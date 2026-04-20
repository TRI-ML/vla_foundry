#!/usr/bin/env python3
"""
Refactored data adapter for converting between robotics pipeline format and policy format.

The refactored design separates configuration, buffering, and conversion logic so the
main adapter class focuses on orchestrating the flow between observations, the robotics
processor, and policy-facing outputs.
"""

import copy
import logging
import math
from typing import Any

import numpy as np
import torch
from PIL import Image

import vla_foundry.visualizers.visualizer as vz
from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod, resize_and_crop_image
from vla_foundry.data.robotics.utils import (
    calculate_relative_pose,
    pose_to_9d,
    to_pose_matrix,
)
from vla_foundry.inference.robotics.lbm_mapping import ActionMapping, ObservationMapping
from vla_foundry.inference.robotics.utils import (
    any_to_actual_map,
    center_crop,
    relative_to_absolute_map,
)


class PolicyDataAdapter:
    """Adapter for converting policy data to training format.

    It requires buffers to keep track of past observations and actions.
    It also requires to manage normalization of the data and adapting relative fields to the correct reference.
    For that, the buffers are always using the unnormalized absolute fields and the adapter is tracking the reference.
    When producing the model input format, the adapter converts absolute buffers to the relative format if needed and
    normalizes the results using the robotics processor normalizer.

    Args:
        robotics_processor: The robotics processor to process the data.
        data_config: The data configuration containing action_fields, proprioception_fields, and pose_groups.
        field_mapping_path: The path to the field mapping file that references the field names in the gym format.
        image_names: The names of the images in the observation (in the format of camera_name_t<timestep>).
        preprocessor_image_size: The size of the image preprocessor.
    """

    def __init__(
        self,
        robotics_processor,
        data_config,
        field_mapping_path: str,
        image_names: list[str],
        preprocessor_image_size: tuple[int, int],
        preprocessor_image_resize_method: ImageResizingMethod = ImageResizingMethod.CENTER_CROP,
        num_past_timesteps: int = 1,
        num_future_timesteps: int = 14,
        image_indices: tuple[int, ...] = (-1, 0),
        gripper_debounce_open_threshold: float = 0.6,
        gripper_debounce_close_threshold: float = 0.4,
    ):
        self.robotics_processor = robotics_processor
        self.data_config = data_config
        self.action_fields = list(data_config.action_fields)
        self.relative_action_fields = [field for field in self.action_fields if field.endswith("_relative")]
        self.proprioception_fields = list(data_config.proprioception_fields)
        self.relative_proprioception_fields = [
            field for field in self.proprioception_fields if field.endswith("_relative")
        ]
        self.pose_groups = list(data_config.pose_groups)

        # Create lookup table for fast pose group access during inference
        self.pose_group_lookup = {}
        for pose_group in self.pose_groups:
            self.pose_group_lookup[pose_group["position_key"]] = pose_group
            self.pose_group_lookup[pose_group["rotation_key"]] = pose_group
        self.num_past_timesteps = num_past_timesteps
        self.num_future_timesteps = num_future_timesteps
        self.image_indices = image_indices
        self.image_names = image_names
        self.field_mapping = ObservationMapping(field_mapping_path, image_names)
        self.action_mapping = ActionMapping(
            field_mapping_path,
            self.action_fields,
            self.robotics_processor,
            pose_groups=self.pose_groups,
            num_past_timesteps=num_past_timesteps,
        )
        self.action_dim = self.action_mapping.action_dim
        self.language_instruction = "Do the task"

        self.preprocessor_image_size = preprocessor_image_size
        self.preprocessor_image_resize_method = preprocessor_image_resize_method
        self.image_crop_size = self.data_config.augmentation.image.crop.shape

        self.total_action_timesteps = self.num_past_timesteps + 1 + self.num_future_timesteps

        self.action_buffer = []
        self.action_buffer_mask = []
        self.proprioception_buffer = []
        self.image_buffer = []
        self.reference = {}
        self.reference_initialized = False
        self.has_predicted = False  # Whether at least one inference has been done

        # Gripper hysteresis debouncer: when open, value must drop below close_threshold to close;
        # when closed, value must rise above open_threshold to open.
        # Both thresholds must be set to enable debouncing (None = disabled).
        self.gripper_debounce_close_threshold = gripper_debounce_close_threshold
        self.gripper_debounce_open_threshold = gripper_debounce_open_threshold
        # Per-field gripper state: True = open, False = closed
        self.gripper_state: dict[str, bool] = {}
        self.gripper_fields = [field for field in self.action_fields if "gripper" in field]
        # Precompute snap values (min=closed, max=open) from normalizer stats
        self.gripper_snap_values: dict[str, tuple[float, float]] = {}
        normalizer = self.robotics_processor.normalizer
        for field in self.gripper_fields:
            absolute_field = relative_to_absolute_map(field)
            if absolute_field in normalizer.stats:
                field_stats = normalizer.stats[absolute_field]
                snap_closed = float(np.min(field_stats["min"]))
                snap_open = float(np.max(field_stats["max"]))
                self.gripper_snap_values[absolute_field] = (snap_closed, snap_open)
                logging.info(
                    f"Gripper debounce for '{absolute_field}': snap_closed={snap_closed:.4f}, snap_open={snap_open:.4f}"
                )

    def initialize_action_buffer(self, observation) -> None:
        logging.debug("Initializing action buffer")
        action = {}
        for field in self.action_fields:
            # Get the absolute field name (strip _relative suffix if present)
            absolute_field = relative_to_absolute_map(field)
            # Read from the actual robot state
            actual_absolute_field = any_to_actual_map(absolute_field)
            robot_data = self.field_mapping.get_field(observation, actual_absolute_field)
            # Store with the absolute field name as key
            action[absolute_field] = np.asarray(robot_data, dtype=np.float64)

        self.action_buffer = [copy.deepcopy(action) for _ in range(self.total_action_timesteps)]
        self.action_buffer_mask = [True] + [False] * (len(self.action_buffer) - 1)

    def initialize_proprioception_buffer(self, observation) -> None:
        logging.debug("Initializing proprioception buffer")
        if not self.proprioception_fields:
            self.proprioception_buffer = []
            return

        proprioception = {}
        for field in self.proprioception_fields:
            absolute_field = relative_to_absolute_map(field)
            actual_field = any_to_actual_map(absolute_field)
            robot_data = self.field_mapping.get_field(observation, actual_field)
            proprioception[absolute_field] = np.asarray(robot_data, dtype=np.float64)

        buffer_size = self.num_past_timesteps + 1
        self.proprioception_buffer = [copy.deepcopy(proprioception) for _ in range(buffer_size)]

    def initialize_image_buffer(self, observation) -> None:
        logging.debug("Initializing image buffer")
        if len(self.image_names) == 0:
            self.image_buffer = []
            return

        images = self.field_mapping.get_all_images(observation)
        max_time = -np.inf
        min_time = np.inf
        for image_name in self.image_names:
            _, timestep_str = image_name.rsplit("_t", 1)
            timestep = int(timestep_str)
            max_time = max(max_time, timestep)
            min_time = min(min_time, timestep)

        range_images = int(max_time - min_time + 1)
        for _ in range(range_images):
            self.image_buffer.append(copy.deepcopy(images))

    def reset(self, initial_observation):
        logging.debug("Resetting data adapter")
        self.action_buffer = []
        self.action_buffer_mask = []
        self.proprioception_buffer = []
        self.image_buffer = []
        self.reference_initialized = False
        self.has_predicted = False
        self.gripper_state = {}
        self.initialize_action_buffer(initial_observation)
        self.initialize_proprioception_buffer(initial_observation)
        self.initialize_image_buffer(initial_observation)
        self.update_reference(initial_observation)
        self.past_mask = torch.zeros(1, self.num_past_timesteps + 1 + self.num_future_timesteps, dtype=torch.bool)

    def step_proprioception(self, observation) -> None:
        if not self.proprioception_fields:
            return

        proprioception = {}
        for field in self.proprioception_fields:
            absolute_field = relative_to_absolute_map(field)
            actual_field = any_to_actual_map(absolute_field)
            robot_data = self.field_mapping.get_field(observation, actual_field)
            proprioception[absolute_field] = np.asarray(robot_data, dtype=np.float64)
        self.proprioception_buffer.append(proprioception)
        if len(self.proprioception_buffer) > self.num_past_timesteps + 1:
            self.proprioception_buffer.pop(0)

    def step_action(self, lag_compensation: float = 0.0):
        # Compute the read position with lag compensation (fractional offset into the future)
        base_idx = self.num_past_timesteps
        offset_idx = base_idx + lag_compensation
        lo = int(math.floor(offset_idx))
        hi = int(math.ceil(offset_idx))
        frac = offset_idx - lo

        # Clamp to valid buffer range
        max_idx = len(self.action_buffer) - 1
        lo = max(0, min(lo, max_idx))
        hi = max(0, min(hi, max_idx))

        if frac < 1e-9 or lo == hi:
            current_action_dict = copy.deepcopy(self.action_buffer[lo])
        else:
            a, b = self.action_buffer[lo], self.action_buffer[hi]
            current_action_dict = {k: a[k] * (1.0 - frac) + b[k] * frac for k in a}

        # Shift buffer left by 1 (consume one timestep)
        last_action = copy.deepcopy(self.action_buffer[-1])
        self.action_buffer.pop(0)
        self.action_buffer.append(last_action)
        self.action_buffer_mask.pop(0)
        self.action_buffer_mask.append(False)

        self._debounce_gripper(current_action_dict)

        output = self.action_mapping.create_pose_and_gripper(current_action_dict)
        vz.log_robot_gym_poses_and_grippers("current_action_arm_poses", output)
        return output

    def _debounce_gripper(self, action_dict: dict) -> None:
        """Apply hysteresis debouncing to gripper fields in-place.

        Thresholds are in normalized [0, 1] space (0 = closed, 1 = open).
        When currently open, the value must drop below close_threshold to close.
        When currently closed, the value must rise above open_threshold to open.
        Snap values are taken from the normalizer stats (min = closed, max = open).
        """
        for field in self.gripper_fields:
            absolute_field = relative_to_absolute_map(field)
            if absolute_field not in action_dict or absolute_field not in self.gripper_snap_values:
                continue

            snap_closed, snap_open = self.gripper_snap_values[absolute_field]
            raw_value = float(np.asarray(action_dict[absolute_field]).flatten()[0])

            # Normalize raw value to [0, 1] for threshold comparison
            value_range = snap_open - snap_closed
            if abs(value_range) < 1e-8:
                continue
            normalized = (raw_value - snap_closed) / value_range

            if absolute_field not in self.gripper_state:
                midpoint = (self.gripper_debounce_close_threshold + self.gripper_debounce_open_threshold) / 2
                self.gripper_state[absolute_field] = normalized > midpoint

            is_open = self.gripper_state[absolute_field]
            if is_open and normalized < self.gripper_debounce_close_threshold:
                self.gripper_state[absolute_field] = False
            elif not is_open and normalized > self.gripper_debounce_open_threshold:
                self.gripper_state[absolute_field] = True

            snapped = snap_open if self.gripper_state[absolute_field] else snap_closed
            action_dict[absolute_field] = np.array([snapped], dtype=np.float64)

    def get_remaining_actions_in_buffer(self) -> tuple[int, int]:
        """Return remaining valid actions and total remaining slots from current execution index."""
        start_idx = min(self.num_past_timesteps, len(self.action_buffer_mask))
        remaining_slots = max(0, len(self.action_buffer_mask) - start_idx)
        remaining_actions = int(sum(self.action_buffer_mask[start_idx:]))
        return remaining_actions, remaining_slots

    def update_reference(self, observation):
        """
        The data adapter has buffers for the past actions and proprioception that are kept in absolute format.
        If the model needs relative fields input or if we need to interpret the relative action output of the model,
        we need to compute them using a reference observation.
        This function updates the reference with the given observation.
        """
        logging.debug("Updating reference")
        for field in self.action_fields:
            absolute_actual_field = relative_to_absolute_map(any_to_actual_map(field))
            robot_data = self.field_mapping.get_field(observation, absolute_actual_field)
            self.reference[absolute_actual_field] = np.asarray(robot_data, dtype=np.float64)
        vz.log_robot_gym_poses_and_grippers(
            "reference_arm_poses", self.field_mapping.create_pose_and_gripper(self.reference)
        )
        for field in self.proprioception_fields:
            absolute_actual_field = relative_to_absolute_map(any_to_actual_map(field))
            robot_data = self.field_mapping.get_field(observation, absolute_actual_field)
            self.reference[absolute_actual_field] = np.asarray(robot_data, dtype=np.float64)
        vz.log_robot_gym_poses_and_grippers(
            "reference_proprioception", self.field_mapping.create_pose_and_gripper(self.reference)
        )

    def preprocess_images(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        logging.debug(f"Preprocessing images resize {self.preprocessor_image_size} crop {self.image_crop_size}")
        processed: dict[str, np.ndarray] = {}
        for camera_name, image in images.items():
            resized = resize_and_crop_image(
                image,
                target_size=self.preprocessor_image_size,
                resize_method=self.preprocessor_image_resize_method,
            )
            cropped = center_crop(resized, self.image_crop_size[0], self.image_crop_size[1])
            if isinstance(cropped, Image.Image):
                cropped = np.array(cropped)
            processed[camera_name] = cropped
        return processed

    def step_image(self, observation) -> None:
        logging.debug("Stepping image")
        images = self.field_mapping.get_all_images(observation)
        vz.log_images("observation_images", images)
        processed_images = self.preprocess_images(images)
        self.image_buffer.append(processed_images)
        self.image_buffer.pop(0)

    def step_task(self, observation) -> None:
        logging.debug(f"Stepping task to instruction {observation.language_instruction}")
        self.language_instruction = observation.language_instruction
        vz.log_text("language_instruction", self.language_instruction)

    def step_past_mask(self) -> None:
        logging.debug("Stepping past mask")
        if self.num_past_timesteps == 0:
            return

        if self.num_past_timesteps > 1:
            # Roll the past mask to the left as we are stepping the buffer to track the real past actions in the buffer
            self.past_mask[:, : self.num_past_timesteps - 1] = self.past_mask[:, 1 : self.num_past_timesteps].clone()
        self.past_mask[:, self.num_past_timesteps - 1] = True

    def step_observations(
        self,
        observation,
    ):
        self.step_proprioception(observation)
        self.step_image(observation)
        self.step_task(observation)
        self.step_past_mask()
        vz.log_robot_gym_poses_and_grippers("current_pose", observation.robot.actual)

    def get_images_for_processor(self) -> dict[str, np.ndarray]:
        logging.debug("Getting images for processor")
        images: dict[str, np.ndarray] = {}
        for image_name in self.image_names:
            camera_name, timestep_str = image_name.rsplit("_t", 1)
            timestep = int(timestep_str)
            timestep = len(self.image_buffer) - 1 + timestep
            buffer_t = self.image_buffer[timestep]
            logging.debug(f"Getting image {camera_name} at timestep {timestep} of buffer for {image_name}")
            if camera_name in buffer_t:
                images[image_name] = buffer_t[camera_name]
        processed_images = {}
        for name, image in images.items():
            image = np.asarray(image, dtype=np.float32)
            image_min = image.min()
            image_max = image.max()
            if image_max == image_min:
                processed_images[name] = np.zeros_like(image, dtype=np.uint8)
                continue
            scaled = (image - image_min) / (image_max - image_min)
            processed_images[name] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
        vz.log_images("processed_images", processed_images)
        return images

    def get_lowdim_for_processor(self) -> dict[str, torch.Tensor]:
        logging.debug("Getting lowdim for processor")
        lowdim = self._stack_fields(
            self.action_buffer,
            self.action_fields,
            self.relative_action_fields,
        )

        if self.proprioception_fields:
            lowdim.update(
                self._stack_fields(
                    self.proprioception_buffer,
                    self.proprioception_fields,
                    self.relative_proprioception_fields,
                )
            )

        return lowdim

    def _to_relative(self, buffer: list[dict], field: str, timesteps_slice: slice | None = None) -> np.ndarray:
        """Convert absolute field to relative based on field type.

        Automatically detects if field belongs to a pose group and applies appropriate conversion.

        Args:
            buffer: Buffer containing absolute field data
            field: Field name (with _relative suffix)
            timesteps_slice: Optional slice to apply to buffer

        Returns:
            Relative values as numpy array
        """
        absolute_field = relative_to_absolute_map(field)

        # Check if this field belongs to a pose group
        pose_group = self.pose_group_lookup.get(absolute_field)
        if pose_group is not None:
            # Handle pose group fields (xyz + rot_6d)
            xyz_key = pose_group["position_key"]
            rot_6d_key = pose_group["rotation_key"]

            # Calculate relative pose
            relative_xyz, relative_rot_6d = self._pose_to_relative(buffer, xyz_key, rot_6d_key, timesteps_slice)

            # Return the appropriate component
            if absolute_field == xyz_key:
                return relative_xyz
            else:  # rot_6d_key
                return relative_rot_6d

        # Handle non-pose relative fields (e.g., joint positions)
        values = [np.asarray(entry[absolute_field], dtype=np.float64) for entry in buffer]
        if timesteps_slice is not None:
            values = values[timesteps_slice]
        stacked = np.stack(values, axis=0)

        actual_field = any_to_actual_map(absolute_field)
        reference = self.reference.get(actual_field)
        if reference is None:
            raise KeyError(f"Reference for '{actual_field}' not initialized")

        if "joint_position" in field:
            return self._wrap_to_pi(stacked - reference)
        else:
            # TODO: This does not current support generic subtraction
            raise ValueError(f"Unsupported relative field type for '{field}'")

    def _stack_fields(
        self,
        buffer: list[dict],
        fields: list[str],
        relative_fields: list[str],
        timesteps_slice: slice | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stack fields from buffer into tensors, applying relative conversions as needed.

        Note: If both xyz_relative and rot_6d_relative from the same pose group are in fields,
        _pose_to_relative() will be called twice. This is acceptable for code clarity - the
        alternative would require pre-processing pose groups which adds complexity. The duplicate
        calculation is minimal compared to model inference time. Consider refactoring in the future.
        """
        stacked_fields: dict[str, torch.Tensor] = {}
        if not buffer or not fields:
            return stacked_fields

        for field in fields:
            absolute_field = relative_to_absolute_map(field)

            if field in relative_fields:
                # Apply relative conversion
                stacked = self._to_relative(buffer, field, timesteps_slice)
            else:
                # Absolute field - extract and stack
                values = [np.asarray(entry[absolute_field], dtype=np.float64) for entry in buffer]
                if timesteps_slice is not None:
                    values = values[timesteps_slice]
                stacked = np.stack(values, axis=0)

            # Convert to 2D tensor
            stacked = stacked.astype(np.float32)
            if stacked.ndim == 1:
                stacked = stacked[:, None]
            if stacked.ndim != 2:
                raise ValueError(f"Expected stacked tensor to be 2D for field '{field}', got shape {stacked.shape}")

            logging.debug(f"Stacked shape: {stacked.shape} {field}")
            stacked_fields[field] = torch.as_tensor(stacked, dtype=torch.float32)

        return stacked_fields

    def _pose_to_relative(
        self, buffer: list[dict], xyz_key: str, rot_6d_key: str, timesteps_slice: slice | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert absolute pose fields to relative pose.

        Args:
            buffer: Buffer containing absolute pose data
            xyz_key: Key for position field (absolute)
            rot_6d_key: Key for rotation field (absolute)
            timesteps_slice: Optional slice to apply to buffer

        Returns:
            Tuple of (relative_xyz, relative_rot_6d) as numpy arrays
        """
        # Extract and stack values from buffer
        values_xyz = [np.asarray(entry[xyz_key], dtype=np.float64) for entry in buffer]
        values_rot = [np.asarray(entry[rot_6d_key], dtype=np.float64) for entry in buffer]
        if timesteps_slice is not None:
            values_xyz = values_xyz[timesteps_slice]
            values_rot = values_rot[timesteps_slice]
        xyz_stacked = np.stack(values_xyz, axis=0)
        rot_6d_stacked = np.stack(values_rot, axis=0)

        # Get reference pose
        xyz_actual = any_to_actual_map(xyz_key)
        rot_6d_actual = any_to_actual_map(rot_6d_key)
        reference_xyz = self.reference.get(xyz_actual)
        reference_rot_6d = self.reference.get(rot_6d_actual)

        if reference_xyz is None or reference_rot_6d is None:
            raise KeyError(f"Reference missing for pose: xyz={xyz_actual}, rot_6d={rot_6d_actual}")

        # Calculate relative pose using pose matrices
        reference_pose_matrix = to_pose_matrix(reference_xyz, reference_rot_6d)
        current_pose_matrices = to_pose_matrix(xyz_stacked, rot_6d_stacked)
        relative_pose_matrices = calculate_relative_pose(current_pose_matrices, reference_pose_matrix)
        relative_xyz, relative_rot_6d = pose_to_9d(relative_pose_matrices)

        return relative_xyz, relative_rot_6d

    @staticmethod
    def _wrap_to_pi(delta: np.ndarray) -> np.ndarray:
        return (delta + np.pi) % (2 * np.pi) - np.pi

    def get_processor_input(self) -> dict[str, Any]:
        logging.debug("Getting processor input")
        processor_input = {
            "images": [self.get_images_for_processor()],
            "lowdim": [self.get_lowdim_for_processor()],
            "metadata": [
                {
                    "anchor_relative_idx": self.num_past_timesteps,
                    "original_anchor_relative_idx": self.num_past_timesteps,
                }
            ],
            "language_instruction": [self.language_instruction],
        }

        return processor_input

    def get_model_input(self, observation) -> dict[str, torch.Tensor]:
        logging.debug("Getting model input")
        self.update_reference(observation)
        processor_input = self.get_processor_input()
        processed = self.robotics_processor.process_inputs(processor_input, image_names=self.image_names)
        processed = self.robotics_processor.add_action_and_proprioception_fields(
            processed,
            action_fields=self.action_fields,
            proprioception_fields=self.proprioception_fields,
        )
        processed["past_mask"] = self.past_mask.clone()
        return processed

    def update_action(self, observation, model_output: torch.Tensor):
        logging.debug("Updating action buffer with fresh predictions")
        model_output = model_output.cpu()
        n_missing_actions = max(0, len(self.action_buffer) - model_output.shape[1])
        if n_missing_actions > 0:
            filler = model_output[:, -1:].expand(-1, n_missing_actions, -1)
            model_output = torch.cat((model_output, filler), dim=1)
        self.has_predicted = True
        action_list = self.action_mapping.from_action_model(
            model_output,
            self.robotics_processor.normalizer,
            self.reference,
        )
        # Update the action buffer with the new actions from the model
        self.action_buffer = [copy.deepcopy(action) for action in action_list]
        self.action_buffer_mask = [True] * (len(self.action_buffer) - n_missing_actions) + [False] * n_missing_actions
        vz.log_robot_gym_action_predictions(
            "action_predictions", [self.action_mapping.create_pose_and_gripper(action) for action in self.action_buffer]
        )
