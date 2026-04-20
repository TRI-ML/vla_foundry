import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import fsspec
import numpy as np
import torch
import yaml
from pydrake.math import RigidTransform, RotationMatrix
from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers

from vla_foundry.data.robotics.utils import (
    apply_relative_pose,
    get_rot_6d,
    get_xyz,
    pose_to_9d,
    rot_6d_to_matrix,
    to_pose_matrix,
)
from vla_foundry.inference.robotics.utils import (
    any_to_actual_map,
    relative_to_absolute_map,
)


class ObservationMapping:
    """
    Utility wrapper to map observation fields of the robot in gym format to the policy format.

    Args:
        mapping_path: The path to the YAML field mapping definition.
        image_names: The names of the images in the observation (in the format of camera_name_t<timestep>).
    """

    def __init__(self, mapping_path: str, image_names: list[str], num_past_timesteps: int = None):
        with fsspec.open(mapping_path, "r") as handle:
            mapping = yaml.safe_load(handle)
        self._field_paths = mapping["field_paths"]
        self._laterality = ("left", "right")
        self._components = ("xyz", "rot_6d", "gripper", "joint_position")
        self._types = ("actual", "action", "desired")
        self._type_map = self._build_type_map()
        self._camera_names = self._get_camera_names(image_names)
        self.num_past_timesteps = num_past_timesteps

    def _build_type_map(self) -> dict[str, dict[str, dict[str, str]]]:
        type_map: dict[str, dict[str, dict[str, str]]] = {
            type_name: {side: {} for side in self._laterality} for type_name in self._types
        }

        for field in self._field_paths:
            type_name = None
            for _type in self._types:
                if _type in field:
                    type_name = _type
                    break
            side_name = None
            for side in self._laterality:
                if side in field:
                    side_name = side
                    break
            component_name = None
            for component in self._components:
                if component in field:
                    component_name = component
                    break
            if type_name is None or side_name is None or component_name is None:
                raise ValueError(
                    f"Field '{field}' could not be mapped: "
                    f"type_name={type_name}, side_name={side_name}, component_name={component_name}"
                )
            type_map[type_name][side_name][component_name] = field

        return type_map

    def _get_camera_names(self, image_names: list[str]) -> list[str]:
        camera_names: list[str] = []
        for img_name in image_names:
            if "_t" in img_name:
                camera_name, _timestep_str = img_name.rsplit("_t", 1)
            else:
                camera_name = img_name
            if camera_name not in camera_names:
                camera_names.append(camera_name)
        return camera_names

    def _path(self, field: str) -> Sequence[str]:
        absolute = relative_to_absolute_map(field)
        return self._field_paths[absolute]

    def get_type(self, observation: MultiarmObservation, type_name: str) -> dict[str, dict[str, str]]:
        if "actual" in type_name:
            return observation.robot.actual
        elif "desired" in type_name:
            return observation.robot.desired
        else:
            raise ValueError(f"Invalid type name: {type_name}")

    def get_component(self, observation: MultiarmObservation, type: str, laterality: str, component: str) -> np.ndarray:
        """
        Get the component of the pose or gripper of the robot.
        Args:
            observation: The observation of the robot.
            type: The type of the component in (actual, action, desired).
            laterality: The laterality of the component in (left, right).
            component: The component of the pose or gripper in (xyz, rot_6d, gripper).
        Returns:
            The component of the pose or gripper of the robot.
        """

        path = self._path(self._type_map[type][laterality][component])
        robot_data = self.get_type(observation, type)
        if component == "gripper":
            robot_data = robot_data.grippers
            return robot_data[path[0]]
        elif component == "joint_position":
            robot_data = robot_data.joint_position
            return robot_data[path[0]]
        else:
            robot_data = robot_data.poses
            if "xyz" in component:
                return get_xyz(robot_data[path[0]])
            elif "rot_6d" in component:
                return get_rot_6d(robot_data[path[0]])
            else:
                raise ValueError(f"Invalid component: {component} for pose")

    def get_type_name(self, field: str) -> str:
        for type_name in self._types:
            if type_name in field:
                return type_name
        raise ValueError(f"Invalid field: {field}")

    def get_component_name(self, field: str) -> str:
        for component in self._components:
            if component in field:
                return component
        raise ValueError(f"Invalid field: {field}")

    def get_laterality_name(self, field: str) -> str:
        for laterality in self._laterality:
            if laterality in field:
                return laterality
        raise ValueError(f"Invalid field: {field}")

    def get_field(self, observation: MultiarmObservation, field: str) -> np.ndarray:
        """
        Get the field of the robot.
        Args:
            observation: The observation of the robot.
            field: The field of the robot.
        Returns:
            The field of the robot.
        """
        type_name, laterality, component = (
            self.get_type_name(field),
            self.get_laterality_name(field),
            self.get_component_name(field),
        )
        return self.get_component(observation, type_name, laterality, component)

    def get_all_images(self, observation: MultiarmObservation) -> dict[str, Any]:
        """
        Get all the images of the observation in the order listed in the image_names.
        Args:
            observation: The observation of the robot.
        Returns:
            The all the images of the robot.
        """
        images: dict[str, Any] = {}
        for camera_name in self._camera_names:
            if camera_name in observation.visuo:
                images[camera_name] = observation.visuo[camera_name].rgb.array.copy()
        return images

    def get_all_depth_images(self, observation: MultiarmObservation) -> dict[str, Any]:
        """
        Get all the depth images of the observation.
        Args:
            observation: The observation of the robot.
        Returns:
            Dictionary mapping camera names to depth arrays (H, W) uint16.
        """
        depth_images: dict[str, Any] = {}
        for camera_name in self._camera_names:
            if camera_name in observation.visuo and observation.visuo[camera_name].depth is not None:
                depth_images[camera_name] = observation.visuo[camera_name].depth.array.copy()
        return depth_images if depth_images else None

    def get_all_intrinsics(self, observation: MultiarmObservation) -> dict[str, np.ndarray]:
        """
        Get intrinsics matrices for all cameras.
        Args:
            observation: The observation of the robot.
        Returns:
            Dictionary mapping camera names to intrinsics matrices (3, 3).
        """
        intrinsics: dict[str, np.ndarray] = {}
        for camera_name in self._camera_names:
            if camera_name in observation.visuo:
                intrinsics[camera_name] = observation.visuo[camera_name].rgb.K.copy()
        return intrinsics

    def get_all_extrinsics(self, observation: MultiarmObservation) -> dict[str, np.ndarray]:
        """
        Get extrinsics transforms for all cameras.
        Args:
            observation: The observation of the robot.
        Returns:
            Dictionary mapping camera names to extrinsics matrices (4, 4).
        """
        extrinsics: dict[str, np.ndarray] = {}
        for camera_name in self._camera_names:
            if camera_name in observation.visuo:
                X_TC: RigidTransform = observation.visuo[camera_name].rgb.X_TC
                extrinsics[camera_name] = X_TC.GetAsMatrix4()
        return extrinsics

    def create_pose_and_gripper(
        self,
        data_dict: dict,
    ) -> PosesAndGrippers:
        return create_pose_and_grippers(
            field_paths=self._field_paths,
            fields=list(data_dict.keys()),
            value_getter=lambda _field, absolute_field: data_dict[absolute_field],
            source_label="data",
        )


class ActionMapping:
    """
    Utility wrapper to map action fields of the robot in gym format to the policy format and vice versa.
    """

    def __init__(
        self,
        mapping_path: str,
        action_fields: list[str],
        robotics_processor,
        pose_groups: list[dict[str, str]] = None,
        num_past_timesteps: int = None,
    ):
        with fsspec.open(mapping_path, "r") as handle:
            mapping = yaml.safe_load(handle)
        self._field_paths = mapping["field_paths"]
        self.action_fields = action_fields
        self.relative_action_fields = [field for field in self.action_fields if field.endswith("_relative")]
        self.pose_groups = pose_groups or []
        self.field_dims = {
            field: robotics_processor.normalizer.get_field_dimension(field) for field in self.action_fields
        }
        self.action_dim = sum(self.field_dims.values())
        self.normalizer = robotics_processor.normalizer
        self.num_past_timesteps = num_past_timesteps

        # Create lookup table for fast pose group access during inference
        self.pose_group_lookup = {}
        for pose_group in self.pose_groups:
            self.pose_group_lookup[pose_group["position_key"]] = pose_group
            self.pose_group_lookup[pose_group["rotation_key"]] = pose_group

        # Log relative action mode for verification
        if self.relative_action_fields:
            logging.info(f"🔄 RELATIVE ACTION MODE: {len(self.relative_action_fields)} relative fields detected")
            logging.info(f"   Relative fields: {self.relative_action_fields}")
            logging.info(f"   Pose groups: {len(self.pose_groups)} groups configured")
        else:
            logging.info("📍 ABSOLUTE ACTION MODE: No relative fields detected")

    def _pose_to_absolute(
        self,
        relative_xyz: np.ndarray,
        relative_rot_6d: np.ndarray,
        reference_xyz: np.ndarray,
        reference_rot_6d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert relative pose to absolute using reference pose.

        Args:
            relative_xyz: Relative position component
            relative_rot_6d: Relative rotation component
            reference_xyz: Reference position
            reference_rot_6d: Reference rotation

        Returns:
            Tuple of (absolute_xyz, absolute_rot_6d)
        """
        # Flatten references if needed
        reference_xyz = reference_xyz.flatten() if reference_xyz.ndim > 1 else reference_xyz
        reference_rot_6d = reference_rot_6d.flatten() if reference_rot_6d.ndim > 1 else reference_rot_6d

        # Remove extra batch dimension if present
        if relative_xyz.ndim == 3 and relative_xyz.shape[0] == 1:
            relative_xyz = relative_xyz.squeeze(0)
        if relative_rot_6d.ndim == 3 and relative_rot_6d.shape[0] == 1:
            relative_rot_6d = relative_rot_6d.squeeze(0)

        # Convert relative to absolute: absolute = reference @ relative
        reference_pose_matrix = to_pose_matrix(reference_xyz, reference_rot_6d)
        relative_pose_matrices = to_pose_matrix(relative_xyz, relative_rot_6d)
        absolute_pose_matrices = apply_relative_pose(relative_pose_matrices, reference_pose_matrix)

        # Extract absolute components
        return pose_to_9d(absolute_pose_matrices)

    def _to_absolute(
        self, field: str, relative_data: np.ndarray, action: dict, reference: dict[str, np.ndarray]
    ) -> None:
        """Convert relative field to absolute and store in action dict.

        Automatically detects if field belongs to a pose group and applies appropriate conversion.

        Args:
            field: Field name (with _relative suffix)
            relative_data: The relative field data
            action: Action dictionary to update with absolute values
            reference: Reference values for conversion
        """
        absolute_field = relative_to_absolute_map(field)

        # Check if this field belongs to a pose group
        pose_group = self.pose_group_lookup.get(absolute_field)
        if pose_group is not None:
            # Handle pose group fields - compute once for both xyz and rot_6d
            xyz_key = pose_group["position_key"]
            rot_6d_key = pose_group["rotation_key"]
            xyz_relative = f"{xyz_key}_relative"
            rot_6d_relative = f"{rot_6d_key}_relative"

            # Only compute if we haven't processed this pose group yet
            if xyz_key not in action or rot_6d_key not in action:
                # Get reference pose
                xyz_actual = any_to_actual_map(xyz_key)
                rot_6d_actual = any_to_actual_map(rot_6d_key)

                # Get relative data
                relative_xyz = action[xyz_relative]
                relative_rot_6d = action[rot_6d_relative]

                # Convert to absolute
                absolute_xyz, absolute_rot_6d = self._pose_to_absolute(
                    relative_xyz, relative_rot_6d, reference[xyz_actual], reference[rot_6d_actual]
                )

                # Store absolute values
                action[xyz_key] = absolute_xyz
                action[rot_6d_key] = absolute_rot_6d
        else:
            # Handle non-pose relative fields (e.g., joint positions, grippers)
            absolute_actual_field = any_to_actual_map(absolute_field)
            action[absolute_field] = relative_data + reference[absolute_actual_field]

    def from_action_model(
        self,
        action_from_model: torch.Tensor,
        normalizer,
        reference: dict[str, np.ndarray],
    ) -> dict:
        """
        Convert the action from the action model output format to the buffer action format.

        Args:
            action_from_model: The action from the action model output format. (B, T, D)
            normalizer: The normalizer to denormalize the action.
            reference: The reference to convert the relative action fields to absolute action fields.
        Returns:
            The action in the buffer action format.

        Note: If both xyz_relative and rot_6d_relative from the same pose group are in action_fields,
        _pose_to_absolute() will be called during the first field's processing. This is acceptable
        for code clarity - the pose group check prevents duplicate computation.
        """
        action = {}
        start_idx = 0

        # Denormalize all action fields
        for field in self.action_fields:
            dim = self.field_dims[field]
            action[field] = (
                normalizer.denormalize_tensor(
                    action_from_model[..., start_idx : start_idx + dim], field, anchor_timestep=self.num_past_timesteps
                )
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            start_idx += dim

        # Convert relative fields to absolute
        for field in self.relative_action_fields:
            self._to_absolute(field, action[field], action, reference)

        # Create a list of action dictionaries for each timestep instead of a dictionary of sequences
        action_list = []
        absolute_fields = [relative_to_absolute_map(field) for field in self.action_fields]
        for t in range(action_from_model.shape[1]):
            action_dict = {}
            for absolute_field in absolute_fields:
                field_data = action[absolute_field]
                # Handle both 2D (T, dim) and 3D (1, T, dim) cases
                if field_data.ndim == 3:
                    if field_data.shape[0] != 1:
                        raise ValueError(
                            f"Expected batch size 1 for 3D action field '{absolute_field}', "
                            f"but got shape {field_data.shape}.  "
                            f"Only batch size 1 is supported in action conversion."
                        )
                    action_dict[absolute_field] = field_data[0, t, :]
                elif field_data.ndim == 2:
                    action_dict[absolute_field] = field_data[t, :]
                else:
                    # Don't silently broadcast - this indicates a bug
                    raise ValueError(
                        f"Unexpected dimensionality for action field '{absolute_field}': "
                        f"expected 2D (T, {field_data.shape[-1] if field_data.ndim > 0 else '? '}) "
                        f"or 3D (1, T, {field_data.shape[-1] if field_data.ndim > 0 else '? '}), "
                        f"but got shape {field_data.shape}"
                    )
            action_list.append(action_dict)
        return action_list

    def create_pose_and_gripper(
        self,
        action: dict,
    ) -> PosesAndGrippers:
        return create_pose_and_grippers(
            field_paths=self._field_paths,
            fields=self.action_fields,
            value_getter=lambda _field, absolute_field: action[absolute_field],
            source_label="action",
        )

    def from_sim(self, action_from_sim: PosesAndGrippers) -> dict:
        action = {}
        for field in self.action_fields:
            absolute_field = relative_to_absolute_map(field)
            mapping_fields = self._field_paths[absolute_field]
            # TODO: find a better way than to check the length of the mapping fields
            if len(mapping_fields) == 1:
                action[absolute_field] = action_from_sim.grippers[mapping_fields[0]]
            elif len(mapping_fields) == 2:
                if "xyz" in field:
                    action[absolute_field] = get_xyz(action_from_sim.poses[mapping_fields[0]])
                elif "rot" in field:
                    action[absolute_field] = get_rot_6d(action_from_sim.poses[mapping_fields[0]])
                else:
                    raise ValueError(f"Unknown field: {field}")
            else:
                raise ValueError(f"Unknown action field: {field}")
        return action


def resolve_pose_component(component: np.ndarray) -> tuple[str, Any]:
    if component.shape[0] == 3:
        return "p", component
    if component.shape[0] == 6:
        rot_matrix = rot_6d_to_matrix(component)
        return "R", RotationMatrix(rot_matrix)
    raise ValueError(f"Unknown pose component dimensionality: {component.shape}")


def create_pose_and_grippers(
    field_paths: dict[str, Sequence[str]],
    fields: Iterable[str],
    value_getter: Callable[[str, str], np.ndarray],
    source_label: str,
) -> PosesAndGrippers:
    grippers: dict[str, Any] = {}
    poses: dict[str, Any] = {}
    for field in fields:
        absolute_field = relative_to_absolute_map(field)
        mapping_fields = field_paths[absolute_field]
        data_field = value_getter(field, absolute_field)
        if len(mapping_fields) == 1:
            grippers[mapping_fields[0]] = float(np.asarray(data_field).flat[0])
            continue
        if len(mapping_fields) != 2:
            raise ValueError(f"Unknown {source_label} field: {field}")
        pose_name = mapping_fields[0]
        field_name, processed_field = resolve_pose_component(data_field)
        current_pose = poses.get(pose_name)
        if current_pose is None:
            poses[pose_name] = {field_name: processed_field}
            continue
        if isinstance(current_pose, dict):
            current_pose[field_name] = processed_field
            if "p" in current_pose and "R" in current_pose:
                poses[pose_name] = RigidTransform(R=RotationMatrix(current_pose["R"]), p=current_pose["p"])
            else:
                poses[pose_name] = current_pose
            continue
        raise ValueError(f"Pose '{pose_name}' already constructed for field '{field}'")
    return PosesAndGrippers(poses=poses, grippers=grippers)
