import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

import fsspec
import numpy as np
import ray
import yaml

from vla_foundry.data.preprocessing.robotics.converters.base import BaseRoboticsConverter
from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks
from vla_foundry.data.preprocessing.utils import (
    is_still_sample,
    validate_pose_groups,
)
from vla_foundry.data.robotics.utils import any_to_actual_key, load_action_field_config


@dataclass
class SampleMetadata:
    """Metadata for each preprocessed sample."""

    episode_id: str
    sample_id: str
    anchor_timestep: int | None
    anchor_relative_idx: int | None
    image_timesteps: list[int]
    lowdim_start_timestep: int
    lowdim_end_timestep: int
    past_padding: int
    future_padding: int
    camera_names: list[str]
    original_episode_length: int
    original_image_sizes: dict[str, tuple[int, int]]
    is_padded: bool


@ray.remote
def check_episode_validity_ray(episode_path: str) -> str | None:
    """Check if an episode directory has valid processed data. Returns episode path if valid, None otherwise."""
    fs, _ = fsspec.core.url_to_fs(episode_path)
    processed_path = os.path.join(episode_path, "processed")
    fs_processed_path = processed_path.replace("s3://", "")

    try:
        if not fs.exists(fs_processed_path):
            return None

        # Check for required files
        required_files = ["metadata.yaml", "observations.npz"]
        for required_file in required_files:
            file_path = os.path.join(processed_path, required_file)
            fs_file_path = file_path.replace("s3://", "")
            if not fs.exists(fs_file_path):
                return None
        return episode_path
    except Exception:
        return None


@ray.remote
def discover_and_validate_episodes_in_directory(
    diffusion_spartan_path: str, max_episodes: int = -1, validation_episodes=None
) -> list[str]:
    """Discover and validate episodes in a diffusion_spartan directory in parallel."""
    fs, _ = fsspec.core.url_to_fs(diffusion_spartan_path)
    fs_path = diffusion_spartan_path.replace("s3://", "")

    try:
        items = fs.listdir(fs_path)
    except Exception as e:
        print(f"Warning: Cannot list directory {diffusion_spartan_path}: {e}")
        return []

    # First pass: identify episode directories only
    episode_paths = []
    for item in items:
        item_name = item["name"] if isinstance(item, dict) else item
        item_basename = os.path.basename(item_name.rstrip("/"))

        # Only process directories that start with "episode_" - skip all files
        if item_basename.startswith("episode_") and not any(
            item_basename.endswith(ext) for ext in [".pkl", ".npz", ".txt", ".json", ".yaml", ".tar", ".gz"]
        ):
            episode_path = os.path.join(diffusion_spartan_path, item_basename)
            if episode_path.startswith("s3://"):
                # Extract task name from S3 path — assumes structure: s3://<bucket>/.../tasks/<task>/...
                task_name = (
                    episode_path.split("/tasks/")[-1].split("/")[0]
                    if "/tasks/" in episode_path
                    else episode_path.split("/")[-1]
                )
            else:
                # Somewhat hacky but I think this is always correct.
                task_name = episode_path.split("tasks/")[-1].split("/")[0]
            episode_num = int(item_basename.split("_")[-1])
            if validation_episodes and episode_num in validation_episodes[task_name]:
                continue
            episode_paths.append(episode_path)

            # Early exit if we have enough episodes
            if max_episodes > 0 and len(episode_paths) >= max_episodes:
                break

    if not episode_paths:
        return []

    # Second pass: validate episodes in parallel using Ray
    validation_futures = [check_episode_validity_ray.remote(ep_path) for ep_path in episode_paths]
    validation_results = ray.get(validation_futures)

    # Filter out None results (invalid episodes)
    valid_episodes = [ep for ep in validation_results if ep is not None]

    return valid_episodes


class SpartanConverter(BaseRoboticsConverter):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Load language annotations
        print("📚 Loading language annotations...")
        with open(cfg.language_annotations_path) as f:
            data = yaml.safe_load(f)
        self.language_annotations = data.get("language_dict", {})
        print(f"Loaded language annotations for {len(self.language_annotations)} tasks")

        # Load action field configuration
        print("📘 Loading action field configuration...")
        action_field_config = load_action_field_config(cfg.action_fields_config_path)
        self.action_key_fields = action_field_config["action_key_fields"]
        self.action_index_fields = action_field_config["action_index_fields"]

        # Load pose groups from action field config
        if "pose_groups" in action_field_config:
            self.pose_groups = action_field_config["pose_groups"]
            print(f"Loaded {len(action_field_config['pose_groups'])} pose groups from action field config")

            # Validate pose groups are complete
            validate_pose_groups(self.pose_groups)
        else:
            # Pose groups must be explicitly provided in action field config
            raise ValueError(
                "pose_groups not found in action field config. "
                "Please add pose_groups to your action field configuration file to enable \
                    relative coordinate computation."
            )

        print(f"Loaded {len(self.action_key_fields)} action fields")
        if self.action_key_fields:
            prev_index = 0
            debug_slices = []
            for name, cumulative_index in zip(self.action_key_fields, self.action_index_fields, strict=False):
                debug_slices.append(f"{name} (dim={cumulative_index - prev_index})")
                prev_index = cumulative_index
            print("🧭 Action field slices:", debug_slices)

        # Load validation episodes
        self.validation_episodes = None
        if cfg.validation_episodes_path:
            with open(cfg.validation_episodes_path) as f:
                self.validation_episodes = json.load(f)
            print(f"Loaded validation episodes: {self.validation_episodes}")

        if self.action_key_fields and len(self.action_key_fields) != len(self.action_index_fields):
            raise ValueError("Action field configuration mismatch: key and index lists differ in length")

        self.action_field_sizes = []
        if self.action_key_fields:
            previous_index = 0
            for key, cumulative_index in zip(self.action_key_fields, self.action_index_fields, strict=False):
                field_size = cumulative_index - previous_index
                if field_size <= 0:
                    raise ValueError(
                        f"Action field indices must be strictly increasing. Field {key} produced size {field_size}."
                    )
                self.action_field_sizes.append(field_size)
                previous_index = cumulative_index

            print(
                "🧭 Action field slices:",
                [
                    f"{name} (dim={size})"
                    for name, size in zip(self.action_key_fields, self.action_field_sizes, strict=False)
                ],
            )

    def discover_episodes(self, source_paths: list[str], max_episodes_to_process: int = -1) -> list[str]:
        """
        Discover episodes efficiently by assuming all source_paths are diffusion_spartan directories.
        Uses Ray for parallel validation.
        """
        if isinstance(source_paths, str):
            source_paths = [source_paths]

        # Assume all source_paths are diffusion_spartan directories
        diffusion_spartan_dirs = source_paths
        print(f"Using {len(diffusion_spartan_dirs)} diffusion_spartan directories")

        # Discover and validate episodes in parallel using Ray
        print("Discovering and validating episodes in parallel...")
        discover_futures = [
            discover_and_validate_episodes_in_directory.remote(
                dir_path, -1, self.validation_episodes
            )  # Let each dir discover all episodes
            for dir_path in diffusion_spartan_dirs
        ]
        discover_results = ray.get(discover_futures)

        # Merge results from all directories
        all_episodes = []
        for result in discover_results:
            all_episodes.extend(result)

        # Apply max_episodes_to_process limit if specified
        if max_episodes_to_process > 0 and len(all_episodes) > max_episodes_to_process:
            all_episodes = all_episodes[:max_episodes_to_process]

        print(f"Total episodes discovered: {len(all_episodes)}")
        return sorted(all_episodes)

    def extract_sample_camera_calibration(
        self,
        episode_intrinsics: dict[str, Any],
        episode_extrinsics: dict[str, Any],
        valid_start: int,
        valid_end: int,
        past_padding: int,
        future_padding: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract camera calibration data for the lowdim sequence timespan."""
        sample_intrinsics = {}
        sample_extrinsics = {}

        # Extract intrinsics for the sequence timespan
        for camera_name, intrinsics_data in episode_intrinsics.items():
            if isinstance(intrinsics_data, np.ndarray):
                if intrinsics_data.ndim == 3:  # Time-varying intrinsics (timesteps, 3, 3)
                    # Extract the sequence from valid_start to valid_end (inclusive)
                    valid_intrinsics = intrinsics_data[valid_start : valid_end + 1]

                    # Apply padding to match lowdim data
                    if past_padding > 0 or future_padding > 0:
                        # Use edge padding for camera calibration
                        pad_width = [(past_padding, future_padding)] + [(0, 0)] * (valid_intrinsics.ndim - 1)
                        valid_intrinsics = np.pad(valid_intrinsics, pad_width, mode="edge")

                    sample_intrinsics[camera_name] = valid_intrinsics
                elif intrinsics_data.ndim == 2:  # Static intrinsics (3, 3)
                    # For static calibration, repeat for the sequence length
                    sequence_length = (valid_end - valid_start + 1) + past_padding + future_padding
                    sample_intrinsics[camera_name] = np.tile(intrinsics_data[np.newaxis, :, :], (sequence_length, 1, 1))

        # Extract extrinsics for the sequence timespan
        for camera_name, extrinsics_data in episode_extrinsics.items():
            if isinstance(extrinsics_data, np.ndarray):
                if extrinsics_data.ndim == 3:  # Time-varying extrinsics (timesteps, 4, 4)
                    # Extract the sequence from valid_start to valid_end (inclusive)
                    valid_extrinsics = extrinsics_data[valid_start : valid_end + 1]

                    # Apply padding to match lowdim data
                    if past_padding > 0 or future_padding > 0:
                        # Use edge padding for camera calibration
                        pad_width = [(past_padding, future_padding)] + [(0, 0)] * (valid_extrinsics.ndim - 1)
                        valid_extrinsics = np.pad(valid_extrinsics, pad_width, mode="edge")

                    sample_extrinsics[camera_name] = valid_extrinsics
                elif extrinsics_data.ndim == 2:  # Static extrinsics (4, 4)
                    # For static calibration, repeat for the sequence length
                    sequence_length = (valid_end - valid_start + 1) + past_padding + future_padding
                    sample_extrinsics[camera_name] = np.tile(extrinsics_data[np.newaxis, :, :], (sequence_length, 1, 1))

        return sample_intrinsics, sample_extrinsics

    def transform_camera_calibration_keys(
        self, intrinsics: dict[str, Any], extrinsics: dict[str, Any], metadata: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform camera calibration keys from camera IDs to semantic names."""
        # Get camera mapping from metadata
        camera_mapping = metadata.get("camera_id_to_semantic_name", {})
        if not camera_mapping:
            # If no mapping available, return original data
            return intrinsics, extrinsics

        # Transform intrinsics keys
        transformed_intrinsics = {}
        for camera_id, semantic_name in camera_mapping.items():
            if camera_id in intrinsics:
                transformed_intrinsics[semantic_name] = intrinsics[camera_id]

        # Transform extrinsics keys
        transformed_extrinsics = {}
        for camera_id, semantic_name in camera_mapping.items():
            if camera_id in extrinsics:
                transformed_extrinsics[semantic_name] = extrinsics[camera_id]

        return transformed_intrinsics, transformed_extrinsics

    def get_language_instructions(self, episode_path: str, instruction_types: list[str] = None) -> dict[str, list[str]]:
        """Get language instructions for a given task, organized by type.

        Args:
            episode_path: Path to the episode
            instruction_types: List of instruction types to include. If None, includes all types.
                            Valid types: "original", "randomized", "verbose", "alternative"

        Returns:
            Dictionary mapping instruction type to list of instructions
        """
        path_parts = episode_path.split("/")
        task_name = path_parts[path_parts.index("tasks") + 1]

        if task_name not in self.language_annotations:
            return {}

        task_annotations = self.language_annotations[task_name]
        instructions_by_type = {}

        # Default to all types if none specified
        if instruction_types is None:
            instruction_types = ["original", "randomized", "verbose", "alternative"]

        # Add instructions for each requested type
        for instruction_type in instruction_types:
            if instruction_type in task_annotations:
                instructions_by_type[instruction_type] = task_annotations[instruction_type]

        return instructions_by_type

    def load_episode_data(self, episode_path: str) -> dict[str, Any]:
        """Load episode data with optimization and retry logic."""
        processed_path = os.path.join(episode_path, "processed")

        # Load metadata
        metadata_path = os.path.join(processed_path, "metadata.yaml")
        with fsspec.open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        # Load observations
        obs_path = os.path.join(processed_path, "observations.npz")
        with fsspec.open(obs_path, "rb") as f:
            observations = np.load(f, allow_pickle=True)
            observations = {k: v for k, v in observations.items() if k not in self.cfg.data_discard_keys}

        # Load actions
        actions = {}
        actions_path = os.path.join(processed_path, "actions.npz")
        with fsspec.open(actions_path, "rb") as f:
            actions_archive = np.load(f, allow_pickle=True)
            # Extract the 'actions' key specifically
            if "actions" in actions_archive:
                actions = {"actions": actions_archive["actions"]}
            else:
                print(f"Warning: 'actions' key not found in {actions_path}")
                print(f"Available keys: {list(actions_archive.keys())}")
                actions = {}

        # Load camera params
        intrinsics, extrinsics = {}, {}
        intrinsics_path = os.path.join(processed_path, "intrinsics.npz")
        extrinsics_path = os.path.join(processed_path, "extrinsics.npz")
        with fsspec.open(intrinsics_path, "rb") as f:
            intrinsics_archive = np.load(f)
            intrinsics = {key: intrinsics_archive[key] for key in intrinsics_archive.files}
        with fsspec.open(extrinsics_path, "rb") as f:
            extrinsics_archive = np.load(f)
            extrinsics = {key: extrinsics_archive[key] for key in extrinsics_archive.files}

        # Transform camera calibration keys from camera IDs to semantic names
        intrinsics, extrinsics = self.transform_camera_calibration_keys(intrinsics, extrinsics, metadata)

        return {
            "metadata": metadata,
            "observations": observations,
            "actions": actions,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

    def get_episode_length(self, episode_data: dict[str, Any]) -> int:
        """Get episode length from episode data."""
        first_obs_key = next(iter(episode_data["observations"].keys()))
        episode_length = episode_data["observations"][first_obs_key].shape[0]
        return episode_length

    def extract_camera_data(self, episode_data: dict[str, Any]) -> dict[str, np.ndarray]:
        """Extract camera data with filtering, including depth images.

        Behavior controlled by cfg.skip_episodes_missing_cameras:
        - If False (default): Process with available cameras, warn about missing ones
        - If True: Return empty dict (skip episode) if any requested cameras are missing
        """
        camera_mapping = episode_data["metadata"].get("camera_id_to_semantic_name", {})

        if self.cfg.camera_names:
            available_cameras = set(camera_mapping.values())
            missing_cameras = [cam for cam in self.cfg.camera_names if cam not in available_cameras]

            if missing_cameras:
                if self.cfg.skip_episodes_missing_cameras:
                    # Skip this episode entirely
                    print(f"⚠️  Skipping episode - missing cameras: {missing_cameras}")
                    print(f"   Available cameras: {list(available_cameras)}")
                    return {}
                else:
                    # Continue with available cameras
                    print(f"⚠️  Warning: Processing with partial cameras. Missing: {missing_cameras}")
                    print(f"   Available cameras: {list(available_cameras)}")

            filtered_mapping = {
                cid: sname
                for cid, sname in camera_mapping.items()
                if sname in self.cfg.camera_names and cid in episode_data["observations"]
            }
        else:
            filtered_mapping = {
                cid: sname for cid, sname in camera_mapping.items() if cid in episode_data["observations"]
            }

        # Check if any cameras matched
        if not filtered_mapping:
            available_cameras = set(camera_mapping.values())
            print("⚠️  Warning: No cameras matched for this episode!")
            print(f"   Requested cameras: {self.cfg.camera_names}")
            print(f"   Available cameras: {list(available_cameras)}")
            return {}

        # Extract RGB images for each camera
        result = {}
        for cid, sname in filtered_mapping.items():
            if cid in episode_data["observations"]:
                result[sname] = episode_data["observations"][cid]

        return result

    def extract_lowdim_data(self, episode_data: dict[str, Any]):
        result = {}

        # Extract low-dimensional observations
        if episode_data["observations"]:
            result.update(
                {
                    key: value
                    for key, value in episode_data["observations"].items()
                    if len(value.shape) <= 2 or key.startswith(("robot__", "language_"))
                }
            )

        # Extract 'actions' if available
        if episode_data["actions"] and "actions" in episode_data["actions"] and self.action_key_fields:
            total_action_dim = episode_data["actions"]["actions"].shape[1]
            expected_action_dim = self.action_index_fields[-1]
            if total_action_dim < expected_action_dim:
                raise ValueError(
                    "Action tensor has insufficient dimension. "
                    f"Expected at least {expected_action_dim}, got {total_action_dim}."
                )

            prev_index = 0
            for key, index in zip(self.action_key_fields, self.action_index_fields, strict=False):
                result[key] = episode_data["actions"]["actions"][:, prev_index:index]
                prev_index = index

            if prev_index != expected_action_dim:
                raise ValueError(
                    "Action slicing did not consume expected dimensions. "
                    f"Expected {expected_action_dim}, consumed {prev_index}."
                )
        elif self.action_key_fields:
            raise ValueError(
                "Configured action fields but no actions were found in episode data. "
                "Ensure actions.npz is present for each episode."
            )

        return result

    def extract_intrinsics_extrinsics_data(self, episode_data: dict[str, Any]):
        return episode_data.get("intrinsics", {}), episode_data.get("extrinsics", {})

    def extract_sample_data(
        self,
        anchor_timestep: int,
        episode_path: str,
        episode_length: int,
        camera_data: dict[str, np.ndarray],
        lowdim_data: dict[str, np.ndarray],
        intrinsics_data: dict[str, np.ndarray],
        extrinsics_data: dict[str, np.ndarray],
        metadata_data: dict[str, Any],
        statistics_ray_actor,
        logger_actor,
    ):
        logger_actor.increment_total_potential_samples.remote()

        # Calculate windows
        lowdim_start = anchor_timestep - self.cfg.past_lowdim_steps
        lowdim_end = anchor_timestep + self.cfg.future_lowdim_steps

        # Check padding
        past_padding = max(0, -lowdim_start)
        future_padding = max(0, lowdim_end - episode_length + 1)

        if past_padding > self.cfg.max_padding_left or future_padding > self.cfg.max_padding_right:
            logger_actor.increment_padding_samples_filtered.remote()
            return None, None, None, None, None

        valid_start = max(0, lowdim_start)
        valid_end = min(episode_length - 1, lowdim_end)

        # Check if robot is stationary (e.g. to filter pauses)
        if self.cfg.filter_still_samples and is_still_sample(
            lowdim_data, valid_start, valid_end, self.cfg.still_threshold
        ):
            logger_actor.increment_still_samples_filtered.remote()
            return None, None, None, None, None

        # Extract images
        sample_images = {}
        actual_image_timesteps = []

        for img_offset in self.cfg.image_indices:
            img_timestep = np.clip(anchor_timestep + img_offset, 0, episode_length - 1)
            actual_image_timesteps.append(int(img_timestep))

            for camera_name, camera_images in camera_data.items():
                key = f"{camera_name}_t{img_offset}"
                sample_images[key] = camera_images[img_timestep]

        # Extract sequence-specific camera calibration first (needed for depth distortion)
        sample_intrinsics, sample_extrinsics = self.extract_sample_camera_calibration(
            intrinsics_data,
            extrinsics_data,
            valid_start,
            valid_end,
            past_padding,
            future_padding,
        )

        # Process lowdim data (which includes actions)
        sample_lowdim = {}
        reference_data = {}
        for key, data in lowdim_data.items():
            valid_data = data[valid_start : valid_end + 1]
            if past_padding > 0 or future_padding > 0:
                valid_data = self.pad_fn(valid_data, past_padding, future_padding)
            sample_lowdim[key] = valid_data
            actual_key = any_to_actual_key(key)
            if actual_key is not None and actual_key in lowdim_data:
                reference_data[key] = lowdim_data[actual_key][anchor_timestep]

        # Add relative lowdim data with respect to the actual position at the current timestep
        sample_lowdim_relative = self.create_relative_lowdim_data(sample_lowdim, reference_data)
        sample_lowdim.update(sample_lowdim_relative)

        # Create masks
        past_mask, future_mask = create_past_and_future_masks(
            anchor_timestep, self.cfg.past_lowdim_steps, self.cfg.future_lowdim_steps, episode_length
        )

        # Create metadata
        episode_id = self.get_episode_id(episode_path)
        sample_metadata = SampleMetadata(
            episode_id=episode_id,
            sample_id=f"{uuid.uuid4()}_{episode_id}_t{anchor_timestep:04d}",
            anchor_timestep=int(anchor_timestep),
            anchor_relative_idx=int(self.cfg.past_lowdim_steps),
            image_timesteps=actual_image_timesteps,
            lowdim_start_timestep=int(lowdim_start),
            lowdim_end_timestep=int(lowdim_end),
            past_padding=int(past_padding),
            future_padding=int(future_padding),
            camera_names=list(camera_data.keys()),
            original_episode_length=int(episode_length),
            original_image_sizes={},  # To be filled in after image resizing
            is_padded=bool(past_padding > 0 or future_padding > 0),
        )

        # Build stats_sample for batched statistics update (don't send immediately)
        stats_sample = None
        if statistics_ray_actor is not None:
            stats_sample = {
                "lowdim": {k: v.copy() for k, v in sample_lowdim.items()},  # Copy before modifying
                "past_mask": past_mask,
                "future_mask": future_mask,
            }

        # Add intrinsics, extrinsics, past_mask, future_mask to lowdim (after building stats_sample)
        # Only include cameras that have images (camera_data keys are semantic names)
        available_camera_names = set(camera_data.keys()) if camera_data else set()
        for key, value in sample_intrinsics.items():
            if not available_camera_names or key in available_camera_names:
                sample_lowdim[f"original_intrinsics.{key}"] = value
        for key, value in sample_extrinsics.items():
            if not available_camera_names or key in available_camera_names:
                sample_lowdim[f"extrinsics.{key}"] = value
        sample_lowdim["past_mask"] = past_mask
        sample_lowdim["future_mask"] = future_mask

        language_instructions = self.get_language_instructions(episode_path)

        return (
            sample_images,
            sample_lowdim,
            sample_metadata,
            language_instructions,
            stats_sample,
        )
