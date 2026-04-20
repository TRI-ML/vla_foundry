#!/usr/bin/env python3
"""
Raw Robotics Data Loader

This module provides functionality to load and explore raw (non-preprocessed) robotics data
in the format expected by the preprocessing pipeline.

The raw data format expects:
- Episode directories containing processed/ subdirectories
- metadata.yaml: Episode metadata including camera mappings
- observations.npz: All observation data including camera images and low-dim data
- intrinsics.npz: Camera intrinsic parameters (optional)
- extrinsics.npz: Camera extrinsic parameters (optional)
"""

import logging
import os
from typing import Any

import fsspec
import numpy as np
import yaml
from tqdm import tqdm


class RawRoboticsDataLoader:
    """Load and process raw robotics episode data for exploration."""

    def __init__(
        self,
        episode_paths: list[str],
        max_samples: int = -1,
        max_episodes_to_process: int = -1,
        camera_names: list[str] | None = None,
        trajectory_length: int = 50,
        stride: int = 1,
    ):
        self.episode_paths = episode_paths
        self.max_samples = max_samples
        self.max_episodes_to_process = max_episodes_to_process
        self.camera_names = camera_names
        self.trajectory_length = trajectory_length
        self.stride = stride
        self.samples = []

    @staticmethod
    def _convert_image_array_to_numpy(img_array: np.ndarray) -> np.ndarray:
        """Convert image array to proper numpy format for display."""
        if img_array.dtype != np.uint8:
            # Assume values are in [0, 1] range if not uint8
            img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)

        return img_array

    def load_episode_data(self, episode_path: str) -> dict[str, Any] | None:
        """Load raw episode data from directory structure."""
        processed_path = os.path.join(episode_path, "processed")

        try:
            # Load metadata
            metadata_path = os.path.join(processed_path, "metadata.yaml")
            with fsspec.open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)

            # Load observations
            obs_path = os.path.join(processed_path, "observations.npz")
            with fsspec.open(obs_path, "rb") as f:
                observations = np.load(f, allow_pickle=True)
                observations = {k: v for k, v in observations.items()}

            # Load camera calibration data (optional)
            intrinsics, extrinsics = {}, {}
            try:
                intrinsics_path = os.path.join(processed_path, "intrinsics.npz")
                extrinsics_path = os.path.join(processed_path, "extrinsics.npz")

                with fsspec.open(intrinsics_path, "rb") as f:
                    intrinsics_archive = np.load(f)
                    intrinsics = {key: intrinsics_archive[key] for key in intrinsics_archive.files}

                with fsspec.open(extrinsics_path, "rb") as f:
                    extrinsics_archive = np.load(f)
                    extrinsics = {key: extrinsics_archive[key] for key in extrinsics_archive.files}
            except Exception as e:
                logging.warning(f"Could not load camera calibration for {episode_path}: {e}")

            return {
                "episode_path": episode_path,
                "metadata": metadata,
                "observations": observations,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
            }

        except Exception as e:
            logging.error(f"Failed to load episode {episode_path}: {e}")
            return None

    def extract_camera_data(
        self, observations: dict[str, np.ndarray], metadata: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Extract camera data with semantic naming."""
        camera_mapping = metadata.get("camera_id_to_semantic_name", {})

        camera_data = {}
        for camera_id, semantic_name in camera_mapping.items():
            if camera_id in observations and self.camera_names is None or semantic_name in self.camera_names:
                camera_data[semantic_name] = observations[camera_id]

        return camera_data

    def extract_lowdim_data(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Extract low-dimensional data (non-image data)."""
        lowdim_data = {}

        for key, value in observations.items():
            # Skip image data (usually high-dimensional arrays)
            if len(value.shape) <= 2 or key.startswith(("robot__", "language_")):
                lowdim_data[key] = value

        return lowdim_data

    def transform_camera_calibration_keys(
        self, intrinsics: dict[str, Any], extrinsics: dict[str, Any], metadata: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform camera calibration keys from camera IDs to semantic names."""
        camera_mapping = metadata.get("camera_id_to_semantic_name", {})
        if not camera_mapping:
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

    def create_samples_from_episode(self, episode_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Create exploration samples from a raw episode."""
        samples = []

        episode_id = os.path.basename(episode_data["episode_path"].rstrip("/"))
        observations = episode_data["observations"]
        metadata = episode_data["metadata"]

        # Get episode length from first observation
        first_obs_key = next(iter(observations.keys()))
        episode_length = observations[first_obs_key].shape[0]

        # Extract camera and low-dim data
        camera_data = self.extract_camera_data(observations, metadata)
        lowdim_data = self.extract_lowdim_data(observations)

        # Transform camera calibration keys
        intrinsics, extrinsics = self.transform_camera_calibration_keys(
            episode_data["intrinsics"], episode_data["extrinsics"], metadata
        )

        # Create samples at specified stride
        for timestep in range(0, episode_length, self.stride):
            # Extract images for this timestep
            sample_images = {}
            for camera_name, camera_images in camera_data.items():
                if timestep < len(camera_images):
                    img_array = camera_images[timestep]
                    sample_images[camera_name] = self._convert_image_array_to_numpy(img_array)

            # Extract lowdim data slice around this timestep
            sample_lowdim = {}
            for key, data in lowdim_data.items():
                if timestep < len(data):
                    # For raw data exploration, we'll show a small window around the current timestep
                    start_idx = max(0, timestep - 5)
                    end_idx = min(len(data), timestep + 21)  # 5 past + current + 20 future
                    sample_lowdim[key] = data[start_idx:end_idx]

            # Create sample metadata
            sample_metadata = {
                "episode_id": episode_id,
                "sample_id": f"{episode_id}_t{timestep:04d}",
                "anchor_timestep": timestep,
                "lowdim_start_timestep": max(0, timestep - 5),
                "original_episode_length": episode_length,
                "camera_names": list(camera_data.keys()),
                "original_image_sizes": {},
                "is_raw_data": True,
            }

            # Add original image sizes
            for camera_name, img_array in sample_images.items():
                if len(img_array.shape) >= 2:
                    h, w = img_array.shape[:2]
                    sample_metadata["original_image_sizes"][camera_name] = (w, h)

            sample = {
                "images": sample_images,
                "lowdim": sample_lowdim,
                "metadata": sample_metadata,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "actions": {},  # Raw data doesn't have separate actions
            }

            samples.append(sample)

            # Stop if we've reached max samples
            if self.max_samples > 0 and len(samples) >= self.max_samples:
                break

        return samples

    def load_samples(self) -> list[dict[str, Any]]:
        """Load samples from raw episode data."""
        print(f"🔍 Loading raw episode data from {len(self.episode_paths)} episodes...")

        episodes_to_process = self.episode_paths
        if self.max_episodes_to_process > 0:
            episodes_to_process = episodes_to_process[: self.max_episodes_to_process]

        for episode_path in tqdm(episodes_to_process, desc="Loading episodes"):
            episode_data = self.load_episode_data(episode_path)
            if episode_data is None:
                continue

            episode_samples = self.create_samples_from_episode(episode_data)
            self.samples.extend(episode_samples)

            # Stop if we've reached max samples
            if self.max_samples > 0 and len(self.samples) >= self.max_samples:
                self.samples = self.samples[: self.max_samples]
                break

        print(f"✅ Loaded {len(self.samples)} samples from raw data")
        return self.samples


def discover_raw_episodes(source_paths: list[str], max_episodes_to_process: int = -1) -> list[str]:
    """Discover raw episode directories."""
    if isinstance(source_paths, str):
        source_paths = [source_paths]

    episodes = []

    for source_path in source_paths:
        fs, fsspec_path = fsspec.core.url_to_fs(source_path)

        try:
            # Check if source_path is already an episode directory
            if os.path.basename(source_path.rstrip("/")).startswith("episode_"):
                processed_path = os.path.join(source_path, "processed")
                if fs.exists(processed_path if not source_path.startswith("s3://") else processed_path[5:]):
                    episodes.append(source_path)
                    if max_episodes_to_process > 0 and len(episodes) >= max_episodes_to_process:
                        return sorted(episodes)
                continue

            # List items in source directory
            items = fs.listdir(fsspec_path)

            for item in items:
                item_path = item["name"] if isinstance(item, dict) else item
                full_item_path = item_path if not source_path.startswith("s3://") else f"s3://{item_path}"

                item_name = os.path.basename(full_item_path.rstrip("/"))

                # Check if this is an episode directory
                if item_name.startswith("episode_"):
                    processed_path = os.path.join(full_item_path, "processed")
                    fs_processed_path = processed_path if not full_item_path.startswith("s3://") else processed_path[5:]

                    try:
                        if fs.exists(fs_processed_path):
                            episodes.append(full_item_path)
                            if max_episodes_to_process > 0 and len(episodes) >= max_episodes_to_process:
                                return sorted(episodes)
                    except Exception:
                        continue

                # Also check subdirectories for episode folders
                try:
                    if not item_name.startswith("episode_"):
                        sub_items = fs.listdir(item_path if not source_path.startswith("s3://") else item_path)
                        for sub_item in sub_items:
                            sub_item_path = sub_item["name"] if isinstance(sub_item, dict) else sub_item
                            if not source_path.startswith("s3://"):
                                full_sub_path = sub_item_path
                            else:
                                full_sub_path = f"s3://{sub_item_path}"

                            sub_name = os.path.basename(full_sub_path.rstrip("/"))
                            if sub_name.startswith("episode_"):
                                processed_path = os.path.join(full_sub_path, "processed")
                                fs_processed_path = (
                                    processed_path if not full_sub_path.startswith("s3://") else processed_path[5:]
                                )

                                try:
                                    if fs.exists(fs_processed_path):
                                        episodes.append(full_sub_path)
                                        if max_episodes_to_process > 0 and len(episodes) >= max_episodes_to_process:
                                            return sorted(episodes)
                                except Exception:
                                    continue
                except Exception:
                    continue

        except Exception as e:
            print(f"Error scanning source path {source_path}: {e}")

    return sorted(episodes)
