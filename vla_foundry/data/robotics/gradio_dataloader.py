#!/usr/bin/env python3
"""
Gradio-based Robotics Data Explorer

This tool provides a gradio web interface for exploring preprocessed robotics data
with interactive trajectory visualization overlaid on camera images.

Usage:
    python vla_foundry/data/robotics/data_explorer_gradio.py \
        --dataset-path /path/to/processed/dataset/ \
        --max-samples 100 \
        --port 7860

Features:
    - Modern web-based interface using Gradio
    - Interactive sliders and dropdowns
    - Real-time trajectory overlay on camera images
    - 3D trajectory visualization
    - Multiple camera support
    - Gripper state visualization
    - Sample metadata display
"""

import json
import logging
import os
import traceback
from types import SimpleNamespace
from typing import Any

import fsspec
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

from vla_foundry.data.dataloader import get_datastring_input, get_wds_dataloader
from vla_foundry.data.pipelines.robotics import extract_robotics_fields
from vla_foundry.data.robotics.normalization import RoboticsNormalizer
from vla_foundry.data.robotics.utils import (
    any_to_actual_key,
    apply_relative_pose,
    load_action_field_config,
    pose_to_9d,
    rot_6d_to_matrix,
    to_pose_matrix,
)
from vla_foundry.params.data_params import RoboticsDataParams

ACTION_FIELDS_CONFIG_PATH = "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml"


def build_action_field_lookup(action_field_keys: list[str]) -> dict[str, str | None]:
    """Build a lookup dictionary for action fields based on their semantic names."""
    lookup: dict[str, str | None] = {
        "left_xyz": None,
        "left_rot_6d": None,
        "left_gripper": None,
        "right_xyz": None,
        "right_rot_6d": None,
        "right_gripper": None,
    }

    for field_name in action_field_keys:
        prefix, _, suffix = field_name.partition("::")
        normalized_prefix = prefix.lower()
        normalized_suffix = suffix.lower()

        if "left" in normalized_prefix:
            if "poses__left" in normalized_prefix and normalized_suffix.endswith("__xyz"):
                lookup["left_xyz"] = field_name
            elif "poses__left" in normalized_prefix and normalized_suffix.endswith("__rot_6d"):
                lookup["left_rot_6d"] = field_name
            elif "grippers__left" in normalized_prefix or normalized_suffix.endswith("hand"):
                lookup["left_gripper"] = field_name
        elif "right" in normalized_prefix:
            if "poses__right" in normalized_prefix and normalized_suffix.endswith("__xyz"):
                lookup["right_xyz"] = field_name
            elif "poses__right" in normalized_prefix and normalized_suffix.endswith("__rot_6d"):
                lookup["right_rot_6d"] = field_name
            elif "grippers__right" in normalized_prefix or normalized_suffix.endswith("hand"):
                lookup["right_gripper"] = field_name

    return lookup


def arm_xyz_to_gripper_xyz(
    arm_xyz: np.ndarray,
    arm_rot_6d: np.ndarray,
    gripper_offset: float = 0.22,
) -> np.ndarray:
    """Convert arm xyz and 6D rotation to gripper xyz using the standard convention."""

    rot_mat = rot_6d_to_matrix(arm_rot_6d)
    gripper_offset_vector = np.array([0.0, 0.0, gripper_offset])

    gripper_xyz = arm_xyz + np.matmul(rot_mat, gripper_offset_vector)

    return gripper_xyz


def _reconstruct_relative_coordinates(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Reconstruct absolute coordinates from relative coordinates by adding back the reference position.
    Uses robot__actual as the reference for ALL trajectory types.

    Args:
        sample: Sample data dictionary

    Returns:
        Dictionary with reconstructed absolute coordinates
    """

    reconstructed = {}
    lowdim = sample["lowdim"]

    # Get metadata to find the reference timestep (last past timestep)
    metadata = sample.get("metadata", {})
    # The reference index for relative coordinates is anchor_relative_idx - 1
    # (anchor_relative_idx points to the anchor, but relative coords are computed relative to the previous timestep)
    anchor_relative_idx = metadata.get("anchor_relative_idx", 1)
    reference_index = anchor_relative_idx

    # Find all relative coordinate keys
    relative_keys = [key for key in lowdim if key.endswith("_relative")]

    # Group relative keys by pose groups (xyz + rot_6d pairs)
    pose_groups = {}
    standalone_keys = []

    for relative_key in relative_keys:
        absolute_key = relative_key.replace("_relative", "")
        if absolute_key not in lowdim:
            continue

        if "xyz" in absolute_key.lower():
            # This is a position field - find its rotation counterpart
            base_key = absolute_key.replace("__xyz", "")
            rot_key = base_key + "__rot_6d"
            rot_relative_key = rot_key + "_relative"

            if rot_relative_key in relative_keys and rot_key in lowdim:
                # This is a pose group
                pose_groups[base_key] = {
                    "xyz_key": absolute_key,
                    "rot_key": rot_key,
                    "xyz_relative_key": relative_key,
                    "rot_relative_key": rot_relative_key,
                }
            else:
                # Standalone xyz without rotation
                standalone_keys.append(relative_key)
        elif "rot_6d" in absolute_key.lower():
            # Check if already processed as part of a pose group
            base_key = absolute_key.replace("__rot_6d", "")
            if base_key not in pose_groups:
                standalone_keys.append(relative_key)
        else:
            standalone_keys.append(relative_key)

    # Process pose groups (xyz + rot_6d pairs)
    for _base_key, keys in pose_groups.items():
        xyz_key = keys["xyz_key"]
        rot_key = keys["rot_key"]
        xyz_relative_key = keys["xyz_relative_key"]
        rot_relative_key = keys["rot_relative_key"]

        # Get reference keys
        reference_xyz_key = any_to_actual_key(xyz_key)
        reference_rot_key = any_to_actual_key(rot_key)

        if reference_xyz_key not in lowdim or reference_rot_key not in lowdim:
            continue

        # Get reference pose at reference timestep
        reference_xyz = lowdim[reference_xyz_key][reference_index]
        reference_rot = lowdim[reference_rot_key][reference_index]
        reference_pose_matrix = to_pose_matrix(reference_xyz, reference_rot)

        # Get relative poses
        relative_xyz = lowdim[xyz_relative_key]
        relative_rot = lowdim[rot_relative_key]
        relative_pose_matrices = to_pose_matrix(relative_xyz, relative_rot)

        # Convert to absolute
        absolute_pose_matrices = apply_relative_pose(relative_pose_matrices, reference_pose_matrix)
        absolute_xyz, absolute_rot = pose_to_9d(absolute_pose_matrices)

        reconstructed[xyz_key] = absolute_xyz
        reconstructed[rot_key] = absolute_rot

    # Process standalone relative fields (if any remain)
    for relative_key in standalone_keys:
        absolute_key = relative_key.replace("_relative", "")
        reference_key = any_to_actual_key(absolute_key)

        if reference_key is None or reference_key not in lowdim:
            continue

        # For standalone fields, just add the reference back
        reference_value = lowdim[reference_key][reference_index]
        reconstructed[absolute_key] = lowdim[relative_key] + reference_value

    return reconstructed


def extract_trajectories(
    sample: dict[str, Any],
    include_desired: bool = False,
    include_action: bool = False,
    use_reconstructed: bool = False,
) -> dict[str, np.ndarray]:
    """Extract robot trajectories from a sample.

    Args:
        sample: Sample data dictionary
        include_desired: If True, include both actual and desired poses
        include_action: If True, include actions
        use_reconstructed: If True, use reconstructed coordinates from relative coordinates
    """
    trajectories = {}

    lowdim = sample["lowdim"]

    # Check if we have relative coordinates and reconstruct absolute coordinates
    if use_reconstructed:
        reconstructed_coords = _reconstruct_relative_coordinates(sample)
        if reconstructed_coords:
            # Use reconstructed coordinates for trajectory extraction
            lowdim = {**lowdim, **reconstructed_coords}

    # Extract left arm trajectory
    left_xyz_key = "robot__actual__poses__left::panda__xyz"
    left_gripper_key = "robot__actual__grippers__left::panda_hand"
    left_6d_key = "robot__actual__poses__left::panda__rot_6d"
    if left_xyz_key in lowdim:
        trajectories["left_arm_xyz"] = lowdim[left_xyz_key]
        trajectories["left_arm_gripper"] = lowdim[left_gripper_key]
        trajectories["left_arm_6d"] = lowdim[left_6d_key]
        trajectories["left_gripper_xyz"] = arm_xyz_to_gripper_xyz(
            trajectories["left_arm_xyz"], trajectories["left_arm_6d"]
        )

        # Add desired poses if requested and available
        if include_desired:
            left_desired_xyz_key = "robot__desired__poses__left::panda__xyz"
            left_desired_gripper_key = "robot__desired__grippers__left::panda_hand"
            left_desired_6d_key = "robot__desired__poses__left::panda__rot_6d"

            if left_desired_xyz_key in lowdim:
                trajectories["left_arm_xyz_desired"] = lowdim[left_desired_xyz_key]
                trajectories["left_arm_gripper_desired"] = lowdim[left_desired_gripper_key]
                trajectories["left_arm_6d_desired"] = lowdim[left_desired_6d_key]
                trajectories["left_gripper_xyz_desired"] = arm_xyz_to_gripper_xyz(
                    trajectories["left_arm_xyz_desired"],
                    trajectories["left_arm_6d_desired"],
                )

        if include_action:
            left_action_xyz_key = "robot__action__poses__left::panda__xyz"
            left_action_gripper_key = "robot__action__grippers__left::panda_hand"
            left_action_6d_key = "robot__action__poses__left::panda__rot_6d"
            if left_action_xyz_key in lowdim:
                trajectories["left_arm_xyz_action"] = lowdim[left_action_xyz_key]
                trajectories["left_arm_gripper_action"] = lowdim[left_action_gripper_key]
                trajectories["left_arm_6d_action"] = lowdim[left_action_6d_key]
                trajectories["left_gripper_xyz_action"] = arm_xyz_to_gripper_xyz(
                    trajectories["left_arm_xyz_action"], trajectories["left_arm_6d_action"]
                )

    # Extract right arm trajectory
    right_xyz_key = "robot__actual__poses__right::panda__xyz"
    right_gripper_key = "robot__actual__grippers__right::panda_hand"
    right_6d_key = "robot__actual__poses__right::panda__rot_6d"
    if right_xyz_key in lowdim:
        trajectories["right_arm_xyz"] = lowdim[right_xyz_key]
        trajectories["right_arm_gripper"] = lowdim[right_gripper_key]
        trajectories["right_arm_6d"] = lowdim[right_6d_key]
        trajectories["right_gripper_xyz"] = arm_xyz_to_gripper_xyz(
            trajectories["right_arm_xyz"], trajectories["right_arm_6d"]
        )

        # Add desired poses if requested and available
        if include_desired:
            right_desired_xyz_key = "robot__desired__poses__right::panda__xyz"
            right_desired_gripper_key = "robot__desired__grippers__right::panda_hand"
            right_desired_6d_key = "robot__desired__poses__right::panda__rot_6d"

            if right_desired_xyz_key in lowdim:
                trajectories["right_arm_xyz_desired"] = lowdim[right_desired_xyz_key]
                trajectories["right_arm_gripper_desired"] = lowdim[right_desired_gripper_key]
                trajectories["right_arm_6d_desired"] = lowdim[right_desired_6d_key]
                trajectories["right_gripper_xyz_desired"] = arm_xyz_to_gripper_xyz(
                    trajectories["right_arm_xyz_desired"],
                    trajectories["right_arm_6d_desired"],
                )

        if include_action:
            right_action_xyz_key = "robot__action__poses__right::panda__xyz"
            right_action_gripper_key = "robot__action__grippers__right::panda_hand"
            right_action_6d_key = "robot__action__poses__right::panda__rot_6d"
            if right_action_xyz_key in lowdim:
                trajectories["right_arm_xyz_action"] = lowdim[right_action_xyz_key]
                trajectories["right_arm_gripper_action"] = lowdim[right_action_gripper_key]
                trajectories["right_arm_6d_action"] = lowdim[right_action_6d_key]
                trajectories["right_gripper_xyz_action"] = arm_xyz_to_gripper_xyz(
                    trajectories["right_arm_xyz_action"],
                    trajectories["right_arm_6d_action"],
                )

    return trajectories


class RoboticsDataLoader:
    """Load and process robotics WebDataset data using the training pipeline."""

    def __init__(
        self,
        params: RoboticsDataParams,
        max_samples: int = -1,
        max_shards: int = -1,
        use_dataloader: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        augmentation_params: object | None = None,
    ):
        self.params = params
        self.max_samples = max_samples
        self.max_shards = max_shards
        self.use_dataloader = use_dataloader
        self.samples = []
        self.device = device  # Used for model predictions only
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.augmentation_params = augmentation_params
        # Initialize normalizer for denormalization if normalization is enabled
        self.normalizer = None
        if self.params.normalization.enabled:
            self.normalizer = RoboticsNormalizer(
                normalization_params=self.params.normalization,
                statistics_path=self.params.dataset_statistics,
            )
            if self.normalizer:
                logging.info("RoboticsDataLoader: Normalizer initialized for denormalization")
            else:
                logging.warning("RoboticsDataLoader: Failed to initialize normalizer for denormalization")

        action_field_config = load_action_field_config(ACTION_FIELDS_CONFIG_PATH)
        self.action_field_lookup = build_action_field_lookup(action_field_config["action_key_fields"])

    @staticmethod
    def _convert_image_tensor_to_numpy(img_tensor):
        """Convert image tensor to numpy array with proper pixel range handling."""
        img_numpy = img_tensor.permute(1, 2, 0).float().cpu().numpy()  # [H, W, C]

        # Handle different pixel value ranges
        min_val = img_numpy.min()
        max_val = img_numpy.max()
        img_numpy = img_numpy - min_val
        img_numpy = img_numpy / (max_val - min_val)
        img_numpy = img_numpy * 255
        img_numpy = img_numpy.astype(np.uint8)
        return img_numpy

    def load_samples(self) -> list[dict[str, Any]]:
        """Fallback method to load samples directly from shards."""

        # Try to load manifest
        manifest = []
        try:
            for manifest_path in self.params.dataset_manifest:
                with fsspec.open(manifest_path, "r") as f:
                    manifest.extend([json.loads(line.strip()) for line in f if line.strip()])
        except Exception as e:
            print(f"Warning: Could not load manifest: {e}")
            return []

        # Apply shard limit
        if self.max_shards > 0:
            shards_to_load = manifest[: self.max_shards]
            print(f"Loading samples from {len(shards_to_load)} shards (limited from {len(manifest)} total)...")
        else:
            shards_to_load = manifest  # Load all available shards
            print(f"Loading samples from {len(shards_to_load)} shards...")

        for shard_info in tqdm(shards_to_load, desc="Loading shards"):
            base_path = self.params.dataset_manifest[0].rsplit("/", 1)[0]
            shard_path = os.path.join(base_path, f"{shard_info['shard']}.tar")

            try:
                dataset = (
                    wds.WebDataset(shard_path)
                    .decode("pilrgb")
                    .map(
                        lambda sample: extract_robotics_fields(
                            sample,
                            action_fields=self.params.action_fields,
                            proprioception_fields=self.params.proprioception_fields,
                            intrinsics_fields=self.params.intrinsics_fields,
                            extrinsics_fields=self.params.extrinsics_fields,
                        )
                    )
                )

                for sample in dataset:
                    self.samples.append(sample)

                    if self.max_samples > 0 and len(self.samples) >= self.max_samples:
                        break

            except Exception as e:
                print(f"Warning: Failed to load shard {shard_path}: {e}")
                continue

            if self.max_samples > 0 and len(self.samples) >= self.max_samples:
                break

        return self.samples

    def load_samples_auto(self) -> list[dict[str, Any]]:
        """Load samples using the preferred method (dataloader or direct file loading)."""
        if self.use_dataloader:
            print("🔧 Loading samples using dataloader pipeline...")
            return self.load_samples_from_dataloader()
        else:
            print("📁 Loading samples directly from files...")
            return self.load_samples()

    def load_samples_from_dataloader(self) -> list[dict[str, Any]]:
        """Load samples using the data loader and denormalize the output."""
        # Create config for dataloader
        cfg = SimpleNamespace()
        cfg.distributed = SimpleNamespace(world_size=1, rank=0)
        cfg.vit = SimpleNamespace()
        # Create a copy of params with the correct batch size for the data explorer
        cfg.data = self.params
        cfg.hparams = SimpleNamespace(global_batch_size=1)
        cfg.augmentations = self.augmentation_params

        # Load all samples
        num_samples = -1

        # Use get_datastring_input to generate datastrings
        curr_shard_idx_per_dataset = [0]
        shard_shuffle_seed_per_dataset = [0]
        manifest_paths = self.params.dataset_manifest
        dataset_weighting = [1]  # Single dataset
        allow_multiple_epochs = False
        num_workers_per_gpu = 1
        world_size = 1

        (
            datastrings,
            num_samples_list_per_dataset,
            next_shard_idx_per_dataset,
            next_shard_shuffle_seed_per_dataset,
        ) = get_datastring_input(
            num_samples,
            curr_shard_idx_per_dataset,
            shard_shuffle_seed_per_dataset,
            manifest_paths,
            dataset_weighting,
            allow_multiple_epochs,
            num_workers_per_gpu,
            world_size,
        )
        print(f"Creating dataloader with datastrings: {datastrings}")
        print(f"Num samples per dataset: {num_samples_list_per_dataset}")

        # Create dataloader
        dataloader_info = get_wds_dataloader(
            datastrings=datastrings,
            num_samples_per_dataset=num_samples_list_per_dataset,
            checkpoint_num=0,
            cfg=cfg,
        )

        dataloader = dataloader_info.dataloader
        print(f"Dataloader created with {dataloader.num_batches} batches, {dataloader.num_samples} samples")
        print(f"Dataloader type: {type(dataloader)}")
        print(f"Dataloader has __iter__: {hasattr(dataloader, '__iter__')}")
        print(f"Dataloader has __len__: {hasattr(dataloader, '__len__')}")

        samples = []

        # Load samples from dataloader
        batch_count = 0
        try:
            print("Starting to iterate over dataloader...")

            for batch in dataloader:
                batch_count += 1

                if isinstance(batch, dict):
                    # Convert batch to individual samples
                    batch_samples = self._batch_to_samples(batch)
                    samples.extend(batch_samples)

                    # Stop if we have enough samples
                    if self.max_samples > 0 and len(samples) >= self.max_samples:
                        samples = samples[: self.max_samples]
                        break

                else:
                    print(f"WARNING: Batch is not a dict, it's a {type(batch)}")
            print(f"Loaded {len(samples)} samples")

        except Exception as e:
            print(f"Error loading samples from dataloader: {e}")
            traceback.print_exc()

        print(f"Finished loading. Total batches processed: {batch_count}, Total samples: {len(samples)}")

        return samples

    def load_samples_from_model_predictions(self, model, num_inference_steps=10, cfg=None):
        """Load samples using the data loader and denormalize the output."""
        # Create config for dataloader
        if cfg is None:
            cfg = SimpleNamespace()
            cfg.distributed = SimpleNamespace(world_size=1, rank=0)
            cfg.vit = SimpleNamespace()
            # Create a copy of params with the correct batch size for the data explorer
            cfg.data = self.params
            cfg.hparams = SimpleNamespace(global_batch_size=1)
            cfg.augmentations = self.augmentation_params

        # Load all samples
        num_samples = -1

        # Use get_datastring_input to generate datastrings
        curr_shard_idx_per_dataset = [0]
        shard_shuffle_seed_per_dataset = [0]
        manifest_paths = self.params.dataset_manifest
        dataset_weighting = [1]  # Single dataset
        allow_multiple_epochs = False
        num_workers_per_gpu = 1
        world_size = 1

        (
            datastrings,
            num_samples_list_per_dataset,
            next_shard_idx_per_dataset,
            next_shard_shuffle_seed_per_dataset,
        ) = get_datastring_input(
            num_samples,
            curr_shard_idx_per_dataset,
            shard_shuffle_seed_per_dataset,
            manifest_paths,
            dataset_weighting,
            allow_multiple_epochs,
            num_workers_per_gpu,
            world_size,
        )
        print(f"Creating dataloader with datastrings: {datastrings}")
        print(f"Num samples per dataset: {num_samples_list_per_dataset}")

        # Create dataloader
        dataloader_info = get_wds_dataloader(
            datastrings=datastrings, num_samples_per_dataset=num_samples_list_per_dataset, checkpoint_num=0, cfg=cfg
        )

        dataloader = dataloader_info.dataloader

        samples = []

        # Load samples from dataloader
        batch_count = 0
        try:
            print("Starting to iterate over dataloader...")

            for batch in dataloader:
                batch_count += 1

                if isinstance(batch, dict):
                    model.eval()
                    # Don't pass the ground truth actions to the model to avoid cheating
                    actions = batch["actions"].to(self.device) * batch["past_mask"][:, :, None].to(
                        self.device, dtype=self.dtype
                    ) + torch.randn_like(batch["actions"].to(self.device, dtype=self.dtype)) * (
                        batch["past_mask"][:, :, None].to(self.device, dtype=self.dtype) == 0
                    )
                    # actions = batch["actions"].to(self.device)
                    proprioception = batch.get("proprioception")
                    if proprioception is not None:
                        proprioception = proprioception.to(self.device, dtype=self.dtype)
                    with torch.no_grad():
                        # Generate actions from model
                        predictions = model.generate_actions(
                            batch["input_ids"].to(self.device, dtype=torch.long),
                            batch["pixel_values"].to(self.device, dtype=self.dtype),
                            actions.to(self.device, dtype=self.dtype),
                            batch["attention_mask"].to(self.device, dtype=torch.bool),
                            past_mask=batch["past_mask"].to(self.device, dtype=torch.bool),
                            proprioception=proprioception,
                            num_inference_steps=num_inference_steps,
                        )

                    # Replace actions with generated actions
                    start_idx = 0
                    for action_field in self.params.action_fields:
                        end_idx = start_idx + batch["lowdim"][action_field].shape[-1]
                        difference = (
                            (predictions[:, :, start_idx:end_idx] - batch["lowdim"][action_field].to(self.device))
                            .abs()
                            .mean()
                        )
                        print(f"Normalized difference of {action_field} is {difference}")
                        batch["lowdim"][action_field] = predictions[:, :, start_idx:end_idx]
                        start_idx = end_idx

                    # Convert batch to individual samples
                    batch_samples = self._batch_to_samples(batch)
                    samples.extend(batch_samples)

                    # Stop if we have enough samples
                    if self.max_samples > 0 and len(samples) >= self.max_samples:
                        samples = samples[: self.max_samples]
                        break

                else:
                    print(f"WARNING: Batch is not a dict, it's a {type(batch)}")
            print(f"Loaded {len(samples)} samples")

        except Exception as e:
            print(f"Error loading samples from dataloader: {e}")
            traceback.print_exc()

        print(f"Finished loading. Total batches processed: {batch_count}, Total samples: {len(samples)}")

        return samples

    def _denormalize_lowdim_data(self, lowdim_data: dict[str, Any]) -> dict[str, Any]:
        """Denormalize lowdim data if normalizer is available."""
        if self.normalizer is None:
            return lowdim_data

        denormalized_lowdim = {}
        for field_name, tensor in lowdim_data.items():
            if isinstance(tensor, (np.ndarray, torch.Tensor)):
                # Convert numpy array to torch tensor for denormalization
                tensor_torch = torch.from_numpy(tensor)
                denormalized_tensor = self.normalizer.denormalize_tensor(tensor_torch, field_name)
                denormalized_lowdim[field_name] = denormalized_tensor.float().cpu().numpy()
            else:
                denormalized_lowdim[field_name] = tensor

        return denormalized_lowdim

    def _batch_to_samples(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a batch to a list of individual samples."""
        samples = []
        batch_size = batch["input_ids"].shape[0]

        if batch_size == 0:
            return samples

        for i in range(batch_size):
            sample = {}

            # Extract metadata
            if "metadata" in batch and i < len(batch["metadata"]):
                sample["metadata"] = batch["metadata"][i]
            # Add __key__ propagation
            if "__key__" in batch and i < len(batch["__key__"]):
                sample["__key__"] = batch["__key__"][i]

            # Extract lowdim data
            if "lowdim" in batch:
                sample["lowdim"] = {}
                for key, value in batch["lowdim"].items():
                    if isinstance(value, torch.Tensor):
                        # Extract single sample from batch tensor
                        if value.dim() > 0:
                            sample["lowdim"][key] = value[i].float().cpu().numpy()
                        else:
                            sample["lowdim"][key] = value.float().cpu().numpy()
                    else:
                        sample["lowdim"][key] = value

                # Denormalize lowdim data if normalization is enabled
                if self.params.normalization.enabled and self.normalizer is not None:
                    sample["lowdim"] = self._denormalize_lowdim_data(sample["lowdim"])

            # Extract images - handle the processed pixel_values from the pipeline
            if "pixel_values" in batch and batch["pixel_values"] is not None:
                # The pipeline processes images into pixel_values tensor
                # Shape: [batch_size, num_images, channels, height, width]
                pixel_values = batch["pixel_values"]
                if pixel_values.dim() == 5:  # [B, N, C, H, W]
                    num_images = pixel_values.shape[1]
                    sample["images"] = {}

                    # Get camera names from metadata or batch
                    camera_names = batch.get("camera_names", [])
                    if not camera_names and "metadata" in sample:
                        camera_names = sample["metadata"].get("camera_names", [])

                    # Convert each image back to numpy and store with camera name
                    for img_idx in range(num_images):
                        if img_idx < len(camera_names):
                            camera_name = camera_names[img_idx]
                            # Convert tensor to numpy: [C, H, W] -> [H, W, C]
                            img_tensor = pixel_values[i, img_idx]  # [C, H, W]
                            img_numpy = RoboticsDataLoader._convert_image_tensor_to_numpy(img_tensor)
                            sample["images"][camera_name] = img_numpy
                        else:
                            # Fallback naming if camera names not available
                            sample["images"][f"camera_{img_idx}"] = img_tensor.permute(1, 2, 0).float().cpu().numpy()

            # Extract past and future masks
            if "past_mask" in batch:
                sample["past_mask"] = batch["past_mask"][i].float().cpu().numpy()
            if "future_mask" in batch:
                sample["future_mask"] = batch["future_mask"][i].float().cpu().numpy()

            # Extract actions
            if "actions" in batch:
                # Denormalize actions data if normalization is enabled
                if self.params.normalization.enabled and self.normalizer is not None:
                    denormalized_actions = self.normalizer.denormalize_batch(
                        batch["actions"], self.params.action_fields
                    )
                    sample["actions"] = denormalized_actions[i].float().cpu().numpy()
                else:
                    sample["actions"] = batch["actions"][i].float().cpu().numpy()

            # Extract calibration data (intrinsics and extrinsics)
            if "intrinsics" in batch:
                sample["intrinsics"] = {}
                for key, value in batch["intrinsics"][i].items():
                    if isinstance(value, torch.Tensor):
                        sample["intrinsics"][key] = value[i].float().cpu().numpy()
                    else:
                        sample["intrinsics"][key] = value

            if "extrinsics" in batch:
                sample["extrinsics"] = {}
                for key, value in batch["extrinsics"][i].items():
                    if isinstance(value, torch.Tensor):
                        sample["extrinsics"][key] = value[i].float().cpu().numpy()
                    else:
                        sample["extrinsics"][key] = value

            if "language_instructions" in batch:
                print(f"Found language instructions in batch: {batch['language_instructions'][i]}")
                sample["language_instructions"] = batch["language_instructions"][i]

            if "__key__" in batch and i < len(batch["__key__"]):
                sample["__key__"] = batch["__key__"][i]

            samples.append(sample)

        return samples
