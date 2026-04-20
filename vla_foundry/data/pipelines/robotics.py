import os
import random

import numpy as np
import torch
import webdataset as wds

from vla_foundry.data.augmentations.decode_and_augment import Augmentations
from vla_foundry.data.pipelines.base import BaseWebDatasetPipeline
from vla_foundry.data.pipelines.webdataset_cache import get_tarfile_to_samples_stage
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.data.robotics.utils import crop_sequence
from vla_foundry.data.utils import deterministic_shuffle, log_and_continue
from vla_foundry.params.data_params import RoboticsDataParams


def filter_robotics_sample(sample):
    """Filter to ensure sample has required robotics data components."""
    has_lowdim = any(k.endswith("lowdim.npz") for k in sample)
    has_metadata = any(k.endswith("metadata.json") for k in sample)
    has_images = any(k.endswith(".jpg") for k in sample)
    return has_lowdim and has_metadata and has_images


def select_language_instruction(language_instructions, instruction_types):
    """Select a random language instruction from the specified types."""
    if not language_instructions or not instruction_types:
        return ""

    # Collect all instructions from the specified types
    available_instructions = []
    for instruction_type in instruction_types:
        if instruction_type in language_instructions:
            if isinstance(language_instructions[instruction_type], str):
                language_instructions[instruction_type] = [language_instructions[instruction_type]]
            available_instructions.extend(language_instructions[instruction_type])
    return random.choice(available_instructions) if available_instructions else ""


def extract_robotics_fields(
    sample,
    language_instruction_types=None,
    action_fields=None,
    proprioception_fields=None,
    intrinsics_fields=None,
    extrinsics_fields=None,
    lowdim_past_timesteps=None,
    lowdim_future_timesteps=None,
):
    """Extract robotics fields from sample."""
    if extrinsics_fields is None:
        extrinsics_fields = []
    if intrinsics_fields is None:
        intrinsics_fields = []
    if proprioception_fields is None:
        proprioception_fields = []
    if action_fields is None:
        action_fields = []

    images, data = {}, {}

    for key, value in sample.items():
        if key.endswith(".jpg"):
            # Extract camera name and timestep from key.
            # Use rsplit to handle camera names containing dots (e.g., "observation.image_t-1")
            img_key = key.rsplit(".", 1)[0]  # e.g., "observation.image_t-1"
            # Also store under a short key (without dotted prefix like "observation.images.")
            # so that image_names computed from camera_names match regardless of prefix.
            # The short key is the part after the last dot-separated segment that precedes
            # the camera+timestep pattern (e.g., "observation.images.cam_t0" -> "cam_t0").
            # We detect the timestep suffix "_t" to find where the camera name starts.
            parts = img_key.rsplit("_t", 1)
            if len(parts) == 2 and "." in parts[0]:
                short_key = parts[0].rsplit(".", 1)[-1] + "_t" + parts[1]
            else:
                short_key = img_key
            # Keep tensor images as tensors for tensor-native downstream paths.
            img = value if isinstance(value, torch.Tensor) else np.asarray(value)
            images[img_key] = img
            if short_key != img_key:
                images[short_key] = img
        else:
            suffix_map = ["lowdim.npz", "metadata.json", "language_instructions.json"]
            for suffix in suffix_map:
                if key.endswith(suffix):
                    data[suffix] = value

    instruction = select_language_instruction(data.get("language_instructions.json"), language_instruction_types)

    lowdim_data = data.get("lowdim.npz")
    metadata = data.get("metadata.json", {})

    # Get the anchor index from metadata (where the current timestep is in the sequence)
    original_anchor_idx = metadata.get("anchor_relative_idx", None)

    # Crop sequences if requested
    extracted_lowdim = {}
    for key in action_fields + proprioception_fields:
        field_data = lowdim_data.get(key)
        if (
            field_data is not None
            and original_anchor_idx is not None
            and lowdim_past_timesteps is not None
            and lowdim_future_timesteps is not None
        ):
            extracted_lowdim[key] = crop_sequence(
                field_data, original_anchor_idx, lowdim_past_timesteps, lowdim_future_timesteps
            )
        else:
            extracted_lowdim[key] = field_data

    # Also crop masks if cropping is enabled
    past_mask = lowdim_data.get("past_mask")
    future_mask = lowdim_data.get("future_mask")
    if original_anchor_idx is not None and lowdim_past_timesteps is not None and lowdim_future_timesteps is not None:
        if past_mask is not None:
            past_mask = crop_sequence(past_mask, original_anchor_idx, lowdim_past_timesteps, lowdim_future_timesteps)
        if future_mask is not None:
            future_mask = crop_sequence(
                future_mask, original_anchor_idx, lowdim_past_timesteps, lowdim_future_timesteps
            )

        # Update metadata with new anchor index after cropping
        # The new anchor is always at lowdim_past_timesteps in the cropped sequence
        metadata = metadata.copy()
        metadata["anchor_relative_idx"] = lowdim_past_timesteps
        # Store original anchor for alignment with normalization statistics
        metadata["original_anchor_relative_idx"] = original_anchor_idx

    return {
        "images": images,
        "lowdim": extracted_lowdim,
        "past_mask": past_mask,
        "future_mask": future_mask,
        "metadata": metadata,
        "intrinsics": {key: lowdim_data.get(key) for key in intrinsics_fields},
        "extrinsics": {key: lowdim_data.get(key) for key in extrinsics_fields},
        "language_instruction": instruction,
        "language_instruction_full": data.get("language_instructions.json", {}),
    }


class RoboticsPipeline(BaseWebDatasetPipeline):
    def __init__(self, modality, data_params: RoboticsDataParams, batch_size: int):
        super().__init__(modality, data_params, batch_size)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.data_params = data_params
        self.robotics_processor = RoboticsProcessor(data_params)
        self.augmentations = Augmentations(data_params.augmentation)

    def __len__(self):
        """Return the number of samples in the dataset (cached)."""
        if not hasattr(self, "_cached_num_samples"):
            num_samples = 0
            for i in range(len(self.data_params.dataset_manifest)):
                num_samples += self.data_params.dataset_manifest[i]["num_sequences"]
            self._cached_num_samples = num_samples
        return self._cached_num_samples

    def create_pipeline(self, datastring, checkpoint_num):
        cache_cfg = self.data_params.dataset_cache

        pipeline = [
            wds.SimpleShardList(datastring),
            deterministic_shuffle(
                bufsize=self.data_params.shuffle_buffer_size,
                initial=self.data_params.shuffle_initial,
                seed=self.data_params.seed,
                epoch=checkpoint_num,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            get_tarfile_to_samples_stage(
                cache_cfg=cache_cfg,
                handler=log_and_continue,
            ),
            wds.map(self.augmentations.decode_and_augment_sample, handler=log_and_continue),
            wds.select(filter_robotics_sample),
            wds.map(
                lambda sample: extract_robotics_fields(
                    sample,
                    language_instruction_types=self.data_params.language_instruction_types,
                    action_fields=self.data_params.action_fields,
                    proprioception_fields=self.data_params.proprioception_fields,
                    intrinsics_fields=self.data_params.intrinsics_fields,
                    extrinsics_fields=self.data_params.extrinsics_fields,
                    lowdim_past_timesteps=self.data_params.lowdim_past_timesteps,
                    lowdim_future_timesteps=self.data_params.lowdim_future_timesteps,
                ),
                handler=log_and_continue,
            ),
            wds.batched(self.batch_size, partial=False),
            wds.map(
                lambda batch: self.robotics_processor.process_inputs(
                    batch,
                    image_names=self.data_params.image_names,
                    max_text_seq_len=self.data_params.max_text_seq_len,
                ),
                handler=log_and_continue,
            ),
            wds.map(
                lambda batch: self.robotics_processor.add_action_and_proprioception_fields(
                    batch,
                    action_fields=self.data_params.action_fields,
                    proprioception_fields=self.data_params.proprioception_fields,
                ),
                handler=log_and_continue,
            ),
            wds.map(lambda batch: {**batch, "images": None}, handler=log_and_continue),  # Save memory
        ]

        return pipeline

    def save_configs(self, experiment_path: str):
        # Save normalizer config
        # Can be loaded with RoboticsNormalizer.load(config_path, statistics_path)
        if self.robotics_processor.normalizer is not None:
            self.robotics_processor.normalizer.save(experiment_path)

        # Save processor config
        # Can be loaded with RoboticsProcessor.load(config_path)
        self.robotics_processor.save(experiment_path)
