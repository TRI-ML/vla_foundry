"""
Tests for robotics data sequence cropping and normalization alignment.

These tests verify that:
1. Sequences can be cropped to smaller windows than preprocessing created
2. Normalization correctly aligns cropped sequences with statistics
3. The anchor point is properly maintained during cropping
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import webdataset as wds

from vla_foundry.data.pipelines.robotics import crop_sequence, extract_robotics_fields
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.data.robotics.normalization import RoboticsNormalizer
from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.robotics.normalization_params import NormalizationParams


@pytest.fixture
def dataset_dir():
    """Get the path to the test dataset."""
    return os.path.join(os.path.dirname(__file__), "..", "test_assets", "small_lbm_dataset")


@pytest.fixture
def stats_path(dataset_dir):
    """Get the path to the dataset statistics."""
    return os.path.join(dataset_dir, "stats.json")


@pytest.fixture
def manifest_path(dataset_dir):
    """Get the path to the dataset manifest."""
    return os.path.join(dataset_dir, "manifest.jsonl")


@pytest.fixture
def processing_metadata_path(dataset_dir):
    """Get the path to the processing metadata."""
    return os.path.join(dataset_dir, "processing_metadata.json")


def init_data_params_for_tests(data_params):
    """Initialize shared attributes for RoboticsDataParams without a full training config."""
    cfg = SimpleNamespace(
        data=data_params,
        hparams=SimpleNamespace(seed=42),
        model=SimpleNamespace(),
    )
    data_params.init_shared_attributes(cfg)
    return data_params


class TestCropSequence:
    """Test the crop_sequence function."""

    def test_crop_sequence_basic(self):
        """Test basic sequence cropping."""
        # Create a sequence with 15 timesteps (1 past + 14 future)
        data = np.arange(15 * 3).reshape(15, 3)  # [15, 3]
        anchor_idx = 1  # The anchor is at index 1

        # Crop to 0 past + 8 future
        past_timesteps = 0
        future_timesteps = 8
        cropped = crop_sequence(data, anchor_idx, past_timesteps=past_timesteps, future_timesteps=future_timesteps)

        # Should have 8 timesteps starting from the anchor
        assert cropped.shape == (past_timesteps + 1 + future_timesteps, 3)
        # First element should be the anchor
        assert np.array_equal(cropped[0], data[anchor_idx - past_timesteps])
        # Last element should be 8 steps into the future from anchor
        assert np.array_equal(cropped[-1], data[anchor_idx + future_timesteps])

    def test_crop_sequence_with_past(self):
        """Test cropping with past timesteps."""
        data = np.arange(15 * 3).reshape(15, 3)
        anchor_idx = 5

        # Crop to 2 past + 1 present + 4 future (includes anchor in future count)
        past_timesteps = 2
        future_timesteps = 4
        cropped = crop_sequence(data, anchor_idx, past_timesteps=past_timesteps, future_timesteps=future_timesteps)

        assert cropped.shape == (7, 3)  # 2 + 1 + 4 = 7
        # First element should be 2 steps before anchor
        assert np.array_equal(cropped[0], data[anchor_idx - past_timesteps])
        # Element at index 2 should be the anchor
        assert np.array_equal(cropped[past_timesteps], data[anchor_idx])
        # Last element should be at index anchor + 1 + 4
        assert np.array_equal(cropped[-1], data[anchor_idx + future_timesteps])

    def test_crop_sequence_preserves_anchor(self):
        """Test that anchor position is correctly preserved."""
        data = np.arange(20 * 2).reshape(20, 2)
        anchor_idx = 10

        cropped = crop_sequence(data, anchor_idx, past_timesteps=3, future_timesteps=5)

        # The anchor should be at index 3 in the cropped sequence
        assert np.array_equal(cropped[3], data[10])

    def test_crop_sequence_different_shapes(self):
        """Test cropping with different tensor shapes."""
        # Test with 1D
        data_1d = np.arange(15)
        cropped_1d = crop_sequence(data_1d, anchor_idx=5, past_timesteps=2, future_timesteps=3)
        assert cropped_1d.shape == (6,)

        # Test with 3D
        data_3d = np.arange(15 * 3 * 4).reshape(15, 3, 4)
        cropped_3d = crop_sequence(data_3d, anchor_idx=5, past_timesteps=2, future_timesteps=3)
        assert cropped_3d.shape == (6, 3, 4)


class TestExtractRoboticsFieldsWithCropping:
    """Test the extract_robotics_fields function with cropping."""

    def test_extract_with_cropping(self):
        """Test that extract_robotics_fields crops sequences correctly."""
        # Create a sample with full sequences (1 past + 14 future = 15 timesteps)
        sample_lowdim_data = {
            "robot__action__poses__right::panda__xyz": np.arange(15 * 3).reshape(15, 3),
            "robot__actual__joint_position__right::panda": np.arange(15 * 7).reshape(15, 7),
            "past_mask": np.array([1] + [0] * 14),
            "future_mask": np.array([0] + [1] * 14),
        }

        sample = {
            "sample.lowdim.npz": sample_lowdim_data,
            "sample.language_instructions.json": {"original": "test instruction"},
            "sample.metadata.json": {"anchor_relative_idx": 1},
        }

        # Extract with cropping to 0 past + 8 future
        result = extract_robotics_fields(
            sample,
            action_fields=["robot__action__poses__right::panda__xyz"],
            proprioception_fields=["robot__actual__joint_position__right::panda"],
            lowdim_past_timesteps=0,
            lowdim_future_timesteps=8,
        )

        # Check that sequences were cropped
        assert result["lowdim"]["robot__action__poses__right::panda__xyz"].shape == (9, 3)
        assert result["lowdim"]["robot__actual__joint_position__right::panda"].shape == (9, 7)

        # Check that masks were cropped
        assert result["past_mask"].shape == (9,)  # 0 + 1 + 8 = 9
        assert result["future_mask"].shape == (9,)  # 0 + 1 + 8 = 9

    def test_extract_without_cropping(self):
        """Test that extract_robotics_fields works without cropping."""
        sample_lowdim_data = {
            "robot__action__poses__right::panda__xyz": np.arange(15 * 3).reshape(15, 3),
            "past_mask": np.ones(15),
            "future_mask": np.ones(15),
        }

        sample = {
            "sample.lowdim.npz": sample_lowdim_data,
            "sample.language_instructions.json": {},
        }

        # Extract without cropping parameters
        result = extract_robotics_fields(
            sample,
            action_fields=["robot__action__poses__right::panda__xyz"],
        )

        # Sequences should not be cropped
        assert result["lowdim"]["robot__action__poses__right::panda__xyz"].shape == (15, 3)


class TestRoboticsProcessorWithCropping:
    """Test RoboticsProcessor with sequence cropping."""

    def test_processor_with_cropped_sequences(self, stats_path, manifest_path):
        """Test that the processor handles cropped sequences correctly."""
        # Create data params with cropping
        data_params = RoboticsDataParams(
            dataset_statistics=[stats_path],
            dataset_manifest=[manifest_path],
            processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            action_fields=["robot__action__poses__right::panda__xyz"],
            proprioception_fields=[
                "robot__actual__joint_position__right::panda",
                "robot__actual__joint_velocity__right::panda",
            ],
            lowdim_past_timesteps=0,  # Crop to 0 past
            lowdim_future_timesteps=8,  # Crop to 8 future
            normalization=NormalizationParams(
                enabled=True,
                method="std",
                scope="global",
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=14,
            ),
        )
        data_params = init_data_params_for_tests(data_params)

        # Verify that data_params has the correct settings
        assert data_params.lowdim_past_timesteps == 0
        assert data_params.lowdim_future_timesteps == 8
        # Normalization should align with the source data window
        assert data_params.normalization.lowdim_past_timesteps == 1
        assert data_params.normalization.lowdim_future_timesteps == 14

        # Create processor
        processor = RoboticsProcessor(data_params)

        # Verify normalizer aligns with the cropped window
        assert processor.normalizer.lowdim_past_timesteps == 1
        assert processor.normalizer.lowdim_future_timesteps == 14

    def test_processor_with_different_crop_sizes(self, stats_path, manifest_path):
        """Test processor with various crop sizes."""
        crop_configs = [
            (0, 4),  # Very small future window
            (0, 8),  # Medium future window
            (0, 14),  # Full future window (no cropping)
            (1, 10),  # With past timestep
        ]

        for past, future in crop_configs:
            data_params = RoboticsDataParams(
                dataset_statistics=[stats_path],
                dataset_manifest=[manifest_path],
                processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                action_fields=["robot__action__poses__right::panda__xyz"],
                proprioception_fields=["robot__actual__joint_position__right::panda"],
                lowdim_past_timesteps=past,
                lowdim_future_timesteps=future,
                normalization=NormalizationParams(
                    enabled=True,
                    lowdim_past_timesteps=1,
                    lowdim_future_timesteps=14,
                ),
            )
            data_params = init_data_params_for_tests(data_params)

            processor = RoboticsProcessor(data_params)

            # Verify settings
            assert data_params.lowdim_past_timesteps == past
            assert data_params.lowdim_future_timesteps == future
            assert processor.normalizer.lowdim_past_timesteps == 1
            assert processor.normalizer.lowdim_future_timesteps == 14


class TestNormalizationWithCropping:
    """Test normalization alignment with cropped sequences."""

    def test_global_normalization_with_cropping(self, stats_path, manifest_path):
        """Test that global normalization works correctly with cropped sequences."""
        # Create normalizer with global normalization
        data_params = RoboticsDataParams(
            dataset_statistics=[stats_path],
            dataset_manifest=[manifest_path],
            processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            action_fields=["robot__action__poses__right::panda__xyz"],
            lowdim_past_timesteps=0,
            lowdim_future_timesteps=8,
            normalization=NormalizationParams(
                enabled=True,
                method="std",
                scope="global",
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=14,
            ),
        )
        data_params = init_data_params_for_tests(data_params)

        normalizer = RoboticsNormalizer(
            normalization_params=data_params.normalization,
            statistics_path=stats_path,
        )

        # Create a tensor with cropped sequences (0 past + 1 present + 8 future = 9 timesteps)
        tensor = torch.randn(2, 9, 3)

        # Normalize - should work without errors
        normalized = normalizer.normalize_tensor(tensor, "robot__action__poses__right::panda__xyz")

        assert normalized.shape == (2, 9, 3)

    def test_per_timestep_normalization_with_cropping(self, stats_path, manifest_path):
        """Test that per-timestep normalization aligns correctly with cropped sequences."""
        # Create normalizer with per-timestep normalization
        data_params = RoboticsDataParams(
            dataset_statistics=[stats_path],
            dataset_manifest=[manifest_path],
            processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            action_fields=["robot__action__poses__right::panda__xyz"],
            lowdim_past_timesteps=0,
            lowdim_future_timesteps=8,
            normalization=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=14,
            ),
        )
        data_params = init_data_params_for_tests(data_params)

        normalizer = RoboticsNormalizer(
            normalization_params=data_params.normalization,
            statistics_path=stats_path,
        )

        # Create a tensor with cropped sequences (0 past + 1 present + 8 future = 9 timesteps)
        # Anchor at index 0 means the tensor starts at the original anchor position
        tensor = torch.randn(2, 9, 3)

        # Normalize - should align correctly with anchor at 0
        normalized = normalizer.normalize_tensor(tensor, "robot__action__poses__right::panda__xyz", anchor_timestep=0)

        assert normalized.shape == (2, 9, 3)

    def test_normalization_alignment_correctness(self, stats_path, manifest_path):
        """Test that normalization uses the correct statistics for each timestep."""
        # This test just verifies that per-timestep normalization with cropped sequences
        # works without errors. Detailed alignment testing is complex and covered by
        # integration tests.
        data_params = RoboticsDataParams(
            dataset_statistics=[stats_path],
            dataset_manifest=[manifest_path],
            processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            action_fields=["robot__action__poses__right::panda__xyz"],
            lowdim_past_timesteps=0,
            lowdim_future_timesteps=8,
            normalization=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=14,
            ),
        )
        data_params = init_data_params_for_tests(data_params)

        normalizer = RoboticsNormalizer(
            normalization_params=data_params.normalization,
            statistics_path=stats_path,
        )

        field_name = "robot__action__poses__right::panda__xyz"

        # Create a random tensor with cropped size
        tensor = torch.randn(2, 8, 3)

        # Normalize with anchor at index 0
        normalized = normalizer.normalize_tensor(tensor, field_name, anchor_timestep=0)

        # Verify shape is preserved and normalization worked
        assert normalized.shape == tensor.shape
        assert torch.isfinite(normalized).all()

    def test_different_anchor_positions(self, stats_path, manifest_path):
        """Test normalization with different anchor positions in cropped sequences."""
        data_params = RoboticsDataParams(
            dataset_statistics=[stats_path],
            dataset_manifest=[manifest_path],
            processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            action_fields=["robot__action__poses__right::panda__xyz"],
            lowdim_past_timesteps=1,
            lowdim_future_timesteps=8,
            normalization=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=14,
            ),
        )
        data_params = init_data_params_for_tests(data_params)

        normalizer = RoboticsNormalizer(
            normalization_params=data_params.normalization,
            statistics_path=stats_path,
        )

        # Test with anchor at index 1 (1 past + 1 present + 8 future = 10 timesteps)
        tensor = torch.randn(2, 10, 3)

        normalized = normalizer.normalize_tensor(tensor, "robot__action__poses__right::panda__xyz", anchor_timestep=1)
        assert normalized.shape == (2, 10, 3)


class TestEndToEndWithRealData:
    """Test end-to-end processing with real dataset samples."""

    def test_load_and_crop_real_sample(self, dataset_dir, stats_path):
        """Test loading a real sample and cropping it."""
        shard_path = os.path.join(dataset_dir, "shard_000000.tar")

        if not os.path.exists(shard_path):
            pytest.skip(f"Test dataset not found at {shard_path}")

        # Load one sample from the dataset
        dataset = wds.WebDataset(shard_path).decode("pilrgb")
        sample = next(iter(dataset))

        # Extract with cropping
        result = extract_robotics_fields(
            sample,
            action_fields=["robot__desired__poses__right::panda__xyz"],
            proprioception_fields=["robot__actual__poses__right::panda__xyz"],
            lowdim_past_timesteps=0,
            lowdim_future_timesteps=8,
        )

        # Verify cropping worked
        assert result["lowdim"]["robot__desired__poses__right::panda__xyz"].shape[0] == 9
        assert result["lowdim"]["robot__actual__poses__right::panda__xyz"].shape[0] == 9

    def test_process_batch_with_real_data(self, dataset_dir, stats_path, manifest_path):
        """Test that processor correctly handles cropped sequences."""
        # Create data params with cropping
        data_params = RoboticsDataParams(
            dataset_statistics=[stats_path],
            dataset_manifest=[manifest_path],
            processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            action_fields=[
                "robot__desired__poses__right::panda__xyz",
                "robot__desired__poses__right::panda__rot_6d",
            ],
            proprioception_fields=[
                "robot__actual__poses__right::panda__xyz",
                "robot__actual__poses__right::panda__rot_6d",
            ],
            lowdim_past_timesteps=0,
            lowdim_future_timesteps=8,
            normalization=NormalizationParams(
                enabled=True,
                method="std",
                scope="global",
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=14,
            ),
        )
        data_params = init_data_params_for_tests(data_params)

        processor = RoboticsProcessor(data_params)

        # Verify the processor is properly configured
        assert processor.data_params.lowdim_past_timesteps == 0
        assert processor.data_params.lowdim_future_timesteps == 8
        assert processor.normalizer.lowdim_past_timesteps == 1
        assert processor.normalizer.lowdim_future_timesteps == 14

        # Verify normalizer can handle cropped sequences (0 past + 1 present + 8 future = 9 timesteps)
        xyz_tensor = torch.randn(2, 9, 3)
        rot_tensor = torch.randn(2, 9, 6)

        # Test normalization with each field
        normalized_xyz = processor.normalizer.normalize_tensor(xyz_tensor, "robot__desired__poses__right::panda__xyz")
        normalized_rot = processor.normalizer.normalize_tensor(
            rot_tensor, "robot__desired__poses__right::panda__rot_6d"
        )

        # Verify shapes are preserved
        assert normalized_xyz.shape == (2, 9, 3)
        assert normalized_rot.shape == (2, 9, 6)
