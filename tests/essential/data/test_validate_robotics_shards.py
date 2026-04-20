#!/usr/bin/env python3
"""
Pytest tests for validating processed robotics shards.

Usage:
    # Run unit tests only
    pytest tests/essential/test_validate_robotics_shards.py -v

    # Run integration tests (requires S3 access)
    pytest tests/essential/test_validate_robotics_shards.py -v -m integration

    # Run integration tests with custom dataset path
    pytest tests/essential/test_validate_robotics_shards.py -v -m integration --dataset-path /path/to/dataset

    # Run with custom number of samples to validate
    pytest tests/essential/test_validate_robotics_shards.py -v -m integration --dataset-path /path --num-samples 9

    # Run all tests
    pytest tests/essential/test_validate_robotics_shards.py -v
"""

import os
from typing import Any

import numpy as np
import pytest
import webdataset as wds

import vla_foundry.data.utils  # noqa: F401  # Ensure s3:// gopen hook is installed.
from vla_foundry.file_utils import json_load, load_dataset_manifest


@pytest.fixture
def mock_sample_valid():
    """Create a valid mock sample for testing."""
    return {
        "__key__": "test_sample_001",
        "camera_front.jpg": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "camera_wrist.jpg": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "lowdim.npz": {
            "robot_joint_positions": np.random.randn(10, 7).astype(np.float32),
            "robot_joint_velocities": np.random.randn(10, 7).astype(np.float32),
            "robot_end_effector_pose": np.random.randn(10, 7).astype(np.float32),
            "observation_text": ["pick up the red block"] * 10,
        },
        "actions.npz": {
            "robot_joint_positions": np.random.randn(10, 7).astype(np.float32),
            "robot_joint_velocities": np.random.randn(10, 7).astype(np.float32),
        },
        "masks.npz": {
            "past_mask": np.ones(10, dtype=bool),
            "future_mask": np.ones(10, dtype=bool),
        },
        "metadata.json": {
            "episode_id": "episode_001",
            "sample_id": "sample_001",
            "anchor_timestep": 5,
            "camera_names": ["camera_front", "camera_wrist"],
            "is_padded": False,
        },
    }


@pytest.fixture
def mock_sample_with_nan():
    """Create a mock sample with NaN values for testing error detection."""
    sample = {
        "__key__": "test_sample_nan",
        "lowdim.npz": {
            "robot_joint_positions": np.array([[1.0, 2.0, np.nan, 4.0]]).astype(np.float32),
        },
        "actions.npz": {
            "robot_joint_positions": np.array([[np.nan, 2.0, 3.0]]).astype(np.float32),
        },
        "masks.npz": {
            "past_mask": np.ones(1, dtype=bool),
            "future_mask": np.ones(1, dtype=bool),
        },
        "metadata.json": {
            "episode_id": "episode_001",
            "sample_id": "sample_001",
            "anchor_timestep": 0,
            "camera_names": [],
            "is_padded": False,
        },
    }
    return sample


@pytest.fixture
def mock_sample_missing_fields():
    """Create a mock sample with missing required fields."""
    return {
        "__key__": "test_sample_missing",
        "lowdim.npz": {
            "robot_joint_positions": np.random.randn(5, 7).astype(np.float32),
        },
        "masks.npz": {
            "past_mask": np.ones(5, dtype=bool),
            # Missing future_mask
        },
        "metadata.json": {
            "episode_id": "episode_001",
            # Missing required fields
        },
    }


def validate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Validate a single sample and return statistics."""
    stats = {
        "has_images": False,
        "has_lowdim": False,
        "has_actions": False,
        "has_masks": False,
        "has_metadata": False,
        "num_images": 0,
        "lowdim_keys": 0,
        "action_keys": 0,
        "errors": [],
    }

    try:
        # Check images
        image_keys = [k for k in sample if k.endswith(".jpg")]
        if image_keys:
            stats["has_images"] = True
            stats["num_images"] = len(image_keys)

        # Check lowdim data
        if "lowdim.npz" in sample:
            stats["has_lowdim"] = True
            lowdim_data = sample["lowdim.npz"]
            if isinstance(lowdim_data, dict):
                stats["lowdim_keys"] = len(lowdim_data.keys())

                # Check for NaN values
                for key, data in lowdim_data.items():
                    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number) and np.isnan(data).any():
                        stats["errors"].append(f"NaN values in lowdim.{key}")
            else:
                stats["errors"].append(f"lowdim.npz unexpected type: {type(lowdim_data)}")

        # Check actions
        if "actions.npz" in sample:
            stats["has_actions"] = True
            actions_data = sample["actions.npz"]
            if isinstance(actions_data, dict):
                stats["action_keys"] = len(actions_data.keys())

                # Check for NaN values
                for key, data in actions_data.items():
                    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number) and np.isnan(data).any():
                        stats["errors"].append(f"NaN values in actions.{key}")
            else:
                stats["errors"].append(f"actions.npz unexpected type: {type(actions_data)}")

        # Check masks
        if "masks.npz" in sample:
            stats["has_masks"] = True
            masks_data = sample["masks.npz"]
            if isinstance(masks_data, dict):
                if "past_mask" not in masks_data or "future_mask" not in masks_data:
                    stats["errors"].append("Missing past_mask or future_mask")
            else:
                stats["errors"].append(f"masks.npz unexpected type: {type(masks_data)}")

        # Check metadata
        if "metadata.json" in sample:
            stats["has_metadata"] = True
            metadata = sample["metadata.json"]
            if isinstance(metadata, dict):
                required_fields = ["episode_id", "sample_id", "anchor_timestep", "camera_names", "is_padded"]
                for field in required_fields:
                    if field not in metadata:
                        stats["errors"].append(f"Missing metadata field: {field}")
            else:
                stats["errors"].append(f"metadata.json unexpected type: {type(metadata)}")

    except Exception as e:
        stats["errors"].append(f"Exception during validation: {str(e)}")

    return stats


class TestSampleValidation:
    """Test individual sample validation logic."""

    def test_validate_valid_sample(self, mock_sample_valid):
        """Test validation of a valid sample."""
        stats = validate_sample(mock_sample_valid)

        assert stats["has_images"] is True
        assert stats["has_lowdim"] is True
        assert stats["has_actions"] is True
        assert stats["has_masks"] is True
        assert stats["has_metadata"] is True
        assert stats["num_images"] == 2
        assert stats["lowdim_keys"] == 4
        assert stats["action_keys"] == 2
        assert len(stats["errors"]) == 0

    def test_validate_sample_with_nan(self, mock_sample_with_nan):
        """Test validation detects NaN values."""
        stats = validate_sample(mock_sample_with_nan)

        assert len(stats["errors"]) == 2
        assert any("NaN values in lowdim.robot_joint_positions" in error for error in stats["errors"])
        assert any("NaN values in actions.robot_joint_positions" in error for error in stats["errors"])

    def test_validate_sample_missing_fields(self, mock_sample_missing_fields):
        """Test validation detects missing required fields."""
        stats = validate_sample(mock_sample_missing_fields)

        assert len(stats["errors"]) > 0
        assert any("Missing past_mask or future_mask" in error for error in stats["errors"])
        assert any("Missing metadata field:" in error for error in stats["errors"])

    def test_validate_empty_sample(self):
        """Test validation of empty sample."""
        stats = validate_sample({})

        assert stats["has_images"] is False
        assert stats["has_lowdim"] is False
        assert stats["has_actions"] is False
        assert stats["has_masks"] is False
        assert stats["has_metadata"] is False
        assert stats["num_images"] == 0
        assert stats["lowdim_keys"] == 0
        assert stats["action_keys"] == 0

    def test_validate_sample_with_exception(self):
        """Test validation handles exceptions gracefully."""
        # Create a sample that will cause an exception
        sample = {"lowdim.npz": "invalid_data_type"}  # Should be dict

        stats = validate_sample(sample)
        assert len(stats["errors"]) > 0
        assert any("unexpected type" in error for error in stats["errors"])


class TestDatasetValidation:
    """Test dataset-level validation functionality."""

    @pytest.mark.integration
    def test_load_manifest(self, dataset_config):
        """Test loading dataset manifest."""
        manifest = load_dataset_manifest(os.path.join(dataset_config["path"], "manifest.jsonl"))

        assert isinstance(manifest, list)
        assert len(manifest) > 0

        # Check manifest structure
        first_entry = manifest[0]
        assert "shard" in first_entry
        assert "num_sequences" in first_entry
        assert isinstance(first_entry["num_sequences"], int)
        assert first_entry["num_sequences"] > 0

    @pytest.mark.integration
    def test_load_dataset_statistics(self, dataset_config):
        """Test loading dataset statistics."""
        stats = json_load(os.path.join(dataset_config["path"], "stats.json"))

        # Statistics might not exist, but if they do, check structure
        if stats is not None:
            assert isinstance(stats, dict)
            # Check for expected fields
            expected_fields = ["temporal_length", "overall"]
            for field in expected_fields:
                if field in stats:
                    assert stats[field] is not None

    @pytest.mark.integration
    def test_validate_real_dataset_samples(self, dataset_config):
        """Test validation on real dataset samples."""
        dataset_path = dataset_config["path"]
        num_samples = dataset_config["num_samples"]

        # Load manifest
        manifest = load_dataset_manifest(os.path.join(dataset_path, "manifest.jsonl"))
        assert len(manifest) > 0, f"No manifest entries found in {dataset_path}"

        # Create shard URLs for first few shards
        shard_urls = []
        for entry in manifest[:3]:  # Test first 3 shards
            shard_path = os.path.join(dataset_path, f"{entry['shard']}.tar")
            shard_urls.append(shard_path)

        # Create dataset
        dataset = wds.WebDataset(shard_urls, shardshuffle=False).decode()

        # Validate samples
        validation_stats = []
        sample_count = 0
        sample_keys = []

        for sample in dataset:
            if sample_count >= num_samples:
                break

            sample_keys.append(sample.get("__key__", f"sample_{sample_count}"))
            stats = validate_sample(sample)
            validation_stats.append(stats)
            sample_count += 1

        # Print summary of what was actually tested
        if sample_count > 0:
            print(f"\nValidated {sample_count} samples from dataset: {dataset_path}")
            print(f"Sample keys: {sample_keys}")
            total_errors = sum(len(stats["errors"]) for stats in validation_stats)
            if total_errors > 0:
                print(f"Total validation issues found: {total_errors}")
            else:
                print("All samples passed validation ✓")

        # Assertions on validation results
        assert len(validation_stats) == min(num_samples, sample_count)
        assert sample_count > 0, "No samples were loaded from the dataset"

        # Check that most samples have required components
        samples_with_lowdim = sum(1 for stats in validation_stats if stats["has_lowdim"])
        samples_with_metadata = sum(1 for stats in validation_stats if stats["has_metadata"])

        assert samples_with_lowdim > 0, "No samples found with lowdim data"
        assert samples_with_metadata > 0, "No samples found with metadata"

        # Check for critical errors
        total_errors = sum(len(stats["errors"]) for stats in validation_stats)
        critical_errors = []
        for stats in validation_stats:
            for error in stats["errors"]:
                if "NaN values" in error or "Missing metadata field" in error:
                    critical_errors.append(error)

        # Report but don't fail on non-critical errors
        if total_errors > 0:
            print(f"\nFound {total_errors} total validation issues:")
            if critical_errors:
                print(f"Critical errors: {len(critical_errors)}")
                for error in set(critical_errors):
                    count = critical_errors.count(error)
                    print(f"  - {error} (occurred {count} times)")

        # Only fail on critical errors
        assert len(critical_errors) == 0, f"Found {len(critical_errors)} critical validation errors"


class TestManifestValidation:
    """Test manifest file validation."""

    def test_validate_manifest_structure(self):
        """Test validation of manifest structure."""
        valid_manifest = [
            {"shard": "shard_001", "num_sequences": 100},
            {"shard": "shard_002", "num_sequences": 150},
        ]

        # Test valid manifest
        assert all("shard" in entry and "num_sequences" in entry for entry in valid_manifest)
        assert all(isinstance(entry["num_sequences"], int) for entry in valid_manifest)
        assert all(entry["num_sequences"] > 0 for entry in valid_manifest)

    def test_validate_manifest_with_missing_fields(self):
        """Test detection of invalid manifest entries."""
        invalid_manifest = [
            {"shard": "shard_001"},  # Missing num_sequences
            {"num_sequences": 100},  # Missing shard
        ]

        for entry in invalid_manifest:
            assert not ("shard" in entry and "num_sequences" in entry)


if __name__ == "__main__":
    # Allow running as script for backwards compatibility
    import sys

    if len(sys.argv) > 1 and "--help" not in sys.argv:
        print("To run tests, use: pytest tests/essential/test_validate_robotics_shards.py")
        print("For integration tests, add: --dataset-path /path/to/dataset")
    else:
        pytest.main([__file__] + sys.argv[1:])
