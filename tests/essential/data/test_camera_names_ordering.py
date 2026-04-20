"""Test that camera names are discovered and saved from data when not specified."""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import yaml_load
from vla_foundry.params.data_params import RoboticsDataParams


@pytest.fixture
def dataset_stats_path():
    """Get the path to the real dataset statistics file."""
    return os.path.join(os.path.dirname(__file__), "..", "test_assets", "small_lbm_dataset", "stats.json")


@pytest.fixture
def dataset_manifest_path():
    """Get the path to the dataset manifest file."""
    return os.path.join(os.path.dirname(__file__), "..", "test_assets", "small_lbm_dataset", "manifest.jsonl")


@pytest.fixture
def sample_batch():
    """Create a sample batch with multiple images from different cameras."""
    return {
        "images": [
            {
                "camera_front_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "camera_wrist_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "camera_side_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            },
            {
                "camera_front_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "camera_wrist_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "camera_side_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            },
        ],
        "language_instruction": ["pick up the red block", "place the blue cube"],
        "lowdim": [
            {"robot_joint_positions": np.random.randn(10, 7).astype(np.float32)},
            {"robot_joint_positions": np.random.randn(10, 7).astype(np.float32)},
        ],
    }


@patch("vla_foundry.data.processor.robotics_processor.get_processor")
def test_camera_names_discovered_from_preprocessing_config(
    mock_get_processor, dataset_stats_path, dataset_manifest_path
):
    """Test that camera names are discovered from preprocessing config when not specified."""
    # Setup mocks
    mock_vlm_processor = Mock()
    mock_vlm_processor.chat_template = None
    mock_vlm_processor.tokenizer = None
    mock_vlm_processor.return_value = {
        "input_ids": np.array([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": np.array([[1, 1, 1], [1, 1, 1]]),
    }
    mock_get_processor.return_value = mock_vlm_processor

    preprocessing_config_path = os.path.join(os.path.dirname(dataset_stats_path), "preprocessing_config.yaml")
    preprocessing_config = yaml_load(preprocessing_config_path)
    expected_camera_names = preprocessing_config["camera_names"]
    expected_image_indices = sorted(preprocessing_config["image_indices"])
    expected_image_names = [f"{cname}_t{idx}" for idx in expected_image_indices for cname in expected_camera_names]

    # Create RoboticsDataParams WITHOUT camera_names, image_indices, or image_names
    data_params = RoboticsDataParams(
        dataset_statistics=[dataset_stats_path],
        dataset_manifest=[dataset_manifest_path],
        processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        image_indices=[],
        camera_names=[],
        image_names=[],
        proprioception_fields=["robot__actual__joint_position__right::panda"],
        action_fields=["robot__actual__poses__right::panda__xyz"],
    )

    # Verify camera and image configuration derived from preprocessing config
    assert data_params.camera_names == expected_camera_names
    assert data_params.image_indices == expected_image_indices
    assert data_params.image_names == expected_image_names

    # Create processor to ensure initialization does not modify discovered values
    processor = RoboticsProcessor(data_params)

    assert processor.data_params.camera_names == expected_camera_names
    assert processor.data_params.image_names == expected_image_names


@patch("vla_foundry.data.processor.robotics_processor.get_processor")
def test_camera_names_not_overwritten_when_specified(mock_get_processor, dataset_stats_path, sample_batch):
    """Test that camera names are NOT overwritten when already specified."""
    # Setup mocks
    mock_vlm_processor = Mock()
    mock_vlm_processor.chat_template = None
    mock_vlm_processor.tokenizer = None
    mock_vlm_processor.return_value = {
        "input_ids": np.array([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": np.array([[1, 1, 1], [1, 1, 1]]),
    }
    mock_get_processor.return_value = mock_vlm_processor

    # Create RoboticsDataParams WITH camera_names specified
    specified_cameras = ["camera_wrist", "camera_front"]  # Different order than data
    data_params = RoboticsDataParams(
        dataset_statistics=[dataset_stats_path],
        processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        image_indices=[0],
        camera_names=specified_cameras,  # User-specified camera names
        proprioception_fields=["robot__actual__joint_position__right::panda"],
        action_fields=["robot__actual__poses__right::panda__xyz"],
    )

    # Verify camera_names is set before processing
    assert data_params.camera_names == specified_cameras
    assert data_params.image_names == ["camera_wrist_t0", "camera_front_t0"]

    # Create processor
    processor = RoboticsProcessor(data_params)

    # Process the batch - should NOT modify camera_names
    processor.process_inputs(sample_batch, image_names=data_params.image_names)

    # Verify camera names were NOT changed
    assert processor.data_params.camera_names == specified_cameras
    assert processor.data_params.image_names == ["camera_wrist_t0", "camera_front_t0"]


@patch("vla_foundry.data.processor.robotics_processor.get_processor")
def test_camera_names_ordering_preserved(mock_get_processor, dataset_stats_path, dataset_manifest_path):
    """Test that discovered camera names maintain consistent ordering across batches."""
    # Setup mocks
    mock_vlm_processor = Mock()
    mock_vlm_processor.chat_template = None
    mock_vlm_processor.tokenizer = None
    mock_vlm_processor.return_value = {
        "input_ids": np.array([[1, 2, 3]]),
        "attention_mask": np.array([[1, 1, 1]]),
    }
    mock_get_processor.return_value = mock_vlm_processor

    # Create RoboticsDataParams WITHOUT camera_names
    data_params = RoboticsDataParams(
        dataset_statistics=[dataset_stats_path],
        dataset_manifest=[dataset_manifest_path],
        processor="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        image_indices=[0],
        camera_names=[],
        image_names=[],
        proprioception_fields=["robot__actual__joint_position__right::panda"],
        action_fields=["robot__actual__poses__right::panda__xyz"],
    )

    # Create processor
    processor = RoboticsProcessor(data_params)

    # First batch with certain camera order
    batch1 = {
        "images": [
            {
                "camera_a_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "camera_b_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            }
        ],
        "language_instruction": ["test"],
        "lowdim": [{"robot_joint_positions": np.random.randn(10, 7).astype(np.float32)}],
    }

    # Process first batch
    processor.process_inputs(batch1, image_names=processor.data_params.image_names)
    first_camera_names = processor.data_params.camera_names.copy()
    first_image_names = processor.data_params.image_names.copy()

    # Second batch with DIFFERENT camera order in keys
    batch2 = {
        "images": [
            {
                "camera_b_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "camera_a_t0": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            }
        ],
        "language_instruction": ["test2"],
        "lowdim": [{"robot_joint_positions": np.random.randn(10, 7).astype(np.float32)}],
    }

    # Process second batch - should use the same ordering as first batch
    processor.process_inputs(batch2, image_names=processor.data_params.image_names)

    # Verify ordering is preserved (not changed by second batch)
    assert processor.data_params.camera_names == first_camera_names
    assert processor.data_params.image_names == first_image_names
