import sys
from unittest.mock import patch

import draccus
import pytest
from draccus.utils import DecodingError

from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.train_experiment_params import TrainExperimentParams


def get_args_robotics_auto_action_dim():
    """Test args for robotics data with automatic action dimension computation."""
    test_args = [
        "--name",
        "test_robotics_auto_action_dim",
        "--total_train_samples",
        "1000",
        "--model.type",
        "transformer",
        "--data.type",
        "robotics",
        "--data.dataset_manifest",
        ["tests/essential/test_assets/small_lbm_dataset/manifest.jsonl"],
        "--data.dataset_modality",
        ["robotics"],
        "--data.dataset_weighting",
        ["1.0"],
        "--data.dataset_statistics",
        ["tests/essential/test_assets/small_lbm_dataset/stats.json"],
        "--data.action_fields",
        [
            "robot__desired__poses__left::panda__xyz",
            "robot__desired__poses__right::panda__xyz",
            "robot__desired__poses__left::panda__rot_6d",
            "robot__desired__poses__right::panda__rot_6d",
            "robot__desired__grippers__left::panda_hand",
            "robot__desired__grippers__right::panda_hand",
        ],
        "--data.proprioception_fields",
        [
            "robot__actual__poses__right::panda__xyz",
            "robot__actual__poses__left::panda__xyz",
        ],
    ]
    with patch.object(sys, "argv", ["test"] + test_args):
        args = draccus.parse(config_class=TrainExperimentParams)
    return args


def get_args_robotics_manual_action_dim(action_dim: int):
    """Test args for robotics data with manually specified action dimension."""
    test_args = [
        "--name",
        "test_robotics_manual_action_dim",
        "--total_train_samples",
        "1000",
        "--model.type",
        "transformer",
        "--data.type",
        "robotics",
        "--data.dataset_manifest",
        ["tests/essential/test_assets/small_lbm_dataset/manifest.jsonl"],
        "--data.dataset_modality",
        ["robotics"],
        "--data.dataset_weighting",
        ["1.0"],
        "--data.dataset_statistics",
        ["tests/essential/test_assets/small_lbm_dataset/stats.json"],
        "--data.action_fields",
        [
            "robot__action__poses__left::panda__xyz_relative",
            "robot__action__poses__right::panda__xyz_relative",
            "robot__action__poses__left::panda__rot_6d_relative",
            "robot__action__poses__right::panda__rot_6d_relative",
            "robot__action__grippers__left::panda_hand",
            "robot__action__grippers__right::panda_hand",
        ],
        "--data.proprioception_fields",
        [
            "robot__actual__poses__right::panda__xyz",
            "robot__actual__poses__left::panda__xyz",
        ],
        "--data.action_dim",
        str(action_dim),
    ]
    with patch.object(sys, "argv", ["test"] + test_args):
        args = draccus.parse(config_class=TrainExperimentParams)
    return args


def test_action_dim_auto_computation():
    """Test that action_dim is automatically computed from action fields and statistics."""
    args = get_args_robotics_auto_action_dim()

    # Verify the data config is RoboticsDataParams
    assert isinstance(args.data, RoboticsDataParams)

    # Verify action fields are set correctly
    expected_action_fields = [
        "robot__desired__poses__left::panda__xyz",
        "robot__desired__poses__right::panda__xyz",
        "robot__desired__poses__left::panda__rot_6d",
        "robot__desired__poses__right::panda__rot_6d",
        "robot__desired__grippers__left::panda_hand",
        "robot__desired__grippers__right::panda_hand",
    ]
    assert args.data.action_fields == expected_action_fields

    # Verify action dimension is automatically computed
    # Expected: 3 (xyz) + 3 (xyz) + 6 (rot_6d) + 6 (rot_6d) + 1 (gripper) + 1 (gripper) = 20
    expected_action_dim = 20
    assert args.data.action_dim == expected_action_dim


def test_action_dim_manual_correct():
    """Test that manually specified action_dim works when it matches computed dimension."""
    expected_action_dim = 20
    args = get_args_robotics_manual_action_dim(expected_action_dim)

    # Verify the manually set action_dim is preserved
    assert args.data.action_dim == expected_action_dim


def test_action_dim_manual_incorrect():
    """Test that assertion error is raised when manually specified action_dim doesn't match computed."""
    incorrect_action_dim = 15  # Should be 20

    with pytest.raises(DecodingError, match="Action dimension mismatch"):
        get_args_robotics_manual_action_dim(incorrect_action_dim)


def test_action_dim_subset_fields():
    """Test action_dim computation with a subset of action fields."""
    test_args = [
        "--name",
        "test_robotics_subset_action_dim",
        "--total_train_samples",
        "1000",
        "--model.type",
        "transformer",
        "--data.type",
        "robotics",
        "--data.dataset_manifest",
        ["tests/essential/test_assets/small_lbm_dataset/manifest.jsonl"],
        "--data.dataset_modality",
        ["robotics"],
        "--data.dataset_weighting",
        ["1.0"],
        "--data.dataset_statistics",
        ["tests/essential/test_assets/small_lbm_dataset/stats.json"],
        "--data.action_fields",
        [
            "robot__desired__poses__left::panda__xyz",  # 3D
            "robot__desired__grippers__left::panda_hand",  # 1D
        ],
        "--data.proprioception_fields",
        [
            "robot__actual__poses__right::panda__xyz",
        ],
    ]

    with patch.object(sys, "argv", ["test"] + test_args):
        args = draccus.parse(config_class=TrainExperimentParams)

    # Expected: 3 (xyz) + 1 (gripper) = 4
    expected_action_dim = 4
    assert args.data.action_dim == expected_action_dim


def test_action_dim_empty_fields():
    """Test action_dim computation with empty action fields."""
    test_args = [
        "--name",
        "test_robotics_empty_action_dim",
        "--total_train_samples",
        "1000",
        "--model.type",
        "transformer",
        "--data.type",
        "robotics",
        "--data.dataset_manifest",
        ["tests/essential/test_assets/small_lbm_dataset/manifest.jsonl"],
        "--data.dataset_modality",
        ["robotics"],
        "--data.dataset_weighting",
        ["1.0"],
        "--data.dataset_statistics",
        ["tests/essential/test_assets/small_lbm_dataset/stats.json"],
        "--data.action_fields",
        [],  # Empty action fields
        "--data.proprioception_fields",
        [
            "robot__actual__poses__right::panda__xyz",
        ],
    ]

    with patch.object(sys, "argv", ["test"] + test_args):
        args = draccus.parse(config_class=TrainExperimentParams)

    # Expected: 0 (no action fields)
    expected_action_dim = 0
    assert args.data.action_dim == expected_action_dim
