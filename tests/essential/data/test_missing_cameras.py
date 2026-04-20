"""Integration tests for graceful handling of missing cameras in SpartanConverter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from vla_foundry.data.preprocessing.robotics.converters.spartan import SpartanConverter


@pytest.fixture
def mock_config():
    """Create a mock config for SpartanConverter."""
    cfg = MagicMock()
    cfg.type = "spartan"
    cfg.language_annotations_path = None
    cfg.action_fields_config_path = None
    cfg.data_discard_keys = []
    cfg.camera_names = []
    cfg.resize_images_size = [384, 384]
    cfg.image_indices = [-1, 0]
    cfg.jpeg_quality = 90
    cfg.padding_strategy = "copy"
    cfg.past_lowdim_steps = 5
    cfg.future_lowdim_steps = 10
    cfg.max_padding_left = 5
    cfg.max_padding_right = 10
    cfg.filter_still_samples = False
    cfg.still_threshold = 0.01
    cfg.stride = 1
    cfg.num_workers = 1
    cfg.output_dir = "/tmp/test_output"
    cfg.fail_on_nan = False
    cfg.image_resizing_method = "resize_and_crop"
    cfg.camera_rotations = {}
    return cfg


@pytest.fixture
def spartan_episode_dir(tmp_path):
    """Write a minimal but real Spartan episode to disk (one camera: camera_left)."""
    T = 5
    task_dir = tmp_path / "tasks" / "test_task"
    ep_dir = task_dir / "episode_0000" / "processed"
    ep_dir.mkdir(parents=True)

    metadata = {
        "camera_id_to_semantic_name": {"cam_l": "camera_left"},
        "episode_length": T,
    }
    (ep_dir / "metadata.yaml").write_text(yaml.dump(metadata))

    np.savez(
        ep_dir / "observations.npz",
        cam_l=np.random.randint(0, 255, (T, 64, 64, 3), dtype=np.uint8),
        robot_joint_positions=np.random.randn(T, 7).astype(np.float32),
    )
    np.savez(ep_dir / "actions.npz", actions=np.random.randn(T, 7).astype(np.float32))

    K = np.tile(np.eye(3, dtype=np.float32)[None], (T, 1, 1))
    E = np.tile(np.eye(4, dtype=np.float32)[None], (T, 1, 1))
    np.savez(ep_dir / "intrinsics.npz", cam_l=K)
    np.savez(ep_dir / "extrinsics.npz", cam_l=E)

    return str(ep_dir.parent)  # episode_0000/


def _make_converter(mock_config):
    """Build a real SpartanConverter (only language/action config is faked)."""
    with patch("builtins.open"), patch("yaml.safe_load") as mock_yaml:
        mock_yaml.return_value = {"language_dict": {"test_task": {"original": ["do the thing"]}}}
        mock_config.language_annotations_path = "/tmp/fake.yaml"
        mock_config.action_fields_config_path = "/tmp/fake.yaml"
        mock_config.validation_episodes_path = None
        with patch(
            "vla_foundry.data.robotics.utils.load_action_field_config",
            return_value={
                "action_key_fields": ["action"],
                "action_index_fields": [7],
                "pose_groups": [],
            },
        ):
            return SpartanConverter(mock_config)


def test_load_and_extract_all_cameras_present(mock_config, spartan_episode_dir):
    """Load real episode from disk; request only camera_left which exists."""
    mock_config.camera_names = ["camera_left"]
    mock_config.skip_episodes_missing_cameras = False
    converter = _make_converter(mock_config)

    episode_data = converter.load_episode_data(spartan_episode_dir)
    camera_data = converter.extract_camera_data(episode_data)

    assert "camera_left" in camera_data
    assert camera_data["camera_left"].shape[0] == 5


def test_load_and_extract_missing_camera_skip_true(mock_config, spartan_episode_dir):
    """Request camera_right which doesn't exist on disk; skip_episodes_missing_cameras=True -> empty."""
    mock_config.camera_names = ["camera_left", "camera_right"]
    mock_config.skip_episodes_missing_cameras = True
    converter = _make_converter(mock_config)

    episode_data = converter.load_episode_data(spartan_episode_dir)
    camera_data = converter.extract_camera_data(episode_data)

    assert camera_data == {}


def test_load_and_extract_missing_camera_skip_false(mock_config, spartan_episode_dir):
    """Request camera_right which doesn't exist on disk; skip=False -> partial cameras returned."""
    mock_config.camera_names = ["camera_left", "camera_right"]
    mock_config.skip_episodes_missing_cameras = False
    converter = _make_converter(mock_config)

    episode_data = converter.load_episode_data(spartan_episode_dir)
    camera_data = converter.extract_camera_data(episode_data)

    assert "camera_left" in camera_data
    assert "camera_right" not in camera_data


def test_process_episode_skips_on_missing_camera(mock_config, spartan_episode_dir):
    """Full process_episode: episode is skipped (returns []) when cameras are missing."""
    mock_config.camera_names = ["camera_left", "camera_right"]
    mock_config.skip_episodes_missing_cameras = True
    converter = _make_converter(mock_config)

    logger_actor = MagicMock()
    result = converter.process_episode(spartan_episode_dir, statistics_ray_actor=None, logger_actor=logger_actor)

    assert result == []
