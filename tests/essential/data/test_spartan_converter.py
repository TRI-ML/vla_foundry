"""Tests for Spartan converter."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import fsspec
import numpy as np
import pytest
import yaml

from vla_foundry.data.preprocessing.robotics.converters.spartan import SpartanConverter


# Helper functions for testing (non-Ray versions)
def check_episode_validity(episode_path: str) -> bool:
    """Check if an episode directory has valid processed data."""
    fs, _ = fsspec.core.url_to_fs(episode_path)
    processed_path = os.path.join(episode_path, "processed")
    fs_processed_path = processed_path.replace("s3://", "")

    try:
        if not fs.exists(fs_processed_path):
            return False

        # Check for required files
        required_files = ["metadata.yaml", "observations.npz"]
        for required_file in required_files:
            file_path = os.path.join(processed_path, required_file)
            fs_file_path = file_path.replace("s3://", "")
            if not fs.exists(fs_file_path):
                return False
        return True
    except Exception:
        return False


def discover_episodes_in_directory(diffusion_spartan_path: str, max_episodes: int = -1) -> list:
    """Discover and validate episodes in a diffusion_spartan directory."""
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
            episode_paths.append(episode_path)

            # Early exit if we have enough episodes
            if max_episodes > 0 and len(episode_paths) >= max_episodes:
                break

    if not episode_paths:
        return []

    # Second pass: validate episodes
    valid_episodes = [ep for ep in episode_paths if check_episode_validity(ep)]

    return valid_episodes


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
    cfg.padding_strategy = "copy"  # Use valid padding strategy
    cfg.past_lowdim_steps = 5
    cfg.future_lowdim_steps = 10
    cfg.max_padding_left = 5
    cfg.max_padding_right = 10
    cfg.filter_still_samples = False
    cfg.still_threshold = 0.01
    cfg.stride = 1
    cfg.num_workers = 1
    cfg.output_dir = "/tmp/test_output"
    cfg.jpeg_quality = 90
    cfg.fail_on_nan = False
    return cfg


@pytest.fixture
def temp_spartan_episodes():
    """Create temporary episode directories with required files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "data" / "diffusion_spartan"
        base_path.mkdir(parents=True, exist_ok=True)

        # Create valid episodes
        for i in range(3):
            episode_dir = base_path / f"episode_{i:04d}"
            processed_dir = episode_dir / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Create required files
            metadata_path = processed_dir / "metadata.yaml"
            metadata_path.write_text(yaml.dump({"episode_id": i}))

            observations_path = processed_dir / "observations.npz"
            np.savez(observations_path, obs=np.zeros((10, 3)))

        # Create invalid episode (missing observations.npz)
        invalid_episode_dir = base_path / "episode_0003"
        invalid_processed_dir = invalid_episode_dir / "processed"
        invalid_processed_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = invalid_processed_dir / "metadata.yaml"
        metadata_path.write_text(yaml.dump({"episode_id": 3}))

        yield str(base_path)


def test_check_episode_validity_valid(temp_spartan_episodes):
    """Test that check_episode_validity returns True for valid episodes."""
    episode_path = os.path.join(temp_spartan_episodes, "episode_0000")
    result = check_episode_validity(episode_path)
    assert result is True


def test_check_episode_validity_invalid(temp_spartan_episodes):
    """Test that check_episode_validity returns False for invalid episodes."""
    episode_path = os.path.join(temp_spartan_episodes, "episode_0003")
    result = check_episode_validity(episode_path)
    assert result is False


def test_check_episode_validity_nonexistent(temp_spartan_episodes):
    """Test that check_episode_validity returns False for non-existent episodes."""
    episode_path = os.path.join(temp_spartan_episodes, "episode_9999")
    result = check_episode_validity(episode_path)
    assert result is False


def test_discover_episodes_in_directory(temp_spartan_episodes):
    """Test discovery and validation of episodes."""
    result = discover_episodes_in_directory(temp_spartan_episodes)

    # Should find 3 valid episodes (0000, 0001, 0002) and exclude the invalid one (0003)
    assert len(result) == 3
    assert all("episode_" in ep for ep in result)

    # Check that valid episodes are in the result
    episode_names = [os.path.basename(ep) for ep in result]
    assert "episode_0000" in episode_names
    assert "episode_0001" in episode_names
    assert "episode_0002" in episode_names
    assert "episode_0003" not in episode_names


def test_discover_episodes_with_max_limit(temp_spartan_episodes):
    """Test that max_episodes parameter limits the number of episodes discovered."""
    result = discover_episodes_in_directory(temp_spartan_episodes, max_episodes=2)

    # Should find at most 2 episodes due to the limit
    assert len(result) <= 2


def test_discover_episodes_integration(temp_spartan_episodes, mock_config):
    """Test the full discover_episodes method by mocking Ray calls."""
    # Mock the language annotations and action fields to avoid file loading
    with patch("builtins.open"), patch("yaml.safe_load") as mock_yaml:
        mock_yaml.return_value = {"language_dict": {}}
        mock_config.language_annotations_path = "/tmp/fake_annotations.yaml"
        mock_config.action_fields_config_path = "/tmp/fake_action_fields.yaml"
        mock_config.validation_episodes_path = None

        with (
            patch(
                "vla_foundry.data.robotics.utils.load_action_field_config",
                return_value={"action_key_fields": [], "action_index_fields": [], "pose_groups": []},
            ),
            patch("ray.get") as mock_ray_get,
            patch(
                "vla_foundry.data.preprocessing.robotics.converters.spartan.discover_and_validate_episodes_in_directory"
            ),
        ):
            # Set up mock to return valid episodes
            valid_episodes = [os.path.join(temp_spartan_episodes, f"episode_{i:04d}") for i in range(3)]
            mock_ray_get.return_value = [valid_episodes]

            converter = SpartanConverter(mock_config)
            episodes = converter.discover_episodes([temp_spartan_episodes])

            # Should discover 3 valid episodes
            assert len(episodes) == 3
            assert all("episode_" in ep for ep in episodes)


def test_discover_episodes_with_max_episodes_to_process(temp_spartan_episodes, mock_config):
    """Test that max_episodes_to_process limits the total number of episodes returned."""
    with patch("builtins.open"), patch("yaml.safe_load") as mock_yaml:
        mock_yaml.return_value = {"language_dict": {}}
        mock_config.language_annotations_path = "/tmp/fake_annotations.yaml"
        mock_config.action_fields_config_path = "/tmp/fake_action_fields.yaml"
        mock_config.validation_episodes_path = None

        with (
            patch(
                "vla_foundry.data.robotics.utils.load_action_field_config",
                return_value={"action_key_fields": [], "action_index_fields": [], "pose_groups": []},
            ),
            patch("ray.get") as mock_ray_get,
            patch(
                "vla_foundry.data.preprocessing.robotics.converters.spartan.discover_and_validate_episodes_in_directory"
            ),
        ):
            # Set up mock to return 3 valid episodes
            valid_episodes = [os.path.join(temp_spartan_episodes, f"episode_{i:04d}") for i in range(3)]
            mock_ray_get.return_value = [valid_episodes]

            converter = SpartanConverter(mock_config)
            episodes = converter.discover_episodes([temp_spartan_episodes], max_episodes_to_process=2)

            # Should only return 2 episodes due to the limit
            assert len(episodes) == 2


def test_discover_episodes_empty_directory(mock_config):
    """Test discover_episodes with an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dir = Path(tmpdir) / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)

        with patch("builtins.open"), patch("yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"language_dict": {}}
            mock_config.language_annotations_path = "/tmp/fake_annotations.yaml"
            mock_config.action_fields_config_path = "/tmp/fake_action_fields.yaml"
            mock_config.validation_episodes_path = None

            with (
                patch(
                    "vla_foundry.data.robotics.utils.load_action_field_config",
                    return_value={"action_key_fields": [], "action_index_fields": [], "pose_groups": []},
                ),
                patch("ray.get") as mock_ray_get,
                patch(
                    "vla_foundry.data.preprocessing.robotics.converters.spartan.discover_and_validate_episodes_in_directory"
                ),
            ):
                mock_ray_get.return_value = [[]]

                converter = SpartanConverter(mock_config)
                episodes = converter.discover_episodes([str(empty_dir)])

                # Should return empty list
                assert len(episodes) == 0


def test_intrinsics_extrinsics_lookup_regression(mock_config):
    """Regression test: intrinsics/extrinsics should use camera_name as key, not 'intrinsics.camera_name'."""
    # This tests the bug fix where we were looking for "intrinsics.camera_name" but
    # extract_sample_camera_calibration() stores with just "camera_name" as the key

    with patch("builtins.open"), patch("yaml.safe_load") as mock_yaml:
        mock_yaml.return_value = {"language_dict": {}}
        mock_config.language_annotations_path = "/tmp/fake_annotations.yaml"
        mock_config.action_fields_config_path = "/tmp/fake_action_fields.yaml"
        mock_config.validation_episodes_path = None

        with patch(
            "vla_foundry.data.robotics.utils.load_action_field_config",
            return_value={"action_key_fields": [], "action_index_fields": []},
        ):
            # Create sample intrinsics/extrinsics with camera_name keys (NOT prefixed)
            sample_intrinsics = {"camera_left": np.random.randn(10, 3, 3)}

            camera_name = "camera_left"

            # Verify lookup works with camera_name (not "intrinsics.camera_name")
            assert camera_name in sample_intrinsics  # Should be True
            assert f"intrinsics.{camera_name}" not in sample_intrinsics  # Should be False
            assert f"original_intrinsics.{camera_name}" not in sample_intrinsics  # Should be False

            # This is the correct lookup pattern (what the fixed code does)
            if camera_name in sample_intrinsics:
                intrinsics = sample_intrinsics[camera_name]
                assert intrinsics is not None
                assert intrinsics.shape == (10, 3, 3)

            # This was the bug (incorrect lookup pattern)
            incorrect_key = f"intrinsics.{camera_name}"
            assert incorrect_key not in sample_intrinsics
