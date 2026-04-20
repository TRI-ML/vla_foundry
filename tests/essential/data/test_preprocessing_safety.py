"""Tests for preprocessing safety checks."""

from unittest.mock import MagicMock, patch

import pytest


class TestPreprocessRoboticsToTarSafety:
    """Tests for preprocessing safety checks in preprocess_robotics_to_tar.py."""

    @patch("vla_foundry.data.preprocessing.preprocess_robotics_to_tar.check_directory_has_files_with_substring")
    @patch("vla_foundry.data.preprocessing.preprocess_robotics_to_tar.draccus.parse")
    @patch("argparse.ArgumentParser.parse_known_args")
    def test_main_fails_with_existing_episode_files(self, mock_parse_args, mock_draccus_parse, mock_check_dir):
        """Test that main() fails when output directory has existing episode files."""
        # Mock argparse
        mock_args = MagicMock()
        mock_args.type = "lerobot"
        mock_parse_args.return_value = (mock_args, [])

        # Mock the config
        mock_cfg = MagicMock()
        mock_cfg.source_episodes = "/some/source"
        mock_cfg.output_dir = "s3://bucket/output"
        mock_draccus_parse.return_value = mock_cfg

        # Mock existing files
        mock_check_dir.return_value = ["episode_001_frame_00000.tar", "episode_002_frame_00000.tar"]

        from vla_foundry.data.preprocessing.preprocess_robotics_to_tar import main

        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            main()

        assert "Output directory is not empty" in str(exc_info.value)
        assert "episode_001_frame_00000.tar" in str(exc_info.value)
        mock_check_dir.assert_called_once_with("s3://bucket/output/frames", "_frame_")

    @patch("vla_foundry.data.preprocessing.preprocess_robotics_to_tar.get_converter")
    @patch("vla_foundry.data.preprocessing.preprocess_robotics_to_tar.check_directory_has_files_with_substring")
    @patch("vla_foundry.data.preprocessing.preprocess_robotics_to_tar.draccus.parse")
    @patch("argparse.ArgumentParser.parse_known_args")
    @patch("vla_foundry.data.preprocessing.preprocess_robotics_to_tar.ray.init")
    def test_main_continues_with_empty_directory(
        self, mock_ray_init, mock_parse_args, mock_draccus_parse, mock_check_dir, mock_get_converter
    ):
        """Test that main() continues when output directory is empty."""
        # Mock argparse
        mock_args = MagicMock()
        mock_args.type = "lerobot"
        mock_parse_args.return_value = (mock_args, [])

        # Mock the config
        mock_cfg = MagicMock()
        mock_cfg.source_episodes = "/some/source"
        mock_cfg.output_dir = "s3://bucket/output"
        mock_cfg.ray_address = None
        mock_cfg.ray_num_cpus = 4
        mock_cfg.type = "lerobot"
        mock_draccus_parse.return_value = mock_cfg

        # Mock empty directory
        mock_check_dir.return_value = []

        # Mock converter
        mock_converter = MagicMock()
        mock_converter.discover_episodes.return_value = []
        mock_get_converter.return_value = mock_converter

        from vla_foundry.data.preprocessing.preprocess_robotics_to_tar import main

        # Should not raise, but will exit early due to no episodes
        # We're just testing that it passes the safety check
        try:
            main()
        except Exception as e:
            # It's ok if it fails later, we just want to ensure it didn't fail on our safety check
            assert "Output directory is not empty" not in str(e)

        mock_check_dir.assert_called_once_with("s3://bucket/output/frames", "_frame_")
