"""Tests for file_utils module."""

import os
import tempfile

from vla_foundry.file_utils import check_directory_has_files_with_prefix


class TestCheckDirectoryHasFilesWithPrefix:
    """Tests for check_directory_has_files_with_prefix function."""

    def test_empty_directory(self):
        """Test with an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_directory_has_files_with_prefix(tmpdir, "episode_")
            assert result == []

    def test_nonexistent_directory(self):
        """Test with a non-existent directory."""
        result = check_directory_has_files_with_prefix("/nonexistent/path", "episode_")
        assert result == []

    def test_directory_with_matching_files(self):
        """Test with directory containing files with matching prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files with matching prefix
            open(os.path.join(tmpdir, "episode_001.tar"), "w").close()
            open(os.path.join(tmpdir, "episode_002.tar"), "w").close()
            open(os.path.join(tmpdir, "other_file.txt"), "w").close()

            result = check_directory_has_files_with_prefix(tmpdir, "episode_")
            assert len(result) == 2
            assert "episode_001.tar" in result
            assert "episode_002.tar" in result
            assert "other_file.txt" not in result

    def test_directory_with_no_matching_files(self):
        """Test with directory containing no files with matching prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different prefix
            open(os.path.join(tmpdir, "shard_001.tar"), "w").close()
            open(os.path.join(tmpdir, "manifest.json"), "w").close()

            result = check_directory_has_files_with_prefix(tmpdir, "episode_")
            assert result == []

    def test_different_prefixes(self):
        """Test with different prefixes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different prefixes
            open(os.path.join(tmpdir, "shard_001.tar"), "w").close()
            open(os.path.join(tmpdir, "shard_002.tar"), "w").close()
            open(os.path.join(tmpdir, "episode_001.tar"), "w").close()

            result = check_directory_has_files_with_prefix(tmpdir, "shard_")
            assert len(result) == 2
            assert "shard_001.tar" in result
            assert "shard_002.tar" in result

    def test_subdirectories_not_included(self):
        """Test that subdirectories are included in the results (they're listed as items)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory with matching prefix
            os.makedirs(os.path.join(tmpdir, "episode_subdir"))
            open(os.path.join(tmpdir, "episode_file.tar"), "w").close()

            result = check_directory_has_files_with_prefix(tmpdir, "episode_")
            assert len(result) == 2
            assert "episode_file.tar" in result
            assert "episode_subdir" in result
