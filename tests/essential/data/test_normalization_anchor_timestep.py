"""
Tests for normalization and denormalization with anchor_timestep parameter.

These tests verify that:
1. normalize_tensor correctly aligns cropped sequences with statistics using anchor_timestep
2. denormalize_tensor correctly inverts the normalization
3. The round-trip (normalize -> denormalize) preserves the original data
4. Edge cases are handled correctly (boundary conditions, different anchor positions)
"""

import os
import tempfile

import pytest
import torch

from vla_foundry.data.robotics.normalization import RoboticsNormalizer
from vla_foundry.params.robotics.normalization_params import FieldNormalizationParams, NormalizationParams


@pytest.fixture
def dataset_dir():
    """Get the path to the test dataset."""
    return os.path.join(os.path.dirname(__file__), "..", "test_assets", "small_lbm_dataset")


@pytest.fixture
def stats_path(dataset_dir):
    """Get the path to the dataset statistics."""
    return os.path.join(dataset_dir, "stats.json")


@pytest.fixture
def temp_stats():
    """Create temporary statistics for testing."""
    # Create statistics with known values for easier testing
    # Assume 16 timesteps total (1 past + 15 future from preprocessing)
    num_timesteps = 16
    feature_dim = 3

    stats = {
        "test_field": {
            "mean": [float(d * 10) for d in range(feature_dim)],
            "std": [float(1.0 + d * 0.5) for d in range(feature_dim)],
            "mean_per_timestep": [[float(t * 10 + d) for d in range(feature_dim)] for t in range(num_timesteps)],
            "std_per_timestep": [[float(1.0 + t * 0.1) for d in range(feature_dim)] for t in range(num_timesteps)],
            "percentile_5": [float(-d * 5) for d in range(feature_dim)],
            "percentile_95": [float(d * 5 + 10) for d in range(feature_dim)],
            "percentile_5_per_timestep": [
                [float(-t * 5 - d) for d in range(feature_dim)] for t in range(num_timesteps)
            ],
            "percentile_95_per_timestep": [
                [float(t * 5 + d + 10) for d in range(feature_dim)] for t in range(num_timesteps)
            ],
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json

        json.dump(stats, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


class TestNormalizationWithAnchorTimestep:
    """Test normalize_tensor with anchor_timestep parameter."""

    def test_normalize_without_anchor(self, temp_stats):
        """Test normalization without anchor_timestep (backward compatibility)."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        # Create a tensor with shape [B, T, D]
        batch_size, num_timesteps, feature_dim = 2, 16, 3
        tensor = torch.randn(batch_size, num_timesteps, feature_dim)

        # Normalize without anchor_timestep
        normalized = normalizer.normalize_tensor(tensor, "test_field")

        # Verify shape is preserved
        assert normalized.shape == tensor.shape

        # Verify statistics were applied (check it's not the same as input)
        assert not torch.allclose(normalized, tensor)

    def test_normalize_with_anchor_at_start(self, temp_stats):
        """Test normalization with anchor at the start of the sequence."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        # Create a cropped tensor with anchor at index 0 (no past, 8 future)
        batch_size, num_timesteps, feature_dim = 2, 9, 3
        tensor = torch.randn(batch_size, num_timesteps, feature_dim)

        # Normalize with anchor at index 0
        normalized = normalizer.normalize_tensor(tensor, "test_field", anchor_timestep=0)

        # Verify shape is preserved
        assert normalized.shape == tensor.shape
        assert torch.isfinite(normalized).all()

    def test_normalize_with_anchor_in_middle(self, temp_stats):
        """Test normalization with anchor in the middle of the sequence."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        # Create a tensor with anchor at index 1 (1 past, 8 future)
        batch_size, num_timesteps, feature_dim = 2, 10, 3
        tensor = torch.randn(batch_size, num_timesteps, feature_dim)

        # Normalize with anchor at index 1
        normalized = normalizer.normalize_tensor(tensor, "test_field", anchor_timestep=1)

        # Verify shape is preserved
        assert normalized.shape == tensor.shape
        assert torch.isfinite(normalized).all()

    def test_normalize_with_different_crop_sizes(self, temp_stats):
        """Test normalization with various crop sizes."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=2,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        batch_size, feature_dim = 2, 3

        # Test various combinations of past and future timesteps
        test_cases = [
            (0, 5, 0),  # anchor at 0, 0 past + 5 future + 1 anchor = 6 timesteps
            (1, 5, 1),  # anchor at 1, 1 past + 5 future + 1 anchor = 7 timesteps
            (2, 8, 2),  # anchor at 2, 2 past + 8 future + 1 anchor = 11 timesteps
            (1, 10, 1),  # anchor at 1, 1 past + 10 future + 1 anchor = 12 timesteps
        ]

        for past, future, anchor in test_cases:
            num_timesteps = past + future + 1
            tensor = torch.randn(batch_size, num_timesteps, feature_dim)

            normalized = normalizer.normalize_tensor(tensor, "test_field", anchor_timestep=anchor)

            assert normalized.shape == tensor.shape
            assert torch.isfinite(normalized).all()

    def test_normalize_global_scope_ignores_anchor(self, temp_stats):
        """Test that global scope normalization ignores anchor_timestep."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="global",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="global",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        batch_size, num_timesteps, feature_dim = 2, 10, 3
        tensor = torch.randn(batch_size, num_timesteps, feature_dim)

        # Normalize with and without anchor should give same result for global scope
        normalized_with_anchor = normalizer.normalize_tensor(tensor, "test_field", anchor_timestep=1)
        normalized_without_anchor = normalizer.normalize_tensor(tensor, "test_field")

        assert torch.allclose(normalized_with_anchor, normalized_without_anchor)


class TestDenormalizationWithAnchorTimestep:
    """Test denormalize_tensor with anchor_timestep parameter."""

    def test_denormalize_without_anchor(self, temp_stats):
        """Test denormalization without anchor_timestep (backward compatibility)."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        # Create a normalized tensor
        batch_size, num_timesteps, feature_dim = 2, 16, 3
        normalized = torch.randn(batch_size, num_timesteps, feature_dim)

        # Denormalize without anchor_timestep
        denormalized = normalizer.denormalize_tensor(normalized, "test_field")

        # Verify shape is preserved
        assert denormalized.shape == normalized.shape

        # Verify denormalization was applied
        assert not torch.allclose(denormalized, normalized)

    def test_denormalize_with_anchor_at_start(self, temp_stats):
        """Test denormalization with anchor at the start."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        batch_size, num_timesteps, feature_dim = 2, 9, 3
        normalized = torch.randn(batch_size, num_timesteps, feature_dim)

        # Denormalize with anchor at index 0
        denormalized = normalizer.denormalize_tensor(normalized, "test_field", anchor_timestep=0)

        assert denormalized.shape == normalized.shape
        assert torch.isfinite(denormalized).all()

    def test_denormalize_with_anchor_in_middle(self, temp_stats):
        """Test denormalization with anchor in the middle."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        batch_size, num_timesteps, feature_dim = 2, 10, 3
        normalized = torch.randn(batch_size, num_timesteps, feature_dim)

        # Denormalize with anchor at index 1
        denormalized = normalizer.denormalize_tensor(normalized, "test_field", anchor_timestep=1)

        assert denormalized.shape == normalized.shape
        assert torch.isfinite(denormalized).all()


class TestNormalizationDenormalizationRoundTrip:
    """Test that normalize -> denormalize preserves the original data."""

    def test_roundtrip_without_anchor(self, temp_stats):
        """Test round-trip without anchor_timestep."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        # Create original tensor
        batch_size, num_timesteps, feature_dim = 2, 16, 3
        original = torch.randn(batch_size, num_timesteps, feature_dim)

        # Normalize then denormalize
        normalized = normalizer.normalize_tensor(original, "test_field")
        reconstructed = normalizer.denormalize_tensor(normalized, "test_field")

        # Verify we get back the original (within numerical precision)
        assert torch.allclose(reconstructed, original, rtol=1e-4, atol=1e-5)

    def test_roundtrip_with_anchor_at_start(self, temp_stats):
        """Test round-trip with anchor at start."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        batch_size, num_timesteps, feature_dim = 2, 9, 3
        original = torch.randn(batch_size, num_timesteps, feature_dim)

        # Normalize then denormalize with anchor at 0
        normalized = normalizer.normalize_tensor(original, "test_field", anchor_timestep=0)
        reconstructed = normalizer.denormalize_tensor(normalized, "test_field", anchor_timestep=0)

        assert torch.allclose(reconstructed, original, rtol=1e-4, atol=1e-5)

    def test_roundtrip_with_anchor_in_middle(self, temp_stats):
        """Test round-trip with anchor in middle."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        batch_size, num_timesteps, feature_dim = 2, 10, 3
        original = torch.randn(batch_size, num_timesteps, feature_dim)

        # Normalize then denormalize with anchor at 1
        normalized = normalizer.normalize_tensor(original, "test_field", anchor_timestep=1)
        reconstructed = normalizer.denormalize_tensor(normalized, "test_field", anchor_timestep=1)

        assert torch.allclose(reconstructed, original, rtol=1e-4, atol=1e-5)

    def test_roundtrip_with_various_anchors(self, temp_stats):
        """Test round-trip with various anchor positions."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=6,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        batch_size, feature_dim = 2, 3

        # Test various anchor positions
        test_cases = [
            (0, 9, 0),  # anchor at 0, 9 timesteps total
            (1, 10, 1),  # anchor at 1, 10 timesteps total
            (2, 11, 2),  # anchor at 2, 11 timesteps total
            (5, 14, 5),  # anchor at 5, 14 timesteps total
        ]

        for anchor, num_timesteps, _ in test_cases:
            original = torch.randn(batch_size, num_timesteps, feature_dim)

            normalized = normalizer.normalize_tensor(original, "test_field", anchor_timestep=anchor)
            reconstructed = normalizer.denormalize_tensor(normalized, "test_field", anchor_timestep=anchor)

            assert torch.allclose(reconstructed, original, rtol=1e-4, atol=1e-5), (
                f"Round-trip failed for anchor={anchor}, num_timesteps={num_timesteps}"
            )

    def test_roundtrip_global_scope(self, temp_stats):
        """Test round-trip with global scope normalization."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="global",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="global",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        batch_size, num_timesteps, feature_dim = 2, 10, 3
        original = torch.randn(batch_size, num_timesteps, feature_dim)

        # Global scope should work with or without anchor
        normalized = normalizer.normalize_tensor(original, "test_field", anchor_timestep=1)
        reconstructed = normalizer.denormalize_tensor(normalized, "test_field", anchor_timestep=1)

        assert torch.allclose(reconstructed, original, rtol=1e-4, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_anchor_at_boundary_of_stats(self, temp_stats):
        """Test anchor position at the boundary of available statistics."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        # Test with anchor that would require stats beyond what's available
        # Stats have 16 timesteps (indices 0-15)
        # If we have anchor at 0 and request 20 future timesteps, we exceed stats
        batch_size, num_timesteps, feature_dim = 2, 17, 3  # More than stats available
        original = torch.randn(batch_size, num_timesteps, feature_dim)

        with pytest.raises(ValueError):
            normalizer.normalize_tensor(original, "test_field", anchor_timestep=0)

    def test_single_timestep(self, temp_stats):
        """Test with single timestep tensor."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        batch_size, num_timesteps, feature_dim = 2, 1, 3
        original = torch.randn(batch_size, num_timesteps, feature_dim)

        normalized = normalizer.normalize_tensor(original, "test_field", anchor_timestep=0)
        reconstructed = normalizer.denormalize_tensor(normalized, "test_field", anchor_timestep=0)

        assert torch.allclose(reconstructed, original, rtol=1e-4, atol=1e-5)

    def test_2d_tensor_with_global_scope(self, temp_stats):
        """Test that 2D tensors work with global scope normalization."""
        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="global",
                include_fields=["test_field"],
                field_configs={
                    "test_field": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="global",
                    )
                },
                lowdim_past_timesteps=1,
                lowdim_future_timesteps=15,
            ),
            statistics_path=temp_stats,
        )

        # 2D tensor (no time dimension)
        batch_size, feature_dim = 2, 3
        original = torch.randn(batch_size, feature_dim)

        # Should work with global normalization
        normalized = normalizer.normalize_tensor(original, "test_field")
        reconstructed = normalizer.denormalize_tensor(normalized, "test_field")

        assert torch.allclose(reconstructed, original, rtol=1e-4, atol=1e-5)


class TestConsistencyWithRealData:
    """Test consistency with real dataset statistics."""

    def test_with_real_stats(self, stats_path):
        """Test normalization/denormalization with real dataset statistics."""
        if not os.path.exists(stats_path):
            pytest.skip(f"Stats file not found at {stats_path}")

        normalizer = RoboticsNormalizer(
            normalization_params=NormalizationParams(
                enabled=True,
                method="std",
                scope="per_timestep",
                include_fields=["robot__action__poses__right::panda__xyz"],
                field_configs={
                    "robot__action__poses__right::panda__xyz": FieldNormalizationParams(
                        enabled=True,
                        method="std",
                        scope="per_timestep",
                    )
                },
                lowdim_past_timesteps=2,
                lowdim_future_timesteps=15,
            ),
            statistics_path=stats_path,
        )

        field_name = "robot__action__poses__right::panda__xyz"

        # Test various cropping scenarios
        test_cases = [
            (0, 9),  # anchor at 0
            (1, 10),  # anchor at 1
            (2, 11),  # anchor at 2
        ]

        for anchor, num_timesteps in test_cases:
            batch_size, feature_dim = 2, 3
            original = torch.randn(batch_size, num_timesteps, feature_dim)

            normalized = normalizer.normalize_tensor(original, field_name, anchor_timestep=anchor)
            reconstructed = normalizer.denormalize_tensor(normalized, field_name, anchor_timestep=anchor)

            assert torch.allclose(reconstructed, original, rtol=1e-4, atol=1e-5), (
                f"Round-trip failed with real stats for anchor={anchor}"
            )
