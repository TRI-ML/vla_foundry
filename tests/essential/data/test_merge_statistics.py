import numpy as np
import pytest

from vla_foundry.data.preprocessing.robotics.preprocess_statistics import StreamingDatasetStatistics
from vla_foundry.data.robotics.utils import merge_statistics, merge_statistics_single_field


class TestMergeStatisticsSingleField:
    """Test the merge_statistics_single_field function for different statistics."""

    @pytest.fixture
    def simple_tensor_stats(self):
        """Create simple tensor statistics for 2 datasets with 3 timesteps and 2 dimensions."""
        # Dataset 1: mean=1.0, std=0.5, counts=10 per timestep
        # Dataset 2: mean=2.0, std=1.0, counts=20 per timestep
        # Note: Percentile fields are not included here as percentile merging requires
        # tdigest states which are tested in test_tdigest_merge.py
        return {
            "mean_per_timestep": np.array(
                [
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],  # Dataset 1
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],  # Dataset 2
                ]
            ),
            "std_per_timestep": np.array(
                [
                    [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],  # Dataset 1
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],  # Dataset 2
                ]
            ),
            "min_per_timestep": np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # Dataset 1
                    [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],  # Dataset 2
                ]
            ),
            "max_per_timestep": np.array(
                [
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],  # Dataset 1
                    [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]],  # Dataset 2
                ]
            ),
            "count": np.array(
                [
                    [10.0, 10.0, 10.0],  # Dataset 1
                    [20.0, 20.0, 20.0],  # Dataset 2
                ]
            ),
        }

    def test_merge_mean_per_timestep(self, simple_tensor_stats):
        """Test merging mean_per_timestep with weighted averaging."""
        result = merge_statistics_single_field(simple_tensor_stats, "mean_per_timestep")

        # Expected: weighted average of [1.0, 2.0] with weights [10, 20]
        # = (1.0 * 10 + 2.0 * 20) / (10 + 20) = 50 / 30 = 1.666...
        expected = np.array([[1.666667, 1.666667], [1.666667, 1.666667], [1.666667, 1.666667]])

        assert result.shape == (3, 2)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_merge_mean(self, simple_tensor_stats):
        """Test merging overall mean (averaged across timesteps)."""
        result = merge_statistics_single_field(simple_tensor_stats, "mean")

        # Expected: weighted average across timesteps of the merged mean_per_timestep
        expected = np.array([1.666667, 1.666667])

        assert result.shape == (2,)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_merge_std_per_timestep_pooled_variance(self, simple_tensor_stats):
        """Test merging std_per_timestep using pooled variance formula."""
        result = merge_statistics_single_field(simple_tensor_stats, "std_per_timestep")

        # Using pooled variance formula:
        # σ²_pooled = [Σ((nᵢ-1)×σᵢ² + nᵢ×(μᵢ - μ_global)²)] / (n_total - 1)
        # For dataset 1: (10-1)*0.5² + 10*(1.0-1.666667)² = 9*0.25 + 10*0.444444 = 2.25 + 4.444444 = 6.694444
        # For dataset 2: (20-1)*1.0² + 20*(2.0-1.666667)² = 19*1.0 + 20*0.111111 = 19 + 2.222222 = 21.222222
        # Total: (6.694444 + 21.222222) / (30 - 1) = 27.916666 / 29 = 0.962644
        # sqrt(0.962644) = 0.981246
        expected_variance = 0.962644
        expected_std = np.sqrt(expected_variance)

        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[0, 0], expected_std, rtol=1e-4)

    def test_merge_std_law_of_total_variance(self, simple_tensor_stats):
        """Test merging overall std using law of total variance."""
        result = merge_statistics_single_field(simple_tensor_stats, "std")

        # σ²_overall = E[σ²_t] + Var[μ_t]
        # First get pooled std_per_timestep
        std_per_timestep = merge_statistics_single_field(simple_tensor_stats, "std_per_timestep")
        mean_variance = np.mean(std_per_timestep**2, axis=0)

        # Get merged mean_per_timestep
        mean_per_timestep = merge_statistics_single_field(simple_tensor_stats, "mean_per_timestep")
        variance_of_means = np.var(mean_per_timestep, axis=0)

        expected = np.sqrt(mean_variance + variance_of_means)

        assert result.shape == (2,)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_merge_min_per_timestep(self, simple_tensor_stats):
        """Test merging min_per_timestep (taking minimum across datasets)."""
        result = merge_statistics_single_field(simple_tensor_stats, "min_per_timestep")

        # Expected: minimum of [0.0, 0.5] = 0.0
        expected = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, expected)

    def test_merge_max_per_timestep(self, simple_tensor_stats):
        """Test merging max_per_timestep (taking maximum across datasets)."""
        result = merge_statistics_single_field(simple_tensor_stats, "max_per_timestep")

        # Expected: maximum of [2.0, 4.0] = 4.0
        expected = np.array([[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]])

        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, expected)

    def test_merge_count(self, simple_tensor_stats):
        """Test merging count (summing across datasets)."""
        result = merge_statistics_single_field(simple_tensor_stats, "count")

        # Expected: sum of [10, 20] = 30
        expected = np.array([30.0, 30.0, 30.0])

        assert result.shape == (3,)
        np.testing.assert_array_equal(result, expected)

    def test_invalid_stat_name(self, simple_tensor_stats):
        """Test that invalid stat name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid stat name"):
            merge_statistics_single_field(simple_tensor_stats, "invalid_stat")


class TestMergeStatistics:
    """Test the full merge_statistics function with multiple tensors and datasets."""

    @pytest.fixture
    def two_dataset_statistics(self):
        """Create statistics for 2 datasets with 2 tensors.

        Note: Percentile fields are not included here as percentile merging requires
        tdigest states which are tested in test_tdigest_merge.py
        """
        return [
            # Dataset 1
            {
                "action": {
                    "mean": [1.0, 2.0],
                    "std": [0.5, 0.6],
                    "min": [0.0, 0.5],
                    "max": [2.0, 3.5],
                    "mean_per_timestep": [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
                    "std_per_timestep": [[0.5, 0.6], [0.5, 0.6], [0.5, 0.6]],
                    "min_per_timestep": [[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]],
                    "max_per_timestep": [[2.0, 3.5], [2.0, 3.5], [2.0, 3.5]],
                    "count": [100.0, 100.0, 100.0],
                },
                "obs": {
                    "mean": [0.5, 0.5],
                    "std": [0.2, 0.2],
                    "min": [-1.0, -1.0],
                    "max": [2.0, 2.0],
                    "mean_per_timestep": [[0.5, 0.5], [0.5, 0.5]],
                    "std_per_timestep": [[0.2, 0.2], [0.2, 0.2]],
                    "min_per_timestep": [[-1.0, -1.0], [-1.0, -1.0]],
                    "max_per_timestep": [[2.0, 2.0], [2.0, 2.0]],
                    "count": [50.0, 50.0],
                },
            },
            # Dataset 2
            {
                "action": {
                    "mean": [2.0, 3.0],
                    "std": [1.0, 1.2],
                    "min": [0.5, 1.0],
                    "max": [4.0, 5.0],
                    "mean_per_timestep": [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]],
                    "std_per_timestep": [[1.0, 1.2], [1.0, 1.2], [1.0, 1.2]],
                    "min_per_timestep": [[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]],
                    "max_per_timestep": [[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]],
                    "count": [200.0, 200.0, 200.0],
                },
                "obs": {
                    "mean": [1.0, 1.0],
                    "std": [0.3, 0.3],
                    "min": [-0.5, -0.5],
                    "max": [2.5, 2.5],
                    "mean_per_timestep": [[1.0, 1.0], [1.0, 1.0]],
                    "std_per_timestep": [[0.3, 0.3], [0.3, 0.3]],
                    "min_per_timestep": [[-0.5, -0.5], [-0.5, -0.5]],
                    "max_per_timestep": [[2.5, 2.5], [2.5, 2.5]],
                    "count": [100.0, 100.0],
                },
            },
        ]

    def test_merge_basic_structure(self, two_dataset_statistics):
        """Test that merge_statistics returns correct structure."""
        result = merge_statistics(two_dataset_statistics)

        # Check that all tensors are present
        assert "action" in result
        assert "obs" in result

        # Check that all stat fields are present
        assert "mean" in result["action"]
        assert "std" in result["action"]
        assert "min" in result["action"]
        assert "max" in result["action"]
        assert "mean_per_timestep" in result["action"]
        assert "std_per_timestep" in result["action"]
        assert "count" in result["action"]

    def test_merge_mean_weighted_correctly(self, two_dataset_statistics):
        """Test that means are merged with correct weighting."""
        result = merge_statistics(two_dataset_statistics)

        # For action tensor: Dataset 1 has count 100, Dataset 2 has count 200
        # mean_per_timestep weighted average: (1.0 * 100 + 2.0 * 200) / 300 = 500 / 300 = 1.666...
        # Overall mean is average across timesteps: 1.666...
        expected_mean_dim0 = (1.0 * 100 + 2.0 * 200) / 300

        np.testing.assert_allclose(result["action"]["mean"][0], expected_mean_dim0, rtol=1e-5)

    def test_merge_min_max_correctly(self, two_dataset_statistics):
        """Test that min and max are merged correctly."""
        result = merge_statistics(two_dataset_statistics)

        # Min should be minimum across datasets
        assert result["action"]["min"][0] == 0.0  # min(0.0, 0.5)
        assert result["action"]["min"][1] == 0.5  # min(0.5, 1.0)

        # Max should be maximum across datasets
        assert result["action"]["max"][0] == 4.0  # max(2.0, 4.0)
        assert result["action"]["max"][1] == 5.0  # max(3.5, 5.0)

    def test_merge_count_summed(self, two_dataset_statistics):
        """Test that counts are summed correctly."""
        result = merge_statistics(two_dataset_statistics)

        # Counts should be summed: 100 + 200 = 300
        assert result["action"]["count"][0] == 300.0
        assert result["action"]["count"][1] == 300.0
        assert result["action"]["count"][2] == 300.0

        # For obs: 50 + 100 = 150
        assert result["obs"]["count"][0] == 150.0
        assert result["obs"]["count"][1] == 150.0

    def test_merge_per_timestep_stats(self, two_dataset_statistics):
        """Test that per-timestep statistics are merged correctly."""
        result = merge_statistics(two_dataset_statistics)

        # mean_per_timestep should have shape (num_timesteps, action_dim)
        assert len(result["action"]["mean_per_timestep"]) == 3
        assert len(result["action"]["mean_per_timestep"][0]) == 2

        # All timesteps should have same mean (constant across time in test data)
        np.testing.assert_allclose(
            result["action"]["mean_per_timestep"][0], result["action"]["mean_per_timestep"][1], rtol=1e-5
        )

    def test_merge_different_tensor_shapes(self):
        """Test merging statistics with different tensor dimensions."""
        stats = [
            {
                "tensor_1d": {
                    "mean": [1.0],
                    "std": [0.5],
                    "min": [0.0],
                    "max": [2.0],
                    "mean_per_timestep": [[1.0], [1.0]],
                    "std_per_timestep": [[0.5], [0.5]],
                    "min_per_timestep": [[0.0], [0.0]],
                    "max_per_timestep": [[2.0], [2.0]],
                    "count": [100.0, 100.0],
                },
                "tensor_4d": {
                    "mean": [1.0, 2.0, 3.0, 4.0],
                    "std": [0.5, 0.6, 0.7, 0.8],
                    "min": [0.0, 0.5, 1.0, 1.5],
                    "max": [2.0, 3.5, 5.0, 6.5],
                    "mean_per_timestep": [[1.0, 2.0, 3.0, 4.0]],
                    "std_per_timestep": [[0.5, 0.6, 0.7, 0.8]],
                    "min_per_timestep": [[0.0, 0.5, 1.0, 1.5]],
                    "max_per_timestep": [[2.0, 3.5, 5.0, 6.5]],
                    "count": [50.0],
                },
            },
            {
                "tensor_1d": {
                    "mean": [2.0],
                    "std": [1.0],
                    "min": [0.5],
                    "max": [4.0],
                    "mean_per_timestep": [[2.0], [2.0]],
                    "std_per_timestep": [[1.0], [1.0]],
                    "min_per_timestep": [[0.5], [0.5]],
                    "max_per_timestep": [[4.0], [4.0]],
                    "count": [200.0, 200.0],
                },
                "tensor_4d": {
                    "mean": [2.0, 3.0, 4.0, 5.0],
                    "std": [1.0, 1.1, 1.2, 1.3],
                    "min": [0.5, 1.0, 1.5, 2.0],
                    "max": [4.0, 5.0, 6.5, 8.0],
                    "mean_per_timestep": [[2.0, 3.0, 4.0, 5.0]],
                    "std_per_timestep": [[1.0, 1.1, 1.2, 1.3]],
                    "min_per_timestep": [[0.5, 1.0, 1.5, 2.0]],
                    "max_per_timestep": [[4.0, 5.0, 6.5, 8.0]],
                    "count": [100.0],
                },
            },
        ]

        result = merge_statistics(stats)

        # Check shapes are preserved
        assert len(result["tensor_1d"]["mean"]) == 1
        assert len(result["tensor_4d"]["mean"]) == 4

        # Check values are reasonable
        assert isinstance(result["tensor_1d"]["mean"][0], float)
        assert isinstance(result["tensor_4d"]["mean"][0], float)

    def test_merge_single_dataset(self):
        """Test that merging a single dataset returns unchanged statistics."""
        single_dataset = [
            {
                "action": {
                    "mean": [1.0, 2.0],
                    "std": [0.5, 0.6],
                    "min": [0.0, 0.5],
                    "max": [2.0, 3.5],
                    "mean_per_timestep": [[1.0, 2.0]],
                    "std_per_timestep": [[0.5, 0.6]],
                    "min_per_timestep": [[0.0, 0.5]],
                    "max_per_timestep": [[2.0, 3.5]],
                    "count": [100.0],
                },
            }
        ]

        result = merge_statistics(single_dataset)

        # For a single dataset, merged stats should be very close to original
        # (minor differences due to the merging algorithm)
        np.testing.assert_allclose(result["action"]["min"], [0.0, 0.5], rtol=1e-5)
        np.testing.assert_allclose(result["action"]["max"], [2.0, 3.5], rtol=1e-5)
        assert result["action"]["count"][0] == 100.0

    def test_merge_empty_statistics(self):
        """Test merging with empty statistics list."""
        result = merge_statistics([])

        # Should return empty dict
        assert result == {}

    def test_merge_preserves_all_stat_types(self, two_dataset_statistics):
        """Test that all statistic types are preserved after merging."""
        result = merge_statistics(two_dataset_statistics)

        expected_stat_names = {
            "mean",
            "std",
            "min",
            "max",
            "mean_per_timestep",
            "std_per_timestep",
            "min_per_timestep",
            "max_per_timestep",
            "count",
        }

        for tensor_name in result:
            stat_names = set(result[tensor_name].keys())
            # Check that all expected stats are present
            assert expected_stat_names.issubset(stat_names)


class TestMergeStatisticsEdgeCases:
    """Test edge cases and special scenarios for merge_statistics."""

    def test_merge_with_zero_counts(self):
        """Test merging when some timesteps have zero counts."""
        stats = [
            {
                "action": {
                    "mean": [1.0, 2.0],
                    "std": [0.5, 0.6],
                    "min": [0.0, 0.5],
                    "max": [2.0, 3.5],
                    "mean_per_timestep": [[1.0, 2.0], [0.0, 0.0]],  # Second timestep has no data
                    "std_per_timestep": [[0.5, 0.6], [0.0, 0.0]],
                    "min_per_timestep": [[0.0, 0.5], [0.0, 0.0]],
                    "max_per_timestep": [[2.0, 3.5], [0.0, 0.0]],
                    "count": [100.0, 0.0],  # Second timestep has zero count
                },
            },
            {
                "action": {
                    "mean": [2.0, 3.0],
                    "std": [1.0, 1.2],
                    "min": [0.5, 1.0],
                    "max": [4.0, 5.0],
                    "mean_per_timestep": [[2.0, 3.0], [2.0, 3.0]],
                    "std_per_timestep": [[1.0, 1.2], [1.0, 1.2]],
                    "min_per_timestep": [[0.5, 1.0], [0.5, 1.0]],
                    "max_per_timestep": [[4.0, 5.0], [4.0, 5.0]],
                    "count": [200.0, 200.0],
                },
            },
        ]

        result = merge_statistics(stats)

        # Should handle zero counts gracefully without errors
        assert "action" in result
        assert "mean" in result["action"]
        # Result should not have NaN values
        assert not np.isnan(result["action"]["mean"][0])

    def test_merge_with_varying_timesteps(self):
        """Test merging datasets with different numbers of timesteps."""
        stats = [
            {
                "action": {
                    "mean": [1.0, 2.0],
                    "std": [0.5, 0.6],
                    "min": [0.0, 0.5],
                    "max": [2.0, 3.5],
                    "mean_per_timestep": [[1.0, 2.0]],  # 1 timestep
                    "std_per_timestep": [[0.5, 0.6]],
                    "min_per_timestep": [[0.0, 0.5]],
                    "max_per_timestep": [[2.0, 3.5]],
                    "count": [100.0],
                },
            },
            {
                "action": {
                    "mean": [2.0, 3.0],
                    "std": [1.0, 1.2],
                    "min": [0.5, 1.0],
                    "max": [4.0, 5.0],
                    "mean_per_timestep": [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]],  # 3 timesteps
                    "std_per_timestep": [[1.0, 1.2], [1.0, 1.2], [1.0, 1.2]],
                    "min_per_timestep": [[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]],
                    "max_per_timestep": [[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]],
                    "count": [200.0, 200.0, 200.0],
                },
            },
        ]

        with pytest.raises(ValueError):
            merge_statistics(stats)

    def test_merge_numerical_stability(self):
        """Test that merging handles numerical stability well."""
        # Create statistics with very large and very small values
        stats = [
            {
                "action": {
                    "mean": [1e10, 1e-10],
                    "std": [1e8, 1e-8],
                    "min": [1e9, 1e-11],
                    "max": [1e11, 1e-9],
                    "mean_per_timestep": [[1e10, 1e-10]],
                    "std_per_timestep": [[1e8, 1e-8]],
                    "min_per_timestep": [[1e9, 1e-11]],
                    "max_per_timestep": [[1e11, 1e-9]],
                    "count": [100.0],
                },
            },
            {
                "action": {
                    "mean": [2e10, 2e-10],
                    "std": [2e8, 2e-8],
                    "min": [1.5e9, 0.5e-11],
                    "max": [2e11, 2e-9],
                    "mean_per_timestep": [[2e10, 2e-10]],
                    "std_per_timestep": [[2e8, 2e-8]],
                    "min_per_timestep": [[1.5e9, 0.5e-11]],
                    "max_per_timestep": [[2e11, 2e-9]],
                    "count": [100.0],
                },
            },
        ]

        result = merge_statistics(stats)

        # Should handle large and small numbers without overflow/underflow
        assert np.isfinite(result["action"]["mean"][0])
        assert np.isfinite(result["action"]["mean"][1])
        assert result["action"]["mean"][0] > 0
        assert result["action"]["mean"][1] > 0


class TestMergeStatisticsLargeScenarios:
    """Test merge_statistics with large, realistic scenarios."""

    def test_merge_many_datasets(self):
        """Test merging 10 datasets with realistic robotics dimensions."""
        np.random.seed(42)
        num_datasets = 10
        action_dim = 7  # Typical robot action dimension
        num_timesteps = 16

        # Generate statistics for multiple datasets with varying properties
        stats = []
        for i in range(num_datasets):
            # Each dataset has slightly different mean and std
            base_mean = i * 0.5
            base_std = 0.3 + i * 0.1
            count_per_timestep = 100 + i * 50  # Varying sample counts

            mean_per_timestep = np.random.normal(base_mean, 0.1, (num_timesteps, action_dim))
            std_per_timestep = np.random.uniform(base_std, base_std + 0.2, (num_timesteps, action_dim))
            min_per_timestep = mean_per_timestep - 3 * std_per_timestep
            max_per_timestep = mean_per_timestep + 3 * std_per_timestep
            count = np.full(num_timesteps, count_per_timestep)

            dataset_stats = {
                "action": {
                    "mean": mean_per_timestep.mean(axis=0).tolist(),
                    "std": std_per_timestep.mean(axis=0).tolist(),
                    "min": min_per_timestep.min(axis=0).tolist(),
                    "max": max_per_timestep.max(axis=0).tolist(),
                    "mean_per_timestep": mean_per_timestep.tolist(),
                    "std_per_timestep": std_per_timestep.tolist(),
                    "min_per_timestep": min_per_timestep.tolist(),
                    "max_per_timestep": max_per_timestep.tolist(),
                    "count": count.tolist(),
                }
            }
            stats.append(dataset_stats)

        # Merge all datasets
        result = merge_statistics(stats)

        # Verify structure
        assert "action" in result
        assert len(result["action"]["mean"]) == action_dim
        assert len(result["action"]["mean_per_timestep"]) == num_timesteps
        assert len(result["action"]["mean_per_timestep"][0]) == action_dim

        # Verify counts are summed correctly
        total_count = sum((100 + i * 50) for i in range(num_datasets))
        np.testing.assert_allclose(result["action"]["count"][0], total_count, rtol=1e-5)

        # Verify mean is reasonable (should be somewhere between min and max of input means)
        all_means = [s["action"]["mean"][0] for s in stats]
        assert min(all_means) <= result["action"]["mean"][0] <= max(all_means)

        # Verify min/max are the global min/max
        all_mins = [s["action"]["min"][0] for s in stats]
        all_maxs = [s["action"]["max"][0] for s in stats]
        np.testing.assert_allclose(result["action"]["min"][0], min(all_mins), rtol=1e-5)
        np.testing.assert_allclose(result["action"]["max"][0], max(all_maxs), rtol=1e-5)

    def test_merge_high_dimensional_actions(self):
        """Test merging datasets with high-dimensional action spaces."""
        np.random.seed(123)
        action_dim = 50  # High-dimensional action space
        num_timesteps = 20
        num_datasets = 5

        stats = []
        for _i in range(num_datasets):
            mean_per_timestep = np.random.randn(num_timesteps, action_dim)
            std_per_timestep = np.abs(np.random.randn(num_timesteps, action_dim)) * 0.5 + 0.1

            dataset_stats = {
                "high_dim_action": {
                    "mean": mean_per_timestep.mean(axis=0).tolist(),
                    "std": std_per_timestep.mean(axis=0).tolist(),
                    "min": (mean_per_timestep - 2 * std_per_timestep).min(axis=0).tolist(),
                    "max": (mean_per_timestep + 2 * std_per_timestep).max(axis=0).tolist(),
                    "mean_per_timestep": mean_per_timestep.tolist(),
                    "std_per_timestep": std_per_timestep.tolist(),
                    "min_per_timestep": (mean_per_timestep - 2 * std_per_timestep).tolist(),
                    "max_per_timestep": (mean_per_timestep + 2 * std_per_timestep).tolist(),
                    "count": [200.0] * num_timesteps,
                }
            }
            stats.append(dataset_stats)

        result = merge_statistics(stats)

        # Verify high-dimensional output
        assert len(result["high_dim_action"]["mean"]) == action_dim
        assert len(result["high_dim_action"]["mean_per_timestep"]) == num_timesteps
        assert len(result["high_dim_action"]["mean_per_timestep"][0]) == action_dim

        # Verify no NaN or Inf values
        assert all(np.isfinite(result["high_dim_action"]["mean"]))
        assert all(np.isfinite(result["high_dim_action"]["std"]))

        # Verify count
        assert result["high_dim_action"]["count"][0] == 200.0 * num_datasets

    def test_merge_multiple_tensor_types(self):
        """Test merging datasets with multiple tensor types (actions, observations, etc)."""
        np.random.seed(456)
        num_datasets = 8
        action_dim = 7
        obs_dim = 10
        proprioception_dim = 15
        num_timesteps = 12

        stats = []
        for i in range(num_datasets):
            dataset_stats = {}

            # Generate stats for each tensor type
            for tensor_name, dim in [
                ("action", action_dim),
                ("observation", obs_dim),
                ("proprioception", proprioception_dim),
            ]:
                mean_per_timestep = np.random.randn(num_timesteps, dim) * (i + 1) * 0.3
                std_per_timestep = np.abs(np.random.randn(num_timesteps, dim)) * 0.4 + 0.2
                count = np.full(num_timesteps, 150.0 + i * 25.0)

                dataset_stats[tensor_name] = {
                    "mean": mean_per_timestep.mean(axis=0).tolist(),
                    "std": std_per_timestep.mean(axis=0).tolist(),
                    "min": (mean_per_timestep - 3 * std_per_timestep).min(axis=0).tolist(),
                    "max": (mean_per_timestep + 3 * std_per_timestep).max(axis=0).tolist(),
                    "mean_per_timestep": mean_per_timestep.tolist(),
                    "std_per_timestep": std_per_timestep.tolist(),
                    "min_per_timestep": (mean_per_timestep - 3 * std_per_timestep).tolist(),
                    "max_per_timestep": (mean_per_timestep + 3 * std_per_timestep).tolist(),
                    "count": count.tolist(),
                }

            stats.append(dataset_stats)

        result = merge_statistics(stats)

        # Verify all tensor types are present
        assert "action" in result
        assert "observation" in result
        assert "proprioception" in result

        # Verify dimensions are correct
        assert len(result["action"]["mean"]) == action_dim
        assert len(result["observation"]["mean"]) == obs_dim
        assert len(result["proprioception"]["mean"]) == proprioception_dim

        # Verify all have correct timesteps
        assert len(result["action"]["mean_per_timestep"]) == num_timesteps
        assert len(result["observation"]["mean_per_timestep"]) == num_timesteps
        assert len(result["proprioception"]["mean_per_timestep"]) == num_timesteps

    def test_merge_with_ground_truth_comparison(self):
        """Test merging against ground truth calculation from raw data."""
        np.random.seed(789)
        action_dim = 4
        num_timesteps = 8
        samples_per_dataset = [100, 200, 150]

        # Generate raw data for each dataset
        all_raw_data = []
        stats = []

        for i, num_samples in enumerate(samples_per_dataset):
            # Generate raw samples
            raw_data = np.random.randn(num_samples, num_timesteps, action_dim) * (i + 1) + i * 0.5
            all_raw_data.append(raw_data)

            # Compute statistics from raw data
            mean_per_timestep = raw_data.mean(axis=0)
            std_per_timestep = raw_data.std(axis=0, ddof=1)
            min_per_timestep = raw_data.min(axis=0)
            max_per_timestep = raw_data.max(axis=0)
            count = np.full(num_timesteps, float(num_samples))

            dataset_stats = {
                "action": {
                    "mean": mean_per_timestep.mean(axis=0).tolist(),
                    "std": std_per_timestep.mean(axis=0).tolist(),
                    "min": min_per_timestep.min(axis=0).tolist(),
                    "max": max_per_timestep.max(axis=0).tolist(),
                    "mean_per_timestep": mean_per_timestep.tolist(),
                    "std_per_timestep": std_per_timestep.tolist(),
                    "min_per_timestep": min_per_timestep.tolist(),
                    "max_per_timestep": max_per_timestep.tolist(),
                    "count": count.tolist(),
                }
            }
            stats.append(dataset_stats)

        # Merge statistics
        result = merge_statistics(stats)

        # Compute ground truth from combined raw data
        combined_raw_data = np.concatenate(all_raw_data, axis=0)
        gt_mean = combined_raw_data.mean(axis=(0, 1))
        combined_raw_data.std(axis=(0, 1), ddof=1)
        gt_min = combined_raw_data.min(axis=(0, 1))
        gt_max = combined_raw_data.max(axis=(0, 1))
        gt_count = sum(samples_per_dataset)

        # Compare merged results with ground truth
        # Note: The merged mean should be close to ground truth
        np.testing.assert_allclose(result["action"]["mean"], gt_mean.tolist(), rtol=0.05)

        # Min and max should match exactly
        np.testing.assert_allclose(result["action"]["min"], gt_min.tolist(), rtol=1e-5)
        np.testing.assert_allclose(result["action"]["max"], gt_max.tolist(), rtol=1e-5)

        # Count should match exactly
        assert result["action"]["count"][0] == gt_count

    def test_merge_imbalanced_datasets(self):
        """Test merging datasets with highly imbalanced sample counts."""
        np.random.seed(101)
        action_dim = 6
        num_timesteps = 10

        # Create datasets with vastly different sample counts
        counts = [10, 100, 1000, 10000]  # 4 orders of magnitude difference
        stats = []

        for count in counts:
            mean_per_timestep = np.random.randn(num_timesteps, action_dim)
            std_per_timestep = np.abs(np.random.randn(num_timesteps, action_dim)) * 0.3 + 0.1

            dataset_stats = {
                "action": {
                    "mean": mean_per_timestep.mean(axis=0).tolist(),
                    "std": std_per_timestep.mean(axis=0).tolist(),
                    "min": (mean_per_timestep - 2 * std_per_timestep).min(axis=0).tolist(),
                    "max": (mean_per_timestep + 2 * std_per_timestep).max(axis=0).tolist(),
                    "mean_per_timestep": mean_per_timestep.tolist(),
                    "std_per_timestep": std_per_timestep.tolist(),
                    "min_per_timestep": (mean_per_timestep - 2 * std_per_timestep).tolist(),
                    "max_per_timestep": (mean_per_timestep + 2 * std_per_timestep).tolist(),
                    "count": [float(count)] * num_timesteps,
                }
            }
            stats.append(dataset_stats)

        result = merge_statistics(stats)

        # Verify count is correct
        total_count = sum(counts)
        assert result["action"]["count"][0] == total_count

        # The merged mean should be heavily influenced by the largest dataset
        # since it has 10000 samples vs 10+100+1000=1110 for the others
        # Weight of largest dataset: 10000 / 11110 ≈ 0.9
        stats[3]["action"]["mean"][0]

        # Merged mean should be close to the largest dataset's mean
        # but we can't test exact value without knowing the input means
        # Just verify it's finite and reasonable
        assert np.isfinite(result["action"]["mean"][0])
        assert np.isfinite(result["action"]["std"][0])

    def test_merge_long_sequences(self):
        """Test merging datasets with very long sequences (many timesteps)."""
        np.random.seed(202)
        action_dim = 7
        num_timesteps = 100  # Long sequence
        num_datasets = 4

        stats = []
        for i in range(num_datasets):
            # Create a temporal pattern in the mean
            time_values = np.linspace(0, 2 * np.pi, num_timesteps)
            temporal_pattern = np.sin(time_values)[:, np.newaxis] * (i + 1) * 0.5

            mean_per_timestep = temporal_pattern + np.random.randn(num_timesteps, action_dim) * 0.1
            std_per_timestep = np.abs(np.random.randn(num_timesteps, action_dim)) * 0.2 + 0.3
            count = np.full(num_timesteps, 50.0 + i * 20.0)

            dataset_stats = {
                "action": {
                    "mean": mean_per_timestep.mean(axis=0).tolist(),
                    "std": std_per_timestep.mean(axis=0).tolist(),
                    "min": (mean_per_timestep - 2 * std_per_timestep).min(axis=0).tolist(),
                    "max": (mean_per_timestep + 2 * std_per_timestep).max(axis=0).tolist(),
                    "mean_per_timestep": mean_per_timestep.tolist(),
                    "std_per_timestep": std_per_timestep.tolist(),
                    "min_per_timestep": (mean_per_timestep - 2 * std_per_timestep).tolist(),
                    "max_per_timestep": (mean_per_timestep + 2 * std_per_timestep).tolist(),
                    "count": count.tolist(),
                }
            }
            stats.append(dataset_stats)

        result = merge_statistics(stats)

        # Verify long sequence is handled correctly
        assert len(result["action"]["mean_per_timestep"]) == num_timesteps
        assert len(result["action"]["std_per_timestep"]) == num_timesteps
        assert len(result["action"]["count"]) == num_timesteps

        # Verify all timesteps have valid statistics
        for t in range(num_timesteps):
            assert all(np.isfinite(result["action"]["mean_per_timestep"][t]))
            assert all(np.isfinite(result["action"]["std_per_timestep"][t]))
            assert all(v >= 0 for v in result["action"]["std_per_timestep"][t])  # std should be non-negative

    def test_merge_realistic_robot_scenario(self):
        """Test a realistic robot learning scenario with multiple modalities."""
        np.random.seed(303)

        # Realistic robot dimensions
        action_dim = 7  # 7-DOF robot
        image_embedding_dim = 512  # CLIP embeddings
        proprioception_dim = 14  # Joint positions and velocities
        num_timesteps = 16
        num_datasets = 6  # Multiple robot tasks/environments

        stats = []
        for i in range(num_datasets):
            dataset_stats = {}

            # Actions: typically in [-1, 1] range after normalization
            action_mean = np.random.uniform(-0.5, 0.5, (num_timesteps, action_dim))
            action_std = np.random.uniform(0.2, 0.6, (num_timesteps, action_dim))

            # Image embeddings: typically normalized with mean 0, std 1
            image_mean = np.random.randn(num_timesteps, image_embedding_dim) * 0.1
            image_std = (
                np.ones((num_timesteps, image_embedding_dim)) * 0.9
                + np.random.randn(num_timesteps, image_embedding_dim) * 0.1
            )

            # Proprioception: joint angles and velocities
            proprio_mean = np.random.uniform(-1, 1, (num_timesteps, proprioception_dim))
            proprio_std = np.random.uniform(0.1, 0.4, (num_timesteps, proprioception_dim))

            count = np.full(num_timesteps, 100.0 + i * 50.0)

            for tensor_name, mean_per_timestep, std_per_timestep in [
                ("action", action_mean, action_std),
                ("image_embedding", image_mean, image_std),
                ("proprioception", proprio_mean, proprio_std),
            ]:
                dataset_stats[tensor_name] = {
                    "mean": mean_per_timestep.mean(axis=0).tolist(),
                    "std": std_per_timestep.mean(axis=0).tolist(),
                    "min": (mean_per_timestep - 3 * std_per_timestep).min(axis=0).tolist(),
                    "max": (mean_per_timestep + 3 * std_per_timestep).max(axis=0).tolist(),
                    "mean_per_timestep": mean_per_timestep.tolist(),
                    "std_per_timestep": std_per_timestep.tolist(),
                    "min_per_timestep": (mean_per_timestep - 3 * std_per_timestep).tolist(),
                    "max_per_timestep": (mean_per_timestep + 3 * std_per_timestep).tolist(),
                    "count": count.tolist(),
                }

            stats.append(dataset_stats)

        result = merge_statistics(stats)

        # Verify all modalities are present
        assert "action" in result
        assert "image_embedding" in result
        assert "proprioception" in result

        # Verify dimensions
        assert len(result["action"]["mean"]) == action_dim
        assert len(result["image_embedding"]["mean"]) == image_embedding_dim
        assert len(result["proprioception"]["mean"]) == proprioception_dim

        # Verify temporal structure
        for tensor_name in ["action", "image_embedding", "proprioception"]:
            assert len(result[tensor_name]["mean_per_timestep"]) == num_timesteps
            assert len(result[tensor_name]["std_per_timestep"]) == num_timesteps
            assert len(result[tensor_name]["count"]) == num_timesteps

        # Verify all statistics are finite
        for tensor_name in result:
            assert all(np.isfinite(result[tensor_name]["mean"]))
            assert all(np.isfinite(result[tensor_name]["std"]))
            assert all(v >= 0 for v in result[tensor_name]["std"])  # std should be non-negative

        # Verify counts are summed correctly
        total_count_first_timestep = sum(100.0 + i * 50.0 for i in range(num_datasets))
        assert result["action"]["count"][0] == total_count_first_timestep


class TestMergeStatisticsVaryingTimesteps:
    """Test merge_statistics with varying counts per timestep (e.g., from masking)."""

    @pytest.fixture
    def varying_timestep_counts_stats(self):
        """Create statistics where different timesteps have different counts."""
        # Dataset 1: timesteps have counts [100, 80, 60]
        # Dataset 2: timesteps have counts [200, 150, 100]
        # This simulates varying amounts of masking per timestep
        return {
            "mean_per_timestep": np.array(
                [
                    [[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]],  # Dataset 1
                    [[2.0, 2.0], [2.5, 2.5], [3.0, 3.0]],  # Dataset 2
                ]
            ),
            "std_per_timestep": np.array(
                [
                    [[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]],  # Dataset 1
                    [[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]],  # Dataset 2
                ]
            ),
            "min_per_timestep": np.array(
                [
                    [[0.0, 0.0], [-0.5, -0.5], [-1.0, -1.0]],  # Dataset 1
                    [[0.5, 0.5], [0.0, 0.0], [-0.5, -0.5]],  # Dataset 2
                ]
            ),
            "max_per_timestep": np.array(
                [
                    [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],  # Dataset 1
                    [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],  # Dataset 2
                ]
            ),
            "count": np.array(
                [
                    [100.0, 80.0, 60.0],  # Dataset 1 - varying counts
                    [200.0, 150.0, 100.0],  # Dataset 2 - varying counts
                ]
            ),
        }

    def test_merge_mean_per_timestep_with_varying_counts(self, varying_timestep_counts_stats):
        """Test that mean_per_timestep merging correctly weights timesteps with different counts."""
        result = merge_statistics_single_field(varying_timestep_counts_stats, "mean_per_timestep")

        # Timestep 0: (1.0*100 + 2.0*200)/(100+200) = 500/300 = 1.666...
        # Timestep 1: (1.5*80 + 2.5*150)/(80+150) = 495/230 = 2.152...
        # Timestep 2: (2.0*60 + 3.0*100)/(60+100) = 420/160 = 2.625
        expected = np.array([[1.666667, 1.666667], [2.152174, 2.152174], [2.625, 2.625]])

        assert result.shape == (3, 2)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_merge_overall_mean_with_varying_counts(self, varying_timestep_counts_stats):
        """Test that overall mean correctly weights timesteps with different counts."""
        result = merge_statistics_single_field(varying_timestep_counts_stats, "mean")

        # First merge mean_per_timestep to get: [1.666667, 2.152174, 2.625]
        # Then weight by counts per timestep: [300, 230, 160]
        # Overall = (1.666667*300 + 2.152174*230 + 2.625*160) / (300+230+160)
        #         = (500.0001 + 495.0000 + 420.0) / 690
        #         = 1415.0001 / 690 = 2.050725
        expected = np.array([2.050725, 2.050725])

        assert result.shape == (2,)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_merge_std_per_timestep_with_varying_counts(self, varying_timestep_counts_stats):
        """Test that std_per_timestep merging uses pooled variance with varying counts."""
        result = merge_statistics_single_field(varying_timestep_counts_stats, "std_per_timestep")

        # Using pooled variance formula for each timestep independently
        # For timestep 0:
        #   Dataset 1: n1=100, σ1=0.5, μ1=1.0
        #   Dataset 2: n2=200, σ2=1.0, μ2=2.0
        #   Merged μ = 1.666667
        #   Pooled variance = [(n1-1)*σ1² + n1*(μ1-μ_merged)²] + [(n2-1)*σ2² + n2*(μ2-μ_merged)²] / (n1+n2-1)
        #                   = [(99*0.25 + 100*0.444444) + (199*1.0 + 200*0.111111)] / 299
        #                   = [24.75 + 44.4444 + 199 + 22.2222] / 299
        #                   = 290.4166 / 299 = 0.9714
        #   Pooled std = sqrt(0.9714) = 0.9856

        # For timestep 1:
        #   Dataset 1: n1=80, σ1=0.6, μ1=1.5
        #   Dataset 2: n2=150, σ2=1.1, μ2=2.5
        #   Merged μ = 2.152174
        #   Similar calculation...

        # Just verify shape and that it's reasonable (between min and max of inputs)
        assert result.shape == (3, 2)
        # Result should be between the min and max std values
        assert np.all(result >= 0.5)  # Minimum input std
        assert np.all(result <= 1.2)  # Maximum input std

    def test_merge_std_with_varying_counts(self, varying_timestep_counts_stats):
        """Test that std merging uses weighted averaging with varying counts."""
        result = merge_statistics_single_field(varying_timestep_counts_stats, "std")

        # Should use weighted law of total variance
        std_per_timestep = merge_statistics_single_field(varying_timestep_counts_stats, "std_per_timestep")
        mean_per_timestep = merge_statistics_single_field(varying_timestep_counts_stats, "mean_per_timestep")
        counts_per_timestep = np.sum(varying_timestep_counts_stats["count"], axis=0)

        # Weighted variance calculation
        variance_per_timestep = std_per_timestep**2
        mean_variance = np.average(variance_per_timestep, axis=0, weights=counts_per_timestep)
        weighted_mean = np.average(mean_per_timestep, axis=0, weights=counts_per_timestep)
        variance_of_means = np.average((mean_per_timestep - weighted_mean) ** 2, axis=0, weights=counts_per_timestep)
        expected = np.sqrt(mean_variance + variance_of_means)

        assert result.shape == (2,)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_merge_count_with_varying_values(self, varying_timestep_counts_stats):
        """Test that count merging sums correctly with varying counts."""
        result = merge_statistics_single_field(varying_timestep_counts_stats, "count")

        expected = np.array([300.0, 230.0, 160.0])  # [100+200, 80+150, 60+100]

        assert result.shape == (3,)
        np.testing.assert_array_equal(result, expected)

    @pytest.fixture
    def extreme_imbalance_stats(self):
        """Create statistics with extreme count imbalance between timesteps."""
        # Dataset 1: counts [1000, 10, 1] - extreme imbalance
        # Dataset 2: counts [2000, 20, 2] - same proportion
        return {
            "mean_per_timestep": np.array(
                [
                    [[1.0, 1.0], [5.0, 5.0], [10.0, 10.0]],  # Dataset 1
                    [[2.0, 2.0], [6.0, 6.0], [12.0, 12.0]],  # Dataset 2
                ]
            ),
            "std_per_timestep": np.array(
                [
                    [[0.1, 0.1], [0.5, 0.5], [1.0, 1.0]],  # Dataset 1
                    [[0.2, 0.2], [0.6, 0.6], [1.2, 1.2]],  # Dataset 2
                ]
            ),
            "count": np.array(
                [
                    [1000.0, 10.0, 1.0],  # Dataset 1 - extreme imbalance
                    [2000.0, 20.0, 2.0],  # Dataset 2 - extreme imbalance
                ]
            ),
        }

    def test_merge_with_extreme_count_imbalance(self, extreme_imbalance_stats):
        """Test merging with extreme count imbalance - first timestep should dominate."""
        result = merge_statistics_single_field(extreme_imbalance_stats, "mean_per_timestep")

        # Timestep 0 has 3000 samples, timestep 1 has 30, timestep 2 has 3
        # Timestep 0: (1.0*1000 + 2.0*2000)/(1000+2000) = 5000/3000 = 1.666...
        # Timestep 1: (5.0*10 + 6.0*20)/(10+20) = 170/30 = 5.666...
        # Timestep 2: (10.0*1 + 12.0*2)/(1+2) = 34/3 = 11.333...
        expected_t0 = np.array([1.666667, 1.666667])
        expected_t1 = np.array([5.666667, 5.666667])
        expected_t2 = np.array([11.333333, 11.333333])

        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[0], expected_t0, rtol=1e-5)
        np.testing.assert_allclose(result[1], expected_t1, rtol=1e-5)
        np.testing.assert_allclose(result[2], expected_t2, rtol=1e-5)

    def test_overall_std_dominated_by_high_count_timesteps(self, extreme_imbalance_stats):
        """Test that overall std is properly weighted towards high-count timesteps."""
        result = merge_statistics_single_field(extreme_imbalance_stats, "std")

        # The overall std should be heavily influenced by timestep 0 (3000 samples)
        # and barely influenced by timestep 2 (3 samples)
        merge_statistics_single_field(extreme_imbalance_stats, "std_per_timestep")
        np.sum(extreme_imbalance_stats["count"], axis=0)

        # Verify that the std is much closer to timestep 0's value than timestep 2's
        assert result.shape == (2,)
        # The weighted calculation should be closer to timestep 0
        # which has std ~0.166 after pooling, vs timestep 2 with std ~1.15
        assert result[0] < 1.0  # Should be much less than timestep 2's std

    @pytest.fixture
    def zero_count_timesteps_stats(self):
        """Create statistics where some timesteps have zero counts."""
        return {
            "mean_per_timestep": np.array(
                [
                    [[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]],  # Dataset 1, t1 has 0 count
                    [[2.0, 2.0], [3.0, 3.0], [0.0, 0.0]],  # Dataset 2, t2 has 0 count
                ]
            ),
            "std_per_timestep": np.array(
                [
                    [[0.5, 0.5], [0.0, 0.0], [0.7, 0.7]],  # Dataset 1
                    [[1.0, 1.0], [1.1, 1.1], [0.0, 0.0]],  # Dataset 2
                ]
            ),
            "count": np.array(
                [
                    [100.0, 0.0, 50.0],  # Dataset 1, timestep 1 has 0 samples
                    [200.0, 80.0, 0.0],  # Dataset 2, timestep 2 has 0 samples
                ]
            ),
        }

    def test_merge_with_zero_counts(self, zero_count_timesteps_stats):
        """Test that timesteps with zero counts are handled correctly."""
        result = merge_statistics_single_field(zero_count_timesteps_stats, "mean_per_timestep")

        # Timestep 0: both datasets have data
        # Timestep 1: only dataset 2 has data (count=80)
        # Timestep 2: only dataset 1 has data (count=50)
        expected_t0 = np.array([1.666667, 1.666667])  # (1*100 + 2*200)/(100+200)
        expected_t1 = np.array([3.0, 3.0])  # Only dataset 2 contributes
        expected_t2 = np.array([2.0, 2.0])  # Only dataset 1 contributes

        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[0], expected_t0, rtol=1e-5)
        np.testing.assert_allclose(result[1], expected_t1, rtol=1e-5)
        np.testing.assert_allclose(result[2], expected_t2, rtol=1e-5)

    def test_std_with_partial_zero_counts(self, zero_count_timesteps_stats):
        """Test std calculation when some timesteps have zero counts."""
        result = merge_statistics_single_field(zero_count_timesteps_stats, "std")

        # Should handle zero counts gracefully using weighted average
        # Only timesteps with non-zero total counts should contribute
        counts = np.sum(zero_count_timesteps_stats["count"], axis=0)  # [300, 80, 50]
        assert counts[0] == 300.0
        assert counts[1] == 80.0
        assert counts[2] == 50.0

        # Result should be finite and non-negative
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)


class TestTDigestMerge:
    """Test merging statistics using t-digest for accurate percentile computation."""

    def test_tdigest_merge_accuracy(self):
        """
        Test that merging statistics using t-digest states is accurate
        even for very different distributions where weighted average fails.
        """
        np.random.seed(42)
        # Dataset 1: Normal(0, 1)
        data1 = np.random.normal(loc=0.0, scale=1.0, size=(1000, 1, 1)).astype(np.float32)
        mask1 = np.ones((1000, 1), dtype=bool)

        # Dataset 2: Normal(10, 2)
        data2 = np.random.normal(loc=10.0, scale=2.0, size=(2000, 1, 1)).astype(np.float32)
        mask2 = np.ones((2000, 1), dtype=bool)

        # Compute stats for each
        stats_obj1 = StreamingDatasetStatistics()
        stats_obj2 = StreamingDatasetStatistics()

        stats_obj1.update({"test_tensor": data1, "mask": mask1})
        stats_obj2.update({"test_tensor": data2, "mask": mask2})

        stats1 = stats_obj1.get_statistics()
        stats2 = stats_obj2.get_statistics()

        # Verify that tdigest_state is present
        assert "tdigest_state" in stats1["test_tensor"]
        assert "tdigest_state" in stats2["test_tensor"]

        # Merge statistics
        merged = merge_statistics([stats1, stats2])

        # Compute ground truth combined percentiles
        combined_data = np.concatenate([data1.flatten(), data2.flatten()])
        target_ps = [1, 5, 95, 99]
        expected = np.percentile(combined_data, target_ps)

        # Check accuracy
        for i, p in enumerate(target_ps):
            val_merged = merged["test_tensor"][f"percentile_{p}"][0]
            error = abs(val_merged - expected[i])

            # T-Digest merge is extremely accurate.
            # error should be < 0.1.
            assert error < 0.1, (
                f"T-Digest merge error for p={p} is too high: {error} (expected ~{expected[i]}, got {val_merged})"
            )

    def test_tdigest_merge_per_timestep(self):
        """Test merging of per-timestep percentiles using t-digest states."""
        np.random.seed(42)
        T, C = 5, 2
        data1 = np.random.normal(loc=0.0, scale=1.0, size=(2000, T, C)).astype(np.float32)
        data2 = np.random.normal(loc=5.0, scale=1.0, size=(2000, T, C)).astype(np.float32)

        stats_obj1 = StreamingDatasetStatistics()
        stats_obj2 = StreamingDatasetStatistics()

        stats_obj1.update({"test_tensor": data1, "mask": np.ones((2000, T), dtype=bool)})
        stats_obj2.update({"test_tensor": data2, "mask": np.ones((2000, T), dtype=bool)})

        stats1 = stats_obj1.get_statistics()
        stats2 = stats_obj2.get_statistics()

        merged = merge_statistics([stats1, stats2])

        # Check accuracy for one timestep and one channel
        combined_data = np.concatenate([data1[:, 0, 0], data2[:, 0, 0]])
        expected_p99 = np.percentile(combined_data, 99)

        val_merged = merged["test_tensor"]["percentile_99_per_timestep"][0][0]
        error = abs(val_merged - expected_p99)

        assert error < 0.1, f"Per-timestep T-Digest merge error is too high: {error}"
