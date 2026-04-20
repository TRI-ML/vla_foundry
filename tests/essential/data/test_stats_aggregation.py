"""Tests for the worker-side aggregation + actor-side merge refactor.

Verifies that compute_batch_aggregates -> merge_from_aggregates produces
identical results to the legacy merge_from_samples and direct update paths.
"""

import numpy as np
import pytest

from vla_foundry.data.preprocessing.robotics.preprocess_statistics import (
    StreamingDatasetStatistics,
    TDigestEstimator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(num_samples, T=5, C=3, seed=42):
    """Generate a list of raw stat samples matching the format produced by converters."""
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(num_samples):
        mask = rng.randint(0, 2, T).astype(bool)
        # Ensure at least one True so stats are non-degenerate
        mask[0] = True
        sample = {
            "lowdim": {
                "action": rng.randn(T, C).astype(np.float32),
                "state": rng.randn(T, C).astype(np.float32),
            },
            "past_mask": mask,
            "future_mask": mask,
        }
        samples.append(sample)
    return samples


def _compare_stats(stats_a, stats_b, atol=1e-10):
    """Assert two get_statistics() outputs are numerically identical."""
    assert set(stats_a.keys()) == set(stats_b.keys()), f"Key mismatch: {set(stats_a.keys())} vs {set(stats_b.keys())}"
    for key in stats_a:
        for metric in [
            "mean",
            "std",
            "min",
            "max",
            "mean_per_timestep",
            "std_per_timestep",
            "min_per_timestep",
            "max_per_timestep",
            "count",
        ]:
            a = np.array(stats_a[key][metric])
            b = np.array(stats_b[key][metric])
            np.testing.assert_allclose(a, b, atol=atol, err_msg=f"{key}.{metric} mismatch")


# ---------------------------------------------------------------------------
# TDigestEstimator.merge_from_state
# ---------------------------------------------------------------------------


class TestTDigestMergeFromState:
    def test_merge_two_estimators(self):
        """Merging state from estimator B into estimator A should give the same
        result as feeding all data into a single estimator."""
        rng = np.random.RandomState(0)
        shape = (3, 2)
        data_a = rng.randn(50, *shape).astype(np.float32)
        data_b = rng.randn(50, *shape).astype(np.float32)

        # Single estimator with all data
        est_all = TDigestEstimator(shape)
        for row in np.concatenate([data_a, data_b]):
            est_all.update(row)

        # Two estimators merged via state
        est_a = TDigestEstimator(shape)
        for row in data_a:
            est_a.update(row)
        est_b = TDigestEstimator(shape)
        for row in data_b:
            est_b.update(row)

        est_a.merge_from_state(est_b.get_state())

        for p in [0.01, 0.5, 0.99]:
            q_all = est_all.get_quantile(p)
            q_merged = est_a.get_quantile(p)
            np.testing.assert_allclose(q_merged, q_all, atol=0.3, err_msg=f"Quantile {p} mismatch after merge")

    def test_merge_into_empty(self):
        """Merging into a fresh estimator should reproduce the source."""
        rng = np.random.RandomState(1)
        shape = (4,)
        est_src = TDigestEstimator(shape, max_buffer=10)
        data = rng.randn(30, *shape).astype(np.float32)
        for row in data:
            est_src.update(row)

        est_dst = TDigestEstimator(shape, max_buffer=10)
        est_dst.merge_from_state(est_src.get_state())

        for p in [0.05, 0.5, 0.95]:
            np.testing.assert_allclose(est_dst.get_quantile(p), est_src.get_quantile(p), atol=0.2)

    def test_merge_buffer_only(self):
        """Merge when source has only buffered data (not yet promoted to digest)."""
        shape = (2,)
        est_src = TDigestEstimator(shape, max_buffer=9999)  # huge buffer -> stays in buffer
        est_src.update(np.array([1.0, 2.0], dtype=np.float32))
        est_src.update(np.array([3.0, 4.0], dtype=np.float32))

        est_dst = TDigestEstimator(shape, max_buffer=9999)
        est_dst.merge_from_state(est_src.get_state())

        np.testing.assert_allclose(est_dst.get_quantile(0.5), [2.0, 3.0], atol=0.1)


# ---------------------------------------------------------------------------
# compute_batch_aggregates + merge_from_aggregates vs legacy
# ---------------------------------------------------------------------------


class TestAggregatePathMatchesLegacy:
    @pytest.mark.parametrize("num_samples", [1, 5, 50])
    def test_single_batch(self, num_samples):
        """Single batch: aggregate path should match legacy merge_from_samples."""
        samples = _make_samples(num_samples, seed=42)

        stats_legacy = StreamingDatasetStatistics(compute_stats=True)
        stats_legacy.merge_from_samples(samples)

        stats_new = StreamingDatasetStatistics(compute_stats=True)
        agg = StreamingDatasetStatistics.compute_batch_aggregates(samples)
        stats_new.merge_from_aggregates(agg)

        _compare_stats(stats_legacy.get_statistics(), stats_new.get_statistics())

    def test_multiple_batches(self):
        """Incremental aggregation over several batches should match one big merge."""
        all_samples = _make_samples(60, seed=99)

        stats_one_shot = StreamingDatasetStatistics(compute_stats=True)
        stats_one_shot.merge_from_samples(all_samples)

        stats_incremental = StreamingDatasetStatistics(compute_stats=True)
        for i in range(0, 60, 10):
            batch = all_samples[i : i + 10]
            agg = StreamingDatasetStatistics.compute_batch_aggregates(batch)
            stats_incremental.merge_from_aggregates(agg)

        _compare_stats(
            stats_one_shot.get_statistics(),
            stats_incremental.get_statistics(),
            atol=1e-6,  # tiny float drift from splitting batches
        )


# ---------------------------------------------------------------------------
# Aggregate path vs direct update()
# ---------------------------------------------------------------------------


class TestAggregatePathMatchesDirectUpdate:
    @pytest.mark.parametrize(
        "mean,std,num_samples,shape",
        [
            (-3.0, 0.5, 100, (1, 3, 2)),
            (0.0, 1.0, 200, (4, 10, 3)),
            (5.0, 2.0, 300, (2, 5, 6)),
        ],
    )
    def test_matches_direct_update(self, mean, std, num_samples, shape):
        """Feeding raw batches via update() and via compute_batch_aggregates+merge
        should give close results. Different batch sizes cause Welford accumulation
        drift for std, so we check mean/min/max tightly and std loosely."""
        rng = np.random.RandomState(42)
        key = "test_key"

        stats_direct = StreamingDatasetStatistics(compute_stats=True)
        all_samples = []

        for _ in range(0, num_samples, shape[0]):
            data = rng.normal(loc=mean, scale=std, size=shape).astype(np.float32)
            mask = rng.randint(0, 2, (shape[0], shape[1]), dtype=bool)
            mask[:, 0] = True  # ensure at least one valid
            sample_lowdim = {key: data.copy(), "mask": mask.copy()}
            stats_direct.update(sample_lowdim)

            # Build raw samples for the aggregate path
            for b in range(shape[0]):
                all_samples.append(
                    {
                        "lowdim": {key: data[b].copy()},
                        "past_mask": mask[b],
                        "future_mask": mask[b],
                    }
                )

        stats_agg = StreamingDatasetStatistics(compute_stats=True)
        for i in range(0, len(all_samples), 20):
            batch = all_samples[i : i + 20]
            agg = StreamingDatasetStatistics.compute_batch_aggregates(batch)
            stats_agg.merge_from_aggregates(agg)

        direct = stats_direct.get_statistics()[key]
        aggregated = stats_agg.get_statistics()[key]

        # mean, min, max, count should be tight — only float precision drift
        for metric in ["mean", "mean_per_timestep", "min", "max", "min_per_timestep", "max_per_timestep", "count"]:
            a = np.array(direct[metric])
            b = np.array(aggregated[metric])
            np.testing.assert_allclose(a, b, atol=1e-4, err_msg=f"{metric} mismatch")

        # std drifts more due to different batch boundaries in Welford's algorithm
        for metric in ["std", "std_per_timestep"]:
            a = np.array(direct[metric])
            b = np.array(aggregated[metric])
            np.testing.assert_allclose(a, b, rtol=0.1, err_msg=f"{metric} mismatch")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAggregateEdgeCases:
    def test_single_sample(self):
        """A batch with a single sample should work."""
        samples = _make_samples(1, seed=0)
        stats = StreamingDatasetStatistics(compute_stats=True)
        agg = StreamingDatasetStatistics.compute_batch_aggregates(samples)
        stats.merge_from_aggregates(agg)
        result = stats.get_statistics()
        assert "action" in result
        assert "state" in result

    def test_all_masked(self):
        """Samples where most timesteps are masked should not crash."""
        sample = {
            "lowdim": {
                "action": np.zeros((5, 3), dtype=np.float32),
            },
            "past_mask": np.array([True, False, False, False, False]),
            "future_mask": np.array([True, False, False, False, False]),
        }
        stats = StreamingDatasetStatistics(compute_stats=True)
        agg = StreamingDatasetStatistics.compute_batch_aggregates([sample])
        stats.merge_from_aggregates(agg)
        result = stats.get_statistics()
        counts = np.array(result["action"]["count"])
        assert counts[0] == 1
        assert counts[1] == 0

    def test_compute_stats_disabled(self):
        """When compute_stats=False, merge_from_aggregates is a no-op."""
        samples = _make_samples(5)
        stats = StreamingDatasetStatistics(compute_stats=False)
        agg = StreamingDatasetStatistics.compute_batch_aggregates(samples)
        stats.merge_from_aggregates(agg)
        assert stats.get_statistics() == {"statistics_disabled": True}
