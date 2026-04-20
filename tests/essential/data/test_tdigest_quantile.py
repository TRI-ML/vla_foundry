import os
import tempfile

import numpy as np

from vla_foundry.data.preprocessing.robotics.preprocess_statistics import StreamingDatasetStatistics, TDigestEstimator


def test_tdigest_accuracy():
    """Test the accuracy of the TDigestEstimator against numpy.percentile."""
    p = 0.05
    n_samples = 1000
    shape = (2,)
    est = TDigestEstimator(shape)

    # Generate some data
    data = np.random.normal(0, 1, size=(n_samples, *shape))
    for i in range(n_samples):
        est.update(data[i])

    # Exact quantile
    exact = np.percentile(data, p * 100, axis=0)
    td_est = est.get_quantile(p)

    # Check error - T-Digest is very accurate
    error = np.abs(exact - td_est)
    assert np.all(error < 0.05), f"T-Digest error too high: {error} for exact {exact}"


def test_streaming_stats_tdigest():
    """Test that StreamingDatasetStatistics correctly uses TDigest estimators for multiple quantiles."""
    np.random.seed(42)
    stats = StreamingDatasetStatistics()

    n_samples = 2000
    shape = (1, 3)  # T=1, C=3

    data_list = []
    for _ in range(n_samples):
        val = np.random.uniform(0, 100, size=(1, shape[0], shape[1]))  # B=1, T=1, C=3
        mask = np.ones((1, shape[0]), dtype=bool)
        sample = {"test_key": val[0], "mask": mask[0]}
        stats.update(sample)
        data_list.append(val[0])

    res = stats.get_statistics()
    data_all = np.concatenate(data_list, axis=0)

    # Verify the specific percentiles we track
    for p_val in [0.01, 0.05, 0.95, 0.99]:
        exact = np.percentile(data_all, p_val * 100, axis=0)
        p_key = f"percentile_{int(p_val * 100)}_per_timestep"
        # Recorded is (T, C), we have T=1. exact is (C,)
        recorded = np.array(res["test_key"][p_key])[0]

        # T-Digest estimates are very accurate
        # For range 0-100, error < 1.0 is reasonable
        error = np.abs(exact - recorded)
        assert np.all(error < 1.0), f"Error for p={p_val} too high: {error}. Exact: {exact}, Recorded: {recorded}"


def test_serialization():
    """Test that the state of TDigest estimators is correctly preserved through serialization."""

    stats = StreamingDatasetStatistics()
    val = np.random.uniform(0, 100, size=(10, 1, 3))
    mask = np.ones((10, 1), dtype=bool)
    stats.update({"test_key": val, "mask": mask})

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        stats.save_state(tmp_path)

        stats2 = StreamingDatasetStatistics()
        stats2.load_state(tmp_path)

        # Compare counts
        assert np.all(stats.counts["test_key"] == stats2.counts["test_key"])

        # Compare T-Digest estimates for one of the quantiles
        for p in [0.01, 0.05, 0.95, 0.99]:
            est1 = stats.quantile_estimators["test_key"].get_quantile(p)
            est2 = stats2.quantile_estimators["test_key"].get_quantile(p)
            assert np.allclose(est1, est2), f"Serialization mismatch for p={p}: {est1} != {est2}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
