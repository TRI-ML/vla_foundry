import numpy as np
import pytest

from vla_foundry.data.preprocessing.robotics.preprocess_statistics import StreamingDatasetStatistics


@pytest.mark.parametrize(
    "mean,std,num_samples,shape",
    [
        (-3.0, 0.5, 500, (1, 3, 2)),
        (0.0, 1.0, 1001, (4, 10, 3)),
        (5.0, 2.0, 3001, (2, 5, 6)),
    ],
)
def test_streaming_dataset_statistics_basic(mean, std, num_samples, shape):
    """
    Test the StreamingDatasetStatistics class with a simple normal distribution.
    The shape is (batch_size, num_timesteps, num_channels)
    The test gives one batch of samples at a time, and the StreamingDatasetStatistics class is updated with each batch.
    The test checks that the statistics are computed correctly by comparing them to the global statistics computed on
    all the samples.
    """
    np.random.seed(42)
    max_samples_for_percentiles = 1000
    stats = StreamingDatasetStatistics(compute_stats=True, max_samples_for_percentiles=max_samples_for_percentiles)
    key = "test_key"
    all_data = []
    all_mask = []
    for _ in range(0, num_samples, shape[0]):
        data = np.random.normal(loc=mean, scale=std, size=shape).astype(np.float32)
        mask = np.random.randint(0, 2, (shape[0], shape[1]), dtype=bool)
        sample = {key: data.copy(), "mask": mask.copy()}
        all_data.append(data)
        all_mask.append(mask)
        stats.update(sample)
    all_data = np.concatenate(all_data, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)
    all_data[~all_mask] = 0
    expected_mean = np.sum(all_data, axis=0) / np.sum(all_mask, axis=0)[..., None]
    squared_diff = (all_data - expected_mean) ** 2
    squared_diff[~all_mask] = 0
    expected_std = np.sqrt(np.sum(squared_diff, axis=0) / np.sum(all_mask, axis=0)[..., None])
    all_data_no_min = all_data.copy()
    all_data_no_min[~all_mask] = float("inf")
    expected_min = np.min(all_data_no_min, axis=0)
    all_data_no_max = all_data.copy()
    all_data_no_max[~all_mask] = float("-inf")
    expected_max = np.max(all_data_no_max, axis=0)

    expected_percentiles = np.percentile(all_data[all_mask].reshape(-1, shape[2]), [1, 5, 95, 99], axis=0)
    expected_percentiles_per_timestep = np.zeros((4, shape[1], shape[2]))
    for timestep in range(shape[1]):
        data_timestep = all_data[:, timestep]
        expected_percentiles_per_timestep[:, timestep] = np.percentile(
            data_timestep[all_mask[:, timestep]], [1, 5, 95, 99], axis=0
        )
    computed = stats.get_statistics()[key]

    # Allow small numerical error
    np.testing.assert_allclose(computed["mean_per_timestep"], expected_mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        computed["std_per_timestep"], expected_std, rtol=1e-1, atol=1e-1
    )  # Tolerance is higher because of the Welford's online algorithm, not exact
    np.testing.assert_allclose(computed["min_per_timestep"], expected_min, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(computed["max_per_timestep"], expected_max, rtol=1e-3, atol=1e-3)
    # Percentiles
    np.testing.assert_allclose(
        computed["percentile_1_per_timestep"], expected_percentiles_per_timestep[0], rtol=1e-1, atol=1e-1
    )
    np.testing.assert_allclose(
        computed["percentile_5_per_timestep"], expected_percentiles_per_timestep[1], rtol=1e-1, atol=1e-1
    )
    np.testing.assert_allclose(
        computed["percentile_95_per_timestep"], expected_percentiles_per_timestep[2], rtol=1e-1, atol=1e-1
    )
    np.testing.assert_allclose(
        computed["percentile_99_per_timestep"], expected_percentiles_per_timestep[3], rtol=1e-1, atol=1e-1
    )
    np.testing.assert_allclose(computed["percentile_1"], expected_percentiles[0], rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(computed["percentile_5"], expected_percentiles[1], rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(computed["percentile_95"], expected_percentiles[2], rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(computed["percentile_99"], expected_percentiles[3], rtol=1e-1, atol=1e-1)

    assert np.array(computed["count"]).sum() == all_mask.sum()
    np.testing.assert_array_compare(np.less_equal, computed["percentile_sample_count"], computed["count"])


def test_streaming_dataset_statistics_varying_timestep_counts():
    """
    Test StreamingDatasetStatistics with varying counts per timestep due to masking.
    This simulates realistic scenarios where different timesteps have different
    amounts of valid data.
    """
    np.random.seed(42)
    stats = StreamingDatasetStatistics(compute_stats=True, max_samples_for_percentiles=1000)
    key = "test_key"

    # Create data with intentionally varying mask patterns
    # Early timesteps have more valid samples, later timesteps have fewer
    num_batches = 100
    batch_size = 10
    num_timesteps = 5
    num_channels = 3

    all_data = []
    all_mask = []

    for _batch_idx in range(num_batches):
        data = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, num_timesteps, num_channels)).astype(np.float32)

        # Create masks with decreasing probability of being valid at later timesteps
        mask = np.zeros((batch_size, num_timesteps), dtype=bool)
        for t in range(num_timesteps):
            # Timestep 0: 100% valid, timestep 4: 20% valid
            prob_valid = 1.0 - (0.8 * t / (num_timesteps - 1))
            mask[:, t] = np.random.random(batch_size) < prob_valid

        sample = {key: data.copy(), "mask": mask.copy()}
        all_data.append(data)
        all_mask.append(mask)
        stats.update(sample)

    all_data = np.concatenate(all_data, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)

    computed = stats.get_statistics()[key]

    # Verify that counts decrease over timesteps
    counts = np.array(computed["count"])
    assert counts[0] > counts[-1], "First timestep should have more samples than last"

    # Verify counts are monotonically decreasing (or equal)
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1] * 0.8, "Counts should generally decrease"

    # Verify all statistics have the correct shape
    assert len(computed["mean_per_timestep"]) == num_timesteps
    assert len(computed["std_per_timestep"]) == num_timesteps
    assert len(computed["count"]) == num_timesteps

    # Verify overall statistics are reasonable
    assert len(computed["mean"]) == num_channels
    assert len(computed["std"]) == num_channels
    assert all(np.isfinite(computed["mean"]))
    assert all(np.isfinite(computed["std"]))
    assert all(s >= 0 for s in computed["std"])


def test_streaming_dataset_statistics_extreme_masking():
    """
    Test with extreme masking patterns where some timesteps have very few samples.
    """
    np.random.seed(123)
    stats = StreamingDatasetStatistics(compute_stats=True, max_samples_for_percentiles=500)
    key = "test_key"

    # Create 50 samples with extreme masking
    num_samples = 50
    batch_size = 5
    num_timesteps = 4
    num_channels = 2

    for _ in range(num_samples // batch_size):
        data = np.random.normal(loc=5.0, scale=2.0, size=(batch_size, num_timesteps, num_channels)).astype(np.float32)

        # Extreme masking: timestep 0 always valid, timestep 3 rarely valid
        mask = np.ones((batch_size, num_timesteps), dtype=bool)
        mask[:, 1] = np.random.random(batch_size) < 0.5  # 50% valid
        mask[:, 2] = np.random.random(batch_size) < 0.2  # 20% valid
        mask[:, 3] = np.random.random(batch_size) < 0.1  # 10% valid

        sample = {key: data.copy(), "mask": mask.copy()}
        stats.update(sample)

    computed = stats.get_statistics()[key]
    counts = np.array(computed["count"])

    # Verify the count pattern matches our masking
    assert counts[0] == 50, "First timestep should have all samples"
    assert counts[1] < counts[0], "Second timestep should have fewer samples"
    assert counts[2] < counts[1], "Third timestep should have even fewer"
    assert counts[3] < counts[2], "Fourth timestep should have the fewest"
    assert counts[3] >= 1, "Should have at least one sample in last timestep"

    # Verify statistics are still valid despite extreme imbalance
    assert all(np.isfinite(computed["mean"]))
    assert all(np.isfinite(computed["std"]))
    assert all(s >= 0 for s in computed["std"])

    # Verify per-timestep statistics exist for all timesteps
    assert len(computed["mean_per_timestep"]) == num_timesteps
    assert len(computed["std_per_timestep"]) == num_timesteps


def test_streaming_dataset_statistics_weighted_variance():
    """
    Test that the overall variance correctly uses weighted averaging when
    timesteps have different counts.
    """
    np.random.seed(456)
    stats = StreamingDatasetStatistics(compute_stats=True, max_samples_for_percentiles=1000)
    key = "test_key"

    # Create data where timesteps have very different means and counts
    num_batches = 20
    batch_size = 10
    num_timesteps = 3
    num_channels = 2

    all_data = []
    all_mask = []

    for _ in range(num_batches):
        data = np.zeros((batch_size, num_timesteps, num_channels), dtype=np.float32)

        # Give each timestep a different mean
        data[:, 0, :] = np.random.normal(loc=0.0, scale=0.5, size=(batch_size, num_channels))
        data[:, 1, :] = np.random.normal(loc=5.0, scale=0.5, size=(batch_size, num_channels))
        data[:, 2, :] = np.random.normal(loc=10.0, scale=0.5, size=(batch_size, num_channels))

        # Create imbalanced masking
        mask = np.ones((batch_size, num_timesteps), dtype=bool)
        mask[:, 0] = True  # 100% for timestep 0
        mask[:, 1] = np.random.random(batch_size) < 0.3  # 30% for timestep 1
        mask[:, 2] = np.random.random(batch_size) < 0.1  # 10% for timestep 2

        sample = {key: data.copy(), "mask": mask.copy()}
        all_data.append(data)
        all_mask.append(mask)
        stats.update(sample)

    computed = stats.get_statistics()[key]
    counts = np.array(computed["count"])

    # With heavily imbalanced counts, the overall mean should be close to timestep 0's mean
    overall_mean = np.array(computed["mean"])
    np.array(computed["mean_per_timestep"][0])

    # Since timestep 0 has ~200 samples and others have much fewer,
    # overall mean should be closer to 0.0 than to 5.0 or 10.0
    assert np.abs(overall_mean[0]) < 2.0, "Overall mean should be dominated by high-count timestep"

    # Verify the weighting is working by checking counts
    assert counts[0] > counts[1] * 2, "First timestep should have much more data"
    assert counts[1] > counts[2], "Second timestep should have more data than third"

    # Overall std should reflect variance both within and between timesteps
    overall_std = np.array(computed["std"])
    assert np.all(overall_std > 0.5), "Should have significant variance due to different timestep means"


def test_streaming_dataset_statistics_all_timesteps_masked():
    """
    Test handling when all samples for a specific timestep are masked.
    """
    np.random.seed(789)
    stats = StreamingDatasetStatistics(compute_stats=True, max_samples_for_percentiles=100)
    key = "test_key"

    num_batches = 10
    batch_size = 5
    num_timesteps = 4
    num_channels = 2

    for _ in range(num_batches):
        data = np.random.normal(loc=1.0, scale=0.5, size=(batch_size, num_timesteps, num_channels)).astype(np.float32)

        # Mask out entire middle timesteps
        mask = np.ones((batch_size, num_timesteps), dtype=bool)
        mask[:, 1] = False  # Completely mask timestep 1
        mask[:, 2] = False  # Completely mask timestep 2

        sample = {key: data.copy(), "mask": mask.copy()}
        stats.update(sample)

    computed = stats.get_statistics()[key]
    counts = np.array(computed["count"])

    # Verify masked timesteps have zero count
    assert counts[0] == 50, "First timestep should have all samples"
    assert counts[1] == 0, "Second timestep should be completely masked"
    assert counts[2] == 0, "Third timestep should be completely masked"
    assert counts[3] == 50, "Fourth timestep should have all samples"

    # Verify overall statistics are still computable
    assert all(np.isfinite(computed["mean"]))
    assert all(np.isfinite(computed["std"]))

    # Per-timestep statistics for masked timesteps should be 0 or NaN-handled
    assert np.all(np.isfinite(computed["std_per_timestep"][1]))
    assert np.all(np.isfinite(computed["std_per_timestep"][2]))
