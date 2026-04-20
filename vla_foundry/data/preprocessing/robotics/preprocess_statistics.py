import json
from typing import Any

import numpy as np
import ray
from tdigest_rs import TDigest


@ray.remote
class LoggerActor:
    def __init__(self):
        self.total_potential_samples = 0
        self.still_samples_filtered = 0
        self.padding_samples_filtered = 0

    def get_values(self):
        return {
            "total_potential_samples": self.total_potential_samples,
            "still_samples_filtered": self.still_samples_filtered,
            "padding_samples_filtered": self.padding_samples_filtered,
        }

    def increment_total_potential_samples(self):
        self.total_potential_samples += 1

    def increment_still_samples_filtered(self):
        self.still_samples_filtered += 1

    def increment_padding_samples_filtered(self):
        self.padding_samples_filtered += 1


class TDigestEstimator:
    """
    T-Digest algorithm for streaming quantile estimation with accurate tail percentiles.
    Uses tdigest-rs (Rust-based) for performance and accuracy.

    Reference: Dunning, Ted, and Otmar Ertl. "Computing extremely accurate quantiles using t-digests."
    arXiv preprint arXiv:1902.04023 (2019).
    """

    def __init__(self, shape: tuple, max_buffer: int = 1000, compression: int = 100):
        self.shape = shape
        self.max_buffer = max_buffer  # Collect samples before creating t-digest
        self.compression = compression
        self.counts = np.zeros(self.shape, dtype=int)
        self.buffers: dict[tuple, list[float]] = {}
        self.digests: dict[tuple, Any] = {}

    def get_quantile(self, p: float):
        """Estimate the p-th quantile for all indices."""
        res = np.zeros(self.shape)
        for idx in np.ndindex(self.shape):
            if idx in self.digests:
                res[idx] = self.digests[idx].quantile(p)
            elif idx in self.buffers and self.buffers[idx]:
                res[idx] = np.percentile(self.buffers[idx], p * 100)
            else:
                res[idx] = 0.0
        return res

    def update(self, x: np.ndarray, mask: np.ndarray = None, idx_override: tuple = None):
        """Update the estimator with new samples."""
        if idx_override is not None:
            # All samples in x are for index idx_override
            self._update_index(idx_override, x, mask)
            return

        if x.ndim > len(self.shape):
            # Batch update
            for i in range(x.shape[0]):
                if mask is not None:
                    self._update_single(x[i], mask[i])
                else:
                    self._update_single(x[i])
        else:
            self._update_single(x, mask)

    def _update_index(self, idx: tuple, x: np.ndarray, mask: np.ndarray = None):
        """Update a specific index with an array of samples."""
        if mask is not None:
            # If idx_override is used, mask should be 1D corresponding to x
            if mask.ndim > 0:
                x = x[mask]
            elif not mask:
                return

        if len(x) == 0:
            return

        if idx in self.digests:
            # Merge with existing digest
            new_digest = TDigest.from_array(x.astype(np.float32), self.compression)
            self.digests[idx] = self.digests[idx].merge(new_digest)
            self.counts[idx] += len(x)
        else:
            if idx not in self.buffers:
                self.buffers[idx] = []

            self.buffers[idx].extend(x.astype(float).tolist())
            self.counts[idx] += len(x)

            if len(self.buffers[idx]) >= self.max_buffer:
                self.digests[idx] = TDigest.from_array(
                    np.array(self.buffers[idx], dtype=np.float32),
                    self.compression,
                )
                del self.buffers[idx]

    def _update_single(self, x: np.ndarray, mask: np.ndarray = None):
        """Update with a single sample of shape self.shape."""
        if mask is not None:
            if not np.any(mask):
                return
            if mask.ndim < x.ndim:
                mask = np.broadcast_to(mask[..., None] if mask.ndim > 0 else mask, x.shape)
        else:
            mask = np.ones(self.shape, dtype=bool)

        for idx in np.ndindex(self.shape):
            if not mask[idx]:
                continue

            if idx in self.digests:
                # Already have a t-digest, merge new sample
                self.digests[idx] = self.digests[idx].merge(
                    TDigest.from_array(np.array([x[idx]], dtype=np.float32), self.compression)
                )
                self.counts[idx] += 1
            else:
                # Still in buffer mode
                if idx not in self.buffers:
                    self.buffers[idx] = []

                self.buffers[idx].append(float(x[idx]))
                self.counts[idx] += 1

                # Check if buffer is full, switch to t-digest
                if len(self.buffers[idx]) >= self.max_buffer:
                    self.digests[idx] = TDigest.from_array(
                        np.array(self.buffers[idx], dtype=np.float32),
                        self.compression,
                    )
                    del self.buffers[idx]

    # Free memory and reduce serialized state size

    def get_state(self):
        """Serialize state for storage and merging."""
        # Compact serialization: lists of indices, means, and weights
        indices = []
        means = []
        weights = []
        for idx, digest in self.digests.items():
            d = digest.to_dict()
            indices.append(list(idx))
            means.append(d["means"].tolist())
            weights.append(d["weights"].tolist())

        # Sparse buffer serialization: only store what's actually in buffer
        buffer_indices = []
        buffer_data = []
        for idx, data in self.buffers.items():
            if data:
                buffer_indices.append(list(idx))
                buffer_data.append(data)

        return {
            "shape": list(self.shape),
            "counts": self.counts.tolist(),
            "max_buffer": self.max_buffer,
            "compression": self.compression,
            "buffers": {"indices": buffer_indices, "data": buffer_data},
            "digests": {"indices": indices, "means": means, "weights": weights},
        }

    def merge_from_state(self, state):
        """Merge another estimator's serialized state into this one.

        For each index in the incoming state, merge its TDigest (or buffer)
        into the corresponding local TDigest/buffer.
        """
        compression = state.get("compression", self.compression)
        incoming_counts = np.array(state["counts"], dtype=int) if state.get("counts") is not None else None

        # Merge digest entries
        if state.get("digests") and isinstance(state["digests"], dict) and "indices" in state["digests"]:
            for idx_list, means_list, weights_list in zip(
                state["digests"]["indices"],
                state["digests"]["means"],
                state["digests"]["weights"],
                strict=True,
            ):
                idx = tuple(idx_list)
                incoming_digest = TDigest.from_means_weights(
                    np.array(means_list, dtype=np.float32),
                    np.array(weights_list, dtype=np.uint32),
                    compression,
                )
                if idx in self.digests:
                    self.digests[idx] = self.digests[idx].merge(incoming_digest)
                elif idx in self.buffers:
                    # Flush buffer to digest, then merge
                    self.digests[idx] = TDigest.from_array(
                        np.array(self.buffers.pop(idx), dtype=np.float32), self.compression
                    ).merge(incoming_digest)
                else:
                    self.digests[idx] = incoming_digest

        # Merge buffer entries
        if state.get("buffers") and isinstance(state["buffers"], dict) and "indices" in state["buffers"]:
            for idx_list, data in zip(state["buffers"]["indices"], state["buffers"]["data"], strict=True):
                idx = tuple(idx_list)
                if idx in self.digests:
                    # Merge buffer data into existing digest
                    if data:
                        new_digest = TDigest.from_array(np.array(data, dtype=np.float32), self.compression)
                        self.digests[idx] = self.digests[idx].merge(new_digest)
                else:
                    if idx not in self.buffers:
                        self.buffers[idx] = []
                    self.buffers[idx].extend(data)
                    if len(self.buffers[idx]) >= self.max_buffer:
                        self.digests[idx] = TDigest.from_array(
                            np.array(self.buffers.pop(idx), dtype=np.float32), self.compression
                        )

        # Merge counts
        if incoming_counts is not None:
            self.counts += incoming_counts

    def load_state(self, state):
        """Load state from serialized form."""
        self.shape = tuple(state["shape"])
        self.max_buffer = state.get("max_buffer", 1000)
        self.compression = state.get("compression", 100)

        if state.get("counts") is not None:
            self.counts = np.array(state["counts"], dtype=int)

        self.buffers = {}
        if state.get("buffers") and isinstance(state["buffers"], dict) and "indices" in state["buffers"]:
            for idx_list, data in zip(state["buffers"]["indices"], state["buffers"]["data"], strict=True):
                self.buffers[tuple(idx_list)] = data

        self.digests = {}
        if state.get("digests") and isinstance(state["digests"], dict) and "indices" in state["digests"]:
            for idx_list, means_list, weights_list in zip(
                state["digests"]["indices"],
                state["digests"]["means"],
                state["digests"]["weights"],
                strict=True,
            ):
                idx = tuple(idx_list)
                means = np.array(means_list, dtype=np.float32)
                weights = np.array(weights_list, dtype=np.uint32)
                self.digests[idx] = TDigest.from_means_weights(means, weights, self.compression)


class StreamingDatasetStatistics:
    """Memory-efficient streaming statistics computation.

    This class is **not** internally synchronized. When used as a Ray actor,
    it *must* be configured with ``max_concurrency=1`` so that FIFO ordering is
    guaranteed and no additional locking is needed.
    """

    def __init__(
        self,
        compute_stats: bool = True,
        max_samples_for_percentiles: int = 1000,
        quantile_compression: int = 100,
    ):
        self.compute_stats = compute_stats
        self.max_samples_for_percentiles = max_samples_for_percentiles
        self.quantile_compression = quantile_compression
        if not compute_stats:
            return

        self.running_means = {}
        self.running_m2s = {}
        self.counts = {}
        self.mins = {}
        self.maxs = {}
        self.quantiles_to_track = [0.01, 0.02, 0.05, 0.95, 0.98, 0.99]
        self.quantile_estimators = {}  # key -> TDigestEstimator (per-timestep)
        self.global_quantile_estimators = {}  # key -> TDigestEstimator (global, per-channel)

    def update(self, sample_lowdim: dict[str, np.ndarray]):
        """Update statistics with new sample (Welford's online algorithm).

        Serialization is handled by the Ray actor (max_concurrency=1).
        """
        if not self.compute_stats:
            return

        raw_mask = sample_lowdim["mask"]
        # Ensure mask is 2D (B, T) regardless of input shape
        if raw_mask.ndim == 1:
            raw_mask = raw_mask[None, :]
        elif raw_mask.ndim > 2:
            raw_mask = raw_mask.reshape(raw_mask.shape[0], raw_mask.shape[1])
        # base_mask is (B, T, 1) - never mutated, used as template for each key
        base_mask = raw_mask[..., None]
        for key, data in sample_lowdim.items():
            if not np.issubdtype(data.dtype, np.number):
                continue
            mask = base_mask
            if data.ndim == 2:
                # Add a batch dimension, assume there is only time and channel dimensions
                data = data[None, ...]
            elif data.ndim != 3:
                raise ValueError(f"Data must have 2 or 3 dimensions, got {data.ndim} for key {key}")

            data = data.copy()  # Avoid modifying the original data
            data[~mask[..., 0]] = 0
            data_nan = data.astype(np.float64).copy()
            data_nan[~mask[..., 0]] = np.nan
            data_min = data.astype(np.float64).copy()
            data_min[~mask[..., 0]] = float("inf")
            data_max = data.astype(np.float64).copy()
            data_max[~mask[..., 0]] = float("-inf")
            if key not in self.counts:
                sum_mask = np.sum(mask, axis=0)
                sum_data = np.sum(data, axis=0)
                mask_sum_mask = sum_mask > 0
                # Use modified sum_mask only for division to avoid division by zero
                sum_mask_for_division = np.where(mask_sum_mask, sum_mask, 1)
                self.running_means[key] = np.where(mask_sum_mask, sum_data / sum_mask_for_division, 0.0)
                # Handle potential NaN/inf values in initial computation
                self.running_means[key] = np.nan_to_num(self.running_means[key], nan=0.0, posinf=0.0, neginf=0.0)

                self.running_m2s[key] = np.where(
                    mask_sum_mask, np.sum((data - self.running_means[key]) ** 2, axis=0), 0.0
                )
                self.running_m2s[key] = np.nan_to_num(self.running_m2s[key], nan=0.0, posinf=0.0, neginf=0.0)
                self.counts[key] = sum_mask
                self.mins[key] = np.where(sum_mask > 0, np.min(data_min, axis=0), float("inf"))
                self.maxs[key] = np.where(sum_mask > 0, np.max(data_max, axis=0), float("-inf"))

                # Initialize T-Digest estimators
                self.quantile_estimators[key] = TDigestEstimator(
                    self.running_means[key].shape,
                    self.max_samples_for_percentiles,
                    self.quantile_compression,
                )
                # Global estimators have shape (C,) - per-channel, global across all samples and timesteps
                self.global_quantile_estimators[key] = TDigestEstimator(
                    (self.running_means[key].shape[-1],),
                    self.max_samples_for_percentiles,
                    self.quantile_compression,
                )
            else:
                # Update running statistics
                n_a = self.counts[key]
                mean_a = self.running_means[key]
                m2_a = self.running_m2s[key]

                n_b = np.sum(mask, axis=0)
                # Safely compute mean_b, handling cases where n_b is 0
                nansum_data = np.nansum(data_nan, axis=0)
                # Use np.divide with where parameter to avoid division warnings
                mean_b = np.divide(nansum_data, n_b, out=np.zeros_like(nansum_data), where=n_b > 0)
                # Handle potential NaN/inf values
                mean_b = np.nan_to_num(mean_b, nan=0.0, posinf=0.0, neginf=0.0)

                # Safely compute m2_b
                m2_b = np.nansum((data_nan - mean_b) ** 2, axis=0)
                m2_b = np.nan_to_num(m2_b, nan=0.0, posinf=0.0, neginf=0.0)

                delta = mean_b - mean_a
                total_count = n_a + n_b

                mask_total_count = total_count > 0
                # Use modified total_count only for division to avoid division by zero
                total_count_for_division = np.where(mask_total_count, total_count, 1)
                self.running_means[key] = np.where(
                    mask_total_count, (n_a * mean_a + n_b * mean_b) / total_count_for_division, 0.0
                )

                self.running_m2s[key] = (
                    m2_a + m2_b + np.where(mask_total_count, delta**2 * (n_a * n_b / total_count_for_division), 0.0)
                )

                self.counts[key] = total_count
                self.mins[key] = np.minimum(self.mins[key], np.min(data_min, axis=0))
                self.maxs[key] = np.maximum(self.maxs[key], np.max(data_max, axis=0))

            # Update T-Digest estimators
            if key in self.quantile_estimators:
                self.quantile_estimators[key].update(data, mask[..., 0])

            if key in self.global_quantile_estimators:
                # data has shape (B, T, C), mask has shape (B, T, 1)
                # We want to update global estimators with all (B*T, C) samples where mask is True
                # Reshape to (B*T, C)
                B, T, C = data.shape
                data_flat = data.reshape(-1, C)
                mask_flat = mask.reshape(-1)
                # Filter data_flat by mask_flat
                data_flat_masked = data_flat[mask_flat]
                # For global we just iterate over channels
                for c in range(C):
                    data_c = data_flat_masked[:, c]
                    if len(data_c) > 0:
                        self.global_quantile_estimators[key].update(data_c, idx_override=(c,))

    @staticmethod
    def _concatenate_samples(samples_batch: list[dict[str, Any]]):
        """Concatenate a list of raw samples into batched arrays."""
        sample_lowdim = {}

        for sample in samples_batch:
            for key, data in sample["lowdim"].items():
                if data.ndim == 1:
                    data = data[..., None]  # (T,) -> (T, 1) — ensure channel dim for scalar-per-timestep data
                if key not in sample_lowdim:
                    sample_lowdim[key] = data[None, ...]
                else:
                    sample_lowdim[key] = np.concatenate([sample_lowdim[key], data[None, ...]], axis=0)
            if "past_mask" in sample and "future_mask" in sample:
                mask = np.logical_or(sample["past_mask"], sample["future_mask"])
                if "mask" not in sample_lowdim:
                    sample_lowdim["mask"] = mask[None, ...]
                else:
                    sample_lowdim["mask"] = np.concatenate([sample_lowdim["mask"], mask[None, ...]], axis=0)

        return sample_lowdim

    @staticmethod
    def _aggregate_lowdim(
        lowdim_dict: dict[str, np.ndarray],
        max_buffer: int = 1000,
        compression: int = 100,
    ) -> dict[str, Any]:
        """Compute aggregates from a batched lowdim dict: {key: (B, T, C), 'mask': (B, T)}.

        Returns a small dict of pre-computed stats per key that can be sent to the actor.
        """
        raw_mask = lowdim_dict["mask"]
        if raw_mask.ndim == 1:
            raw_mask = raw_mask[None, :]
        elif raw_mask.ndim > 2:
            raw_mask = raw_mask.reshape(raw_mask.shape[0], raw_mask.shape[1])
        base_mask = raw_mask[..., None]  # (B, T, 1)

        aggs = {}
        for key, data in lowdim_dict.items():
            if not np.issubdtype(data.dtype, np.number):
                continue
            mask = base_mask
            if data.ndim == 2:
                data = data[None, ...]  # (T, C) -> (1, T, C)
            elif data.ndim != 3:
                raise ValueError(f"Data must have 2 or 3 dimensions, got {data.ndim} for key {key}")

            data = data.copy()
            data[~mask[..., 0]] = 0
            data_nan = data.astype(np.float64).copy()
            data_nan[~mask[..., 0]] = np.nan
            data_min = data.astype(np.float64).copy()
            data_min[~mask[..., 0]] = float("inf")
            data_max = data.astype(np.float64).copy()
            data_max[~mask[..., 0]] = float("-inf")

            n = np.sum(mask, axis=0)  # (T, 1)
            nansum = np.nansum(data_nan, axis=0)  # (T, C)
            safe_n = np.where(n > 0, n, 1)
            mean = np.where(n > 0, nansum / safe_n, 0.0)
            mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            m2 = np.nansum((data_nan - mean) ** 2, axis=0)
            m2 = np.nan_to_num(m2, nan=0.0, posinf=0.0, neginf=0.0)
            mins = np.where(n[..., 0:1] > 0, np.min(data_min, axis=0), float("inf"))
            maxs = np.where(n[..., 0:1] > 0, np.max(data_max, axis=0), float("-inf"))

            B, T, C = data.shape

            # Per-timestep quantile estimator
            per_ts_estimator = TDigestEstimator((T, C), max_buffer, compression)
            per_ts_estimator.update(data, mask[..., 0])

            # Global (per-channel) quantile estimator
            global_estimator = TDigestEstimator((C,), max_buffer, compression)
            data_flat = data.reshape(-1, C)
            mask_flat = mask.reshape(-1)
            data_flat_masked = data_flat[mask_flat]
            for c in range(C):
                data_c = data_flat_masked[:, c]
                if len(data_c) > 0:
                    global_estimator.update(data_c, idx_override=(c,))

            aggs[key] = {
                "n": n,
                "mean": mean,
                "m2": m2,
                "mins": mins,
                "maxs": maxs,
                "per_ts_tdigest_state": per_ts_estimator.get_state(),
                "global_tdigest_state": global_estimator.get_state(),
            }
        return aggs

    @staticmethod
    def compute_batch_aggregates(
        samples_batch: list[dict[str, Any]],
        max_buffer: int = 1000,
        compression: int = 100,
    ) -> dict[str, Any]:
        """Compute per-batch aggregates from raw samples.

        Called internally by ``merge_from_samples`` on the actor. Can also be called
        separately in a worker to pre-aggregate before sending to the actor via
        ``merge_from_aggregates``.
        """
        sample_lowdim = StreamingDatasetStatistics._concatenate_samples(samples_batch)

        result = {"lowdim": StreamingDatasetStatistics._aggregate_lowdim(sample_lowdim, max_buffer, compression)}

        return result

    def merge_from_aggregates(self, aggregates: dict[str, Any]):
        """Merge pre-computed aggregates into running statistics. Runs on the actor.

        This receives only small aggregate arrays (not raw samples), keeping actor memory low.
        """
        if not self.compute_stats:
            return

        for section in ["lowdim"]:
            if section not in aggregates:
                continue
            for key, agg in aggregates[section].items():
                n_b = agg["n"]
                mean_b = agg["mean"]
                m2_b = agg["m2"]

                if key not in self.counts:
                    self.counts[key] = n_b
                    self.running_means[key] = mean_b
                    self.running_m2s[key] = m2_b
                    self.mins[key] = agg["mins"]
                    self.maxs[key] = agg["maxs"]

                    self.quantile_estimators[key] = TDigestEstimator(
                        mean_b.shape,
                        self.max_samples_for_percentiles,
                        self.quantile_compression,
                    )
                    self.global_quantile_estimators[key] = TDigestEstimator(
                        (mean_b.shape[-1],),
                        self.max_samples_for_percentiles,
                        self.quantile_compression,
                    )
                    if "per_ts_tdigest_state" in agg:
                        self.quantile_estimators[key].merge_from_state(agg["per_ts_tdigest_state"])
                    if "global_tdigest_state" in agg:
                        self.global_quantile_estimators[key].merge_from_state(agg["global_tdigest_state"])
                    continue

                n_a = self.counts[key]
                mean_a = self.running_means[key]
                m2_a = self.running_m2s[key]
                delta = mean_b - mean_a
                total_count = n_a + n_b
                mask_tc = total_count > 0
                tc_div = np.where(mask_tc, total_count, 1)
                self.running_means[key] = np.where(mask_tc, (n_a * mean_a + n_b * mean_b) / tc_div, 0.0)
                self.running_m2s[key] = m2_a + m2_b + np.where(mask_tc, delta**2 * (n_a * n_b / tc_div), 0.0)
                self.counts[key] = total_count
                self.mins[key] = np.minimum(self.mins[key], agg["mins"])
                self.maxs[key] = np.maximum(self.maxs[key], agg["maxs"])

                if key in self.quantile_estimators and "per_ts_tdigest_state" in agg:
                    self.quantile_estimators[key].merge_from_state(agg["per_ts_tdigest_state"])
                if key in self.global_quantile_estimators and "global_tdigest_state" in agg:
                    self.global_quantile_estimators[key].merge_from_state(agg["global_tdigest_state"])

    def merge_from_samples(self, samples_batch: list[dict[str, Any]]):
        """Convenience method: compute aggregates from raw samples and merge in one call."""
        if not self.compute_stats:
            return
        aggregates = self.compute_batch_aggregates(
            samples_batch, self.max_samples_for_percentiles, self.quantile_compression
        )
        self.merge_from_aggregates(aggregates)

    def get_statistics(self) -> dict[str, Any]:
        """Get final statistics."""
        if not self.compute_stats:
            return {"statistics_disabled": True}

        stats = {}
        for key in self.counts:
            if any(self.counts[key] > 1):
                # Safe division for variance calculation
                count_minus_one = self.counts[key] - 1
                mask_count_minus_one = count_minus_one > 0
                count_minus_one = np.where(mask_count_minus_one, count_minus_one, 1)
                variance = np.where(mask_count_minus_one, self.running_m2s[key] / count_minus_one, 0.0)
                # Handle potential NaN/inf values in variance
                variance = np.nan_to_num(variance, nan=0.0, posinf=0.0, neginf=0.0)
                std_per_timestep = np.sqrt(np.maximum(variance, 0.0)).tolist()  # Ensure non-negative before sqrt

                # Compute overall std using law of total variance: σ²_overall = E[σ²_t] + Var[μ_t]
                # Use weighted means and variances based on counts per timestep
                weights = self.counts[key][..., 0]
                mean_variance = np.average(variance, axis=0, weights=weights)
                weighted_mean = np.average(self.running_means[key], axis=0, weights=weights)
                variance_of_means = np.average((self.running_means[key] - weighted_mean) ** 2, axis=0, weights=weights)
                overall_variance = mean_variance + variance_of_means
                overall_std = np.sqrt(np.maximum(overall_variance, 0.0)).tolist()
            else:
                std_per_timestep = 0.0
                overall_std = 0.0

            stats[key] = {
                "mean": np.average(self.running_means[key], axis=0, weights=self.counts[key][..., 0]).tolist(),
                "std": overall_std,
                "min": np.min(self.mins[key], axis=0).tolist(),
                "max": np.max(self.maxs[key], axis=0).tolist(),
                "mean_per_timestep": self.running_means[key].tolist(),
                "std_per_timestep": std_per_timestep,
                "min_per_timestep": self.mins[key].tolist(),
                "max_per_timestep": self.maxs[key].tolist(),
            }

            # Get estimates from T-Digest estimators (if available)
            # Point clouds and some other fields may not have percentile estimators
            if key in self.global_quantile_estimators:
                estimates = {
                    p: self.global_quantile_estimators[key].get_quantile(p).tolist() for p in self.quantiles_to_track
                }
                stats[key].update({f"percentile_{int(p * 100)}": estimates[p] for p in self.quantiles_to_track})

                stats[key].update(
                    {
                        "percentile_sample_count": self.counts[key][..., 0].tolist(),
                        "tdigest_state": self.global_quantile_estimators[key].get_state(),
                    }
                )
            else:
                # No percentile estimation available for this field
                for p in self.quantiles_to_track:
                    stats[key][f"percentile_{int(p * 100)}"] = None
                stats[key]["percentile_sample_count"] = 0
                stats[key]["tdigest_state"] = None

            if key in self.quantile_estimators:
                estimates_per_timestep = {
                    p: self.quantile_estimators[key].get_quantile(p).tolist() for p in self.quantiles_to_track
                }
                stats[key].update(
                    {
                        f"percentile_{int(p * 100)}_per_timestep": estimates_per_timestep[p]
                        for p in self.quantiles_to_track
                    }
                )
                stats[key]["tdigest_state_per_timestep"] = self.quantile_estimators[key].get_state()
            else:
                # No per-timestep percentile estimation available
                for p in self.quantiles_to_track:
                    stats[key][f"percentile_{int(p * 100)}_per_timestep"] = None
                stats[key]["tdigest_state_per_timestep"] = None

            stats[key]["count"] = self.counts[key][..., 0].tolist()

        return stats

    def save_state(self, filepath: str):
        """Save the current state of the statistics computation for recovery."""
        if not self.compute_stats:
            return

        # Convert numpy arrays and estimators to serialized state
        state = {
            "compute_stats": self.compute_stats,
            "running_means": {k: v.tolist() for k, v in self.running_means.items()},
            "running_m2s": {k: v.tolist() for k, v in self.running_m2s.items()},
            "counts": {k: v.tolist() for k, v in self.counts.items()},
            "mins": {k: v.tolist() for k, v in self.mins.items()},
            "maxs": {k: v.tolist() for k, v in self.maxs.items()},
            "quantile_estimators": {k: est.get_state() for k, est in self.quantile_estimators.items()},
            "max_samples_for_percentiles": self.max_samples_for_percentiles,
            "quantile_compression": self.quantile_compression,
            "global_quantile_estimators": {k: est.get_state() for k, est in self.global_quantile_estimators.items()},
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load a previously saved state to resume statistics computation."""
        if not self.compute_stats:
            return

        try:
            with open(filepath) as f:
                state = json.load(f)

            # Restore configuration
            self.max_samples_for_percentiles = state.get("max_samples_for_percentiles", 1000)
            self.quantile_compression = state.get("quantile_compression", 100)

            # Convert lists back to numpy arrays
            self.running_means = {k: np.array(v) for k, v in state.get("running_means", {}).items()}
            self.running_m2s = {k: np.array(v) for k, v in state.get("running_m2s", {}).items()}
            self.counts = {k: np.array(v) for k, v in state.get("counts", {}).items()}
            self.mins = {k: np.array(v) for k, v in state.get("mins", {}).items()}
            self.maxs = {k: np.array(v) for k, v in state.get("maxs", {}).items()}

            self.quantile_estimators = {}
            for key, est_state in state.get("quantile_estimators", {}).items():
                est = TDigestEstimator(
                    self.running_means[key].shape,
                    self.max_samples_for_percentiles,
                    self.quantile_compression,
                )
                est.load_state(est_state)
                self.quantile_estimators[key] = est

            self.global_quantile_estimators = {}
            for key, est_state in state.get("global_quantile_estimators", {}).items():
                est = TDigestEstimator(
                    (self.running_means[key].shape[-1],),
                    self.max_samples_for_percentiles,
                    self.quantile_compression,
                )
                est.load_state(est_state)
                self.global_quantile_estimators[key] = est

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load statistics state from {filepath}: {e}")
            # Reset to empty state
            self.running_means = {}
            self.running_m2s = {}
            self.counts = {}
            self.mins = {}
            self.maxs = {}
            self.samples_for_percentiles = {}
            self.sample_mask_for_percentiles = {}

    @classmethod
    def from_saved_state(cls, filepath: str, compute_stats: bool = True, max_samples_for_percentiles: int = 100000):
        """Create a new instance from a saved state file."""
        instance = cls(compute_stats=compute_stats, max_samples_for_percentiles=max_samples_for_percentiles)
        if compute_stats:
            instance.load_state(filepath)
        return instance


@ray.remote(max_concurrency=1)
class StreamingDatasetStatisticsRayActor(StreamingDatasetStatistics):
    pass
