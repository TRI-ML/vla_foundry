from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# Minimum number of samples required to compute reliable median for anti-aliasing filter
MIN_SAMPLES_FOR_ANTIALIASING = 8


class TemporalResampler:
    """
    Resamples multi-rate signals to uniform target frequency with anti-aliasing.

    Supports:
    - Continuous signals (linear interpolation with anti-aliasing filter)
    - Images (nearest-neighbor selection)
    """

    def __init__(self, target_hz: float):
        """
        Initialize resampler.

        Args:
            target_hz: Target sampling frequency in Hz
        """
        if target_hz <= 0:
            raise ValueError(f"target_hz must be positive, got {target_hz}")
        self.target_hz = target_hz

    def create_target_timeline(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Create target timeline with truly uniform frequency spacing.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Array of target timestamps
        """
        return np.arange(start_time, end_time, 1.0 / self.target_hz)

    def _validate_inputs(
        self, source_times: np.ndarray, source_values: np.ndarray | list[Any], target_times: np.ndarray
    ) -> None:
        """Validate resampling inputs. Raises ValueError for invalid inputs."""
        if len(source_times) != len(source_values):
            raise ValueError(
                f"source_times ({len(source_times)}) and source_values ({len(source_values)}) must have the same length"
            )
        if len(source_times) == 0 or len(target_times) == 0:
            raise ValueError("source_times and target_times must not be empty")

    def _broadcast_single_sample(self, source_values: np.ndarray | list[Any], n: int) -> np.ndarray | list[Any]:
        """Broadcast a single source sample to fill n target slots."""
        if isinstance(source_values, list):
            return [source_values[0]] * n
        if source_values.ndim == 1:
            return np.full(n, source_values[0])
        return np.tile(source_values[0], (n, 1))

    def _apply_antialiasing_filter(self, values: np.ndarray, source_hz: float, order=4) -> np.ndarray:
        """
        Apply Butterworth low-pass filter to prevent aliasing when downsampling.

        Args:
            values: Source values to filter
            source_hz: Source sampling frequency
            order: Order of the filter

        Returns:
            Filtered values if downsampling requires anti-aliasing.
        """
        nyquist = source_hz / 2.0
        norm_cutoff = (self.target_hz / 2.0) / nyquist
        if not 0 < norm_cutoff < 1:
            # No filtering needed if source rate is below target Nyquist
            return values

        b, a = butter(order, norm_cutoff, btype="low")
        return filtfilt(b, a, values, axis=0)

    def resample_continuous(
        self, source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray, method: str = "linear"
    ) -> np.ndarray:
        """
        Resample continuous signals with Nyquist-aware anti-aliasing.

        Applies anti-aliasing filter when source frequency > 2x target frequency
        to prevent aliasing artifacts during downsampling.

        Args:
            source_times: Source timestamps
            source_values: Source values (1D or 2D)
            target_times: Target timestamps
            method: Interpolation method ('linear', 'cubic', etc.)

        Returns:
            Resampled values at target times
        """
        values = source_values
        self._validate_inputs(source_times, values, target_times)
        if len(source_times) == 1:
            return self._broadcast_single_sample(values, len(target_times))

        # Apply anti-aliasing if downsampling significantly
        if len(source_times) > MIN_SAMPLES_FOR_ANTIALIASING:
            dt_median = np.median(np.diff(source_times))
            if dt_median > 0 and (1.0 / dt_median) > self.target_hz * 2.0:
                values = self._apply_antialiasing_filter(values, 1.0 / dt_median)

        # Interpolate
        if values.ndim == 1:
            return interp1d(source_times, values, kind=method, fill_value="extrapolate")(target_times)

        resampled = np.zeros((len(target_times), values.shape[1]))
        for i in range(values.shape[1]):
            resampled[:, i] = interp1d(source_times, values[:, i], kind=method, fill_value="extrapolate")(target_times)
        return resampled

    def resample_discrete(
        self, source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray
    ) -> np.ndarray:
        """
        Resample discrete signal using zero-order hold.

        Args:
            source_times: Source timestamps
            source_values: Source values
            target_times: Target timestamps

        Returns:
            Resampled values (held from previous source value)
        """
        self._validate_inputs(source_times, source_values, target_times)
        if len(source_times) == 1:
            return self._broadcast_single_sample(source_values, len(target_times))

        resampled = np.zeros((len(target_times),) + source_values.shape[1:], dtype=source_values.dtype)
        source_idx = 0

        for target_idx, target_time in enumerate(target_times):
            while source_idx < len(source_times) - 1 and source_times[source_idx + 1] <= target_time:
                source_idx += 1
            resampled[target_idx] = source_values[source_idx]

        return resampled

    def resample_images(
        self, source_times: np.ndarray, source_images: list[Any], target_times: np.ndarray
    ) -> list[Any]:
        """
        Temporal resampling for images using nearest-neighbor lookup.

        Args:
            source_times: Source timestamps
            source_images: List of images (bytes or arrays)
            target_times: Target timestamps

        Returns:
            List of resampled images
        """
        self._validate_inputs(source_times, source_images, target_times)
        if len(source_times) == 1:
            return self._broadcast_single_sample(source_images, len(target_times))

        return [source_images[np.argmin(np.abs(source_times - t))] for t in target_times]
