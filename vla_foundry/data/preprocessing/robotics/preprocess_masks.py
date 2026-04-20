import numpy as np


class PaddingStrategy:
    """Optimized padding strategies with vectorized operations."""

    @staticmethod
    def copy_edge(data: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
        """Vectorized edge padding."""
        if pad_before == 0 and pad_after == 0:
            return data

        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, pad_width, mode="edge")

    @staticmethod
    def zero_pad(data: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
        """Vectorized zero padding."""
        if pad_before == 0 and pad_after == 0:
            return data

        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, pad_width, mode="constant", constant_values=0)

    @staticmethod
    def reflect_pad(data: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
        """Vectorized reflect padding."""
        if pad_before == 0 and pad_after == 0:
            return data

        pad_width = [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1)
        return np.pad(data, pad_width, mode="reflect")

    @staticmethod
    def get_pad_fn(padding_strategy: str):
        if padding_strategy == "copy":
            return PaddingStrategy.copy_edge
        elif padding_strategy == "zero":
            return PaddingStrategy.zero_pad
        elif padding_strategy == "reflect":
            return PaddingStrategy.reflect_pad
        else:
            raise ValueError(f"Invalid padding strategy: {padding_strategy}")


def create_past_and_future_masks(idx, num_past, num_future, episode_length):
    total_length = num_past + num_future + 1
    past_mask = np.ones(total_length, dtype=bool)
    future_mask = np.ones(total_length, dtype=bool)
    # Current time step is part of the future for low dim data
    past_mask[num_past:] = False
    future_mask[:num_past] = False

    # Check padding
    past_padding = max(0, -(idx - num_past))
    future_padding = max(0, (idx + num_future) - episode_length + 1)

    if past_padding > 0:
        past_mask[:past_padding] = False
        future_mask[:past_padding] = False
    if future_padding > 0:
        future_mask[-future_padding:] = False
        past_mask[-future_padding:] = False

    return past_mask, future_mask
