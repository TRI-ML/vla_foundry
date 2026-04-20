import io
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from vla_foundry.data.augmentations.decode_and_augment import (
    fast_image_decoder,
)


def _png_bytes(image_hwc: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_hwc, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def test_fast_image_decoder_returns_chw_uint8_tensor():
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        ],
        dtype=np.uint8,
    )

    decoded = fast_image_decoder("frame.png", _png_bytes(image))

    assert isinstance(decoded, torch.Tensor)
    assert decoded.dtype == torch.uint8
    assert decoded.shape == (3, 2, 3)
    assert decoded.is_contiguous()
    assert decoded[:, 0, 0].tolist() == [10, 20, 30]


def test_fast_image_decoder_returns_none_for_non_image_key():
    assert fast_image_decoder("sample.txt", b"not-an-image") is None


def test_pil_fallback_when_torchvision_fails():
    """If torchvision decode_image raises, PIL fallback should still produce a tensor."""
    image = np.full((4, 5, 3), fill_value=42, dtype=np.uint8)
    with patch(
        "vla_foundry.data.augmentations.decode_and_augment.decode_image",
        side_effect=RuntimeError("mock failure"),
    ):
        decoded = fast_image_decoder("img.png", _png_bytes(image))

    assert isinstance(decoded, torch.Tensor)
    assert decoded.dtype == torch.uint8
    assert decoded.shape == (3, 4, 5)
