import io

import numpy as np
import pytest
from PIL import Image

from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod, image_to_bytes
from vla_foundry.data.preprocessing.robotics.converters.base import upload_sample_to_s3


def test_resize_images_size_none_requires_bytes_input():
    """Test that resize_images_size=None fails with numpy arrays."""

    sample_data = {
        "images": {"cam1": np.zeros((480, 640, 3), dtype=np.uint8)},  # numpy array, not bytes
        "lowdim": {},
    }

    with pytest.raises(ValueError, match="resize_images_size is not configured"):
        upload_sample_to_s3(
            sample_data=sample_data,
            output_dir="/tmp/test",
            episode_path="episode_0000",
            episode_id="0000",
            frame_idx=0,
            jpeg_quality=90,
            resize_images_size=None,
        )


def test_center_crop_resizing_method():
    """Test CENTER_CROP method produces exact target size without black bars."""

    # Create a portrait test image 100 x 200 (W x H)
    test_image = np.random.randint(50, 200, (200, 100, 3), dtype=np.uint8)

    image_bytes, original_size = image_to_bytes(
        test_image, target_size=(224, 224), resize_method=ImageResizingMethod.CENTER_CROP, quality=90
    )

    # Original size should be preserved (W, H)
    assert original_size == (100, 200)

    # Decoded image should be resized to target
    decoded = Image.open(io.BytesIO(image_bytes))
    assert decoded.size == (224, 224)

    # CENTER_CROP should not have black bars (no edge pixels should be all black)
    decoded_array = np.array(decoded)
    assert not np.all(decoded_array[:, 0, :] == 0)  # Left edge has content


def test_resize_fit_method_adds_letterboxing():
    """Test RESIZE_FIT method preserves aspect ratio with black bars."""
    # Create a portrait test image 100 x 200 (W x H) with non-black content
    test_image = np.random.randint(50, 200, (200, 100, 3), dtype=np.uint8)

    image_bytes, original_size = image_to_bytes(
        test_image, target_size=(224, 224), resize_method=ImageResizingMethod.RESIZE_FIT, quality=90
    )

    # Original size should be preserved (W, H)
    assert original_size == (100, 200)

    # Decoded image should be resized to target
    decoded = Image.open(io.BytesIO(image_bytes))
    assert decoded.size == (224, 224)

    # RESIZE_FIT should add black bars on sides for portrait->square
    decoded_array = np.array(decoded)
    assert np.all(decoded_array[:, 0, :] == 0)  # Left edge is black
    assert np.all(decoded_array[:, -1, :] == 0)  # Right edge is black

    # Center should have content (not black)
    center_col = decoded_array[:, 112, :]
    assert not np.all(center_col == 0)
