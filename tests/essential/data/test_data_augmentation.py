#!/usr/bin/env python3
"""
Pytest tests for augmentation_params.py.
"""

import io

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms

from vla_foundry.data.augmentations.decode_and_augment import Augmentations
from vla_foundry.data.augmentations.random_ratio_crop import RandomRatioCrop
from vla_foundry.params.robotics.augmentation_params import (
    ColorJitterParams,
    CropParams,
    DataAugmentationParams,
    ImageAugmentationParams,
)


def create_dummy_image(size=(256, 256)):
    """
    Create a dummy image with random colors for testing purposes.

    Args:
        size (tuple): The size of the image (width, height).

    Returns:
        PIL.Image.Image: A dummy image.
    """
    array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


@pytest.mark.parametrize(
    "brightness,contrast,saturation,hue,should_raise",
    [
        (-0.1, 0.2, 0.3, (-0.1, 0.1), True),  # Invalid brightness
        (0.1, -0.2, 0.3, (-0.1, 0.1), True),  # Invalid contrast
        (0.1, 0.2, -0.3, (-0.1, 0.1), True),  # Invalid saturation
        (0.1, 0.2, 0.3, (-0.6, 0.1), True),  # Invalid hue (low out of range)
        (0.1, 0.2, 0.3, (-0.1, 0.6), True),  # Invalid hue (high out of range)
        (0.1, 0.2, 0.3, (0.1, -0.1), True),  # Invalid hue (lo > hi)
        (0.1, 0.2, 0.3, (0.1), True),  # Hue not a tuple of size two
        (0.1, 0.2, 0.3, (0.1, -0.1), True),  # Invalid hue range
        (0.1, 0.2, 0.3, (-0.1, 0.1), False),  # Valid parameters
    ],
)
def test_color_jitter_params_validation(brightness, contrast, saturation, hue, should_raise):
    """Test validation logic in ColorJitterParams."""
    if should_raise:
        with pytest.raises(ValueError):
            ColorJitterParams(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    else:
        ColorJitterParams(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)  # No error


@pytest.mark.parametrize(
    "crop_factory,should_raise",
    [
        (lambda: CropParams(shape=(128, 128), enabled=True), False),  # Valid crop
        (lambda: CropParams(shape=(-128, 128), enabled=True), True),  # Invalid shape (negative height)
        (lambda: CropParams(shape=(128, -128), enabled=True), True),  # Invalid shape (negative width)
        (lambda: CropParams(shape=(0, 128), enabled=True), True),  # Invalid shape (zero height)
        (lambda: CropParams(shape=(128, 0), enabled=True), True),  # Invalid shape (zero width)
        (lambda: CropParams(shape=(128, 128), enabled=False), False),  # Valid when disabled
    ],
)
def test_image_augmentation_params_crop_validation(crop_factory, should_raise):
    """Test validation logic for crop in ImageAugmentationParams."""
    if should_raise:
        with pytest.raises(ValueError):
            ImageAugmentationParams(
                crop=crop_factory(),
            )
    else:
        ImageAugmentationParams(
            crop=crop_factory(),
        )  # Should not raise


@pytest.mark.parametrize(
    "color_jitter_factory,should_raise",
    [
        (
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            False,
        ),  # Valid color jitter
        (
            lambda: ColorJitterParams(brightness=-0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            True,
        ),  # Invalid brightness
        (
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=False),
            False,
        ),  # Valid when disabled
    ],
)
def test_image_augmentation_params_color_jitter_validation(color_jitter_factory, should_raise):
    """Test validation logic for color jitter in ImageAugmentationParams."""
    if should_raise:
        with pytest.raises(ValueError):
            ImageAugmentationParams(
                color_jitter=color_jitter_factory(),
            )
    else:
        ImageAugmentationParams(
            color_jitter=color_jitter_factory(),
        )  # No error


@pytest.mark.parametrize(
    "crop_factory,color_jitter_factory,should_raise",
    [
        (
            lambda: CropParams(shape=(128, 128), enabled=True),
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            False,
        ),  # Valid color jitter and crop
        (
            lambda: CropParams(shape=(-128, 128), enabled=True),
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            True,
        ),  # Invalid crop_shape
        (
            lambda: CropParams(shape=(128, 128), enabled=True),
            lambda: ColorJitterParams(brightness=-0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True),
            True,
        ),  # Invalid color_jitter parameters
        (
            lambda: CropParams(shape=(128, 128), enabled=False),
            lambda: ColorJitterParams(brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=False),
            False,
        ),  # Valid when both disabled
    ],
)
def test_image_augmentation_params_combined_validation(crop_factory, color_jitter_factory, should_raise):
    """Test combined validation logic in ImageAugmentationParams."""
    if should_raise:
        with pytest.raises(ValueError):
            ImageAugmentationParams(
                crop=crop_factory(),
                color_jitter=color_jitter_factory(),
            )
    else:
        ImageAugmentationParams(
            crop=crop_factory(),
            color_jitter=color_jitter_factory(),
        )  # Should not raise


@pytest.mark.parametrize(
    "augmentation_params,has_transforms",
    [
        (None, False),  # No augmentation parameters provided
        (
            DataAugmentationParams(enabled=False),  # Disabled augmentations
            False,  # No transforms expected
        ),
        (
            DataAugmentationParams(),  # Default (empty augmentations)
            False,  # No transforms expected
        ),
        (
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    crop=CropParams(shape=(128, 128), enabled=True),
                )
            ),
            True,  # Should have transforms
        ),
        (
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    color_jitter=ColorJitterParams(
                        brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True
                    ),
                )
            ),
            True,  # Should have transforms
        ),
        (
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    crop=CropParams(shape=(128, 128), enabled=True),
                    color_jitter=ColorJitterParams(
                        brightness=0.2, contrast=0.3, saturation=0.4, hue=(-0.1, 0.1), enabled=True
                    ),
                )
            ),
            True,  # Should have transforms
        ),
    ],
)
def test_augmentations_class_creation(augmentation_params, has_transforms):
    """Test the creation of the Augmentations class."""
    augmentations = Augmentations(augmentation_params)

    # Check that transforms are created appropriately
    if has_transforms:
        assert isinstance(augmentations.image_transforms, transforms.Compose)
        assert len(augmentations.image_transforms.transforms) > 0
    else:
        assert augmentations.image_transforms is None


def test_augmentations_class_invalid_params():
    """Test that invalid augmentation parameters raise appropriate errors."""
    # Invalid crop_shape
    with pytest.raises(ValueError):
        Augmentations(
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    crop=CropParams(shape=(-128, 128), enabled=True),  # Invalid shape
                )
            )
        )

    # Invalid color_jitter parameters
    with pytest.raises(ValueError):
        Augmentations(
            DataAugmentationParams(
                image=ImageAugmentationParams(
                    color_jitter=ColorJitterParams(
                        brightness=-0.2,  # Invalid brightness
                        contrast=0.3,
                        saturation=0.4,
                        hue=(-0.1, 0.1),
                        enabled=True,
                    ),
                )
            )
        )


def test_crop_augmentation():
    """
    Test that the crop augmentation works as expected.
    """
    augmentation_params = DataAugmentationParams(
        image=ImageAugmentationParams(
            crop=CropParams(shape=(128, 128), enabled=True),
        )
    )
    augmentations = Augmentations(augmentation_params)

    # Check that transforms were created
    assert isinstance(augmentations.image_transforms, transforms.Compose)
    assert len(augmentations.image_transforms.transforms) > 0

    # Test the random crop augmentation
    img = create_dummy_image(size=(256, 256))
    cropped_img = augmentations.image_transforms(img)

    # Verify that the cropped image has the correct size
    assert cropped_img.size == (128, 128)


def test_color_jitter_augmentation():
    """
    Test that the color jitter augmentation works as expected.
    """
    augmentation_params = DataAugmentationParams(
        image=ImageAugmentationParams(
            color_jitter=ColorJitterParams(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.1, 0.1), enabled=True),
        )
    )
    augmentations = Augmentations(augmentation_params)

    # Check that transforms were created
    assert isinstance(augmentations.image_transforms, transforms.Compose)
    assert len(augmentations.image_transforms.transforms) > 0

    # Test the color jitter augmentation
    img = create_dummy_image()
    jittered_img = augmentations.image_transforms(img)

    # Verify that the jittered image is different from the original
    original_array = np.array(img)
    jittered_array = np.array(jittered_img)
    assert not np.array_equal(original_array, jittered_array), "Color jitter should modify the image."


def test_combined_augmentations():
    """
    Test that the pipeline works correctly with both crop and color jitter.
    """
    augmentation_params = DataAugmentationParams(
        image=ImageAugmentationParams(
            crop=CropParams(shape=(128, 128), enabled=True),
            color_jitter=ColorJitterParams(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.1, 0.1), enabled=True),
        )
    )
    augmentations = Augmentations(augmentation_params)

    # Check that transforms were created
    assert isinstance(augmentations.image_transforms, transforms.Compose)
    assert len(augmentations.image_transforms.transforms) == 2  # Should have both transforms

    # Test the combined augmentation pipeline
    img = create_dummy_image(size=(256, 256))
    transformed_img = augmentations.image_transforms(img)

    # Verify that the transformed image has the correct size (cropped)
    assert transformed_img.size == (128, 128)

    # Verify that the image was modified (color jitter should make it different)
    # We can't easily test the exact transformation, but we can check the size
    # and that the transforms were applied in sequence


# Tests for RandomRatioCrop


@pytest.mark.parametrize(
    "ratio,should_raise",
    [
        (0.8, False),  # Valid single ratio
        ((0.7, 0.9), False),  # Valid tuple ratio
        ([0.7, 0.9], False),  # Valid list ratio
        (1.0, False),  # Valid edge case (100%)
        ((1.0, 1.0), False),  # Valid edge case (100% both)
        (0.0, True),  # Invalid: ratio cannot be 0
        (-0.5, True),  # Invalid: negative ratio
        (1.5, True),  # Invalid: ratio > 1
        ((0.5, 1.5), True),  # Invalid: width ratio > 1
        ((1.5, 0.5), True),  # Invalid: height ratio > 1
        ((0, 0.5), True),  # Invalid: height ratio = 0
        ((0.5, 0), True),  # Invalid: width ratio = 0
    ],
)
def test_random_ratio_crop_initialization(ratio, should_raise):
    """Test RandomRatioCrop initialization with various ratio values."""
    if should_raise:
        with pytest.raises(ValueError):
            RandomRatioCrop(ratio)
    else:
        transform = RandomRatioCrop(ratio)
        if isinstance(ratio, (tuple, list)):
            assert transform.ratio_h == ratio[0]
            assert transform.ratio_w == ratio[1]
        else:
            assert transform.ratio_h == ratio
            assert transform.ratio_w == ratio


def test_random_ratio_crop_pil_image():
    """Test RandomRatioCrop with PIL images."""
    transform = RandomRatioCrop(0.5)
    img = create_dummy_image(size=(200, 100))  # width=200, height=100

    cropped_img = transform(img)

    # Check that the cropped image has the expected size
    expected_width = int(200 * 0.5)  # 100
    expected_height = int(100 * 0.5)  # 50
    assert cropped_img.size == (expected_width, expected_height)
    assert isinstance(cropped_img, Image.Image)


def test_random_ratio_crop_pil_image_different_ratios():
    """Test RandomRatioCrop with PIL images using different ratios for height and width."""
    transform = RandomRatioCrop((0.7, 0.9))
    img = create_dummy_image(size=(200, 100))  # width=200, height=100

    cropped_img = transform(img)

    # Check that the cropped image has the expected size
    expected_width = int(200 * 0.9)  # 180
    expected_height = int(100 * 0.7)  # 70
    assert cropped_img.size == (expected_width, expected_height)


def test_random_ratio_crop_torch_tensor():
    """Test RandomRatioCrop with PyTorch tensors."""
    transform = RandomRatioCrop(0.5)
    # Create a tensor with shape (C, H, W) = (3, 100, 200)
    img_tensor = torch.rand(3, 100, 200)

    cropped_tensor = transform(img_tensor)

    # Check that the cropped tensor has the expected shape
    expected_height = int(100 * 0.5)  # 50
    expected_width = int(200 * 0.5)  # 100
    assert cropped_tensor.shape == (3, expected_height, expected_width)
    assert isinstance(cropped_tensor, torch.Tensor)


def test_random_ratio_crop_torch_tensor_different_ratios():
    """Test RandomRatioCrop with PyTorch tensors using different ratios."""
    transform = RandomRatioCrop((0.7, 0.9))
    # Create a tensor with shape (C, H, W) = (3, 100, 200)
    img_tensor = torch.rand(3, 100, 200)

    cropped_tensor = transform(img_tensor)

    # Check that the cropped tensor has the expected shape
    expected_height = int(100 * 0.7)  # 70
    expected_width = int(200 * 0.9)  # 180
    assert cropped_tensor.shape == (3, expected_height, expected_width)


def test_random_ratio_crop_batched_torch_tensor():
    """Test RandomRatioCrop with batched image tensors (..., H, W)."""
    transform = RandomRatioCrop((0.5, 0.25))
    # Shape: (B, T, C, H, W)
    img_tensor = torch.rand(2, 4, 3, 120, 200)

    cropped_tensor = transform(img_tensor)

    expected_height = int(120 * 0.5)  # 60
    expected_width = int(200 * 0.25)  # 50
    assert cropped_tensor.shape == (2, 4, 3, expected_height, expected_width)
    assert isinstance(cropped_tensor, torch.Tensor)


def test_random_ratio_crop_randomness():
    """Test that RandomRatioCrop produces different crops on multiple calls."""
    transform = RandomRatioCrop(0.5)
    img = create_dummy_image(size=(200, 200))

    # Apply the transform multiple times
    crops = [transform(img) for _ in range(10)]

    # Convert to numpy arrays for comparison
    crop_arrays = [np.array(crop) for crop in crops]

    # Check that at least some crops are different (very high probability)
    # We compare the first pixel of each crop
    first_pixels = [arr[0, 0].tolist() for arr in crop_arrays]
    unique_first_pixels = len(set(map(tuple, first_pixels)))

    # With 10 random crops, we expect at least 2 different first pixels
    # (unless we're extremely unlucky)
    assert unique_first_pixels >= 2, "RandomRatioCrop should produce varied crops"


def test_random_ratio_crop_full_image():
    """Test RandomRatioCrop with ratio=1.0 (should return same size)."""
    transform = RandomRatioCrop(1.0)
    img = create_dummy_image(size=(200, 100))

    cropped_img = transform(img)

    # Should return the same size image
    assert cropped_img.size == img.size


def test_random_ratio_crop_repr():
    """Test the __repr__ method of RandomRatioCrop."""
    transform = RandomRatioCrop(0.8)
    repr_str = repr(transform)
    assert "RandomRatioCrop" in repr_str
    assert "ratio_h=0.8" in repr_str
    assert "ratio_w=0.8" in repr_str

    transform2 = RandomRatioCrop((0.7, 0.9))
    repr_str2 = repr(transform2)
    assert "ratio_h=0.7" in repr_str2
    assert "ratio_w=0.9" in repr_str2


# ---------------------------------------------------------------------------
# decode_and_augment_sample tests
# ---------------------------------------------------------------------------


def _make_png_bytes(h=4, w=6):
    """Create minimal PNG bytes for testing."""
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _make_tiff_bytes(h=4, w=6, dtype=np.uint16):
    """Create minimal TIFF bytes for testing."""
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.random.randint(0, 1000, (h, w, 3), dtype=dtype))
    return buf.getvalue()


def test_decode_and_augment_sample_decodes_all_field_types():
    """decode_and_augment_sample should handle images, JSON, NPZ, TIFF, txt, and passthrough."""
    import json

    augmentations = Augmentations(None)  # no transforms

    npz_buf = io.BytesIO()
    np.savez(npz_buf, data=np.array([1.0, 2.0]))
    npz_bytes = npz_buf.getvalue()

    sample = {
        "__key__": "sample_001",  # non-bytes passthrough
        "cam0_t0.jpg": _make_png_bytes(8, 10),
        "cam1_t0.png": _make_png_bytes(8, 10),
        "scene_right_0_point_map_t0.tiff": _make_tiff_bytes(8, 10),
        "lowdim.npz": npz_bytes,
        "metadata.json": json.dumps({"anchor": 3}).encode("utf-8"),
        "caption.txt": b"a photo of a robot",
        "other.bin": b"raw-bytes",
    }

    result = augmentations.decode_and_augment_sample(sample)

    # Non-bytes passthrough
    assert result["__key__"] == "sample_001"
    # Images decoded to CHW uint8 tensors
    assert isinstance(result["cam0_t0.jpg"], torch.Tensor)
    assert result["cam0_t0.jpg"].shape == (3, 8, 10)
    assert result["cam0_t0.jpg"].dtype == torch.uint8
    assert isinstance(result["cam1_t0.png"], torch.Tensor)
    # TIFF decoded to numpy array
    assert isinstance(result["scene_right_0_point_map_t0.tiff"], np.ndarray)
    assert result["scene_right_0_point_map_t0.tiff"].shape == (8, 10, 3)
    assert result["scene_right_0_point_map_t0.tiff"].dtype == np.uint16
    # NPZ decoded to dict
    assert isinstance(result["lowdim.npz"], dict)
    np.testing.assert_array_equal(result["lowdim.npz"]["data"], [1.0, 2.0])
    # JSON decoded
    assert result["metadata.json"] == {"anchor": 3}
    # Text decoded to string
    assert result["caption.txt"] == "a photo of a robot"
    # Unknown bytes passed through
    assert result["other.bin"] == b"raw-bytes"


def test_decode_and_augment_sample_applies_transforms():
    """decode_and_augment_sample should apply image transforms when configured."""
    augmentation_params = DataAugmentationParams(
        enabled=True,
        image={"crop": CropParams(enabled=True, mode="center", shape=[4, 4])},
    )
    augmentations = Augmentations(augmentation_params)

    sample = {
        "__key__": "s1",
        "cam.jpg": _make_png_bytes(8, 10),
    }

    result = augmentations.decode_and_augment_sample(sample)

    assert isinstance(result["cam.jpg"], torch.Tensor)
    # Center crop to 4x4
    assert result["cam.jpg"].shape == (3, 4, 4)


def test_decode_and_augment_sample_extension_only_keys():
    """decode_and_augment_sample should handle bare extension keys like 'jpg' and 'txt'."""
    augmentations = Augmentations(None)

    sample = {
        "__key__": "000000",
        "jpg": _make_png_bytes(8, 10),
        "txt": b"a caption",
    }

    result = augmentations.decode_and_augment_sample(sample)

    assert result["__key__"] == "000000"
    assert isinstance(result["jpg"], torch.Tensor)
    assert result["jpg"].shape == (3, 8, 10)
    assert result["txt"] == "a caption"


def test_random_ratio_crop_maintains_content():
    """Test that RandomRatioCrop maintains image content (crops from original)."""
    transform = RandomRatioCrop(0.5)

    # Create a distinctive image with a gradient pattern
    img_array = np.zeros((100, 200, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(200):
            img_array[i, j] = [i * 2, j, 128]  # Distinctive pattern
    img = Image.fromarray(img_array)

    cropped_img = transform(img)
    cropped_array = np.array(cropped_img)

    # Check that all pixel values in the crop are within the range of the original
    assert cropped_array.min() >= img_array.min()
    assert cropped_array.max() <= img_array.max()

    # Check dimensions
    assert cropped_img.size == (100, 50)  # width=100, height=50


def test_random_ratio_crop_with_torch_different_channels():
    """Test RandomRatioCrop with tensors of different channel counts."""
    # Test with 1 channel (grayscale)
    transform = RandomRatioCrop(0.5)
    img_tensor_1ch = torch.rand(1, 100, 200)
    cropped_1ch = transform(img_tensor_1ch)
    assert cropped_1ch.shape == (1, 50, 100)

    # Test with 3 channels (RGB)
    img_tensor_3ch = torch.rand(3, 100, 200)
    cropped_3ch = transform(img_tensor_3ch)
    assert cropped_3ch.shape == (3, 50, 100)

    # Test with 4 channels (RGBA)
    img_tensor_4ch = torch.rand(4, 100, 200)
    cropped_4ch = transform(img_tensor_4ch)
    assert cropped_4ch.shape == (4, 50, 100)


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
