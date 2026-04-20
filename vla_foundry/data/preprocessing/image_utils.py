import io
from enum import Enum

import numpy as np
import tifffile
from PIL import Image

# Global JPEG encoder pool to avoid repeated PIL overhead
_jpeg_quality = 95


def init_jpeg_encoder(quality: int = 95):
    """Initialize global JPEG encoder settings."""
    global _jpeg_quality
    _jpeg_quality = quality


class ImageResizingMethod(Enum):
    CENTER_CROP = "center_crop"
    RESIZE_NO_CROP = "resize_no_crop"
    RESIZE_FIT = "resize_fit"


def resize_and_crop_image(
    image: np.ndarray | Image.Image,
    target_size: tuple,
    resize_method=ImageResizingMethod.CENTER_CROP,
    fill_color: tuple = (0, 0, 0),
) -> np.ndarray:
    """Resize and crop image to target size according to the given `resize_method`."""
    # Calculate aspect-ratio-preserving dimensions
    is_pil = isinstance(image, Image.Image)

    pil_image = image if is_pil else Image.fromarray(image)
    target_width, target_height = target_size
    orig_width, orig_height = pil_image.size

    if resize_method == ImageResizingMethod.CENTER_CROP:
        # Calculate scale to cover target dimensions (no black bars)
        scale = max(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize to cover the target area
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # Center crop to exact target size
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        pil_image = pil_image.crop((left, top, right, bottom))
    elif resize_method == ImageResizingMethod.RESIZE_NO_CROP:
        # Directly resize to target dimensions (may distort)
        pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    elif resize_method == ImageResizingMethod.RESIZE_FIT:
        # Calculate scale to fit within target dimensions
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize to fit within target
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # Create canvas and paste centered
        result = Image.new(pil_image.mode, target_size, fill_color)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        result.paste(pil_image, (paste_x, paste_y))
        pil_image = result
    else:
        raise ValueError(f"Unrecognized image resizing method: {resize_method}")

    return pil_image if is_pil else np.array(pil_image)


def rotate_image(image: np.ndarray | Image.Image, k: int) -> np.ndarray | Image.Image:
    """
    Rotate an image by 90 degrees k times clockwise.

    Args:
        image: Image to rotate (numpy array or PIL Image)
        k: Number of times to rotate by 90 degrees clockwise.
           k=1 rotates 90° clockwise
           k=2 rotates 180°
           k=3 rotates 270° clockwise (90° counterclockwise)
           k=0 or k=4 returns image unchanged

    Returns:
        Rotated image in the same format as input
    """
    if k % 4 == 0:
        return image

    k = k % 4  # Normalize to [0, 1, 2, 3]
    if isinstance(image, Image.Image):
        rotation_map = {
            1: Image.ROTATE_270,  # 90° clockwise
            2: Image.ROTATE_180,  # 180°
            3: Image.ROTATE_90,  # 270° clockwise (90° counterclockwise)
        }
        return image.transpose(rotation_map[k])
    else:
        # NumPy rotation: rot90 with k=1 rotates counterclockwise by default
        # To rotate clockwise, we use k=-k or equivalently (4-k)
        return np.rot90(image, k=-k)


def depth_image_to_bytes(
    image: np.ndarray, target_size: tuple[int, int] | None = None
) -> tuple[bytes, tuple[int, int]]:
    """Convert depth image to PNG with uint16 format (millimeters)."""
    # Ensure uint16 format for depth (mm units)
    assert image.dtype == np.uint16, "depth images must use np.uint16"

    # Convert to PIL
    pil_image = Image.fromarray(image, mode="I;16")
    original_image_size = pil_image.size

    # Resize if needed
    if target_size and pil_image.size != target_size:
        pil_image = resize_and_crop_image(pil_image, target_size)

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue(), original_image_size


def point_map_to_bytes(point_map: np.ndarray, target_size: tuple = None) -> tuple[bytes, tuple[int, int]]:
    """Convert point map (H, W, 3) uint16 to TIFF with 3 channels.

    Args:
        point_map: (H, W, 3) uint16 array with XYZ coordinates in millimeters (offset by +POINT_MAP_UINT16_OFFSET)
        target_size: Optional target size for resizing (typically None to preserve resolution)

    Returns:
        Tuple of (bytes, original_size)
    """

    # Ensure uint16 format with 3 channels
    assert point_map.dtype == np.uint16, f"point maps must use np.uint16, got {point_map.dtype}"
    assert point_map.ndim == 3 and point_map.shape[2] == 3, f"point maps must be (H, W, 3), got {point_map.shape}"

    original_image_size = (point_map.shape[1], point_map.shape[0])  # (W, H)

    # Resize if needed (resize each channel separately using NEAREST to avoid interpolation artifacts)
    if target_size and original_image_size != target_size:
        target_width, target_height = target_size

        # Resize using nearest neighbor to preserve coordinate integrity
        # Use the same center-crop logic as other images
        resized_channels = []
        for channel_idx in range(3):
            channel = point_map[:, :, channel_idx]
            pil_channel = Image.fromarray(channel, mode="I;16")

            orig_width, orig_height = pil_channel.size

            # Calculate scale to cover target dimensions
            scale = max(target_width / orig_width, target_height / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)

            # Resize with NEAREST (no interpolation)
            pil_channel = pil_channel.resize((new_width, new_height), Image.NEAREST)

            # Center crop to exact target size
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            pil_channel = pil_channel.crop((left, top, right, bottom))

            resized_channels.append(np.array(pil_channel))
        point_map = np.stack(resized_channels, axis=-1).astype(np.uint16)

    # Save as TIFF with 3 channels using tifffile (better multi-channel support)
    buf = io.BytesIO()
    tifffile.imwrite(buf, point_map, compression="adobe_deflate")

    return buf.getvalue(), original_image_size


def image_to_bytes(
    image: np.ndarray,
    quality: int | None = None,
    target_size: tuple[int, int] | None = None,
    resize_method=ImageResizingMethod.CENTER_CROP,
) -> tuple[bytes, tuple[int, int]]:
    """Optimized image to JPEG conversion with resize and minimal allocations."""
    if quality is None:
        quality = _jpeg_quality

    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Convert to PIL and resize to target size while maintaining aspect ratio (center crop)
    pil_image = Image.fromarray(image)
    original_image_size = pil_image.size
    if target_size is not None and pil_image.size != target_size:
        pil_image = resize_and_crop_image(pil_image, resize_method=resize_method, target_size=target_size)

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), original_image_size
