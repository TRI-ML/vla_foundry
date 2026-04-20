"""
Image decoding and augmentation for WebDataset pipelines.

Decodes raw image bytes to CHW uint8 tensors via torchvision (with PIL
fallback) and applies configured augmentations in a single pass.
"""

import io
import json
import logging

import numpy as np
import tifffile
import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2 as transforms

from vla_foundry.data.augmentations.random_ratio_crop import RandomRatioCrop
from vla_foundry.params.robotics.augmentation_params import DataAugmentationParams

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
_TIFF_EXTENSIONS = (".tiff", ".tif")


# ---------------------------------------------------------------------------
# Image decoding helpers
# ---------------------------------------------------------------------------


def _pil_to_chw_tensor(image_hwc: np.ndarray) -> torch.Tensor:
    """Convert an HWC numpy image into a contiguous CHW uint8 tensor."""
    if image_hwc.dtype != np.uint8:
        image_hwc = image_hwc.astype(np.uint8, copy=False)
    if not image_hwc.flags.writeable:
        image_hwc = np.array(image_hwc, copy=True)
    return torch.from_numpy(image_hwc).permute(2, 0, 1).contiguous()


def _encoded_bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Convert encoded image bytes into a writable uint8 tensor."""
    return torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())


def _decode_with_pil(data: bytes) -> torch.Tensor:
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    return _pil_to_chw_tensor(np.asarray(pil_img))


def _decode_with_torchvision(data: bytes) -> torch.Tensor:
    encoded = _encoded_bytes_to_tensor(data)
    try:
        return decode_image(encoded, mode=ImageReadMode.RGB).contiguous()
    except RuntimeError:
        return _decode_with_pil(data)


def _is_image_key(key: str) -> bool:
    """Check if a key represents an image (handles both 'cam.jpg' and bare 'jpg')."""
    lower = key.lower()
    return lower.endswith(_IMAGE_EXTENSIONS) or lower in {ext.lstrip(".") for ext in _IMAGE_EXTENSIONS}


def _is_tiff_key(key: str) -> bool:
    """Check if a key represents a TIFF file (handles both 'map.tiff' and bare 'tiff')."""
    lower = key.lower()
    return lower.endswith(_TIFF_EXTENSIONS) or lower in {ext.lstrip(".") for ext in _TIFF_EXTENSIONS}


def fast_image_decoder(key: str, data: bytes) -> torch.Tensor | None:
    """WebDataset-compatible image decoder returning CHW uint8 tensors."""
    if not _is_image_key(key):
        return None
    return _decode_with_torchvision(data)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


class Augmentations:
    def __init__(self, augmentation_params: DataAugmentationParams):
        self.augmentation_params = augmentation_params
        self.construct_transforms()

    def construct_transforms(self):
        self.image_transforms = None
        self.point_cloud_transforms = None

        image_transforms = []

        if self.augmentation_params is None or not self.augmentation_params.enabled:
            self.image_transforms = None
            self._crop_size: tuple[int, int] | None = None
            return

        # Add crop augmentation.
        crop = self.augmentation_params.image.get("crop", None)
        self._crop_size = None

        if crop is not None and crop.enabled:
            crop_h, crop_w = crop.shape
            if crop.mode == "center":
                if crop_h <= 1.0 and crop_w <= 1.0:
                    raise ValueError("Center crop with ratio-based shape is not supported. Use absolute pixel values.")
                self._crop_size = (int(crop_h), int(crop_w))
                image_transforms.append(transforms.CenterCrop(self._crop_size))
            else:  # random mode
                if crop_h <= 1.0 and crop_w <= 1.0:
                    image_transforms.append(RandomRatioCrop((crop_h, crop_w)))
                elif crop_h > 1.0 and crop_w > 1.0:
                    self._crop_size = (int(crop_h), int(crop_w))
                    image_transforms.append(transforms.RandomCrop(self._crop_size))
                else:
                    raise ValueError(f"Invalid crop shape: {crop.shape}")

        # Add color jitter augmentation for images
        if (color_jitter := self.augmentation_params.image.get("color_jitter", None)) and color_jitter.enabled:
            image_transforms.append(
                transforms.ColorJitter(
                    brightness=color_jitter.brightness,
                    contrast=color_jitter.contrast,
                    saturation=color_jitter.saturation,
                    hue=color_jitter.hue,
                )
            )

        # v2.Compose requires at least one transform
        self.image_transforms = transforms.Compose(image_transforms) if image_transforms else None

        self.point_cloud_transforms = None

    def decode_and_augment_sample(self, sample: dict) -> dict:
        """Decode all fields in a raw WebDataset sample and augment images.

        Images are decoded to CHW uint8 tensors and augmented as a batch so
        that random transforms (crop, color jitter) use the same parameters
        across all camera views.  Falls back to per-image transforms when
        images have different spatial dimensions.
        Non-image bytes fields (*.npz, *.json, *.txt) are decoded to native types.
        """
        result = {}
        image_keys: list[str] = []
        image_tensors: list[torch.Tensor] = []

        for key, data in sample.items():
            lower = key.lower()
            if not isinstance(data, bytes):
                result[key] = data
            elif _is_image_key(key):
                tensor = fast_image_decoder(key, data)
                if tensor is not None:
                    image_keys.append(key)
                    image_tensors.append(tensor)
                else:
                    result[key] = data
            elif _is_tiff_key(key):
                try:
                    result[key] = tifffile.imread(io.BytesIO(data))
                except Exception:
                    try:
                        result[key] = np.array(Image.open(io.BytesIO(data)))
                    except Exception:
                        logger.warning(f"All TIFF decoding methods failed for {key}")
                        result[key] = data
            elif lower.endswith(".npz") or lower == "npz":
                with np.load(io.BytesIO(data)) as npz_file:
                    npz = {k: v for k, v in npz_file.items()}
                if self.point_cloud_transforms is not None and key.endswith("point_cloud.npz") and "data" in npz:
                    npz["data"] = self.point_cloud_transforms(npz["data"])
                result[key] = npz
            elif lower.endswith(".json") or lower == "json":
                result[key] = json.loads(data.decode("utf-8"))
            elif lower.endswith(".txt") or lower == "txt":
                result[key] = data.decode("utf-8")
            else:
                result[key] = data

        if image_tensors and self.image_transforms is not None:
            # Validate that images are large enough for the configured crop.
            if self._crop_size is not None:
                crop_h, crop_w = self._crop_size
                for key, t in zip(image_keys, image_tensors, strict=True):
                    _, img_h, img_w = t.shape
                    if img_h < crop_h or img_w < crop_w:
                        raise ValueError(
                            f"Image '{key}' has size ({img_h}, {img_w}) which is smaller than "
                            f"crop size ({crop_h}, {crop_w}). Reduce the crop size or resize "
                            f"images during preprocessing."
                        )

            # Group by shape, batch-transform each group so images of the
            # same size share the same random augmentation parameters.
            groups: dict[tuple, list[int]] = {}
            for i, t in enumerate(image_tensors):
                groups.setdefault(t.shape, []).append(i)
            for indices in groups.values():
                batch = torch.stack([image_tensors[i] for i in indices])
                batch = self.image_transforms(batch)
                for idx, tensor in zip(indices, batch, strict=True):
                    result[image_keys[idx]] = tensor
        else:
            for key, tensor in zip(image_keys, image_tensors, strict=True):
                result[key] = tensor

        return result
