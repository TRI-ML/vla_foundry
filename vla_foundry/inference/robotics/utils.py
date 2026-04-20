from typing import Any

import numpy as np
from PIL import Image


def relative_to_absolute_map(field: str) -> str:
    """Convert a relative field name to its absolute counterpart."""
    return field.replace("_relative", "")


def any_to_actual_map(field: str) -> str:
    """Convert any field name to its 'actual' counterpart for field mapping lookup."""
    parts = field.split("__")
    if len(parts) > 2:
        return "__".join(parts[0:1] + ["actual"] + parts[2:])
    return field.replace("__action__", "__actual__")


def center_crop(img: Any, target_h: int, target_w: int) -> Any:
    if isinstance(img, np.ndarray):
        pil = Image.fromarray(img)
        width, height = pil.size
    elif isinstance(img, Image.Image):
        width, height = img.size
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")
    crop_w = min(target_w, width)
    crop_h = min(target_h, height)
    left = max((width - crop_w) // 2, 0)
    top = max((height - crop_h) // 2, 0)
    right = left + crop_w
    bottom = top + crop_h
    if isinstance(img, np.ndarray):
        return np.array(pil.crop((left, top, right, bottom)))
    if isinstance(img, Image.Image):
        return img.crop((left, top, right, bottom))
