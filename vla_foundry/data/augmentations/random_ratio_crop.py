import random

import torch


class RandomRatioCrop:
    """
    Randomly crop an image to a given ratio of its original dimensions.

    Example:
        # Crop to 80% of original size
        transform = RandomRatioCrop(0.8)

        # Crop to 70% height and 90% width
        transform = RandomRatioCrop((0.7, 0.9))
    """

    def __init__(self, ratio):
        if isinstance(ratio, (tuple, list)):
            self.ratio_h, self.ratio_w = ratio
        else:
            self.ratio_h = self.ratio_w = ratio
        if not (0 < self.ratio_h <= 1 and 0 < self.ratio_w <= 1):
            raise ValueError("Ratios must be between 0 and 1")

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:  # PIL Image
            w, h = img.size

        # Calculate crop dimensions
        crop_h = int(h * self.ratio_h)
        crop_w = int(w * self.ratio_w)

        # Random top-left corner
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # Crop
        if isinstance(img, torch.Tensor):
            return img[..., top : top + crop_h, left : left + crop_w]
        else:  # PIL Image
            return img.crop((left, top, left + crop_w, top + crop_h))

    def __repr__(self):
        return f"{self.__class__.__name__}(ratio_h={self.ratio_h}, ratio_w={self.ratio_w})"
