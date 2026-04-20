"""SimpleVLMProcessor: combines SmolVLM2's tokenizer with simple image preprocessing for a from-scratch ViT."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer


class SimpleVLMProcessor:
    """Combines SmolVLM2's tokenizer with simple image preprocessing for a from-scratch ViT.

    This processor uses:
    - SmolVLM2's tokenizer (49k vocab) for text tokenization
    - Simple image preprocessing (resize + normalize) with no tiling

    Handles both image_caption pipeline (flat list of PIL Images) and robotics
    pipeline (nested list of numpy arrays per sample with multiple cameras).
    """

    def __init__(
        self,
        tokenizer_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        image_size: int = 224,
        image_mean: tuple = (0.5, 0.5, 0.5),
        image_std: tuple = (0.5, 0.5, 0.5),
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.image_size = image_size
        self.image_seq_length = 0  # Will be set by get_processor from data_params.img_num_tokens
        self.chat_template = None  # Required by robotics pipeline
        self._mean = torch.tensor(image_mean).view(3, 1, 1)
        self._std = torch.tensor(image_std).view(3, 1, 1)

    def decode(self, *args, **kwargs):
        """Delegate decoding to the underlying tokenizer."""
        return self.tokenizer.decode(*args, **kwargs)

    def _process_image(self, img):
        """Convert a single image (PIL, numpy, or tensor) to a normalized [C, H, W] tensor."""
        if isinstance(img, Image.Image):
            img = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        elif isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img = torch.as_tensor(img, dtype=torch.float32) / 255.0
            else:
                img = torch.as_tensor(img, dtype=torch.float32)
            if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                img = img.permute(2, 0, 1)  # HWC -> CHW
        elif isinstance(img, torch.Tensor):
            img = img.float()
            if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                img = img.permute(2, 0, 1)
        # Resize
        if img.shape[-2] != self.image_size or img.shape[-1] != self.image_size:
            img = F.interpolate(
                img.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            ).squeeze(0)
        # Normalize
        img = (img - self._mean) / self._std
        return img

    def __call__(self, images, text, return_tensors="pt", padding="max_length", max_length=2048, **kwargs):
        # Accept single image or single string (wrap into lists for uniform handling)
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(text, str):
            text = [text]

        # Expand each <image> token in text into image_seq_length copies so the model
        # knows where to splice in ViT patch embeddings.
        if self.image_seq_length > 0:
            image_token = "<image>"
            expanded = image_token * self.image_seq_length
            text = [t.replace(image_token, expanded) for t in text]

        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            truncation=True,
        )

        # Handle both flat (image_caption) and nested (robotics) image formats
        processed = []
        if images is not None:
            for item in images:
                if isinstance(item, (list, tuple)):
                    # Nested: list of images per sample (robotics multi-camera)
                    for img in item:
                        processed.append(self._process_image(img))
                else:
                    processed.append(self._process_image(item))

        pixel_values = torch.stack(processed) if processed else torch.empty(0, 3, self.image_size, self.image_size)
        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": pixel_values,
        }
