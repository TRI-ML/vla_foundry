import logging
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from vla_foundry.data.processor.stable_diffusion_processor import StableDiffusionProcessor
from vla_foundry.data.utils import text_to_seed
from vla_foundry.params.base_data_params import DataParams


class PassthroughProcessor:
    """Converts images to tensors without normalization. Optionally resizes to ``image_size``."""

    def __init__(self, image_size: int | None = None):
        self.image_token_id = 0
        self.tokenizer = SimpleNamespace(pad_token_id=0, chat_template=None)
        self.chat_template = None
        self.image_size = image_size

    def __call__(self, images, text, return_tensors="pt", padding=True, **kwargs):
        batch_size = len(text)
        if images is not None:
            pixel_values = []
            for sample_images in images:
                for img in sample_images:
                    if isinstance(img, torch.Tensor):
                        # Already a tensor from the new torchvision decoder — skip PIL conversion
                        if self.image_size is not None:
                            img = torch.nn.functional.interpolate(
                                img.unsqueeze(0).float(), size=(self.image_size, self.image_size), mode="bilinear"
                            ).squeeze(0)
                        t = img.float()
                        if t.ndim == 3 and t.shape[0] not in (1, 3, 4):
                            t = t.permute(2, 0, 1)  # HWC -> CHW
                        pixel_values.append(t)
                        continue
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    if self.image_size is not None:
                        img = img.resize((self.image_size, self.image_size))
                    t = torch.as_tensor(np.array(img), dtype=torch.float32)
                    if t.ndim == 3:
                        t = t.permute(2, 0, 1)  # HWC -> CHW
                    pixel_values.append(t)
            pixel_values = torch.stack(pixel_values)
        else:
            pixel_values = torch.empty(0)
        return {
            "input_ids": torch.zeros(batch_size, 1, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 1, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class DebugProcessor:
    def __init__(self):
        self.image_token_id = 0
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    def __call__(
        self, images, text, return_tensors="pt", padding="max_length", padding_side="right", max_length=2048, **kwargs
    ):
        batch_size = len(images)
        seed = text_to_seed(text[0])
        return {
            "input_ids": torch.randint(0, 100, (batch_size, max_length), generator=torch.Generator().manual_seed(seed)),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
            "pixel_values": torch.randn(batch_size, 3, 224, 224, generator=torch.Generator().manual_seed(seed)),
        }


def apply_chat_template(processor, num_images, text):
    """Format text with image placeholders using the processor's chat template.

    Uses the chat template (from processor or tokenizer) when available so that
    model-specific image tokens (e.g. Qwen's <|vision_start|>/<|image_pad|>)
    are inserted correctly.  Falls back to a plain ``<image>`` prefix for
    processors without a chat template (e.g. PaliGemma).
    """
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]

    if getattr(processor, "chat_template", None):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    elif hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "chat_template", None):
        return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        image_tokens = "<image> " * num_images
        return image_tokens + text


def get_processor(data_params: DataParams):
    if data_params.processor == "stable_diffusion":
        return StableDiffusionProcessor(
            image_size=data_params.image_size,
            max_length=data_params.seq_len,
        )
    elif data_params.processor == "debug":
        return DebugProcessor()
    elif data_params.processor == "simple_vlm":
        from vla_foundry.models.vision_language_backbones.simple_vlm_processor import SimpleVLMProcessor

        processor = SimpleVLMProcessor(
            image_size=data_params.image_size,
        )
        processor.image_seq_length = data_params.img_num_tokens
        return processor
    elif data_params.processor == "none":
        return PassthroughProcessor(image_size=getattr(data_params, "image_size", None))
    elif data_params.processor is not None:
        # When using the rust-based fast tokenizer, each process spawns #cpu rayon threads
        # by default. With many GPUs and workers this causes resource contention.
        # Only override env vars when the fast tokenizer is explicitly enabled to avoid
        # clobbering user/job-level settings.
        if data_params.use_hf_fast_tokenizer:
            if data_params.hf_fast_tokenizer_rayon_threads is not None:
                os.environ.setdefault("RAYON_NUM_THREADS", str(data_params.hf_fast_tokenizer_rayon_threads))
            os.environ.setdefault(
                "TOKENIZERS_PARALLELISM", "true" if data_params.hf_fast_tokenizers_parallelism else "false"
            )
        processor = AutoProcessor.from_pretrained(data_params.processor, use_fast=data_params.use_hf_fast_tokenizer)
        # PaliGemma uses ``image_seq_length`` to control how many image token
        # placeholders the processor inserts per image.
        if (
            data_params.img_num_tokens
            and hasattr(processor, "image_seq_length")
            and "paligemma" in data_params.processor.lower()
        ):
            processor.image_seq_length = data_params.img_num_tokens
        return processor
    else:
        raise ValueError(f"{data_params.processor} not yet supported.")
