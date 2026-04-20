import torch
from PIL import Image

from vla_foundry.data.processor import get_processor
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml


class TestProcessorMultiImage:
    """Test processor handling of multiple images per sample using non-gated models."""

    def test_clip_processor_multi_batch_multi_image(self):
        """Test CLIP processor with multiple batches and multiple images per sample."""
        params = load_experiment_params_from_yaml("tests/essential/params/dummy_configs/dummy_vlm_config.yaml")
        object.__setattr__(params.data, "processor", "openai/clip-vit-base-patch32")
        clip_processor = get_processor(params.data)

        # Load test images
        image1 = Image.open("tests/essential/shared/chonky_cat.png")
        image2 = Image.open("tests/essential/shared/chonky_cat.png")
        if image1.mode == "RGBA":
            image1 = image1.convert("RGB")
        if image2.mode == "RGBA":
            image2 = image2.convert("RGB")

        batch_size = 3
        num_images_per_sample = 2

        # Test with multiple batches, each with multiple images
        result = clip_processor(
            images=[
                [image1, image2],
                [image1, image2],
                [image1, image2],
            ],  # 3 samples, 2 images each
            text=[
                "What is in these images?",
                "Describe these images.",
                "What do you see?",
            ],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
        )

        # Test pixel_values shape for multi-batch multi-image
        assert "pixel_values" in result
        # CLIP returns [B*N, C, H, W] for multiple images
        assert result["pixel_values"].dim() == 4, "pixel_values should be 4D for multi-image"
        assert result["pixel_values"].shape[0] == batch_size * num_images_per_sample, (
            f"Should have {batch_size * num_images_per_sample} images (B*N)"
        )
        assert result["pixel_values"].shape[1] == 3, "Should have 3 color channels"
        # CLIP uses 224x224 images
        assert result["pixel_values"].shape[2] == 224, "Height should be 224"
        assert result["pixel_values"].shape[3] == 224, "Width should be 224"

        # Test input_ids shape
        assert "input_ids" in result
        assert result["input_ids"].shape[0] == batch_size, f"Should have {batch_size} samples"

    def test_clip_processor_single_image(self):
        """Test CLIP processor with a single image and text."""
        params = load_experiment_params_from_yaml("tests/essential/params/dummy_configs/dummy_vlm_config.yaml")
        object.__setattr__(params.data, "processor", "openai/clip-vit-base-patch32")
        clip_processor = get_processor(params.data)

        image = Image.open("tests/essential/shared/chonky_cat.png")
        if image.mode == "RGBA":
            image = image.convert("RGB")

        result = clip_processor(
            images=[image],
            text=["What is in this image?"],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
        )

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "pixel_values" in result
        assert result["input_ids"].dim() == 2
        assert result["input_ids"].shape[0] == 1
        assert result["input_ids"].dtype == torch.long
        assert result["pixel_values"].dim() == 4
        assert result["pixel_values"].shape == (1, 3, 224, 224)
        assert result["pixel_values"].dtype == torch.float32
