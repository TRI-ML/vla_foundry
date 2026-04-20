"""Tests for Qwen-style gradient accumulation slicing in batch handlers.

Qwen VLMs return pixel_values as flat (total_patches, patch_dim) tensors
instead of (B, ...). The slice_inputs_for_accumulation method must use
image_grid_thw to correctly partition patches across microbatches.
"""

import torch

from vla_foundry.models.batch_handlers import (
    DiffusionPolicyBatchHandler,
    TransformerBatchHandler,
    VLMBatchHandler,
)


def _make_qwen_style_inputs(batch_size, images_per_sample, grid_h=16, grid_w=16, patch_dim=1536):
    """Create mock Qwen-style model inputs with flat pixel_values.

    Args:
        batch_size: Number of samples in the batch.
        images_per_sample: Number of images per sample.
        grid_h: Height of the patch grid per image.
        grid_w: Width of the patch grid per image.
        patch_dim: Dimension of each patch embedding.

    Returns:
        dict of model inputs mimicking Qwen processor output.
    """
    num_images = batch_size * images_per_sample
    patches_per_image = grid_h * grid_w
    total_patches = num_images * patches_per_image

    # Use sequential values so we can verify correct slicing
    pixel_values = torch.arange(total_patches * patch_dim, dtype=torch.float32).view(total_patches, patch_dim)
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]] * num_images, dtype=torch.long)

    seq_len = 128
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def _make_qwen_style_inputs_variable_resolution(batch_size, grids_per_sample):
    """Create Qwen-style inputs where images have different resolutions.

    Args:
        batch_size: Number of samples.
        grids_per_sample: List of (h, w) tuples for each image within a sample.
            All samples get the same image configuration.

    Returns:
        dict of model inputs.
    """
    grid_rows = []
    total_patches = 0
    for _ in range(batch_size):
        for h, w in grids_per_sample:
            grid_rows.append([1, h, w])
            total_patches += h * w

    patch_dim = 1536
    pixel_values = torch.arange(total_patches * patch_dim, dtype=torch.float32).view(total_patches, patch_dim)
    image_grid_thw = torch.tensor(grid_rows, dtype=torch.long)

    return {
        "input_ids": torch.randint(0, 1000, (batch_size, 128)),
        "attention_mask": torch.ones(batch_size, 128, dtype=torch.bool),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


class TestQwenSlicing:
    """Test slice_inputs_for_accumulation with Qwen-style flat pixel_values."""

    def test_qwen_pixel_values_sliced_correctly(self):
        """Slicing batch_size=4 into microbatches of 2 should halve pixel_values."""
        handler = TransformerBatchHandler()
        inputs = _make_qwen_style_inputs(batch_size=4, images_per_sample=3, grid_h=16, grid_w=16)

        patches_per_image = 16 * 16  # 256
        total_patches = 4 * 3 * patches_per_image  # 3072

        assert inputs["pixel_values"].shape[0] == total_patches

        # Slice first half: samples 0-1
        sliced = handler.slice_inputs_for_accumulation(inputs, 0, 2)
        assert sliced["input_ids"].shape[0] == 2
        assert sliced["pixel_values"].shape[0] == 2 * 3 * patches_per_image  # 1536
        assert sliced["image_grid_thw"].shape[0] == 2 * 3  # 6 images

        # Slice second half: samples 2-3
        sliced2 = handler.slice_inputs_for_accumulation(inputs, 2, 4)
        assert sliced2["pixel_values"].shape[0] == 2 * 3 * patches_per_image
        assert sliced2["image_grid_thw"].shape[0] == 2 * 3

        # Combined slices should reconstruct original pixel_values
        reconstructed = torch.cat([sliced["pixel_values"], sliced2["pixel_values"]], dim=0)
        assert torch.equal(reconstructed, inputs["pixel_values"])

    def test_qwen_slicing_with_variable_resolution(self):
        """Images with different patch counts should be sliced correctly."""
        handler = TransformerBatchHandler()
        # 2 samples, each with 3 images of different sizes
        grids = [(8, 8), (16, 16), (12, 8)]  # 64, 256, 96 patches per image
        inputs = _make_qwen_style_inputs_variable_resolution(batch_size=2, grids_per_sample=grids)

        patches_sample_0 = 64 + 256 + 96  # 416
        patches_sample_1 = 64 + 256 + 96  # 416
        assert inputs["pixel_values"].shape[0] == patches_sample_0 + patches_sample_1

        # Slice sample 0
        sliced0 = handler.slice_inputs_for_accumulation(inputs, 0, 1)
        assert sliced0["pixel_values"].shape[0] == patches_sample_0
        assert sliced0["image_grid_thw"].shape[0] == 3
        # Verify it's the first 416 patches
        assert torch.equal(sliced0["pixel_values"], inputs["pixel_values"][:patches_sample_0])

        # Slice sample 1
        sliced1 = handler.slice_inputs_for_accumulation(inputs, 1, 2)
        assert sliced1["pixel_values"].shape[0] == patches_sample_1
        assert sliced1["image_grid_thw"].shape[0] == 3
        assert torch.equal(sliced1["pixel_values"], inputs["pixel_values"][patches_sample_0:])

    def test_qwen_slicing_preserves_grid_values(self):
        """image_grid_thw values should match the sliced samples."""
        handler = TransformerBatchHandler()
        grids = [(8, 8), (16, 16)]
        inputs = _make_qwen_style_inputs_variable_resolution(batch_size=3, grids_per_sample=grids)

        # Slice middle sample (index 1)
        sliced = handler.slice_inputs_for_accumulation(inputs, 1, 2)
        assert sliced["image_grid_thw"].shape == (2, 3)
        assert sliced["image_grid_thw"][0].tolist() == [1, 8, 8]
        assert sliced["image_grid_thw"][1].tolist() == [1, 16, 16]

    def test_standard_pixel_values_not_affected(self):
        """Standard (B, N, C, H, W) pixel_values should be sliced normally."""
        handler = TransformerBatchHandler()
        batch_size = 4
        inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 32)),
            "attention_mask": torch.ones(batch_size, 32, dtype=torch.bool),
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
        }

        sliced = handler.slice_inputs_for_accumulation(inputs, 1, 3)
        assert sliced["pixel_values"].shape == (2, 3, 224, 224)
        assert torch.equal(sliced["pixel_values"], inputs["pixel_values"][1:3])

    def test_standard_pixel_values_without_image_grid_thw(self):
        """Standard pixel_values without image_grid_thw should use the simple batch-slice path."""
        handler = TransformerBatchHandler()
        batch_size = 4
        inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 32)),
            "pixel_values": torch.randn(batch_size, 3, 224, 224),
        }

        sliced = handler.slice_inputs_for_accumulation(inputs, 0, 2)
        assert sliced["pixel_values"].shape == (2, 3, 224, 224)
        assert torch.equal(sliced["pixel_values"], inputs["pixel_values"][:2])

    def test_qwen_slicing_single_sample(self):
        """Slicing a single sample should return all its patches."""
        handler = TransformerBatchHandler()
        inputs = _make_qwen_style_inputs(batch_size=1, images_per_sample=6, grid_h=16, grid_w=16)

        sliced = handler.slice_inputs_for_accumulation(inputs, 0, 1)
        assert torch.equal(sliced["pixel_values"], inputs["pixel_values"])
        assert torch.equal(sliced["image_grid_thw"], inputs["image_grid_thw"])


class TestQwenSlicingDiffusionPolicy:
    """Test that DiffusionPolicyBatchHandler inherits Qwen slicing correctly."""

    def test_diffusion_policy_qwen_slicing(self):
        """DiffusionPolicy should handle Qwen pixel_values via inherited base class."""
        handler = DiffusionPolicyBatchHandler()
        batch_size = 4
        images_per_sample = 12
        grid_h, grid_w = 16, 16
        patches_per_image = grid_h * grid_w
        action_dim = 20
        total_timesteps = 16

        inputs = _make_qwen_style_inputs(batch_size, images_per_sample, grid_h, grid_w)
        # Add DiffusionPolicy-specific keys
        inputs["actions"] = torch.randn(batch_size, total_timesteps, action_dim)
        inputs["noise"] = torch.randn(batch_size, total_timesteps, action_dim)
        inputs["past_mask"] = torch.ones(batch_size, total_timesteps, dtype=torch.bool)
        inputs["future_mask"] = torch.ones(batch_size, total_timesteps, dtype=torch.bool)

        sliced = handler.slice_inputs_for_accumulation(inputs, 0, 2)

        # Qwen-style pixel_values should be correctly sliced
        expected_patches = 2 * images_per_sample * patches_per_image
        assert sliced["pixel_values"].shape[0] == expected_patches
        assert sliced["image_grid_thw"].shape[0] == 2 * images_per_sample

        # Standard batch-indexed tensors should also be sliced
        assert sliced["actions"].shape[0] == 2
        assert sliced["input_ids"].shape[0] == 2


class TestQwenSlicingVLM:
    """Test VLMBatchHandler with Qwen slicing."""

    def test_vlm_qwen_slicing(self):
        """VLMBatchHandler should handle Qwen pixel_values via inherited base class."""
        handler = VLMBatchHandler()
        inputs = _make_qwen_style_inputs(batch_size=4, images_per_sample=2, grid_h=8, grid_w=8)

        sliced = handler.slice_inputs_for_accumulation(inputs, 1, 3)

        patches_per_image = 8 * 8
        assert sliced["pixel_values"].shape[0] == 2 * 2 * patches_per_image  # 2 samples * 2 images * 64
        assert sliced["image_grid_thw"].shape[0] == 2 * 2
        assert sliced["input_ids"].shape[0] == 2
