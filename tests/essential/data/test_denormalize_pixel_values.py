"""Tests for RoboticsProcessor.denormalize_first_sample_images."""

from types import SimpleNamespace

import numpy as np
import torch

from vla_foundry.data.processor.robotics_processor import RoboticsProcessor


def _make_processor_with_mock_vlm(image_mean, image_std, patch_size=None, temporal_patch_size=None, merge_size=None):
    """Create a RoboticsProcessor with a mock VLM processor for testing denormalization."""
    processor = object.__new__(RoboticsProcessor)
    ip_kwargs = dict(image_mean=image_mean, image_std=image_std)
    if patch_size is not None:
        ip_kwargs.update(patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size)
    processor.vlm_processor = SimpleNamespace(image_processor=SimpleNamespace(**ip_kwargs))
    return processor


class TestDenormalizeFirstSampleImages:
    def test_roundtrip(self):
        """Normalizing then denormalizing should approximately recover the original image."""
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        processor = _make_processor_with_mock_vlm(mean, std)

        # Create a random "original" image in [0, 1] float — (B=1, N=1, C, H, W)
        original = torch.rand(3, 224, 224)
        pixel_values = original.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 224, 224)

        # Normalize (simulating what the VLM processor does)
        mean_t = torch.tensor(mean).view(1, 1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 1, 3, 1, 1)
        normalized = (pixel_values - mean_t) / std_t

        result = processor.denormalize_first_sample_images(normalized)

        assert len(result) == 1
        assert result[0].shape == (224, 224, 3)
        assert result[0].dtype == np.uint8

        # Convert original to uint8 for comparison
        original_uint8 = (original.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        assert np.abs(result[0].astype(int) - original_uint8.astype(int)).max() <= 1

    def test_multiple_images_in_first_sample(self):
        """Should return all images from the first sample in (B, N, C, H, W) input."""
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        processor = _make_processor_with_mock_vlm(mean, std)

        # batch=2, num_images=4 — should only return 4 images from first sample
        pixel_values = torch.randn(2, 4, 3, 64, 64)
        result = processor.denormalize_first_sample_images(pixel_values)

        assert len(result) == 4
        assert all(img.shape == (64, 64, 3) for img in result)
        assert all(img.dtype == np.uint8 for img in result)

    def test_output_range(self):
        """Output should be clamped to [0, 255]."""
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        processor = _make_processor_with_mock_vlm(mean, std)

        # Use extreme values — (B=1, N=1, C=3, H=1, W=1)
        pixel_values = torch.tensor([[[[[10.0]], [[-10.0]], [[0.0]]]]]).float()
        result = processor.denormalize_first_sample_images(pixel_values)

        assert len(result) == 1
        assert result[0].min() >= 0
        assert result[0].max() <= 255

    def test_known_values(self):
        """Test with known normalization values to verify exact computation."""
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        processor = _make_processor_with_mock_vlm(mean, std)

        # With mean=0, std=1: denorm(x) = x, then clamp(0,1)*255
        # 0.5 -> 0.5 * 255 = 127.5 -> 127
        pixel_values = torch.full((1, 1, 3, 1, 1), 0.5)
        result = processor.denormalize_first_sample_images(pixel_values)
        assert result[0][0, 0, 0] == 127  # 0.5 * 255 = 127.5, truncated to 127

        # 1.0 -> 255
        pixel_values = torch.full((1, 1, 3, 1, 1), 1.0)
        result = processor.denormalize_first_sample_images(pixel_values)
        assert result[0][0, 0, 0] == 255

        # -0.5 -> clamped to 0
        pixel_values = torch.full((1, 1, 3, 1, 1), -0.5)
        result = processor.denormalize_first_sample_images(pixel_values)
        assert result[0][0, 0, 0] == 0


class TestDenormalizeQwenPixelValues:
    """Tests for Qwen-style flat patch denormalization."""

    @staticmethod
    def _qwen_forward(images_chw, patch_size, temporal_patch_size, merge_size):
        """Simulate Qwen's image processor patch flattening for a single image.

        Args:
            images_chw: Tensor of shape (num_frames, C, H, W) with normalized values.

        Returns:
            (flatten_patches, image_grid_thw) matching Qwen's output format.
        """
        patches = images_chw  # (num_frames, C, H, W)
        num_frames, channel, h, w = patches.shape
        # Pad temporal dim to multiple of temporal_patch_size
        if num_frames % temporal_patch_size != 0:
            pad = temporal_patch_size - (num_frames % temporal_patch_size)
            patches = torch.cat([patches, patches[-1:].expand(pad, -1, -1, -1)], dim=0)

        grid_t = patches.shape[0] // temporal_patch_size
        grid_h = h // patch_size
        grid_w = w // patch_size

        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])
        return flatten_patches, image_grid_thw

    def test_roundtrip_single_image(self):
        """Normalizing, Qwen-flattening, then denormalizing should recover the original."""
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        patch_size, temporal_patch_size, merge_size = 14, 2, 2
        processor = _make_processor_with_mock_vlm(mean, std, patch_size, temporal_patch_size, merge_size)

        # Image: 1 frame, 3 channels, 224x224 (divisible by patch_size * merge_size = 28)
        original = torch.rand(1, 3, 224, 224)

        # Normalize
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        normalized = (original - mean_t) / std_t

        flat, grid_thw = self._qwen_forward(normalized, patch_size, temporal_patch_size, merge_size)
        result = processor.denormalize_first_sample_images(flat, image_grid_thw=grid_thw)

        # temporal_patch_size=2 but only 1 real frame -> padded to 2 frames in output
        assert len(result) == 2
        assert result[0].shape == (224, 224, 3)
        assert result[0].dtype == np.uint8

        original_uint8 = (original[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        # First frame should match original
        assert np.abs(result[0].astype(int) - original_uint8.astype(int)).max() <= 1

    def test_multiple_images(self):
        """Should handle multiple images concatenated in the flat patch tensor."""
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        patch_size, temporal_patch_size, merge_size = 14, 2, 2
        processor = _make_processor_with_mock_vlm(mean, std, patch_size, temporal_patch_size, merge_size)

        img1 = torch.rand(1, 3, 56, 56)
        img2 = torch.rand(1, 3, 112, 112)

        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)

        flat1, grid1 = self._qwen_forward((img1 - mean_t) / std_t, patch_size, temporal_patch_size, merge_size)
        flat2, grid2 = self._qwen_forward((img2 - mean_t) / std_t, patch_size, temporal_patch_size, merge_size)

        flat = torch.cat([flat1, flat2], dim=0)
        grid_thw = torch.cat([grid1, grid2], dim=0)

        result = processor.denormalize_first_sample_images(flat, image_grid_thw=grid_thw)

        # 2 images, each padded to temporal_patch_size=2 frames -> 4 total frames
        assert len(result) == 4
        assert result[0].dtype == np.uint8
        assert result[0].shape == (56, 56, 3)
        assert result[2].shape == (112, 112, 3)

    def test_qwen3_patch_size(self):
        """Should work with Qwen3's patch_size=16."""
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        patch_size, temporal_patch_size, merge_size = 16, 2, 2
        processor = _make_processor_with_mock_vlm(mean, std, patch_size, temporal_patch_size, merge_size)

        original = torch.rand(1, 3, 128, 128)
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        normalized = (original - mean_t) / std_t

        flat, grid_thw = self._qwen_forward(normalized, patch_size, temporal_patch_size, merge_size)
        result = processor.denormalize_first_sample_images(flat, image_grid_thw=grid_thw)

        assert len(result) == 2
        assert result[0].shape == (128, 128, 3)
        original_uint8 = (original[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        assert np.abs(result[0].astype(int) - original_uint8.astype(int)).max() <= 1
