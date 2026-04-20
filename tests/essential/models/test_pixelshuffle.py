import pytest
import torch

from vla_foundry.models.vlm import ModalityProjector


# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
def original_pixel_shuffle(x, scale_factor=1):
    bsz, seq, embed_dim = x.size()  # x shape [bsz, 16*16, embed_dim]
    seq_root = int(seq**0.5)
    assert seq_root**2 == seq  # Sequence length must be a perfect square for pixel shuffle
    assert seq_root % scale_factor == 0  # Sequence root must be divisible by scale factor

    height = width = seq_root
    x = x.view(bsz, height, width, embed_dim)  # [bsz, 16, 16, embed_dim]
    h_out = height // scale_factor
    w_out = width // scale_factor

    x = x.reshape(bsz, h_out, scale_factor, w_out, scale_factor, embed_dim)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(bsz, h_out * w_out, embed_dim * scale_factor**2)

    return x


class MockVitConfigs:
    def __init__(self, hidden_dim, projector_pixel_shuffle_factor):
        self.hidden_dim = hidden_dim
        self.projector_pixel_shuffle_factor = projector_pixel_shuffle_factor


def test_pixel_shuffle_equivalence():
    """Test that both pixel shuffle implementations produce identical results."""

    # Test parameters
    batch_size = 2
    seq_length = 16 * 16  # 256 (16x16 grid)
    embed_dim = 64
    scale_factor = 2

    # Create test input
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Test original implementation
    result_original = original_pixel_shuffle(x, scale_factor)

    # Test VLM implementation
    mock_configs = MockVitConfigs(embed_dim, scale_factor)
    projector = ModalityProjector(mock_configs, output_dim=128)  # output_dim doesn't matter for this test
    result_vlm = projector.pixel_shuffle(x)

    # Verify results are identical
    assert torch.allclose(result_original, result_vlm, atol=1e-6), "Pixel shuffle results should be identical"

    # Verify output shapes
    expected_seq_length = seq_length // (scale_factor**2)
    expected_embed_dim = embed_dim * (scale_factor**2)
    assert result_original.shape == (batch_size, expected_seq_length, expected_embed_dim)
    assert result_vlm.shape == (batch_size, expected_seq_length, expected_embed_dim)


def test_pixel_shuffle_different_scale_factors():
    """Test pixel shuffle with different scale factors."""

    batch_size = 1
    seq_length = 64 * 64  # 4096 (64x64 grid)
    embed_dim = 32

    x = torch.randn(batch_size, seq_length, embed_dim)

    for scale_factor in [1, 2, 4, 8]:
        # Test original implementation
        result_original = original_pixel_shuffle(x, scale_factor)

        # Test VLM implementation
        mock_configs = MockVitConfigs(embed_dim, scale_factor)
        projector = ModalityProjector(mock_configs, output_dim=128)
        result_vlm = projector.pixel_shuffle(x)

        # Verify results are identical
        assert torch.allclose(result_original, result_vlm, atol=1e-6), (
            f"Results should be identical for scale_factor={scale_factor}"
        )

        # Verify output shapes
        expected_seq_length = seq_length // (scale_factor**2)
        expected_embed_dim = embed_dim * (scale_factor**2)
        assert result_original.shape == (batch_size, expected_seq_length, expected_embed_dim)
        assert result_vlm.shape == (batch_size, expected_seq_length, expected_embed_dim)


def test_pixel_shuffle_4d_input():
    """Test pixel shuffle with 4D input (batch, cameras, seq, embed_dim)."""

    batch_size = 2
    num_cameras = 3
    seq_length = 16 * 16  # 256 (16x16 grid)
    embed_dim = 48
    scale_factor = 2

    x = torch.randn(batch_size, num_cameras, seq_length, embed_dim)

    # Test VLM implementation with 4D input
    mock_configs = MockVitConfigs(embed_dim, scale_factor)
    projector = ModalityProjector(mock_configs, output_dim=128)
    result_vlm = projector.pixel_shuffle(x)

    # Verify output shape
    # The VLM implementation keeps batch_size as n and reshapes (cams w h) together
    expected_seq_length = (seq_length // (scale_factor**2)) * num_cameras
    expected_embed_dim = embed_dim * (scale_factor**2)
    expected_batch_size = batch_size  # Not multiplied by cameras
    assert result_vlm.shape == (expected_batch_size, expected_seq_length, expected_embed_dim)


def test_pixel_shuffle_edge_cases():
    """Test pixel shuffle with edge cases."""

    # Test with scale_factor = 1 (no change)
    batch_size = 1
    seq_length = 4 * 4  # 16 (4x4 grid)
    embed_dim = 8
    scale_factor = 1

    x = torch.randn(batch_size, seq_length, embed_dim)

    result_original = original_pixel_shuffle(x, scale_factor)
    mock_configs = MockVitConfigs(embed_dim, scale_factor)
    projector = ModalityProjector(mock_configs, output_dim=128)
    result_vlm = projector.pixel_shuffle(x)

    # With scale_factor = 1, output should be identical to input
    assert torch.allclose(x, result_original, atol=1e-6)
    assert torch.allclose(x, result_vlm, atol=1e-6)

    # Test with larger grid
    seq_length = 32 * 32  # 1024 (32x32 grid)
    embed_dim = 16
    scale_factor = 4

    x = torch.randn(batch_size, seq_length, embed_dim)

    result_original = original_pixel_shuffle(x, scale_factor)
    mock_configs = MockVitConfigs(embed_dim, scale_factor)
    projector = ModalityProjector(mock_configs, output_dim=128)
    result_vlm = projector.pixel_shuffle(x)

    assert torch.allclose(result_original, result_vlm, atol=1e-6)


def test_pixel_shuffle_invalid_inputs():
    """Test that pixel shuffle fails gracefully with invalid inputs."""

    # Test with non-square sequence length
    batch_size = 1
    seq_length = 255  # Not a perfect square
    embed_dim = 64
    scale_factor = 2

    x = torch.randn(batch_size, seq_length, embed_dim)

    # Original implementation should fail
    with pytest.raises(AssertionError):
        original_pixel_shuffle(x, scale_factor)

    # VLM implementation should fail
    mock_configs = MockVitConfigs(embed_dim, scale_factor)
    projector = ModalityProjector(mock_configs, output_dim=128)
    with pytest.raises(ValueError, match="perfect-square sequence length"):
        projector.pixel_shuffle(x)

    # Test with sequence length not divisible by scale factor
    seq_length = 16 * 16  # 256 (16x16 grid)
    scale_factor = 3  # 16 is not divisible by 3

    x = torch.randn(batch_size, seq_length, embed_dim)

    # Original implementation should fail
    with pytest.raises(AssertionError):
        original_pixel_shuffle(x, scale_factor)

    # VLM implementation should fail
    mock_configs = MockVitConfigs(embed_dim, scale_factor)
    projector = ModalityProjector(mock_configs, output_dim=128)
    with pytest.raises(ValueError, match="divisible by"):
        projector.pixel_shuffle(x)


def test_pixel_shuffle_comprehensive_equivalence():
    """Comprehensive test to ensure both implementations produce identical results."""

    # Test various configurations
    test_configs = [
        (1, 4 * 4, 16, 1),  # Small grid, no scaling
        (2, 8 * 8, 32, 2),  # Medium grid, 2x scaling
        (3, 16 * 16, 64, 4),  # Large grid, 4x scaling
        (4, 32 * 32, 128, 8),  # Very large grid, 8x scaling
    ]

    for batch_size, seq_length, embed_dim, scale_factor in test_configs:
        x = torch.randn(batch_size, seq_length, embed_dim)

        # Test original implementation
        result_original = original_pixel_shuffle(x, scale_factor)

        # Test VLM implementation
        mock_configs = MockVitConfigs(embed_dim, scale_factor)
        projector = ModalityProjector(mock_configs, output_dim=256)
        result_vlm = projector.pixel_shuffle(x)

        # Verify results are identical
        assert torch.allclose(result_original, result_vlm, atol=1e-6), (
            f"Results should be identical for config: batch_size={batch_size}, "
            f"seq_length={seq_length}, embed_dim={embed_dim}, scale_factor={scale_factor}"
        )

        # Verify output shapes
        expected_seq_length = seq_length // (scale_factor**2)
        expected_embed_dim = embed_dim * (scale_factor**2)
        assert result_original.shape == (batch_size, expected_seq_length, expected_embed_dim)
        assert result_vlm.shape == (batch_size, expected_seq_length, expected_embed_dim)


def test_pixel_shuffle_numerical_stability():
    """Test numerical stability and precision of both implementations."""

    batch_size = 1
    seq_length = 16 * 16
    embed_dim = 64
    scale_factor = 2

    # Test with very small values
    x_small = torch.randn(batch_size, seq_length, embed_dim) * 1e-8
    result_original_small = original_pixel_shuffle(x_small, scale_factor)
    mock_configs = MockVitConfigs(embed_dim, scale_factor)
    projector = ModalityProjector(mock_configs, output_dim=128)
    result_vlm_small = projector.pixel_shuffle(x_small)

    assert torch.allclose(result_original_small, result_vlm_small, atol=1e-15)

    # Test with very large values
    x_large = torch.randn(batch_size, seq_length, embed_dim) * 1e8
    result_original_large = original_pixel_shuffle(x_large, scale_factor)
    result_vlm_large = projector.pixel_shuffle(x_large)

    assert torch.allclose(result_original_large, result_vlm_large, atol=1e-6)

    # Test with mixed positive/negative values
    x_mixed = torch.randn(batch_size, seq_length, embed_dim)
    result_original_mixed = original_pixel_shuffle(x_mixed, scale_factor)
    result_vlm_mixed = projector.pixel_shuffle(x_mixed)

    assert torch.allclose(result_original_mixed, result_vlm_mixed, atol=1e-6)
