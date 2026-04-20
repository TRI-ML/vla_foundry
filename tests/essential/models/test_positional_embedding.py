from dataclasses import dataclass

import pytest
import torch

from vla_foundry.positional_embedding import (
    RotaryEmbedding,
    RotaryWithCast,
    apply_rotary_pos_emb,
    get_pos_embed,
    identity_with_cast,
    rotate_half,
)


@dataclass
class MockModelConfig:
    """Mock model config for testing get_pos_embed."""

    hidden_dim: int
    n_heads: int
    max_seq_len: int
    positional_embedding_type: str


class TestRotaryEmbedding:
    """Test standard rotary embedding implementation."""

    @pytest.mark.parametrize(
        "dim_model,seq_len",
        [
            (64, 128),  # Small
            (128, 512),  # Medium
            (256, 1024),  # Large
            (32, 64),  # Minimal
            (512, 2048),  # Very large
        ],
    )
    def test_rotary_embedding_initialization(self, dim_model, seq_len):
        """Test RotaryEmbedding initialization with different dimensions."""
        rotary = RotaryEmbedding(dim_model, seq_len)

        assert rotary.dim_model == dim_model
        assert rotary.seq_len == seq_len
        assert rotary.inv_freq.shape == (dim_model // 2,)
        assert rotary._seq_len_cached == seq_len
        assert rotary._cos_cached is not None
        assert rotary._sin_cached is not None

    @pytest.mark.parametrize(
        "batch_size,seq_len,n_heads,head_dim",
        [
            (1, 8, 4, 16),  # Small
            (2, 16, 8, 32),  # Medium
            (4, 32, 12, 64),  # Large
            (1, 1, 1, 8),  # Single token
            (8, 128, 16, 128),  # Very large
        ],
    )
    def test_rotary_embedding_forward(self, batch_size, seq_len, n_heads, head_dim):
        """Test RotaryEmbedding forward pass with different tensor shapes."""
        rotary = RotaryEmbedding(head_dim, seq_len * 2)  # Allow for longer sequences

        q = torch.randn(batch_size, seq_len, n_heads, head_dim)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim)

        q_rot, k_rot = rotary(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()

    @pytest.mark.parametrize("offset", [0, 5, 10, 20])
    def test_rotary_embedding_with_offset(self, offset):
        """Test RotaryEmbedding with different offsets."""
        dim_model, seq_len = 64, 128
        rotary = RotaryEmbedding(dim_model, seq_len + offset)

        batch_size, n_heads = 2, 8
        q = torch.randn(batch_size, seq_len, n_heads, dim_model)
        k = torch.randn(batch_size, seq_len, n_heads, dim_model)

        q_rot, k_rot = rotary(q, k, offset=offset)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_rotary_embedding_dtypes(self, dtype):
        """Test RotaryEmbedding with different data types."""
        rotary = RotaryEmbedding(64, 128)

        q = torch.randn(2, 16, 8, 64, dtype=dtype)
        k = torch.randn(2, 16, 8, 64, dtype=dtype)

        q_rot, k_rot = rotary(q, k)

        assert q_rot.dtype == dtype
        assert k_rot.dtype == dtype

    def test_rotary_embedding_cos_sin_caching(self):
        """Test that cos/sin tables are properly cached and updated."""
        rotary = RotaryEmbedding(64, 128)

        # Initial state
        assert rotary._seq_len_cached == 128

        # Forward with same length should use cache
        q = torch.randn(1, 64, 4, 64)
        k = torch.randn(1, 64, 4, 64)
        rotary(q, k)
        assert rotary._seq_len_cached == 128

        # Forward with longer sequence should update cache
        q_long = torch.randn(1, 256, 4, 64)
        k_long = torch.randn(1, 256, 4, 64)
        rotary(q_long, k_long)
        assert rotary._seq_len_cached == 256


class TestRotaryWithCast:
    """Test RotaryWithCast wrapper."""

    @pytest.mark.parametrize(
        "q_dtype,v_dtype",
        [
            (torch.float32, torch.float16),
            (torch.float16, torch.float32),
            (torch.float64, torch.float32),
            (torch.float32, torch.float32),  # Same dtype
        ],
    )
    def test_rotary_with_cast_dtypes(self, q_dtype, v_dtype):
        """Test RotaryWithCast handles dtype casting correctly."""
        rotary = RotaryWithCast(64, 128)

        q = torch.randn(2, 16, 8, 64, dtype=q_dtype)
        k = torch.randn(2, 16, 8, 64, dtype=q_dtype)
        v = torch.randn(2, 16, 8, 64, dtype=v_dtype)

        q_out, k_out, v_out = rotary(q, k, v)

        # Output q and k should match v's dtype
        assert q_out.dtype == v_dtype
        assert k_out.dtype == v_dtype
        assert v_out.dtype == v_dtype

        # v should be unchanged
        assert torch.equal(v, v_out)


class TestIdentityWithCast:
    """Test identity positional embedding (no embedding)."""

    @pytest.mark.parametrize(
        "q_dtype,v_dtype",
        [
            (torch.float32, torch.float16),
            (torch.float16, torch.float32),
            (torch.float64, torch.float16),
            (torch.float32, torch.float32),
        ],
    )
    def test_identity_with_cast_dtypes(self, q_dtype, v_dtype):
        """Test identity_with_cast handles dtype casting."""
        q = torch.randn(2, 16, 8, 64, dtype=q_dtype)
        k = torch.randn(2, 16, 8, 64, dtype=q_dtype)
        v = torch.randn(2, 16, 8, 64, dtype=v_dtype)

        q_out, k_out, v_out = identity_with_cast(q, k, v)

        # Should cast q and k to v's dtype
        assert q_out.dtype == v_dtype
        assert k_out.dtype == v_dtype
        assert v_out.dtype == v_dtype

        # Content should be preserved (just cast)
        assert torch.equal(q_out, q.to(v_dtype))
        assert torch.equal(k_out, k.to(v_dtype))
        assert torch.equal(v_out, v)

    def test_identity_with_cast_offset_ignored(self):
        """Test identity_with_cast ignores offset parameter."""
        q = torch.randn(1, 8, 4, 32)
        k = torch.randn(1, 8, 4, 32)
        v = torch.randn(1, 8, 4, 32)

        q_out1, k_out1, v_out1 = identity_with_cast(q, k, v, offset=0)
        q_out2, k_out2, v_out2 = identity_with_cast(q, k, v, offset=100)

        # Results should be identical regardless of offset
        assert torch.equal(q_out1, q_out2)
        assert torch.equal(k_out1, k_out2)
        assert torch.equal(v_out1, v_out2)


class TestPositionalEmbeddingUtilities:
    """Test utility functions for positional embeddings."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            (2, 8, 16),
            (1, 4, 32),
            (4, 16, 64),
            (8, 32, 128),
        ],
    )
    def test_rotate_half(self, input_shape):
        """Test rotate_half utility function."""
        x = torch.randn(*input_shape)
        rotated = rotate_half(x)

        assert rotated.shape == x.shape

        # Check that the rotation is correct
        x1, x2 = x.chunk(2, dim=-1)
        expected = torch.cat((-x2, x1), dim=-1)
        assert torch.allclose(rotated, expected)

    def test_apply_rotary_pos_emb(self):
        """Test apply_rotary_pos_emb utility function."""
        batch_size, seq_len, n_heads, head_dim = 2, 8, 4, 16
        x = torch.randn(batch_size, seq_len, n_heads, head_dim)

        # Create dummy cos/sin tables
        cos = torch.randn(1, seq_len, 1, head_dim)
        sin = torch.randn(1, seq_len, 1, head_dim)

        result = apply_rotary_pos_emb(x, cos, sin)

        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_apply_rotary_pos_emb_with_offset(self):
        """Test apply_rotary_pos_emb with offset."""
        batch_size, seq_len, n_heads, head_dim = 1, 4, 2, 8
        x = torch.randn(batch_size, seq_len, n_heads, head_dim)

        # Cos/sin tables longer than sequence
        cos = torch.randn(1, seq_len + 10, 1, head_dim)
        sin = torch.randn(1, seq_len + 10, 1, head_dim)

        result = apply_rotary_pos_emb(x, cos, sin, offset=5)

        assert result.shape == x.shape

    def test_apply_rotary_pos_emb_assertion_error(self):
        """Test apply_rotary_pos_emb raises assertion for insufficient cos/sin length."""
        x = torch.randn(1, 10, 2, 8)
        cos = torch.randn(1, 5, 1, 8)  # Too short
        sin = torch.randn(1, 5, 1, 8)

        with pytest.raises(AssertionError, match="Offset and/or input sequence is too large"):
            apply_rotary_pos_emb(x, cos, sin, offset=0)


class TestGetPosEmbed:
    """Test the get_pos_embed factory function."""

    @pytest.mark.parametrize(
        "pos_type,expected_class",
        [
            ("rotary", RotaryWithCast),
        ],
    )
    def test_get_pos_embed_types(self, pos_type, expected_class):
        """Test get_pos_embed returns correct positional embedding types."""
        config = MockModelConfig(hidden_dim=512, n_heads=8, max_seq_len=128, positional_embedding_type=pos_type)

        pos_embed = get_pos_embed(config)
        # Check class name instead of isinstance due to import path issues
        assert pos_embed.__class__.__name__ == expected_class.__name__

    def test_get_pos_embed_none_type(self):
        """Test get_pos_embed returns identity function for 'none' type."""
        config = MockModelConfig(hidden_dim=512, n_heads=8, max_seq_len=128, positional_embedding_type="none")

        pos_embed = get_pos_embed(config)
        # Check function name instead of direct comparison
        assert pos_embed.__name__ == "identity_with_cast"

    def test_get_pos_embed_invalid_type(self):
        """Test get_pos_embed raises error for invalid type."""
        config = MockModelConfig(hidden_dim=512, n_heads=8, max_seq_len=128, positional_embedding_type="invalid_type")

        with pytest.raises(RuntimeError, match="Unknown positional embedding type"):
            get_pos_embed(config)

    @pytest.mark.parametrize(
        "hidden_dim,n_heads",
        [
            (512, 8),  # head_dim = 64
            (768, 12),  # head_dim = 64
            (1024, 16),  # head_dim = 64
            (256, 4),  # head_dim = 64
        ],
    )
    def test_get_pos_embed_head_dim_calculation(self, hidden_dim, n_heads):
        """Test that get_pos_embed correctly calculates head_dim."""
        config = MockModelConfig(
            hidden_dim=hidden_dim, n_heads=n_heads, max_seq_len=128, positional_embedding_type="rotary"
        )

        pos_embed = get_pos_embed(config)
        expected_head_dim = hidden_dim // n_heads
        assert pos_embed.dim_model == expected_head_dim


class TestPositionalEmbeddingNumericalStability:
    """Test numerical stability of positional embeddings."""

    @pytest.mark.parametrize("scale_factor", [1e-6, 1e-3, 1.0, 1e3, 1e6])
    @pytest.mark.parametrize(
        "embedding_class",
        [
            RotaryWithCast,
        ],
    )
    def test_positional_embedding_extreme_values(self, scale_factor, embedding_class):
        """Test positional embeddings with extreme input values."""
        pos_embed = embedding_class(64, 128)

        # Create inputs with extreme values
        q = torch.randn(2, 16, 8, 64) * scale_factor
        k = torch.randn(2, 16, 8, 64) * scale_factor
        v = torch.randn(2, 16, 8, 64)

        q_out, k_out, v_out = pos_embed(q, k, v)

        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()
        assert not torch.isnan(v_out).any()
        assert not torch.isinf(q_out).any()
        assert not torch.isinf(k_out).any()
        assert not torch.isinf(v_out).any()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_positional_embedding_different_dtypes(self, dtype):
        """Test positional embeddings with different data types."""
        pos_embed = RotaryWithCast(64, 128)

        q = torch.randn(1, 8, 4, 64, dtype=dtype)
        k = torch.randn(1, 8, 4, 64, dtype=dtype)
        v = torch.randn(1, 8, 4, 64, dtype=dtype)

        q_out, k_out, v_out = pos_embed(q, k, v)

        assert q_out.dtype == dtype
        assert k_out.dtype == dtype
        assert v_out.dtype == dtype

    def test_positional_embedding_single_token(self):
        """Test positional embeddings with single token sequences."""
        pos_embed = RotaryWithCast(64, 128)

        q = torch.randn(1, 1, 8, 64)  # Single token
        k = torch.randn(1, 1, 8, 64)
        v = torch.randn(1, 1, 8, 64)

        q_out, k_out, v_out = pos_embed(q, k, v)

        assert q_out.shape == (1, 1, 8, 64)
        assert k_out.shape == (1, 1, 8, 64)
        assert v_out.shape == (1, 1, 8, 64)
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()


class TestPositionalEmbeddingComparison:
    """Test comparisons between different positional embedding implementations."""

    def test_identity_vs_rotary_difference(self):
        """Test that identity embedding differs from rotary embedding."""
        q = torch.randn(1, 8, 4, 32)
        k = torch.randn(1, 8, 4, 32)
        v = torch.randn(1, 8, 4, 32)

        # Identity embedding
        q_id, k_id, v_id = identity_with_cast(q, k, v)

        # Rotary embedding
        rotary = RotaryWithCast(32, 16)
        q_rot, k_rot, v_rot = rotary(q, k, v)

        # Identity should preserve q and k (after dtype cast)
        assert torch.allclose(q_id, q.to(v.dtype))
        assert torch.allclose(k_id, k.to(v.dtype))

        # Rotary should modify q and k
        assert not torch.allclose(q_rot, q.to(v.dtype))
        assert not torch.allclose(k_rot, k.to(v.dtype))

        # Both should preserve v
        assert torch.equal(v_id, v)
        assert torch.equal(v_rot, v)
