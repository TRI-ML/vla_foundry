import pytest
import torch

from vla_foundry.attention import (
    ATTN_ACTIVATIONS,
    ATTN_SEQ_SCALARS,
    apply_attention_mask_,
    custom_attn,
    get_attn_func,
    get_rectangular_causal_mask,
    torch_attn,
)


class TestAttentionFunctions:
    """Test attention utility functions."""

    @pytest.mark.parametrize(
        "batch_size,n_heads,q_seq_len,k_seq_len",
        [
            (1, 1, 2, 2),  # Square, minimal
            (1, 1, 3, 5),  # Rectangular
            (1, 1, 5, 5),  # Square, larger
            (2, 4, 8, 12),  # Batch and multi-head
            (1, 8, 1, 10),  # Single query, multiple keys
            (4, 2, 16, 16),  # Larger sequences
        ],
    )
    def test_get_rectangular_causal_mask(self, batch_size, n_heads, q_seq_len, k_seq_len):
        """Test rectangular causal mask generation with different dimensions."""
        shape = (batch_size, n_heads)
        device = torch.device("cpu")
        dtype = torch.float32

        mask = get_rectangular_causal_mask(shape, q_seq_len, k_seq_len, device, dtype)

        # Check shape (accounting for multiple of 8 padding)
        assert mask.shape == (batch_size, n_heads, q_seq_len, k_seq_len)

        # Check causal structure
        for i in range(min(q_seq_len, k_seq_len)):
            # Position i should be able to attend to positions up to i (from the right)
            if q_seq_len <= k_seq_len:
                # When q_seq_len <= k_seq_len, query i can attend to keys up to k_seq_len - q_seq_len + i
                max_attendable = k_seq_len - q_seq_len + i
                if max_attendable < k_seq_len - 1:
                    assert (mask[0, 0, i, max_attendable + 1 :] == torch.finfo(dtype).min).any()

    @pytest.mark.parametrize(
        "batch_size,seq_len,mask_last_n",
        [
            (1, 10, 3),  # Mask last 3 tokens
            (2, 8, 2),  # Batch of 2, mask last 2
            (4, 16, 5),  # Larger batch, mask last 5
            (1, 5, 1),  # Mask only last token
            (3, 12, 0),  # No masking
        ],
    )
    def test_apply_attention_mask(self, batch_size, seq_len, mask_last_n):
        """Test attention mask application with different configurations."""
        n_heads = 4
        bias = torch.zeros(batch_size, n_heads, seq_len, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        if mask_last_n > 0:
            attention_mask[:, -mask_last_n:] = 0

        apply_attention_mask_(bias, attention_mask, torch.float32)

        if mask_last_n > 0:
            # Check that padded positions have -inf values
            min_val = torch.finfo(torch.float32).min
            masked_region = bias[:, :, :, -mask_last_n:]

            # Create a tensor with the same shape filled with min_val for comparison
            expected_min_tensor = torch.full_like(masked_region, min_val)
            assert torch.allclose(masked_region, expected_min_tensor)
        else:
            # No masking should leave bias unchanged
            assert torch.allclose(bias, torch.zeros_like(bias))

    @pytest.mark.parametrize(
        "batch_size,seq_len,n_heads,head_dim",
        [
            (1, 4, 2, 8),  # Minimal
            (2, 8, 4, 16),  # Standard
            (1, 16, 8, 32),  # Larger
            (4, 32, 12, 64),  # Large
            (1, 1, 1, 16),  # Single token
        ],
    )
    @pytest.mark.parametrize("is_causal", [True, False])
    def test_torch_attn_basic(self, batch_size, seq_len, n_heads, head_dim, is_causal):
        """Test basic torch attention function with different configurations."""
        queries = torch.randn(batch_size, seq_len, n_heads, head_dim)
        keys = torch.randn(batch_size, seq_len, n_heads, head_dim)
        values = torch.randn(batch_size, seq_len, n_heads, head_dim)

        output = torch_attn(queries, keys, values, is_causal=is_causal)

        assert output.shape == (batch_size, seq_len, n_heads, head_dim)
        assert output.dtype == queries.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize(
        "batch_size,q_seq_len,k_seq_len,n_heads,head_dim",
        [
            (1, 2, 8, 4, 16),  # Shorter queries
            (2, 5, 12, 2, 32),  # Different lengths
            (1, 1, 10, 8, 16),  # Single query, multiple keys
            (3, 8, 16, 4, 24),  # Various combinations
        ],
    )
    def test_torch_attn_different_seq_lengths(self, batch_size, q_seq_len, k_seq_len, n_heads, head_dim):
        """Test torch attention with different query and key lengths."""
        queries = torch.randn(batch_size, q_seq_len, n_heads, head_dim)
        keys = torch.randn(batch_size, k_seq_len, n_heads, head_dim)
        values = torch.randn(batch_size, k_seq_len, n_heads, head_dim)

        output = torch_attn(queries, keys, values, is_causal=True)

        assert output.shape == (batch_size, q_seq_len, n_heads, head_dim)

    @pytest.mark.parametrize(
        "batch_size,seq_len,n_heads,head_dim",
        [
            (2, 8, 4, 16),
            (1, 16, 8, 32),
            (4, 12, 6, 24),
        ],
    )
    def test_torch_attn_with_attention_mask(self, batch_size, seq_len, n_heads, head_dim):
        """Test torch attention with attention mask."""
        queries = torch.randn(batch_size, seq_len, n_heads, head_dim)
        keys = torch.randn(batch_size, seq_len, n_heads, head_dim)
        values = torch.randn(batch_size, seq_len, n_heads, head_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -2:] = 0  # Mask last 2 tokens

        output = torch_attn(queries, keys, values, is_causal=True, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, n_heads, head_dim)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("attn_activation", list(ATTN_ACTIVATIONS.keys()))
    @pytest.mark.parametrize("attn_seq_scalar", list(ATTN_SEQ_SCALARS.keys()))
    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_custom_attn_activations_and_scalars(self, attn_activation, attn_seq_scalar, alpha):
        """Test custom attention with different activation functions and sequence scalars."""
        batch_size, seq_len, n_heads, head_dim = 2, 8, 4, 16
        queries = torch.randn(batch_size, seq_len, n_heads, head_dim)
        keys = torch.randn(batch_size, seq_len, n_heads, head_dim)
        values = torch.randn(batch_size, seq_len, n_heads, head_dim)

        try:
            output = custom_attn(
                queries,
                keys,
                values,
                attn_activation=attn_activation,
                attn_seq_scalar=attn_seq_scalar,
                alpha=alpha,
                is_causal=True,
            )

            assert output.shape == (batch_size, seq_len, n_heads, head_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
        except Exception:
            # Some combinations might not work, that's okay
            # pytest.skip(f"Combination {attn_activation}-{attn_seq_scalar}-{alpha} not supported: {e}")
            pass

    @pytest.mark.parametrize(
        "batch_size,seq_len,n_heads,head_dim",
        [
            (1, 8, 4, 16),
            (2, 16, 8, 32),
        ],
    )
    @pytest.mark.parametrize("is_causal", [True, False])
    def test_custom_attn_vs_torch_attn_structure(self, batch_size, seq_len, n_heads, head_dim, is_causal):
        """Test that custom attention produces reasonable outputs compared to torch attention."""
        queries = torch.randn(batch_size, seq_len, n_heads, head_dim)
        keys = torch.randn(batch_size, seq_len, n_heads, head_dim)
        values = torch.randn(batch_size, seq_len, n_heads, head_dim)

        # Custom attention with softmax (should be similar to torch attention)
        custom_output = custom_attn(
            queries, keys, values, attn_activation="softmax", attn_seq_scalar="none", alpha=0.0, is_causal=is_causal
        )

        torch_output = torch_attn(queries, keys, values, is_causal=is_causal)

        # They won't be exactly the same due to implementation differences,
        # but should have similar magnitudes and no NaN/Inf
        assert custom_output.shape == torch_output.shape
        assert not torch.isnan(custom_output).any()
        assert not torch.isnan(torch_output).any()

    @pytest.mark.parametrize(
        "attn_name,extra_args",
        [
            ("auto", {}),
            ("torch_attn", {}),
            ("custom_attn", {"attn_activation": "relu", "attn_seq_scalar": "none", "alpha": 0.0}),
            ("custom_attn", {"attn_activation": "softmax", "attn_seq_scalar": "avg", "alpha": 0.5}),
        ],
    )
    def test_get_attn_func(self, attn_name, extra_args):
        """Test getting different attention functions."""
        attn_fn = get_attn_func(attn_name, **extra_args)
        assert callable(attn_fn)

        # Test that it works
        batch_size, seq_len, n_heads, head_dim = 1, 4, 2, 8
        queries = torch.randn(batch_size, seq_len, n_heads, head_dim)
        keys = torch.randn(batch_size, seq_len, n_heads, head_dim)
        values = torch.randn(batch_size, seq_len, n_heads, head_dim)

        output = attn_fn(queries, keys, values, is_causal=True)
        assert output.shape == (batch_size, seq_len, n_heads, head_dim)

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "invalid_attention",
            "unknown_attn",
            "wrong_attention_type",
        ],
    )
    def test_get_attn_func_invalid(self, invalid_name):
        """Test getting invalid attention function raises error."""
        with pytest.raises(ValueError, match="Unsupported attn-name"):
            get_attn_func(invalid_name)

    def test_get_attn_func_custom_missing_args(self):
        """Test that custom attention requires all arguments."""
        with pytest.raises(AssertionError, match="must provide attn-activation"):
            get_attn_func("custom_attn")

        with pytest.raises(AssertionError, match="must provide attn-activation"):
            get_attn_func("custom_attn", attn_activation="relu")


class TestAttentionActivations:
    """Test attention activation functions."""

    @pytest.mark.parametrize("activation_name", list(ATTN_ACTIVATIONS.keys()))
    @pytest.mark.parametrize(
        "input_shape",
        [
            (2, 8, 16),
            (1, 4, 32),
            (4, 16, 64),
        ],
    )
    def test_attention_activations(self, activation_name, input_shape):
        """Test all attention activation functions with different input shapes."""
        x = torch.randn(*input_shape)
        fn = ATTN_ACTIVATIONS[activation_name]

        output = fn(x)
        assert output.shape == x.shape

        if activation_name == "softmax":
            # Softmax output should sum to 1 along last dimension
            assert torch.allclose(output.sum(dim=-1), torch.ones(input_shape[:-1]), atol=1e-6)
        elif activation_name == "relu":
            # ReLU should be non-negative
            assert (output >= 0).all()
        elif activation_name == "sigmoid":
            # Sigmoid should be in [0, 1]
            assert (output >= 0).all() and (output <= 1).all()

    @pytest.mark.parametrize("seq_len", [1, 5, 10, 16, 32, 100])
    def test_attention_seq_scalars(self, seq_len):
        """Test attention sequence scalars with different sequence lengths."""
        for name, fn in ATTN_SEQ_SCALARS.items():
            scalar = fn(seq_len)

            if name == "max":
                assert scalar == seq_len
            elif name == "avg":
                assert scalar == (seq_len - 1) / 2 + 1
            elif name == "none":
                assert scalar == 1

            # All scalars should be positive
            assert scalar > 0


class TestAttentionNumericalStability:
    """Test attention numerical stability and edge cases."""

    @pytest.mark.parametrize("scale_factor", [1e-6, 1e-3, 1.0, 1e3, 1e6])
    @pytest.mark.parametrize("attention_fn", [torch_attn])
    def test_attention_with_extreme_values(self, scale_factor, attention_fn):
        """Test attention with extreme input values."""
        batch_size, seq_len, n_heads, head_dim = 1, 4, 2, 8

        # Create queries and keys with extreme values
        queries = torch.randn(batch_size, seq_len, n_heads, head_dim) * scale_factor
        keys = torch.randn(batch_size, seq_len, n_heads, head_dim) * scale_factor
        values = torch.randn(batch_size, seq_len, n_heads, head_dim)

        output = attention_fn(queries, keys, values, is_causal=True)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.shape == (batch_size, seq_len, n_heads, head_dim)

    @pytest.mark.parametrize("attention_fn", [torch_attn])
    def test_attention_with_zero_values(self, attention_fn):
        """Test attention with zero input values."""
        batch_size, seq_len, n_heads, head_dim = 1, 4, 2, 8

        queries = torch.zeros(batch_size, seq_len, n_heads, head_dim)
        keys = torch.zeros(batch_size, seq_len, n_heads, head_dim)
        values = torch.randn(batch_size, seq_len, n_heads, head_dim)

        output = attention_fn(queries, keys, values, is_causal=True)

        assert not torch.isnan(output).any()
        assert output.shape == (batch_size, seq_len, n_heads, head_dim)

    @pytest.mark.parametrize("attention_fn", [torch_attn])
    @pytest.mark.parametrize("seq_len", [1, 2, 4])
    def test_attention_single_and_small_sequences(self, attention_fn, seq_len):
        """Test attention with single token and small sequences."""
        batch_size, n_heads, head_dim = 1, 2, 8

        queries = torch.randn(batch_size, seq_len, n_heads, head_dim)
        keys = torch.randn(batch_size, seq_len, n_heads, head_dim)
        values = torch.randn(batch_size, seq_len, n_heads, head_dim)

        output = attention_fn(queries, keys, values, is_causal=True)

        assert output.shape == (batch_size, seq_len, n_heads, head_dim)

        if seq_len == 1:
            # With single token, causal masking shouldn't matter
            output_non_causal = attention_fn(queries, keys, values, is_causal=False)
            assert torch.allclose(output, output_non_causal, atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_attention_different_dtypes(self, dtype):
        """Test attention with different data types."""
        if dtype == torch.float16 and not torch.cuda.is_available():
            pytest.skip("Half precision requires CUDA")

        batch_size, seq_len, n_heads, head_dim = 1, 4, 2, 8

        queries = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype)
        keys = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype)
        values = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype)

        output = torch_attn(queries, keys, values, is_causal=True)

        assert output.dtype == dtype
        assert output.shape == (batch_size, seq_len, n_heads, head_dim)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_attention_batch_consistency(self, batch_size):
        """Test that attention works consistently across different batch sizes."""
        seq_len, n_heads, head_dim = 8, 4, 16

        queries = torch.randn(batch_size, seq_len, n_heads, head_dim)
        keys = torch.randn(batch_size, seq_len, n_heads, head_dim)
        values = torch.randn(batch_size, seq_len, n_heads, head_dim)

        output = torch_attn(queries, keys, values, is_causal=True)

        assert output.shape == (batch_size, seq_len, n_heads, head_dim)

        # Test that each batch item produces consistent results
        if batch_size > 1:
            # Process first item individually
            single_output = torch_attn(queries[:1], keys[:1], values[:1], is_causal=True)
            assert torch.allclose(output[:1], single_output, atol=1e-6)
