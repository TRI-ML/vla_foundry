import pytest
import torch

from vla_foundry.norms import LayerNorm, LPLayerNorm, RmsNorm, get_norm_class


class TestLayerNorm:
    """Test custom LayerNorm implementation."""

    @pytest.mark.parametrize(
        "normalized_shape,batch_size,seq_len",
        [
            (256, 2, 10),
            (512, 1, 5),
            (128, 4, 20),
            (64, 8, 15),
            ([32, 4], 2, 10),  # Multi-dimensional shape
            ([16, 8, 2], 1, 5),  # 3D normalized shape
        ],
    )
    def test_layernorm_basic(self, normalized_shape, batch_size, seq_len):
        """Test basic LayerNorm functionality with different shapes."""
        ln = LayerNorm(normalized_shape)

        if isinstance(normalized_shape, list):
            input_shape = [batch_size, seq_len] + normalized_shape
        else:
            input_shape = [batch_size, seq_len, normalized_shape]

        x = torch.randn(*input_shape)
        output = ln(x)

        assert output.shape == x.shape

        # For normalization properties, we need to check the right dimensions
        norm_dims = list(range(-len(normalized_shape), 0)) if isinstance(normalized_shape, list) else [-1]

        # Check normalization properties
        assert torch.allclose(
            output.mean(dim=norm_dims, keepdim=True),
            torch.zeros_like(output.mean(dim=norm_dims, keepdim=True)),
            atol=1e-5,
        )
        assert torch.allclose(
            output.std(dim=norm_dims, keepdim=True), torch.ones_like(output.std(dim=norm_dims, keepdim=True)), atol=1e-2
        )

    @pytest.mark.parametrize(
        "elementwise_gain,elementwise_bias",
        [
            (True, True),  # Default - both weight and bias
            (True, False),  # Only weight (gain), no bias
            (False, True),  # Only bias, no weight (gain)
            (False, False),  # Neither weight nor bias
        ],
    )
    def test_layernorm_weight_bias_combinations(self, elementwise_gain, elementwise_bias):
        """Test LayerNorm with different weight/bias combinations."""
        normalized_shape = 256
        ln = LayerNorm(normalized_shape, elementwise_gain=elementwise_gain, elementwise_bias=elementwise_bias)

        if elementwise_gain:
            assert ln.weight is not None
            assert ln.weight.shape == (normalized_shape,)
        else:
            assert ln.weight is None

        if elementwise_bias:
            assert ln.bias is not None
            assert ln.bias.shape == (normalized_shape,)
        else:
            assert ln.bias is None

        # Test forward pass works regardless of configuration
        x = torch.randn(2, 10, normalized_shape)
        output = ln(x)
        assert output.shape == x.shape

    @pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-8, 1e-12])
    def test_layernorm_eps_parameter(self, eps):
        """Test LayerNorm eps parameter with different values."""
        normalized_shape = 256
        ln = LayerNorm(normalized_shape, eps=eps)

        assert ln.eps == eps

        # Test with very small variance (where eps matters)
        x = torch.ones(2, 10, normalized_shape) * 1e-6
        output = ln(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestLPLayerNorm:
    """Test Low Precision LayerNorm implementation."""

    @pytest.mark.parametrize(
        "normalized_shape,dtype",
        [
            (256, torch.float32),
            (512, torch.float32),
            (128, torch.float16),  # Test with half precision
        ],
    )
    def test_lp_layernorm_basic(self, normalized_shape, dtype):
        """Test basic LPLayerNorm functionality with different shapes and dtypes."""
        ln = LPLayerNorm(normalized_shape)

        x = torch.randn(2, 10, normalized_shape, dtype=dtype)
        output = ln(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    @pytest.mark.parametrize(
        "elementwise_gain,elementwise_bias",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_lp_layernorm_inheritance(self, elementwise_gain, elementwise_bias):
        """Test that LPLayerNorm inherits LayerNorm properties."""
        assert issubclass(LPLayerNorm, LayerNorm)

        normalized_shape = 256
        ln = LPLayerNorm(normalized_shape, elementwise_gain=elementwise_gain, elementwise_bias=elementwise_bias)

        if elementwise_gain:
            assert ln.weight is not None
        else:
            assert ln.weight is None

        if elementwise_bias:
            assert ln.bias is not None
        else:
            assert ln.bias is None


class TestRmsNorm:
    """Test RMS Normalization implementation."""

    @pytest.mark.parametrize(
        "normalized_shape,batch_size,seq_len",
        [
            (256, 2, 10),
            (512, 1, 5),
            (128, 4, 20),
            ([64, 4], 2, 10),  # Multi-dimensional shape
        ],
    )
    def test_rmsnorm_basic(self, normalized_shape, batch_size, seq_len):
        """Test basic RmsNorm functionality with different shapes."""
        rms = RmsNorm(normalized_shape)

        if isinstance(normalized_shape, list):
            input_shape = [batch_size, seq_len] + normalized_shape
        else:
            input_shape = [batch_size, seq_len, normalized_shape]

        x = torch.randn(*input_shape)
        output = rms(x)

        assert output.shape == x.shape
        # RMS norm should preserve the mean structure but normalize variance
        assert not torch.allclose(output, x)  # Should change the input

    @pytest.mark.parametrize("eps", [1e-6, 1e-8, 1e-12])
    def test_rmsnorm_eps_parameter(self, eps):
        """Test RmsNorm eps parameter with different values."""
        normalized_shape = 256
        rms = RmsNorm(normalized_shape, eps=eps)

        assert rms.eps == eps

        # Test with very small values (where eps matters)
        x = torch.ones(2, 10, normalized_shape) * 1e-8
        output = rms(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize(
        "normalized_shape",
        [
            256,
            512,
            [64, 4],
            [32, 8, 2],
        ],
    )
    def test_rmsnorm_weight_initialization(self, normalized_shape):
        """Test RmsNorm weight initialization with different shapes."""
        rms = RmsNorm(normalized_shape)

        expected_shape = tuple(normalized_shape) if isinstance(normalized_shape, list) else (normalized_shape,)

        # Weight should be initialized to 1.0
        assert rms.weight.shape == expected_shape
        assert torch.allclose(rms.weight, torch.ones(expected_shape))


class TestNormClassFactory:
    """Test norm class factory function."""

    @pytest.mark.parametrize(
        "norm_type,expected_class",
        [
            ("default_layer_norm", torch.nn.LayerNorm),
            ("lp_layer_norm", LPLayerNorm),
            ("rms_norm", RmsNorm),
        ],
    )
    def test_get_norm_classes(self, norm_type, expected_class):
        """Test getting different norm classes."""
        norm_class = get_norm_class(norm_type)
        assert norm_class == expected_class

    @pytest.mark.parametrize(
        "norm_type,has_weight,has_bias",
        [
            ("gain_only_layer_norm", True, False),
            ("gain_only_lp_layer_norm", True, False),
            ("no_wb_layer_norm", False, False),
        ],
    )
    def test_get_partial_norm_classes(self, norm_type, has_weight, has_bias):
        """Test getting partial norm classes with specific weight/bias settings."""
        norm_class = get_norm_class(norm_type)

        # Should be a partial function or configured class
        norm = norm_class(256)

        if has_weight:
            assert hasattr(norm, "weight") and norm.weight is not None
        else:
            assert norm.weight is None

        if has_bias:
            assert hasattr(norm, "bias") and norm.bias is not None
        else:
            assert norm.bias is None

    @pytest.mark.parametrize(
        "invalid_norm",
        [
            "unsupported_norm",
            "invalid_layernorm",
            "wrong_norm_type",
        ],
    )
    def test_get_unsupported_norm(self, invalid_norm):
        """Test that unsupported norm types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported model-norm"):
            get_norm_class(invalid_norm)


class TestNormNumericalStability:
    """Test normalization layers numerical stability."""

    @pytest.mark.parametrize(
        "norm_class,normalized_shape",
        [
            (LayerNorm, 256),
            (LPLayerNorm, 256),
            (RmsNorm, 256),
        ],
    )
    @pytest.mark.parametrize("scale_factor", [1e-6, 1e-3, 1.0, 1e3, 1e6])
    def test_norm_extreme_values(self, norm_class, normalized_shape, scale_factor):
        """Test normalization layers with extreme input values."""
        norm = norm_class(normalized_shape)

        x = torch.randn(2, 10, normalized_shape) * scale_factor
        output = norm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.shape == x.shape

    @pytest.mark.parametrize("norm_class", [LayerNorm, LPLayerNorm, RmsNorm])
    def test_norm_zero_input(self, norm_class):
        """Test normalization layers with zero input."""
        normalized_shape = 256
        norm = norm_class(normalized_shape)

        x = torch.zeros(2, 10, normalized_shape)
        output = norm(x)

        assert not torch.isnan(output).any()
        assert output.shape == x.shape

    @pytest.mark.parametrize("norm_class", [LayerNorm, LPLayerNorm, RmsNorm])
    def test_norm_constant_input(self, norm_class):
        """Test normalization layers with constant input."""
        normalized_shape = 256
        norm = norm_class(normalized_shape)

        # Constant input (should have zero variance)
        x = torch.ones(2, 10, normalized_shape) * 5.0
        output = norm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.shape == x.shape
