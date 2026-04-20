from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vla_foundry.losses import (
    CrossEntropyLossWithZLoss,
    _ignore_mask,
    get_loss_function,
    masked_mse_loss,
)


class TestCrossEntropyLossWithZLoss:
    """Test the CrossEntropyLossWithZLoss class."""

    def test_initialization(self):
        """Test that the loss function initializes correctly."""
        loss_fn = CrossEntropyLossWithZLoss(eps=1e-3)
        assert loss_fn.eps == 1e-3

        # Test with all parameters
        loss_fn = CrossEntropyLossWithZLoss(eps=1e-2, ignore_index=-100, reduction="sum", label_smoothing=0.1)
        assert loss_fn.eps == 1e-2
        assert loss_fn.ignore_index == -100
        assert loss_fn.reduction == "sum"
        assert loss_fn.label_smoothing == 0.1

    def test_forward_basic(self):
        """Test basic forward pass functionality."""
        batch_size, vocab_size = 2, 10
        loss_fn = CrossEntropyLossWithZLoss(eps=1e-4)

        # Create test inputs
        logits = torch.randn(batch_size, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch_size,))

        # Compute loss
        loss = loss_fn(logits, targets)

        # Should be a scalar tensor
        assert loss.dim() == 0
        assert loss.requires_grad

        # Should be positive
        assert loss.item() > 0

    def test_z_loss_component(self):
        """Test that z-loss component is added correctly."""
        batch_size, vocab_size = 2, 10
        eps = 1e-2

        # Create identical inputs for comparison
        logits = torch.randn(batch_size, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size,))

        # Regular cross entropy loss
        regular_loss = F.cross_entropy(logits, targets)

        # Cross entropy with z-loss
        z_loss_fn = CrossEntropyLossWithZLoss(eps=eps)
        combined_loss = z_loss_fn(logits, targets)

        # Z-loss component should be positive
        z_component = eps * torch.square(torch.logsumexp(logits, dim=-1)).mean()
        expected_loss = regular_loss + z_component

        assert torch.allclose(combined_loss, expected_loss, atol=1e-6)
        assert combined_loss > regular_loss

    def test_different_reductions(self):
        """Test different reduction modes."""
        batch_size, vocab_size = 3, 5
        logits = torch.randn(batch_size, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size,))

        # Test mean reduction (default)
        loss_mean = CrossEntropyLossWithZLoss(eps=1e-4, reduction="mean")
        loss_val_mean = loss_mean(logits, targets)
        assert loss_val_mean.dim() == 0

        # Test sum reduction
        loss_sum = CrossEntropyLossWithZLoss(eps=1e-4, reduction="sum")
        loss_val_sum = loss_sum(logits, targets)
        assert loss_val_sum.dim() == 0
        assert loss_val_sum > loss_val_mean  # Sum should be larger than mean

        # Test none reduction
        loss_none = CrossEntropyLossWithZLoss(eps=1e-4, reduction="none")
        loss_val_none = loss_none(logits, targets)
        assert loss_val_none.shape == (batch_size,)

    def test_ignore_index(self):
        """Test ignore_index functionality."""
        batch_size, vocab_size = 3, 5
        logits = torch.randn(batch_size, vocab_size)
        targets = torch.tensor([0, 1, -100])  # Last target should be ignored

        loss_fn = CrossEntropyLossWithZLoss(eps=1e-4, ignore_index=-100)
        loss = loss_fn(logits, targets)

        # Should still compute successfully
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestMaskedMSELoss:
    """Test the masked_mse_loss function."""

    def test_no_mask(self):
        """Test masked MSE loss without mask (should behave like regular MSE)."""
        batch_size, seq_len, dim = 2, 5, 3
        predicted = torch.randn(batch_size, seq_len, dim)
        target = torch.randn(batch_size, seq_len, dim)

        # Without mask
        loss_masked = masked_mse_loss(predicted, target, mask=None)
        loss_regular = F.mse_loss(predicted, target)

        assert torch.allclose(loss_masked, loss_regular)

    def test_with_mask(self):
        """Test masked MSE loss with mask."""
        batch_size, seq_len, dim = 2, 5, 3
        predicted = torch.randn(batch_size, seq_len, dim)
        target = torch.randn(batch_size, seq_len, dim)

        # Create mask (1 for valid, 0 for invalid)
        mask = torch.ones(batch_size, seq_len)
        mask[0, -2:] = 0  # Mask out last 2 positions for first batch item
        mask[1, -1] = 0  # Mask out last position for second batch item

        loss = masked_mse_loss(predicted, target, mask=mask)

        # Should be a scalar
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_all_masked_out(self):
        """Test when all elements are masked out."""
        batch_size, seq_len, dim = 2, 5, 3
        predicted = torch.randn(batch_size, seq_len, dim)
        target = torch.randn(batch_size, seq_len, dim)

        # All zeros mask
        mask = torch.zeros(batch_size, seq_len)

        loss = masked_mse_loss(predicted, target, mask=mask)

        # Should be very small due to epsilon in denominator
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert loss.item() < 1e-6  # Should be close to zero

    def test_gradient_flow(self):
        """Test that gradients flow correctly through masked loss."""
        batch_size, seq_len, dim = 2, 5, 3
        predicted = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        target = torch.randn(batch_size, seq_len, dim)
        mask = torch.ones(batch_size, seq_len)
        mask[0, -1] = 0  # Mask out one element

        loss = masked_mse_loss(predicted, target, mask=mask)
        loss.backward()

        # Gradients should exist
        assert predicted.grad is not None
        assert predicted.grad.shape == predicted.shape

        # Masked positions should have zero gradients
        assert torch.allclose(predicted.grad[0, -1, :], torch.zeros(dim))


class TestIgnoreMaskWrapper:
    """Test the _ignore_mask wrapper function."""

    def test_wrapper_ignores_mask(self):
        """Test that the wrapper correctly ignores mask parameter."""

        def dummy_loss(x, y):
            return F.mse_loss(x, y)

        wrapped_loss = _ignore_mask(dummy_loss)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        mask = torch.ones(2, 3)

        # Should work with mask parameter
        loss_with_mask = wrapped_loss(x, y, mask=mask)
        loss_without_mask = dummy_loss(x, y)

        assert torch.allclose(loss_with_mask, loss_without_mask)

    def test_wrapper_preserves_other_args(self):
        """Test that wrapper preserves other arguments and kwargs."""

        def dummy_loss(x, y, reduction="mean"):
            return F.mse_loss(x, y, reduction=reduction)

        wrapped_loss = _ignore_mask(dummy_loss)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        mask = torch.ones(2, 3)

        loss = wrapped_loss(x, y, mask=mask, reduction="sum")
        expected = dummy_loss(x, y, reduction="sum")

        assert torch.allclose(loss, expected)


class TestGetLossFunction:
    """Test the get_loss_function factory function."""

    def test_cross_entropy_without_z_loss(self):
        """Test cross entropy loss creation without z-loss."""
        # Mock hparams
        hparams = Mock()
        hparams.z_loss_coefficient = 0.0

        loss_fn = get_loss_function("cross_entropy", hparams)

        # Test that it works
        logits = torch.randn(2, 5)
        targets = torch.randint(0, 5, (2,))
        mask = torch.ones(2)

        loss = loss_fn(logits, targets, mask=mask)
        assert loss.dim() == 0

    def test_cross_entropy_with_z_loss(self):
        """Test cross entropy loss creation with z-loss."""
        hparams = Mock()
        hparams.z_loss_coefficient = 1e-3

        loss_fn = get_loss_function("cross_entropy", hparams)

        # Test that it works
        logits = torch.randn(2, 5)
        targets = torch.randint(0, 5, (2,))
        mask = torch.ones(2)

        loss = loss_fn(logits, targets, mask=mask)
        assert loss.dim() == 0

    def test_mse_loss(self):
        """Test MSE loss creation."""
        hparams = Mock()

        loss_fn = get_loss_function("mse", hparams)

        # Test that it works
        predicted = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)

        loss = loss_fn(predicted, target)
        assert loss > 0
        assert loss.dim() == 0

    def test_masked_mse_loss(self):
        """Test masked MSE loss creation."""
        hparams = Mock()

        loss_fn = get_loss_function("mse", hparams)

        # Test that it works
        predicted = torch.randn(2, 3, 4)
        target = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3)

        loss = loss_fn(predicted, target, mask=mask)
        loss_mse = get_loss_function("mse", hparams)(predicted, target)
        # Here the mask is all ones, so the masked MSE loss should be the same as the regular MSE loss
        assert loss_mse == loss
        assert loss.dim() == 0

        mask = torch.zeros(2, 3)
        loss = loss_fn(predicted, target, mask=mask)
        assert loss == 0

        mask = torch.ones(2, 3)
        mask[0, -1] = 0
        loss_masked = loss_fn(predicted, target, mask=mask)
        assert loss_masked > 0
        assert loss_masked != loss_mse

    def test_unsupported_loss_type(self):
        """Test that unsupported loss types raise ValueError."""
        hparams = Mock()

        with pytest.raises(ValueError, match="Loss function unsupported_loss not supported"):
            get_loss_function("unsupported_loss", hparams)

    def test_all_loss_functions_accept_mask(self):
        """Test that all returned loss functions accept mask parameter."""
        hparams = Mock()
        hparams.z_loss_coefficient = 0.0

        loss_types = ["cross_entropy", "mse"]

        for loss_type in loss_types:
            loss_fn = get_loss_function(loss_type, hparams)

            # Create appropriate test data
            if loss_type == "cross_entropy":
                inputs = torch.randn(2, 3, 5)
                targets = torch.randint(0, 5, (2, 3))
                mask = torch.ones(2, 3)
            else:
                inputs = torch.randn(2, 3, 4)
                targets = torch.randn(2, 3, 4)
                mask = torch.ones(2, 3)

            # Should not raise an error
            loss = loss_fn(inputs, targets, mask=mask)
            assert loss.dim() == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_tensors(self):
        """Test behavior with empty tensors."""
        # Test CrossEntropyLossWithZLoss with empty tensors
        loss_fn = CrossEntropyLossWithZLoss(eps=1e-4)

        # Empty tensors should still work (though may not be meaningful)
        empty_logits = torch.empty(0, 5)
        empty_targets = torch.empty(0, dtype=torch.long)

        loss = loss_fn(empty_logits, empty_targets)
        assert loss.dim() == 0

    def test_single_element_tensors(self):
        """Test behavior with single element tensors."""
        # Single element test
        logits = torch.randn(1, 1)
        targets = torch.tensor([0])

        loss_fn = CrossEntropyLossWithZLoss(eps=1e-4)
        loss = loss_fn(logits, targets)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large logits
        large_logits = torch.tensor([[100.0, -100.0, 50.0]])
        targets = torch.tensor([0])

        loss_fn = CrossEntropyLossWithZLoss(eps=1e-4)
        loss = loss_fn(large_logits, targets)

        assert torch.isfinite(loss)
        assert loss.item() >= 0
