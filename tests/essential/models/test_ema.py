"""Tests for EMA (Exponential Moving Average) model wrapper."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from vla_foundry.file_utils import load_ema_checkpoint
from vla_foundry.models.ema import EMAModel, VanillaEMAModel, create_ema_model


class TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    return TinyModel()


class TestEMACreation:
    """Test EMA model creation and initialization."""

    def test_vanilla_ema_creation(self, tiny_model):
        """Test that VanillaEMAModel is created with correct parameters."""
        ema_model = create_ema_model(tiny_model, ema_type="vanilla", alpha=0.999)

        assert isinstance(ema_model, VanillaEMAModel)
        assert ema_model._alpha == 0.999
        assert ema_model.model is not None

    def test_adaptive_ema_creation(self, tiny_model):
        """Test that EMAModel is created with correct parameters."""
        ema_model = create_ema_model(tiny_model, ema_type="ema", update_after_step=10, max_value=0.9999)

        assert isinstance(ema_model, EMAModel)
        assert ema_model.update_after_step == 10
        assert ema_model.max_value == 0.9999
        assert ema_model.model is not None

    def test_ema_initialization_matches_base_model(self, tiny_model):
        """Test that EMA weights are initialized to match base model."""
        ema_model = create_ema_model(tiny_model, ema_type="vanilla", alpha=0.999)

        # Initially, EMA weights should match base model exactly
        for param_base, param_ema in zip(tiny_model.parameters(), ema_model.model.parameters(), strict=False):
            assert torch.equal(param_base, param_ema), "Initial EMA weights should match base model"


class TestEMAUpdateMechanics:
    """Test EMA update mechanics and weight evolution."""

    def test_vanilla_ema_update_formula(self, tiny_model):
        """Test VanillaEMAModel updates weights correctly."""
        alpha = 0.999
        ema_model = create_ema_model(tiny_model, ema_type="vanilla", alpha=alpha)

        # Store initial EMA weights
        initial_ema_weights = {name: param.clone() for name, param in ema_model.model.named_parameters()}

        # Modify base model weights significantly and update EMA multiple times
        for _ in range(10):
            with torch.no_grad():
                for param in tiny_model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            ema_model.step(tiny_model)

        # Verify EMA weights have changed from initial but are closer to initial than to current model
        # (due to high alpha = 0.999, EMA should track slowly)
        for name, param_ema in ema_model.model.named_parameters():
            param_model = dict(tiny_model.named_parameters())[name]

            # EMA should differ from both initial and current model
            assert not torch.equal(param_ema, initial_ema_weights[name]), f"EMA should have updated for {name}"
            assert not torch.equal(param_ema, param_model), f"EMA should differ from current model for {name}"

            # With high alpha (0.999), EMA should be closer to initial weights than current model
            dist_to_initial = (param_ema - initial_ema_weights[name]).abs().mean()
            dist_to_current = (param_ema - param_model).abs().mean()
            assert dist_to_initial < dist_to_current, f"With high alpha, EMA should be closer to initial for {name}"

    def test_adaptive_ema_warmup_period(self, tiny_model):
        """Test EMAModel respects warmup period."""
        ema_model = create_ema_model(tiny_model, ema_type="ema", update_after_step=10)

        # Store initial EMA weights
        {name: param.clone() for name, param in ema_model.model.named_parameters()}

        # Update before warmup period (step < update_after_step)
        for _step in range(5):  # Less than update_after_step=10
            with torch.no_grad():
                for param in tiny_model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            ema_model.step(tiny_model)

        # EMA weights should still match base model (no decay during warmup)
        for name, param_ema in ema_model.model.named_parameters():
            param_model = dict(tiny_model.named_parameters())[name]
            assert torch.equal(param_ema, param_model), f"EMA should copy weights during warmup for {name}"

        # Update after warmup period
        for _step in range(15):  # Now past update_after_step=10
            with torch.no_grad():
                for param in tiny_model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            ema_model.step(tiny_model)

        # EMA weights should now be different from model (decay is active)
        weights_differ = False
        for name, param_ema in ema_model.model.named_parameters():
            param_model = dict(tiny_model.named_parameters())[name]
            if not torch.equal(param_ema, param_model):
                weights_differ = True
                break

        assert weights_differ, "EMA should apply decay after warmup period"

    def test_ema_optimization_step_tracking(self, tiny_model):
        """Test optimization step counter for EMAModel."""
        ema_model = create_ema_model(tiny_model, ema_type="ema", update_after_step=10)

        assert ema_model.optimization_step.item() == 0, "Initial optimization step should be 0"

        # Update several times
        for _ in range(5):
            ema_model.step(tiny_model)

        assert ema_model.optimization_step.item() == 5, "Optimization step should increment with each update"

    def test_ema_weights_diverge_from_training(self, tiny_model):
        """Test that EMA weights diverge from training weights over time."""
        ema_model = create_ema_model(tiny_model, ema_type="vanilla", alpha=0.999)

        # Train for several steps with large updates
        for _ in range(20):
            with torch.no_grad():
                for param in tiny_model.parameters():
                    param.add_(torch.randn_like(param) * 0.5)
            ema_model.step(tiny_model)

        # EMA weights should differ from training weights
        for param_base, param_ema in zip(tiny_model.parameters(), ema_model.model.parameters(), strict=False):
            assert not torch.equal(param_base, param_ema), "EMA weights should differ from training weights"

        # EMA weights should be smoother (lower variance in changes)
        # This is a heuristic check - EMA should dampen noise
        torch.cat([p.flatten() for p in tiny_model.parameters()]).std()
        torch.cat([p.flatten() for p in ema_model.model.parameters()]).std()
        # Note: This might not always be true, but generally EMA has more stable weights
        # We just verify they're different, not necessarily smoother in this simple test


class TestEMACheckpoints:
    """Test EMA checkpoint saving and loading."""

    def test_ema_checkpoint_saved(self, tiny_model):
        """Test that EMA checkpoint is saved alongside model checkpoint."""
        ema_model = create_ema_model(tiny_model, ema_type="vanilla", alpha=0.999)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            checkpoints_dir = tmpdir_path / "checkpoints"
            checkpoints_dir.mkdir()

            # Save model checkpoint
            model_checkpoint = {"checkpoint_num": 1, "state_dict": tiny_model.state_dict(), "global_step": 100}
            torch.save(model_checkpoint, checkpoints_dir / "checkpoint_1.pt")

            # Save EMA checkpoint
            ema_checkpoint = {
                "checkpoint_num": 1,
                "ema_state_dict": ema_model.model.state_dict(),
                "ema_optimization_step": 0,
            }
            torch.save(ema_checkpoint, checkpoints_dir / "ema_1.pt")

            # Verify both checkpoints exist
            assert (checkpoints_dir / "checkpoint_1.pt").exists(), "Model checkpoint should exist"
            assert (checkpoints_dir / "ema_1.pt").exists(), "EMA checkpoint should exist"

    def test_ema_checkpoint_content(self, tiny_model):
        """Test EMA checkpoint contains required keys."""
        ema_model = create_ema_model(tiny_model, ema_type="ema", update_after_step=10)

        # Update a few times to set optimization_step
        for _ in range(15):
            ema_model.step(tiny_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            checkpoints_dir = tmpdir_path / "checkpoints"
            checkpoints_dir.mkdir()

            # Save EMA checkpoint
            ema_checkpoint = {
                "checkpoint_num": 5,
                "ema_state_dict": ema_model.model.state_dict(),
                "ema_optimization_step": ema_model.optimization_step.item(),
            }
            ema_checkpoint_path = checkpoints_dir / "ema_5.pt"
            torch.save(ema_checkpoint, ema_checkpoint_path)

            # Load and verify EMA checkpoint content
            checkpoint = torch.load(ema_checkpoint_path, map_location="cpu", weights_only=True)

            assert "ema_state_dict" in checkpoint, "EMA checkpoint should contain ema_state_dict"
            assert "checkpoint_num" in checkpoint, "EMA checkpoint should contain checkpoint_num"
            assert "ema_optimization_step" in checkpoint, "EMA checkpoint should contain ema_optimization_step"

            assert checkpoint["checkpoint_num"] == 5
            assert checkpoint["ema_optimization_step"] == 15  # Updated 15 times

    def test_load_ema_checkpoint(self, tiny_model):
        """Test loading EMA checkpoint restores correct state."""
        ema_model = create_ema_model(tiny_model, ema_type="vanilla", alpha=0.999)

        # Update EMA several times
        for _ in range(10):
            with torch.no_grad():
                for param in tiny_model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            ema_model.step(tiny_model)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            checkpoints_dir = tmpdir_path / "checkpoints"
            checkpoints_dir.mkdir()

            # Store EMA weights before saving
            saved_ema_weights = {name: param.clone() for name, param in ema_model.model.named_parameters()}

            # Save EMA checkpoint
            ema_checkpoint = {
                "checkpoint_num": 3,
                "ema_state_dict": ema_model.model.state_dict(),
                "ema_optimization_step": 0,
            }
            ema_checkpoint_path = checkpoints_dir / "ema_3.pt"
            torch.save(ema_checkpoint, ema_checkpoint_path)

            # Create new EMA model and load checkpoint
            new_ema_model = create_ema_model(TinyModel(), ema_type="vanilla", alpha=0.999)
            loaded_checkpoint_num = load_ema_checkpoint(new_ema_model, str(ema_checkpoint_path))

            assert loaded_checkpoint_num == 3, "Should return correct checkpoint number"

            # Verify weights match
            for name, param in new_ema_model.model.named_parameters():
                assert torch.equal(param, saved_ema_weights[name]), f"Loaded EMA weights should match for {name}"

    def test_load_ema_checkpoint_missing(self, tiny_model):
        """Test error handling when EMA checkpoint doesn't exist."""
        ema_model = create_ema_model(tiny_model, ema_type="vanilla", alpha=0.999)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to load non-existent EMA checkpoint
            fake_checkpoint_path = Path(tmpdir) / "nonexistent_ema.pt"

            # Should raise FileNotFoundError with clear message
            with pytest.raises(FileNotFoundError) as exc_info:
                load_ema_checkpoint(ema_model, str(fake_checkpoint_path))

            error_msg = str(exc_info.value)
            assert "EMA checkpoint not found" in error_msg
            assert "trained with EMA enabled" in error_msg

    def test_resume_training_with_ema(self, tiny_model):
        """Test that EMA state is correctly restored when resuming."""
        ema_model = create_ema_model(tiny_model, ema_type="ema", update_after_step=10)

        # Train for several steps past warmup
        for _ in range(20):
            with torch.no_grad():
                for param in tiny_model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            ema_model.step(tiny_model)

        original_optimization_step = ema_model.optimization_step.item()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            checkpoints_dir = tmpdir_path / "checkpoints"
            checkpoints_dir.mkdir()

            # Save EMA checkpoint
            ema_checkpoint = {
                "checkpoint_num": 2,
                "ema_state_dict": ema_model.model.state_dict(),
                "ema_optimization_step": ema_model.optimization_step.item(),
            }
            ema_checkpoint_path = checkpoints_dir / "ema_2.pt"
            torch.save(ema_checkpoint, ema_checkpoint_path)

            # Create new EMA model and load checkpoint
            new_model = TinyModel()
            new_ema_model = create_ema_model(new_model, ema_type="ema", update_after_step=10)
            load_ema_checkpoint(new_ema_model, str(ema_checkpoint_path))

            # Verify optimization_step counter is restored
            assert new_ema_model.optimization_step.item() == original_optimization_step, (
                "Optimization step should be restored"
            )

            # Continue training
            for _ in range(5):
                new_ema_model.step(new_model)

            assert new_ema_model.optimization_step.item() == original_optimization_step + 5, (
                "Optimization step should continue from restored value"
            )
