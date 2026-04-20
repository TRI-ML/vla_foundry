from copy import deepcopy

import torch

from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.model_params import ModelParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml


def test_model_loading():
    model_params = load_params_from_yaml(ModelParams, "tests/essential/shared/tiny_model/config_model.yaml")
    model = create_model(model_params)
    initial_keys = set(model.state_dict().keys())
    initial_state_dict = deepcopy(model.state_dict())

    ckpt = "tests/essential/shared/tiny_model/checkpoint.pt"
    load_model_checkpoint(model, ckpt)
    loaded_keys = set(model.state_dict().keys())
    loaded_state_dict = model.state_dict()
    assert initial_keys == loaded_keys, "State dict keys changed after loading checkpoint"

    # Check all weights changed
    unchanged = [
        name
        for name, initial in initial_state_dict.items()
        if torch.equal(initial, loaded_state_dict[name]) and "pos_embed.inv_freq" not in name
    ]
    assert len(unchanged) == 0, f"These weights didn't change: {unchanged}"

    # Check significant change
    total_change = sum(
        (initial - loaded_state_dict[name]).abs().sum().item() for name, initial in initial_state_dict.items()
    )
    total_params = sum(p.numel() for p in initial_state_dict.values())
    assert total_change / total_params > 1e-4, "Overall parameter change too small"


def test_model_same_seed_same_initialization():
    """Test that model initializes same parameters with same seed."""
    model_params = load_params_from_yaml(ModelParams, "tests/essential/shared/tiny_model/config_model.yaml")

    # Set seed and create model
    torch.manual_seed(42)
    model1 = create_model(model_params)

    # Set same seed and create another model
    torch.manual_seed(42)
    model2 = create_model(model_params)

    # Both models should have identical parameters
    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
        assert torch.equal(p1, p2), "Models with same seed should have identical parameters"


def test_model_deterministic_loading():
    """Test that model loads deterministically regardless of seed."""
    model_params = load_params_from_yaml(ModelParams, "tests/essential/shared/tiny_model/config_model.yaml")

    # Set seed and create model
    torch.manual_seed(42)
    model1 = create_model(model_params)

    # Try different seed. This shouldn't matter because we're loading the same checkpoint.
    torch.manual_seed(43)
    model2 = create_model(model_params)

    ckpt = "tests/essential/shared/tiny_model/checkpoint.pt"
    load_model_checkpoint(model1, ckpt)
    load_model_checkpoint(model2, ckpt)

    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
        assert torch.equal(p1, p2), "Models with same seed should have identical parameters"
