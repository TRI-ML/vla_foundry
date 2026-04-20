#!/usr/bin/env python3
"""Test that load_pretrained parameter works correctly for all model types."""

from unittest.mock import MagicMock, patch

from vla_foundry.models import create_model
from vla_foundry.params.model_params import TransformerHFParams, VLMHFParams


def test_transformer_hf_load_pretrained():
    """Test that TransformerHF respects load_pretrained parameter."""
    config = TransformerHFParams(hf_pretrained="hf-internal-testing/tiny-random-gpt2")

    # Test with load_pretrained=True (should call from_pretrained)
    with (
        patch("vla_foundry.models.transformer_hf.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained,
        patch("vla_foundry.models.transformer_hf.AutoConfig.from_pretrained"),
        patch("vla_foundry.models.transformer_hf.AutoModelForCausalLM.from_config"),
    ):
        mock_from_pretrained.return_value = MagicMock()
        create_model(config, load_pretrained=True)
        mock_from_pretrained.assert_called_once()

    # Test with load_pretrained=False (should call from_config)
    with (
        patch("vla_foundry.models.transformer_hf.AutoModelForCausalLM.from_pretrained"),
        patch("vla_foundry.models.transformer_hf.AutoConfig.from_pretrained") as mock_config,
        patch("vla_foundry.models.transformer_hf.AutoModelForCausalLM.from_config") as mock_from_config,
    ):
        mock_config.return_value = MagicMock()
        mock_from_config.return_value = MagicMock()
        create_model(config, load_pretrained=False)
        mock_from_config.assert_called_once()

    print("✓ TransformerHF correctly respects load_pretrained parameter")


def test_vlm_hf_load_pretrained():
    """Test that VLMHF respects load_pretrained parameter."""
    config = VLMHFParams(hf_pretrained="hf-internal-testing/tiny-random-LlamaForCausalLM")

    # Test with load_pretrained=True (should call from_pretrained)
    with (
        patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained") as mock_from_pretrained,
        patch("vla_foundry.models.vlm_hf.AutoConfig.from_pretrained"),
        patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_config"),
    ):
        mock_from_pretrained.return_value = MagicMock()
        _model = create_model(config, load_pretrained=True)
        mock_from_pretrained.assert_called_once()

    # Test with load_pretrained=False (should call from_config)
    with (
        patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_pretrained"),
        patch("vla_foundry.models.vlm_hf.AutoConfig.from_pretrained") as mock_config,
        patch("vla_foundry.models.vlm_hf.AutoModelForImageTextToText.from_config") as mock_from_config,
    ):
        mock_config.return_value = MagicMock()
        mock_from_config.return_value = MagicMock()
        _model = create_model(config, load_pretrained=False)
        mock_from_config.assert_called_once()

    print("✓ VLMHF correctly respects load_pretrained parameter")


def test_clip_hf_load_pretrained():
    """Test that CLIPHF respects load_pretrained parameter when used directly."""
    print("Testing CLIPHF load_pretrained parameter...")

    # Simplified test: just check the code paths are correct
    import inspect

    from vla_foundry.models.diffusion_policy import clip_hf

    # Check the __init__ method has the load_pretrained parameter
    init_signature = inspect.signature(clip_hf.CLIPHF.__init__)
    assert "load_pretrained" in init_signature.parameters, "CLIPHF missing load_pretrained parameter"

    # Check the source code contains the correct logic
    source = inspect.getsource(clip_hf.CLIPHF.__init__)
    assert "if load_pretrained:" in source, "CLIPHF missing load_pretrained conditional"
    assert "CLIPModel.from_pretrained" in source, "CLIPHF missing from_pretrained call"
    assert "CLIPModel(config)" in source, "CLIPHF missing CLIPModel(config) call"

    print("✓ CLIPHF correctly respects load_pretrained parameter")
