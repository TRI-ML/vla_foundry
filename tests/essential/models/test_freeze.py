#!/usr/bin/env python3

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.transformer import Transformer
from vla_foundry.models.vit import ViT
from vla_foundry.models.vlm import VLM
from vla_foundry.params.model_params import TransformerParams, ViTParams, VLMParams


def test_freeze_functionality():
    """
    Test that the freeze functionality works for the Transformer model.
    """

    # Test with freeze=False (default)
    params = TransformerParams()

    model = Transformer(params)
    assert isinstance(model, BaseModel)

    # Check if parameters are trainable
    trainable_params = sum(p.requires_grad for p in model.parameters())
    total_params = sum(1 for p in model.parameters())
    assert trainable_params == total_params

    # Test with freeze=True
    params_frozen = TransformerParams(freeze=True)

    model_frozen = Transformer(params_frozen)

    # Check if parameters are frozen
    trainable_params_frozen = sum(p.requires_grad for p in model_frozen.parameters())
    total_params_frozen = sum(1 for p in model_frozen.parameters())

    # Verify that frozen model has no trainable parameters
    assert trainable_params_frozen == 0, f"Expected 0 trainable parameters, got {trainable_params_frozen}"
    assert total_params_frozen == total_params, f"Expected {total_params} total parameters, got {total_params_frozen}"


def test_freeze_functionality_vlm():
    """
    Test that the freeze functionality works partially for the VLM model when freezing the vit only.
    """

    vit_params = ViTParams(freeze=True)
    transformer_params = TransformerParams(freeze=False)
    params = VLMParams(vit=vit_params, transformer=transformer_params)
    model = VLM(params, transformer=Transformer(transformer_params), vit=ViT(vit_params))
    assert isinstance(model, BaseModel)

    trainable_params = sum(p.requires_grad for p in model.parameters())
    vit_params = sum(1 for p in model.vit.parameters())
    transformer_params = sum(1 for p in model.transformer.parameters())
    projection_params = sum(1 for p in model.projection.parameters())
    total_params = sum(1 for p in model.parameters())

    assert trainable_params == transformer_params + projection_params
    assert vit_params + transformer_params + projection_params == total_params


if __name__ == "__main__":
    test_freeze_functionality()
