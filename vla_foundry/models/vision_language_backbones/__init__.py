from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.models.vision_language_backbones.clip_hf_backbone import CLIPBackboneWrapper
from vla_foundry.models.vision_language_backbones.vit_backbone import ViTBackboneWrapper
from vla_foundry.models.vision_language_backbones.vlm_foundry_backbone import VLMFoundryBackboneWrapper
from vla_foundry.models.vision_language_backbones.vlm_hf_backbone import VLMHFBackboneWrapper
from vla_foundry.params.model_params import CLIPBackboneParams, VLMBackboneParams, VLMFoundryBackboneParams


def get_vision_language_backbone(backbone_params, load_pretrained: bool = True):
    # Note: We do this instead of isintance() to avoid issues with class overlaps
    t = backbone_params.type
    if t == "clip_backbone":
        return CLIPBackboneWrapper(backbone_params, load_pretrained)
    elif t == "vlm_backbone":
        return VLMHFBackboneWrapper(backbone_params, load_pretrained)
    elif t == "vlm_foundry_backbone":
        return VLMFoundryBackboneWrapper(backbone_params, load_pretrained)
    elif t == "vit_backbone":
        return ViTBackboneWrapper(backbone_params, load_pretrained)
    else:
        raise ValueError(f"Unsupported vision language backbone type: {t}")
