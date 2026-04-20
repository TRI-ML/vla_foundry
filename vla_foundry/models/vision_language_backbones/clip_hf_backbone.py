"""CLIP backbone wrapper for action policy conditioning."""

import torch

from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.params.model_params import CLIPBackboneParams


class CLIPBackboneWrapper(BaseBackboneWrapper):
    """Wraps CLIPHF and provides conditioning embeddings for action policies."""

    def __init__(self, backbone_params: CLIPBackboneParams, load_pretrained: bool = True):
        super().__init__(backbone_params, load_pretrained)
        self.disable_text = backbone_params.disable_text

    def get_conditioning_embeddings_dim(self) -> int:
        """Return CLIP projection dimension."""
        return self._model.get_projection_dim()

    def _concatenate_for_conditioning(self, text_embeds, image_embeds):
        """Helper: Concatenate text and image embeddings into sequence."""
        embeddings_list = []

        if not self.disable_text and text_embeds is not None:
            # [B, D] -> [B, 1, D]
            embeddings_list.append(text_embeds.unsqueeze(1))

        if image_embeds is not None:
            # [B, D] or [B, N, D] -> [B, N, D]
            if image_embeds.ndim == 2:
                image_embeds = image_embeds.unsqueeze(1)
            embeddings_list.append(image_embeds)

        # Concatenate: [B, 1+N, D] or [B, N, D]
        return torch.cat(embeddings_list, dim=1) if embeddings_list else None

    def _extract_action_conditioning(self, outputs) -> VisionLanguageBackboneOutput:
        embeddings = self._concatenate_for_conditioning(outputs.text_embeds, outputs.image_embeds)
        return VisionLanguageBackboneOutput(embeddings=embeddings)
