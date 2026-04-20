"""Base class for backbone wrappers used in action policies."""

from abc import abstractmethod
from typing import Any

import torch
from transformers.utils import ModelOutput

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.registry import create_model
from vla_foundry.params.model_params import BackboneParams


class BaseBackboneWrapper(BaseModel):
    """Base wrapper that adapts a vision-language model for action policy conditioning.

    Subclasses create the underlying model internally and implement
    extraction of conditioning embeddings from model outputs.
    """

    def __init__(self, backbone_params: BackboneParams, load_pretrained: bool = True):
        super().__init__(backbone_params)
        self._model = create_model(backbone_params, load_pretrained)

    @abstractmethod
    def get_conditioning_embeddings_dim(self) -> int:
        """Get the output conditioning embeddings dimension."""
        raise NotImplementedError

    def get_action_conditioning(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> VisionLanguageBackboneOutput:
        """Get embeddings for conditioning action policies.

        Args:
            input_ids: Text token IDs [B, seq_len]
            pixel_values: Image tensors (format depends on model)
            attention_mask: Text attention mask
            attention_mask_images: Image attention mask
            **kwargs: Model-specific args

        Returns:
            VisionLanguageBackboneOutput with embeddings
        """
        (input_ids, pixel_values, attention_mask, attention_mask_images, model_kwargs) = self._prepare_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )

        outputs = self._model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **model_kwargs,
        )

        return self._extract_action_conditioning(outputs=outputs)

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, Any],
    ]:
        """Prepare inputs for conditioning, if necessary."""
        return input_ids, pixel_values, attention_mask, attention_mask_images, kwargs

    @abstractmethod
    def _extract_action_conditioning(self, outputs: ModelOutput) -> VisionLanguageBackboneOutput:
        """Extract conditioning embeddings from model outputs."""
        raise NotImplementedError
