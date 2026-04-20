"""VLM backbone wrapper for action policy conditioning."""

import torch

from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.params.model_params import VLMBackboneParams


class VLMHFBackboneWrapper(BaseBackboneWrapper):
    """VLM backbone wrapper for action policy conditioning.

    This class wraps VLMHF and provides the conditioning embedding interface
    using an action token probe, while keeping VLMHF free from conditioning-specific logic.
    """

    def __init__(self, backbone_params: VLMBackboneParams, load_pretrained: bool = True):
        """Initialize VLM backbone for conditioning.

        Args:
            backbone_params: Configuration parameters for the VLM backbone.
            load_pretrained: If True, download pretrained weights.
        """
        super().__init__(backbone_params, load_pretrained)
        self.num_vlm_layers_to_use = backbone_params.num_vlm_layers_to_use

        # Add action token to vocabulary
        self._action_token_id = self._model.resize_token_embeddings()

        # Detect model type
        self._vlm_model_type = getattr(self._model.model.config, "model_type", "").lower()

        # Initialize the new token's embedding as mean of existing embeddings
        with torch.no_grad():
            embeddings = self._model.model.get_input_embeddings()
            mean_emb = embeddings.weight[:-1].mean(dim=0)
            embeddings.weight[-1] = mean_emb

    def get_conditioning_embeddings_dim(self) -> int:
        """Return dimension for condition embeddings (with layers concatenated)."""
        return self._model.lm_hidden_dim * self.num_vlm_layers_to_use

    @property
    def _is_qwen_style(self) -> bool:
        return "qwen" in self._vlm_model_type

    @property
    def _is_paligemma_style(self) -> bool:
        return "paligemma" in self._vlm_model_type

    def _validate_inputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Validate VLM backbone inputs.

        Ensures required arguments are present and consistent (e.g., Qwen image_grid_thw
        matches pixel_values patches). Raises ValueError if inputs are invalid.
        """
        if self._is_qwen_style and "image_grid_thw" not in kwargs:
            raise ValueError(
                f"Qwen model requires image_grid_thw but it was not provided. "
                f"pixel_values shape: {pixel_values.shape}. "
                f"Ensure the processor passes image_grid_thw through the data pipeline."
            )

    def _prepare_inputs_for_action_token(self, input_ids, attention_mask):
        """Helper: Append action token to inputs (modular and reusable)."""
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Append action token
        action_tokens = torch.full((batch_size, 1), self._action_token_id, device=device, dtype=input_ids.dtype)
        input_ids_with_action = torch.cat([input_ids, action_tokens], dim=1)

        # Extend attention mask
        if attention_mask is not None:
            action_mask = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            attention_mask_with_action = torch.cat([attention_mask, action_mask], dim=1)
        else:
            attention_mask_with_action = None

        return input_ids_with_action, attention_mask_with_action

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ):
        self._validate_inputs(input_ids, pixel_values, attention_mask, attention_mask_images, **kwargs)

        input_ids_with_action, attention_mask_with_action = self._prepare_inputs_for_action_token(
            input_ids, attention_mask
        )

        # VLM needs hidden states for action token embedding extraction
        kwargs["output_hidden_states"] = True

        return input_ids_with_action, pixel_values, attention_mask_with_action, attention_mask_images, kwargs

    def _extract_action_conditioning(self, outputs) -> VisionLanguageBackboneOutput:
        embeddings = self._extract_action_token_embeddings(outputs)
        return VisionLanguageBackboneOutput(embeddings=embeddings)

    def _extract_action_token_embeddings(self, outputs):
        """Helper: Extract and concatenate hidden states from action token position."""
        hidden_states = outputs.hidden_states
        action_hidden_states = []
        for layer_hidden in hidden_states[-self.num_vlm_layers_to_use :]:
            action_hidden = layer_hidden[:, -1, :]  # [B, hidden_dim]
            action_hidden_states.append(action_hidden)

        # Concatenate: [B, 1, hidden_dim * num_vlm_layers_to_use]
        return torch.cat(action_hidden_states, dim=-1).unsqueeze(1)
