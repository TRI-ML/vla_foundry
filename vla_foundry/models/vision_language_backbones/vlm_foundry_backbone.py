"""VLM Foundry backbone wrapper for action policy conditioning."""

import torch

from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.params.model_params import VLMFoundryBackboneParams


class VLMFoundryBackboneWrapper(BaseBackboneWrapper):
    """Backbone wrapper for VLMs trained with VLA Foundry.

    Loads a VLM from an experiment directory (config_model.yaml + checkpoint)
    and provides the conditioning embedding interface using an action token probe.
    """

    def __init__(self, backbone_params: VLMFoundryBackboneParams, load_pretrained: bool = True):
        super().__init__(backbone_params, load_pretrained)
        self.num_vlm_layers_to_use = backbone_params.num_vlm_layers_to_use

        # Add action token to vocabulary by extending the embedding table by 1.
        # Note: processor/tokenizer isn't saved with the checkpoint, so on every load the base
        # VLM is re-instantiated at its original vocab size and the action-token row must be
        # re-created here; weights for that row are then restored by load_model_checkpoint.
        base_vocab_size = self._model.resize_token_embeddings()
        self._action_token_id = base_vocab_size  # new token gets the next available index
        new_vocab_size = self._model.resize_token_embeddings(base_vocab_size + 1)

        embeddings = self._model.get_input_embeddings()
        if new_vocab_size != base_vocab_size + 1 or embeddings.weight.shape[0] != base_vocab_size + 1:
            raise RuntimeError(
                f"Action-token extension failed: expected vocab size {base_vocab_size + 1}, got "
                f"resize return {new_vocab_size} and embedding rows {embeddings.weight.shape[0]}. "
                "Check resize_token_embeddings semantics in the underlying transformer."
            )

        # Initialize new token embedding as mean of existing embeddings
        with torch.no_grad():
            mean_emb = embeddings.weight[:-1].mean(dim=0)
            embeddings.weight[-1] = mean_emb

    def get_conditioning_embeddings_dim(self) -> int:
        return self._model.lm_hidden_dim * self.num_vlm_layers_to_use

    def _prepare_inputs_for_action_token(self, input_ids, attention_mask):
        batch_size, device = input_ids.shape[0], input_ids.device
        action_tokens = torch.full((batch_size, 1), self._action_token_id, device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, action_tokens], dim=1)
        if attention_mask is not None:
            action_mask = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, action_mask], dim=1)
        return input_ids, attention_mask

    def _prepare_inputs(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        attention_mask_images=None,
        **kwargs,
    ):
        input_ids, attention_mask = self._prepare_inputs_for_action_token(input_ids, attention_mask)
        kwargs["output_hidden_states"] = True
        if pixel_values is not None:
            # The data processor (SmolVLM2) upscales images to 512×512 tiles, but the custom
            # ViT was trained at a different resolution. Resize to the ViT's native img_size.
            vit_img_size = self._model.model_params.vit.img_size
            if pixel_values.shape[-1] != vit_img_size or pixel_values.shape[-2] != vit_img_size:
                orig_shape = pixel_values.shape
                pv = pixel_values.reshape(-1, *orig_shape[-3:]).float()
                pv = torch.nn.functional.interpolate(
                    pv, size=(vit_img_size, vit_img_size), mode="bilinear", align_corners=False
                ).to(pixel_values.dtype)
                pixel_values = pv.reshape(*orig_shape[:-2], vit_img_size, vit_img_size)
            if pixel_values.dim() == 5:
                batch, num_images, c, h, w = pixel_values.shape
                pixel_values = pixel_values.reshape(batch * num_images, c, h, w)
        return input_ids, pixel_values, attention_mask, attention_mask_images, kwargs

    def _extract_action_conditioning(self, outputs) -> VisionLanguageBackboneOutput:
        hidden_states = outputs.hidden_states
        action_hidden_states = [hs[:, -1, :] for hs in hidden_states[-self.num_vlm_layers_to_use :]]
        embeddings = torch.cat(action_hidden_states, dim=-1).unsqueeze(1)
        return VisionLanguageBackboneOutput(embeddings=embeddings)
