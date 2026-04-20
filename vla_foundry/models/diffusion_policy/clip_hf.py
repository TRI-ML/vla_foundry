import torch
import torch.nn.functional as F
from transformers import CLIPConfig, CLIPModel

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.model_outputs.clip_output import CLIPOutput
from vla_foundry.params.model_params import CLIPHFParams


class CLIPHF(BaseModel):
    def __init__(self, model_params: CLIPHFParams, load_pretrained: bool = True):
        """Initialize CLIP model.

        Args:
            model_params: CLIP configuration parameters
            load_pretrained: If True, download pretrained weights
        """
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        if load_pretrained:
            self.model = CLIPModel.from_pretrained(self.model_name)
        else:
            config = CLIPConfig.from_pretrained(self.model_name)
            self.model = CLIPModel(config)

    def _post_init(self):
        super()._post_init()
        if self.model_params.freeze_text_encoder:
            self.freeze_text_encoder()
        if self.model_params.freeze_image_encoder:
            self.freeze_image_encoder()

    def freeze_text_encoder(self):
        self.model.text_model.requires_grad_(False)

    def freeze_image_encoder(self):
        self.model.vision_model.requires_grad_(False)

    def get_projection_dim(self):
        return self.model.projection_dim

    def forward(self, input_ids, pixel_values, attention_mask, attention_mask_images, **kwargs):
        if input_ids is not None:
            max_len = self.model.text_model.config.max_position_embeddings
            assert input_ids.shape[-1] <= max_len, (
                f"input_ids length {input_ids.shape[-1]} exceeds CLIP max_position_embeddings {max_len}. "
                "Pass max_text_seq_len to process_inputs to truncate upstream."
            )
            text_output = self.model.text_model(input_ids).pooler_output
            text_embeds = F.normalize(text_output, dim=-1)
            text_embeds = self.model.text_projection(text_embeds)
        else:
            text_output = None
            text_embeds = None
        if pixel_values is None:
            vision_output = None
            image_embeds = None
        else:
            # CLIP processor always returns [B*N, C, H, W] for multiple images
            assert pixel_values.ndim == 4, f"Expected 4D pixel_values [B*N, C, H, W], got {pixel_values.ndim}D"
            vision_output = self.model.vision_model(pixel_values).pooler_output
            image_embeds = self.model.visual_projection(vision_output)
            # [B*N, D] -> [B, N, D]
            image_embeds = image_embeds.view(input_ids.shape[0], -1, *image_embeds.shape[1:])
            image_embeds = F.normalize(image_embeds, dim=-1)
        if image_embeds is not None and attention_mask_images is not None:
            # [B, N, D] * [B, N, 1] -> [B, N, D]
            image_embeds = image_embeds * attention_mask_images.unsqueeze(-1)

        return CLIPOutput(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_output,
            vision_model_output=vision_output,
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError

    def get_fsdp_block_types(self):
        """Return block types for FSDP wrapping."""
        from transformers.models.clip.modeling_clip import CLIPEncoderLayer

        return (CLIPEncoderLayer,)
