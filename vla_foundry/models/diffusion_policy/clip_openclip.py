import torch
from open_clip import create_model_and_transforms

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.model_outputs.clip_output import CLIPOutput
from vla_foundry.params.model_params import CLIP_OpenCLIPParams


class CLIP_OpenCLIP(BaseModel):
    def __init__(self, model_params: CLIP_OpenCLIPParams):
        super().__init__(model_params)
        self.model, _, self.transform = create_model_and_transforms(
            self.model_params.architecture, pretrained=self.model_params.pretrained_weights
        )

    def _post_init(self):
        super()._post_init()
        if self.model_params.freeze_text_encoder:
            self.freeze_text_encoder()
        if self.model_params.freeze_image_encoder:
            self.freeze_image_encoder()

    def freeze_text_encoder(self):
        self.model.transformer.requires_grad_(False)

    def freeze_image_encoder(self):
        self.model.visual.requires_grad_(False)

    def get_projection_dim(self):
        return self.model.visual.output_dim

    def forward(self, input_ids, pixel_values, attention_mask, attention_mask_images, **kwargs):
        if pixel_values.ndim == 5:
            # Handle multiple images per sample
            # [B, N, C, H, W] -> [B*N, C, H, W]
            num_images = pixel_values.shape[1]
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
            image_features, text_features, _ = self.model(image=pixel_values, text=input_ids)
            image_features = image_features.view(input_ids.shape[0], num_images, -1, *image_features.shape[2:])
            text_features = text_features.view(input_ids.shape[0], -1, *text_features.shape[2:])
        else:
            image_features, text_features, _ = self.model(image=pixel_values, text=input_ids)

        # Zero out embeddings where mask is False
        if attention_mask_images is not None:
            image_features = image_features * attention_mask_images.unsqueeze(-1)

        return CLIPOutput(image_embeds=image_features, text_embeds=text_features)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError

    def get_fsdp_block_types(self):
        """Return block types for FSDP wrapping."""
        import open_clip

        return (open_clip.transformer.ResidualAttentionBlock,)
