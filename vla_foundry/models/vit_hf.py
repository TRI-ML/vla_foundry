import timm
import torch

from vla_foundry.models.base_model import BaseModel
from vla_foundry.params.model_params import ViTHFParams


class ViTHF(BaseModel):
    def __init__(self, model_params: ViTHFParams, load_pretrained: bool = True):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        self.model = timm.create_model(self.model_name, num_classes=0, pretrained=load_pretrained)

    def forward(self, image):
        image_embeddings = self.model.forward_intermediates(image)[0]
        return image_embeddings

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError
