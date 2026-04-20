import torch

from vla_foundry.models.model_outputs.base_output import BaseOutput


class CLIPOutput(BaseOutput):
    def __init__(
        self,
        text_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        text_model_output: torch.Tensor | None = None,
        vision_model_output: torch.Tensor | None = None,
    ):
        super().__init__()
        self.text_embeds = text_embeds
        self.image_embeds = image_embeds
        self.text_model_output = text_model_output
        self.vision_model_output = vision_model_output
