from diffusers import UNet2DModel

from vla_foundry.models.base_model import BaseModel
from vla_foundry.params.model_params import UNetParams


class UNetDiffusers(BaseModel):
    def __init__(self, model_params: UNetParams):
        super().__init__(model_params)
        self.in_channels = model_params.in_channels
        self.model = UNet2DModel(
            sample_size=model_params.image_size,  # the target image resolution
            in_channels=model_params.in_channels,  # the number of input channels, 3 for RGB images
            out_channels=model_params.out_channels,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, timesteps):
        return self.model(x, timesteps).sample
