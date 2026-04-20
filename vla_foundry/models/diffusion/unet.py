import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from vla_foundry.model_utils import Float32Module
from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.fsdp_block import FSDPBlock
from vla_foundry.params.model_params import UNetParams


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = self.dim // 2
        emb = math.log(10000) / (self.half_dim - 1)
        emb = torch.exp(torch.arange(self.half_dim) * -emb)  # [half_dim]
        self.register_buffer("emb", emb)

    def forward(self, time):
        emb = time[:, None] * self.emb[None, :]  # [bsz, half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [bsz, dim]
        return emb


class ResnetBlock(FSDPBlock):
    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim=None):
        super().__init__()
        self.out_channels = out_channels
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if text_emb_dim:
            self.text_mlp = nn.Linear(text_emb_dim, out_channels)
        else:
            self.text_mlp = None

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, text_emb=None):
        h = self.block1(x)  # [bsz, out_channels, h, w]

        # Add time conditioning
        h = h + self.time_mlp(time_emb).view(-1, self.out_channels, 1, 1)

        # Add text conditioning if available
        if text_emb is not None and self.text_mlp is not None:
            h = h + self.text_mlp(text_emb).view(-1, self.out_channels, 1, 1)

        h = self.block2(h)
        return h + self.shortcut(x)  # [bsz, out_channels, h, w]


class SelfAttentionBlock(FSDPBlock):
    """Self-attention block for spatial attention"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h)  # [B, C, H, W]
        q = einops.rearrange(q, "b c h w -> b (h w) c")  # [B, H*W, C]
        k = self.k(h)  # [B, C, H, W]
        k = einops.rearrange(k, "b c h w -> b (h w) c")  # [B, H*W, C]
        v = self.v(h)  # [B, C, H, W]
        v = einops.rearrange(v, "b c h w -> b (h w) c")  # [B, H*W, C]

        h = F.scaled_dot_product_attention(q, k, v)
        h = einops.rearrange(h, "b (h w) c -> b c h w", h=H, w=W)
        return x + self.proj_out(h)  # [B, C, H, W]


class CrossAttentionBlock(FSDPBlock):
    """Cross-attention block for text conditioning"""

    def __init__(self, channels, context_dim):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Linear(context_dim, channels)
        self.v = nn.Linear(context_dim, channels)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h)  # [B, C, H, W]
        q = einops.rearrange(q, "b c h w -> b (h w) c")  # [B, H*W, C]
        k = self.k(context)  # [B, seq_len, C]
        v = self.v(context)  # [B, seq_len, C]

        h = F.scaled_dot_product_attention(q, k, v)
        h = einops.rearrange(h, "b (h w) c -> b c h w", h=H, w=W)
        return x + self.proj_out(h)  # [B, C, H, W]


class UNet(BaseModel):
    """U-Net architecture for diffusion model"""

    def __init__(self, model_params: UNetParams):
        super().__init__(model_params)
        self.in_channels = model_params.in_channels
        self.out_channels = model_params.out_channels
        self.time_emb_dim = model_params.time_emb_dim
        self.text_emb_dim = model_params.text_emb_dim
        self.channels = model_params.channels
        self.dim_expansion = 4

        # Time embedding
        time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim * self.dim_expansion),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * self.dim_expansion, self.time_emb_dim * self.dim_expansion),
        )
        if model_params.time_mlp_float32:
            self.time_mlp = Float32Module(time_mlp, cast_outputs_back=True)
        else:
            self.time_mlp = time_mlp
        # Initial projection
        self.init_conv = nn.Conv2d(self.in_channels, self.channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        prev_ch = self.channels[0]
        for i, ch in enumerate(self.channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock(prev_ch, ch, self.time_emb_dim * self.dim_expansion, self.text_emb_dim),
                        ResnetBlock(ch, ch, self.time_emb_dim * self.dim_expansion, self.text_emb_dim),
                        SelfAttentionBlock(ch) if i >= 2 else nn.Identity(),
                        CrossAttentionBlock(ch, self.text_emb_dim) if (i >= 1 and self.text_emb_dim) else nn.Identity(),
                    ]
                )
            )
            if i < len(self.channels) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            prev_ch = ch

        # Middle
        self.mid_block1 = ResnetBlock(
            self.channels[-1], self.channels[-1], self.time_emb_dim * self.dim_expansion, self.text_emb_dim
        )
        self.mid_attn = SelfAttentionBlock(self.channels[-1])
        self.mid_cross_attn = (
            CrossAttentionBlock(self.channels[-1], self.text_emb_dim) if self.text_emb_dim else nn.Identity()
        )
        self.mid_block2 = ResnetBlock(
            self.channels[-1], self.channels[-1], self.time_emb_dim * self.dim_expansion, self.text_emb_dim
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, ch in enumerate(reversed(self.channels)):
            in_ch = ch if i == 0 else ch + self.channels[-(i)]
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResnetBlock(in_ch, ch, self.time_emb_dim * self.dim_expansion, self.text_emb_dim),
                        ResnetBlock(ch, ch, self.time_emb_dim * self.dim_expansion, self.text_emb_dim),
                        SelfAttentionBlock(ch) if len(self.channels) - i - 1 >= 2 else nn.Identity(),
                        CrossAttentionBlock(ch, self.text_emb_dim)
                        if (len(self.channels) - i - 1 >= 1 and self.text_emb_dim)
                        else nn.Identity(),
                    ]
                )
            )
            if i < len(self.channels) - 1:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

        # Final projection
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, self.channels[0]), nn.SiLU(), nn.Conv2d(self.channels[0], self.out_channels, 3, padding=1)
        )

    def _process_block(self, x, time_emb, text_emb, text_emb_pooled, block):
        """Process a single block with ResNet + attention layers"""
        resnet1, resnet2, attn, cross_attn = block
        x = resnet1(x, time_emb, text_emb_pooled)
        x = resnet2(x, time_emb, text_emb_pooled)

        if not isinstance(attn, nn.Identity):
            x = attn(x)
        if not isinstance(cross_attn, nn.Identity) and text_emb is not None:
            x = cross_attn(x, text_emb)
        return x

    def forward(self, x, timesteps, text_embeddings=None):
        # Convert timesteps to tensor with proper batch dimension
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor([timesteps] * x.shape[0], device=x.device, dtype=torch.long)
        elif timesteps.dim() == 0:  # scalar tensor
            timesteps = timesteps.unsqueeze(0).expand(x.shape[0])

        time_emb = self.time_mlp(timesteps).to(x.dtype)  # [bsz, time_emb_dim*4]
        text_emb_pooled = text_embeddings.max(dim=1)[0] if text_embeddings is not None else None  # [bsz, text_emb_dim]

        x = self.init_conv(x)  # [bsz, channels[0], h, w]
        skip_connections = []

        # Encoder
        for i, block in enumerate(self.down_blocks):
            x = self._process_block(x, time_emb, text_embeddings, text_emb_pooled, block)
            if i < len(self.down_samples):
                skip_connections.append(x)  # [bsz, channels[i], h, w]
                x = self.down_samples[i](x)  # [bsz, channels[i], h//2, w//2]

        # Middle
        x = self.mid_block1(x, time_emb, text_emb_pooled)  # [bsz, channels[-1], h, w]
        x = self.mid_attn(x)
        if text_embeddings is not None:
            x = self.mid_cross_attn(x, text_embeddings)
        x = self.mid_block2(x, time_emb, text_emb_pooled)

        # Decoder
        for i, block in enumerate(self.up_blocks):
            if i > 0:
                x = torch.cat([x, skip_connections.pop()], dim=1)  # [bsz, channels+skip_channels, h, w]
            x = self._process_block(x, time_emb, text_embeddings, text_emb_pooled, block)
            if i < len(self.up_samples):
                x = self.up_samples[i](x)  # [bsz, channels[i], h*2, w*2]

        return self.final_conv(x)  # [bsz, out_channels, h, w]
