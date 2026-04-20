import einops
import torch
import torch.nn as nn

from vla_foundry.models.base_model import BaseModel
from vla_foundry.params.model_params import ViTParams


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, model_params: ViTParams):
        super().__init__()
        self.img_size = model_params.img_size
        self.patch_size = model_params.patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = model_params.cls_flag
        self.embd_dim = model_params.hidden_dim

        # Conv layer to extract the patches
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            # TODO: initialize position embedding as rotary embeddings
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, self.embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, self.embd_dim))

    def forward(self, x):
        # x shape [bsz, 3, 224, 224]
        x = self.conv(x)  # extract patches     shape [bsz, hidden_dim, 224 // patch_size, 224 // patch_size]
        x = x.flatten(2)  # flatten the patches into a single dimension
        x = x.transpose(1, 2)  # transpose to (batch_size, num_patches, hidden_dim)

        # Add CLS token (according to original ViT Paper) and position embeddings
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + self.position_embedding
        return x


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L381
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
class ViTMultiHeadAttention(nn.Module):
    def __init__(self, model_params: ViTParams):
        super().__init__()
        self.n_heads = model_params.n_heads
        self.embd_dim = model_params.hidden_dim
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = model_params.dropout

        # Combined projections for all heads
        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)

        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        # Reshape  [B, T, C] -> [B, T, n_heads, head_dim] and transpose -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # ViT attention is bidirectional
        )

        # Transpose back from [B, n_heads, T, head_dim] to [B, T, n_heads * head_dim] and combine all heads to [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L453
class ViTMLP(nn.Module):
    def __init__(self, model_params: ViTParams):
        super().__init__()
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(model_params.hidden_dim, model_params.inter_dim)
        self.fc2 = nn.Linear(model_params.inter_dim, model_params.hidden_dim)
        self.dropout = nn.Dropout(model_params.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# https://github.com/karpathy/nanoGPT/blob/master/model.py#L94
class ViTBlock(nn.Module):
    def __init__(self, model_params: ViTParams):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_params.hidden_dim, eps=model_params.ln_eps)
        self.attn = ViTMultiHeadAttention(model_params)
        self.ln2 = nn.LayerNorm(model_params.hidden_dim, eps=model_params.ln_eps)
        self.mlp = ViTMLP(model_params)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(BaseModel):
    def __init__(self, model_params: ViTParams):
        super().__init__(model_params)
        self.patch_embedding = ViTPatchEmbeddings(model_params)
        self.cls_flag = model_params.cls_flag
        self.dropout = nn.Dropout(model_params.dropout)
        self.blocks = nn.ModuleList([ViTBlock(model_params) for _ in range(model_params.n_layers)])
        self.layer_norm = nn.LayerNorm(model_params.hidden_dim, eps=model_params.ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        bsz = x.shape[0]
        if x.ndim == 5:
            # x shape [bsz, num_cameras, 3, 224, 224]
            several_images = True
            x = einops.rearrange(x, "bsz num_cameras c h w -> (bsz num_cameras) c h w")
        else:
            several_images = False
        x = self.patch_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x[:, 0]) if self.cls_flag else self.layer_norm(x)
        if several_images:
            x = einops.rearrange(x, "(bsz num_cameras) t c -> bsz num_cameras t c", bsz=bsz)
        return x
