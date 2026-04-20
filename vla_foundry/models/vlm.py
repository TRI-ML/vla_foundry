import logging

import torch
import torch.nn as nn
from einops import rearrange

from vla_foundry.file_utils import pt_load, unwrap_state_dict
from vla_foundry.models.registry import register_model
from vla_foundry.models.transformer import Transformer
from vla_foundry.models.transformer_base import TransformerBase
from vla_foundry.models.transformer_hf import TransformerHF
from vla_foundry.models.vit import ViT
from vla_foundry.models.vit_hf import ViTHF
from vla_foundry.params.model_params import ViTParams, VLMParams


class ModalityProjector(nn.Module):
    def __init__(self, vit_params: ViTParams, output_dim: int):
        super().__init__()
        self.input_dim = vit_params.hidden_dim * (vit_params.projector_pixel_shuffle_factor**2)
        self.output_dim = output_dim
        self.scale_factor = vit_params.projector_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # equivalent to:
    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        if x.ndim == 4:
            bsz, cams, seq, embed_dim = x.size()
        else:
            cams = 0
            bsz, seq, embed_dim = x.size()  # x shape [bsz, 16*16, embed_dim]
        # With scale_factor=1 and no camera dimension, pixel_shuffle is a no-op.
        # Return early to support ViTs with non-square token counts (e.g. DINOv2 with CLS token).
        if cams == 0 and self.scale_factor == 1:
            return x

        seq_root = int(seq**0.5)
        if seq_root**2 != seq:
            raise ValueError(
                f"pixel_shuffle requires a perfect-square sequence length, got {seq} "
                f"(sqrt={seq**0.5:.2f}). If your vision encoder outputs a CLS token, "
                f"set projector_pixel_shuffle_factor=1 to skip pixel shuffle, or use a "
                f"vision encoder without CLS tokens."
            )
        if seq_root % self.scale_factor != 0:
            raise ValueError(
                f"pixel_shuffle requires sequence root ({seq_root}) to be divisible by "
                f"scale_factor ({self.scale_factor}), but {seq_root} % {self.scale_factor} = "
                f"{seq_root % self.scale_factor}."
            )
        if cams == 0:
            x = rearrange(
                x,
                "n (w w_scale h h_scale) c -> n (w h) (w_scale h_scale c)",
                w_scale=self.scale_factor,
                h_scale=self.scale_factor,
                w=seq_root // self.scale_factor,
                h=seq_root // self.scale_factor,
            )
        else:
            x = rearrange(
                x,
                "n cams (w w_scale h h_scale) c -> n (cams w h) (w_scale h_scale c)",
                w_scale=self.scale_factor,
                h_scale=self.scale_factor,
                w=seq_root // self.scale_factor,
                h=seq_root // self.scale_factor,
            )
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        return x


class VLM(TransformerBase):
    def __init__(self, model_params: VLMParams, transformer, vit):
        super().__init__(model_params)
        self.vit = vit
        self.transformer = transformer
        self.projection = ModalityProjector(model_params.vit, model_params.transformer.hidden_dim)

    def forward(
        self,
        input_ids,
        pixel_values=None,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        attention_mask_images=None,  # ignored: image tokens are handled via image_token_id masking
        **kwargs,
    ):
        if pixel_values is not None:
            # image shape [bsz, 3, image_size, image_size]
            # input_ids and attention_mask should already allot tokens for the image
            image_embd = self.vit(pixel_values)
            image_embd = self.projection(image_embd)  # [bsz, 16*16, lm_hidden_dim]
            special_image_mask = input_ids == self.model_params.image_token_id
            safe_input_ids = input_ids.clone()
            safe_input_ids[special_image_mask] = 0
            token_embd = self.transformer.embeddings(safe_input_ids).to(image_embd.dtype)
            special_image_mask = special_image_mask.unsqueeze(-1)
            assert special_image_mask.sum() == image_embd.shape[0] * image_embd.shape[1]
            special_image_mask = special_image_mask.expand_as(token_embd).to(token_embd.device)
            inputs_embeds = token_embd.masked_scatter(special_image_mask, image_embd)
        else:
            # Text-only batch: no image tokens to scatter
            inputs_embeds = self.transformer.embeddings(input_ids)

        # Call transformer's forward method directly to get logits and past_key_values and hidden_states
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            **kwargs,
        )
        return output

    def set_grad_checkpointing(self, enable: bool = True):
        """Optional: enable gradient checkpointing on the underlying LM if supported."""
        self.transformer.set_grad_checkpointing(enable)
        self.vit.set_grad_checkpointing(enable)

    def resize_token_embeddings(self, token_id: int = None) -> int:
        """Extend the embedding vocabulary of the underlying LM."""
        return self.transformer.resize_token_embeddings(token_id)

    @property
    def hidden_dim(self) -> int:
        return self.model_params.transformer.hidden_dim

    @property
    def lm_hidden_dim(self) -> int:
        return self.model_params.transformer.hidden_dim

    @property
    def num_hidden_layers(self) -> int:
        return self.model_params.transformer.n_layers

    def get_input_embeddings(self):
        if hasattr(self.transformer, "embeddings"):
            emb = self.transformer.embeddings
            # For TransformerHF, `embeddings` is a method (not an nn.Module).
            # In that case fall through to the HF model's get_input_embeddings().
            if isinstance(emb, nn.Module):
                return emb
        return self.transformer.model.get_input_embeddings()

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: int = None,
        use_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with KV-cache support.

        Args:
            input_ids: Input token ids
            pixel_values: Image pixel values
            attention_mask: Attention mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = neutral, <1.0 = more deterministic, >1.0 = more random)
            top_p: Nucleus sampling threshold (0.0-1.0, lower = more focused)
            top_k: Top-k sampling (0 = disabled)
            eos_token_id: End of sequence token id for early stopping
            use_cache: Whether to use KV-cache for faster generation
            **kwargs: Additional arguments for the forward pass. The processor used with this VLM
                might generate extra kwargs (e.g., image_grid_thw for Qwen) that need to be passed through.
        """
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        attn_mask = attention_mask.clone()
        past_key_values = None

        # First forward pass: process full sequence with images
        outputs = self.forward(
            input_ids=generated,
            pixel_values=pixel_values,
            attention_mask=attn_mask,
            use_cache=use_cache,
            **kwargs,
        )
        logits = outputs.logits[:, -1, :]
        if use_cache:
            past_key_values = outputs.past_key_values

        for _step in range(max_new_tokens):
            # Apply temperature
            if temperature != 0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample from the distribution (or greedy if temperature=0)
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask
            next_token_mask = torch.ones((batch_size, 1), dtype=attn_mask.dtype, device=attn_mask.device)
            attn_mask = torch.cat([attn_mask, next_token_mask], dim=-1)

            # Early stopping on EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Subsequent forward passes: only process new token with cached KV
            if _step < max_new_tokens - 1:  # Don't compute on last iteration
                if use_cache and past_key_values is not None:
                    # Use transformer directly with cached KV (skip image processing)
                    outputs = self.transformer(
                        input_ids=next_token,
                        attention_mask=attn_mask,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        **kwargs,
                    )
                else:
                    # No cache: reprocess full sequence
                    outputs = self.forward(
                        input_ids=generated,
                        pixel_values=pixel_values,
                        attention_mask=attn_mask,
                        use_cache=False,
                    )
                logits = outputs.logits[:, -1, :]
                if use_cache:
                    past_key_values = outputs.past_key_values

        return generated


@register_model("vlm")
def create_vlm(model_params: VLMParams, load_pretrained: bool = True):
    transformer = (
        Transformer(model_params.transformer)
        if model_params.transformer.type == "transformer"
        else TransformerHF(model_params.transformer, load_pretrained=load_pretrained)
    )
    vit = (
        ViT(model_params.vit)
        if model_params.vit.type == "vit"
        else ViTHF(model_params.vit, load_pretrained=load_pretrained)
    )

    # Load sub-component checkpoints if specified (skip when load_pretrained=False,
    # e.g. during inference where a full checkpoint is loaded separately)
    if load_pretrained and model_params.transformer.resume_from_checkpoint is not None:
        checkpoint = pt_load(model_params.transformer.resume_from_checkpoint, map_location="cpu")
        sd = unwrap_state_dict(checkpoint["state_dict"])
        transformer.load_state_dict(sd, strict=True)
        logging.info(f"=> loaded transformer weights from '{model_params.transformer.resume_from_checkpoint}'")

    if load_pretrained and model_params.vit.resume_from_checkpoint is not None:
        checkpoint = pt_load(model_params.vit.resume_from_checkpoint, map_location="cpu")
        sd = unwrap_state_dict(checkpoint["state_dict"])
        vit.load_state_dict(sd, strict=True)
        logging.info(f"=> loaded ViT weights from '{model_params.vit.resume_from_checkpoint}'")
    return VLM(model_params, transformer, vit)
