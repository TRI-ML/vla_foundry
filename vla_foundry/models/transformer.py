import math

import torch
from torch import nn

from vla_foundry.activations import get_feed_forward
from vla_foundry.attention import get_attn_func
from vla_foundry.models.fsdp_block import FSDPBlock
from vla_foundry.models.model_outputs.llm_output import TransformerOutput
from vla_foundry.models.registry import register_model
from vla_foundry.models.transformer_base import TransformerBase
from vla_foundry.norms import get_norm_class
from vla_foundry.params.model_params import TransformerParams
from vla_foundry.positional_embedding import get_pos_embed


class CustomAttn(nn.Module):
    def __init__(self, layer_id: int, model_params: TransformerParams):
        super().__init__()
        self.n_heads = model_params.n_heads
        self.hidden_dim = model_params.hidden_dim
        self.head_dim = model_params.hidden_dim // model_params.n_heads
        self.in_proj = nn.Linear(self.hidden_dim, 3 * self.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_dim, bias=False)
        self.pos_embed = get_pos_embed(model_params)
        self.attn_fn = get_attn_func(model_params.attn_name)
        self.apply_qk_norm = model_params.qk_norm
        self.is_causal = model_params.is_causal

        # initialize norm layers for queries and keys if needed
        self.norm_type = get_norm_class(model_params.norm_type)
        self.q_norm = (
            self.norm_type(
                self.n_heads * self.head_dim,
                eps=model_params.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            self.norm_type(
                self.n_heads * self.head_dim,
                eps=model_params.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )

        self.layer_id = layer_id
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(self.hidden_dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (self.layer_id + 1))
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor, is_causal=None, past_key_value=None, use_cache=False, attention_mask=None):
        batchsize, seq_len, hidden_dim = x.shape
        queries, keys, vals = self.in_proj(x).chunk(3, dim=-1)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = queries.view(batchsize, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(batchsize, seq_len, self.n_heads, self.head_dim)
        vals = vals.view(batchsize, seq_len, self.n_heads, self.head_dim)

        past_length = 0 if past_key_value is None else past_key_value[0].shape[1]
        queries, keys, vals = self.pos_embed(queries, keys, vals, offset=past_length)

        if past_key_value is not None and use_cache:
            keys = torch.cat([past_key_value[0], keys], dim=1)
            vals = torch.cat([past_key_value[1], vals], dim=1)

        if use_cache:
            past_key_value = [keys, vals]

        output = self.attn_fn(
            queries,
            keys,
            vals,
            is_causal=is_causal or self.is_causal,
            attention_mask=attention_mask,
        )
        output = output.view(batchsize, seq_len, -1)
        return self.out_proj(output), past_key_value


class TransformerBlock(FSDPBlock):
    def __init__(self, layer_id: int, model_params: TransformerParams):
        super().__init__()
        self.n_heads = model_params.n_heads
        self.hidden_dim = model_params.hidden_dim

        self.head_dim = model_params.hidden_dim // model_params.n_heads
        self.attention = CustomAttn(layer_id, model_params)
        self.ffn_type = model_params.ffn_type
        self.feed_forward, self.ffn_hidden_dim = get_feed_forward(self.ffn_type, self.hidden_dim)
        self.is_causal = model_params.is_causal

        self.layer_id = layer_id
        self.norm_type = get_norm_class(model_params.norm_type)
        self.attention_norm = self.norm_type(
            model_params.hidden_dim,
            eps=model_params.norm_eps,
        )
        self.ffn_norm = self.norm_type(
            model_params.hidden_dim,
            eps=model_params.norm_eps,
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.ffn_type == "swiglu":
            # initialize weights trunc_normal(1/sqrt(fan_in))
            std = 1.0 / math.sqrt(self.hidden_dim)
            torch.nn.init.trunc_normal_(self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std)
            # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self.layer_id + 1))
            torch.nn.init.trunc_normal_(self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std)
        elif self.ffn_type == "gelu":
            std = 1.0 / math.sqrt(self.hidden_dim)
            torch.nn.init.trunc_normal_(self.feed_forward[0].weight, std=std, a=-3 * std, b=3 * std)

            std = 1.0 / math.sqrt(self.hidden_dim)
            std = std / math.sqrt(2 * (self.layer_id + 1))
            torch.nn.init.trunc_normal_(self.feed_forward[2].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x, past_key_value=None, use_cache=False, attention_mask=None, is_causal=None):
        h, past_key_value = self.attention(
            self.attention_norm(x),
            is_causal=is_causal or self.is_causal,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        h = x + h
        ffn_out = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_out
        return out, past_key_value


class Transformer(TransformerBase):
    def __init__(self, model_params: TransformerParams):
        super().__init__(model_params)
        # for convenience we often share param names with llama
        self._hidden_dim = model_params.hidden_dim
        self.vocab_size = model_params.vocab_size
        self.n_layers = model_params.n_layers
        self.max_seq_len = model_params.max_seq_len
        self.norm_type = get_norm_class(model_params.norm_type)
        self.post_embed_norm = (
            self.norm_type(
                model_params.hidden_dim,
                eps=model_params.norm_eps,
            )
            if model_params.post_embed_norm
            else nn.Identity()
        )
        self.weight_tying = model_params.weight_tying
        self.embeddings = nn.Embedding(model_params.vocab_size, model_params.hidden_dim)
        self.is_causal = model_params.is_causal

        self.layers = torch.nn.ModuleList()
        for layer_id in range(model_params.n_layers):
            self.layers.append(TransformerBlock(layer_id, model_params))

        # get class for normalization layers
        self.norm = self.norm_type(
            model_params.hidden_dim,
            eps=model_params.norm_eps,
        )
        self.output = nn.Linear(model_params.hidden_dim, model_params.vocab_size, bias=False)
        if self.weight_tying:
            self.embeddings.weight = self.output.weight
        self.grad_checkpointing = False
        self.reset_parameters()

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layers

    def reset_parameters(self):
        # initialize weight 1/sqrt(dim)
        # this is 1/fan_in for output, as is default, and Maciej Kilian tried another option
        # for the embed layer (from RWKV paper) but this was better.
        std = 1.0 / math.sqrt(self.hidden_dim)
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.embeddings.weight, std=std, a=-3 * std, b=3 * std)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def resize_token_embeddings(self, new_num_tokens: int = None) -> int:
        """Resize the token embedding table. Mirrors HF's `resize_token_embeddings` semantics:
        - `new_num_tokens is None` → query current size, no modification.
        - `new_num_tokens <= current` → no-op, returns current size.
        - `new_num_tokens > current` → extend embeddings/output to exactly `new_num_tokens` rows.
        """
        if new_num_tokens is None or new_num_tokens <= self.vocab_size:
            return self.vocab_size

        old_embedding_weight = self.embeddings.weight.data.clone()
        old_output_weight = self.output.weight.data.clone()

        new_embeddings = nn.Embedding(new_num_tokens, self.hidden_dim)
        new_embeddings.weight.data[: self.vocab_size] = old_embedding_weight

        std = 1.0 / math.sqrt(self.hidden_dim)
        torch.nn.init.trunc_normal_(new_embeddings.weight.data[self.vocab_size :], std=std, a=-3 * std, b=3 * std)

        new_output = nn.Linear(self.hidden_dim, new_num_tokens, bias=False)
        new_output.weight.data[: self.vocab_size] = old_output_weight
        torch.nn.init.trunc_normal_(new_output.weight.data[self.vocab_size :], std=std, a=-3 * std, b=3 * std)

        self.embeddings = new_embeddings
        self.output = new_output
        self.vocab_size = new_num_tokens

        if self.weight_tying:
            self.embeddings.weight = self.output.weight

        return self.vocab_size

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        output_hidden_states=False,
        is_causal=None,
        **kwargs,
    ):
        """
        Args:
            input
            past_key_values
            use_cache (bool)
            attention_mask (torch.Tensor): Shape (batch_size, sequence_len), indicates tokens that should not be
                attended to. attention_mask[s, i] = False indicates that token i should not be attended to by any other
                token for sequence s.
            output_hidden_states (bool): Whether to return the hidden states of the transformer.
            is_causal (bool): Whether the transformer is causal, default set in init, can be overridden by the caller.
        """
        if input_ids is not None:
            x = self.embeddings(input_ids)
        elif inputs_embeds is not None:
            x = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        x = self.post_embed_norm(x)

        # For causal models during training (no KV-cache), the causal mask already
        # prevents real tokens from attending to right-padded positions, so we can
        # skip the explicit padding attention_mask. During generation with KV-cache,
        # the mask is still needed for variable prefill lengths.
        if (is_causal or self.is_causal) and not use_cache:
            attention_mask = None

        if past_key_values is None:
            past_key_values = [None] * self.n_layers
        elif isinstance(past_key_values, tuple):
            past_key_values = list(past_key_values)
        hidden_states = []
        for i, layer in enumerate(self.layers):
            if self.grad_checkpointing:
                x, past_key_values[i] = torch.utils.checkpoint.checkpoint(
                    layer, x, past_key_values[i], use_cache, attention_mask, is_causal or self.is_causal
                )
            else:
                x, past_key_values[i] = layer(
                    x,
                    past_key_values[i],
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    is_causal=is_causal or self.is_causal,
                )
            if output_hidden_states:
                hidden_states.append(x)
        if past_key_values[0] is None:
            past_key_values = None
        x = self.norm(x)
        output = self.output(x)
        if self.model_params.cast_output_to_float32:
            output = output.float()

        return TransformerOutput(
            logits=output,
            past_key_values=past_key_values,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=20,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        eos_token_id=None,
        use_cache=True,
        **kwargs,
    ):
        """
        Generate tokens autoregressively with KV-cache support.

        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = neutral, <1.0 = more deterministic, >1.0 = more random)
            top_p: Nucleus sampling threshold (0.0-1.0, lower = more focused)
            top_k: Top-k sampling (0 = disabled)
            eos_token_id: End of sequence token id for early stopping
            use_cache: Whether to use KV-cache for faster generation
        """
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        attn_mask = attention_mask.clone()
        past_key_values = None

        for _step in range(max_new_tokens):
            # Only process last token if we have cached K/V, otherwise process full sequence
            curr_input_ids = generated[:, -1:] if use_cache and past_key_values is not None else generated

            outputs = self.forward(
                input_ids=curr_input_ids,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            logits = outputs.logits[:, -1, :]
            if use_cache:
                past_key_values = outputs.past_key_values

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

            # Update attention mask: 1 for non-padding tokens
            next_token_mask = torch.ones((batch_size, 1), dtype=attn_mask.dtype, device=attn_mask.device)
            attn_mask = torch.cat([attn_mask, next_token_mask], dim=-1)

            # Early stopping on EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


@register_model("transformer")
def create_transformer(model_params: TransformerParams, load_pretrained: bool = True):
    return Transformer(model_params)
