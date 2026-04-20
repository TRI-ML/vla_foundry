# Transformer Models

VLA Foundry provides two transformer implementations: a from-scratch GPT-style model and a Hugging Face-backed variant. Both are registered in the model registry and can be used as standalone language models or as components within larger architectures (VLMs, Diffusion Policies).

## Transformer (From Scratch)

**Type key:** `transformer`
**Source:** `vla_foundry/models/transformer.py`
**Params:** [`TransformerParams`](../params/model-params.md#transformerparams)

A decoder-only causal transformer built in pure PyTorch. This is the default building block for language modeling in VLA Foundry.

### Architecture

```
Input Tokens
    |
    v
Token Embedding + Positional Embedding (rotary by default)
    |
    v
[Optional] Post-Embedding LayerNorm
    |
    v
N x Transformer Block:
    |-- Multi-Head Self-Attention (causal mask)
    |-- LayerNorm
    |-- FFN (SwiGLU by default)
    |-- LayerNorm
    |
    v
Final LayerNorm
    |
    v
Output Projection (vocab_size) [optional weight tying]
```

### Key Design Choices

- **Rotary positional embeddings** (RoPE) by default, enabling length extrapolation.
- **SwiGLU** feed-forward network for improved training efficiency.
- **Optional QK normalization** for training stability at scale.
- **Causal masking** enabled by default (`is_causal=True`). Set to `False` for bidirectional attention (e.g., when used as a denoiser in Diffusion Policy).

### Config Presets

| Preset | Parameters | hidden_dim | n_layers | n_heads | File |
|--------|-----------|------------|----------|---------|------|
| **transformer_tiny** | ~3M | 64 | 2 | 2 | `config_presets/models/transformer_tiny.yaml` |
| **transformer_11m** | ~11M | 96 | 8 | 4 | `config_presets/models/transformer_11m.yaml` |
| **transformer_100m** | ~100M | 512 | 12 | 8 | `config_presets/models/transformer_100m.yaml` |
| **transformer_410m** | ~410M | 1024 | 24 | 16 | `config_presets/models/transformer_410m.yaml` |
| **transformer_1b** | ~1B | 2048 | 24 | 16 | `config_presets/models/transformer_1b.yaml` |

### Example: 410M Transformer

```yaml
# config_presets/models/transformer_410m.yaml
type: transformer
hidden_dim: 1024
n_layers: 24
n_heads: 16
max_seq_len: 2048
vocab_size: 50432
post_embed_norm: false
weight_tying: false
is_causal: true
```

### Usage

=== "Standalone LLM"

    ```bash
    torchrun --nproc_per_node=8 vla_foundry/main.py \
        --model "include vla_foundry/config_presets/models/transformer_410m.yaml" \
        --data.type text \
        --data.dataset_manifest '["s3://path/to/manifest.jsonl"]' \
        --data.dataset_modality '["text"]' \
        --data.dataset_weighting '[1.0]' \
        --total_train_samples 10_000_000
    ```

=== "As Diffusion Policy Denoiser"

    ```yaml
    model:
      type: diffusion_policy
      transformer:
        <<: !include transformer_100m.yaml
        is_causal: false  # Bidirectional for denoising
    ```

---

## TransformerHF (Hugging Face)

**Type key:** `transformer_hf`
**Source:** `vla_foundry/models/transformer_hf.py`
**Params:** [`TransformerHFParams`](../params/model-params.md#transformerhfparams)

Wraps any Hugging Face `AutoModelForCausalLM` checkpoint. The architecture is determined entirely by the pretrained model --- `hidden_dim` and `vocab_size` are read dynamically from the HF config.

### Usage

```yaml
model:
  type: transformer_hf
  hf_pretrained: "Qwen/Qwen2.5-0.5B"
```

```python
from vla_foundry.models import create_model
from vla_foundry.params.model_params import TransformerHFParams

params = TransformerHFParams(hf_pretrained="Qwen/Qwen2.5-0.5B")
model = create_model(params)

print(params.hidden_dim)  # 1024 (from Qwen config)
print(params.vocab_size)  # 151936 (from Qwen config)
```

### Config Preset

| Preset | Model | File |
|--------|-------|------|
| **qwen_05b** | Qwen2.5-0.5B | `config_presets/models/qwen_05b.yaml` |

```yaml
# config_presets/models/qwen_05b.yaml
type: transformer_hf
hidden_dim: 1024
n_layers: 24
n_heads: 16
max_seq_len: 2048
vocab_size: 151936
post_embed_norm: false
weight_tying: false
```

!!! note
    For `transformer_hf`, the fields like `hidden_dim` in the YAML are informational hints. The actual values are read from the Hugging Face model config at runtime via the `hf_pretrained` identifier.

## Choosing Between Implementations

| Consideration | `transformer` | `transformer_hf` |
|---------------|---------------|-------------------|
| **Pretrained weights** | No (train from scratch) | Yes (any HF checkpoint) |
| **Architecture control** | Full (norm type, FFN type, etc.) | Limited to HF model design |
| **Customizability** | High (pure PyTorch) | Lower (HF abstractions) |
| **Use as sub-component** | VLM, Diffusion Policy denoiser | VLM language backbone |
| **FSDP compatibility** | Native FSDP block wrapping | HF-compatible wrapping |
