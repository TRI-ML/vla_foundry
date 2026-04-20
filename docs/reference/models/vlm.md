# Vision-Language Models

VLA Foundry provides two VLM implementations: a compositional model that pairs a ViT encoder with a transformer decoder, and a Hugging Face-backed variant that loads a complete pretrained VLM.

## VLM (Compositional)

**Type key:** `vlm`
**Source:** `vla_foundry/models/vlm.py`
**Params:** [`VLMParams`](../params/model-params.md#vlmparams)

The compositional VLM assembles two independent modules:

1. A **Vision Transformer (ViT)** that encodes images into a sequence of visual tokens.
2. A **Transformer decoder** that processes the interleaved sequence of visual tokens and text tokens autoregressively.

### Architecture

```
Image Input                   Text Input
    |                             |
    v                             v
ViT Encoder                 Token Embedding
    |                             |
    v                             |
Visual Token Projector            |
    |                             |
    +-------- Interleave ---------+
                  |
                  v
          Transformer Decoder
                  |
                  v
          Output Logits (vocab_size)
```

### ViT Options

The `vit` field accepts either a from-scratch ViT or a Hugging Face ViT:

=== "ViTParams (from scratch)"

    ```yaml
    model:
      type: vlm
      vit:
        type: vit
        img_size: 224
        hidden_dim: 1152
        inter_dim: 4304
        n_heads: 16
        n_layers: 27
        patch_size: 14
        projector_pixel_shuffle_factor: 1
    ```

=== "ViTHFParams (Hugging Face)"

    ```yaml
    model:
      type: vlm
      vit:
        type: vit_hf
        hf_pretrained: "google/siglip-so400m-patch14-384"
        hidden_dim: 1152
        projector_pixel_shuffle_factor: 2
    ```

### Transformer Options

The `transformer` field accepts either a from-scratch or HF transformer:

=== "TransformerParams"

    ```yaml
    model:
      type: vlm
      transformer:
        type: transformer
        hidden_dim: 2048
        n_layers: 18
        n_heads: 8
        vocab_size: 257216
    ```

=== "TransformerHFParams"

    ```yaml
    model:
      type: vlm
      transformer:
        type: transformer_hf
        hf_pretrained: "Qwen/Qwen2.5-0.5B"
    ```

### Shared Attribute Resolution

`VLMParams` automatically resolves several values during `init_shared_attributes`:

| Attribute | Source | Behavior |
|-----------|--------|----------|
| `processor` | `DataParams.processor` | Inherited from data config if not set on the model. |
| `image_token_id` | Processor or `DataParams` | Resolved from the loaded processor's tokenizer. |
| `transformer.vocab_size` | Processor's tokenizer length | Overridden to match the actual tokenizer vocabulary if the default value is present. |

### Config Preset

```yaml
# config_presets/models/vlm_3b.yaml
type: vlm
transformer:
  type: transformer
  hidden_dim: 2048
  n_layers: 18
  n_heads: 8
  max_seq_len: 2048
  vocab_size: 257216
  post_embed_norm: false
  weight_tying: false
vit: !include vit_paligemma.yaml
```

The PaliGemma-style ViT preset:

```yaml
# config_presets/models/vit_paligemma.yaml
type: vit
img_size: 224
hidden_dim: 1152
inter_dim: 4304
n_heads: 16
n_layers: 27
patch_size: 14
projector_pixel_shuffle_factor: 1
```

### Usage

```bash
torchrun --nproc_per_node=8 vla_foundry/main.py \
    --model "include vla_foundry/config_presets/models/vlm_3b.yaml" \
    --data.type image_caption \
    --data.processor "google/paligemma-3b-pt-224" \
    --data.dataset_manifest '["s3://path/to/manifest.jsonl"]' \
    --data.dataset_modality '["image_caption"]' \
    --data.dataset_weighting '[1.0]' \
    --total_train_samples 14_000_000
```

---

## VLM HF (Hugging Face)

**Type key:** `vlm_hf`
**Source:** `vla_foundry/models/vlm_hf.py`
**Params:** [`VLMHFParams`](../params/model-params.md#vlmhfparams)

Loads a complete pretrained VLM from Hugging Face (e.g., PaliGemma, Qwen-VL). The full architecture --- ViT, projector, and language model --- is loaded as a single unit from the HF checkpoint.

### Usage

```yaml
model:
  type: vlm_hf
  hf_pretrained: "google/paligemma-3b-pt-224"
```

```python
from vla_foundry.models import create_model
from vla_foundry.params.model_params import VLMHFParams

params = VLMHFParams(hf_pretrained="google/paligemma-3b-pt-224")
model = create_model(params)
```

---

## Choosing Between Implementations

| Consideration | `vlm` | `vlm_hf` |
|---------------|-------|----------|
| **Component control** | Mix and match ViT + LLM independently | Single HF checkpoint |
| **ViT pretrained weights** | Load separately via `ViTParams.pretrained` | Included in HF checkpoint |
| **LLM pretrained weights** | Optional via `TransformerHFParams` sub-component | Included in HF checkpoint |
| **Custom architectures** | Full control over both components | Limited to HF model architectures |
| **Fine-tuning** | Freeze ViT or LLM independently | Freeze entire model or nothing |
| **Typical use case** | Research, custom VLMs, mixed pretraining | Quick experiments with existing VLMs |

!!! tip "PaliGemma 3B"
    The recommended starting point for VLM training is the `vlm_3b` preset, which uses a PaliGemma-style ViT (SigLIP architecture, 27 layers, 1152 hidden dim) paired with an 18-layer transformer decoder with 2048 hidden dim.
