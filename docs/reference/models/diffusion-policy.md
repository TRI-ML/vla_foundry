# Diffusion-Based Models

VLA Foundry includes two diffusion-based architectures: Diffusion Policy for robotics action prediction and Stable Diffusion for image generation.

## Diffusion Policy

**Type key:** `diffusion_policy`
**Source:** `vla_foundry/models/diffusion_policy/diffusion_policy.py`
**Params:** [`DiffusionPolicyParams`](../params/model-params.md#diffusionpolicyparams)

Diffusion Policy predicts robot actions by iteratively denoising a noise vector conditioned on visual and language observations. It is the primary architecture for robotics policy training in VLA Foundry.

### Architecture

```
Camera Images + Language Instruction
              |
              v
    CLIP Backbone (frozen or trainable)
     |                    |
     v                    v
  Visual Tokens      Text Tokens
     |                    |
     +---- Concatenate ---+
              |
              v
     Conditioning Sequence
              |
              +--- Noisy Action Sequence (+ timestep embedding)
              |
              v
        Transformer Denoiser (bidirectional)
              |
              v
        Predicted Clean Actions
```

### Key Design Choices

- **CLIP backbone** for visual-language conditioning. The text encoder can be frozen independently of the image encoder.
- **Bidirectional transformer** as the denoiser (set `is_causal: false` on the transformer sub-component).
- **Flow matching** scheduler by default (`use_flow_matching_scheduler: true`), which provides faster convergence than DDPM.
- **Diffusion step conditioning** via concatenation (`"concat"`) or addition (`"add"`).
- **Action and proprioception dimensions** are automatically derived from `DataParams`.

### Config Preset

```yaml
# config_presets/models/diffusion_policy.yaml
type: diffusion_policy
transformer:
  <<: !include transformer_100m.yaml
  is_causal: false
vision_language_backbone:
  type: clip_backbone
  hf_pretrained: openai/clip-vit-base-patch32
  disable_text: false
noise_scheduler:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  clamp_range: [-3, 3]
use_flow_matching_scheduler: true
```

### Training Job Presets

| Preset | Task | File |
|--------|------|------|
| `diffusion_policy_bellpepper` | LBM BellPepper bimanual task | `training_jobs/diffusion_policy_bellpepper.yaml` |
| `diffusion_policy_lbm1` | LBM1 bimanual manipulation | `training_jobs/diffusion_policy_lbm1.yaml` |

### Usage

```bash
torchrun --nproc_per_node=8 vla_foundry/main.py \
    --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
    --total_train_samples 30_000_000 \
    --num_checkpoints 10 \
    --remote_sync s3://my-bucket/diffusion_policy
```

---

## Stable Diffusion

**Type key:** `stable_diffusion`
**Source:** `vla_foundry/models/diffusion/stable_diffusion.py`
**Params:** [`StableDiffusionParams`](../params/model-params.md#stablediffusionparams)

A text-conditioned latent diffusion model for image generation. Supports classifier-free guidance (CFG).

### Architecture

```
Text Input
    |
    v
CLIP Text Encoder
    |
    v
Text Embeddings ---> UNet Denoiser <--- Noisy Latents + Timestep
                          |
                          v
                   Predicted Noise / Clean Latents
```

### Components

| Component | Params | Description |
|-----------|--------|-------------|
| **UNet** | `UNetParams` | The denoising backbone. Configurable channel counts per resolution level. |
| **Noise Scheduler** | `NoiseSchedulerParams` | DDPM or flow matching noise schedule. |
| **CLIP** | `CLIPHFParams` | Text encoder for conditioning. |

### Config Preset

```yaml
# Example Stable Diffusion configuration
model:
  type: stable_diffusion
  unet:
    type: unet
    in_channels: 3
    out_channels: 3
    time_emb_dim: 256
    text_emb_dim: 512
    channels: [128, 256, 512, 1024]
  noise_scheduler:
    num_timesteps: 1000
    beta_start: 0.0001
    beta_end: 0.02
  clip:
    type: clip_hf
    hf_pretrained: openai/clip-vit-base-patch32
  do_classifier_free_guidance: true
  guidance_scale: 4.0
  dropout_percent: 0.2
```

---

## Comparison

| Feature | Diffusion Policy | Stable Diffusion |
|---------|-----------------|------------------|
| **Domain** | Robotics actions | Image generation |
| **Input** | Camera images + language | Text |
| **Denoiser** | Transformer | UNet |
| **Conditioning** | CLIP visual-language | CLIP text |
| **Scheduler** | Flow matching (default) | DDPM or flow matching |
| **Output** | Action trajectory | Generated image |


