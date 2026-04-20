# Config Presets

Config presets are reusable YAML fragments stored in `vla_foundry/config_presets/`. They define standard configurations for models, data pipelines, hyperparameters, and complete training jobs. Presets are composed using YAML `!include` directives and can be overridden by command-line arguments.

## Usage

Presets can be loaded in two ways:

=== "CLI include"

    ```bash
    --model "include vla_foundry/config_presets/models/vlm_3b.yaml"
    ```

=== "YAML !include"

    ```yaml
    model:
      <<: !include ../models/diffusion_policy.yaml
    ```

Command-line arguments always take precedence over values from presets.

---

## Model Presets

Located in `vla_foundry/config_presets/models/`.

### Transformers

From-scratch causal transformers at various scales.

| Preset | Type | hidden_dim | n_layers | n_heads | max_seq_len | Notes |
|--------|------|-----------|----------|---------|-------------|-------|
| `transformer_tiny.yaml` | `transformer` | 64 | 2 | 2 | 128 | ~3M params. Testing only. |
| `transformer_11m.yaml` | `transformer` | 96 | 8 | 4 | 2048 | ~11M params. Quick experiments. |
| `transformer_100m.yaml` | `transformer` | 512 | 12 | 8 | 2048 | ~100M params. Also used as Diffusion Policy denoiser. |
| `transformer_410m.yaml` | `transformer` | 1024 | 24 | 16 | 2048 | ~410M params. |
| `transformer_1b.yaml` | `transformer` | 2048 | 24 | 16 | 2048 | ~1B params. |

### Hugging Face Transformers

| Preset | Type | Model | File |
|--------|------|-------|------|
| `qwen_05b.yaml` | `transformer_hf` | Qwen2.5-0.5B | `config_presets/models/qwen_05b.yaml` |

### Vision Models

| Preset | Type | Description | File |
|--------|------|-------------|------|
| `vit_tiny.yaml` | `vit` | Tiny ViT for smoke testing (4 layers, 64 dim, patch 14, 224px) | `config_presets/models/vit_tiny.yaml` |
| `vit_paligemma.yaml` | `vit` | PaliGemma-style SigLIP ViT (27 layers, 1152 dim, patch 14) | `config_presets/models/vit_paligemma.yaml` |
| `vit_smolvlm2_256m.yaml` | `vit` | SmolVLM2-style ViT (12 layers, 768 dim, patch 16, 512px, pixel shuffle 2x) | `config_presets/models/vit_smolvlm2_256m.yaml` |
| `vit_smolvlm2_256m_224.yaml` | `vit` | SmolVLM2-style ViT at 224px (12 layers, 768 dim, patch 16) | `config_presets/models/vit_smolvlm2_256m_224.yaml` |
| `unet.yaml` | `unet` | UNet for Stable Diffusion (channels: 128/256/512/1024) | `config_presets/models/unet.yaml` |

### Vision-Language Models

| Preset | Type | Description | File |
|--------|------|-------------|------|
| `vlm_11m.yaml` | `vlm` | 11M VLM: tiny ViT + 11M transformer. Smoke testing. | `config_presets/models/vlm_11m.yaml` |
| `vlm_100m.yaml` | `vlm` | 100M VLM: tiny ViT + 100M transformer | `config_presets/models/vlm_100m.yaml` |
| `vlm_1b.yaml` | `vlm` | 1B VLM: PaliGemma ViT + 1B transformer | `config_presets/models/vlm_1b.yaml` |
| `vlm_3b.yaml` | `vlm` | 3B VLM: PaliGemma ViT + 18-layer Transformer (2048 dim) | `config_presets/models/vlm_3b.yaml` |
| `vlm_3b_gemma2_2b.yaml` | `vlm` | 3B VLM: PaliGemma ViT + Gemma2-2B HF backbone | `config_presets/models/vlm_3b_gemma2_2b.yaml` |
| `smolvlm_load_llm.yaml` | `vlm` | VLM initialized from 1B transformer + SmolVLM2 224px ViT | `config_presets/models/smolvlm_load_llm.yaml` |
| `paligemma_load_llm.yaml` | `vlm` | VLM initialized from 1B transformer + PaliGemma ViT | `config_presets/models/paligemma_load_llm.yaml` |

### Policy Models

| Preset | Type | Description | File |
|--------|------|-------------|------|
| `diffusion_policy.yaml` | `diffusion_policy` | CLIP backbone + 100M transformer denoiser + flow matching | `config_presets/models/diffusion_policy.yaml` |

### VLA Diffusion Models

VLA Diffusion models use a VLM backbone (loaded from checkpoint) with a diffusion transformer head for action prediction.

| Preset | Type | Description | File |
|--------|------|-------------|------|
| `vla_diffusion_11m.yaml` | `diffusion_policy` | VLM backbone + tiny transformer denoiser. Smoke testing. | `config_presets/models/vla_diffusion_11m.yaml` |
| `vla_diffusion_100m.yaml` | `diffusion_policy` | VLM backbone + 11M transformer denoiser | `config_presets/models/vla_diffusion_100m.yaml` |
| `vla_diffusion_1b.yaml` | `diffusion_policy` | VLM backbone + 100M transformer denoiser | `config_presets/models/vla_diffusion_1b.yaml` |
| `vla_diffusion_paligemma2.yaml` | `diffusion_policy` | PaliGemma2-3B HF backbone + 100M transformer denoiser | `config_presets/models/vla_diffusion_paligemma2.yaml` |

---

## Data Presets

Located in `vla_foundry/config_presets/data/`.

### Base Data Configurations

| Preset | Type | Description | File |
|--------|------|-------------|------|
| `diffusion_policy.yaml` | `robotics` | Base robotics data params for Diffusion Policy (CLIP processor, 1 past + 14 future timesteps) | `config_presets/data/diffusion_policy.yaml` |
| `vla_diffusion.yaml` | `robotics` | VLA Diffusion data params (PaliGemma2 processor, 224px, 256 img tokens, seq_len 2048) | `config_presets/data/vla_diffusion.yaml` |

### LBM (Large Behavior Model)

Stored in `config_presets/data/lbm/`.

| Preset | Description | File |
|--------|-------------|------|
| `lbm_data_params.yaml` | Base LBM robotics data (bimanual Panda, 6 cameras, proprioception + action fields) | `data/lbm/lbm_data_params.yaml` |
| `lbm_action_fields.yaml` | LBM action field definitions | `data/lbm/lbm_action_fields.yaml` |
| `lbm_data_camera_names_4cameras.yaml` | 4-camera configuration | `data/lbm/lbm_data_camera_names_4cameras.yaml` |
| `lbm_data_camera_names_6cameras.yaml` | 6-camera configuration | `data/lbm/lbm_data_camera_names_6cameras.yaml` |
| `lbm_language_annotations.yaml` | Language instruction type configuration | `data/lbm/lbm_language_annotations.yaml` |
| `lbm_image_augmentation_params.yaml` | Image augmentation settings | `data/lbm/lbm_image_augmentation_params.yaml` |
| `lbm_data_discard_key.yaml` | Keys to discard from the dataset | `data/lbm/lbm_data_discard_key.yaml` |

### Preprocessing Parameters

| Preset | Description | File |
|--------|-------------|------|
| `robotics_preprocessing_params_1past_14future.yaml` | Standard 1 past + 14 future timesteps | `data/robotics_preprocessing_params_1past_14future.yaml` |
| `robotics_preprocessing_params_5past_20future_lbmsize.yaml` | 5 past + 20 future timesteps, 342x256 images | `data/robotics_preprocessing_params_5past_20future_lbmsize.yaml` |

---

## Hyperparameter Presets

Located in `vla_foundry/config_presets/hparams/`.

| Preset | Description | Key Settings | File |
|--------|-------------|-------------|------|
| `diffusion_policy.yaml` | Diffusion Policy hparams | `lr: 5e-4`, `loss: mse`, `grad_clip: 1.0`, `lr_cooldown_end: 1e-5` | `hparams/diffusion_policy.yaml` |

---

## Training Job Presets

Located in `vla_foundry/config_presets/training_jobs/`. These are complete experiment configurations that compose model, data, and hparam presets with task-specific overrides.

| Preset | Model | Task | File |
|--------|-------|------|------|
| `diffusion_policy_bellpepper.yaml` | Diffusion Policy | LBM BellPepper bimanual manipulation | `training_jobs/diffusion_policy_bellpepper.yaml` |
| `diffusion_policy_lbm1.yaml` | Diffusion Policy | LBM1 bimanual manipulation (full config) | `training_jobs/diffusion_policy_lbm1.yaml` |
| `lbm_hparams_4cams.yaml` | LBM | 4-camera hparam configuration | `training_jobs/lbm_hparams_4cams.yaml` |
| `lbm_hparams_6cams.yaml` | LBM | 6-camera hparam configuration | `training_jobs/lbm_hparams_6cams.yaml` |
| `lbm_multitask_4cams.yaml` | Diffusion Policy | LBM multitask 4-camera with 410M transformer, EMA | `training_jobs/lbm_multitask_4cams.yaml` |
| `vla_diffusion_bellpepper.yaml` | VLA Diffusion | BellPepper task with PaliGemma2 VLM backbone | `training_jobs/vla_diffusion_bellpepper.yaml` |
| `vla_diffusion_tiny_test.yaml` | VLA Diffusion | Tiny VLA diffusion for smoke testing (local data) | `training_jobs/vla_diffusion_tiny_test.yaml` |

### Anatomy of a Training Job Preset

A training job preset composes presets from other categories:

```yaml
# training_jobs/diffusion_policy_bellpepper.yaml
model:
  <<: !include ../models/diffusion_policy.yaml        # Model preset
  vision_language_backbone:
    type: clip_backbone
    hf_pretrained: openai/clip-vit-base-patch32
    freeze_text_encoder: True
  transformer:
    <<: !include ../models/transformer_100m.yaml       # Nested model preset
    is_causal: True

data:
  <<: !include ../data/lbm/lbm_data_params.yaml       # Robot-specific fields
  <<: !include ../data/diffusion_policy.yaml           # Base data params
  dataset_manifest:                                     # Task-specific data
    - s3://bucket/BimanualPutRedBellPepperInBin/manifest.jsonl
  dataset_statistics:
    - s3://bucket/BimanualPutRedBellPepperInBin/stats.json
  dataset_modality:
    - robotics
  dataset_weighting:
    - 1.0

distributed:
  fsdp: True

hparams:
  <<: !include ../hparams/diffusion_policy.yaml        # Hparam preset
  per_gpu_batch_size: 16
  global_batch_size: 128
```

---

## Directory Structure

```
vla_foundry/config_presets/
|-- models/
|   |-- transformer_tiny.yaml
|   |-- transformer_11m.yaml
|   |-- transformer_100m.yaml
|   |-- transformer_410m.yaml
|   |-- transformer_1b.yaml
|   |-- qwen_05b.yaml
|   |-- vit_tiny.yaml
|   |-- vit_paligemma.yaml
|   |-- vit_smolvlm2_256m.yaml
|   |-- vit_smolvlm2_256m_224.yaml
|   |-- vlm_11m.yaml
|   |-- vlm_100m.yaml
|   |-- vlm_1b.yaml
|   |-- vlm_3b.yaml
|   |-- vlm_3b_gemma2_2b.yaml
|   |-- smolvlm_load_llm.yaml
|   |-- paligemma_load_llm.yaml
|   |-- unet.yaml
|   |-- diffusion_policy.yaml
|   |-- vla_diffusion_11m.yaml
|   |-- vla_diffusion_100m.yaml
|   |-- vla_diffusion_1b.yaml
|   +-- vla_diffusion_paligemma2.yaml
|-- data/
|   |-- diffusion_policy.yaml
|   |-- vla_diffusion.yaml
|   |-- robotics_preprocessing_params_1past_14future.yaml
|   |-- robotics_preprocessing_params_5past_20future_lbmsize.yaml
|   |-- lbm/
|   +-- preprocessing/
|-- hparams/
|   +-- diffusion_policy.yaml
+-- training_jobs/
    |-- diffusion_policy_bellpepper.yaml
    |-- diffusion_policy_lbm1.yaml
    |-- lbm_hparams_4cams.yaml
    |-- lbm_hparams_6cams.yaml
    |-- lbm_multitask_4cams.yaml
    |-- vla_diffusion_bellpepper.yaml
    +-- vla_diffusion_tiny_test.yaml
```

!!! tip "Creating Your Own Presets"
    The easiest way to start a new experiment is to copy an existing training job preset and modify the dataset paths, camera names, and field definitions. The model and hparam presets can be reused as-is in most cases.
