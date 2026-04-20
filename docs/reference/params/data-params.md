# DataParams

`DataParams` is the polymorphic base class for all dataset configurations. Like `ModelParams`, it uses draccus `ChoiceRegistry` --- the concrete subclass is selected by the `type` field.

**Source:** `vla_foundry/params/base_data_params.py` (base), `vla_foundry/params/data_params.py` (subclasses)

## Base Fields

Every `DataParams` subclass inherits these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `None` | Registry key that selects the concrete subclass (`"text"`, `"text_untokenized"`, `"image_caption"`, `"robotics"`). |
| `dataset_manifest` | `List[str]` | `[]` | List of paths to dataset manifest files (local or S3). One entry per dataset. |
| `dataset_weighting` | `List[float]` | `[]` | Sampling weight for each dataset. Must have the same length as `dataset_manifest`. |
| `dataset_modality` | `List[str]` | `[]` | Modality label for each dataset (e.g., `"text"`, `"image_caption"`, `"robotics"`). |
| `val_dataset_manifest` | `List[str]` | `[]` | Manifest paths for validation datasets. |
| `val_dataset_weighting` | `List[float]` | `[]` | Sampling weights for validation datasets. |
| `allow_multiple_epochs` | `bool` | `False` | Allow the dataloader to loop over the dataset more than once. Required when using `num_epochs`. |
| `num_workers` | `Optional[int]` | `None` | Number of dataloader workers per GPU. If `None`, auto-calculated as `cpu_count // world_size`. |
| `prefetch_factor` | `int` | `4` | Number of batches to prefetch per worker (PyTorch DataLoader). |
| `seq_len` | `int` | `2048` | Sequence length for tokenized data. |
| `shuffle` | `bool` | `True` | Shuffle the dataset. |
| `shuffle_buffer_size` | `int` | `2000` | Size of the shuffle buffer (WebDataset streaming shuffle). |
| `shuffle_initial` | `int` | `500` | Initial shuffle buffer fill size. |
| `seed` | `int` | `42` | **Shared.** Random seed, inherited from `HyperParams.seed`. |

---

## TextDataParams

**Type key:** `text`

For pre-tokenized text datasets. Adds no fields beyond the base class.

```yaml
data:
  type: text
  dataset_manifest: ["s3://my-bucket/text-data/manifest.jsonl"]
  dataset_modality: ["text"]
  dataset_weighting: [1.0]
  seq_len: 2048
```

---

## TextUntokenizedDataParams

**Type key:** `text_untokenized`

For raw (untokenized) text datasets. Tokenization happens on-the-fly using the specified tokenizer.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tokenizer` | `str` | `"EleutherAI/gpt-neox-20b"` | HuggingFace tokenizer identifier. |

**Computed property:**

| Property | Description |
|----------|-------------|
| `pad_token_id` | The pad token ID from the loaded tokenizer. If the tokenizer has no pad token, `[PAD]` is added automatically. |

```yaml
data:
  type: text_untokenized
  tokenizer: "EleutherAI/gpt-neox-20b"
  dataset_manifest: ["s3://my-bucket/raw-text/manifest.jsonl"]
  dataset_modality: ["text"]
  dataset_weighting: [1.0]
```

---

## ImageCaptionDataParams

**Type key:** `image_caption`

For image-caption paired datasets, typically used for VLM training.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `processor` | `str` | `"google/paligemma-3b-pt-224"` | HuggingFace processor identifier for image and text preprocessing. |
| `img_num_tokens` | `int` | `256` | Number of image tokens in the sequence. |
| `image_size` | `int` | `224` | Input image resolution in pixels. |
| `augmentation` | `DataAugmentationParams` | `DataAugmentationParams()` | Image augmentation configuration. |

**Computed properties:**

| Property | Description |
|----------|-------------|
| `image_token_id` | The image token ID from the loaded processor. |
| `pad_token_id` | The pad token ID from the processor's tokenizer. |

```yaml
data:
  type: image_caption
  processor: "google/paligemma-3b-pt-224"
  img_num_tokens: 256
  image_size: 224
  dataset_manifest: ["s3://my-bucket/image-caption/manifest.jsonl"]
  dataset_modality: ["image_caption"]
  dataset_weighting: [1.0]
```

---

## RoboticsDataParams

**Type key:** `robotics`

The most feature-rich data params subclass. Configures camera inputs, proprioception/action field mappings, normalization, augmentation, and temporal windowing for robotics policy training.

**Source:** `vla_foundry/params/data_params.py`

### Dataset and Processor

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_statistics` | `list[str]` | `[]` | Paths to dataset statistics JSON files (one per manifest). Required for normalization. |
| `val_dataset_statistics` | `list[str]` | `[]` | Paths to validation dataset statistics. |
| `processor` | `str` | `None` | HuggingFace processor identifier for image preprocessing. |
| `img_num_tokens` | `int` | `256` | Number of image tokens. |
| `image_size` | `int` | `224` | Input image resolution. |

### Camera Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `camera_names` | `list[str]` | `[]` | Camera names (e.g., `["scene_right_0", "wrist_left_plus"]`). Auto-detected from preprocessing config if empty. |
| `image_indices` | `list[int]` | `[]` | Temporal indices for images (e.g., `[-1, 0]` for previous and current frame). Auto-detected if empty. |
| `image_names` | `list[str]` | `[]` | Computed from `camera_names` and `image_indices` (e.g., `"scene_right_0_t-1"`). |
| `pad_missing_images` | `bool` | `False` | Pad missing camera images with zeros instead of erroring. |
| `mask_padded_images` | `bool` | `False` | Provide a mask indicating which images were padded. Requires `pad_missing_images`. |

### Field Definitions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `proprioception_fields` | `list[str]` | `[]` | Names of proprioception fields from the dataset (e.g., joint positions, gripper state). |
| `action_fields` | `list[str]` | `[]` | Names of action fields from the dataset (e.g., relative poses, gripper commands). |
| `pose_groups` | `list[Dict[str, str]]` | `[]` | Groups of position/rotation fields for relative coordinate transforms. |
| `intrinsics_fields` | `list[str]` | `[]` | Camera intrinsics field names. |
| `extrinsics_fields` | `list[str]` | `[]` | Camera extrinsics field names. |

### Normalization and Augmentation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `normalization` | `NormalizationParams` | `NormalizationParams()` | Global and per-field normalization configuration. |
| `augmentation` | `DataAugmentationParams` | `DataAugmentationParams()` | Image augmentation settings. |

### Temporal Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lowdim_past_timesteps` | `Optional[int]` | `None` | Number of past observation timesteps. Falls back to `normalization.lowdim_past_timesteps`. |
| `lowdim_future_timesteps` | `Optional[int]` | `None` | Number of future action timesteps. Falls back to `normalization.lowdim_future_timesteps`. |

### Computed Dimensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action_dim` | `int` | `None` | Auto-computed by summing dimensions of all `action_fields` from normalization statistics. |
| `proprioception_dim` | `Optional[int]` | `None` | Auto-computed by summing dimensions of all `proprioception_fields` from normalization statistics. |

### Language Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `language_instruction_types` | `list[str]` | `["original"]` | Which language instruction variants to use. Valid values: `"original"`, `"randomized"`, `"verbose"`, `"alternative"`. |

!!! info "Automatic Dimension Computation"
    `action_dim` and `proprioception_dim` are computed during `__post_init__` by loading the dataset statistics file and summing the dimensions of each named field. You generally do not need to set these manually. If you do set them, the values are validated against the computed values.

### Example YAML

```yaml
data:
  type: robotics
  dataset_manifest:
    - s3://my-bucket/dataset/shards/manifest.jsonl
  dataset_statistics:
    - s3://my-bucket/dataset/shards/stats.json
  dataset_modality:
    - robotics
  dataset_weighting:
    - 1.0
  camera_names:
    - scene_right_0
    - scene_left_0
    - wrist_left_plus
    - wrist_right_minus
  image_indices:
    - -1
    - 0
  proprioception_fields:
    - robot__actual__poses__left::panda__xyz
    - robot__actual__poses__left::panda__rot_6d
    - robot__actual__grippers__left::panda_hand
  action_fields:
    - robot__action__poses__left::panda__xyz_relative
    - robot__action__poses__left::panda__rot_6d_relative
    - robot__action__grippers__left::panda_hand
  normalization:
    enabled: true
    method: percentile_1_99
    scope: global
    epsilon: 1e-2
    centered_norm: true
  augmentation:
    image:
      crop:
        enabled: true
        shape: [224, 224]
        mode: center
  image_size: 224
  processor: "openai/clip-vit-base-patch32"
  allow_multiple_epochs: true
  lowdim_past_timesteps: 1
  lowdim_future_timesteps: 14
```
