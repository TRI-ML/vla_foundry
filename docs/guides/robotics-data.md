# Robotics Data

This guide describes the robotics dataset structure used by VLA Foundry, including shard contents, configuration, normalization, and image augmentation.

## Dataset Structure

### S3 Path Layout

```
dataset_directory_in_s3/
  manifest.jsonl
  stats.json
  shard_00000.tar
  shard_00001.tar
  ...
```

The `stats.json` file is a dictionary with keys corresponding to camera names, states, observations, and other tensors. Each key maps to a dictionary containing `mean`, `std`, and other statistical fields.

### Shard Contents

Each shard (e.g., `shard_00000.tar`) contains the following per sample:

```
(unique_id_1).lowdim.npz
(unique_id_1).camera_name_{1,2,...n}.jpg
(unique_id_1).language_instructions.json
(unique_id_1).metadata.json
...
```

**`lowdim.npz`** is a dictionary that must contain the following keys:

- `past_mask`
- `future_mask`

It will also contain the keys used to construct actions, proprioceptions, intrinsics, and extrinsics (see sections below).

**`language_instructions.json`** is a dictionary with keys from the set: `original`, `randomized`, `verbose`, `alternative`.

!!! note "Normalization keys"
    The key names in `lowdim.npz` that you wish to normalize should also exist as keys in `stats.json`. Not all keys in `lowdim.npz` will be normalized. During training, you supply flags like `--data.action_fields` and `--data.proprioception_fields` (which can be empty), and these must exist as fields in `lowdim.npz`. The normalizer only normalizes the keys specified in these two fields.

## Sample YAML Config

In the main launcher or command line, reference a data config with:

```
--data "include vla_foundry/config_presets/data/lbm/lbm_data_params.yaml"
```

Below is what a sample YAML configuration looks like (some lines removed for clarity):

```yaml
type: robotics

proprioception_fields:
  - robot__actual__poses__left::panda__xyz        # 3
  - robot__actual__poses__right::panda__xyz       # 3
  - robot__actual__poses__left::panda__rot_6d     # 6
  - robot__actual__poses__right::panda__rot_6d    # 6
  - robot__actual__grippers__left::panda_hand     # 1
  - robot__actual__grippers__right::panda_hand    # 1

action_fields:
  - robot__desired__poses__left::panda__xyz       # 3
  - robot__desired__poses__right::panda__xyz      # 3
  - robot__desired__poses__left::panda__rot_6d    # 6
  - robot__desired__poses__right::panda__rot_6d   # 6
  - robot__desired__grippers__left::panda_hand    # 1
  - robot__desired__grippers__right::panda_hand   # 1

intrinsics_fields:
  - intrinsics.scene_right_0
  - intrinsics.scene_left_0
  - intrinsics.wrist_left_minus
  - intrinsics.wrist_left_plus
  - intrinsics.wrist_right_minus
  - intrinsics.wrist_right_plus

extrinsics_fields:
  - extrinsics.scene_right_0
  - extrinsics.scene_left_0
  - extrinsics.wrist_left_minus
  - extrinsics.wrist_left_plus
  - extrinsics.wrist_right_minus
  - extrinsics.wrist_right_plus

img_num_tokens: 81
image_size: 224

normalization:
  enabled: true
  method: std
  scope: global
  epsilon: 1e-8
```

The full file can be found at `vla_foundry/config_presets/data/lbm/lbm_data_params.yaml`.

## Action and Proprioception

The `RoboticsDataParams` class has `action_fields` and `proprioception_fields` attributes, each accepting a list of field names. The contents of these lists must exist as keys in `lowdim.npz`. The `RoboticsProcessor.add_action_and_proprioception_fields` function parses these fields and extracts their contents to create action and proprioception tensors.

!!! info "LeRobot data"
    For LeRobot-converted data, `action_fields` is usually a single-element list containing just `actions`.

For examples on how to specify these fields, see `vla_foundry/config_presets/data/lbm/lbm_data_params.yaml`.

## Intrinsics and Extrinsics

These fields are **optional**. They are used by the visualization scripts but are not required during training.

Similar to the action and proprioception fields, the `RoboticsDataParams` class has `intrinsics_fields` and `extrinsics_fields` attributes. For examples on how to specify these, see `vla_foundry/config_presets/data/lbm/lbm_data_params.yaml`.

## Normalization

Normalization has its own dedicated params class, `NormalizationParams`, which is used to instantiate the `RoboticsNormalizer` class (defined in `vla_foundry/data/robotics/normalization.py`). The `NormalizationParams` class lives as an attribute inside `RoboticsDataParams` and can be set using the prefix `--data.normalization`.

By default, the fields that get normalized are the union of `action_fields` and `proprioception_fields`.

The `NormalizationParams` class contains:

- Parameters controlling the type of normalization applied.
- A `field_configs` dictionary for specifying how individual fields should be normalized differently from the default settings.

For implementation details, see the `RoboticsNormalizer` class in `vla_foundry/data/robotics/normalization.py`.

### RoboticsNormalizer Class

The `RoboticsNormalizer` object is instantiated inside the `RoboticsProcessor` class (defined in `vla_foundry/data/processor/robotics_processor.py`). Instantiation requires supplying the appropriate `dataset_statistics` path (a `stats.json` file) in the `RoboticsDataParams` object.

Before training starts, `RoboticsNormalizer` and `RoboticsProcessor` automatically save their configs to the output folder. These saved configs can later be loaded using the `from_pretrained` keyword by pointing to the directory containing the saved config files.

## Image Augmentation and Transforms

Image augmentation has its own dedicated params class, `DataAugmentationParams`, which is used to instantiate the `Augmentations` class (defined in `vla_foundry/data/augmentations/base.py`). The `DataAugmentationParams` class lives as an attribute inside `RoboticsDataParams` and can be set using the prefix `--data.augmentation`.

Each augmentation (e.g., color jitter, random crop) has its own parameters and can be toggled using the `enabled` keyword. When the `apply_transforms` function is invoked in the pipeline, it runs through all enabled augmentations and sequentially applies them to all images.
