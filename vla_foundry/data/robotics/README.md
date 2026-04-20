# Robotics Data

## 1. Robotics Dataset Structure (Bare Minimum Requirements)

**S3 Path:**
```
dataset_directory_in_s3/
├── manifest.jsonl
├── stats.json
├── shard_00000.tar
├── shard_00001.tar
└── ...
```
- `stats.json` should be a dict with keys corresponding to camera names, states, observations, etc. (basically all tensors). These should themselves contain dicts with keys `mean`, `std`, etc.


**Inside each shard (e.g., shard_00000.tar):**
```
├── (unique_id_1).lowdim.npz
├── (unique_id_1).camera_name_{1,2,...n}.jpg
├── (unique_id_1).language_instructions.json
├── (unique_id_1).metadata.json
└── ...
```
- `lowdim.npz` is a dict.
    - It MUST contain the following keys
        - `past_mask`
        - `future_mask`
    - It will also contain the keys that will be used to construct the actions, proprioceptions, intrinsics, and extrinsics. See sections below for more details.
- `language_instructions.json` should be a dict with keys in set ["original", "randomized", "verbose", "alternative"]
- The names of keys you wish to normalize in the `lowdim.npz` dict should also exist as keys in `stats.json`.
    - Not all the keys in `lowdim.npz` will get normalized. During training time, you will need to supply flags like `--data.action_fields` and `--data.proprioception_fields` (can be empty),which should exist as fields in `lowdim.npz`. The normalizer will only normalize the keys in these two fields.
**Sample yaml config**

In the main launcher / command line, you can use the flag `--data "include vla_foundry/config_presets/data/lbm/lbm_data_params.yaml"`

Below is what a sample yaml file would look like (with some lines removed for clarity).

```yaml
type: robotics

proprioception_fields:
  - robot__actual__poses__left::panda__xyz # 3
  - robot__actual__poses__right::panda__xyz # 3
  - robot__actual__poses__left::panda__rot_6d # 6
  - robot__actual__poses__right::panda__rot_6d # 6
  - robot__actual__grippers__left::panda_hand # 1
  - robot__actual__grippers__right::panda_hand # 1

action_fields:
  - robot__desired__poses__left::panda__xyz # 3
  - robot__desired__poses__right::panda__xyz # 3
  - robot__desired__poses__left::panda__rot_6d # 6
  - robot__desired__poses__right::panda__rot_6d # 6
  - robot__desired__grippers__left::panda_hand # 1
  - robot__desired__grippers__right::panda_hand # 1

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
The full file can be found in [vla_foundry/config_presets/data/lbm/lbm_data_params.yaml](/vla_foundry/config_presets/data/lbm/lbm_data_params.yaml)

## 2. Action and Proprioception
The `RoboticsDataParams` class has attributes `action_fields` and `proprioception_fields` which take in lists. The contents of these lists should exist as keys in `lowdim.npz`, and the `RoboticsProcessor.add_action_and_proprioception_fields` function will parse these fields and extract their contents to create the action and proprioception tensors. For examples on how to specify these, see `vla_foundry/config_presets/data/lbm/lbm_data_params.yaml`.

For LeRobot converted data, the `action_fields` is usually a single-element list with the element `actions`. 

## 3. Intrinsics and Extrinsics
These are optional. They are used for the visualization scripts but are not used during training. Similar to the action and proprioception fields above, the `RoboticsDataParams` class also has attributes `intrinsics_fields` and `extrinsics_fields`. Once again, for examples on how to specify these, see `vla_foundry/config_presets/data/lbm/lbm_data_params.yaml`.


## 4. Normalization
Normalization has its own dedicated params class `NormalizationParams`, which are used to instantiate the [RoboticsNormalizer class](/vla_foundry/data/robotics/normalization.py). This `NormalizationParams` class lives as an attribute inside `RoboticsDataParams` and can be set using the prefix `--data.normalization`. 

By default, the keys in `action_fields` $\bigcup$ `proprioception_fields` are the fields that get normalized. 
The `NormalizationParams` class contains parameters on what type of normalization is done. In addition, `NormalizationParams` class also contains a `field_configs` dict, which can be used to specify how to handle certain fields that we may want to normalize differently from the default setting in `NormalizationParams`.

For details on how the normalization is implemented, see the [RoboticsNormalizer class](/vla_foundry/data/robotics/normalization.py).

### 4.1 RoboticsNormalizer Class
The `RoboticsNormalizer` object is instantiated inside the [RoboticsProcessor class](/vla_foundry/data/processor/robotics_processor.py). This instantitiation is done by supplying the appropriate `dataset_statistics` stats.json path in the `RoboticsDataParams` object. Before the start of training, `RoboticsNormalizer` and `RoboticsProcessor` will automatically save their configs to the output folder, and these can be loaded using the `from_pretrained` keyword by pointing to the directory containing their saved config files.


## 5. Image Augmentation / Transforms
Image augmentation also has its own dedicated params class `DataAugmentationParams`, which are used to instantiate the [Augmentations class](/vla_foundry/data/augmentations/base.py). This `DataAugmentationParams` class lives as an attribute inside `RoboticsDataParams` and can be set using the prefix `--data.augmentation`. Each augmentation (e.g., color jitter, random crop) has its own parameters and can be toggled using the `enabled` keyword. When the `apply_transforms` function is invoked in the pipeline, it will run through all enabled augmentations and sequentially apply them to all the images.
