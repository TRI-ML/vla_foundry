# Data Preprocessing

This guide covers the data preprocessing pipeline in VLA Foundry, including dependency setup, Ray parallelization, and conversion scripts for various dataset formats.

## Dependencies

The preprocessing scripts use dependencies that are isolated from the training code. Install them with:

```bash
uv sync --group=preprocessing
```

## Using Ray

Many preprocessing scripts use [Ray](https://docs.ray.io/) for parallelization. The general workflow is:

1. Create your Ray-compatible script.
2. Start a Ray instance -- either locally or on AWS EC2 clusters.
3. Run your script from within the Ray environment.

### Local Ray

For local usage, start Ray on your machine:

```bash
ray start --head
```

!!! tip "Ray Dashboard"
    Add `--include-dashboard=True` before `--head` to enable the Ray dashboard for diagnostics.

### AWS Ray Cluster

!!! warning "Template — adapt to your AWS account"
    `ray_cluster_configs.yaml` is a starting template, **not a turn-key config**. It will not `ray up` cleanly in a fresh AWS account. Before first use you must fill in the `PLACEHOLDER` fields to match your infrastructure:

    - `SubnetIds: [subnet-PLACEHOLDER]` — a subnet in your VPC with outbound internet access
    - `SecurityGroupIds: [sg-PLACEHOLDER]` — a security group that permits intra-cluster traffic (Ray ports) and outbound HTTPS
    - `ImageId: ami-PLACEHOLDER` — an Ubuntu AMI compatible with the listed `InstanceType`s (`m5.xlarge`, `i4i.4xlarge`)
    - `IamInstanceProfile.Arn: arn:aws:iam::ACCOUNT_ID:instance-profile/ray-autoscaler-v1` — an instance profile with EC2 autoscaling + the S3 read/write permissions your workload needs
    - Tags (`owner.email`, `project`) — adjust to your organization's tagging convention

    The template has been validated on our internal setup but has not been tested against a clean AWS account. If `ray up` or `ray attach` fails, expect to iterate on the AMI / subnet / security group settings. Prefer the [Local Ray](#local-ray) path above if your preprocessing volume fits on a single machine.

#### 1. Create the cluster

```bash
ray up vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

!!! warning "Before running"
    You may also want to edit the following in `ray_cluster_configs.yaml`:

    - `min_workers` / `max_workers` — scale to your preprocessing volume
    - `file_mounts`: By default it copies your HF token from `~/.cache/huggingface/token`. Change this if your token is stored elsewhere.
    - `rsync_exclude`: Currently excludes `.venv` and `wandb`. Add additional paths you want to exclude (e.g., large checkpoints).

#### 2. Attach to the cluster

```bash
ray attach vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

#### 3. Run your script inside the cluster

```bash
# Optionally start a persistent terminal like tmux
cd vla_foundry
python some_preprocessing_script.py
```

!!! note
    Ray scripts currently do **not** work well with `uv run`. Use `uv sync --group=preprocessing` (automatically done in `ray_cluster_configs.yaml`) and `source .venv/bin/activate` instead.

#### 4. Shut down the cluster

When finished, exit the cluster, then from your local machine:

```bash
ray down vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

## Conversion Scripts

### Downloading a Hugging Face Dataset to S3

```bash
python vla_foundry/data/preprocessing/hf_utils/hf_dataset_downloader.py \
  --dataset IPEC-COMMUNITY/droid_lerobot \
  --mode s3 \
  --s3-output-path s3://your-bucket/your-path/hf_datasets/droid_lerobot \
  --local-output-dir /datasets/hf_datasets/droid_lerobot \
  --preserve-structure
```

### Converting VLM Hugging Face Captions to Tar Shards

This uses [img2dataset](https://github.com/rom1504/img2dataset) for image downloading and webdataset shard creation. The HF dataset must already be downloaded to S3 (see section above).

```bash
python vla_foundry/data/preprocessing/preprocess_captionshf_to_tar.py \
  --cluster ray \
  --input_path s3://your-bucket/your-path/downloads/ \
  --output_path s3://your-bucket/your-path/downloads2/ \
  --url_col images \
  --caption_col texts \
  --save_additional_columns metadata
```

### Converting Text Hugging Face Datasets to Tar Shards

```bash
python vla_foundry/data/preprocessing/preprocess_untokenized_to_tar.py \
  --s3_input_path s3://your-bucket/your-path/hf_datasets/fineweb-edu-350BT \
  --s3_output_path s3://your-bucket/your-path/datasets/text/fineweb-edu-350BT \
  --tmp_dir /tmp/finewebshards
```

### Converting LeRobot to Tar Shards

The HF dataset must already be staged (local dir or S3 bucket). The example
below uses Physical Intelligence's publicly-available `pi_libero` dataset.

Stage it first:

```bash
hf download physical-intelligence/pi_libero --repo-type dataset \
    --local-dir /tmp/pi_libero
aws s3 sync /tmp/pi_libero s3://your-bucket/your-path/hf_datasets/pi_libero/
```

Then run the preprocessor (swap `s3://...` for a local path like `/tmp/pi_libero/` to skip the S3 staging):

```bash
source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type "lerobot" \
  --source_episodes "['s3://your-bucket/your-path/hf_datasets/pi_libero/']" \
  --output_dir s3://your-bucket/your-path/lerobotdata/pi_libero/ \
  --camera_names "['image', 'wrist_image']" \
  --samples_per_shard 100 \
  --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml" \
  --observation_keys "['state']" \
  --action_keys "['actions']"
```

The `--camera_names`, `--observation_keys`, and `--action_keys` shown above
match `pi_libero`'s schema; adjust them if you swap in a different LeRobot
dataset (e.g., `lerobot/pusht` uses single-camera `['observation.image']`,
state `['observation.state']`, and action `['action']`).

### Converting LBM Spartan Data to Tar Shards

This is a two-step process: first download the raw Spartan data from the public registry, then convert it to tar shards.

??? note "Available tasks and raw data sizes"

    | Task | Size |
    |------|------|
    | BimanualHangMugsOnMugHolderFromDryingRack | 92.4 GB |
    | BimanualHangMugsOnMugHolderFromTable | 93.3 GB |
    | BimanualLayCerealBoxOnCuttingBoardFromTopShelf | 53.2 GB |
    | BimanualLayCerealBoxOnCuttingBoardFromUnderShelf | 80.6 GB |
    | BimanualPlaceAppleFromBowlIntoBin | 76.1 GB |
    | BimanualPlaceAppleFromBowlOnCuttingBoard | 73.0 GB |
    | BimanualPlaceAvocadoFromBowlOnCuttingBoard | 113.1 GB |
    | BimanualPlaceFruitFromBowlIntoBin | 164.6 GB |
    | BimanualPlaceFruitFromBowlOnCuttingBoard | 140.3 GB |
    | BimanualPlacePearFromBowlIntoBin | 75.6 GB |
    | BimanualPlacePearFromBowlOnCuttingBoard | 113.4 GB |
    | BimanualPutMugsOnPlatesFromDryingRack | 135.3 GB |
    | BimanualPutMugsOnPlatesFromTable | 69.6 GB |
    | BimanualPutRedBellPepperInBin | 70.5 GB |
    | BimanualPutSpatulaOnPlateFromDryingRack | 50.6 GB |
    | BimanualPutSpatulaOnPlateFromTable | 38.5 GB |
    | BimanualPutSpatulaOnTableFromDryingRack | 61.9 GB |
    | BimanualPutSpatulaOnTableFromUtensilCrock | 88.5 GB |
    | BimanualStackPlatesOnTableFromDryingRack | 149.9 GB |
    | BimanualStackPlatesOnTableFromTable | 215.8 GB |
    | BimanualStoreCerealBoxUnderShelf | 74.9 GB |
    | PickAndPlaceBox | 15.6 GB |
    | PlaceCupByCoaster | 121.7 GB |
    | PlaceCupOnCoaster | 199.3 GB |
    | PushCoasterToCenterOfTable | 164.4 GB |
    | PushCoasterToMug | 182.8 GB |
    | PutBananaInCenterOfTable | 20.2 GB |
    | PutBananaOnSaucer | 23.3 GB |
    | PutCupInCenterOfTable | 95.8 GB |
    | PutCupOnSaucer | 201.2 GB |
    | PutGreenAppleInCenterOfTable | 55.9 GB |
    | PutGreenAppleOnSaucer | 19.9 GB |
    | PutKiwiInCenterOfTable | 19.6 GB |
    | PutKiwiOnSaucer | 43.0 GB |
    | PutMugOnSaucer | 115.6 GB |
    | PutOrangeInCenterOfTable | 44.8 GB |
    | PutOrangeOnSaucer | 20.9 GB |
    | PutSpatulaInUtensilCrock | 54.2 GB |
    | PutSpatulaInUtensilCrockFromDryingRack | 46.2 GB |
    | TurnCupUpsideDown | 394.2 GB |
    | TurnMugRightsideUp | 220.3 GB |

    **Total: 40 tasks, ~3.8 TB**. Raw data is not available for PushBox — use the pre-processed download instead.

#### Step 1: Download the raw data

```bash
python vla_foundry/data/scripts/download_dataset.py \
  --task PutGreenAppleOnSaucer \
  --local_path /data/raw \
  --raw

python vla_foundry/data/scripts/download_dataset.py \
  --task PickAndPlaceBox \
  --local_path /data/raw \
  --raw
```

This downloads and extracts the raw Spartan episodes to `/data/raw/tasks/<TaskName>/`.

!!! tip
    Use `--dry_run` to preview what will be downloaded. Use `--all --raw` to download raw data for every task.

#### Step 2: Preprocess into tar shards

The `--source_episodes` argument expects paths to `diffusion_spartan/` directories.
After downloading, locate them with:

```bash
find /data/raw/tasks -name diffusion_spartan -type d
# Example output:
# /data/raw/tasks/PutGreenAppleOnSaucer/cabot/sim/bc/teleop/2024-11-14T15-59-59-08-00/diffusion_spartan
# /data/raw/tasks/PutGreenAppleOnSaucer/cabot/sim/bc/teleop/2024-11-14T16-33-05-08-00/diffusion_spartan
# /data/raw/tasks/PickAndPlaceBox/cabot/sim/bc/teleop/2025-09-08T15-18-57-04-00/diffusion_spartan
```

Then pass them to the preprocessor:

```bash
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type "spartan" \
  --source_episodes "[
      '/data/raw/tasks/PutGreenAppleOnSaucer/cabot/sim/bc/teleop/2024-11-14T15-59-59-08-00/diffusion_spartan/',
      '/data/raw/tasks/PutGreenAppleOnSaucer/cabot/sim/bc/teleop/2024-11-14T16-33-05-08-00/diffusion_spartan/',
      '/data/raw/tasks/PickAndPlaceBox/cabot/sim/bc/teleop/2025-09-08T15-18-57-04-00/diffusion_spartan/']" \
  --output_dir /data/preprocessed/mixed_tasks/ \
  --camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \
  --language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
  --action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
  --data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
  --samples_per_shard 100 \
  --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
```

