# Dependencies
The scripts in this folder use certain preprocessing-specific dependencies, which we have isolated from the training code. 

To run these scripts locally, use
```
uv sync --group=preprocessing
```

# Using Ray
Many scripts use Ray for parallelization.

The general flow is as follows: (1) Create your script that's compatible with Ray. (2) Start a Ray instance on AWS clusters with `ray_cluster_configs.yaml`. This will start a head node with several worker instances. (3) From within the Ray cluster, run your script.

Alternatively, Ray also works on local instances, so Steps 2 and 3 can be skipped. 

1. Start ray (if not already running)
```bash
ray start --head
```

Add the --include-dashboard=True arg before the --head to include the ray dashboard for diagnostics.

2. [Optional] Create the ray cluster

> **⚠️ `ray_cluster_configs.yaml` is a template — adapt to your AWS account before use.**
> The file ships with `PLACEHOLDER` values for `SubnetIds`, `SecurityGroupIds`, `ImageId`, and the IAM instance-profile `Arn` (`ACCOUNT_ID`). `ray up` will not succeed until these are replaced with resources that exist in your VPC / account. The template has been validated on our internal setup but has not been tested in a clean AWS account — expect to iterate on AMI / subnet / security group / IAM settings. Prefer the local Ray path above if your preprocessing volume fits on a single machine.

```bash
####################
# IMPORTANT: You may also want to edit the following in ray_cluster_configs.yaml before running.
# - tags, username, number of nodes, etc.
# - file_mounts: It copies your HF token from ~/.cache/huggingface/token. Change this if it's somewhere else.
# - rsync_exclude: It currently excludes rsyncing `.venv` and `wandb`. Add here if there are other paths you want to exclude (e.g. large checkpoints).
####################
ray up vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

3. [Optional] Attach the ray cluster. This will take you "inside" the cluster.
```bash
ray attach vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

4. Run your script inside the cluster. Note that ray scripts currently do **not** work well with `uv run`. The requirements can still be used with `uv sync --group=preprocessing` (automatically done in `ray_cluster_configs.yaml`) and `source .venv/bin/activate`.
```bash
# [optional] Start a persistent terminal like tmux
cd vla_foundry
python (some-script-here)
```

5. When finished, exit the cluster. Then, from your own machine, shut down the ray cluster with `ray down`.
```bash
ray down vla_foundry/config_presets/data/preprocessing/ray_cluster_configs.yaml
```

# Downloading a Hugging Face dataset to S3
```bash
python vla_foundry/data/preprocessing/hf_utils/hf_dataset_downloader.py --dataset IPEC-COMMUNITY/droid_lerobot --mode s3 --s3-output-path s3://your-bucket/your-path/hf_datasets/droid_lerobot --local-output-dir /datasets/hf_datasets/droid_lerobot --preserve-structure
```

# Converting VLM Hugging Face captions to tar shards

We use [img2dataset](https://github.com/rom1504/img2dataset) to handle image downloading and webdataset shard creation. 

This assumes that the HF dataset is already downloaded to S3 (see above section).

```bash
python vla_foundry/data/preprocessing/preprocess_captionshf_to_tar.py --cluster ray --input_path s3://your-bucket/your-path/vla_foundry_scratch/downloads/ --output_path s3://your-bucket/your-path/vla_foundry_scratch/downloads2/ --url_col images --caption_col texts --save_additional_columns metadata
```

# Converting a text Hugging Face dataset to tar shards
```bash
python vla_foundry/data/preprocessing/preprocess_untokenized_to_tar.py --s3_input_path s3://your-bucket/your-path/hf_datasets/fineweb-edu-350BT --s3_output_path s3://your-bucket/your-path/datasets/text/fineweb-edu-350BT --tmp_dir /tmp/finewebshards
```

# Converting LeRobot to tar shards

The preprocessor reads and writes either local paths or S3 URIs. The example
below uses Physical Intelligence's publicly-available `pi_libero` LeRobot
dataset and streams through S3 (the typical cluster workflow).

First, stage the HF dataset in your own S3 bucket:

```bash
hf download physical-intelligence/pi_libero --repo-type dataset \
    --local-dir /tmp/pi_libero
aws s3 sync /tmp/pi_libero s3://your-bucket/your-path/hf_datasets/pi_libero/
```

Then run the preprocessor (swap `s3://...` for local paths to run fully
offline — see `tutorials/lerobot.ipynb` for the local workflow):

```bash
source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "lerobot" \
--source_episodes "['s3://your-bucket/your-path/hf_datasets/pi_libero/']" \
--output_dir s3://your-bucket/your-path/vla_foundry_scratch/lerobotdata/pi_libero/ \
--camera_names "['image', 'wrist_image']" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml" \
--observation_keys "['state']" \
--action_keys "['actions']"
```

# Converting LBM Spartan data to tar shards

First, download the public raw Spartan data for one or more tasks using the
included downloader. It pulls from `https://tri-ml-public.s3.amazonaws.com/`
— no credentials needed. Run `--list` to see all available public tasks.

```bash
python vla_foundry/data/scripts/download_dataset.py --task BimanualPutRedBellPepperInBin --local_path /tmp/raw_data --raw
python vla_foundry/data/scripts/download_dataset.py --task BimanualPlaceAppleFromBowlIntoBin --local_path /tmp/raw_data --raw
python vla_foundry/data/scripts/download_dataset.py --task BimanualStackPlatesOnTableFromDryingRack --local_path /tmp/raw_data --raw
```

After extraction, each task's episodes live at `/tmp/raw_data/tasks/<TaskName>/`.
The `--source_episodes` flag accepts a list of episode directories — passing
multiple tasks merges them into a single output dataset (with per-task
language annotations preserved via `--language_annotations_path`).

```bash
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    '/tmp/raw_data/tasks/BimanualPutRedBellPepperInBin/',
    '/tmp/raw_data/tasks/BimanualPlaceAppleFromBowlIntoBin/',
    '/tmp/raw_data/tasks/BimanualStackPlatesOnTableFromDryingRack/',
    ]" \
--output_dir /tmp/preprocessed/lbm_multitask/ \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \
--language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
--action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
```

Swap the local paths for `s3://...` URIs to read/write remotely instead.
