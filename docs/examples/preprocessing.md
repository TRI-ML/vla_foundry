# Preprocessing Examples

This page walks through the data preprocessing scripts that convert raw
datasets into the WebDataset tar shard format used by VLA Foundry at
training time. Most robotics preprocessing is handled by a single entrypoint
(`preprocess_robotics_to_tar.py`) with a `--type` flag that selects the
source format.

---

## HF Dataset Download to S3

Download a Hugging Face dataset to S3 using Ray for parallel transfers.

**Script:** `vla_foundry/data/preprocessing/hf_utils/hf_dataset_downloader.py`

```bash
source .venv/bin/activate && python \
    vla_foundry/data/preprocessing/hf_utils/hf_dataset_downloader.py \
    --repo_id <hf-dataset-repo-id> \                # (1)!
    --output_dir s3://your-bucket/your-path/hf_datasets/      # (2)!
```

1. The Hugging Face dataset repository ID (e.g., `lerobot/aloha_sim_insertion_human`).
2. S3 or local path where the downloaded dataset files are stored.

This utility downloads all files from a HF dataset repo and optionally
uploads them to S3 in parallel using Ray workers. It is typically a first
step before converting HF data into tar shards.

---

## VLM HF Captions to Tar Shards (img2dataset)

Convert a Hugging Face image-caption dataset (in Parquet format) into
WebDataset tar shards using `img2dataset`.

**Script:** `vla_foundry/data/preprocessing/preprocess_captionshf_to_tar.py`

```bash
source .venv/bin/activate && python \
    vla_foundry/data/preprocessing/preprocess_captionshf_to_tar.py \
    --cluster local \                               # (1)!
    --input_path s3://your-bucket/your-path/hf_dataset/ \     # (2)!
    --output_path s3://your-bucket/your-path/shards/ \        # (3)!
    --url_col url \                                 # (4)!
    --caption_col re_caption                        # (5)!
```

1. Run locally (`local`) or distributed on a Ray cluster (`ray`).
2. S3 or local path to the Parquet files containing image URLs and captions.
3. Output path for the WebDataset tar shards.
4. Column name in the Parquet file that contains the image URLs.
5. Column name that contains the caption text.

Under the hood, this uses the `img2dataset` library to download images,
resize them (keeping aspect ratio, max 512px), and encode them as WebP files
in WebDataset shards of 512 samples each.

---

## Text HF to Tar Shards

Convert Parquet-format text datasets from S3 into WebDataset tar shards with
JSON files.

**Script:** `vla_foundry/data/preprocessing/preprocess_text_untokenized_to_tar.py`

```bash
source .venv/bin/activate && python \
    vla_foundry/data/preprocessing/preprocess_text_untokenized_to_tar.py \
    --input_path s3://your-bucket/your-path/text_parquets/ \  # (1)!
    --output_path s3://your-bucket/your-path/text_shards/ \   # (2)!
    --samples_per_shard 512                         # (3)!
```

1. S3 path to the directory containing Parquet files with raw text data.
2. Output path for the tar shards. A `manifest.jsonl` is automatically created.
3. Number of text samples per tar shard.

Each Parquet row becomes a JSON file inside the tar shard, identified by a
UUID filename. This format is consumed by the `text_untokenized` data type
in VLA Foundry, which tokenizes on-the-fly during training.

---

## LeRobot to Tar Shards

Convert a LeRobot-format dataset into WebDataset tar shards for robotics
training.

**Source:** `examples/preprocessing/preprocess_robotics_data_lerobot.sh`

The preprocessor reads from and writes to either local paths or S3 URIs. The
example below shows the S3-to-S3 pattern that's typical for cluster training;
for a fully local walkthrough see `tutorials/lerobot.ipynb`.

First, stage the HF dataset in your own S3 bucket:

```bash
hf download physical-intelligence/pi_libero --repo-type dataset \
    --local-dir /tmp/pi_libero
aws s3 sync /tmp/pi_libero s3://your-bucket/your-path/hf_datasets/pi_libero/
```

Then run the preprocessor against the S3 paths:

```bash
source .venv/bin/activate && python \
    vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type "lerobot" \                              # (1)!
    --source_episodes "['s3://your-bucket/your-path/hf_datasets/pi_libero/']" \  # (2)!
    --output_dir s3://your-bucket/your-path/vla_foundry_datasets/lerobot/pi_libero/ \  # (3)!
    --camera_names "['image', 'wrist_image']" \     # (4)!
    --samples_per_shard 100 \                       # (5)!
    --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml" \  # (6)!
    --observation_keys "['state']" \                # (7)!
    --action_keys "['actions']"                     # (8)!
```

1. Source format type — `lerobot` for LeRobot HF datasets.
2. S3 path to the staged LeRobot dataset. Swap in a local directory (e.g., `./hf_datasets/pi_libero/`) to skip the S3 staging step entirely.
3. Output directory for the tar shards, stats, and manifest. Also accepts a local path.
4. Camera names to extract — must match the LeRobot dataset's column/video names. `pi_libero` has two cameras (`image`, `wrist_image`). Single-camera datasets like `lerobot/pusht` use `['observation.image']` instead.
5. Number of trajectory samples per tar shard file.
6. Preprocessing params YAML that specifies the number of past/future timesteps, chunk sizes, etc.
7. Observation keys to extract (proprioceptive state). `pi_libero` uses `state`; some other LeRobot datasets use `observation.state`.
8. Action key(s) to extract. `pi_libero` uses `actions` (plural); some other LeRobot datasets use `action` (singular).

---

## LBM Spartan to Tar Shards

Convert LBM Spartan-format simulation data into WebDataset tar shards.

**Source:** `examples/preprocessing/preprocess_robotics_data_lbm.sh`

First, download the public raw Spartan data for one or more tasks (pulled
from `https://tri-ml-public.s3.amazonaws.com/` — no credentials needed). Use
`--list` to see all available public tasks.

```bash
python vla_foundry/data/scripts/download_dataset.py --task BimanualPutRedBellPepperInBin --local_path /tmp/raw_data --raw
python vla_foundry/data/scripts/download_dataset.py --task BimanualPlaceAppleFromBowlIntoBin --local_path /tmp/raw_data --raw
python vla_foundry/data/scripts/download_dataset.py --task BimanualStackPlatesOnTableFromDryingRack --local_path /tmp/raw_data --raw
```

After extraction, each task's episodes live at `/tmp/raw_data/tasks/<TaskName>/`.
`--source_episodes` takes a **list** of episode directories — passing
multiple tasks merges them into a single output dataset, with per-task
language annotations preserved via `--language_annotations_path`:

```bash
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
    --type "spartan" \                              # (1)!
    --source_episodes "[
        '/tmp/raw_data/tasks/BimanualPutRedBellPepperInBin/',
        '/tmp/raw_data/tasks/BimanualPlaceAppleFromBowlIntoBin/',
        '/tmp/raw_data/tasks/BimanualStackPlatesOnTableFromDryingRack/',
        ]" \                                        # (2)!
    --output_dir /tmp/preprocessed/lbm_multitask/ \  # (3)!
    --camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \  # (4)!
    --language_annotations_path "vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml" \   # (5)!
    --action_fields_config_path "vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml" \          # (6)!
    --data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \       # (7)!
    --samples_per_shard 100 \
    --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
```

1. Source format type — `spartan` for LBM Spartan simulation data.
2. **List** of extracted Spartan task directories to merge into one output dataset. For single-task preprocessing, pass a 1-element list (e.g., `['/tmp/raw_data/tasks/BimanualPutRedBellPepperInBin/']`). Swap in `s3://...` URIs to read remotely.
3. Output directory for the processed tar shards (one unified manifest/stats across all input tasks). Swap in an S3 URI to write remotely.
4. Camera names loaded from a YAML config preset using the `include` syntax.
5. Path to a YAML file mapping tasks to language instructions (one file covers all tasks).
6. Path to a YAML file specifying which action fields to extract.
7. Keys to discard from the raw data during preprocessing.
