# LBM Tutorial: Train and Evaluate a Diffusion Policy

This tutorial walks through training a diffusion policy on the **PickAndPlaceBox** task — a single-arm pick-and-place task with 5,175 sequences (~1.4 GB). You'll download the data, train a model, and run it in simulation.

> 42 tasks are available in total. Run `python vla_foundry/data/scripts/download_dataset.py --list` to see them all. Replace `PickAndPlaceBox` with any task name to train on a different one.

## 1. Install

```bash
uv sync
```

## 2. Download the dataset

```bash
python vla_foundry/data/scripts/download_dataset.py \
  --task PickAndPlaceBox \
  --local_path tutorials/data/PickAndPlaceBox
```

This downloads pre-processed WebDataset shards, normalization statistics, and a manifest. After download the manifest is rewritten with local paths — ready for training.

The download resumes if interrupted and skips files already present. Add `--dry_run` to preview sizes first.

## 3. Train

```bash
.venv/bin/torchrun --nproc_per_node=1 --nnodes=1 vla_foundry/main.py \
  --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
  --data.dataset_manifest '["tutorials/data/PickAndPlaceBox/manifest.jsonl"]' \
  --data.dataset_statistics '["tutorials/data/PickAndPlaceBox/stats.json"]' \
  --total_train_samples 5000 \
  --num_checkpoints 1 \
  --data.num_workers 2 \
  --save_path tutorials/checkpoints/pick_and_place_box \
  --wandb false
```

This trains a diffusion policy for 5,000 samples (~1 epoch) and saves one checkpoint. On a single GPU it takes a few minutes.

For a real training run, increase `--total_train_samples` (e.g. 100000), add more GPUs with `--nproc_per_node`, and enable W&B logging with `--wandb true`.

Training creates a timestamped directory under `tutorials/checkpoints/pick_and_place_box/` containing checkpoints, configs, and logs.

## 4. Evaluate in simulation

Evaluation uses two processes: a **simulation container** (Docker) and a **policy server** (host GPU).

### 4a. Pull the simulation image

```bash
docker pull toyotaresearch/lbm-eval-oss:vla-foundry
```

Requires Docker with NVIDIA Container Toolkit.

### 4b. Start the simulation

In one terminal:

```bash
mkdir -p tutorials/rollouts

docker run --rm --network host \
  --runtime=nvidia --gpus all \
  --device /dev/dri \
  --group-add video \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RECORD_VIDEO=1 \
  -v "$(pwd)/tutorials/rollouts:/tmp/lbm/rollouts" \
  toyotaresearch/lbm-eval-oss:vla-foundry \
  bash /opt/lbm_eval/launch_sim.sh PickAndPlaceBox
```

### 4c. Start the policy server

In another terminal, find your experiment directory and start the server:

```bash
uv sync --group inference

# Find the experiment directory (it has a timestamp in the name)
EXPERIMENT_DIR=$(ls -td tutorials/checkpoints/pick_and_place_box/*/ | head -1)

CUDA_VISIBLE_DEVICES=0 uv run --group inference \
  python vla_foundry/inference/robotics/inference_policy.py \
    --checkpoint_directory "$EXPERIMENT_DIR" \
    --device cuda
```

The policy server loads the latest checkpoint and listens on `localhost:50051`. The simulation sends observations, the policy returns actions.

### 4d. View results

Videos and results appear in `tutorials/rollouts/<timestamp>/pick_and_place_box/demonstration_<N>/`.

---

## Going further

**Bigger datasets** — Bimanual tasks (e.g. `BimanualPutRedBellPepperInBin`, 38k sequences) are more challenging and require more training.

**More GPUs** — Use `--nproc_per_node=N` for multi-GPU training.

**Finetuning** — Resume from a pretrained checkpoint:
```bash
--model.resume_from_checkpoint path/to/checkpoint_N.pt \
--model.resume_weights_only True
```
These flags can also be set in your training config YAML. The checkpoint path is validated early in training to fail fast with a clear error if invalid. Use `--model.resume_weights_only=False` (default) to fully resume training including optimizer state.

**Inference tuning** — Key policy server flags:
- `--num_flow_steps 8` — diffusion denoising steps (more = better quality, slower)
- `--open_loop_steps 4` — steps to execute before re-planning
- `DEBUG=1` — enable verbose logging

**Raw data** — To preprocess from raw Spartan episodes instead of using pre-processed shards:
```bash
# Download raw data (~70 GB per task)
python vla_foundry/data/scripts/download_dataset.py --task PickAndPlaceBox --local_path data/raw --raw

# Preprocess to WebDataset shards
uv sync --group preprocessing
python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type "spartan" \
  --source_episodes "['data/raw/tasks/PickAndPlaceBox/']" \
  --output_dir tutorials/data/PickAndPlaceBox_preprocessed/ \
  --camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \
  --language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
  --action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
  --data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
  --samples_per_shard 100 \
  --config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
```

The raw data follows the [Spartan format](https://github.com/ToyotaResearchInstitute/lbm_eval/blob/main/TRAINING_DATA_FORMAT.md): NPZ archives with actions (20-element bimanual vectors), observations (6 camera views at 10 Hz), and calibration data. See [preprocessing README](vla_foundry/data/preprocessing/README.md) for Ray cluster setup.
