# Quickstart

The main entrypoint is `vla_foundry/main.py`. This page walks through running your first training job.

## Minimal Example: LLM Training

The simplest starting point is training a small transformer:

```bash
.venv/bin/torchrun --nproc_per_node=1 --nnodes=1 vla_foundry/main.py \
    --model "include vla_foundry/config_presets/models/transformer_11m.yaml" \
    --data.type text \
    --data.dataset_manifest '["s3://path-to-your-dataset/manifest.jsonl"]' \
    --data.dataset_modality '["text"]' \
    --data.dataset_weighting '[1.0]' \
    --data.seq_len 2048 \
    --total_train_samples 1_000_000 \
    --num_checkpoints 5 \
    --hparams.per_gpu_batch_size 8 \
    --hparams.global_batch_size 8 \
    --distributed.fsdp False
```

See [tutorials/training_llm_vlm_vla.ipynb](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/training_llm_vlm_vla.ipynb) for an end-to-end walkthrough.

## VLM Training

Train a 3B VLM with PaliGemma:

```bash
.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 vla_foundry/main.py \
    --model "include vla_foundry/config_presets/models/vlm_3b.yaml" \
    --model.vit "include vla_foundry/config_presets/models/vit_paligemma.yaml" \
    --data.type image_caption \
    --data.processor google/paligemma-3b-pt-224 \
    --data.dataset_manifest '["s3://your-dataset/manifest.jsonl"]' \
    --data.dataset_modality '["image_caption"]' \
    --data.dataset_weighting '[1.0]' \
    --data.img_num_tokens 256 \
    --total_train_samples 14_000_000 \
    --num_checkpoints 5 \
    --hparams.per_gpu_batch_size 2 \
    --hparams.global_batch_size 64 \
    --remote_sync s3://your-bucket/vlm_paligemma_3b
```

## Robotics Policy Training

Train a Diffusion Policy for robotics:

```bash
.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
    --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
    --remote_sync s3://your-bucket/diffusion_policy \
    --num_checkpoints 5 \
    --total_train_samples 100000
```

See [tutorials/lerobot.ipynb](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/lerobot.ipynb) and [tutorials/training_llm_vlm_vla.ipynb](https://github.com/TRI-ML/vla_foundry/blob/main/tutorials/training_llm_vlm_vla.ipynb) for end-to-end robotics training walkthroughs.

## Using Config Presets

Config presets simplify command-line arguments. Use the `include` keyword:

```bash
--model "include vla_foundry/config_presets/models/vlm_3b.yaml"
--config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml
```

Command-line arguments always take precedence over preset values.

See [Configuration](../concepts/configuration.md) for full details on the config system.

## Resolving Configs

Preview the resolved configuration before training:

```bash
.venv/bin/torchrun --nproc_per_node=1 --nnodes=1 vla_foundry/main.py \
    --config_path your_config.yaml \
    --resolve_configs True \
    --resolve_configs_path ./
```

This prints the fully resolved config and optionally saves it to `resolved_config.yaml`.

## Running on SageMaker

Create a `secrets.env` file in the project root:

```bash
WANDB_API_KEY=<your wandb key>
HF_TOKEN=<your hf token>
```

Launch with the SageMaker wrapper:

```bash
uv run --group sagemaker sagemaker/launch_training.py \
    --sagemaker.user your.user.name \
    --sagemaker.instance_count 1 \
    --sagemaker.instance_type p4de \
    --model "include vla_foundry/config_presets/models/vlm_3b.yaml" \
    # ... other training arguments
```

!!! note
    Use `uv run --group sagemaker` instead of `torchrun` for SageMaker launches. The SageMaker argument parser wraps the main one, so all training arguments work the same way with a `--sagemaker.` prefix for SageMaker-specific options.

## Next Steps

- [Architecture Overview](../concepts/architecture.md) — understand how VLA Foundry is structured
- [Configuration System](../concepts/configuration.md) — deep dive into draccus config
- [Adding New Models](../guides/adding-new-models.md) — add your own model architecture
- [Training Loop](../concepts/training-loop.md) — understand batch sizing, checkpoints, and resuming
