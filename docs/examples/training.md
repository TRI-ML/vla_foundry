# Training Examples

This page walks through the annotated training scripts in the
`examples/training/` directory. Each example shows the full command with
explanations for key flags.

## Key Flags Reference

Before diving into examples, here is a summary of the most important
training flags:

| Flag | Description |
|------|-------------|
| `--total_train_samples` | Total number of training samples to process across the entire run. Training ends when this count is reached. |
| `--num_checkpoints` | Number of evenly-spaced checkpoints to save during training. |
| `--hparams.per_gpu_batch_size` | Micro-batch size per GPU. Determines memory usage per device. |
| `--hparams.global_batch_size` | Effective batch size across all GPUs and gradient accumulation steps. Must be a multiple of `per_gpu_batch_size * nproc_per_node * nnodes`. |
| `--remote_sync` | S3 path where checkpoints and logs are uploaded. |
| `--config_path` | Path to a YAML config preset that sets all parameters for a training job. CLI flags override preset values. |
| `--model "include ..."` | Inline include of a YAML model config preset. |
| `--distributed.fsdp` | Enable Fully Sharded Data Parallel (FSDP2) training. |
| `--wandb` | Enable Weights & Biases logging. |
| `--resolve_configs` | Print the fully resolved config and optionally save it (useful for debugging). |

---

## LLM Training (transformer_11m) — Minimal Example

The simplest training script. Trains an 11M-parameter transformer on text
data using 3 GPUs.

**Source:** `examples/training/llm_11m.sh`

```bash
.venv/bin/torchrun --nproc_per_node=3 --nnodes=1 vla_foundry/main.py \
    --model.type transformer \                                              # (1)!
    --model "include vla_foundry/config_presets/models/transformer_11m.yaml" \  # (2)!
    --model.cast_output_to_float32 True \                                   # (3)!
    --distributed.fsdp True \                                               # (4)!
    --data.type text \                                                      # (5)!
    --data.dataset_manifest ["s3://your-bucket/your-path/your_llm_pretraining_dataset/manifest.jsonl"] \  # (6)!
    --data.dataset_modality ["text"] \
    --data.dataset_weighting [1.0] \                                        # (7)!
    --data.seq_len 2048 \                                                   # (8)!
    --total_train_samples 14_000_000 \                                      # (9)!
    --num_checkpoints 5 \                                                   # (10)!
    --hparams.per_gpu_batch_size 32 \                                       # (11)!
    --hparams.global_batch_size 96 \                                        # (12)!
    --remote_sync s3://your-bucket/your-path/vla_foundry_scratch/models/llm_11m \  # (13)!
    --resolve_configs True \                                                # (14)!
    --resolve_configs_path ./ \
    "$@"
```

1. Model architecture type — `transformer` for native LLM.
2. Include a YAML preset that defines the 11M-param transformer config (layers, hidden dim, heads, etc.).
3. Cast the output logits to float32 for numerically stable loss computation.
4. Enable FSDP2 for distributed training across GPUs.
5. Data modality — `text` for pre-tokenized text data.
6. S3 path to the WebDataset manifest file listing all tar shards.
7. Dataset weight — used for mixing multiple datasets; `1.0` means this dataset gets 100% of samples.
8. Sequence length in tokens per sample.
9. Stop training after processing 14 million samples total.
10. Save 5 evenly-spaced checkpoints over the course of training.
11. Each GPU processes 32 samples per micro-batch.
12. Effective batch size is 96 (= 32 per GPU × 3 GPUs, no gradient accumulation here).
13. Upload checkpoints and logs to this S3 path.
14. Print the fully resolved config for debugging; save to `./resolved_config.yaml`.

---

## VLM Training with PaliGemma

Train a 3B Vision-Language Model using a PaliGemma processor and
image-caption data.

**Source:** `examples/training/vlm_paligemma3b.sh`

```bash
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
.venv/bin/torchrun --nproc_per_node=3 --nnodes=1 vla_foundry/main.py \
    --model "include vla_foundry/config_presets/models/vlm_3b.yaml" \       # (1)!
    --distributed.fsdp True \
    --data.type image_caption \                                             # (2)!
    --data.processor google/paligemma-3b-pt-224 \                           # (3)!
    --data.dataset_manifest ["s3://your-bucket/your-path/datasets/datacompdr_1b/manifest.jsonl"] \
    --data.dataset_modality ["image_caption"] \
    --data.dataset_weighting [1.0] \
    --data.seq_len 2048 \
    --data.img_num_tokens 256 \                                             # (4)!
    --total_train_samples 14_000_000 \
    --num_checkpoints 5 \
    --hparams.per_gpu_batch_size 1 \                                        # (5)!
    --hparams.global_batch_size 3 \
    --remote_sync s3://your-bucket/your-path/vla_foundry_scratch/models/vlm_paligemma_3b \
    "$@"
```

1. The `vlm_3b.yaml` preset defines the full VLM architecture including the vision encoder (ViT) and language model.
2. `image_caption` data type expects WebDataset shards with image-text pairs.
3. The processor handles image tokenization and text tokenization using the PaliGemma tokenizer.
4. Number of image tokens per image — PaliGemma uses 256 tokens for 224×224 images.
5. With a 3B model, per-GPU batch size is typically small to fit in memory.

---

## VLM Training Initialized from a Pretrained LLM

Train a SmolVLM-style VLM where the language-model tower is initialized from
a pretrained LLM checkpoint instead of random weights.

**Source:** `examples/training/vlm_smolvlm_full_fromllm.sh`

```bash
.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 vla_foundry/main.py \
    --model "include vla_foundry/config_presets/models/smolvlm_load_llm.yaml" \  # (1)!
    --model.transformer.resume_from_checkpoint s3://your-bucket/your-path/your_pretrained_llm_run/checkpoints/checkpoint_N.pt \  # (2)!
    --model.transformer.resume_weights_only True \                          # (3)!
    --distributed.fsdp True \
    --data.type image_caption \
    --data.processor HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
    --data.dataset_manifest ["s3://your-bucket/your-path/datasets/datacompdr_1b/manifest.jsonl"] \
    --data.dataset_modality ["image_caption"] \
    --data.dataset_weighting [1.0] \
    --data.seq_len 2048 \
    --data.img_num_tokens 64 \                                              # (4)!
    --data.image_size 224 \
    --model.vit.img_size 224 \
    --model.vit.patch_size 14 \
    --model.vit.projector_pixel_shuffle_factor 2 \
    --total_train_samples 50_000_000 \
    --num_checkpoints 10 \
    --hparams.per_gpu_batch_size 8 \
    --hparams.global_batch_size 512 \
    --hparams.torchcompile True \
    --remote_sync s3://your-bucket/your-path/vla_foundry_scratch/models/vlm_smolvlm_fromllm_samples50m
```

1. The `smolvlm_load_llm.yaml` preset is configured to expect an external LLM checkpoint for the transformer tower.
2. Point at a previously-trained LLM checkpoint to initialize the language tower weights.
3. `resume_weights_only=True` means the optimizer state, scheduler, and training step are **not** restored — only the weights. This is the transfer-learning pattern (as opposed to resuming an interrupted run).
4. SmolVLM uses fewer image tokens (64) at 224×224 with patch 14 and pixel-shuffle factor 2 — smaller image-token budget than PaliGemma's 256.

---

## VLA DiffusionPolicy (PaliGemma2 backbone)

Train a vision-language-action DiffusionPolicy with a PaliGemma2 backbone on
the `BimanualPutRedBellPepperInBin` robotics task.

**Source:** `examples/training/vla_diffusion_redbellpepper_paligemma2.sh`

```bash
.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
    --hparams.torchcompile True \
    --config_path vla_foundry/config_presets/training_jobs/vla_diffusion_bellpepper.yaml \  # (1)!
    --remote_sync s3://your-bucket/your-path/vla_foundry/model_checkpoints/vla_diffusion \
    --num_checkpoints 5 \
    --total_train_samples 1000 \
    "$@"
```

1. The `vla_diffusion_bellpepper.yaml` preset bundles the DiffusionPolicy head, the PaliGemma2 VLM backbone, the Spartan data pipeline, and the LBM hyperparameters.

## VLA DiffusionPolicy (Qwen3-VL-2B-Thinking backbone)

Same VLA recipe, but swap in a Qwen3-VL-2B-Thinking backbone at the CLI:

**Source:** `examples/training/vla_diffusion_redbellpepper_qwen_2b_thinking.sh`

```bash
uv run torchrun --nproc_per_node=1 --nnodes=1 vla_foundry/main.py \
    --hparams.torchcompile False \
    --config_path vla_foundry/config_presets/training_jobs/vla_diffusion_bellpepper.yaml \
    --remote_sync s3://your-bucket/your-path/model_checkpoints/vla_diffusion \
    --num_checkpoints 5 \
    --total_train_samples 1000 \
    --model.vision_language_backbone.hf_pretrained "Qwen/Qwen3-VL-2B-Thinking" \  # (1)!
    --data.processor "Qwen/Qwen3-VL-2B-Thinking" \                                # (2)!
    "$@"
```

1. Override the backbone from the YAML preset by pointing `hf_pretrained` at a different HF model.
2. The data processor must match the backbone's tokenizer/image processor.

---

## Diffusion Policy Training with config_path

Train a standalone DiffusionPolicy (no VLM) for robotics using a single YAML
config preset.

**Source:** `examples/training/diffusion_policy.sh`

```bash
.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
    --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \  # (1)!
    --remote_sync s3://your-bucket/your-path/vla_foundry/model_checkpoints/diffusion_policy \
    --num_checkpoints 5 \
    --total_train_samples 100000 \
    "$@"
```

1. The `config_path` flag loads a complete training job config — model architecture, data pipeline, hyperparameters, and all. CLI flags like `--total_train_samples` override preset values.

!!! note
    When using `--config_path`, you typically only need to specify overrides on the command line. The YAML preset contains all default values for the model, data, and hyperparameters.

---

## Finetuning from a Pretrained Checkpoint

Resume training from an existing checkpoint, loading only the model weights
(not the optimizer state or training step).

**Source:** `examples/training/resume.sh`

```bash
.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 vla_foundry/main.py \
    --config_path vla_foundry/config_presets/training_jobs/lbm_hparams_4cams.yaml \
    --remote_sync s3://your-bucket/model_checkpoints/diffusion_policy \
    --num_checkpoints 5 \
    --total_train_samples 1_000_000 \
    --data.dataset_manifest "['s3://...manifest.jsonl']" \
    --data.dataset_statistics "['s3://...stats.json']" \
    --data.dataset_modality "['robotics']" \
    --data.dataset_weighting "[1.0]" \
    --model.resume_from_checkpoint "s3://...checkpoint_3_converted.pt" \    # (1)!
    --model.resume_weights_only True \                                      # (2)!
    --hparams.per_gpu_batch_size 128 \
    --hparams.global_batch_size 1024
```

1. S3 path to a previously saved checkpoint file. The model weights are loaded from this checkpoint.
2. When `True`, only model weights are loaded — the optimizer state, learning rate schedule, and training step counter are **not** restored. This is useful for finetuning on a new dataset or task with a fresh optimizer.

!!! tip "Full resume vs. weight-only resume"
    - **Full resume** (`resume_weights_only=False`): Restores everything (weights, optimizer, scheduler, step). Use this to continue a training run that was interrupted.
    - **Weight-only resume** (`resume_weights_only=True`): Only restores model weights. Use this for finetuning or transfer learning where you want a fresh training state.

---

## Running on SageMaker

For cluster-scale training, wrap any of the above commands with
`sagemaker/launch_training.py`. The launcher builds and pushes a Docker
image, then submits the job to SageMaker via `estimator.fit()`.

```bash
uv run --group sagemaker sagemaker/launch_training.py \
    --sagemaker.user firstname.lastname \
    --sagemaker.instance_count 1 \
    --sagemaker.instance_type p5en \
    --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
    --remote_sync s3://your-bucket/your-path/model_checkpoints/diffusion_policy
```

See [`sagemaker/README.md`](https://github.com/TRI-ML/vla_foundry/blob/main/sagemaker/README.md) for the full list of `--sagemaker.*` options and prerequisites.

!!! warning "Secrets file required"
    Create a `secrets.env` file in the project root with your `WANDB_API_KEY` and `HF_TOKEN` before launching SageMaker jobs.
