#!/bin/bash
# Train VLADiffusion model with Qwen/Qwen3-VL-2B-Thinking encoder on robotics data

uv run torchrun --nproc_per_node=1 --nnodes=1 vla_foundry/main.py \
--hparams.torchcompile False \
--config_path vla_foundry/config_presets/training_jobs/vla_diffusion_bellpepper.yaml \
--remote_sync s3://your-bucket/your-path/model_checkpoints/vla_diffusion \
--num_checkpoints 5 \
--total_train_samples 1000 \
--model.vision_language_backbone.hf_pretrained "Qwen/Qwen3-VL-2B-Thinking" \
--data.processor "Qwen/Qwen3-VL-2B-Thinking" \
"$@"
