#!/bin/bash
.venv/bin/torchrun --nproc_per_node=2 --nnodes=1 vla_foundry/main.py \
--config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
--remote_sync s3://your-bucket/your-path/vla_foundry/model_checkpoints/diffusion_policy \
--num_checkpoints 5 \
--total_train_samples 100000 \
"$@"