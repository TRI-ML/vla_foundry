#!/bin/bash
.venv/bin/torchrun --nproc_per_node=8 --nnodes=1 vla_foundry/main.py \
--config_path vla_foundry/config_presets/training_jobs/lbm_hparams_4cams.yaml \
--remote_sync s3://your-bucket/model_checkpoints/diffusion_policy \
--num_checkpoints 5 \
--total_train_samples 1_000_000 \
--data.dataset_manifest "['s3://...manifest.jsonl']" \
--data.dataset_statistics "['s3://...stats.json']" \
--data.dataset_modality "['robotics']" \
--data.dataset_weighting "[1.0]" \
--model.resume_from_checkpoint "s3://...checkpoint_3_converted.pt" \
--model.resume_weights_only True \
--hparams.per_gpu_batch_size 128 \
--hparams.global_batch_size 1024 \
"$@"
