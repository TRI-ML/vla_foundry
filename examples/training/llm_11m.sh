#!/bin/bash
.venv/bin/torchrun --nproc_per_node=3 --nnodes=1 vla_foundry/main.py \
--model.type transformer \
--model "include vla_foundry/config_presets/models/transformer_11m.yaml" \
--model.cast_output_to_float32 True \
--distributed.fsdp True \
--data.type text \
--data.dataset_manifest ["s3://your-bucket/your-path/your_llm_pretraining_dataset/manifest.jsonl"] \
--data.dataset_modality ["text"] \
--data.dataset_weighting [1.0] \
--data.seq_len 2048 \
--total_train_samples 14_000_000 \
--num_checkpoints 5 \
--hparams.per_gpu_batch_size 32 \
--hparams.global_batch_size 96 \
--remote_sync s3://your-bucket/your-path/vla_foundry_scratch/models/llm_11m \
--resolve_configs True \
--resolve_configs_path ./ \
"$@"