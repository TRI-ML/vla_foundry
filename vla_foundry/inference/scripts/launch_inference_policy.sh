#!/usr/bin/env bash
# Download a trained policy checkpoint from S3 (if missing) and launch it as a
# gRPC inference server using DiffusionPolicy.
#
# Usage:
#   launch_inference_policy.sh <experiment_name> [checkpoint_number] [s3_base]
#
# Arguments:
#   experiment_name    Name of the experiment directory under the S3 model root.
#   checkpoint_number  (Optional) Specific checkpoint number to use. If omitted,
#                      the highest numbered checkpoint in S3 is selected.
#   s3_base            (Optional) S3 root for model checkpoints. Defaults to the
#                      VLA_MODELS_S3 env var, or
#                      s3://your-bucket/your-path/vla_foundry/model_checkpoints/diffusion_policy
#
# Environment:
#   AWS_PROFILE        (Optional) AWS profile used by `aws s3 cp` calls.
#   CUDA_VISIBLE_DEVICES  Defaults to 0 if unset.

set -euo pipefail

if [ $# -eq 0 ]; then
    sed -n '2,20p' "$0"
    exit 1
fi

EXPERIMENT_NAME="$1"
CHECKPOINT_NUMBER="${2:-}"
S3_BASE="${3:-${VLA_MODELS_S3:-s3://your-bucket/your-path/vla_foundry/model_checkpoints/diffusion_policy}}"
DESTINATION_DIR="${VLA_INFERENCE_DEST_DIR:-.}"

AWS_PROFILE_ARGS=()
if [ -n "${AWS_PROFILE:-}" ]; then
    AWS_PROFILE_ARGS=(--profile "$AWS_PROFILE")
fi

S3_PATH="$S3_BASE/$EXPERIMENT_NAME"
EXP_DIR="$DESTINATION_DIR/experiments/$EXPERIMENT_NAME"
CKPT_DIR="$EXP_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

# --- Download config files ------------------------------------------------
CONFIG_FILES=(
    "config.yaml"
    "config_normalizer.yaml"
    "config_processor.yaml"
    "stats.json"
    "preprocessing_config.yaml"
)
for file in "${CONFIG_FILES[@]}"; do
    if [ ! -f "$EXP_DIR/$file" ]; then
        echo "Downloading $file..."
        aws s3 cp "$S3_PATH/$file" "$EXP_DIR/$file" "${AWS_PROFILE_ARGS[@]}" || true
    fi
done

# Legacy file names (older checkpoints).
[ -f "$EXP_DIR/stats.json" ] || aws s3 cp "$S3_PATH/stats_normalizer.json" "$EXP_DIR/stats.json" "${AWS_PROFILE_ARGS[@]}" || true
[ -f "$EXP_DIR/preprocessing_config.yaml" ] || aws s3 cp "$S3_PATH/preprocessing_configs.yaml" "$EXP_DIR/preprocessing_config.yaml" "${AWS_PROFILE_ARGS[@]}" || true

# --- Resolve checkpoint number -------------------------------------------
if [ -z "$CHECKPOINT_NUMBER" ]; then
    echo "Finding highest checkpoint in S3..."
    CHECKPOINT_NUMBER=$(aws s3 ls "$S3_PATH/checkpoints/" "${AWS_PROFILE_ARGS[@]}" \
        | grep "checkpoint_.*\.pt" \
        | sed 's/.*checkpoint_\([0-9]*\)\.pt.*/\1/' \
        | sort -n | tail -1)
    if [ -z "$CHECKPOINT_NUMBER" ]; then
        echo "ERROR: no checkpoint files found at $S3_PATH/checkpoints/" >&2
        exit 1
    fi
fi
echo "Using checkpoint number: $CHECKPOINT_NUMBER"

# --- Download checkpoint (skip if already present) -----------------------
if [ ! -f "$CKPT_DIR/checkpoint_${CHECKPOINT_NUMBER}.pt" ]; then
    echo "Downloading checkpoint_${CHECKPOINT_NUMBER}.pt..."
    aws s3 sync "$S3_PATH/checkpoints" "$CKPT_DIR" \
        --exclude "*" \
        --include "checkpoint_${CHECKPOINT_NUMBER}.pt" \
        --include "ema_${CHECKPOINT_NUMBER}.pt" \
        "${AWS_PROFILE_ARGS[@]}"
fi

# --- Launch the gRPC inference server ------------------------------------
echo "Launching inference policy server for $EXPERIMENT_NAME..."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run --group inference python vla_foundry/inference/robotics/inference_policy.py \
    --checkpoint_directory "$EXP_DIR" \
    --num_flow_steps 8 \
    --device cuda \
    --open_loop_steps 8 \
    --gripper_debounce_open_threshold 0.6 \
    --gripper_debounce_close_threshold 0.4
