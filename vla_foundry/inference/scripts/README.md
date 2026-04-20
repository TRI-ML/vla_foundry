# Inference Scripts

Shell helpers for serving trained VLA policies over gRPC. All scripts assume
they are run from the repository root and rely on the `inference` dependency
group (`uv sync --group inference`).

## Scripts

### `launch_inference_policy.sh`

Downloads a trained DiffusionPolicy checkpoint from S3 (if not already
present) and starts a gRPC inference server that serves it.

```bash
bash vla_foundry/inference/scripts/launch_inference_policy.sh <experiment_name> [checkpoint_number]
```

Example:

```bash
bash vla_foundry/inference/scripts/launch_inference_policy.sh \
    2026_01_12-21_27_03-model_diffusion_policy-lr_5e-05-bsz_1024
```

Optional environment variables:

- `VLA_MODELS_S3` — S3 root for model checkpoints. Defaults to a placeholder
  path; set this (or pass it as the third positional argument) to point at
  your own bucket.
- `AWS_PROFILE` — AWS profile used for `aws s3 cp` calls.
- `CUDA_VISIBLE_DEVICES` — defaults to `0`.
- `VLA_INFERENCE_DEST_DIR` — local directory where checkpoints are written
  (defaults to the current directory).

### `launch_wave_policy.sh`

Starts a deterministic scripted policy that streams sinusoidal joint poses
over gRPC. Useful for validating the gRPC stack without a trained model.

```bash
bash vla_foundry/inference/scripts/launch_wave_policy.sh
```

## Connecting a Client

These scripts only start the policy server. To exercise it end-to-end, point
your gRPC simulation client (for example,
[`lbm_eval`](https://github.com/ToyotaResearchInstitute/lbm_eval)) at the
running server.
