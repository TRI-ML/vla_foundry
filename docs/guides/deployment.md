# Deployment

This guide covers deploying and evaluating LBM robotics policies using the
inference utilities provided in VLA Foundry.

## Overview

The `vla_foundry/inference/scripts/` directory provides lightweight scripts
for serving trained LBM robotics policies over gRPC. Each script is ready
to run with `uv` so that you can reuse the repository's managed environment.

| Script | Description |
|---|---|
| `launch_wave_policy.sh` | Launches a dummy gRPC policy server that waves the robot end-effectors in a simple sinusoidal pattern. Useful for verifying the gRPC stack. |
| `launch_inference_policy.sh` | Downloads a DiffusionPolicy checkpoint from S3 (if needed) and launches a gRPC policy server that uses it. |

## Prerequisites

Before running the deployment scripts, ensure the following:

- Complete the project setup in the repository root (see main README for `uv sync --frozen` instructions).
- Install the `inference` group: `uv sync --group inference`.
- Provide any required credentials (e.g., AWS, W&B, Hugging Face tokens) for accessing checkpoints or datasets referenced in your configuration.
- Run scripts from the repository root.

## Wave Policy Demo

The wave policy is a deterministic scripted policy that validates the gRPC
stack.

### Launch the wave policy server

```bash
bash vla_foundry/inference/scripts/launch_wave_policy.sh
```

Key behavior:

- Starts a gRPC server that streams sinusoidal joint poses to connected
  clients until interrupted.

## Inference Policy Demo

`launch_inference_policy.sh` handles both the checkpoint download and the
server launch in a single step.

### Launch the inference policy server

From the repository root:

```bash
bash vla_foundry/inference/scripts/launch_inference_policy.sh <experiment_name>
```

For example:

```bash
bash vla_foundry/inference/scripts/launch_inference_policy.sh \
  2025_11_05-21_34_11-model_diffusion_policy-lr_5e-05-bsz_1024
```

To pin a specific checkpoint number (the highest available is selected by
default):

```bash
bash vla_foundry/inference/scripts/launch_inference_policy.sh <experiment_name> 5
```

The S3 root and download destination can be overridden via the
`VLA_MODELS_S3` and `VLA_INFERENCE_DEST_DIR` environment variables. See
`vla_foundry/inference/scripts/README.md` for the full list of options.

### Connecting a client

These scripts only start the policy server. To exercise it end-to-end,
point your gRPC simulation client (for example,
[`lbm_eval`](https://github.com/ToyotaResearchInstitute/lbm_eval)) at the
running server.
