# SageMaker Training

`launch_training.py` submits a VLA Foundry training job to AWS SageMaker. It
builds a Docker image from `sagemaker/Dockerfile`, pushes it to ECR, and then
calls `Estimator.fit()` directly to start the job. The launcher's argument
parser wraps `vla_foundry/main.py`, so any training argument can be passed on
the command line as-is, with SageMaker-specific options under the
`--sagemaker.` prefix.

## Prerequisites

1. **AWS access.** You need an IAM role and credentials with permission to
   pull/push ECR images, create SageMaker training jobs, and read/write the
   S3 bucket you sync to. Set `AWS_PROFILE` (or pass `--sagemaker.profile`)
   to select the profile.
2. **SageMaker execution role.** Set the `SAGEMAKER_ARN` environment variable
   to the IAM role ARN that the training job will assume, or pass
   `--sagemaker.arn arn:aws:iam::<acct>:role/<role>`.
3. **Docker.** A working local Docker installation; it is used to build the
   training image before each launch.
4. **`uv` environment.** Sync the SageMaker dependency group:
   ```bash
   uv sync --group sagemaker
   ```
5. **`secrets.env` file in the repo root.** Each non-comment line should be of
   the form `KEY=VALUE`. Common entries:
   ```bash
   WANDB_API_KEY=<your wandb key>
   HF_TOKEN=<your huggingface token>
   ```
   Blank lines and lines starting with `#` are ignored. These entries are
   forwarded into the SageMaker container as environment variables. The
   launcher prints a usage message and exits early if this file is missing.

## Quick Start

```bash
uv run --group sagemaker sagemaker/launch_training.py \
    --sagemaker.user firstname.lastname \
    --sagemaker.instance_count 1 \
    --sagemaker.instance_type p4de \
    --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
    --remote_sync s3://your-bucket/your-path/model_checkpoints/diffusion_policy
```

The script prints `Submitted <job_name>` once the job is accepted. Use the
SageMaker console or `aws sagemaker describe-training-job --training-job-name
<job_name>` to monitor progress.

## SageMaker Options

| Flag | Default | Description |
|---|---|---|
| `--sagemaker.user` | required | Used in the job name and as a tag. |
| `--sagemaker.name_prefix` | `None` | Optional prefix prepended to the job name. |
| `--sagemaker.local` | `false` | Run with `LocalSession` and a local-GPU container instead of submitting to AWS. Useful for debugging the entry point. |
| `--sagemaker.region` | `us-west-2` | AWS region. |
| `--sagemaker.profile` | `default` | AWS profile used for ECR login and STS. |
| `--sagemaker.arn` | `$SAGEMAKER_ARN` | Execution role ARN used by the training job. |
| `--sagemaker.instance_count` | `1` | Number of training instances. |
| `--sagemaker.instance_type` | `p4de` | One of `p4de`, `p5`, `p5en`, `p6` — see `INSTANCE_MAPPER` in `launch_training.py` for the full type strings. |
| `--sagemaker.max_run` | `10` | Max wall-clock duration in days. |
| `--sagemaker.volume_size` | `30` | EBS volume size in GB. |
| `--sagemaker.check_config` | `false` | Parse and validate the config, dump the hyperparameters yaml, then exit before docker build. Useful as a fast smoke test when you can't submit jobs directly. |
| `--sagemaker.build_only` | `false` | Build and push the training image, then exit without submitting to SageMaker. Verifies the Dockerfile and dependency resolution without running a job. |

## Outputs

- **Checkpoints.** Written to the S3 path you pass via `--remote_sync`.
- **W&B logs.** Forwarded under the `vla_foundry` project (controlled by the
  `WANDB_PROJECT` env var).
- **Git provenance.** The launcher captures the current commit hash, branch,
  and any uncommitted diff into `sagemaker/git_diffs/git_diff_<uuid>.txt` and
  bakes that file into the training image so the job's environment records
  exactly which code ran.

## Local Mode

Set `--sagemaker.local true` to run the same training entry point inside a
local Docker container with `instance_type=local_gpu`. This is the fastest
way to validate that the image builds correctly and that your
`hyperparameters_<uuid>.yaml` is well-formed before paying for a remote
instance.
