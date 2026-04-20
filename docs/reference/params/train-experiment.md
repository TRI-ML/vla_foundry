# TrainExperimentParams

`TrainExperimentParams` is the top-level, immutable configuration dataclass for a training experiment. It composes all other parameter groups and runs cross-field validation on construction.

**Source:** `vla_foundry/params/train_experiment_params.py`

## Fields

### Logging and Tracking

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `None` | Optional experiment name. If `None`, a name is generated at runtime by `vla_foundry.utils.get_experiment_name`. |
| `resolve_configs` | `bool` | `False` | If `True`, print the fully resolved config and exit without training. |
| `resolve_configs_path` | `Optional[str]` | `None` | When set alongside `resolve_configs`, dumps the resolved config to `{path}/resolved_config.yaml`. |
| `save_path` | `str` | `None` | Base directory where the experiment folder is created. Defaults to `experiments/` if `None`. |
| `wandb` | `bool` | `True` | Enable Weights & Biases logging. |
| `wandb_entity` | `str` | `$WANDB_ENTITY` | W&B entity (team or user). Defaults to the `WANDB_ENTITY` environment variable. |
| `wandb_project_name` | `str` | `"vla_foundry"` | W&B project name. |
| `wandb_tags` | `list[str]` | `[]` | Tags attached to the W&B run. |
| `log_every_n_steps` | `int` | `20` | Frequency (in training steps) of metric logging. |
| `log_level` | `str` | `"INFO"` | Python logging level. |
| `remote_sync` | `str` | `None` | S3 path to which the experiment directory is periodically synced. |
| `remote_sync_fixed_path` | `str` | `"s3://your-bucket/your-path/vla_foundry_models_fixed/"` | Sibling S3 path where each run is also synced under its UUID, so runs can be located by ID regardless of `remote_sync`. |

### Training Budget

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_train_samples` | `int` | `None` | Total number of samples to train on. **Mutually exclusive** with `num_epochs`. |
| `num_epochs` | `int` | `None` | Number of passes over the dataset. Converted to `total_train_samples` during init. **Mutually exclusive** with `total_train_samples`. |
| `num_checkpoints` | `int` | `5` | Number of evenly-spaced checkpoint windows across the training budget. |
| `max_checkpoint_limit` | `int` | `None` | Maximum number of checkpoint files to keep on disk. Older checkpoints are deleted when this limit is exceeded. |

### Validation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_val_samples` | `int` | `None` | Total number of samples for validation. If `None`, validation is skipped. |
| `val_every_n_checkpoints` | `int` | `1` | Run validation every N checkpoint windows. |

### Nested Parameter Groups

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | [`DataParams`](data-params.md) | `DataParams()` | Dataset and preprocessing configuration. |
| `distributed` | `DistributedParams` | `DistributedParams()` | Multi-GPU and multi-node settings. |
| `ema` | `EMAParams` | `EMAParams()` | Exponential moving average settings. |
| `hparams` | [`HyperParams`](hyper-params.md) | `HyperParams()` | Optimization hyperparameters. |
| `model` | [`ModelParams`](model-params.md) | `ModelParams()` | Model architecture configuration. |

## DistributedParams

Controls multi-GPU and multi-node training. Most fields are auto-populated by `init_distributed_device()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dist_url` | `str` | `"env://"` | URL for distributed initialization. |
| `dist_backend` | `str` | `"nccl"` | Communication backend. |
| `fsdp` | `bool` | `False` | Enable Fully Sharded Data Parallel (FSDP2). |
| `fsdp_cpu_offload` | `bool` | `False` | Offload FSDP parameters to CPU. |
| `fsdp_reshard_after_forward` | `bool` | `False` | Reshard parameters after forward pass. |
| `ddp_static_graph` | `bool` | `False` | Enable DDP static graph optimization. |
| `use_distributed` | `bool` | `False` | Auto-set. Whether distributed training is active. |
| `world_size` | `int` | `1` | Auto-set. Total number of processes. |
| `rank` | `int` | `0` | Auto-set. Global rank of this process. |
| `local_rank` | `int` | `0` | Auto-set. Local rank on this node. |
| `device` | `str` | `None` | Auto-set. Device string (e.g., `"cuda:0"`). |

## EMAParams

Controls exponential moving average of model weights.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Whether to use EMA. |
| `type` | `str` | `"ema"` | EMA variant: `"vanilla"` (fixed decay) or `"ema"` (adaptive). |
| `alpha` | `float` | `0.999` | Fixed decay rate for vanilla EMA. |
| `update_after_step` | `int` | `0` | Start EMA updates after N steps (adaptive mode). |
| `inv_gamma` | `float` | `1.0` | Inverse gamma warmup factor (adaptive mode). |
| `power` | `float` | `0.75` | Warmup power. Use `2/3` for long training, `3/4` for short. |
| `min_value` | `float` | `0.0` | Minimum decay rate (adaptive mode). |
| `max_value` | `float` | `0.9999` | Maximum decay rate (adaptive mode). |

## Validation Rules

`TrainExperimentParams` enforces the following invariants during construction:

- `total_train_samples` and `num_epochs` are **mutually exclusive** --- set one or the other, not both.
- `global_batch_size` must be evenly divisible by `world_size`.
- `accum_freq * world_size * per_gpu_batch_size == global_batch_size`.
- `dataset_manifest`, `dataset_modality`, and `dataset_weighting` must all have the same length.
- `total_train_samples` must be resolved (non-`None`) after init.
- If `num_epochs` is set, `data.allow_multiple_epochs` must be `True`.

## Loading From YAML

```python
from vla_foundry.params.train_experiment_params import (
    load_experiment_params_from_yaml,
    load_params_from_yaml,
)

# Load full experiment config
params = load_experiment_params_from_yaml("path/to/config.yaml")

# Load any params subclass from YAML
from vla_foundry.params.model_params import ModelParams
model_params = load_params_from_yaml(ModelParams, "path/to/model.yaml")
```

!!! warning "S3 Paths"
    When loading from S3, the file is copied to a temporary location. `!include` directives inside S3-hosted configs will not resolve correctly because they use relative paths.

## Example YAML

```yaml
name: my_experiment
save_path: /tmp/experiments
wandb: true
wandb_project_name: vla_foundry
wandb_tags: [diffusion_policy, bellpepper]
total_train_samples: 30_000_000
num_checkpoints: 10
log_every_n_steps: 20
remote_sync: s3://my-bucket/experiments

model:
  type: diffusion_policy
  # ...

data:
  type: robotics
  # ...

hparams:
  lr: 5e-4
  global_batch_size: 128
  per_gpu_batch_size: 16
  # ...

distributed:
  fsdp: true

ema:
  enabled: true
  alpha: 0.999
```
