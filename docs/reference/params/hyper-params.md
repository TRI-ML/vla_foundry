# HyperParams

`HyperParams` controls all optimization-related settings: learning rate, scheduler, optimizer, precision, batch sizing, and gradient handling.

**Source:** `vla_foundry/params/hyper_params.py`

## Fields

### Precision

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `precision` | `str` | `"amp_bfloat16"` | Training precision mode. See [Precision Modes](#precision-modes) below. |

### Batch Sizing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `global_batch_size` | `int` | `512` | Total batch size across all GPUs and accumulation steps. |
| `per_gpu_batch_size` | `int` | `8` | Microbatch size per GPU per forward pass. |

### Randomness

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed for reproducibility. Propagated to `DataParams` and other components. |

### Learning Rate

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lr` | `float` | `1e-4` | Peak learning rate. |
| `lr_scheduler` | `str` | `"cosine"` | Learning rate schedule. Options: `"cosine"`, `"const"`, `"linear"`. |
| `warmup` | `str` | `"1000"` | Warmup duration. Integer values are treated as step counts; decimal values (e.g., `"0.025"`) are treated as fractions of total training steps. |
| `decay` | `str` | `"0.3"` | Decay phase duration. Same format as `warmup`. |
| `lr_cooldown_end` | `float` | `0.0` | Learning rate at the end of the cooldown phase. Must be less than or equal to `lr`. |
| `force_min_lr` | `float` | `0.0` | Absolute minimum learning rate floor. |

### Optimizer

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimizer` | `str` | `"adamw"` | Optimizer type. |
| `wd` | `float` | `0.01` | Weight decay. |
| `beta1` | `float` | `0.9` | Adam beta1 (first moment decay). |
| `beta2` | `float` | `0.95` | Adam beta2 (second moment decay). |
| `eps` | `float` | `1e-8` | Adam epsilon for numerical stability. |

### Loss

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `loss_function` | `str` | `"cross_entropy"` | Loss function. Options include `"cross_entropy"`, `"mse"`. |
| `z_loss_coefficient` | `float` | `0.0` | Coefficient for the auxiliary z-loss (logit regularization). Set to `0.0` to disable. |

### Gradient Handling

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `grad_clip_norm` | `float` | `None` | Maximum gradient norm for clipping. `None` disables clipping. |
| `grad_checkpointing` | `bool` | `False` | Enable gradient checkpointing (activation recomputation) to reduce memory usage. |

### Compilation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `torchcompile` | `bool` | `False` | Enable `torch.compile` for the model. |

### Internal / Shared

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `world_size` | `int` | `1` | **Shared.** Auto-set from `DistributedParams.world_size`. |

## Computed Properties

### `accum_freq`

Gradient accumulation frequency, computed from batch sizing:

```python
@property
def accum_freq(self):
    return global_batch_size // (world_size * per_gpu_batch_size)
```

For example, with `global_batch_size=512`, `world_size=8`, and `per_gpu_batch_size=8`:

```
accum_freq = 512 // (8 * 8) = 8
```

This means 8 microbatch forward passes are accumulated before each optimizer step.

## Precision Modes

| Value | AMP | Pure BF16 | Description |
|-------|-----|-----------|-------------|
| `"amp_bfloat16"` / `"amp_bf16"` / `"amp"` | Yes | No | Automatic mixed precision with bfloat16. **Recommended.** |
| `"pure_bf16"` | No | Yes | All operations in bfloat16. Lower memory but may reduce stability. |
| `"fp32"` / `"float32"` | No | No | Full float32 precision. Highest memory usage. |

!!! tip "Choosing Precision"
    `"amp_bfloat16"` is the default and recommended setting. It provides a good balance of speed, memory, and numerical stability. Use `"pure_bf16"` only if you are memory-constrained and have validated training stability. Use `"fp32"` for debugging numerical issues.

## Validation

The following assertions are checked during construction:

- `lr >= lr_cooldown_end` --- the cooldown end rate cannot exceed the peak learning rate.
- `global_batch_size % (world_size * per_gpu_batch_size) == 0` --- the global batch must divide evenly into accumulation steps.

## Example Configurations

### LLM Training (Cosine Schedule)

```yaml
hparams:
  precision: "amp_bfloat16"
  lr: 1e-4
  lr_scheduler: "cosine"
  warmup: "1000"
  decay: "0.3"
  global_batch_size: 512
  per_gpu_batch_size: 8
  optimizer: "adamw"
  wd: 0.01
  loss_function: "cross_entropy"
```

### Diffusion Policy (Constant LR)

```yaml
hparams:
  precision: "amp_bf16"
  lr: 5e-4
  lr_scheduler: "cosine"
  lr_cooldown_end: 1e-5
  grad_clip_norm: 1.0
  loss_function: "mse"
  global_batch_size: 128
  per_gpu_batch_size: 16
```

