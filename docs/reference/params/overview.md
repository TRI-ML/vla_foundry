# Parameter System Overview

VLA Foundry uses a hierarchical, immutable dataclass system built on [draccus](https://github.com/dlwh/draccus) to configure every aspect of a training experiment. All configuration flows through a single top-level object --- `TrainExperimentParams` --- which composes several specialized parameter groups.

## Hierarchy

```
TrainExperimentParams
|-- model: ModelParams          # Architecture definition
|-- data: DataParams            # Dataset and preprocessing
|-- hparams: HyperParams        # Optimization and training
|-- distributed: DistributedParams  # Multi-GPU / multi-node
|-- ema: EMAParams              # Exponential moving average
```

Each nested parameter class is an immutable frozen dataclass. Fields are populated from YAML config files, command-line arguments, or a combination of both, with command-line arguments always taking precedence.

## Design Principles

### Separation of Concerns

Each parameter group owns a distinct slice of the configuration:

| Group | Responsibility |
|-------|---------------|
| **ModelParams** | What to train --- architecture type, layer counts, pretrained weights |
| **DataParams** | What to train on --- dataset paths, modality, field definitions, normalization |
| **HyperParams** | How to train --- learning rate, batch size, optimizer, precision |
| **DistributedParams** | Where to train --- FSDP, world size, backend |
| **EMAParams** | Whether and how to maintain an EMA copy of the model |

This separation means you can swap a model preset without touching data configuration, or change the optimizer without modifying the architecture.

### Immutability

All parameter dataclasses are `frozen=True`. Once constructed, fields cannot be reassigned. This guarantees that the configuration logged to Weights & Biases or saved alongside checkpoints is exactly what was used during training.

Derived fields (like `accum_freq` on `HyperParams`) are computed as `@property` accessors rather than mutable attributes.

### Shared Attributes

Some values need to flow between parameter groups. For example, `DiffusionPolicyParams` needs the `action_dim` that is computed inside `DataParams`. This is handled by `init_shared_attributes(cfg)`, which is called during `TrainExperimentParams.__post_init__` and passes the full config to each sub-parameter so it can read cross-cutting values.

```python
# Inside DiffusionPolicyParams
def init_shared_attributes(self, cfg):
    super().init_shared_attributes(cfg)
    object.__setattr__(self, "action_dim", cfg.data.action_dim)
    object.__setattr__(self, "proprioception_dim", cfg.data.proprioception_dim)
```

### Registry-Based Polymorphism

Both `ModelParams` and `DataParams` use draccus `ChoiceRegistry` for subclass selection at runtime. You specify the concrete type via a `type` field:

```yaml
model:
  type: diffusion_policy   # selects DiffusionPolicyParams
data:
  type: robotics            # selects RoboticsDataParams
```

New types are registered with decorators:

```python
@register_model_params("my_new_model")
@dataclass(frozen=True)
class MyNewModelParams(ModelParams):
    ...
```

## How Configs Are Loaded

1. **YAML files** are the primary source. Use `!include` directives to compose from presets.
2. **Command-line arguments** override any YAML value using dot-notation (`--hparams.lr 3e-4`).
3. **`__post_init__`** runs validation and derives computed fields.
4. **`init_shared_attributes`** propagates cross-cutting values between sub-params.
5. **`check_asserts`** validates consistency (batch sizes, dataset manifest lengths, etc.).

```python
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml

params = load_experiment_params_from_yaml("path/to/config.yaml")
```

!!! tip "Resolving Configs"
    Set `--resolve_configs True` to print the fully merged configuration and exit without training. Optionally set `--resolve_configs_path ./` to also save it to `resolved_config.yaml`.

## Source Locations

| File | Contents |
|------|----------|
| `vla_foundry/params/train_experiment_params.py` | `TrainExperimentParams`, YAML loading helpers |
| `vla_foundry/params/model_params.py` | `ModelParams` and all model-specific subclasses |
| `vla_foundry/params/base_data_params.py` | `DataParams` base class |
| `vla_foundry/params/data_params.py` | `TextDataParams`, `ImageCaptionDataParams`, `RoboticsDataParams` |
| `vla_foundry/params/hyper_params.py` | `HyperParams` |
| `vla_foundry/params/distributed_params.py` | `DistributedParams` |
| `vla_foundry/params/ema_params.py` | `EMAParams` |

## Next Steps

- [TrainExperimentParams](train-experiment.md) --- top-level fields
- [ModelParams](model-params.md) --- architecture configuration
- [DataParams](data-params.md) --- dataset configuration
- [HyperParams](hyper-params.md) --- optimization configuration
