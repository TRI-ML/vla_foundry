# Model Registry

VLA Foundry uses a decorator-based registry system that allows models and batch handlers to self-register. This means adding a new model does not require editing any central configuration file --- you just decorate your creation function and import the module.

**Source:** `vla_foundry/models/registry.py`, `vla_foundry/models/__init__.py`

## How It Works

### Registration

Models register themselves using the `@register_model` decorator. Each model type has a unique string key that matches the `type` field in `ModelParams`.

```python
from vla_foundry.models.registry import register_model

@register_model("my_model")
def create_my_model(model_params, load_pretrained=True):
    return MyModel(model_params)
```

Batch handlers register similarly with `@register_batch_handler`:

```python
from vla_foundry.models.registry import register_batch_handler

@register_batch_handler("my_model")
class MyModelBatchHandler(BatchHandler):
    def __call__(self, batch, model, loss_fn):
        ...
```

### Import-Time Registration

Registration happens at import time. The `vla_foundry/models/__init__.py` module imports all model submodules, which triggers their decorators:

```python
import vla_foundry.models.transformer       # registers "transformer"
import vla_foundry.models.transformer_hf    # registers "transformer_hf"
import vla_foundry.models.vlm              # registers "vlm"
import vla_foundry.models.vlm_hf           # registers "vlm_hf"
import vla_foundry.models.diffusion        # registers "stable_diffusion"
import vla_foundry.models.diffusion_policy # registers "diffusion_policy", "clip_hf", "clip_openclip"
```

### Factory Function

The `create_model` function looks up the registered creation function and calls it:

```python
from vla_foundry.models import create_model

model = create_model(model_params, load_pretrained=True)
```

Internally this resolves to:

```python
def create_model(model_params: ModelParams, load_pretrained: bool = True) -> nn.Module:
    model_type = model_params.type
    create_fn = _MODEL_REGISTRY[model_type]
    return create_fn(model_params, load_pretrained)
```

### Batch Handler Factory

Each model type also has a batch handler that knows how to unpack a data batch, run the forward pass, and compute the loss:

```python
from vla_foundry.models import create_batch_handler

handler = create_batch_handler("diffusion_policy")
loss = handler(batch, model, loss_fn)
```

## Registered Models

| Type Key | Params Class | Description | Source |
|----------|-------------|-------------|--------|
| `transformer` | `TransformerParams` | From-scratch causal transformer (GPT-style) | `models/transformer.py` |
| `transformer_hf` | `TransformerHFParams` | Hugging Face pretrained transformer | `models/transformer_hf.py` |
| `vlm` | `VLMParams` | Vision-Language Model (ViT + Transformer) | `models/vlm.py` |
| `vlm_hf` | `VLMHFParams` | Hugging Face pretrained VLM | `models/vlm_hf.py` |
| `stable_diffusion` | `StableDiffusionParams` | Stable Diffusion for image generation | `models/diffusion/` |
| `diffusion_policy` | `DiffusionPolicyParams` | Diffusion Policy for robotics actions | `models/diffusion_policy/` |

## Registered Batch Handlers

Batch handlers are registered alongside their models. They define how data flows through the model during training:

| Type Key | Description |
|----------|-------------|
| `transformer` | Autoregressive language model batch handling (token shift, cross-entropy loss) |
| `transformer_hf` | Same as `transformer`, using HF model interface |
| `vlm` | Image-caption batch handling (image tokens + text tokens) |
| `vlm_hf` | Same as `vlm`, using HF model interface |
| `stable_diffusion` | Diffusion training loop (noise, denoise, reconstruction loss) |
| `diffusion_policy` | Robotics diffusion policy batch handling (condition on obs, denoise actions) |

## Registry API

```python
from vla_foundry.models.registry import (
    create_model,              # Create a model from params
    create_batch_handler,      # Create a batch handler by type key
    register_model,            # Decorator to register a model
    register_batch_handler,    # Decorator to register a batch handler
    list_registered_models,    # List all registered model type keys
    list_registered_batch_handlers,  # List all registered handler type keys
    is_model_registered,       # Check if a model type is registered
)
```

!!! warning "Duplicate Registration"
    Registering the same type key twice raises a `ValueError`. Each type key must be unique across the entire codebase.

## Adding a New Model

To register a new model:

1. Create the model class in `vla_foundry/models/`.
2. Create a params subclass in `vla_foundry/params/model_params.py` with `@register_model_params("your_key")`.
3. Create a model factory function with `@register_model("your_key")`.
4. Create a batch handler class with `@register_batch_handler("your_key")`.
5. Add an import line to `vla_foundry/models/__init__.py`.

See the [Adding New Models](../../guides/adding-new-models.md) guide for a full walkthrough.
