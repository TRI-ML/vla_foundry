# Adding New Models

This tutorial shows how to add a new model to VLA Foundry using the registry system.

## Overview

The models module uses a **registry pattern** where models self-register using decorators. This keeps models self-contained and avoids merge conflicts.

**Key concepts**:
- Models register with `@register_model("model_name")`
- Batch handlers register with `@register_batch_handler("model_name")`
- FSDP blocks inherit from `FSDPBlock` (or implement `get_fsdp_block_types()` for HuggingFace models)

## Quick Start

### 1. Define Model Params

Add a params class to `params/model_params.py`:

```python
from vla_foundry.params.model_params import ModelParams

@register_model_params("my_model")
@dataclass(frozen=True)
class MyModelParams(ModelParams):
    hidden_dim: int = 768
    n_layers: int = 12
    # ... other model-specific params
```

### 2. Implement Your Model

Create `models/my_model.py`:

```python
import torch.nn as nn
from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.fsdp_block import FSDPBlock
from vla_foundry.models.registry import register_model
from vla_foundry.params.model_params import MyModelParams


class MyModelBlock(FSDPBlock):  # ← Inherit from FSDPBlock for FSDP wrapping
    """A single block/layer of your model."""
    def __init__(self, layer_id: int, model_params: MyModelParams):
        super().__init__()
        self.linear = nn.Linear(model_params.hidden_dim, model_params.hidden_dim)

    def forward(self, x):
        return self.linear(x)


class MyModel(BaseModel):
    """Your model implementation."""
    def __init__(self, model_params: MyModelParams):
        super().__init__(model_params)
        self.layers = nn.ModuleList([
            MyModelBlock(i, model_params)
            for i in range(model_params.n_layers)
        ])

    def forward(self, input_ids, **kwargs):
        # Your forward implementation
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


# ===== Model Registration =====

@register_model("my_model")
def create_my_model(model_params: MyModelParams, load_pretrained: bool = True):
    return MyModel(model_params)
```

### 3. Add Import

Add one line to `models/__init__.py`:

```python
import vla_foundry.models.my_model  # registers "my_model"
```

### 4. Add Batch Handler

Add to `models/batch_handlers.py`:

```python
from vla_foundry.models.registry import register_batch_handler

@register_batch_handler("my_model")
class MyModelBatchHandler(BatchHandler):
    def prepare_inputs(self, batch, device, model_dtype, cfg):
        return {
            "input_ids": batch["input_ids"].to(device, non_blocking=True, dtype=torch.long),
        }

    def prepare_inputs_and_targets(self, batch, device, model_dtype, cfg):
        inputs = self.prepare_inputs(batch, device, model_dtype, cfg)
        targets = batch["labels"].to(device, non_blocking=True, dtype=torch.long)
        return inputs, targets, None

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        return loss_fn(outputs.logits, targets, mask=mask)
```

### 5. Use Your Model

```python
from vla_foundry.models import create_model

model = create_model(my_model_params, load_pretrained=True)
```

## FSDP Block Registration

### Option 1: Custom Blocks (Recommended)

For models you control, inherit from `FSDPBlock`:

```python
from vla_foundry.models.fsdp_block import FSDPBlock

class MyBlock(FSDPBlock):  # ✅ Automatically wrapped by FSDP!
    def __init__(self, ...):
        super().__init__()
```

**No registration needed** - FSDP automatically finds all `FSDPBlock` subclasses.

### Option 2: HuggingFace Models

For models using HF library blocks, implement `get_fsdp_block_types()`:

```python
from transformers import AutoModelForCausalLM
from torch import nn

class MyHFModel(BaseModel):
    def __init__(self, model_params):
        super().__init__(model_params)
        self.model = AutoModelForCausalLM.from_pretrained(...)

    def get_fsdp_block_types(self):
        """Return HF block types for FSDP."""
        for _name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return (type(module[0]),)
        raise ValueError("Could not find model block class.")
```

## Project Structure

### Option A: Single File (Simple Models)
```
models/
  my_model.py        # Model + registration
```

### Option B: Directory (Complex Models)
```
models/
  my_model/
    __init__.py      # Registration
    model.py         # Model classes
    components.py    # Sub-components
```

## Examples

### Example 1: Simple Custom Model

```python
# models/simple_mlp.py
from vla_foundry.models.fsdp_block import FSDPBlock
from vla_foundry.models.registry import register_model

class MLPBlock(FSDPBlock):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))

class SimpleMLP(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLPBlock(model_params.hidden_dim)
            for _ in range(model_params.n_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

@register_model("simple_mlp")
def create_simple_mlp(model_params, load_pretrained=True):
    return SimpleMLP(model_params)
```

### Example 2: Model with Sub-Models

```python
# models/multimodal/__init__.py
from vla_foundry.models.registry import register_model

@register_model("multimodal")
def create_multimodal(model_params, load_pretrained=True):
    from vla_foundry.models.registry import create_model

    vision_encoder = create_model(model_params.vision, load_pretrained)
    text_encoder = create_model(model_params.text, load_pretrained)

    return MultimodalModel(model_params, vision_encoder, text_encoder)
```

### Example 3: HuggingFace Wrapper

```python
# models/my_hf_model.py
from transformers import AutoModelForCausalLM
from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.registry import register_model

class MyHFModel(BaseModel):
    def __init__(self, model_params, load_pretrained=True):
        super().__init__(model_params)
        if load_pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(model_params.hf_pretrained)
        else:
            config = AutoConfig.from_pretrained(model_params.hf_pretrained)
            self.model = AutoModelForCausalLM.from_config(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_fsdp_block_types(self):
        for _name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                return (type(module[0]),)
        raise ValueError("Could not find model block class.")

@register_model("my_hf_model")
def create_my_hf_model(model_params, load_pretrained=True):
    return MyHFModel(model_params, load_pretrained)
```

## Best Practices

### Model Design

1. **Inherit from `BaseModel` or `TransformerBase`**
   - Provides consistent interfaces
   - Handles parameter freezing automatically

2. **Use type hints**
   ```python
   def create_my_model(model_params: MyModelParams, load_pretrained: bool = True) -> MyModel:
   ```

3. **Handle `load_pretrained` parameter**
   - `True`: Load pretrained weights
   - `False`: Initialize from scratch (useful for testing)

4. **For FSDP blocks**:
   - Custom code: Inherit from `FSDPBlock`
   - Library code: Implement `get_fsdp_block_types()`

### Registration

1. **Register at end of file** to avoid circular imports
   ```python
   # Model classes...

   # ===== Model Registration =====
   # (decorators at the very end)
   ```

2. **Use local imports** in registration functions
   ```python
   @register_model("my_model")
   def create_my_model(model_params, load_pretrained=True):
       from vla_foundry.models.sub_model import SubModel  # Local import
       return MyModel(model_params, SubModel())
   ```

3. **Use descriptive names**: `"my_model"`, not `"model1"`

### Batch Handlers

1. **Keep stateless** - No instance variables between calls

2. **Use `non_blocking=True`**
   ```python
   input_ids = batch["input_ids"].to(device, non_blocking=True, dtype=torch.long)
   ```

3. **Handle optional inputs**
   ```python
   if "attention_mask" in batch and batch["attention_mask"] is not None:
       inputs["attention_mask"] = batch["attention_mask"].to(device, ...)
   ```

## Testing Your Model

```python
# Test registration
from vla_foundry.models.registry import list_registered_models
assert "my_model" in list_registered_models()

# Test model creation
from vla_foundry.models import create_model
model = create_model(my_model_params)
assert model is not None

# Test batch handler
from vla_foundry.models import create_batch_handler
handler = create_batch_handler("my_model")
assert handler is not None
```

## Disabling a Model

Comment out the import in `models/__init__.py`:

```python
import vla_foundry.models.transformer
# import vla_foundry.models.experimental_model  # DISABLED
```

## Troubleshooting

### "Model type 'X' is not registered"
**Solution**: Add `import vla_foundry.models.X` to `models/__init__.py`

### "Batch handler for 'X' not registered"
**Solution**: Add `@register_batch_handler("X")` decorator in `batch_handlers.py`

### Circular import errors
**Solution**: Use local imports inside registration functions

### FSDP errors
**Solution**: Check that blocks either:
1. Inherit from `FSDPBlock`, OR
2. Model implements `get_fsdp_block_types()`

---

**See also**:
- `models/registry.py` - Registry implementation
- `models/batch_handlers.py` - Batch handler examples
- `models/fsdp_block.py` - FSDP marker class
- Existing model files for reference
