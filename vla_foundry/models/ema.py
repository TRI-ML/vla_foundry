"""Exponential Moving Average (EMA) for model parameters."""

import copy

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def _to_tensor(param):
    """Convert DTensor to regular tensor if needed (for FSDP2 compatibility)."""
    # Check if this is a DTensor (FSDP2 uses DTensor for sharded parameters)
    if hasattr(param, "full_tensor"):
        # DTensor - convert to full tensor
        return param.full_tensor()
    # Regular tensor - return as-is
    return param


class EMABaseModel(torch.nn.Module):
    """Base class for EMA models."""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def step(self, new_model):
        """Update EMA weights with new model weights."""
        raise NotImplementedError()

    @property
    def model(self):
        """Return the averaged model."""
        raise NotImplementedError()


class VanillaEMAModel(EMABaseModel):
    """Simple EMA with fixed decay rate."""

    def __init__(
        self,
        model,
        alpha,
        use_buffers=True,
    ):
        """
        Args:
           model: The model to take an average of.
           alpha: The factor used to weight the averaged model in each update.
               So alpha = 0.99 would mean 0.99 weight on the averaged model
               and (1 - 0.99) weight on the updated model.
           use_buffers: Whether to also average pytorch buffers in the model
               in addition to parameters. This is set to True by default in
               order to properly handle batch norm global statistics.
        """
        super().__init__()
        self._alpha = alpha
        assert alpha > 0.0 and alpha < 1, f"Invalid EMA alpha: {alpha}"

        def ema_func(ema_param, model_param, num_averaged):
            return alpha * ema_param + (1 - alpha) * model_param

        # AveragedModel does a deepcopy of model.
        self.averaged_model = torch.optim.swa_utils.AveragedModel(
            model,
            # this should be ema = alpha * ema + (1 - alpha) * model
            avg_fn=ema_func,
            use_buffers=use_buffers,
        )
        self.averaged_model.module.eval()
        self.averaged_model.module.requires_grad_(False)

    @torch.no_grad()
    def step(self, new_model):
        """Update EMA weights."""
        self.averaged_model.update_parameters(new_model)

    @property
    def model(self):
        """Return the averaged model."""
        return self.averaged_model.module


class EMAModel(EMABaseModel):
    """Exponential Moving Average with adaptive warmup schedule."""

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999,
    ):
        """
        Args:
            model: The model to take an average of.
            update_after_step: Only start updating EMA after this many steps.
            inv_gamma: Inverse multiplicative factor of EMA warmup. Default: 1.
            power: Exponential factor of EMA warmup. Default: 0.75.
            min_value: The minimum EMA decay rate. Default: 0.
            max_value: The maximum EMA decay rate. Default: 0.9999.
        """
        super().__init__()
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.register_buffer("optimization_step", torch.zeros(1, dtype=torch.long))

    @property
    def model(self):
        """Return the averaged model."""
        return self.averaged_model

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        """Update EMA weights with adaptive decay rate."""
        decay = self.get_decay(self.optimization_step.item())

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules(), strict=False):
            # Handle BatchNorm statistics separately
            if isinstance(module, _BatchNorm):
                assert isinstance(ema_module, _BatchNorm)
                # Convert DTensor to regular tensor if needed (FSDP2 compatibility)
                running_mean = _to_tensor(module.running_mean)
                running_var = _to_tensor(module.running_var)

                ema_module.running_mean.mul_(decay)
                ema_module.running_mean.add_(running_mean, alpha=1 - decay)

                ema_module.running_var.mul_(decay)
                ema_module.running_var.add_(running_var, alpha=1 - decay)

            # Update parameters
            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
                strict=False,
            ):
                # Iterate over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")

                # Convert DTensor to regular tensor if needed (FSDP2 compatibility)
                param_data = _to_tensor(param.data)

                if not param.requires_grad:
                    # Copy non-trainable parameters directly
                    ema_param.copy_(param_data.to(dtype=ema_param.dtype))
                else:
                    # Apply EMA update
                    ema_param.mul_(decay)
                    ema_param.add_(param_data.to(dtype=ema_param.dtype), alpha=1 - decay)

        self.optimization_step += 1


def create_ema_model(model, ema_type="ema", **kwargs):
    """
    Factory function to create EMA model.

    Args:
        model: The model to wrap with EMA.
        ema_type: Type of EMA to use ("vanilla" or "ema").
        **kwargs: Additional arguments for the EMA model.

    Returns:
        EMABaseModel instance.
    """
    if ema_type == "vanilla":
        return VanillaEMAModel(model, **kwargs)
    elif ema_type == "ema":
        return EMAModel(model, **kwargs)
    else:
        raise ValueError(f"Unknown EMA type: {ema_type}")
