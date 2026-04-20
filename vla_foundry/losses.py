import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CrossEntropyLossWithZLoss(CrossEntropyLoss):
    def __init__(
        self,
        eps: float = 1e-4,
        weight: Tensor = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, mask=None) -> Tensor:
        if mask is not None:
            while target.ndim > mask.ndim:
                mask = mask.unsqueeze(-1)
            target = target.masked_fill(mask.bool(), self.ignore_index)
        input = input.reshape(-1, input.shape[-1])
        target = target.reshape(-1)
        if self.eps != 0.0:
            return super().forward(input, target) + self.eps * torch.square(torch.logsumexp(input, dim=-1)).mean()
        else:
            return super().forward(input, target)


def masked_mse_loss(input: Tensor, target: Tensor, mask=None) -> Tensor:
    """
    Compute the MSE loss with masking. If no mask is provided, all values are considered valid.

    Args:
        input: Predicted values
        target: Ground truth values
        mask: Mask of valid actions. Either same shape as input, or same shape as input without last dimension.

    Returns:
        Masked MSE loss between inputs and targets
    """
    if mask is None:
        mask = torch.ones_like(input)
    elif mask.shape == input.shape:
        pass
    elif mask.shape == input.shape[:-1]:
        mask = mask.unsqueeze(-1).expand_as(input)
    else:
        raise ValueError(f"Mask shape {mask.shape} is not compatible with input shape {input.shape}")

    if mask.sum() == 0:
        return torch.tensor(0.0)
    else:
        return torch.nn.functional.mse_loss(input, target, weight=mask)


def _ignore_mask(loss_fn):
    def wrapper(*args, mask=None, **kwargs):
        return loss_fn(*args, **kwargs)

    return wrapper


def get_loss_function(loss_function_type, hparams):
    """
    Get the appropriate loss function for the given type.
    The output loss function should take inputs and targets as arguments and also a mask
    but it doesn't need to use the mask. In that case, the mask can be ignored with the wrapper _ignore_mask.
    """
    if loss_function_type == "cross_entropy":
        loss = CrossEntropyLossWithZLoss(hparams.z_loss_coefficient)
    elif loss_function_type == "mse":
        loss = masked_mse_loss
    else:
        raise ValueError(f"Loss function {loss_function_type} not supported.")

    return loss
