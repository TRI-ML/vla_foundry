import itertools
import logging
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.distributed_c10d import ReduceOp

from vla_foundry.distributed import is_master
from vla_foundry.models.registry import create_batch_handler
from vla_foundry.params.train_experiment_params import TrainExperimentParams
from vla_foundry.precision import get_autocast


def validate_one_checkpoint(
    model: nn.Module,
    val_dataloader,
    loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
    checkpoint_num: int,
    step: int,
    cfg: TrainExperimentParams,
) -> float:
    """
    Validates model on the provided validation data.

    Args:
        model: torch.nn.Module or a distributed-wrapped module.
        dataloader: dataloader object.
        loss: Callable loss function mapping (logits, targets, mask) -> scalar loss.
        checkpoint_num: Index of the current checkpoint window.
        step: Current global training step **before** this window starts.
        cfg: Training config.

    Returns:
        val_loss (float): Average validation loss over the dataset.
    """
    device = torch.device(cfg.distributed.device)
    autocast = get_autocast(cfg.hparams.precision)

    # Create batch handler for the model type
    batch_handler = create_batch_handler(cfg.model.type)

    # Check if model was previously training, then switch to eval model.
    model_was_training = model.training
    model.eval()

    total_val_loss = torch.zeros(1, device=device)
    total_samples = 0

    # Let the dataloader know which window/checkpoint it's on.
    val_dataloader.set_checkpoint_num(checkpoint_num)
    data_iterator = iter(val_dataloader.dataloader)

    for _i in itertools.count():
        # Try to fetch the next batch on this rank.
        try:
            batch = next(data_iterator)
            has_data = torch.tensor(1, dtype=torch.long, device=device)
        except StopIteration:
            has_data = torch.tensor(0, dtype=torch.long, device=device)

        # Ensure all ranks still have data; if any rank is out, break.
        if cfg.distributed.world_size > 1:
            dist.all_reduce(has_data, op=ReduceOp.SUM)
        if has_data < cfg.distributed.world_size:  # Not all gpus have data
            break

        with torch.no_grad():
            # Prepare model inputs and targets (including chunking) using batch handler.
            model_inputs, targets, mask = batch_handler.prepare_inputs_and_targets(batch, device, cfg)
            # Use returned mask if present, otherwise check for future_mask (diffusion policy)
            if mask is None:
                mask = model_inputs.get("future_mask", None)

            with autocast():
                outputs = model(**model_inputs)
                batch_loss = batch_handler.compute_loss(outputs, targets, loss, cfg, mask=mask)

        # Accumulate validation loss per rank.
        bs = len(model_inputs["input_ids"])
        total_val_loss += batch_loss * bs
        total_samples += bs

    # Aggregate across ranks.
    if cfg.distributed.world_size > 1:
        dist.all_reduce(total_val_loss, op=ReduceOp.SUM)
        total_samples_tensor = torch.tensor([total_samples], dtype=torch.long, device=device)
        dist.all_reduce(total_samples_tensor, op=ReduceOp.SUM)
        total_samples = int(total_samples_tensor.item())

    avg_loss = (total_val_loss / total_samples).item()

    # Master-only logging & W&B
    if is_master(cfg):
        logging.info(f"[VAL] ckpt={checkpoint_num} step={step} avg_loss={avg_loss:.6f} samples={total_samples}")
        if cfg.wandb:
            import wandb

            wandb.log(
                {
                    "val/loss": avg_loss,
                    "val/samples": total_samples,
                },
                step=step,
            )

    if model_was_training:
        model.train()

    return avg_loss
