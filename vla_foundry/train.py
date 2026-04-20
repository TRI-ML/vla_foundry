import itertools
import logging
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FSDPModule
from tqdm import tqdm

from vla_foundry.distributed import is_master
from vla_foundry.file_utils import get_unwrapped_model
from vla_foundry.meters import Metrics
from vla_foundry.models.registry import create_batch_handler
from vla_foundry.params.train_experiment_params import TrainExperimentParams
from vla_foundry.precision import get_autocast


def train_one_checkpoint(
    model: nn.Module,
    dataloader,
    loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
    checkpoint_num: int,
    step: int,
    optimizer: optim.Optimizer,
    scheduler: Callable[[int], None],
    cfg: TrainExperimentParams,
    ema_model: nn.Module | None = None,
) -> tuple[bool, int]:
    """
    Trains model for one checkpoint on the provided data.

    This function:
      - Drives LR scheduling.
      - Performs forward/backward/step with optional gradient accumulation.
      - Computes and (optionally) all-reduces loss across ranks for logging.
      - Tracks timing/throughput metrics and logs periodically.
      - Exits when either:
          * the global training budget in samples is exhausted, or
          * the dataloader is depleted on any rank.

    Args:
        model: torch.nn.Module or a distributed-wrapped module.
        dataloader: dataloader object.
        loss: Callable loss function mapping (logits, targets, mask) -> scalar loss.
        checkpoint_num: Index of the current checkpoint window (for logs).
        step: Current global training step **before** this window starts.
        optimizer: torch.optim.Optimizer instance.
        scheduler: Callable taking `step` and adjusting LR, etc.
        cfg: Training config.
        ema_model: Optional EMA model for maintaining exponential moving average of weights.

    Returns:
        success (bool): Whether training completed successfully
        step (int): Global step at the end of the checkpoint.
    """
    device = torch.device(cfg.distributed.device)
    autocast = get_autocast(cfg.hparams.precision)

    # Create batch handler for the model type
    batch_handler = create_batch_handler(cfg.model.type)

    model.train()

    # Set EMA model if provided (for models that support it)
    if ema_model is not None:
        unwrapped_model = get_unwrapped_model(model)
        unwrapped_model.set_ema_model(ema_model)

    # Let the dataloader know which window/checkpoint it's on.
    dataloader.set_checkpoint_num(checkpoint_num)
    num_batches_per_checkpoint = dataloader.dataloader.num_batches

    # Meters for logging.
    metrics = Metrics()

    end = time.time()
    data_iterator = iter(dataloader.dataloader)

    # Progress bar setup - show step progress with proper starting value
    total_steps = cfg.total_train_samples // cfg.hparams.global_batch_size
    progress_bar = tqdm(
        initial=step, total=total_steps, desc=f"Checkpoint {checkpoint_num}", disable=not is_master(cfg), unit="step"
    )

    # Open-ended loop; we break on budget or data exhaustion.
    for i in itertools.count():
        scheduler(step)

        # Hard-stop when we reach the sample budget translated into steps.
        if step >= total_steps:
            logging.warning(f"step: {step} has reached/exceeded total_steps: {total_steps}. ending training.")
            break

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

        metrics.stats["data_time"].update(time.time() - end)
        optimizer.zero_grad()

        # Prepare model inputs and targets (including chunking) using batch handler
        model_inputs, targets, mask = batch_handler.prepare_inputs_and_targets(batch, device, cfg)

        # Validate that mask and future_mask are mutually exclusive
        if mask is not None and "future_mask" in model_inputs:
            raise ValueError(
                "mask and future_mask should not both be present. "
                "Use mask for LLM/VLM or future_mask for diffusion policy, not both."
            )

        forward_total_time = 0
        backward_total_time = 0
        total_lm_loss = 0
        for ii in range(cfg.hparams.accum_freq):
            # Don't sync gradients until the final microbatch for FSDP.
            # Check if model is actually wrapped with FSDP (not just if FSDP is enabled in config)
            if isinstance(model, FSDPModule):
                is_final_accum = ii == cfg.hparams.accum_freq - 1
                model.set_requires_gradient_sync(is_final_accum)
                model.set_requires_all_reduce(is_final_accum)
                model.set_reshard_after_backward(is_final_accum)
                model.set_is_last_backward(is_final_accum)

            with autocast():
                forward_start = time.time()
                # Slice the microbatch for this accumulation step.
                start_idx = ii * cfg.hparams.per_gpu_batch_size
                end_idx = (ii + 1) * cfg.hparams.per_gpu_batch_size
                model_inputs_ii = batch_handler.slice_inputs_for_accumulation(model_inputs, start_idx, end_idx)

                if model_inputs_ii["input_ids"].shape[0] == 0:
                    break

                targets_ii = batch_handler.slice_targets_for_accumulation(
                    targets, start_idx, end_idx, sliced_inputs=model_inputs_ii
                )
                # Get mask for microbatch: use mask if present, otherwise use future_mask (diffusion policy)
                mask_ii = mask[start_idx:end_idx] if mask is not None else model_inputs_ii.get("future_mask", None)

                # Forward pass - same for all model types!
                outputs = model(**model_inputs_ii)
                forward_total_time += time.time() - forward_start

                local_loss = batch_handler.compute_loss(outputs, targets_ii, loss, cfg, mask=mask_ii)

                # Scale loss by microbatch size ratio
                local_loss = local_loss * (model_inputs_ii["input_ids"].shape[0] / model_inputs["input_ids"].shape[0])

            # Backward per microbatch.
            backward_start = time.time()
            local_loss.backward()
            backward_total_time += time.time() - backward_start
            total_lm_loss += local_loss

        metrics.stats["forward_time"].update(forward_total_time)
        metrics.stats["backward_time"].update(backward_total_time)
        total_loss = total_lm_loss

        # Optimizer step
        optim_step_start = time.time()
        # (Optional) grad clipping
        if cfg.hparams.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hparams.grad_clip_norm, norm_type=2.0)
        optimizer.step()
        metrics.stats["optim_step_time"].update(time.time() - optim_step_start)

        # Update EMA model after optimizer step
        if ema_model is not None:
            unwrapped_model = get_unwrapped_model(model)
            ema_model.step(unwrapped_model)

        # For logging: clone a tensor copy of the loss and average across ranks.
        global_loss_tensor = total_loss.detach().clone()

        sync_start = time.time()
        if cfg.distributed.world_size > 1:
            dist.all_reduce(global_loss_tensor, op=ReduceOp.AVG)
        metrics.stats["sync_time"].update(time.time() - sync_start)

        # Update timing meters for this iteration.
        metrics.stats["batch_time"].update(time.time() - end)
        end = time.time()

        batch_count = i + 1
        step += 1  # Advance the global step after completing this batch.

        # Master-only logging & W&B
        if is_master(cfg):
            batch_size = len(model_inputs["input_ids"])
            # update the loss meter with the global loss tensor every iteration,
            # so that the logging is of the avg of loss of the last cfg.log_every_n_steps iterations
            metrics.stats["loss"].update(global_loss_tensor.item(), batch_size)

            # Periodic log or end-of-window/end-of-training log.
            if (
                (i % cfg.log_every_n_steps == 0 and i > 0)
                or batch_count == num_batches_per_checkpoint
                or step == total_steps - 1
            ):
                metrics.update_and_log_state(
                    cfg=cfg,
                    batch_size=batch_size,
                    batch_num_tokens=model_inputs["input_ids"].numel(),
                    batch_count=batch_count,
                    num_batches_per_checkpoint=num_batches_per_checkpoint,
                    step=step,
                    dataloader=dataloader,
                    lr=optimizer.param_groups[0]["lr"],
                    checkpoint_num=checkpoint_num,
                )

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"loss": f"{global_loss_tensor.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"}
            )

    progress_bar.close()

    return True, step
