"""
Batch handlers for different model types.

This module contains batch preparation and loss computation logic for each model type,
eliminating the need for if/elif statements in the training loop.

Batch handlers are registered using decorators from the registry module.
"""

from abc import ABC, abstractmethod

import torch

from vla_foundry.data.sampler import sample_chunk
from vla_foundry.models.registry import register_batch_handler


class BatchHandler(ABC):
    """Abstract base class for model-specific batch handlers."""

    def _move_to_device(self, batch, device):
        """Move all tensor values in batch to device in-place."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
        return batch

    @abstractmethod
    def prepare_inputs(self, batch, device, cfg):
        """
        Prepare model inputs from batch data.

        Args:
            batch: Raw batch dictionary from dataloader
            device: Target device for tensors
            cfg: Training configuration

        Returns:
            Dictionary of inputs ready for model(**inputs)
        """
        pass

    @abstractmethod
    def prepare_inputs_and_targets(self, batch, device, cfg):
        """
        Prepare model inputs and targets from batch data, including chunking if needed.

        Args:
            batch: Raw batch dictionary from dataloader
            device: Target device for tensors
            cfg: Training configuration

        Returns:
            Tuple of (model_inputs_dict, targets_tensor, mask_tensor)

        Note:
            The returned mask and model_inputs["future_mask"] are mutually exclusive:
            - LLM/VLM handlers return a mask (for padding/image tokens) and no future_mask
            - Diffusion policy handlers return mask=None and put future_mask in model_inputs
            The training loop validates this invariant.
        """
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        """
        Compute loss from model outputs and targets.

        Args:
            outputs: Model outputs
            targets: Target tensor (if needed)
            loss_fn: Loss function
            cfg: Training configuration
            mask: Mask of valid actions (should be broadcastable to the shape of outputs)

        Returns:
            Loss tensor
        """
        pass

    def slice_inputs_for_accumulation(self, model_inputs, start_idx, end_idx):
        """Slice model inputs for gradient accumulation microbatches."""
        if "image_grid_thw" in model_inputs:
            return self._slice_inputs_qwen(model_inputs, start_idx, end_idx)

        batch_size = model_inputs["input_ids"].shape[0]
        sliced_inputs = {}
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                if key == "pixel_values" and value.ndim == 4 and value.shape[0] != batch_size:
                    # CLIP/PaliGemma processors return pixel_values as [B*N, C, H, W].
                    # Scale slice indices to match the B*N first dimension.
                    scale = value.shape[0] // batch_size
                    sliced_inputs[key] = value[start_idx * scale : end_idx * scale]
                else:
                    sliced_inputs[key] = value[start_idx:end_idx]
            else:
                sliced_inputs[key] = value
        return sliced_inputs

    def _slice_inputs_qwen(self, model_inputs, start_idx, end_idx):
        """Slice inputs for Qwen-style models with flat pixel_values.

        Qwen processors return pixel_values as a flat (total_patches, patch_dim)
        tensor instead of (B, ...), with a companion image_grid_thw tensor
        describing per-image patch grid sizes. Both must be sliced together
        using the grid metadata.

        Assumes all samples in the batch have the same number of images. Missing
        images must be padded (set data_params.pad_missing_images=True) to satisfy
        this invariant.
        """
        batch_size = model_inputs["input_ids"].shape[0]
        grid = model_inputs["image_grid_thw"]

        assert grid.shape[0] % batch_size == 0, (
            f"image_grid_thw.shape[0] ({grid.shape[0]}) must be divisible by batch_size ({batch_size}). "
            f"Ensure all samples have the same number of images (set pad_missing_images=True)."
        )

        images_per_sample = grid.shape[0] // batch_size
        img_start = start_idx * images_per_sample
        img_end = end_idx * images_per_sample

        sliced_inputs = {}
        for key, value in model_inputs.items():
            if key in ("pixel_values", "image_grid_thw"):
                continue
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
                sliced_inputs[key] = value[start_idx:end_idx]
            else:
                sliced_inputs[key] = value

        sliced_inputs["image_grid_thw"] = grid[img_start:img_end]

        # Each row in image_grid_thw is (t, h, w). The number of patches for
        # that image is t * h * w. Compute cumulative offsets to slice pixel_values.
        patches_per_image = grid[:, 0] * grid[:, 1] * grid[:, 2]
        patch_start = patches_per_image[:img_start].sum().item()
        patch_end = patch_start + patches_per_image[img_start:img_end].sum().item()
        sliced_inputs["pixel_values"] = model_inputs["pixel_values"][patch_start:patch_end]

        return sliced_inputs

    def slice_targets_for_accumulation(self, targets, start_idx, end_idx, sliced_inputs=None):
        """Slice targets for gradient accumulation microbatches.

        Args:
            targets: Full targets tensor.
            start_idx: Start index for slicing.
            end_idx: End index for slicing.
            sliced_inputs: The already-sliced model inputs (from slice_inputs_for_accumulation).
                Subclasses may use this to recompute targets when slicing changes inputs
                (e.g., fresh noise generation with num_action_head_repeats).
        """
        return targets[start_idx:end_idx]


@register_batch_handler("transformer")
@register_batch_handler("transformer_hf")
class TransformerBatchHandler(BatchHandler):
    """Handles batch preparation for transformer and transformer_hf models."""

    def prepare_inputs(self, batch, device, cfg):
        self._move_to_device(batch, device)
        batch["output_hidden_states"] = False
        return batch

    def prepare_inputs_and_targets(self, batch, device, cfg):
        model_inputs = self.prepare_inputs(batch, device, cfg)

        # Sample a contiguous chunk to the configured sequence length
        input_ids, attention_mask, targets = sample_chunk(
            model_inputs["input_ids"], model_inputs.get("attention_mask"), cfg.data.seq_len
        )
        model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # Mask out pad tokens from loss computation
        if hasattr(cfg.data, "pad_token_id") and cfg.data.pad_token_id is not None:
            mask = targets == cfg.data.pad_token_id
        else:
            mask = None

        return model_inputs, targets, mask

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        return loss_fn(outputs.logits, targets, mask=mask)


@register_batch_handler("vlm")
@register_batch_handler("vlm_hf")
class VLMBatchHandler(BatchHandler):
    """Handles batch preparation for vlm and vlm_hf models."""

    def prepare_inputs(self, batch, device, cfg):
        self._move_to_device(batch, device)
        batch["output_hidden_states"] = False
        return batch

    def prepare_inputs_and_targets(self, batch, device, cfg):
        model_inputs = self.prepare_inputs(batch, device, cfg)

        # Sample a contiguous chunk to the configured sequence length
        input_ids, attention_mask, targets = sample_chunk(
            model_inputs["input_ids"], model_inputs.get("attention_mask"), cfg.data.seq_len
        )
        model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        mask = (targets == cfg.data.pad_token_id) | (targets == cfg.data.image_token_id)
        return model_inputs, targets, mask

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        return loss_fn(outputs.logits, targets, mask=mask)


@register_batch_handler("stable_diffusion")
class StableDiffusionBatchHandler(BatchHandler):
    """Handles batch preparation for stable_diffusion models."""

    # Keys the StableDiffusion.forward() signature accepts.
    _MODEL_KEYS = frozenset({"input_ids", "image", "attention_mask", "noise"})

    def prepare_inputs(self, batch, device, cfg):
        self._move_to_device(batch, device)
        # Model expects "image" not "pixel_values"
        batch["image"] = batch.pop("pixel_values")
        batch["noise"] = torch.randn_like(batch["image"])
        # Only keep keys accepted by StableDiffusion.forward() to avoid
        # passing unexpected WebDataset metadata (e.g. __key__).
        return {k: v for k, v in batch.items() if k in self._MODEL_KEYS}

    def prepare_inputs_and_targets(self, batch, device, cfg):
        model_inputs = self.prepare_inputs(batch, device, cfg)

        # For diffusion, targets are the noise (or noise direction for flow matching)
        targets = model_inputs["noise"]
        if cfg.model.use_flow_matching_scheduler:
            # In flow-matching variant: target is (noise - image) direction
            targets = model_inputs["noise"] - model_inputs["image"]

        return model_inputs, targets, None

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        predicted_direction = outputs
        return loss_fn(predicted_direction, targets, mask=mask)


@register_batch_handler("diffusion_policy")
class DiffusionPolicyBatchHandler(BatchHandler):
    """Handles batch preparation for diffusion policy models."""

    def prepare_inputs(self, batch, device, cfg):
        self._move_to_device(batch, device)
        batch["noise"] = torch.randn_like(batch["actions"])
        self._num_action_head_repeats = getattr(cfg.model, "num_action_head_repeats", None)
        return batch

    def prepare_inputs_and_targets(self, batch, device, cfg):
        inputs = self.prepare_inputs(batch, device, cfg)
        targets = inputs["noise"] - inputs["actions"]
        return inputs, targets, None

    # Keys whose batch dimension corresponds to the action head (tiled to [B*N]).
    _ACTION_SIDE_KEYS = frozenset({"actions", "noise", "past_mask", "future_mask", "proprioception"})

    def slice_inputs_for_accumulation(self, model_inputs, start_idx, end_idx):
        """Slice inputs for gradient accumulation, then apply num_repeats tiling.

        All tensors in model_inputs are at uniform batch size [B_full].
        After slicing the microbatch [start_idx:end_idx], action-side tensors
        are repeat_interleaved to [micro_batch * N] and N distinct noises are
        generated, while VLM-side tensors stay at [micro_batch].
        """
        sliced = super().slice_inputs_for_accumulation(model_inputs, start_idx, end_idx)

        num_repeats = getattr(self, "_num_action_head_repeats", None)
        if num_repeats is not None and num_repeats > 1:
            for key in self._ACTION_SIDE_KEYS:
                if key in sliced and isinstance(sliced[key], torch.Tensor):
                    sliced[key] = sliced[key].repeat_interleave(num_repeats, dim=0)
            # Generate N distinct noise samples per microbatch element.
            actions = sliced["actions"]
            sliced["noise"] = torch.randn(
                actions.shape,
                device=actions.device,
                dtype=actions.dtype,
            )

        return sliced

    def slice_targets_for_accumulation(self, targets, start_idx, end_idx, sliced_inputs=None):
        """Recompute targets from sliced inputs when num_repeats > 1.

        Fresh noise is generated in slice_inputs_for_accumulation, so the
        pre-computed targets (from the original noise) are stale.  Recompute
        as ``noise - actions`` from the already-sliced (and possibly repeated)
        model inputs.
        """
        num_repeats = getattr(self, "_num_action_head_repeats", None)
        if num_repeats is not None and num_repeats > 1:
            assert sliced_inputs is not None, (
                "sliced_inputs is required to recompute targets with num_action_head_repeats"
            )
            return sliced_inputs["noise"] - sliced_inputs["actions"]
        return targets[start_idx:end_idx]

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        # Reshape inputs and masks to match shapes
        predicted_direction = outputs
        target_direction = targets

        # Depending on the input strategy (past given in the same sequence or separate),
        # the mask may be shorter or longer than the loss
        if mask is not None:
            seq_len = min(mask.shape[1], predicted_direction.shape[1])
            predicted_direction = predicted_direction[:, -seq_len:]
            target_direction = target_direction[:, -seq_len:]
            mask = mask[:, -seq_len:]

        return loss_fn(input=predicted_direction, target=target_direction, mask=mask)
