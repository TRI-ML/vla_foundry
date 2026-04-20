"""
Main training entrypoint for the VLA Foundry training.

This module wires together configuration parsing, model/optimizer/scheduler
construction, dataset selection per checkpoint, and the core training loop.

Notes
- The training budget is specified in *samples*, not *steps*.
- This file aims to remain a *thin orchestrator*; most heavy lifting is
delegated to subpackages (data, models, opt, train, etc.).
"""

import json
import logging
import os
import uuid

import draccus
import torch
import yaml

from vla_foundry.data.dataloader import get_datastring_input, get_wds_dataloader
from vla_foundry.data.utils import load_data_chunks
from vla_foundry.distributed import get_model_precision, is_master, move_buffers_to_device, wrap_fsdp_ddp
from vla_foundry.file_utils import (
    collect_preprocessing_configs,
    collect_processing_metadata,
    file_exists,
    load_ema_checkpoint,
    load_model_checkpoint,
    remote_sync,
    save_checkpoint,
)
from vla_foundry.logger import setup_logging
from vla_foundry.losses import get_loss_function
from vla_foundry.models import create_model
from vla_foundry.models.ema import create_ema_model
from vla_foundry.optimizer import create_optimizer, load_optimizer
from vla_foundry.params.train_experiment_params import TrainExperimentParams
from vla_foundry.scheduler import create_scheduler
from vla_foundry.train import train_one_checkpoint
from vla_foundry.utils import get_experiment_name, set_random_seed, summarize_datastrings
from vla_foundry.validate import validate_one_checkpoint


def main():
    """
    Entry point for launching training.

    Arguments are parsed with draccus.parse to instantiate a TrainExperimentParams object.
    They are provided as a preset yaml file, or as command line arguments or both.
    When using both a preset yaml file and command line arguments, the command line arguments take precedence.
    The preset yaml file is loaded with draccus.load, which supports !include statements to link a sub-preset yaml file.
    Other sub-preset yaml files can be passed as command line arguments with '--arg.subarg "include <path>"'.

    This function orchestrates experiment directory setup, (distributed) model
    construction, optimizer/scheduler creation, dataset selection per
    checkpoint, the training loop, and periodic checkpointing + remote sync.

    See README.md for more details.
    """
    # Parse config.
    cfg = draccus.parse(config_class=TrainExperimentParams)
    if cfg.resolve_configs:
        # Resolve configs for debugging. Program stops here if the flag is received.
        if is_master(cfg):
            print("Resolved config: ", cfg)
            if cfg.resolve_configs_path is not None:
                with open(os.path.join(cfg.resolve_configs_path, "resolved_config.yaml"), "w") as f:
                    draccus.dump(cfg, f)
                print(f"The resolved config was saved to {cfg.resolve_configs_path.rstrip('/')}/resolved_config.yaml")
            print(
                "=" * 50 + "\nThe flag --resolve_configs was received, stopping here. "
                "If the configuration is the one you want to launch, re-run without that flag."
            )
        return

    device = cfg.distributed.device
    # Seed rank-0 before any object creation for reproducibility.
    set_random_seed(cfg.hparams.seed, 0)

    # Early validation: check checkpoint path exists if provided
    if cfg.model.resume_from_checkpoint is not None:
        if not file_exists(cfg.model.resume_from_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found at '{cfg.model.resume_from_checkpoint}'. "
                "Please verify the path is correct and accessible."
            )
        else:
            logging.info(
                f"Checkpoint validation passed. Will load weights from '{cfg.model.resume_from_checkpoint}' for "
                f"{'finetuning' if cfg.model.resume_weights_only else 'resuming training'}."
            )

    # Set path for experiment, log, checkpoints.
    experiment_name = get_experiment_name(cfg)
    experiment_uuid = str(uuid.uuid4())

    if cfg.save_path is None:
        experiment_path = os.path.join("experiments", experiment_name)
    else:
        experiment_path = os.path.join(cfg.save_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    log_path = os.path.join(experiment_path, "out.log")
    # Convert string log level to logging constant
    log_level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    setup_logging(log_path, log_level)
    checkpoint_path = os.path.join(experiment_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    if is_master(cfg):
        # Persist the resolved config.
        with open(os.path.join(experiment_path, "config.yaml"), "w") as f:
            draccus.dump(cfg, f)
        with open(os.path.join(experiment_path, "config_model.yaml"), "w") as f:
            draccus.dump(cfg.model, f)

        # Collect and save processing metadata and configs from all data sources
        processing_metadata = collect_processing_metadata(cfg.data.dataset_manifest, experiment_path)
        if processing_metadata:
            with open(os.path.join(experiment_path, "processing_metadata.json"), "w") as f:
                json.dump(processing_metadata, f, indent=2)
        preprocessing_configs = collect_preprocessing_configs(cfg.data.dataset_manifest)
        if preprocessing_configs:
            with open(os.path.join(experiment_path, "preprocessing_config.yaml"), "w") as f:
                yaml.dump(preprocessing_configs, f)

        # Initial sync to check that remote_sync works.
        if cfg.remote_sync:
            remote_sync(experiment_path, os.path.join(cfg.remote_sync, experiment_name))
            remote_sync(experiment_path, os.path.join(cfg.remote_sync_fixed_path, experiment_uuid))

    if cfg.distributed.use_distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {cfg.distributed.device}."
            f"Process (global: {cfg.distributed.rank}, local {cfg.distributed.local_rank}), "
            f"total {cfg.distributed.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {cfg.distributed.device}.")

    # Model construction
    model = create_model(cfg.model)
    # Re-seed with rank to randomize across workers.
    set_random_seed(cfg.hparams.seed, cfg.distributed.rank)
    if cfg.hparams.grad_checkpointing:
        model.set_grad_checkpointing()

    # Create EMA model if enabled (BEFORE FSDP wrapping to avoid deepcopy issues)
    ema_model = None
    if cfg.ema.enabled:
        # Prepare kwargs based on EMA type
        if cfg.ema.type == "vanilla":
            ema_kwargs = {"alpha": cfg.ema.alpha}
        else:  # "ema" (adaptive)
            ema_kwargs = {
                "update_after_step": cfg.ema.update_after_step,
                "inv_gamma": cfg.ema.inv_gamma,
                "power": cfg.ema.power,
                "min_value": cfg.ema.min_value,
                "max_value": cfg.ema.max_value,
            }

        ema_model = create_ema_model(model, ema_type=cfg.ema.type, **ema_kwargs)

        # Move EMA model to device in FP32 (BF16 causes precision issues with high decay values)
        # EMA needs FP32 because decay values like 0.9999 round to 1.0 in BF16, stopping all updates
        ema_model = ema_model.to(device, dtype=torch.float32)

        if cfg.ema.type == "vanilla":
            logging.info(f"Created Vanilla EMA model with alpha={cfg.ema.alpha}")
        else:
            logging.info(
                f"Created Adaptive EMA model with power={cfg.ema.power}, inv_gamma={cfg.ema.inv_gamma}, "
                f"max_value={cfg.ema.max_value}"
            )

    # Wrap for distributed or move to device with the configured precision.
    if cfg.distributed.use_distributed:
        model = wrap_fsdp_ddp(model, device, cfg)
    else:
        model = model.to(device, dtype=get_model_precision(cfg))

    # Optionally resume model from a checkpoint or finetune from pretrained weights.
    start_checkpoint_num, global_step = 0, 0
    total_steps = cfg.total_train_samples // cfg.hparams.global_batch_size
    shard_shuffle_seed_per_dataset = None
    if cfg.model.resume_from_checkpoint is not None:
        if cfg.model.resume_weights_only:
            logging.info(f"Finetuning: Loading pretrained weights from '{cfg.model.resume_from_checkpoint}'")
            load_model_checkpoint(model, cfg.model.resume_from_checkpoint)
        else:
            start_checkpoint_num, global_step, shard_shuffle_seed_per_dataset = load_model_checkpoint(
                model, cfg.model.resume_from_checkpoint
            )

    # Create optimizer before torchcompile
    optimizer = create_optimizer(cfg.hparams, model)

    if cfg.hparams.torchcompile:
        # Ensure all buffers are on the correct device before compiling.
        # Some HuggingFace models (like SigLIP) have buffers that stay on CPU
        # which causes device mismatch errors during compilation.
        move_buffers_to_device(model, device)
        logging.info("Compiling model with torch.compile()...")
        # Note: torch.compile compatibility can vary across VLMs; some configurations may not compile cleanly.
        model = torch.compile(model)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

    count_parameters(model)

    # Optionally resume optimizer state from a checkpoint.
    # This needs to be after torchcompile.
    if cfg.model.resume_from_checkpoint is not None and not cfg.model.resume_weights_only:
        load_optimizer(optimizer, checkpoint_path=cfg.model.resume_from_checkpoint, use_fsdp=cfg.distributed.fsdp)

        # Load EMA checkpoint if EMA is enabled and checkpoint exists
        if ema_model is not None:
            # Construct EMA checkpoint path from model checkpoint path
            checkpoint_dir = os.path.dirname(cfg.model.resume_from_checkpoint)
            checkpoint_file = os.path.basename(cfg.model.resume_from_checkpoint)
            # Replace "checkpoint_" with "ema_" to get EMA checkpoint path
            ema_checkpoint_file = checkpoint_file.replace("checkpoint_", "ema_")
            ema_checkpoint_path = os.path.join(checkpoint_dir, ema_checkpoint_file)
            load_ema_checkpoint(ema_model, ema_checkpoint_path)

    # Create LR scheduler, loss function.
    scheduler = create_scheduler(cfg.hparams, optimizer, cfg.total_train_samples)
    loss = get_loss_function(cfg.hparams.loss_function, cfg.hparams)

    # Logging.
    if cfg.wandb and is_master(cfg):
        import wandb  # imported lazily to avoid hard dependency when disabled

        logging.debug("Starting wandb.")
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project_name,
            name=experiment_name,
            tags=cfg.wandb_tags,
            resume=None,
            config=vars(cfg),
        )
        logging.debug("Wandb initialized.")

    done_training = global_step >= total_steps
    checkpoint_num = start_checkpoint_num
    # Per-dataset cursors and shuffle seeds allow resuming mixed datasets.
    curr_shard_idx_per_dataset = [0 for dataset in range(len(cfg.data.dataset_manifest))]
    if shard_shuffle_seed_per_dataset is None:
        shard_shuffle_seed_per_dataset = [cfg.hparams.seed for dataset in range(len(cfg.data.dataset_manifest))]

    samples_seen = 0
    if cfg.model.resume_from_checkpoint is not None and not cfg.model.resume_weights_only:
        # Also restore which shards were consumed and how many samples were seen.
        curr_shard_idx_per_dataset, samples_seen = load_data_chunks(cfg.model.resume_from_checkpoint)

    # Load validation dataloader once here to reuse across checkpoints.
    do_validation = cfg.total_val_samples is not None
    if do_validation:
        val_datastrings, val_num_samples_per_dataset, _, _ = get_datastring_input(
            num_samples=cfg.total_val_samples,
            curr_shard_idx_per_dataset=[0 for _ in cfg.data.val_dataset_manifest],
            shard_shuffle_seed_per_dataset=[cfg.hparams.seed for _ in cfg.data.val_dataset_manifest],
            manifest_paths=cfg.data.val_dataset_manifest,
            dataset_weighting=cfg.data.val_dataset_weighting,
            allow_multiple_epochs=True,
            num_workers_per_gpu=cfg.data.num_workers,
            world_size=cfg.distributed.world_size,
        )
        val_dataloader = get_wds_dataloader(val_datastrings, val_num_samples_per_dataset, 0, cfg)

    # Main training loop
    while not done_training:
        if is_master(cfg):
            logging.info(f"Start checkpoint {checkpoint_num}")

        # Partition the global sample budget into evenly-sized checkpoint chunks.
        samples_per_checkpoint = cfg.total_train_samples // cfg.num_checkpoints
        datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = (
            get_datastring_input(
                num_samples=samples_per_checkpoint,
                curr_shard_idx_per_dataset=curr_shard_idx_per_dataset,
                shard_shuffle_seed_per_dataset=shard_shuffle_seed_per_dataset,
                manifest_paths=cfg.data.dataset_manifest,
                dataset_weighting=cfg.data.dataset_weighting,
                allow_multiple_epochs=cfg.data.allow_multiple_epochs,
                num_workers_per_gpu=cfg.data.num_workers,
                world_size=cfg.distributed.world_size,
            )
        )

        if is_master(cfg):
            logging.info(f"Now training on: {summarize_datastrings(datastrings)}")
            logging.info(f"Samples: {samples_seen} / {cfg.total_train_samples}")
            logging.info(f"Samples in this checkpoint (per dataset): {num_samples_per_dataset}")

        # Safety check: ensure all ranks see the same data slice.
        if cfg.distributed.use_distributed:
            all_datastrings = ["" for _ in range(cfg.distributed.world_size)]
            torch.distributed.all_gather_object(all_datastrings, datastrings)
            assert all([x == datastrings for x in all_datastrings]), (
                "Dataset to train on is not the same across all nodes. This should not happen normally, "
                "unless there is an issue with shard shuffling during the dataset generation."
            )

        dataloader = get_wds_dataloader(datastrings, num_samples_per_dataset, checkpoint_num, cfg)
        if is_master(cfg):
            # Save any necessary dataloader/pipeline configs.
            dataloader.save_configs(experiment_path)
            if cfg.remote_sync:
                remote_sync(experiment_path, os.path.join(cfg.remote_sync, experiment_name))
                remote_sync(experiment_path, os.path.join(cfg.remote_sync_fixed_path, experiment_uuid))

        prev_step = global_step

        if cfg.distributed.use_distributed:
            torch.distributed.barrier()

        success, global_step = train_one_checkpoint(
            model,
            dataloader,
            loss,
            checkpoint_num,
            global_step,
            optimizer,
            scheduler,
            cfg,
            ema_model=ema_model,
        )
        if cfg.distributed.use_distributed:
            torch.distributed.barrier()

        # Translate newly completed steps into samples.
        samples_seen = samples_seen + (global_step - prev_step) * cfg.hparams.global_batch_size
        checkpoint_num += 1
        done_training = global_step >= total_steps

        # Persist training state (model/opt/scheduler + data cursors).
        save_checkpoint(
            cfg,
            checkpoint_num,
            checkpoint_path,
            cfg.max_checkpoint_limit,
            model,
            optimizer,
            datastrings,
            curr_shard_idx_per_dataset,
            samples_seen,
            global_step,
            shard_shuffle_seed_per_dataset,
            ema_model=ema_model,
        )

        # Validate checkpoint.
        if do_validation and checkpoint_num % cfg.val_every_n_checkpoints == 0:
            if cfg.distributed.use_distributed:
                torch.distributed.barrier()
            avg_val_loss = validate_one_checkpoint(
                model, val_dataloader, loss, 0, global_step, cfg
            )  # checkpoint_num=0 for val for consistent data and seeding
            if is_master(cfg):
                logging.info(f"Validation after checkpoint {checkpoint_num}: avg_loss={avg_val_loss:.6f}")
            if cfg.distributed.use_distributed:
                torch.distributed.barrier()

        # Optionally push artifacts to remote storage after each checkpoint.
        if is_master(cfg) and cfg.remote_sync:
            remote_sync(experiment_path, os.path.join(cfg.remote_sync, experiment_name))
            remote_sync(experiment_path, os.path.join(cfg.remote_sync_fixed_path, experiment_uuid))

        if cfg.distributed.use_distributed:
            torch.distributed.barrier()

        if done_training:
            if is_master(cfg):
                logging.info("Model has seen the desired number of samples. Ending training.")
            break

    if cfg.wandb and is_master(cfg):
        wandb.finish()

    if cfg.distributed.use_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
