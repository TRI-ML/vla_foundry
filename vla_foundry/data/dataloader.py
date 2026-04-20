import copy
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vla_foundry.data.pipelines import create_wds_pipeline
from vla_foundry.data.utils import SharedCheckpointCounter
from vla_foundry.file_utils import load_dataset_manifest


def seed_worker(worker_id: int) -> None:
    """
    Seed NumPy and Python RNGs inside a dataloader worker process.

    Args:
        worker_id: The worker id provided by PyTorch's DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class DataInfo:
    """
    This is a wrapper around WebDataset's DataLoader that allows us to store a few extra information.
    """

    dataloader: DataLoader
    dataset_pipelines: list[wds.DataPipeline] = None
    sampler: DistributedSampler = None
    shared_checkpoint_counter: SharedCheckpointCounter = None

    # Optional token IDs for padding and image tokenization.
    pad_token_id: int = None
    image_token_id: int = None

    def set_checkpoint_num(self, checkpoint_num: int) -> None:
        """
        Propagate the current checkpoint window number to helpers.
        """
        if self.shared_checkpoint_counter is not None:
            self.shared_checkpoint_counter.set_value(checkpoint_num)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_checkpoint_num(checkpoint_num)

    def save_configs(self, experiment_path: str):
        for dataset_pipeline in self.dataset_pipelines:
            dataset_pipeline.save_configs(experiment_path)


def get_wds_dataloader(
    datastrings: Sequence[str],
    num_samples_per_dataset: Sequence[int],
    checkpoint_num: int,
    cfg: object,
) -> DataInfo:
    """
    Build a mixed WebDataset dataloader for a single checkpoint window.
    Args:
        datastrings: Per-dataset WebDataset input strings.
        num_samples_per_dataset: The sample budget to draw from each dataset for this window.
            These are used as mixing probabilities in `wds.mix.RandomMix`.
        checkpoint_num: Current checkpoint window index.
        cfg: Training configuration object.

    Returns:
        DataInfo: A wrapper containing the `WebLoader` and helper objects.
    """
    shared_checkpoint_counter = SharedCheckpointCounter(checkpoint_num=checkpoint_num)

    # Per-rank batch size (global batch is split evenly across ranks).
    if cfg.hparams.global_batch_size // cfg.distributed.world_size == 0:
        logging.error(
            f"Global batch size {cfg.hparams.global_batch_size} is smaller than world size "
            f"{cfg.distributed.world_size}, setting it to world size"
        )
    batch_size = max(cfg.hparams.global_batch_size // cfg.distributed.world_size, 1)

    # Build one pipeline per dataset, then mix them by target sample counts.
    dataset_pipelines = []
    for datastring, modality in zip(datastrings, cfg.data.dataset_modality, strict=False):
        dataset_pipelines.append(create_wds_pipeline(datastring, modality, batch_size, checkpoint_num, cfg.data))
    dataset = wds.mix.RandomMix(dataset_pipelines, probs=num_samples_per_dataset, longest=True)

    # Start a generator to have control over reproducibility.
    if cfg.data.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(
            cfg.data.seed + shared_checkpoint_counter.get_value() * cfg.distributed.world_size + cfg.distributed.rank
        )
        worker_init_fn = seed_worker
    else:
        generator = None
        worker_init_fn = None

    prefetch_factor = cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,  # batching handled in the pipeline
        shuffle=False,  # mixing is handled by RandomMix
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        generator=generator,
        worker_init_fn=worker_init_fn,
        in_order=cfg.data.dataloader_in_order,
    )

    # Compute total batches/samples this loader will emit in this window.
    # We want each worker to process the same number of shard-groups.
    if cfg.data.num_workers == 0:
        logging.warning("num_workers is <= 0, setting to 1 per GPU")
    num_workers_per_gpu = max(1, cfg.data.num_workers)
    total_samples = sum(num_samples_per_dataset)
    denominator = cfg.hparams.global_batch_size * num_workers_per_gpu
    num_worker_batches = total_samples // denominator
    if num_worker_batches == 0:
        raise ValueError(
            f"Zero batches: total_samples ({total_samples}) < "
            f"global_batch_size ({cfg.hparams.global_batch_size}) × "
            f"num_workers ({num_workers_per_gpu}) = {denominator}. "
            f"Reduce global_batch_size or increase total_train_samples."
        )

    num_batches = num_worker_batches * num_workers_per_gpu
    num_samples = num_batches * cfg.hparams.global_batch_size

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    dataloader = DataInfo(
        dataloader=dataloader, dataset_pipelines=dataset_pipelines, shared_checkpoint_counter=shared_checkpoint_counter
    )
    return dataloader


def _shuffle_manifest_inplace(manifest, seed):
    """Shuffle manifest entries in-place with a deterministic seed, matching load_dataset_manifest."""
    if seed is not None:
        np.random.default_rng(seed).shuffle(manifest)
    return manifest


def get_datastring_input(
    num_samples: int,
    curr_shard_idx_per_dataset: int,
    shard_shuffle_seed_per_dataset: int,
    manifest_paths: str,
    dataset_weighting: str,
    allow_multiple_epochs: str,
    num_workers_per_gpu: int,
    world_size: int,
) -> tuple[list[str], list[int], list[int], list[int]]:
    """
    Select shards for the next checkpoint window and build datastrings.

    Given one or more dataset manifests, this function determines how many
    samples to draw from each dataset (according to `dataset_weighting`), then
    selects enough shards so that every worker across all ranks receives the
    same count of shards. It returns WebDataset datastrings suitable for
    `create_wds_pipeline`.

    Args:
        num_samples: Total number of samples to fetch across all
            datasets for this window.
        curr_shard_idx_per_dataset: Current shard cursor per dataset.
        shard_shuffle_seed_per_dataset: Current shuffle seed per dataset.
        manifest_paths: Paths/URIs to per-dataset manifest JSON files.
        dataset_weighting: Optional per-dataset weights; if `None`,
            uses uniform weighting.
        allow_multiple_epochs: Whether to reshuffle and wrap around when
            shards are exhausted.
        num_workers_per_gpu: Number of dataloader workers per rank.
        world_size: Total number of ranks in the distributed job.

    Returns:
        datastrings: Per-dataset WebDataset input strings (local or S3 pipe).
        num_samples_list_per_dataset: Per-dataset total samples scheduled
            for this window (after shard selection).
        next_shard_idx_per_dataset: Updated shard cursors after accounting
            for selected shards.
        next_shard_shuffle_seed_per_dataset: Updated shuffle seeds.
    """
    # Load raw manifests once (for potential in-memory re-shuffling in multi-epoch),
    # then apply the initial shuffle to get the working copies.
    raw_manifests = [load_dataset_manifest(path, shard_shuffle_seed=None) for path in manifest_paths]
    manifests = [
        _shuffle_manifest_inplace(list(raw), seed)
        for raw, seed in zip(raw_manifests, shard_shuffle_seed_per_dataset, strict=False)
    ]

    # Default to uniform weighting if not provided.
    if dataset_weighting is None:
        dataset_weighting = [1 for i in range(len(manifests))]

    if num_samples > 0:
        needed_samples_per_dataset = [
            int(np.ceil(dataset_weighting[i] * num_samples / sum(dataset_weighting))) for i in range(len(manifests))
        ]
    else:
        needed_samples_per_dataset = [-1 for i in range(len(manifests))]
        # Avoid infinite loop when num_samples is -1
        assert not allow_multiple_epochs, "allow_multiple_epochs must be False when num_samples is -1"

    next_shard_idx_per_dataset = copy.deepcopy(curr_shard_idx_per_dataset)
    next_shard_shuffle_seed_per_dataset = copy.deepcopy(shard_shuffle_seed_per_dataset)

    # Build lists of shard names and their sample counts selected for this window.
    shard_list_per_dataset = [[] for i in range(len(manifests))]
    num_samples_list_per_dataset = [[] for i in range(len(manifests))]
    total_num_workers = num_workers_per_gpu * world_size

    # Greedily add shards until we satisfy both:
    # (a) enough samples for the weighting target, and
    # (b) at least one shard per worker (to balance work).
    for i in range(len(manifests)):
        needed = needed_samples_per_dataset[i]
        manifest = manifests[i]
        accumulated_samples = 0

        if needed == -1:
            # Use all remaining shards in a single pass (no multi-epoch).
            start_idx = curr_shard_idx_per_dataset[i]
            for idx in range(start_idx, len(manifest)):
                shard_list_per_dataset[i].append(manifest[idx]["shard"])
                num_samples_list_per_dataset[i].append(manifest[idx]["num_sequences"])
            curr_shard_idx_per_dataset[i] = len(manifest)
            continue

        # Phase 1: Add remaining shards from current position in the current epoch.
        start_idx = curr_shard_idx_per_dataset[i]
        for idx in range(start_idx, len(manifest)):
            shard_list_per_dataset[i].append(manifest[idx]["shard"])
            num_samples_list_per_dataset[i].append(manifest[idx]["num_sequences"])
            accumulated_samples += manifest[idx]["num_sequences"]
            curr_shard_idx_per_dataset[i] = idx + 1
            if accumulated_samples >= needed and len(shard_list_per_dataset[i]) >= total_num_workers:
                break

        # Phase 2: If we still need more samples, handle multi-epoch wrapping in bulk.
        if accumulated_samples < needed or len(shard_list_per_dataset[i]) < total_num_workers:
            if not allow_multiple_epochs:
                logging.error(
                    "Number of shards requested for a single epoch is more than the number of shards available. "
                    "Consider using --allow-multiple-epochs."
                )
                raise IndexError(
                    f"Dataset {i}: needed {needed} samples but only {accumulated_samples} available "
                    f"in {len(manifest)} shards without multi-epoch."
                )

            # Pre-compute samples per full epoch to skip in bulk.
            samples_per_epoch = sum(entry["num_sequences"] for entry in manifest)
            shards_per_epoch = len(manifest)

            remaining_needed = needed - accumulated_samples
            remaining_shards = max(0, total_num_workers - len(shard_list_per_dataset[i]))

            # Calculate how many full epochs we can add in bulk.
            if samples_per_epoch > 0:
                full_epochs = remaining_needed // samples_per_epoch
                # Ensure we also satisfy the minimum-shards-per-worker constraint.
                if remaining_shards > full_epochs * shards_per_epoch:
                    full_epochs = max(full_epochs, (remaining_shards + shards_per_epoch - 1) // shards_per_epoch)
            else:
                full_epochs = 0

            if full_epochs > 0:
                logging.info(
                    f"Dataset {i}: adding {full_epochs} full epochs in bulk "
                    f"({full_epochs * shards_per_epoch} shards, {full_epochs * samples_per_epoch} samples)"
                )

            # Add full epochs using in-memory shuffling (no repeated S3 loads).
            raw = raw_manifests[i]
            for _ in range(full_epochs):
                if shard_shuffle_seed_per_dataset[i] is not None:
                    shard_shuffle_seed_per_dataset[i] += 1
                shuffled = _shuffle_manifest_inplace(list(raw), shard_shuffle_seed_per_dataset[i])
                for entry in shuffled:
                    shard_list_per_dataset[i].append(entry["shard"])
                    num_samples_list_per_dataset[i].append(entry["num_sequences"])
                accumulated_samples += samples_per_epoch

            # Phase 3: Add shards from one more partial epoch if still needed.
            if accumulated_samples < needed or len(shard_list_per_dataset[i]) < total_num_workers:
                if shard_shuffle_seed_per_dataset[i] is not None:
                    shard_shuffle_seed_per_dataset[i] += 1
                shuffled = _shuffle_manifest_inplace(list(raw), shard_shuffle_seed_per_dataset[i])
                for j, entry in enumerate(shuffled):
                    shard_list_per_dataset[i].append(entry["shard"])
                    num_samples_list_per_dataset[i].append(entry["num_sequences"])
                    accumulated_samples += entry["num_sequences"]
                    curr_shard_idx_per_dataset[i] = j + 1
                    if accumulated_samples >= needed and len(shard_list_per_dataset[i]) >= total_num_workers:
                        break

    # Normalize shard lists: ensure each dataset's shard count is divisible by total workers.
    for i in range(len(manifests)):
        # Ensure number of shards is a multiple of number of workers, so each worker has same number of shards.
        idx_div = (
            (len(shard_list_per_dataset[i]) // total_num_workers) * total_num_workers
            if total_num_workers > 0
            else len(shard_list_per_dataset[i])
        )
        shard_list_per_dataset[i] = shard_list_per_dataset[i][:idx_div]
        num_samples_list_per_dataset[i] = num_samples_list_per_dataset[i][:idx_div]

        # Only add used shards. Put back unused shards.
        next_shard_idx_per_dataset[i] += len(shard_list_per_dataset[i])
        if next_shard_shuffle_seed_per_dataset[i] is not None:
            next_shard_shuffle_seed_per_dataset[i] += next_shard_idx_per_dataset[i] // len(manifests[i])
        next_shard_idx_per_dataset[i] = next_shard_idx_per_dataset[i] % len(manifests[i])

    # Build WebDataset datastrings per dataset from selected shard names.
    datastrings = []
    for i, manifest_path in enumerate(manifest_paths):
        shard_root_source = "/".join(manifest_path.split("/")[:-1]) + "/"
        if len(shard_list_per_dataset[i]) > 1:
            curr_datastring = shard_root_source + "{" + ",".join(shard_list_per_dataset[i]) + "}.tar"
        elif len(shard_list_per_dataset[i]) == 1:
            curr_datastring = shard_root_source + shard_list_per_dataset[i][0] + ".tar"
        else:
            logging.debug(f"No shards found for dataset {i} in {manifest_path}")
            continue
        datastrings.append(curr_datastring)

    # Collapse per-shard sample counts into per-dataset totals.
    total_num_samples_per_dataset = [sum(i) for i in num_samples_list_per_dataset]

    return datastrings, total_num_samples_per_dataset, next_shard_idx_per_dataset, next_shard_shuffle_seed_per_dataset
