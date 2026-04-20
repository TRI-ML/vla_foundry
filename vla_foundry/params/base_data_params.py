import multiprocessing
from dataclasses import dataclass, field

import draccus

from vla_foundry.params.base_params import BaseParams


@dataclass(frozen=True)
class DatasetCacheParams(BaseParams):
    """Controls WebDataset shard caching behavior."""

    enabled: bool = field(default=False)
    cache_dir: str | None = field(default=None)
    cache_size_gb: int | None = field(default=None)
    cache_verbose: bool | None = field(default=None)


@dataclass(frozen=True)
class DataParams(draccus.ChoiceRegistry, BaseParams):
    type: str = field(default=None)
    dataset_manifest: list[str] = field(default_factory=list)
    dataset_weighting: list[float] = field(default_factory=list)
    dataset_modality: list[str] = field(default_factory=list)
    val_dataset_manifest: list[str] = field(default_factory=list)
    val_dataset_weighting: list[float] = field(default_factory=list)
    allow_multiple_epochs: bool = False
    num_workers: int | None = field(default=None)  # Auto-calculated per-GPU if None
    prefetch_factor: int = field(default=4)  # Number of batches to prefetch per worker (PyTorch DataLoader)
    seq_len: int = field(default=2048)
    shuffle: bool = field(default=True)
    shuffle_buffer_size: int = field(default=2000)
    shuffle_initial: int = field(default=500)
    use_hf_fast_tokenizer: bool = True
    hf_fast_tokenizers_parallelism: bool = True
    hf_fast_tokenizer_rayon_threads: int | None = None
    # Whether the dataloader returns batches in strict worker order. Robotics data pipelines
    # involve heavy per-sample processing (decoding, augmentation), so workers often finish at
    # different rates. Setting this to False allows batches to be yielded as soon as any worker
    # is ready, avoiding idle time waiting for the next in-order worker. The trade-off is that
    # batch ordering becomes non-deterministic across runs.
    dataloader_in_order: bool = False
    dataset_cache: DatasetCacheParams = field(default_factory=DatasetCacheParams)
    # Shared attributes. Overwritten in init_shared_attributes.
    seed: int = field(default=42)

    def __init__(self):
        raise NotImplementedError("DataParams should not be instantiated directly. Use a subclass with data.type=...")

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "dataset_weighting", [float(i) for i in self.dataset_weighting])
        if not self.shuffle:
            object.__setattr__(self, "shuffle_buffer_size", 0)
            object.__setattr__(self, "shuffle_initial", 0)
        if self.type is None:
            object.__setattr__(self, "type", getattr(self.__class__, "_type", None))

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        object.__setattr__(self, "seed", cfg.hparams.seed)

        # Set num_workers if not explicitly configured
        if self.num_workers is None:
            world_size = cfg.distributed.world_size if hasattr(cfg, "distributed") else 1
            cpu_count = multiprocessing.cpu_count()
            # Calculate per-GPU workers to match total CPU count
            default_workers = max(1, cpu_count // world_size)
            object.__setattr__(self, "num_workers", default_workers)
