import logging
from dataclasses import dataclass, field

from vla_foundry.params.base_params import BaseParams


@dataclass(frozen=True)
class HyperParams(BaseParams):
    precision: str = field(default="amp_bfloat16")
    global_batch_size: int = field(default=512)
    per_gpu_batch_size: int = field(default=8)
    seed: int = field(default=42)
    lr: float = field(default=1e-4)
    lr_scheduler: str = field(default="cosine")
    warmup: str = field(default="1000")
    decay: str = field(default="0.3")
    lr_cooldown_end: float = field(default=0.0)
    force_min_lr: float = field(default=0.0)
    optimizer: str = field(default="adamw")
    wd: float = field(default=0.01)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    eps: float = field(default=1.0e-8)
    loss_function: str = field(default="cross_entropy")
    z_loss_coefficient: float = field(default=0.0)
    grad_clip_norm: float = field(default=None)
    grad_checkpointing: bool = field(default=False)
    torchcompile: bool = field(default=False)

    # Shared attributes. Overwritten in init_shared_attributes.
    world_size: int = field(default=1)

    def __post_init__(self):
        super().__post_init__()
        assert self.lr >= self.lr_cooldown_end, "lr must be greater than lr_cooldown_end"
        if self.precision == "pure_bf16":
            object.__setattr__(self, "precision_amp", False)
            object.__setattr__(self, "precision_pure_bf16", True)
        elif self.precision == "amp" or self.precision == "amp_bf16" or self.precision == "amp_bfloat16":
            object.__setattr__(self, "precision_pure_bf16", False)
            object.__setattr__(self, "precision_amp", True)
        elif self.precision == "fp32" or self.precision == "float32":
            object.__setattr__(self, "precision_amp", False)
            object.__setattr__(self, "precision_pure_bf16", False)
        else:
            logging.warning(f"Precision {self.precision} uknown, using default float32")
            object.__setattr__(self, "precision", "float32")
            object.__setattr__(self, "precision_amp", False)
            object.__setattr__(self, "precision_pure_bf16", False)

    @property
    def accum_freq(self):
        combined_batch_size = self.world_size * self.per_gpu_batch_size
        assert self.global_batch_size % combined_batch_size == 0
        return self.global_batch_size // combined_batch_size

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        object.__setattr__(self, "world_size", cfg.distributed.world_size)
