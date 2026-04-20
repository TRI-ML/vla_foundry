from dataclasses import dataclass, field

from vla_foundry.params.base_params import BaseParams


@dataclass(frozen=True)
class EMAParams(BaseParams):
    enabled: bool = field(default=False)  # Whether to use EMA
    type: str = field(default="ema")  # "vanilla" or "ema" (adaptive)
    alpha: float = field(default=0.999)  # For vanilla EMA - fixed decay rate
    update_after_step: int = field(default=0)  # For adaptive EMA - start updating after N steps
    inv_gamma: float = field(default=1.0)  # For adaptive EMA - inverse gamma warmup factor
    power: float = field(default=0.75)  # For adaptive EMA - warmup power (2/3 for long, 3/4 for short training)
    min_value: float = field(default=0.0)  # For adaptive EMA - minimum decay rate
    max_value: float = field(default=0.9999)  # For adaptive EMA - maximum decay rate
