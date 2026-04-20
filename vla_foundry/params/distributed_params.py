from dataclasses import dataclass, field

from vla_foundry.distributed import init_distributed_device
from vla_foundry.params.base_params import BaseParams


@dataclass(frozen=True)
class DistributedParams(BaseParams):
    dist_url: str = field(default="env://")
    dist_backend: str = field(default="nccl")
    fsdp: bool = field(default=False)
    fsdp_cpu_offload: bool = field(default=False)
    fsdp_reshard_after_forward: bool = field(default=False)
    ddp_static_graph: bool = field(default=False)

    # The following should not be initialized by the user.
    # These will be initialized automatically in init_distributed_device()
    use_distributed: bool = field(default=False)
    world_size: int = field(default=1)
    rank: int = field(default=0)
    local_rank: int = field(default=0)
    device: str = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        init_distributed_device(self)
