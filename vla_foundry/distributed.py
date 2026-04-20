# Partly from open_clip.
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
)
from torch.distributed.fsdp import (
    # FullyShardedDataParallel as FSDP,
    fully_shard as FSDP2,
)

from vla_foundry.models.fsdp_block import FSDPBlock


def is_global_master(cfg):
    return cfg.distributed.rank == 0


def is_local_master(cfg):
    return cfg.distributed.local_rank == 0


def is_master(cfg, local=False):
    return is_local_master(cfg) if local else is_global_master(cfg)


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def init_distributed_device(distributed_params):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    object.__setattr__(distributed_params, "use_distributed", False)  # bypass Frozen=True
    object.__setattr__(distributed_params, "world_size", 1)
    object.__setattr__(distributed_params, "rank", 0)
    object.__setattr__(distributed_params, "local_rank", 0)
    if is_using_distributed():
        # DDP via torchrun, torch.distributed.launch
        # Note that this currently assumes that the world size is all gpus in a node.
        local_rank, _, _ = world_info_from_env()
        object.__setattr__(distributed_params, "local_rank", local_rank)
        torch.distributed.init_process_group(
            backend=distributed_params.dist_backend, init_method=distributed_params.dist_url
        )
        object.__setattr__(distributed_params, "world_size", torch.distributed.get_world_size())
        object.__setattr__(distributed_params, "rank", torch.distributed.get_rank())
        object.__setattr__(distributed_params, "use_distributed", True)

    if torch.cuda.is_available():
        device = f"cuda:{distributed_params.local_rank}" if distributed_params.use_distributed else "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    object.__setattr__(distributed_params, "device", device)
    device = torch.device(device)
    return device


def broadcast_object(cfg, obj, src=0):
    objects = [obj] if cfg.distributed.rank == src else [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_gather_object(cfg, obj, dst=0):
    # gather a pickle-able python object across all ranks
    objects = [None for _ in range(cfg.distributed.world_size)]
    dist.all_gather_object(objects, obj)
    return objects


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_model_precision(cfg):
    """
    Determine the appropriate model precision based on distributed configuration.
    Returns the dtype that should be used for model parameters.
    """
    # FSDP handles precision through MixedPrecision policy but in either case the model precision should be bfloat16
    if cfg.hparams.precision_amp or cfg.hparams.precision_pure_bf16:
        return torch.bfloat16
    else:
        return torch.float32  # Default precision if not specified


def move_buffers_to_device(model, device, verbose=False):
    """
    Recursively move all buffers to the specified device.

    Some HuggingFace models (like SigLIP) have buffers (e.g., position_ids) that
    are created on CPU and not automatically moved by FSDP. This causes issues
    with torch.compile which requires all tensors to be on the same device.
    """
    device = torch.device(device) if isinstance(device, str) else device
    moved_count = 0
    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.device != device:
                if verbose:
                    print(f"Moving buffer {module.__class__.__name__}.{name} from {buf.device} to {device}")
                module._buffers[name] = buf.to(device)
                moved_count += 1
    if moved_count > 0:
        print(f"=> Moved {moved_count} buffers to {device}")


def wrap_fsdp_ddp(model, device, cfg):
    # Move all buffers to device before FSDP wrapping.
    # Some HuggingFace models (like SigLIP) have buffers (e.g., position_ids) that
    # are created on CPU and not automatically moved by FSDP.
    move_buffers_to_device(model, device)

    if cfg.distributed.fsdp:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=None,
            reduce_dtype=None,
            output_dtype=None,
        )
        if cfg.hparams.precision_amp:
            print("=> using bfloat16 params as part of fsdp amp policy.")
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
            )
        elif cfg.hparams.precision_pure_bf16:
            print("=> using pure bfloat16 params as part of fsdp amp policy.")
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
            )

        if cfg.distributed.rank == 0:
            print(f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Before FSDP {torch.cuda.memory_allocated() / 1024**3:.3} GB")

        fsdp_kwargs = {
            "mp_policy": mp_policy,
            "offload_policy": CPUOffloadPolicy() if cfg.distributed.fsdp_cpu_offload else None,
            "reshard_after_forward": cfg.distributed.fsdp_reshard_after_forward,
        }
        print("=> FSDP kwargs: ", fsdp_kwargs)

        # Initialize FSDP. Use the same seed across workers to ensure reset_parameters is the same across workers.
        random_seed(cfg.hparams.seed, rank=0)

        # Find scalar parameters to ignore (FSDP2 doesn't support scalar parameters)
        # CLIP model has scalar parameters and needs this.
        scalar_params = set()
        scalar_param_names = []
        for name, param in model.named_parameters():
            if param.dim() == 0:  # scalar parameter
                scalar_params.add(param)
                scalar_param_names.append(name)

        if scalar_param_names:
            print(f"=> Ignoring {len(scalar_param_names)} scalar parameters for FSDP: {scalar_param_names}")

        # Convert to frozenset to avoid mutation during FSDP operations
        ignored_params = frozenset(scalar_params) if scalar_params else None

        # Get block types from model if it provides them (for HF models)
        hf_block_types = None
        if hasattr(model, "get_fsdp_block_types"):
            hf_block_types = model.get_fsdp_block_types()

        for p in model.modules():
            # Check if module should be wrapped:
            # 1. Custom blocks inherit from FSDPBlock
            # 2. HF library blocks match types from get_fsdp_block_types()
            should_wrap = isinstance(p, FSDPBlock) or (hf_block_types and isinstance(p, hf_block_types))

            if should_wrap:
                # Get scalar parameters specific to this module
                module_scalar_params = set()
                for param in p.parameters():
                    if param.dim() == 0 and param in scalar_params:
                        module_scalar_params.add(param)

                module_ignored_params = frozenset(module_scalar_params) if module_scalar_params else None
                if module_ignored_params:
                    FSDP2(p, ignored_params=module_ignored_params, **fsdp_kwargs)
                else:
                    FSDP2(p, **fsdp_kwargs)

        # Wrap the entire model with ignored scalar parameters
        # ignored_params is a frozenset of scalar parameters to ignore if any exist
        FSDP2(model, ignored_params=ignored_params, **fsdp_kwargs)

        print(
            f"After FSDP parameter num: {sum(p.numel() for p in model.parameters()):,} on rank {cfg.distributed.rank}"
        )
        print(f"After FSDP {torch.cuda.memory_allocated() / 1024**3:.3} GB on rank {cfg.distributed.rank}")
    else:
        # Move model to device before DDP wrapping
        model = model.to(device, dtype=get_model_precision(cfg))

        ddp_args = {
            "find_unused_parameters": True,  # Model has unused parameters (e.g., CLIP logit scale)
            "static_graph": False,  # Must be False when find_unused_parameters=True
        }
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    return model
