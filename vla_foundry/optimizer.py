import logging

from torch import optim
from torch.distributed.checkpoint.state_dict import _init_optim_state
from torch.distributed.tensor import DTensor, distribute_tensor

from vla_foundry.file_utils import pt_load


def create_optimizer(hparams, model):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    if hparams.optimizer == "adamw":
        optimizer = optim.AdamW(
            [
                {"params": no_decay_params, "weight_decay": 0.0},
                {"params": decay_params, "weight_decay": hparams.wd},
            ],
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
            eps=hparams.eps,
        )
    else:
        raise ValueError("Only adamw supported for now")

    return optimizer


def load_optimizer(optimizer, checkpoint_path, use_fsdp):
    optimizer_path = checkpoint_path.replace("checkpoint_", "optimizer_")
    optimizer_checkpoint = pt_load(optimizer_path, map_location="cpu")
    if "optimizer" in optimizer_checkpoint:
        osd = optimizer_checkpoint["optimizer"]
        if use_fsdp:
            _init_optim_state(optimizer)
            param_groups = optimizer.state_dict()["param_groups"]
            state = optimizer.state_dict()["state"]

            full_param_groups = osd["param_groups"]
            full_state = osd["state"]

            for param_group, full_param_group in zip(param_groups, full_param_groups, strict=False):
                for key, value in full_param_group.items():
                    if key == "params":
                        continue
                    param_group[key] = value
                for pid, full_pid in zip(param_group["params"], full_param_group["params"], strict=False):
                    if pid not in state:
                        continue
                    if full_pid not in full_state:
                        continue
                    param_state = state[pid]
                    full_param_state = full_state[full_pid]
                    for attr, full_tensor in full_param_state.items():
                        sharded_tensor = param_state[attr]
                        if isinstance(sharded_tensor, DTensor):
                            param_state[attr] = distribute_tensor(
                                full_tensor,
                                sharded_tensor.device_mesh,
                                sharded_tensor.placements,
                            )
                        else:
                            param_state[attr] = full_tensor
            optimizer.load_state_dict(
                {
                    "param_groups": param_groups,
                    "state": state,
                }
            )
        else:
            optimizer.load_state_dict(osd)
        logging.info("=> resuming optimizer")
    else:
        logging.info("=> WARNING: not resuming optimizer.")
