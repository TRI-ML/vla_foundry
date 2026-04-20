from contextlib import suppress

import torch


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16" or precision == "pure_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        def autocast_fn():
            return torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)

        return autocast_fn
    else:
        return suppress
