import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length):
    def _lr_adjuster(step):
        lr = _warmup_lr(base_lr, warmup_length, step) if step < warmup_length else base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def const_lr_cooldown(
    optimizer,
    base_lr,
    warmup_length,
    steps,
    cooldown_steps,
    cooldown_power=1.0,
    cooldown_end_lr=0.0,
):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e / es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps, min_lr, force_min_lr):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr)
            lr = max(lr, force_min_lr)
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def warmup_constant_decay_lr(optimizer, base_lr, warmup_length, decay_length, steps, min_lr):
    """
    Learning rate schedule with three phases:
    1. Linear warmup from 0 to base_lr
    2. Constant at base_lr
    3. Cosine decay from base_lr to min_lr

    Args:
        optimizer: The optimizer to adjust
        base_lr: Peak learning rate
        warmup_length: Number of warmup steps (linear increase)
        decay_length: Number of decay steps (cosine decrease)
        steps: Total number of training steps
        min_lr: Minimum learning rate at end of decay
    """
    decay_start = steps - decay_length

    def _lr_adjuster(step):
        if step < warmup_length:
            # Linear warmup
            lr = _warmup_lr(base_lr, warmup_length, step)
        elif step < decay_start:
            # Constant phase
            lr = base_lr
        else:
            # Cosine decay
            e = step - decay_start
            lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / decay_length)) * (base_lr - min_lr)
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def _parse_steps_or_fraction(value, total_steps):
    """Parse a value that can be either absolute steps or a fraction of total steps."""
    val = float(value)
    if val < 1:
        return int(val * total_steps)
    return int(val)


def create_scheduler(hparams, optimizer, total_train_samples):
    total_steps = total_train_samples // hparams.global_batch_size
    warmup = _parse_steps_or_fraction(hparams.warmup, total_steps)
    if hparams.lr_scheduler == "cosine":
        scheduler = cosine_lr(
            optimizer,
            hparams.lr,
            warmup,
            total_steps,
            hparams.lr_cooldown_end,
            hparams.force_min_lr,
        )
    elif hparams.lr_scheduler == "const":
        scheduler = const_lr(
            optimizer,
            hparams.lr,
            warmup,
        )
    elif hparams.lr_scheduler == "warmup_constant_decay":
        decay = _parse_steps_or_fraction(hparams.decay, total_steps)
        scheduler = warmup_constant_decay_lr(
            optimizer,
            hparams.lr,
            warmup,
            decay,
            total_steps,
            hparams.lr_cooldown_end,
        )
    else:
        raise ValueError(
            f"Unknown scheduler, {hparams.lr_scheduler}. Available options are: cosine, const, warmup_constant_decay."
        )
    return scheduler
