import torch


def sample_chunk(input_ids, attention_mask, seq_len, seed=None):
    if input_ids.shape[1] == seq_len + 1:
        start_idx = 0
    elif input_ids.shape[1] > seq_len + 1:
        if seed is not None:
            start_idx = torch.randint(
                0,
                input_ids.shape[1] - seq_len,
                (1,),
                generator=torch.Generator().manual_seed(seed),
            ).item()
        else:
            start_idx = torch.randint(0, input_ids.shape[1] - seq_len, (1,)).item()
    else:
        start_idx = 0
        seq_len = input_ids.shape[1] - 1

    inputs = input_ids[:, start_idx : start_idx + seq_len]
    mask = attention_mask[:, start_idx : start_idx + seq_len] if attention_mask is not None else None
    targets = input_ids[:, start_idx + 1 : start_idx + seq_len + 1]
    return inputs, mask, targets
