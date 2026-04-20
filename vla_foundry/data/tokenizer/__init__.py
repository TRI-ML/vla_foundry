import torch
from transformers import AutoTokenizer

from vla_foundry.data.utils import text_to_seed


class DebugTokenizer:
    def __init__(self):
        self.pad_token = "[PAD]"

    def __call__(self, texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt"):
        batch_size = len(texts)
        seed = text_to_seed(texts[0])
        return {
            "input_ids": torch.randint(0, 100, (batch_size, max_length), generator=torch.Generator().manual_seed(seed)),
            "attention_mask": torch.ones((batch_size, max_length), dtype=torch.long),
        }


def get_tokenizer(tokenizer_name):
    if tokenizer_name == "debug":
        return DebugTokenizer()
    else:
        return AutoTokenizer.from_pretrained(tokenizer_name)
