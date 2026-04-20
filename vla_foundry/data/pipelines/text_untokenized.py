import json

import webdataset as wds

from vla_foundry.data.pipelines.base import BaseWebDatasetPipeline
from vla_foundry.data.pipelines.webdataset_cache import get_tarfile_to_samples_stage
from vla_foundry.data.tokenizer import get_tokenizer
from vla_foundry.data.utils import deterministic_shuffle, log_and_continue
from vla_foundry.params.base_data_params import DataParams


def batch_tokenize(batch, tokenizer, seq_len):
    texts = []
    for item in batch[0]:
        text = item.decode("utf-8") if isinstance(item, bytes) else item
        # Parse JSON if the text looks like JSON with a "text" field
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                text = parsed.get("text", text)
            except json.JSONDecodeError:
                pass
        texts.append(text)
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len + 1,  # +1 because next token prediction
        return_tensors="pt",
    )
    return tokenized["input_ids"]


class TextUntokenizedPipeline(BaseWebDatasetPipeline):
    def __init__(self, modality: str, data_params: DataParams, batch_size: int):
        super().__init__(modality, data_params, batch_size)
        self.tokenizer = get_tokenizer(data_params.tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def create_pipeline(self, datastring: str, checkpoint_num: int):
        cache_cfg = self.data_params.dataset_cache
        pipeline = [
            wds.SimpleShardList(datastring),
            deterministic_shuffle(
                bufsize=self.data_params.shuffle_buffer_size,
                initial=self.data_params.shuffle_initial,
                seed=self.data_params.seed,
                epoch=checkpoint_num,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            get_tarfile_to_samples_stage(
                cache_cfg=cache_cfg,
                handler=log_and_continue,
            ),
            wds.to_tuple("json", handler=log_and_continue),
            wds.batched(self.batch_size, partial=False),
            wds.map(self.tokenize_wrapper),
        ]
        return pipeline

    def tokenize_wrapper(self, batch):
        input_ids = batch_tokenize(batch, self.tokenizer, self.data_params.seq_len)
        return {"input_ids": input_ids, "attention_mask": None}
