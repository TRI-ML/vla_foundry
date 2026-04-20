import logging

import torch
import webdataset as wds

from vla_foundry.data.pipelines.base import BaseWebDatasetPipeline
from vla_foundry.data.pipelines.webdataset_cache import get_tarfile_to_samples_stage
from vla_foundry.data.utils import deterministic_shuffle, log_and_continue


def filter_lt_seqlen(seq_len: int, x: list) -> bool:
    valid_sample = len(x) > seq_len
    if not valid_sample:
        logging.warning(
            f"Sample sequence length: {len(x)} not larger than seq_len: {seq_len}. "
            "Skipping sample. NOTE: sample sequence length should be one greater than seq_len."
        )
    return valid_sample


class TextPipeline(BaseWebDatasetPipeline):
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
            wds.decode(handler=log_and_continue),
            wds.map(lambda sample: {"input_ids": sample["json.gz"]}, handler=log_and_continue),
            wds.select(lambda x: filter_lt_seqlen(self.data_params.seq_len, x["input_ids"])),
            wds.batched(self.batch_size, partial=False),
            wds.map(lambda batch: {"input_ids": torch.LongTensor(batch["input_ids"]), "attention_mask": None}),
        ]
        return pipeline
