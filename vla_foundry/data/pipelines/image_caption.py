import webdataset as wds

from vla_foundry.data.augmentations.decode_and_augment import Augmentations
from vla_foundry.data.pipelines.base import BaseWebDatasetPipeline
from vla_foundry.data.pipelines.webdataset_cache import get_tarfile_to_samples_stage
from vla_foundry.data.processor import apply_chat_template, get_processor
from vla_foundry.data.utils import deterministic_shuffle, log_and_continue
from vla_foundry.params.base_data_params import DataParams


def filter_no_caption_or_no_image(sample):
    has_caption = any(k == "txt" or k.endswith(".txt") for k in sample)
    has_image = any(k == ext or k.endswith(f".{ext}") for k in sample for ext in ("png", "jpg", "jpeg", "webp"))
    return has_caption and has_image


def _list_collation_fn(batch):
    """Collate samples into lists without stacking tensors.

    Image-caption data has variable-size images that cannot be stacked.
    The HF processor applied after batching handles resizing and padding.
    """
    return {k: [s[k] for s in batch] for k in batch[0]}


class ImageCaptionPipeline(BaseWebDatasetPipeline):
    def __init__(self, modality: str, data_params: DataParams, batch_size: int):
        super().__init__(modality, data_params, batch_size)
        self.processor = get_processor(data_params)
        self.processor_kwargs = getattr(data_params, "processor_kwargs", {})
        self.augmentations = Augmentations(data_params.augmentation)

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
            wds.select(filter_no_caption_or_no_image),
            wds.map(self.augmentations.decode_and_augment_sample, handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map(
                lambda sample: {
                    **sample,
                    "text": apply_chat_template(self.processor, 1, sample["text"]),
                }
            ),
            wds.batched(self.batch_size, partial=False, collation_fn=_list_collation_fn),
            wds.map(
                lambda sample: self.processor(
                    images=[[img] for img in sample["image"]],
                    text=sample["text"],
                    return_tensors="pt",
                    padding="max_length",
                    padding_side="right",
                    max_length=self.data_params.seq_len + 1,
                    truncation=True,
                    **self.processor_kwargs,
                ),
                handler=log_and_continue,
            ),
            # BatchFeature is preserved so batch_handlers can use .to(device)
            # to forward all VLM-specific tensor keys automatically.
            wds.map(lambda sample: (sample.pop("text", None), sample)[1]),
        ]
        return pipeline
