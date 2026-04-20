from collections.abc import Callable
from itertools import islice

import webdataset as wds

from vla_foundry.data.pipelines.image_caption import ImageCaptionPipeline
from vla_foundry.data.pipelines.robotics import RoboticsPipeline
from vla_foundry.data.pipelines.text import TextPipeline
from vla_foundry.data.pipelines.text_untokenized import TextUntokenizedPipeline


class FiniteDataPipeline(wds.DataPipeline):
    def __init__(self, *args, save_configs: Callable[[str], None] = None, **kwargs):
        self.save_configs = save_configs  # This is a function
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """Iterate through up to self.nsamples steps.

        Note: wds.DataPipeline.__iter__ inexplicably only limits the number of samples with self.nsamples if
        self.repetitions != 1. Here, we always slice using self.nsamples, if self.nsamples > 0.
        """
        # Handle case where nsamples might not be set (None) or is 0
        nsamples = getattr(self, "nsamples", 0)
        if nsamples and nsamples > 0:
            return islice(self.iterator(), nsamples)
        else:
            return self.iterator()


def create_wds_pipeline(datastring, modality, batch_size, checkpoint_num, data_params):
    if modality == "text":
        pipeline = TextPipeline(modality, data_params, batch_size)
    elif modality == "text_untokenized":
        pipeline = TextUntokenizedPipeline(modality, data_params, batch_size)
    elif modality == "image_caption":
        pipeline = ImageCaptionPipeline(modality, data_params, batch_size)
    elif modality == "robotics":
        pipeline = RoboticsPipeline(modality, data_params, batch_size)
    else:
        raise ValueError(f"{modality} webdataset pipeline not supported")

    pipeline_components = pipeline.create_pipeline(datastring, checkpoint_num)
    return FiniteDataPipeline(*pipeline_components, save_configs=pipeline.save_configs)
