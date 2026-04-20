from vla_foundry.params.base_data_params import DataParams


class BaseWebDatasetPipeline:
    def __init__(self, modality: str, data_params: DataParams, batch_size: int):
        self.modality = modality
        self.data_params = data_params
        self.batch_size = batch_size

    def create_pipeline(self, datastring: str, checkpoint_num: int):
        raise NotImplementedError("Implemented in individual classes.")

    def save_configs(self, experiment_path: str):
        # Save any necessary dataloader/pipeline configs (e.g., normalization config)
        pass
