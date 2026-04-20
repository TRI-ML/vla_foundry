import torch
import torch.nn as nn

from vla_foundry.params.model_params import ModelParams


class BaseModelMeta(type):
    """Metaclass that automatically calls _post_init after __init__."""

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        instance.__init__(*args, **kwargs)

        # Call _post_init if it exists
        if hasattr(instance, "_post_init"):
            instance._post_init()

        return instance


class BaseModel(nn.Module, metaclass=BaseModelMeta):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, checkpoint: str | None = None, device: str = "cuda"):
        """Load a model from a HF Hub repo, S3 path, or local directory.

        Args:
            pretrained_model_name_or_path: One of:
                - HF repo ID: ``"TRI-ML/my-model"``
                - hf:// URI: ``"hf://TRI-ML/my-model"``
                - S3 path: ``"s3://bucket/path/to/experiment/"``
                - Local path: ``"/path/to/experiment/"``
            checkpoint: Checkpoint filename (e.g. ``"checkpoint_11.pt"``).
                If None, loads the latest checkpoint.
            device: Device to move the model to.

        Returns:
            Model in eval mode with weights loaded.

        Example::

            from vla_foundry.models.base_model import BaseModel
            model = BaseModel.from_pretrained("TRI-ML/vla-foundry-qwen3-5-2b-diffusion-policy")
        """
        # Lazy: distributed -> models/__init__ -> stable_diffusion -> base_model
        #   -> train_experiment_params -> distributed_params -> distributed
        from vla_foundry.file_utils import get_latest_checkpoint, load_model_checkpoint
        from vla_foundry.hf_hub import normalize_checkpoint_locator
        from vla_foundry.models.registry import create_model
        from vla_foundry.params.train_experiment_params import TrainExperimentParams, load_params_from_yaml

        path = normalize_checkpoint_locator(pretrained_model_name_or_path)

        # Load config and build model architecture (no pretrained backbone weights needed).
        # load_params_from_yaml stashes _config_origin on params so the
        # vlm_foundry_backbone constructor can resolve vlm_config_model.yaml
        # from the published repo on its own.
        config_path = f"{path}/config.yaml" if not path.endswith(".yaml") else path
        # localize_params rewrites s3:// entries in the loaded config to sibling
        # files next to config.yaml (when they exist), so a locally-synced
        # training directory doesn't require AWS credentials just to read its
        # preprocessing_config / stats.json. Skip for S3-hosted configs; those
        # are already remote, and we can't localize what we can't fetch.
        train_params = load_params_from_yaml(
            TrainExperimentParams,
            config_path,
            localize_params=not config_path.startswith("s3://"),
        )

        model = create_model(train_params.model, load_pretrained=False)

        # Resolve checkpoint path
        if checkpoint is not None:
            ckpt_path = f"{path}/checkpoints/{checkpoint}"
        else:
            ckpt_path = get_latest_checkpoint(path)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoints found in {path}")

        load_model_checkpoint(model, ckpt_path)
        model = model.to(device)
        model.eval()
        return model

    def __init__(self, model_params: ModelParams):
        super().__init__()
        self.model_params = model_params
        # Use object.__setattr__ to prevent ema_model from being registered as a submodule
        # This prevents it from being included in state_dict() during checkpoint saving
        object.__setattr__(self, "ema_model", None)

    def _post_init(self):
        """Called automatically after subclass initialization is complete."""
        if self.model_params.freeze:
            self.freeze_parameters()

    def freeze_parameters(self):
        """Freeze all parameters in the model by setting requires_grad to False."""
        for param in self.parameters():
            param.requires_grad = False

    def set_ema_model(self, ema_model):
        """Set the EMA model for inference/evaluation.

        Args:
            ema_model: EMA model instance (e.g., from create_ema_model())
        """
        # Use object.__setattr__ to bypass nn.Module's __setattr__
        # This prevents the ema_model from being registered as a child module
        # and therefore prevents it from being included in state_dict()
        object.__setattr__(self, "ema_model", ema_model)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        raise NotImplementedError
