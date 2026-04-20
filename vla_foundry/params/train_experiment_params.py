import logging
import os
import tempfile
import weakref
from dataclasses import dataclass, field

import yaml

from vla_foundry.data.utils import epochs_to_samples
from vla_foundry.file_utils import localize_paths, yaml_load
from vla_foundry.hf_hub import resolve_hf_path as _resolve_hf_path
from vla_foundry.params.base_params import BaseParams
from vla_foundry.params.data_params import DataParams  # not from base_data_params so it loads registered params
from vla_foundry.params.distributed_params import DistributedParams
from vla_foundry.params.ema_params import EMAParams
from vla_foundry.params.hyper_params import HyperParams
from vla_foundry.params.model_params import ModelParams


@dataclass(frozen=True)
class TrainExperimentParams(BaseParams):
    """
    Top-level, immutable configuration for a training experiment.
    """

    # -- Logging and remote sync
    # Optional explicit experiment name.
    # If `None`, a name will be generated at runtime (see `vla_foundry.utils.get_experiment_name`).
    name: str = field(default=None)

    # If resolve_configs is True, main.py will print the resolved config and stop.
    # The optional resolve_configs_path field will dump that printed output to {path}/resolved_config.yaml.
    resolve_configs: bool = field(default=False)
    resolve_configs_path: str | None = field(default=None)

    # Optional base directory where the experiment folder is created. If `None`, defaults to `experiments/`.
    save_path: str = field(default=None)
    wandb: bool = field(default=True)
    db_logging: bool = field(default=True)  # Log training runs to DynamoDB for dashboard tracking
    wandb_entity: str = field(default=os.getenv("WANDB_ENTITY"))
    wandb_project_name: str = field(default=os.getenv("WANDB_PROJECT", "vla_foundry"))
    wandb_tags: list[str] = field(default_factory=list)
    log_every_n_steps: int = field(default=20)
    log_level: str = field(default="INFO")
    # Optional path to S3 to which the experiment directory is synced.
    remote_sync: str = field(default=None)
    remote_sync_fixed_path: str = field(default="s3://your-bucket/your-path/vla_foundry_models_fixed/")

    # --Training
    # Total number of samples to train on. If `num_epochs` is also set, it must
    # resolve to the same value.
    total_train_samples: int = field(default=None)
    # Number of epochs over the input datasets. If set, it is converted to
    # `total_train_samples` using `epochs_to_samples`. If
    # `total_train_samples` is also set, the two must agree.
    num_epochs: int = field(default=None)
    # Number of checkpoint windows the total budget is split into.
    num_checkpoints: int = field(default=5)
    max_checkpoint_limit: int = field(default=None)

    # --Validation
    total_val_samples: int = field(default=None)
    val_every_n_checkpoints: int = field(default=1)

    # --Params Subclasses
    data: DataParams = field(default_factory=DataParams)
    distributed: DistributedParams = field(default_factory=DistributedParams)
    ema: EMAParams = field(default_factory=EMAParams)
    hparams: HyperParams = field(default_factory=HyperParams)
    model: ModelParams = field(default_factory=ModelParams)

    def __post_init__(self):
        """
        Derive fields, initialize shared attributes, and validate consistency.
        """
        super().__post_init__()

        # Allow sub-params to read the full config and set shared/derived fields.
        self.init_shared_attributes(self)

        derived_total_train_samples = None
        if self.num_epochs is not None:
            derived_total_train_samples = epochs_to_samples(self.data.dataset_manifest, self.num_epochs)

        if self.total_train_samples is not None and derived_total_train_samples is not None:
            assert self.total_train_samples == derived_total_train_samples, (
                "Both total_train_samples and num_epochs are set, but they resolve to different training budgets: "
                f"total_train_samples={self.total_train_samples}, "
                f"derived_total_train_samples={derived_total_train_samples}."
            )
            logging.warning(
                "Both total_train_samples and num_epochs are set and consistent; "
                "using the explicit total_train_samples value."
            )

        # If total_train_samples is already provided, keep it as the source of truth.
        # Otherwise derive it from num_epochs.
        if self.total_train_samples is None and derived_total_train_samples is not None:
            logging.info(f"Setting total_train_samples based on self.num_epochs={self.num_epochs} epochs.")
            object.__setattr__(self, "total_train_samples", derived_total_train_samples)

        self.check_asserts()

    def check_asserts(self):
        """
        Validate cross-field invariants for batch sizing and dataset config.
        """
        # Global batch must shard evenly across processes.
        assert self.hparams.global_batch_size % self.distributed.world_size == 0

        # Consistency between accumulation, per-GPU microbatch, and global batch.
        assert (
            self.hparams.accum_freq * self.distributed.world_size * self.hparams.per_gpu_batch_size
            == self.hparams.global_batch_size
        )

        # Dataset-related lists must align in length.
        assert len(self.data.dataset_manifest) == len(self.data.dataset_modality)
        assert len(self.data.dataset_manifest) == len(self.data.dataset_weighting)

        # Each dataset_modality entry must be a valid pipeline type, and each
        # pipeline accesses type-specific attributes on the DataParams subclass.
        # Validate that each modality is compatible with the data type.
        valid_modalities = {"text", "text_untokenized", "image_caption", "robotics"}
        # Map each modality to the data type(s) whose DataParams subclass exposes
        # the attributes that the corresponding pipeline needs.
        modality_compatible_types = {
            "robotics": {"robotics"},
            "image_caption": {"image_caption"},
            "text": {"text", "text_untokenized"},
            "text_untokenized": {"text_untokenized"},
        }
        for modality in self.data.dataset_modality:
            if modality not in valid_modalities:
                raise ValueError(f"Unknown dataset_modality '{modality}'. Valid modalities: {sorted(valid_modalities)}")
            compatible = modality_compatible_types[modality]
            if self.data.type not in compatible:
                raise ValueError(
                    f"dataset_modality '{modality}' is incompatible with data.type '{self.data.type}'. "
                    f"The '{modality}' pipeline requires data.type to be one of {sorted(compatible)}."
                )

        # For data types with images, seq_len must be >= img_num_tokens to avoid
        # truncating image token placeholders (causes assertion failures downstream).
        if (
            hasattr(self.data, "img_num_tokens")
            and self.data.img_num_tokens is not None
            and self.data.seq_len < self.data.img_num_tokens
        ):
            raise ValueError(
                f"data.seq_len ({self.data.seq_len}) must be >= data.img_num_tokens "
                f"({self.data.img_num_tokens}). A shorter seq_len truncates image token "
                f"placeholders, causing shape mismatches during training."
            )

        # Training budget must be resolved at this point.
        assert self.total_train_samples is not None

        # If epochs were requested, multiple passes must be allowed.
        if self.num_epochs is not None:
            assert self.data.allow_multiple_epochs

        # This check causes an error when loading from yaml due to load-time constraints.
        # Commenting out for now.
        # if self.distributed.fsdp and not self.distributed.use_distributed:
        #     raise ValueError(f"--fsdp can only be specified in distributed mode.")


def load_params_from_yaml(params_class: type[BaseParams], path: str, localize_params: bool = False) -> BaseParams:
    """
    Load a draccus params object from a yaml file with support for s3 and hf:// paths.

    Warning:
    If loading from s3 or hf://, the file will be copied to a temporary file and deleted after loading.
    This does not allow !include statements in the yaml files because those need to be relative to the file.
    Hopefully remote configs do not have !include statements (they shouldn't).

    Args:
        params_class: dataclass type to load.
        path: local filesystem path, S3 URI, or ``hf://repo_id/file.yaml`` to the YAML file.
        localize_params: if True, first localize all paths in the config before loading.
    """
    original_path = path

    # Resolve hf:// paths to local cache before anything else
    if path.startswith("hf://"):
        path = _resolve_hf_path(path)

    if localize_params:
        config_dict = yaml_load(path)
        base_path, _ = os.path.split(path)
        yaml_dict = localize_paths(config_dict, base_path)

        # Create a temporary file
        fd, temp_file_path = tempfile.mkstemp(suffix=".yaml", prefix="localized_config_")
        os.close(fd)  # Close the file descriptor
        with open(temp_file_path, "w") as f:
            yaml.dump(yaml_dict, f)
        path = temp_file_path

    # Use from_file method which handles unknown key stripping
    params = params_class.from_file(path)

    # Stash the directory the config was loaded from so backbone constructors
    # (e.g. vlm_foundry_backbone) can resolve sibling files like
    # ``vlm_config_model.yaml`` without needing the caller to plumb paths.
    # Preserve the original URI scheme (hf://, s3://) so remote sibling
    # resolution still goes through the right fetch path. Use a weak-ref map
    # rather than mutating __dict__ (which would break callers that round-trip
    # the dataclass via **kwargs).
    origin = original_path.rsplit("/", 1)[0] if "/" in original_path else ""
    _record_config_origin(params, origin)
    # The backbone may live at params.model.vision_language_backbone
    # (TrainExperimentParams) or params.vision_language_backbone (ModelParams
    # loaded directly). Handle both shapes.
    vlb = getattr(getattr(params, "model", None), "vision_language_backbone", None) or getattr(
        params, "vision_language_backbone", None
    )
    if vlb is not None and getattr(vlb, "type", None) == "vlm_foundry_backbone":
        _record_config_origin(vlb, origin)

    return params


# Map from params instance id -> origin path. Cleared via weakref finalizer
# when the params object is garbage-collected. Using a side table rather than
# an attribute on the params object itself avoids leaking into __dict__, which
# would break callers that round-trip the dataclass via ``**kwargs``.
_ORIGIN_BY_ID: dict[int, str] = {}


def _record_config_origin(params_obj, origin: str) -> None:
    pid = id(params_obj)
    _ORIGIN_BY_ID[pid] = origin
    weakref.finalize(params_obj, _ORIGIN_BY_ID.pop, pid, None)


def get_config_origin(params_obj) -> str | None:
    """Return the directory a params object was loaded from, or None if unknown."""
    return _ORIGIN_BY_ID.get(id(params_obj))


def load_experiment_params_from_yaml(path: str, localize_params: bool = False) -> TrainExperimentParams:
    """
    Convenience wrapper to load `TrainExperimentParams` from YAML.
    If `localize_params` is True, first localize all paths in the config before loading.
    """
    return load_params_from_yaml(TrainExperimentParams, path, localize_params)
