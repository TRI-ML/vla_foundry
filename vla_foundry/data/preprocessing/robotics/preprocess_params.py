from dataclasses import dataclass, field

import draccus

from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod
from vla_foundry.params.base_params import BaseParams


def register_preprocess_params(key: str):
    """
    Registers a PreprocessParams subclass and sets its type attribute.
    Use decorator wrapper because draccus's class selection with --preprocess_params.type doesn't
    automatically populate the attribute cfg.preprocess_params.type
    """

    def decorator(cls):
        registered_cls = PreprocessParams.register_subclass(key)(cls)
        registered_cls._type = key
        return registered_cls

    return decorator


@dataclass(frozen=True)
class PreprocessParams(draccus.ChoiceRegistry, BaseParams):
    type: str = field(default=None)

    # Core I/O
    source_episodes: list[str] | None = field(default=None)
    output_dir: str | None = field(default=None)
    output_dir_fixed_path: str | None = field(default="s3://your-bucket/your-path/vla_foundry_datasets_fixed/")

    # Sampling/windowing
    past_lowdim_steps: int = field(default=1)
    future_lowdim_steps: int = field(default=14)
    camera_names: list[str] | None = field(default=None)
    # If True, skip episodes that don't have ALL requested cameras.
    # If False (default), process episodes with whatever cameras are available and warn about missing ones.
    skip_episodes_missing_cameras: bool = field(default=False)
    image_indices: list[int] = field(default_factory=lambda: [-1, 0])
    stride: int = field(default=1)
    max_padding_left: int = field(default=1)
    max_padding_right: int = field(default=7)
    padding_strategy: str = field(default="copy")  # one of: copy, zero, reflect

    # Filtering
    filter_still_samples: bool = field(default=False)
    still_threshold: float = field(default=0.01)

    # Sharding / parallelism
    samples_per_shard: int = field(default=128)
    num_workers: int = field(default=20)

    # Runtime
    max_episodes_to_process: int = field(default=-1)
    fail_on_nan: bool = field(default=False)

    # Statistics and reproducibility
    compute_statistics: bool = field(default=True)
    auto_tag: bool = field(default=True)

    # Testing flags
    skip_git_tagging: bool = field(default=False)  # Skip git operations for testing

    # Image preprocessing
    resize_images_size: list[int] | None = field(default=None)
    image_resizing_method: ImageResizingMethod = ImageResizingMethod.CENTER_CROP
    camera_rotations: dict[str, int] | None = field(default=None)
    jpeg_quality: int = field(default=95)

    # Ray configuration
    ray_address: str = field(default=None)  # Ray cluster address, default to auto-detect
    ray_num_cpus: int = field(default=None)  # Number of CPUs for Ray, default to auto-detect

    # Database logging
    db_logging: bool = field(default=False)  # Whether to log preprocessing to DynamoDB

    def __post_init__(self):
        super().__post_init__()

        if self.type is None:
            object.__setattr__(self, "type", getattr(self, "_type", None))

        # Validate required paths
        assert self.source_episodes is not None, "--source_episodes is required (or set in config_path)"
        assert self.output_dir is not None, "--output_dir is required (or set in config_path)"

        # Validate image resizing method (not a strictly necessary check due to enum)
        allowed_image_resizing_methods = set(ImageResizingMethod)
        if self.image_resizing_method not in allowed_image_resizing_methods:
            raise ValueError(
                f"image_resizing_method must be one of {[m.value for m in allowed_image_resizing_methods]}, "
                f"got {self.image_resizing_method.value}"
            )


@register_preprocess_params("spartan")
@dataclass(frozen=True)
class SpartanPreprocessParams(PreprocessParams):
    data_discard_keys: list[str] | None = field(default=None)

    language_annotations_path: str = field(
        default="vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml",
    )
    action_fields_config_path: str = field(
        default="vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml",
    )
    validation_episodes_path: str | None = field(default=None)


@register_preprocess_params("lerobot")
@dataclass(frozen=True)
class LeRobotPreprocessParams(PreprocessParams):
    observation_keys: list[str] = field(default=None)
    action_keys: list[str] = field(default=None)


TYPE_MAPPER = {
    "spartan": SpartanPreprocessParams,
    "lerobot": LeRobotPreprocessParams,
}
