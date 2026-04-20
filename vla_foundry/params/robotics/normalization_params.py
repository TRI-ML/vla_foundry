import os
from dataclasses import dataclass, field

from vla_foundry.file_utils import yaml_load
from vla_foundry.params.base_params import BaseParams


@dataclass(frozen=True)
class FieldNormalizationParams:
    """Configuration for a specific field's normalization.

    We create a dictionary of these objects in NormalizationParams which is not properly serialized by draccus.
    So we use dataclasses_json to make sure the result is still properly serializable.
    """

    method: str = field(default="std")  # "std", "percentile_5_95", "percentile_1_99" "min_max"
    scope: str = field(default="global")  # "global" or "per_timestep"
    epsilon: float = field(default=1e-8)
    enabled: bool = field(default=True)

    def to_dict(self):
        return {
            "method": self.method,
            "scope": self.scope,
            "epsilon": self.epsilon,
            "enabled": self.enabled,
        }

    def __reduce__(self):
        """Control how this object is pickled/serialized.

        This helps avoid the !!python/object tag in YAML output.
        """
        # Just return the data as a tuple of (class, args)
        # This will make it serialize as a plain mapping
        return (self.__class__, (self.method, self.scope, self.epsilon, self.enabled))

    # This is what PyYAML will use for representing the object
    def __repr__(self):
        return str(self.to_dict())


@dataclass(frozen=True)
class NormalizationParams(BaseParams):
    """
    Configuration for robotics data normalization that defines which fields to normalize and how.

    Note about per-timestep normalization:
    Some fields may be normalized "per-timestep". In such a case, the time sequences are normalized with respective
    time sequences in the statistics. However, we may want to use different windows of the time sequences for data
    loading. To handle this, we may have a different `lowdim_past_timesteps` and `lowdim_future_timesteps` for data
    loading than the ones used for normalization.
    Normalization will always use the `lowdim_past_timesteps` and `lowdim_future_timesteps` from the statistics.
    """

    enabled: bool = field(default=True)

    # Default parameters to be used for all fields if not specified in field_configs
    method: str = field(default="std")  # "std", "percentile_5_95", "percentile_1_99" "min_max"
    scope: str = field(default="global")  # "global" or "per_timestep"
    epsilon: float = field(default=1e-8)
    include_fields: list[str] = field(default_factory=list)
    centered_norm: bool = field(default=False)

    # Field-specific configurations (initialized in __post_init__)
    field_configs: dict[str, FieldNormalizationParams] = field(default_factory=dict)

    # Shared attributes. Overwritten in init_shared_attributes.
    # Low-dimensional trajectory window captured during preprocessing
    lowdim_past_timesteps: int | None = field(default=None)
    lowdim_future_timesteps: int | None = field(default=None)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "method": self.method,
            "scope": self.scope,
            "epsilon": self.epsilon,
            "include_fields": self.include_fields,
            "centered_norm": self.centered_norm,
            "field_configs": {k: v.to_dict() for k, v in self.field_configs.items()},
            "lowdim_past_timesteps": self.lowdim_past_timesteps,
            "lowdim_future_timesteps": self.lowdim_future_timesteps,
        }

    def __post_init__(self):
        self.check_asserts()
        if self.method == "std":
            # std is always centered so we force centered_norm to True
            object.__setattr__(self, "centered_norm", True)

    def check_asserts(self):
        if self.method not in ["std", "percentile_5_95", "percentile_1_99", "min_max"]:
            raise ValueError(f"Invalid normalization method: {self.method}")
        if self.scope not in ["global", "per_timestep"]:
            raise ValueError(f"Invalid normalization scope: {self.scope}")

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        include_fields = list(cfg.data.proprioception_fields + cfg.data.action_fields)
        # Currently we don't support normalization of intrinsics and extrinsics fields
        object.__setattr__(self, "include_fields", include_fields)

        field_configs = dict(self.field_configs)
        for field_name in include_fields:
            if field_name not in field_configs:
                field_configs[field_name] = FieldNormalizationParams(
                    method=self.method,
                    scope=self.scope,
                    epsilon=self.epsilon,
                    enabled=self.enabled,
                )

        object.__setattr__(self, "field_configs", field_configs)

        dataset_statistics = cfg.data.dataset_statistics
        requested_past_from_data_params = cfg.data.lowdim_past_timesteps
        requested_future_from_data_params = cfg.data.lowdim_future_timesteps
        requested_past = (
            max(self.lowdim_past_timesteps, requested_past_from_data_params)
            if self.lowdim_past_timesteps is not None and requested_past_from_data_params is not None
            else requested_past_from_data_params or self.lowdim_past_timesteps
        )
        requested_future = (
            max(self.lowdim_future_timesteps, requested_future_from_data_params)
            if self.lowdim_future_timesteps is not None and requested_future_from_data_params is not None
            else requested_future_from_data_params or self.lowdim_future_timesteps
        )

        if not dataset_statistics:
            raise ValueError("Robotics normalization requires dataset_statistics.")

        statistics_paths = [dataset_statistics] if isinstance(dataset_statistics, str) else list(dataset_statistics)
        # Published checkpoints have sanitized (None) stats paths; skip the lowdim
        # availability check in that case since there's no preprocessing_config.yaml to read.
        statistics_paths = [p for p in statistics_paths if p]
        if not statistics_paths:
            return

        past_lowdim_candidates = set()
        future_lowdim_candidates = set()
        for statistics_path in statistics_paths:
            path = os.path.dirname(statistics_path)
            processing_config = yaml_load(os.path.join(path, "preprocessing_config.yaml"))
            # Handle indexed format from collect_preprocessing_configs (e.g. {0: {...}, 1: {...}})
            if processing_config and all(isinstance(k, int) for k in processing_config):
                processing_config = processing_config[0]
            past_lowdim_candidates.add(processing_config["past_lowdim_steps"])
            future_lowdim_candidates.add(processing_config["future_lowdim_steps"])

        available_past = min(past_lowdim_candidates)
        available_future = min(future_lowdim_candidates)

        if requested_past is not None and requested_past > available_past:
            raise ValueError(
                f"Requested lowdim_past_timesteps {requested_past} exceeds available past timesteps "
                f"{available_past} from at least one data source."
            )
        if requested_future is not None and requested_future > available_future:
            raise ValueError(
                f"Requested lowdim_future_timesteps {requested_future} exceeds available future timesteps "
                f"{available_future} from at least one data source."
            )

        # We set the lowdim_past_timesteps and lowdim_future_timesteps to the available past and future timesteps
        # Or the ones present in the data params if provided. Not to the requested values.
        # Requested sequence lengths are used for data loading, not normalization.
        object.__setattr__(self, "lowdim_past_timesteps", self.lowdim_past_timesteps or available_past)
        object.__setattr__(self, "lowdim_future_timesteps", self.lowdim_future_timesteps or available_future)
