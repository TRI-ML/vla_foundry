"""
Normalization utilities for robotics data.
"""

import json
import logging
import os
from typing import Any

import draccus
import torch

from vla_foundry.data.robotics.utils import crop_sequence, merge_statistics
from vla_foundry.file_utils import json_load
from vla_foundry.params.robotics.normalization_params import FieldNormalizationParams, NormalizationParams


class RoboticsNormalizer:
    """
    Normalizer for robotics data with configurable strategies.

    Supports:
    - Global normalization: normalize across all timesteps
    - Per-timestep normalization: normalize each timestep separately
    - Std-based normalization: use mean/std
    - Quantile-based normalization: use percentiles (e.g., 5th/95th)
    """

    def __init__(
        self,
        normalization_params: dict[str, Any] | NormalizationParams,
        statistics_data: dict[str, Any] | None = None,
        statistics_path: str | list[str] | None = None,
    ):
        """
        Initialize normalizer.

        Args:
            normalization_params: NormalizationParams instance with field definitions and normalization settings
            statistics_data: Pre-loaded statistics dict
            statistics_path: Path to statistics JSON file
        """
        self.normalization_params = normalization_params

        self.enabled = self.normalization_params.enabled
        self.lowdim_past_timesteps = self.normalization_params.lowdim_past_timesteps
        self.lowdim_future_timesteps = self.normalization_params.lowdim_future_timesteps

        # Always load statistics when available, regardless of whether normalization is enabled
        # This allows action dimension computation even when normalization is disabled
        if statistics_data is not None:
            self.stats = statistics_data
        elif statistics_path is not None:
            self.stats = self._load_statistics(statistics_path)
        else:
            logging.warning("No statistics provided - normalization will be disabled")
            self.enabled = False
            self.stats = None
            return

        self._norm_param_cache = {}
        if isinstance(self.stats, list):
            if len(self.stats) > 1:
                self.stats = merge_statistics(self.stats)
            else:
                self.stats = self.stats[0]

        # Parse configuration from dataclass
        self.method = self.normalization_params.method
        self.scope = self.normalization_params.scope
        self.epsilon = self.normalization_params.epsilon
        self.field_configs = self.normalization_params.field_configs
        self.include_fields = self.normalization_params.include_fields
        self.centered_norm = self.normalization_params.centered_norm

        logging.info(f"RoboticsNormalizer initialized: method={self.method}, scope={self.scope}")

    def save(self, experiment_path: str):
        with open(os.path.join(experiment_path, "config_normalizer.yaml"), "w") as f:
            draccus.dump(self.normalization_params, f)
        with open(os.path.join(experiment_path, "stats.json"), "w") as f:
            json.dump(self.stats, f)

    @classmethod
    def load(cls, config_path: str, statistics_path: str):
        return cls(NormalizationParams.from_file(config_path), statistics_path=statistics_path)

    @classmethod
    def from_pretrained(cls, config_path: str):
        return cls(
            NormalizationParams.from_file(os.path.join(config_path, "config_normalizer.yaml")),
            statistics_path=os.path.join(config_path, "stats.json"),
        )

    def get_field_dimension(self, field_name: str) -> int:
        """Get the dimension of a field."""
        if field_name in self.stats:
            return len(self.stats[field_name]["mean"])
        else:
            raise ValueError(f"Field {field_name} not found in dataset statistics")

    def _load_statistics(self, statistics_path: str) -> dict[str, Any]:
        """Load statistics from JSON file."""
        if isinstance(statistics_path, str):
            stats = json_load(statistics_path)
        elif isinstance(statistics_path, list):
            stats = []
            for path in statistics_path:
                stats.append(json_load(path))
        else:
            raise ValueError(f"Invalid statistics path: {statistics_path}")
        logging.info(f"Loaded statistics from {statistics_path}")
        return stats

    def _get_field_config(self, field_name: str) -> FieldNormalizationParams:
        """Get configuration for a specific field."""
        # Check for exact match first
        if field_name in self.field_configs:
            return self.field_configs[field_name]

        # Check for pattern matches (mainly applies the same config to relative fields as the main field)
        for pattern, config in self.field_configs.items():
            if pattern in field_name:
                return config

        # Return default config
        return FieldNormalizationParams(
            method=self.method, scope=self.scope, epsilon=self.epsilon, enabled=self.enabled
        )

    def _should_normalize_field(self, field_name: str) -> bool:
        """Check if a field should be normalized."""
        if not self.enabled or not self._get_field_config(field_name).enabled:
            return False

        # Only normalize fields that are in the include_fields
        if field_name not in self.include_fields:
            return False

        # Skip text and mask fields
        if any(keyword in field_name.lower() for keyword in ["text", "language", "instruction", "mask", "valid"]):
            return False

        # Skip if no statistics available
        if field_name not in self.stats:
            logging.warning(f"No statistics available for field: {field_name}")
            return False

        return True

    def _get_normalization_params(self, field_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get normalization parameters (center, scale) for a field.
        Use a cache for efficiency.

        Args:
            field_name: Name of the field

        Returns:
            Tuple of (center, scale) tensors
        """
        if field_name in self._norm_param_cache:
            return self._norm_param_cache[field_name]
        center, scale = self._compute_normalization_params(field_name)
        self._norm_param_cache[field_name] = (center, scale)
        return center, scale

    def _compute_normalization_params(self, field_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        field_stats = self.stats[field_name]
        field_config = self._get_field_config(field_name)

        method = field_config.method
        scope = field_config.scope
        epsilon = field_config.epsilon

        if not self.enabled or not field_config.enabled:
            return torch.zeros(1), torch.ones(1)

        if scope == "global":
            if method == "std":
                center = torch.tensor(field_stats["mean"], dtype=torch.float32)
                scale = torch.tensor(field_stats["std"], dtype=torch.float32)
            elif method == "percentile_5_95":
                center = torch.tensor(field_stats["percentile_5"], dtype=torch.float32)
                scale = torch.tensor(field_stats["percentile_95"], dtype=torch.float32) - torch.tensor(
                    field_stats["percentile_5"], dtype=torch.float32
                )
            elif method == "percentile_1_99":
                center = torch.tensor(field_stats["percentile_1"], dtype=torch.float32)
                scale = torch.tensor(field_stats["percentile_99"], dtype=torch.float32) - torch.tensor(
                    field_stats["percentile_1"], dtype=torch.float32
                )
            elif method == "min_max":
                center = torch.tensor(field_stats["min"], dtype=torch.float32)
                scale = torch.tensor(field_stats["max"], dtype=torch.float32) - torch.tensor(
                    field_stats["min"], dtype=torch.float32
                )
            else:
                raise ValueError(f"Invalid normalization method: {method}")
        else:
            if method == "std":
                center = torch.tensor(field_stats["mean_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(field_stats["std_per_timestep"], dtype=torch.float32)
            elif method == "percentile_5_95":
                center = torch.tensor(field_stats["percentile_5_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(
                    field_stats["percentile_95_per_timestep"],
                    dtype=torch.float32,
                ) - torch.tensor(
                    field_stats["percentile_5_per_timestep"],
                    dtype=torch.float32,
                )
            elif method == "percentile_1_99":
                center = torch.tensor(field_stats["percentile_1_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(
                    field_stats["percentile_99_per_timestep"],
                    dtype=torch.float32,
                ) - torch.tensor(
                    field_stats["percentile_1_per_timestep"],
                    dtype=torch.float32,
                )
            elif method == "min_max":
                center = torch.tensor(field_stats["min_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(field_stats["max_per_timestep"], dtype=torch.float32) - torch.tensor(
                    field_stats["min_per_timestep"], dtype=torch.float32
                )
            else:
                raise ValueError(f"Invalid normalization method: {method}")

        if self.centered_norm and ("percentile" in method or "min_max" in method):
            center = center + 0.5 * scale
            scale = scale * 0.5

        # Avoid division by zero
        scale = torch.clamp(scale, min=epsilon)
        return center, scale

    def normalize_tensor(
        self, tensor: torch.Tensor, field_name: str, anchor_timestep: int | None = None
    ) -> torch.Tensor:
        """
        Normalize a tensor.

        Args:
            tensor: Input tensor of shape [batch_size, timesteps, features] or [batch_size, features]
            field_name: Name of the field being normalized
            anchor_timestep: The index of the anchor timestep in the input tensor
            (only usedfor per-timestep normalization with cropped sequences)

        Returns:
            Normalized tensor
        """
        if not self._should_normalize_field(field_name):
            return tensor

        field_config = self._get_field_config(field_name)
        scope = field_config.scope

        center, scale = self._get_normalization_params(field_name)
        center = center.to(tensor.device)
        scale = scale.to(tensor.device)

        if scope == "global" or len(tensor.shape) == 2:
            # Global normalization or no time dimension
            # Broadcast to match tensor dimensions - add singleton dims for all but last
            target_shape = [1] * (len(tensor.shape) - 1) + [-1]
            center = center.view(target_shape)
            scale = scale.view(target_shape)
            normalized = (tensor - center) / scale

        elif scope == "per_timestep" and len(tensor.shape) == 3:
            # Per-timestep normalization
            # If anchor_timestep is provided, we need to align the tensor with the statistics
            # The statistics were computed with lowdim_past_timesteps past steps
            # The tensor has anchor_timestep as the index of the current timestep
            _batch_size, num_timesteps, _feature_dim = tensor.shape

            if anchor_timestep is not None and num_timesteps != center.shape[0]:
                if self.lowdim_past_timesteps is None:
                    raise ValueError(
                        "The normalizer is asked to normalize a tensor with a different number"
                        f"of timesteps {num_timesteps} than the statistics ({center.shape[0]})"
                        "but lowdim_past_timesteps must be set to align the statistics with the tensor."
                        "This is likely because the data preprocessing metadata is not available."
                    )
                # The statistics were computed with lowdim_past_timesteps past steps
                # We need to crop them to align with the tensor's anchor_timestep
                # This maps: tensor[anchor_timestep] -> stats[lowdim_past_timesteps]

                # Calculate how many past and future timesteps the tensor has relative to its anchor
                tensor_past = anchor_timestep
                tensor_future = num_timesteps - anchor_timestep - 1

                # Crop statistics around stats_anchor_idx to match tensor's time range
                stats_anchor_idx = self.lowdim_past_timesteps

                # Check if crop would be valid
                start_idx = stats_anchor_idx - tensor_past
                end_idx = stats_anchor_idx + tensor_future + 1

                if start_idx >= 0 and end_idx <= len(center):
                    # Normal case: statistics fully cover the tensor's time range
                    cropped_center = crop_sequence(center, stats_anchor_idx, tensor_past, tensor_future)
                    cropped_scale = crop_sequence(scale, stats_anchor_idx, tensor_past, tensor_future)
                else:
                    raise ValueError("The statistics do not cover the requested tensor's time range")

                # Add batch dimension and broadcast
                cropped_center = cropped_center.unsqueeze(0)  # [1, T', D]
                cropped_scale = cropped_scale.unsqueeze(0)  # [1, T', D]

                normalized = (tensor - cropped_center) / cropped_scale
            else:
                # No anchor provided, assume the tensor is already aligned
                center = center.unsqueeze(0)  # [1, T, D]
                scale = scale.unsqueeze(0)  # [1, T, D]
                normalized = (tensor - center) / scale
        else:
            # Unsupported tensor shape
            logging.warning(f"Unsupported tensor shape for normalization: {tensor.shape}")
            normalized = tensor

        return normalized

    def denormalize_tensor(
        self, normalized_tensor: torch.Tensor, field_name: str, anchor_timestep: int = None
    ) -> torch.Tensor:
        """
        Denormalize a tensor (inverse of normalize_tensor).

        Args:
            normalized_tensor: Normalized tensor
            field_name: Name of the field being denormalized
            anchor_timestep: The index of the anchor timestep in the input tensor
                (only used for per-timestep denormalization with cropped sequences)

        Returns:
            Denormalized tensor
        """
        if not self._should_normalize_field(field_name):
            return normalized_tensor

        field_config = self._get_field_config(field_name)
        scope = field_config.scope

        center, scale = self._get_normalization_params(field_name)
        center = center.to(normalized_tensor.device)
        scale = scale.to(normalized_tensor.device)

        if scope == "global" or len(normalized_tensor.shape) == 2:
            # Global denormalization or no time dimension
            # Broadcast to match tensor dimensions - add singleton dims for all but last
            target_shape = [1] * (len(normalized_tensor.shape) - 1) + [-1]
            center = center.view(target_shape)
            scale = scale.view(target_shape)
            denormalized = normalized_tensor * scale + center

        elif scope == "per_timestep" and len(normalized_tensor.shape) == 3:
            # Per-timestep denormalization
            # If anchor_timestep is provided, we need to align the tensor with the statistics
            # The statistics were computed with lowdim_past_timesteps past steps
            # The tensor has anchor_timestep as the index of the current timestep
            _batch_size, num_timesteps, _feature_dim = normalized_tensor.shape

            if anchor_timestep is not None and num_timesteps != center.shape[0]:
                if self.lowdim_past_timesteps is None:
                    raise ValueError(
                        "The normalizer is asked to denormalize a tensor with a different number"
                        f"of timesteps {num_timesteps} than the statistics ({center.shape[0]})"
                        "but lowdim_past_timesteps must be set to align the statistics with the tensor."
                        "This is likely because the data preprocessing metadata is not available."
                    )
                # The statistics were computed with lowdim_past_timesteps past steps
                # We need to crop them to align with the tensor's anchor_timestep
                # This maps: tensor[anchor_timestep] -> stats[lowdim_past_timesteps]

                # Calculate how many past and future timesteps the tensor has relative to its anchor
                tensor_past = anchor_timestep
                tensor_future = num_timesteps - anchor_timestep - 1

                # Crop statistics around stats_anchor_idx to match tensor's time range
                stats_anchor_idx = self.lowdim_past_timesteps

                # Check if crop would be valid
                start_idx = stats_anchor_idx - tensor_past
                end_idx = stats_anchor_idx + tensor_future + 1

                if start_idx >= 0 and end_idx <= len(center):
                    # Normal case: statistics fully cover the tensor's time range
                    cropped_center = crop_sequence(center, stats_anchor_idx, tensor_past, tensor_future)
                    cropped_scale = crop_sequence(scale, stats_anchor_idx, tensor_past, tensor_future)
                else:
                    # Edge case: need to clamp
                    start_idx = max(0, start_idx)
                    end_idx = min(len(center), end_idx)
                    cropped_center = center[start_idx:end_idx]  # [T', D]
                    cropped_scale = scale[start_idx:end_idx]  # [T', D]

                # Add batch dimension and broadcast
                cropped_center = cropped_center.unsqueeze(0)  # [1, T', D]
                cropped_scale = cropped_scale.unsqueeze(0)  # [1, T', D]

                denormalized = normalized_tensor * cropped_scale + cropped_center
            else:
                # No anchor provided, assume the tensor is already aligned
                center = center.unsqueeze(0)  # [1, T, D]
                scale = scale.unsqueeze(0)  # [1, T, D]
                denormalized = normalized_tensor * scale + center
        else:
            # Unsupported tensor shape
            logging.warning(f"Unsupported tensor shape for denormalization: {normalized_tensor.shape}")
            denormalized = normalized_tensor

        return denormalized
