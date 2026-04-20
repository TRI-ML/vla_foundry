"""Origin tracking for configs loaded via load_params_from_yaml.

The vlm_foundry_backbone constructor needs to resolve ``vlm_config_model.yaml``
from the published repo when ``resume_from_checkpoint`` is scrubbed (true of
all published HF checkpoints). Without origin tracking, instantiating the
backbone outside of ``BaseModel.from_pretrained`` raises ValueError.
"""

import os
import tempfile

import yaml

from vla_foundry.params.model_params import ModelParams
from vla_foundry.params.train_experiment_params import get_config_origin, load_params_from_yaml


def test_config_origin_recorded_for_local_path():
    params = load_params_from_yaml(
        ModelParams, "tests/essential/params/dummy_configs/dummy_vla_diffusion_policy_config.yaml"
    )
    origin = get_config_origin(params)
    assert origin == "tests/essential/params/dummy_configs"


def test_config_origin_survives_round_trip_via_kwargs():
    """Make sure origin tracking doesn't leak into __dict__ (would break **kwargs re-instantiation)."""
    params = load_params_from_yaml(
        ModelParams, "tests/essential/params/dummy_configs/dummy_vla_diffusion_policy_config.yaml"
    )
    # Constructors reject unknown kwargs; if _config_origin leaked into __dict__
    # this would raise TypeError.
    type(params)(**params.__dict__)


def test_config_origin_propagates_to_vlm_foundry_backbone():
    with tempfile.TemporaryDirectory() as tmp:
        vla_cfg = {
            "type": "diffusion_policy",
            "vision_language_backbone": {
                "type": "vlm_foundry_backbone",
                "num_vlm_layers_to_use": 4,
            },
            "transformer": {
                "type": "transformer",
                "vocab_size": 1000,
                "hidden_dim": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_seq_len": 128,
            },
            "noise_scheduler": {"num_timesteps": 100, "beta_start": 0.0001, "beta_end": 0.02},
            "action_dim": 7,
        }
        cfg_path = os.path.join(tmp, "config_model.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(vla_cfg, f)

        params = load_params_from_yaml(ModelParams, cfg_path)

        # Outer params and the vlm_foundry_backbone params both get tagged.
        # Production code matches on the type-string field (draccus union
        # dispatch can pick the wrong dataclass variant — see the gotchas in
        # integration-test.md).
        assert get_config_origin(params) == tmp
        assert params.vision_language_backbone.type == "vlm_foundry_backbone"
        assert get_config_origin(params.vision_language_backbone) == tmp
