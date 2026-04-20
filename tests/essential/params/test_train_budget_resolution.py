from unittest.mock import patch

import pytest

from vla_foundry.params.train_experiment_params import TrainExperimentParams


def make_train_experiment_params_config(**overrides):
    config = {
        "data": {
            "type": "text",
            "dataset_manifest": ["tests/essential/shared/dummy_manifest_small.jsonl"],
            "dataset_modality": ["text"],
            "dataset_weighting": [1.0],
            "allow_multiple_epochs": False,
        },
        "model": {
            "type": "transformer",
        },
        "distributed": {
            "world_size": 1,
        },
        "hparams": {
            "global_batch_size": 1,
            "per_gpu_batch_size": 1,
            "seed": 42,
        },
    }
    config.update(overrides)
    return config


def test_uses_total_train_samples_when_provided():
    cfg = TrainExperimentParams.from_dict(make_train_experiment_params_config(total_train_samples=123))
    assert cfg.total_train_samples == 123


def test_derives_total_train_samples_from_num_epochs():
    config = make_train_experiment_params_config(num_epochs=3)
    config["data"]["allow_multiple_epochs"] = True

    with patch("vla_foundry.params.train_experiment_params.epochs_to_samples", return_value=456) as mock_fn:
        cfg = TrainExperimentParams.from_dict(config)

    assert cfg.total_train_samples == 456
    mock_fn.assert_called_once_with(config["data"]["dataset_manifest"], 3)


def test_derives_total_train_samples_from_multiple_manifests():
    num_epochs = 2
    config = make_train_experiment_params_config(num_epochs=num_epochs)
    config["data"]["dataset_manifest"] = [
        "tests/essential/shared/dummy_manifest_small.jsonl",
        "tests/essential/shared/dummy_manifest_small.jsonl",
    ]
    config["data"]["dataset_modality"] = ["text", "text"]
    config["data"]["dataset_weighting"] = [1.0, 1.0]
    config["data"]["allow_multiple_epochs"] = True

    cfg = TrainExperimentParams.from_dict(config)

    expected_sequences_per_manifest = 27781
    expected_total_sequences = expected_sequences_per_manifest + expected_sequences_per_manifest
    assert cfg.total_train_samples == num_epochs * expected_total_sequences


def test_accepts_explicit_total_train_samples_when_consistent_with_num_epochs():
    config = make_train_experiment_params_config(total_train_samples=123, num_epochs=3)
    config["data"]["allow_multiple_epochs"] = True

    with patch("vla_foundry.params.train_experiment_params.epochs_to_samples", return_value=123) as mock_fn:
        cfg = TrainExperimentParams.from_dict(config)

    assert cfg.total_train_samples == 123
    mock_fn.assert_called_once_with(config["data"]["dataset_manifest"], 3)


def test_rejects_inconsistent_total_train_samples_and_num_epochs():
    config = make_train_experiment_params_config(total_train_samples=123, num_epochs=3)
    config["data"]["allow_multiple_epochs"] = True

    with (
        patch("vla_foundry.params.train_experiment_params.epochs_to_samples", return_value=456),
        pytest.raises(AssertionError, match="resolve to different training budgets"),
    ):
        TrainExperimentParams.from_dict(config)


def test_requires_training_budget_when_neither_value_is_set():
    with pytest.raises(AssertionError):
        TrainExperimentParams.from_dict(make_train_experiment_params_config())
