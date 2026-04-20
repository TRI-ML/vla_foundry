import pytest

from vla_foundry.data.dataloader import get_datastring_input
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml


def test_get_datastring_input_basic():
    datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = (
        get_datastring_input(
            num_samples=500_000,
            curr_shard_idx_per_dataset=[0],
            shard_shuffle_seed_per_dataset=[123],
            manifest_paths=["tests/essential/shared/dummy_manifest_large.jsonl"],
            dataset_weighting=None,
            allow_multiple_epochs=False,
            num_workers_per_gpu=4,
            world_size=8,
        )
    )
    EXPECTED_OUTPUT = (
        "tests/essential/shared/{"
        "00000056,00000043,00000134,00000106,00000186,00000053,00000016,00000069,"
        "00000077,00000151,00000018,00000091,00000195,00000065,00000090,00000031,"
        "00000141,00000189,00000002,00000081,00000054,00000160,00000198,00000168,"
        "00000156,00000087,00000139,00000023,00000041,00000131,00000101,00000172"
        "}.tar"
    )
    assert datastrings[0] == EXPECTED_OUTPUT
    assert num_samples_per_dataset == [262144]
    assert curr_shard_idx_per_dataset == [32]
    assert shard_shuffle_seed_per_dataset == [123]


def test_get_datastring_input_s3_uses_s3_urls(monkeypatch):
    manifest = [
        {"shard": "00000000", "num_sequences": 1},
        {"shard": "00000001", "num_sequences": 1},
        {"shard": "00000002", "num_sequences": 1},
    ]

    monkeypatch.setattr(
        "vla_foundry.data.dataloader.load_dataset_manifest",
        lambda path, shard_shuffle_seed=None: manifest,
    )

    datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = (
        get_datastring_input(
            num_samples=2,
            curr_shard_idx_per_dataset=[0],
            shard_shuffle_seed_per_dataset=[None],
            manifest_paths=["s3://bucket/train/manifest.jsonl"],
            dataset_weighting=None,
            allow_multiple_epochs=False,
            num_workers_per_gpu=1,
            world_size=1,
        )
    )

    assert datastrings == ["s3://bucket/train/{00000000,00000001}.tar"]
    assert num_samples_per_dataset == [2]
    assert curr_shard_idx_per_dataset == [2]
    assert shard_shuffle_seed_per_dataset == [None]


def test_get_datastring_input_smalldata_multiple_epochs():
    datastrings_1, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = (
        get_datastring_input(
            num_samples=100_000,
            curr_shard_idx_per_dataset=[0],
            shard_shuffle_seed_per_dataset=[42],
            manifest_paths=["tests/essential/shared/dummy_manifest_small.jsonl"],
            dataset_weighting=None,
            allow_multiple_epochs=True,
            num_workers_per_gpu=4,
            world_size=8,
        )
    )
    datastrings_1_arr = datastrings_1[0].split("{")[1].split("}")[0].split(",")
    for i in range(0, len(datastrings_1_arr), 6):
        if len(datastrings_1_arr[i : i + 6]) < 6:
            break
        assert sorted(datastrings_1_arr[i : i + 6]) == [
            "00000000",
            "00000001",
            "00000002",
            "00000003",
            "00000004",
            "00000005",
        ]
    assert num_samples_per_dataset == [148241]
    assert curr_shard_idx_per_dataset == [2]
    assert shard_shuffle_seed_per_dataset == [47]

    # Load the same data again
    datastrings_2, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = (
        get_datastring_input(
            num_samples=120_000,
            curr_shard_idx_per_dataset=[0],
            shard_shuffle_seed_per_dataset=[42],
            manifest_paths=["tests/essential/shared/dummy_manifest_small.jsonl"],
            dataset_weighting=None,
            allow_multiple_epochs=True,
            num_workers_per_gpu=4,
            world_size=8,
        )
    )
    assert datastrings_1 == datastrings_2
    datastrings_2_arr = datastrings_2[0].split("{")[1].split("}")[0].split(",")
    for i in range(0, len(datastrings_2_arr), 6):
        if len(datastrings_2_arr[i : i + 6]) < 6:
            break
        assert sorted(datastrings_2_arr[i : i + 6]) == [
            "00000000",
            "00000001",
            "00000002",
            "00000003",
            "00000004",
            "00000005",
        ]
    assert num_samples_per_dataset == [148241]
    assert curr_shard_idx_per_dataset == [2]
    assert shard_shuffle_seed_per_dataset == [47]

    # Restart from where it ended
    datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = (
        get_datastring_input(
            num_samples=150_000,
            curr_shard_idx_per_dataset=curr_shard_idx_per_dataset,
            shard_shuffle_seed_per_dataset=shard_shuffle_seed_per_dataset,
            manifest_paths=["tests/essential/shared/dummy_manifest_small.jsonl"],
            dataset_weighting=None,
            allow_multiple_epochs=True,
            num_workers_per_gpu=1,
            world_size=8,
        )
    )
    datastrings_arr = datastrings[0].split("{")[1].split("}")[0].split(",")
    assert sorted(datastrings_2_arr[-2:] + datastrings_arr[: curr_shard_idx_per_dataset[0]]) == [
        "00000000",
        "00000001",
        "00000002",
        "00000003",
        "00000004",
        "00000005",
    ]
    for i in range(curr_shard_idx_per_dataset[0], len(datastrings_arr), 6):
        if len(datastrings_arr[i : i + 6]) < 6:
            break
        assert sorted(datastrings_arr[i : i + 6]) == [
            "00000000",
            "00000001",
            "00000002",
            "00000003",
            "00000004",
            "00000005",
        ]
    assert num_samples_per_dataset == [148014]
    assert curr_shard_idx_per_dataset == [4]
    assert shard_shuffle_seed_per_dataset == [52]


@pytest.mark.parametrize(
    "num_samples,world_size,param_config_path",
    [
        (50_000, 4, "tests/essential/params/dummy_configs/dummy_vlm_config.yaml"),
        (50_000, 4, "tests/essential/params/dummy_configs/dummy_text_untokenized_config.yaml"),
    ],
)
def test_datastring_text_untokenized_scaling(num_samples, world_size, param_config_path):
    """Test text_untokenized dataloader with different scaling parameters."""
    params = load_experiment_params_from_yaml(param_config_path)

    datastrings, num_samples_per_dataset, _, _ = get_datastring_input(
        num_samples=num_samples,
        curr_shard_idx_per_dataset=[0],
        shard_shuffle_seed_per_dataset=[123],
        manifest_paths=params.data.dataset_manifest,
        dataset_weighting=params.data.dataset_weighting,
        allow_multiple_epochs=True,
        num_workers_per_gpu=1,
        world_size=world_size,
    )

    # Assert scaling properties
    shard_pattern = datastrings[0].split("{")[1].split("}")[0]
    shard_count = len(shard_pattern.split(","))

    # Should have at least world_size shards
    assert shard_count >= world_size
    # Should be divisible by world_size
    assert shard_count % world_size == 0
