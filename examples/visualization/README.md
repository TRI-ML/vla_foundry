# Dataset Visualization

Visualize LBM robotics datasets using [Rerun](https://rerun.io/).

## Usage

```bash
./examples/visualization/visualize_data.sh <dataset_path> <num_samples>
```

### Options

- `--ordered` — Load episodes sequentially from the `episodes/` folder instead of shuffled shards.

### Examples

```bash
# Visualize 5 random samples
./examples/visualization/visualize_data.sh s3://your-bucket/your-path/vla_foundry_datasets/toolhang_202602/BimanualPlaceTtoolOnPegboard 5

# Visualize 5 consecutive episodes
./examples/visualization/visualize_data.sh --ordered s3://your-bucket/your-path/vla_foundry_datasets/toolhang_202602/BimanualPlaceTtoolOnPegboard 5
```

The script reads `preprocessing_config.yaml` from the dataset's `shards/` directory to auto-detect camera names and image indices. The Rerun viewer URL (including the gRPC connection string) is printed to the console at startup.
