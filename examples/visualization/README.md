# Dataset Visualization

Visualize LBM robotics datasets using [Rerun](https://rerun.io/).

## Usage

```bash
./examples/visualization/visualize_data.sh [flags] <dataset_path>
```

### Flags

- `--num_episodes=N` — Number of episodes to visualize (default: 5).
- `--subsample=N` — Visualize every Nth sample (default: 1).
- `--ordered` — Load episodes sequentially from the `episodes/` folder instead of shuffled shards.
- `--print-command` — Print the equivalent Python command (for Colab/Jupyter) instead of executing.

### Examples

```bash
# Visualize 5 random samples from an S3 dataset
./examples/visualization/visualize_data.sh --num_episodes=5 s3://your-bucket/your-path/vla_foundry_datasets/BimanualPutRedBellPepperInBin

# Visualize 5 consecutive episodes from a local dataset
./examples/visualization/visualize_data.sh --ordered --num_episodes=5 /data/datasets/BimanualPutRedBellPepperInBin
```

The script reads `preprocessing_config.yaml` from the dataset's `shards/` directory to auto-detect camera names and image indices. The Rerun viewer URL (including the gRPC connection string) is printed to the console at startup.
