#!/bin/bash
# Generic LBM dataset visualizer that works with any S3 or local robotics dataset
#
# Usage:
#   # S3 paths:
#   ./visualize_data.sh --num_episodes=5 s3://your-bucket/your-path/vla_foundry_datasets/BimanualPutRedBellPepperInBin
#   ./visualize_data.sh --ordered --num_episodes=10 s3://...
#
#   # Local paths:
#   ./visualize_data.sh --num_episodes=5 /data/datasets/BimanualPutRedBellPepperInBin
#   ./visualize_data.sh --ordered --num_episodes=10 /data/datasets/BimanualPutRedBellPepperInBin
#
# Flags:
#   --num_episodes=N   Number of episodes (trajectories) to visualize (default: 5)
#   --subsample=N      Visualize every Nth sample (default: 1, all samples)
#   --ordered          Use ordered episode data instead of shuffled shards
#   --print-command    Print Python command for Colab/Jupyter instead of executing

set -e

# Check for --print-command flag
PRINT_COMMAND=false
if [[ "$*" == *"--print-command"* ]]; then
    PRINT_COMMAND=true
fi

# Parse --num_episodes flag (default: 5)
NUM_EPISODES=5
NUM_EPISODES_ARGS=$(echo "$@" | grep -o -- '--num_episodes=[0-9]*' || true)
if [ -n "$NUM_EPISODES_ARGS" ]; then
    NUM_EPISODES=$(echo "$NUM_EPISODES_ARGS" | cut -d'=' -f2)
fi

# Separate flags (forwarded to Python) from positional args
FLAGS=()
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --print-command) continue ;;  # Skip this flag
        --num_episodes=*) continue ;;  # Skip, handled above
        --*) FLAGS+=("$arg") ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

if [ ${#POSITIONAL[@]} -lt 1 ]; then
    echo "Usage: $0 [--num_episodes=N] [--subsample=N] [--ordered] [--print-command] <dataset_path>"
    echo ""
    echo "Flags:"
    echo "  --num_episodes=N   Number of episodes to visualize (default: 5)"
    echo "  --subsample=N      Visualize every Nth sample (default: 1)"
    echo "  --ordered          Use ordered episode data instead of shuffled shards"
    echo "  --print-command    Print Python command for Colab instead of executing"
    echo ""
    echo "Dataset path can be either S3 or local:"
    echo "  - S3:    s3://bucket/path/to/dataset"
    echo "  - Local: /path/to/dataset"
    echo ""
    echo "Examples:"
    echo "  ./visualize_data.sh --num_episodes=5 s3://your-bucket/your-path/dataset/task"
    echo "  ./visualize_data.sh --ordered /data/datasets/BimanualPutRedBellPepperInBin"
    exit 1
fi

DATASET_PATH="${POSITIONAL[0]%/}"

# Detect if this is an S3 path or local path
if [[ "$DATASET_PATH" == s3://* ]]; then
    IS_S3=true
else
    IS_S3=false
fi

# Determine if shards directory exists in dataset path
if [[ "$DATASET_PATH" == *"/shards" ]]; then
    CONFIG_DIR="$DATASET_PATH"
else
    CONFIG_DIR="$DATASET_PATH/shards"
fi

# Load preprocessing config dynamically
CONFIG_FILE="$CONFIG_DIR/preprocessing_config.yaml"

# Parse YAML to extract config
if [ "$IS_S3" = true ]; then
    # Use aws s3 cp to download YAML temporarily
    TEMP_CONFIG=$(mktemp)
    aws s3 cp "$CONFIG_FILE" "$TEMP_CONFIG" > /dev/null
    CONFIG_FILE="$TEMP_CONFIG"
    CLEANUP=true
else
    # For local paths, verify the file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    CLEANUP=false
fi

# Extract values from YAML using grep/awk
extract_yaml_list() {
    local key=$1
    local file=$2
    # Use sed range to capture only list items between this key and the next YAML key
    sed -n "/^${key}:/,/^[^ -]/{/^- /p}" "$file" | sed 's/^- //' | sed 's/ *$//' | paste -sd ',' -
}

extract_yaml_single() {
    local key=$1
    local file=$2
    grep "^$key:" "$file" | cut -d':' -f2- | sed 's/^ *//' | sed 's/ *$//'
}

CAMERAS=$(extract_yaml_list "camera_names" "$CONFIG_FILE")
IMAGE_INDICES=$(extract_yaml_list "image_indices" "$CONFIG_FILE")

# Convert to proper format for draccus
IFS=',' read -ra CAMERA_ARRAY <<< "$CAMERAS"
CAMERA_STR=$(printf '%s, ' "${CAMERA_ARRAY[@]}" | sed 's/, $//')

IFS=',' read -ra INDICES_ARRAY <<< "$IMAGE_INDICES"
INDICES_STR=$(printf '%s, ' "${INDICES_ARRAY[@]}" | sed 's/, $//')

# Generate image names from cameras and indices
IMAGE_NAMES=()
for camera in "${CAMERA_ARRAY[@]}"; do
    for idx in "${INDICES_ARRAY[@]}"; do
        if [ "$idx" == "-1" ]; then
            IMAGE_NAMES+=("${camera}_t-1")
        else
            IMAGE_NAMES+=("${camera}_t${idx}")
        fi
    done
done

CONFIG_PATH="examples/visualization/visualization_params.yaml"

# Generate image_names list string for draccus
IMAGE_NAMES_STR=$(printf '%s, ' "${IMAGE_NAMES[@]}" | sed 's/, $//')

# Determine which manifest to use for calculating timesteps
if [[ " ${FLAGS[@]} " =~ " --ordered " ]]; then
    # For ordered visualization, use episodes manifest
    MANIFEST_FOR_COUNTING="$DATASET_PATH/episodes/manifest.jsonl"
else
    # For shuffled, use shards manifest (but episodes is better if available)
    MANIFEST_FOR_COUNTING="$DATASET_PATH/shards/manifest.jsonl"
fi

# Always use shards paths for the actual data loading; --ordered flag rewrites to episodes/ in Python
STATS_PATH="$DATASET_PATH/shards/stats.json"
MANIFEST_PATH="$DATASET_PATH/shards/manifest.jsonl"

# Calculate exact number of timesteps in first NUM_EPISODES episodes/shards
# This ensures we get all timesteps from exactly N episodes, not an approximation
if [ "$IS_S3" = true ]; then
    # Download manifest temporarily
    TEMP_MANIFEST=$(mktemp)
    aws s3 cp "$MANIFEST_FOR_COUNTING" "$TEMP_MANIFEST" > /dev/null 2>&1
    MANIFEST_FILE="$TEMP_MANIFEST"
    CLEANUP_MANIFEST=true
else
    # For local paths, verify the file exists
    if [ ! -f "$MANIFEST_FOR_COUNTING" ]; then
        echo "Error: Manifest file not found: $MANIFEST_FOR_COUNTING"
        exit 1
    fi
    MANIFEST_FILE="$MANIFEST_FOR_COUNTING"
    CLEANUP_MANIFEST=false
fi

# Sum up num_sequences from first NUM_EPISODES entries
NUM_SAMPLES=$(head -n "$NUM_EPISODES" "$MANIFEST_FILE" | jq -s 'map(.num_sequences) | add')

if [ "$CLEANUP_MANIFEST" = true ]; then
    rm "$TEMP_MANIFEST"
fi

# echo "Visualizing $NUM_EPISODES episodes ($NUM_SAMPLES total timesteps)"
# echo "Dataset type: $([ "$IS_S3" = true ] && echo "S3" || echo "Local")"

# Build arguments dynamically from config
ARGS=(
    "--config_path=$CONFIG_PATH"
    "--data.dataset_manifest=[$MANIFEST_PATH]"
    "--data.dataset_statistics=[$STATS_PATH]"
    "--data.camera_names=[$CAMERA_STR]"
    "--data.image_indices=[$INDICES_STR]"
    "--data.image_names=[$IMAGE_NAMES_STR]"
    "--data.shuffle=true"
    "--total_train_samples=$NUM_SAMPLES"
    "--num_checkpoints=1"
    "--data.num_workers=0"
    "--data.normalization.enabled=false"
    "--data.normalization.field_configs={}"
)

# Cleanup temp file if created
if [ "$CLEANUP" = true ]; then
    rm "$TEMP_CONFIG"
fi

# Either print the command or execute it
if [ "$PRINT_COMMAND" = true ]; then
    echo "# For Colab, copy and paste this into a cell:"
    echo "# (Make sure VISUALIZER='rerun' is set first)"
    echo ""
    echo "import sys"
    echo "sys.argv = ["
    echo "    'lbm_vis.py',"
    for arg in "${FLAGS[@]}" "${ARGS[@]}"; do
        # Escape single quotes in the argument
        escaped_arg="${arg//\'/\\\'}"
        echo "    '${escaped_arg}',"
    done
    echo "]"
    echo ""
    echo "from vla_foundry.data.scripts.vis.lbm_vis import main"
    echo "main()"
else
    # Call lbm_vis.py (FLAGS like --ordered are forwarded to Python)
    VISUALIZER="${VISUALIZER:-rerun}" uv run --group visualization vla_foundry/data/scripts/vis/lbm_vis.py "${FLAGS[@]}" "${ARGS[@]}"
fi
