#!/usr/bin/env bash
# Convert the publicly-available pi_libero LeRobot dataset into tar shards,
# reading and writing directly from/to S3.
#
# Prerequisite — stage the HF dataset in your own S3 bucket once:
#   hf download physical-intelligence/pi_libero --repo-type dataset \
#       --local-dir /tmp/pi_libero
#   aws s3 sync /tmp/pi_libero s3://your-bucket/your-path/hf_datasets/pi_libero/
#
# To run fully local instead (no S3), pass local paths for --source_episodes
# and --output_dir (see tutorials/lerobot.ipynb for that workflow).

source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "lerobot" \
--source_episodes "['s3://your-bucket/your-path/hf_datasets/pi_libero/']" \
--output_dir s3://your-bucket/your-path/vla_foundry_datasets/lerobot/pi_libero/ \
--camera_names "['image', 'wrist_image']" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml" \
--observation_keys "['state']" \
--action_keys "['actions']"
