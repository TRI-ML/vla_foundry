#!/usr/bin/env bash
# Convert raw Spartan-format LBM data into webdataset tar shards.
#
# Prerequisite — download the public raw Spartan data for one or more tasks
# using the included downloader (pulls from https://tri-ml-public.s3.amazonaws.com/):
#   python vla_foundry/data/scripts/download_dataset.py --task BimanualPutRedBellPepperInBin --local_path /tmp/raw_data --raw
#   python vla_foundry/data/scripts/download_dataset.py --task BimanualPlaceAppleFromBowlIntoBin --local_path /tmp/raw_data --raw
#   python vla_foundry/data/scripts/download_dataset.py --task BimanualStackPlatesOnTableFromDryingRack --local_path /tmp/raw_data --raw
#
# After extraction, each task's episodes live at /tmp/raw_data/tasks/<TaskName>/.
# Run `python vla_foundry/data/scripts/download_dataset.py --list` to see all
# available public tasks.
#
# The --source_episodes flag accepts a list of episode directories — when you
# pass multiple tasks, they are merged into a single output dataset (with
# per-task language annotations preserved via --language_annotations_path).

source .venv/bin/activate && python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
--type "spartan" \
--source_episodes "[
    '/tmp/raw_data/tasks/BimanualPutRedBellPepperInBin/',
    '/tmp/raw_data/tasks/BimanualPlaceAppleFromBowlIntoBin/',
    '/tmp/raw_data/tasks/BimanualStackPlatesOnTableFromDryingRack/',
    ]" \
--output_dir /tmp/preprocessed/lbm_multitask/ \
--camera_names "include vla_foundry/config_presets/data/lbm/lbm_data_camera_names_4cameras.yaml" \
--language_annotations_path vla_foundry/config_presets/data/lbm/lbm_language_annotations.yaml \
--action_fields_config_path vla_foundry/config_presets/data/lbm/lbm_action_fields.yaml \
--data_discard_keys "include vla_foundry/config_presets/data/lbm/lbm_data_discard_key.yaml" \
--samples_per_shard 100 \
--config_path "vla_foundry/config_presets/data/robotics_preprocessing_params_1past_14future.yaml"
