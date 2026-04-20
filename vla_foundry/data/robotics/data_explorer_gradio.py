#!/usr/bin/env python3
"""
Gradio-based Robotics Data Explorer

This tool provides a gradio web interface for exploring preprocessed robotics data
with interactive trajectory visualization overlaid on camera images.

Usage:
    python vla_foundry/data/robotics/data_explorer_gradio.py \
        --dataset-path /path/to/processed/dataset/ \
        --max-samples 100 \
        --port 7860

Features:
    - Modern web-based interface using Gradio
    - Interactive sliders and dropdowns
    - Real-time trajectory overlay on camera images
    - 3D trajectory visualization
    - Multiple camera support
    - Gripper state visualization
    - Sample metadata display
"""

import argparse
import os

import fsspec
import torch
import yaml

from vla_foundry.data.robotics.gradio_dataloader import RoboticsDataLoader
from vla_foundry.data.robotics.gradio_interface import GradioDataExplorer
from vla_foundry.file_utils import load_model_checkpoint
from vla_foundry.models import create_model
from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml


def main():
    parser = argparse.ArgumentParser(description="Gradio-based Robotics Data Explorer")

    parser.add_argument("--dataset-path", required=True, help="Path to processed robotics dataset")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum number of samples to load (-1 for all)")
    parser.add_argument("--max-shards", type=int, default=-1, help="Maximum number of shards to load (-1 for all)")
    parser.add_argument("--trajectory-length", type=int, default=50, help="Number of trajectory points to display")
    parser.add_argument(
        "--port", type=int, default=None, help="Port to run Gradio interface on (auto-select if not specified)"
    )
    parser.add_argument("--share", action="store_true", help="Create shareable link")
    parser.add_argument(
        "--use-dataloader",
        action="store_true",
        default=True,
        help="Use dataloader pipeline for loading (default: True, use --no-use-dataloader for direct file loading)",
    )
    parser.add_argument(
        "--load-from-files",
        default=False,
        action="store_true",
        help="Load samples directly from files instead of using dataloader pipeline",
    )
    parser.add_argument(
        "--model-predictions-path",
        default=None,
        help="Path to model predictions",
    )

    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=100,
        help="Number of inference steps to run for model predictions",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model predictions on",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type to use for model predictions",
    )

    args = parser.parse_args()

    if args.dataset_path.endswith("/"):
        args.dataset_path = args.dataset_path[:-1]

    print("🔍 Loading robotics data...")

    # Determine loading method
    use_dataloader = not args.load_from_files

    # Create data loader and load samples default
    if args.model_predictions_path is None:
        # Load data using direct file access
        config_path = "vla_foundry/config_presets/data/lbm/lbm_data_params.yaml"
        with fsspec.open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Override fields for the data explorer
        config_dict.update(
            {
                "num_workers": 1,
                "seed": 42,
                "processor": "openai/clip-vit-base-patch32",
                "seq_len": 2048,
                "dataset_statistics": [f"{args.dataset_path}/stats.json"],
                "dataset_manifest": [f"{args.dataset_path}/manifest.jsonl"],
                "normalization": {"enabled": True},
                # The dataloader only loads used fields, so we need to add all fields that we want to visualize
                "proprioception_fields": [
                    "robot__actual__poses__left::panda__xyz",
                    "robot__actual__poses__right::panda__xyz",
                    "robot__actual__poses__left::panda__rot_6d",
                    "robot__actual__poses__right::panda__rot_6d",
                    "robot__actual__grippers__left::panda_hand",
                    "robot__actual__grippers__right::panda_hand",
                    "robot__desired__poses__left::panda__xyz",
                    "robot__desired__poses__right::panda__xyz",
                    "robot__desired__poses__left::panda__rot_6d",
                    "robot__desired__poses__right::panda__rot_6d",
                    "robot__desired__grippers__left::panda_hand",
                    "robot__desired__grippers__right::panda_hand",
                    "robot__actual__poses__left::panda__xyz_relative",
                    "robot__actual__poses__right::panda__xyz_relative",
                    "robot__actual__poses__left::panda__rot_6d_relative",
                    "robot__actual__poses__right::panda__rot_6d_relative",
                    "robot__desired__poses__left::panda__xyz_relative",
                    "robot__desired__poses__right::panda__xyz_relative",
                    "robot__desired__poses__left::panda__rot_6d_relative",
                    "robot__desired__poses__right::panda__rot_6d_relative",
                ],
                "action_fields": [
                    "robot__action__poses__left::panda__xyz",
                    "robot__action__poses__right::panda__xyz",
                    "robot__action__poses__left::panda__rot_6d",
                    "robot__action__poses__right::panda__rot_6d",
                    "robot__action__grippers__left::panda_hand",
                    "robot__action__grippers__right::panda_hand",
                ],
                "dataset_weighting": [1.0],
                "dataset_modality": ["robotics"],
            }
        )

        # Load camera names from preprocessing config to get original_intrinsics/extrinsics fields
        try:
            preproc_config_path = f"{args.dataset_path}/preprocessing_config.yaml"
            with fsspec.open(preproc_config_path, "r") as f:
                preproc_config = yaml.safe_load(f)
            camera_names = preproc_config.get("camera_names", [])
            config_dict["intrinsics_fields"] = [f"original_intrinsics.{cam}" for cam in camera_names]
            config_dict["extrinsics_fields"] = [f"extrinsics.{cam}" for cam in camera_names]
            print(f"📷 Found cameras: {camera_names}")
        except Exception as e:
            print(f"⚠️ Could not load camera names from preprocessing config: {e}")

        params = RoboticsDataParams.from_dict(config_dict)

        data_loader = RoboticsDataLoader(
            params,
            max_samples=args.max_samples,
            max_shards=args.max_shards,
            use_dataloader=use_dataloader,
            device=args.device,
            dtype=args.dtype,
        )
        samples = data_loader.load_samples_auto()
    else:  # Use model and load data using model config
        if args.model_predictions_path.endswith("/"):
            model_predictions_path = args.model_predictions_path[:-1] + "/config.yaml"
        elif not args.model_predictions_path.endswith(".yaml"):
            model_predictions_path = args.model_predictions_path + "/config.yaml"
        else:
            model_predictions_path = args.model_predictions_path
        cfg = load_experiment_params_from_yaml(model_predictions_path)
        model = create_model(cfg.model)
        model.to(args.device, dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
        model.eval()
        data_loader = RoboticsDataLoader(
            cfg.data,
            max_samples=args.max_samples,
            max_shards=args.max_shards,
            use_dataloader=use_dataloader,
            device=args.device,
            dtype=args.dtype,
        )
        # Find the checkpoint file and load the latest
        checkpoint_file = [
            f
            for f in os.listdir(args.model_predictions_path + "/checkpoints")
            if f.endswith(".pt") and f.startswith("checkpoint_")
        ]
        # Sort numerically by extracting the checkpoint number
        checkpoint_file = sorted(checkpoint_file, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
        print(f"Loading model from {args.model_predictions_path + '/checkpoints/' + checkpoint_file}")
        # Load checkpoint
        load_model_checkpoint(model, args.model_predictions_path + "/checkpoints/" + checkpoint_file)
        model.to("cuda")
        model.eval()
        object.__setattr__(cfg.hparams, "global_batch_size", 1)
        object.__setattr__(cfg.hparams, "per_gpu_batch_size", 1)
        samples = data_loader.load_samples_from_model_predictions(
            model, num_inference_steps=args.num_inference_steps, cfg=cfg
        )

    if not samples:
        print("❌ No samples loaded!")
        return

    print(f"✅ Loaded {len(samples)} samples")

    # Create explorer
    print("🚀 Starting Gradio data explorer...")
    explorer = GradioDataExplorer(samples, trajectory_length=args.trajectory_length)

    # Create and launch interface
    interface = explorer.create_interface()

    print("📊 Features:")
    print("  - Interactive sample navigation with slider")
    print("  - Camera selection dropdown")
    print("  - Real-time trajectory overlay on images")
    print("  - 3D trajectory visualization")
    print("  - Gripper state indicators")
    print("  - Sample metadata display")
    print("  - Configurable shard and sample limits for faster loading")
    print("  - Choice between dataloader pipeline and direct file loading")

    # Handle port selection
    launch_kwargs = {"share": args.share, "show_error": True}

    if args.port is not None:
        launch_kwargs["server_port"] = args.port
        print(f"🌐 Launching web interface on port {args.port}")
    else:
        print("🌐 Launching web interface on auto-selected port")

    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
