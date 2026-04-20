"""Rerun visualizer for LBM samples.

This uses the same argument parsing `draccus.parse(config_class=TrainExperimentParams)` as `main.py`
so you can take any script in `./examples` and copy paste the arguments exactly.

Example:
    VISUALIZER=rerun uv run --group visualization vla_foundry/data/scripts/vis/lbm_vis.py \
    --config_path vla_foundry/config_presets/training_jobs/diffusion_policy_bellpepper.yaml \
    --total_train_samples 100
"""

import logging
import os
import sys
from collections.abc import Mapping

import draccus
import numpy as np

from vla_foundry.data.dataloader import get_datastring_input, get_wds_dataloader
from vla_foundry.data.robotics.cv_utils import (
    create_images_with_projected_trace,
    intrinsics_3x3_to_4,
    scale_intrinsics_4_for_resize_and_crop,
    transform_points_to_camera_frame,
)
from vla_foundry.data.robotics.normalization import RoboticsNormalizer
from vla_foundry.data.robotics.utils import (
    invert_homogeneous_transform,
    to_pose_matrix,
)
from vla_foundry.logger import setup_logging
from vla_foundry.params.train_experiment_params import TrainExperimentParams
from vla_foundry.utils import get_experiment_name, set_random_seed
from vla_foundry.visualizers import visualizer as vz

NDArray = np.ndarray


class FrameAssembler:
    """Compute and package pose frames for visualization."""

    def base_to_frame_poses(self, lowdim: Mapping[str, NDArray]) -> dict[str, NDArray]:
        """Return mapping of base-anchored frame trajectories."""

        # Get left end-effector poses.
        base_T_left_xyz_actual = lowdim.get("robot__actual__poses__left::panda__xyz")[0]  # (N, 3)
        base_T_left_rot6d_actual = lowdim.get("robot__actual__poses__left::panda__rot_6d")[0]  # (N, 6)
        base_T_left_actual = to_pose_matrix(base_T_left_xyz_actual, base_T_left_rot6d_actual)

        # Get right end-effector poses.
        base_T_right_xyz_actual = lowdim.get("robot__actual__poses__right::panda__xyz")[0]  # (N, 3)
        base_T_right_rot6d_actual = lowdim.get("robot__actual__poses__right::panda__rot_6d")[0]  # (N, 6)
        base_T_right_actual = to_pose_matrix(base_T_right_xyz_actual, base_T_right_rot6d_actual)

        if base_T_left_actual is None:
            raise ValueError("'base_T_left_actual' could not be calculated from lowdim data")
        if base_T_right_actual is None:
            raise ValueError("'base_T_right_actual' could not be calculated from lowdim data")

        return {
            "base/left_eef_actual": base_T_left_actual,
            "base/right_eef_actual": base_T_right_actual,
        }


class Plotter:
    """Logging helpers that call into the visualization backend."""

    @staticmethod
    def log_images(path: str, img_data: Mapping[str, NDArray]) -> None:
        vz.log_images(path, img_data)


class RerunSampleVisualizer:
    """High-level visualizer that ties together reading, frames, and plotting."""

    def __init__(self, normalizer: RoboticsNormalizer | None = None):
        self._frames = FrameAssembler()
        self._plot = Plotter()
        self._normalizer = normalizer

    def visualize_sample(self, sample_id: str, payload: Mapping[str, object], image_names: list[str]) -> None:
        """Visualize a single sample's payload."""
        pixel_values: NDArray | None = payload.get("pixel_values").numpy()  # type: ignore[assignment]
        lowdim: Mapping[str, NDArray] | None = payload.get("lowdim")  # type: ignore[assignment]
        metadata: Mapping[str, object] | None = payload.get("metadata")[0]  # type: ignore[assignment]

        # Denormalize low-dim data and convert to numpy arrays.
        if self._normalizer is not None and lowdim is not None:
            anchor_relative_idx = metadata.get("anchor_relative_idx") if metadata else None
            for field_name, array in lowdim.items():
                try:
                    lowdim[field_name] = self._normalizer.denormalize_tensor(
                        array, field_name, anchor_timestep=anchor_relative_idx
                    ).numpy()
                except Exception as exc:
                    logging.warning("Denormalization failed for %s: %s", field_name, exc)

        # Convert lowdim entries to numpy arrays if not already.
        if lowdim is not None:
            for field_name, array in lowdim.items():
                if not isinstance(array, np.ndarray):
                    lowdim[field_name] = array.numpy()

        # Compute frame poses of interest (e.g. end effectors poses).
        frame_poses = self._frames.base_to_frame_poses(lowdim)

        # Handle pixel_values shape - expect (B, N, C, H, W) where B=batch_size, N=num_images
        # but some branches return (N, C, H, W) directly
        if pixel_values.ndim == 5:
            # Expected: (B, N, C, H, W) -> extract first batch element -> (N, C, H, W)
            pixel_values = pixel_values[0, :, :, :]
        elif pixel_values.ndim == 4:
            # Already (N, C, H, W), no need to index
            pass
        else:
            raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

        # (num_cameras * num_image_timesteps, H, W, C)
        pixel_values = pixel_values.transpose(0, 2, 3, 1)

        # Populate data dictionaries keyed by image name (one for each camera and timestep).
        image_data: dict[str, NDArray] = {}
        extrinsics_data: dict[str, NDArray] = {}
        intrinsics_data: dict[str, NDArray] = {}
        camera_data: dict[str, str] = {}

        num_available_images = pixel_values.shape[0]
        if len(image_names) != num_available_images:
            logging.warning(
                "Image count mismatch: got %d image_names but %d image tensors; using first %d entries",
                len(image_names),
                num_available_images,
                min(len(image_names), num_available_images),
            )

        extrinsics_dict = payload.get("extrinsics")[0]
        intrinsics_dict = payload.get("intrinsics")[0]

        for i, image_name in enumerate(image_names[:num_available_images]):
            # Find corresponding camera name to image name (camera name should be a substring).
            camera_name = next((x for x in payload["camera_names"] if x in image_name), None)
            if not camera_name:
                continue

            extrinsics_value = extrinsics_dict.get(f"extrinsics.{camera_name}", None)
            intrinsics_value = intrinsics_dict.get(f"original_intrinsics.{camera_name}", None)
            if intrinsics_value is None:
                intrinsics_value = intrinsics_dict.get(f"intrinsics.{camera_name}", None)

            if extrinsics_value is None or intrinsics_value is None:
                continue

            camera_data[image_name] = camera_name
            image_data[image_name] = pixel_values[i, :, :, :]  # (H, W, C)
            extrinsics_data[image_name] = extrinsics_value.numpy()  # (N, 4, 4)
            intrinsics_data[image_name] = intrinsics_value.numpy()  # (N, 3, 3)

        if not image_data:
            logging.warning("No valid images with camera intrinsics/extrinsics found in sample %s", sample_id)
            return

        # Normalize all images.
        for image_key in image_data:
            img_numpy = image_data[image_key]
            min_val = img_numpy.min()
            max_val = img_numpy.max()
            img_numpy = img_numpy - min_val
            denom = max_val - min_val

            img_numpy = img_numpy / denom if denom > 0 else np.zeros_like(img_numpy)

            img_numpy = img_numpy * 255
            img_numpy = img_numpy.astype(np.uint8)
            image_data[image_key] = img_numpy

        # Visualize all images.
        if image_data is not None:
            self._plot.log_images("", image_data)

        # Scale intrinsics to handle processing on original image (resize then crop).
        scaled_intrinsics_data = {}
        for image_key in intrinsics_data:
            H, W, C = image_data[image_key].shape
            W0, H0 = metadata["original_image_sizes"][f"{camera_data[image_key]}"]
            intrinsics_4 = intrinsics_3x3_to_4(intrinsics_data[image_key])
            scaled_intrinsics = scale_intrinsics_4_for_resize_and_crop(
                intrinsics_4,
                (W0, H0),
                (W, H),
            )
            # Use static intrinsics per image source.
            scaled_intrinsics_data[image_key] = scaled_intrinsics[0]

        # Get offsets between image timesteps and lowdim timesteps.
        assert metadata["image_timesteps"][0] >= metadata["lowdim_start_timestep"]
        tm1_offset = metadata["image_timesteps"][0] - metadata["lowdim_start_timestep"]  # t-1 image offset
        t0_offset = metadata["image_timesteps"][1] - metadata["lowdim_start_timestep"]  # t0 image offset

        # Pass in a different list of trace points for each image, and offset the trajectory
        # points based on the lowdim start timestep.
        trace_pts_dict = dict()
        for image_key in image_data:
            if "t0" in image_key:
                time_offset = t0_offset
            elif "t-1" in image_key:
                time_offset = tm1_offset
            else:
                raise ValueError(f"Cannot infer time offset from image_key: {image_key}")

            camera_T_base = invert_homogeneous_transform(extrinsics_data[image_key][time_offset])

            camera_t_left_pts_actual = transform_points_to_camera_frame(
                camera_T_base, frame_poses["base/left_eef_actual"][:, :3, 3]
            )
            camera_t_right_pts_actual = transform_points_to_camera_frame(
                camera_T_base, frame_poses["base/right_eef_actual"][:, :3, 3]
            )

            trace_pts_dict[image_key] = [
                camera_t_left_pts_actual[time_offset:],
                camera_t_right_pts_actual[time_offset:],
            ]

        # Visualize projected trajectory traces on images.
        img_data_with_traces = create_images_with_projected_trace(image_data, scaled_intrinsics_data, trace_pts_dict)
        self._plot.log_images("projected", img_data_with_traces)


def main(argv: list[str] | None = None) -> None:
    # Parse --ordered and --subsample flags before draccus processes the rest of argv.
    ordered = "--ordered" in sys.argv
    if ordered:
        sys.argv = [a for a in sys.argv if a != "--ordered"]

    # Parse --subsample=N argument (default 1 = no subsampling)
    subsample = 1
    subsample_args = [arg for arg in sys.argv if arg.startswith("--subsample=")]
    if subsample_args:
        subsample = int(subsample_args[0].split("=")[1])
        sys.argv = [a for a in sys.argv if not a.startswith("--subsample=")]

    # Use dataloader to iterate through samples.
    cfg = draccus.parse(config_class=TrainExperimentParams)

    # When --ordered is set, rewrite manifest paths to use episodes/ and disable shuffling.
    if ordered:
        new_manifests = []
        for path in cfg.data.dataset_manifest:
            new_manifests.append(path.replace("/shards/manifest.jsonl", "/episodes/manifest.jsonl"))
        object.__setattr__(cfg.data, "dataset_manifest", new_manifests)
        object.__setattr__(cfg.data, "shuffle", False)

    # Force batch size to 1
    object.__setattr__(cfg.hparams, "per_gpu_batch_size", 1)
    object.__setattr__(cfg.hparams, "global_batch_size", 1)

    # Seed rank-0 before any object creation for reproducibility.
    set_random_seed(cfg.hparams.seed, 0)

    # Set path for experiment, log, checkpoints.
    experiment_name = get_experiment_name(cfg)
    if cfg.save_path is None:
        experiment_path = os.path.join("experiments", experiment_name)
    else:
        experiment_path = os.path.join(cfg.save_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    log_path = os.path.join(experiment_path, "out.log")
    setup_logging(log_path, logging.INFO)
    checkpoint_path = os.path.join(experiment_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    start_checkpoint_num = 0
    # Per-dataset cursors and shuffle seeds allow resuming mixed datasets.
    curr_shard_idx_per_dataset = [0 for dataset in range(len(cfg.data.dataset_manifest))]
    if cfg.data.shuffle:
        shard_shuffle_seed_per_dataset = [cfg.hparams.seed for dataset in range(len(cfg.data.dataset_manifest))]
    else:
        # None seeds preserve original manifest order for sequential loading.
        shard_shuffle_seed_per_dataset = [None for dataset in range(len(cfg.data.dataset_manifest))]

    # Partition the global sample budget into evenly-sized checkpoint chunks.
    samples_per_checkpoint = cfg.total_train_samples // cfg.num_checkpoints
    datastrings, num_samples_per_dataset, curr_shard_idx_per_dataset, shard_shuffle_seed_per_dataset = (
        get_datastring_input(
            num_samples=samples_per_checkpoint,
            curr_shard_idx_per_dataset=curr_shard_idx_per_dataset,
            shard_shuffle_seed_per_dataset=shard_shuffle_seed_per_dataset,
            manifest_paths=cfg.data.dataset_manifest,
            dataset_weighting=cfg.data.dataset_weighting,
            allow_multiple_epochs=cfg.data.allow_multiple_epochs,
            num_workers_per_gpu=cfg.data.num_workers,
            world_size=cfg.distributed.world_size,
        )
    )

    dataloader = get_wds_dataloader(datastrings, num_samples_per_dataset, start_checkpoint_num, cfg)

    normalizer = None
    if cfg.data.normalization.enabled:
        normalizer = RoboticsNormalizer(
            normalization_params=cfg.data.normalization, statistics_path=cfg.data.dataset_statistics
        )
        if normalizer:
            logging.info("RoboticsDataLoader: Normalizer initialized for denormalization")
        else:
            logging.warning("RoboticsDataLoader: Failed to initialize normalizer for denormalization")

    # Check if we're in notebook/Colab mode to prevent auto-shutdown
    # The backend is hardcoded to colab=True, so use spawn=False to match
    if not vz._STATE.initialized:
        vz.init(run_name="lbm_visualization", spawn=False, open_browser=False)

    visualizer = RerunSampleVisualizer(normalizer)

    print(f"Starting visualization loop for {cfg.total_train_samples} samples...")
    if subsample > 1:
        print(f"Subsampling: visualizing every {subsample} samples")

    sample_count = 0
    visualized_count = 0
    for idx, batch in enumerate(dataloader.dataloader):
        sample_count += 1

        # Skip samples based on subsampling
        if subsample > 1 and (idx % subsample != 0):
            continue

        if visualized_count % 10 == 0:  # Print progress every 10 visualized samples
            print(f"Visualizing sample {idx}...")

        vz.set_time("sample", sequence=idx)
        visualizer.visualize_sample(str(idx), batch, cfg.data.image_names)
        visualized_count += 1

    print(f"Finished processing {sample_count} samples (visualized {visualized_count}).")
    print("In Colab/Jupyter, call: rr.notebook_show().")


if __name__ == "__main__":
    main()
