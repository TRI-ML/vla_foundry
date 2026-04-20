import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import ray

from vla_foundry.data.preprocessing.image_utils import init_jpeg_encoder
from vla_foundry.data.preprocessing.robotics.preprocess_masks import PaddingStrategy
from vla_foundry.data.preprocessing.robotics.preprocess_statistics import StreamingDatasetStatistics
from vla_foundry.data.preprocessing.utils import upload_sample_to_s3
from vla_foundry.data.robotics.utils import (
    calculate_relative_pose,
    pose_to_9d,
    to_pose_matrix,
)


class BaseRoboticsConverter:
    """
    Base class for all robotics converters.
    This class handles the logic for discovering episodes, loading episode data, and extracting the relevant fields.

    All converters must inherit from this class and implement the methods in this file.
    Some methods are already implemented in this file, and you can probably use them as is.
    Notably, preprocess_robotics_to_tar.py calls process_episode(), which is already defined in this file.
    You need to define all methods called within process_episode(), as well as any other auxiliary methods you need.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.resize_images_size = cfg.resize_images_size
        self.image_resizing_method = cfg.image_resizing_method
        self.camera_rotations = cfg.camera_rotations
        self.image_indices = sorted(cfg.image_indices) if cfg.image_indices is not None else [-1, 0]

        # Initialize JPEG encoder
        init_jpeg_encoder(cfg.jpeg_quality)

        # Set padding function
        self.pad_fn = PaddingStrategy.get_pad_fn(cfg.padding_strategy)

    def discover_episodes(self, source_paths: list[str], max_episodes_to_process: int = -1) -> list[str]:
        """
        Given a list of source paths, return a list of all full episode paths in the directories.
        """
        raise NotImplementedError("Subclasses must implement discover_episodes()")

    def load_episode_data(self, episode_path: str) -> Any:
        """
        The output here will be a dictionary (or anything, really).
        No strict format for the keys. Return whatever is needed for the extract() methods below.
        This output dictionary will be passed to the extract_camera_data() and extract_lowdim_data() methods.
        """
        raise NotImplementedError("Subclasses must implement load_episode_data()")

    def get_episode_length(self, episode_data: Any) -> int:
        """
        Given the episode_data, return the number of timesteps in the episode.
        """
        raise NotImplementedError("Subclasses must implement get_episode_length()")

    def extract_camera_data(self, episode_data: Any):
        """
        Return a dictionary with camera names as keys and image data as values.
        Camera data can be images or bytes. Both are supported in upload_sample_to_s3.
        Can be as simple as `return episode_data["observations"]`
        The values here will cover all the timesteps, then the process_episode() will extract the specific frames.
        """
        return None

    def extract_lowdim_data(self, episode_data: Any):
        """
        Return a dictionary with lowdim keys as keys and lowdim data as values.
        lowdim covers all low dimensional numpy arrays, including actions, proprioception, intrinsics, extrinsics, etc.
        Can be as simple as `return episode_data["lowdim"]`
        The values here will cover all the timesteps, then the process_episode() will extract the specific frames.
        """
        return None

    def extract_intrinsics_extrinsics_data(self, episode_data: Any):
        """
        Return a dictionary with intrinsics and extrinsics keys as keys and data as values.
        Can be as simple as `return episode_data["intrinsics"], episode_data["extrinsics"]`
        This is optional. Can return None, None if not available.
        """
        return None, None

    def extract_metadata_data(self, episode_data: Any):
        """
        Return a either a dictionary or a SampleMetadata object.
        The values here are global values that are shared across all timesteps.
        Alternatively, they can be lists of values, one for each timestep (e.g. timestamps in seconds).
        """
        return None

    def extract_sample_data(
        self,
        anchor_timestep: int,
        episode_path: str,
        episode_length: int,
        camera_data: dict[str, Any],
        lowdim_data: dict[str, Any],
        intrinsics_data: dict[str, Any],
        extrinsics_data: dict[str, Any],
        metadata_data: dict[str, Any],
        statistics_ray_actor,
        logger_actor,
    ):
        """
        Takes in camera_data, lowdim_data, intrinsics_data, extrinsics_data, metadata_data.
        Uses anchor_timestep to extract the specific frames.

        Arguments:
        - anchor_timestep: the current timestep to extract the sample data for.
        - episode_path: the path to the current episode.
        - episode_length: the number of timesteps in the current episode. Returned from get_episode_length().
        - camera_data: a dict with camera names as keys and images (array or bytes) as values.
        Returned from extract_camera_data().
        - lowdim_data: a dict with lowdim keys as keys and lowdim data as values. Returned from extract_lowdim_data().
        - intrinsics_data: Returned from extract_intrinsics_extrinsics_data().
        - extrinsics_data: Returned from extract_intrinsics_extrinsics_data().
        - metadata_data: a dict or a SampleMetadata object (some fields can be blank).
        Returned from extract_metadata_data().

        Returns:
        - sample_images: a dictionary with camera names as keys and image data as values.
        - sample_lowdim: a dictionary with lowdim keys as keys and lowdim data as values.
        - sample_metadata: a dictionary or a SampleMetadata object (some fields can be blank).
        - language_instructions: a dictionary with keys "original", etc. and language instructions as values.
        IMPORTANT: Make sure to also update statistics data in this function, as well as the sample counts.
        You can use the statistics_ray_actor and the logger_actor to update the statistics and sample counts.
        """
        raise NotImplementedError("Subclasses must implement extract_sample_data()")

    def create_relative_lowdim_data(
        self, lowdim_data: dict[str, np.ndarray], reference_data: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Create relative coordinate data using configuration-based pose matching."""
        if not hasattr(self, "pose_groups") or not self.pose_groups:
            # No pose groups configured - return empty dict (no relative coordinates needed)
            return {}

        relative_data = {}
        for pose_group in self.pose_groups:
            xyz_key = pose_group["position_key"]
            rot_6d_key = pose_group["rotation_key"]

            xyz_data = lowdim_data[xyz_key]
            rot_6d_data = lowdim_data[rot_6d_key]
            reference_xyz = reference_data[xyz_key]
            reference_rot_6d = reference_data[rot_6d_key]

            # Create reference pose matrix
            reference_pose_matrix = to_pose_matrix(reference_xyz, reference_rot_6d)

            # Create pose matrices for all timesteps (vectorized)
            current_pose_matrices = to_pose_matrix(xyz_data, rot_6d_data)

            # Calculate relative poses (vectorized)
            relative_pose_matrices = calculate_relative_pose(current_pose_matrices, reference_pose_matrix)

            # Extract xyz and rot_6d from relative pose matrices (vectorized)
            relative_xyz, relative_rot_6d = pose_to_9d(relative_pose_matrices)

            # Store relative data with appropriate names
            relative_data[f"{xyz_key}_relative"] = relative_xyz
            relative_data[f"{rot_6d_key}_relative"] = relative_rot_6d

        return relative_data

    def get_episode_id(self, episode_path: str) -> str:
        """Get episode ID from episode path."""
        return os.path.basename(episode_path.rstrip("/"))

    def process_episode(self, episode_path: str, statistics_ray_actor, logger_actor) -> None:
        """
        Process an episode and return a dictionary of the processed episode.

        Here, "processing" an episode means:
        1. Take in episode_path
        2. Load the episode data
        3. Extract the relevant fields (camera data, lowdim data, intrinsics data, extrinsics data, and metadata data)
            - Other modalities should be added here as needed.
        4. For each timestep in the episode:
            - Extract the sample data for the current timestep
            - Upload the sample data to S3
        """
        try:
            episode_data = self.load_episode_data(episode_path)
            episode_length = self.get_episode_length(episode_data)
            camera_data = self.extract_camera_data(episode_data)

            # Skip episode if no camera data was found
            if not camera_data:
                print(f"⚠️  Skipping episode {episode_path} - no matching cameras found")
                return []

            lowdim_data = self.extract_lowdim_data(episode_data)
            intrinsics_data, extrinsics_data = self.extract_intrinsics_extrinsics_data(episode_data)
            metadata_data = self.extract_metadata_data(episode_data)

            # Free episode_data — camera_data/lowdim_data now hold what we need
            del episode_data

            # Convert camera arrays to lists of per-frame copies so old frames
            # can be individually freed as we iterate (numpy views would keep the
            # entire contiguous array alive).
            if camera_data is not None:
                for cam_name in list(camera_data.keys()):
                    arr = camera_data[cam_name]
                    if isinstance(arr, np.ndarray):
                        camera_data[cam_name] = [arr[i].copy() for i in range(len(arr))]
                        del arr

            # Determine minimum image offset for frame eviction
            min_img_offset = min(self.image_indices) if self.image_indices else 0
            last_evicted_up_to = -1

            # Use ThreadPoolExecutor with bounded queue to prevent memory blowup
            with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as executor:
                futures = set()
                results = []
                stats_samples_batch = []  # Collect stats samples for batched update
                stats_flush_size = 100  # Flush stats every N samples to bound memory
                stats_futures = []  # Track stats actor calls to ensure completion

                for anchor_timestep in range(0, episode_length, self.cfg.stride):
                    # If we have max_workers futures in flight, wait for one to complete
                    # This bounds memory usage to ~max_workers samples
                    if len(futures) >= self.cfg.num_workers:
                        done_future = next(as_completed(futures))
                        futures.remove(done_future)
                        result = done_future.result()  # Raise any exceptions
                        results.append(result)

                    # Create sample_images, sample_lowdim, sample_metadata, language_instructions,
                    # and optionally stats_sample
                    result = self.extract_sample_data(
                        anchor_timestep,
                        episode_path,
                        episode_length,
                        camera_data,
                        lowdim_data,
                        intrinsics_data,
                        extrinsics_data,
                        metadata_data,
                        statistics_ray_actor,
                        logger_actor,
                    )

                    # Evict camera frames no longer needed by future anchors.
                    # Next anchor needs frame >= (anchor + stride + min_img_offset).
                    if camera_data is not None:
                        evict_below = max(0, anchor_timestep + self.cfg.stride + min_img_offset)
                        if evict_below > last_evicted_up_to + 1:
                            for cam_name in camera_data:
                                num_frames = len(camera_data[cam_name])
                                for idx in range(last_evicted_up_to + 1, min(evict_below, num_frames)):
                                    camera_data[cam_name][idx] = None
                            last_evicted_up_to = evict_below - 1

                    # Handle 4+ tuple returns (stats_sample is optional 5th element)
                    sample_images, sample_lowdim, sample_metadata, language_instructions, *extra = result
                    stats_sample = extra[0] if len(extra) >= 1 else None

                    if sample_images is None and sample_lowdim is None:
                        # Filtered out either by max_padding or still_samples
                        continue

                    # Collect stats sample for batched update, flush periodically to bound memory
                    if stats_sample is not None:
                        stats_samples_batch.append(stats_sample)
                        if len(stats_samples_batch) >= stats_flush_size:
                            if statistics_ray_actor is not None:
                                aggregates = StreamingDatasetStatistics.compute_batch_aggregates(stats_samples_batch)
                                stats_futures.append(statistics_ray_actor.merge_from_aggregates.remote(aggregates))
                            stats_samples_batch = []

                    sample_data = {
                        "images": sample_images,
                        "lowdim": sample_lowdim,
                        "metadata": sample_metadata,
                        "language_instructions": language_instructions,
                    }

                    # Submit upload task
                    future = executor.submit(
                        upload_sample_to_s3,
                        sample_data=sample_data,
                        output_dir=self.output_dir,
                        episode_path=episode_path,
                        episode_id=self.get_episode_id(episode_path),
                        frame_idx=anchor_timestep,
                        jpeg_quality=self.cfg.jpeg_quality,
                        resize_images_size=self.resize_images_size,
                        image_resizing_method=self.image_resizing_method,
                        camera_rotations=self.camera_rotations,
                    )
                    futures.add(future)

                # Wait for remaining uploads to complete and collect results
                for future in as_completed(futures):
                    result = future.result()  # Raise any exceptions
                    results.append(result)

            # Flush any remaining stats samples
            if statistics_ray_actor is not None and stats_samples_batch:
                aggregates = StreamingDatasetStatistics.compute_batch_aggregates(stats_samples_batch)
                stats_futures.append(statistics_ray_actor.merge_from_aggregates.remote(aggregates))

            # Wait for all stats calls to complete before returning,
            # so the stats actor has all data when get_statistics() is called later.
            if stats_futures:
                ray.get(stats_futures)

            return results

        except Exception as e:
            if self.cfg.fail_on_nan:
                raise e
            print(f"Warning: Failed to process episode {episode_path}: {e}")
            return None
