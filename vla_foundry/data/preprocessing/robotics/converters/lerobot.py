import os
import re
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import ray

from vla_foundry.data.preprocessing.robotics.converters.base import BaseRoboticsConverter
from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks
from vla_foundry.data.preprocessing.utils import is_still_sample
from vla_foundry.file_utils import copy_to_temp_file, file_exists, json_load, jsonl_load, list_directory


def resolve_path(base_path: str, relative_path: str) -> str:
    # If relative_path is already absolute (starts with s3:// or /), return as-is
    if relative_path.startswith("s3://") or relative_path.startswith("/"):
        return relative_path
    return f"{base_path.rstrip('/')}/{relative_path.lstrip('/')}"


def detect_fps(info_path: str) -> float:
    """Automatically detect FPS from info.json file."""
    info = json_load(info_path)
    if "fps" in info:
        fps = float(info["fps"])
        print(f"Detected FPS from global setting: {fps}")
    else:
        fps = 30.0
        print(f"Warning: Could not detect FPS from info.json, using default FPS {fps}")
    return fps


@ray.remote
def build_episode_lookup_chunk(chunk_dir: str) -> dict[int, str]:
    """Build episode lookup for a single chunk directory."""
    episode_lookup = {}
    for file in list_directory(chunk_dir):
        if file.endswith(".parquet") and "episode_" in file:
            match = re.search(r"episode_(\d+)\.parquet", file)
            if match:
                ep_num = int(match.group(1))
                episode_lookup[ep_num] = f"{chunk_dir.rstrip('/')}/{file}"
    return episode_lookup


def build_episode_lookup(data_chunks: list[str]) -> dict[int, str]:
    print(f"Building episode lookup dict from {len(data_chunks)} chunks...")
    # Process chunks in parallel
    chunk_futures = [build_episode_lookup_chunk.remote(chunk) for chunk in data_chunks]
    chunk_results = ray.get(chunk_futures)

    # Merge results
    episode_lookup = {}
    for chunk_result in chunk_results:
        episode_lookup.update(chunk_result)
    return episode_lookup


@ray.remote
def discover_episodes_chunk(chunk_dir: str) -> list[str]:
    episodes = []
    for file in list_directory(chunk_dir):
        if file.endswith(".parquet") and "episode_" in file:
            match = re.search(r"episode_(\d+)\.parquet", file)
            if match:
                episodes.append(f"{chunk_dir.rstrip('/')}/{file}")
    return episodes


def discover_image_columns(data_chunks: list[str], episode_file_pattern: str) -> list[str]:
    """Discover image columns by checking first available episode."""
    for chunk_dir in data_chunks:
        for episode_idx in range(10):
            episode_path = f"{chunk_dir.rstrip('/')}/{episode_file_pattern.format(episode_idx)}"
            if not file_exists(episode_path):
                continue

            # Read parquet to examine columns
            if episode_path.startswith("s3"):
                with copy_to_temp_file(episode_path) as temp_parquet:
                    df = pq.read_table(temp_parquet).to_pandas()
            else:
                df = pq.read_table(episode_path).to_pandas()

            # Find columns containing image bytes
            image_columns = []
            for col in df.columns:
                if df[col].dtype == object and len(df[col]) > 0:
                    first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if isinstance(first_val, (dict, bytes)) and (isinstance(first_val, bytes) or "bytes" in first_val):
                        image_columns.append(col)

            if image_columns:
                print(f"Discovered image columns: {image_columns}")
                return image_columns

    print("Warning: Could not discover image columns")
    return []


def decode_video_frames(video_path: str) -> list[np.ndarray]:
    """Decode all frames from a video file, returning a list of (H, W, 3) RGB uint8 numpy arrays."""
    if video_path.startswith("s3"):
        with copy_to_temp_file(video_path) as local_path:
            return decode_video_frames(local_path)

    frames = []
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def discover_cameras(video_chunks: list[str]) -> dict[str, str]:
    """Discover available cameras by scanning video chunk directories.

    Returns a dict mapping camera directory name -> camera directory name.
    The directory name is the full name as it appears on disk (e.g., 'observation.image').
    """
    cameras = {}

    if not video_chunks:
        return cameras

    first_chunk = video_chunks[0]
    for item in list_directory(first_chunk):
        cameras[item] = item

    print(f"Discovered cameras: {list(cameras.keys())}")
    return cameras


@ray.remote
def build_video_lookup_chunk(chunk_dir: str, cameras: dict[str, str]) -> dict[tuple[int, str], str]:
    """Build video lookup for a single chunk directory."""
    video_lookup = {}
    for _camera_name, camera_path in cameras.items():
        camera_dir = f"{chunk_dir.rstrip('/')}/{camera_path}"
        files = list_directory(camera_dir)
        for file in files:
            if file.endswith(".mp4") and "episode_" in file:
                match = re.search(r"episode_(\d+)\.mp4", file)
                if match:
                    ep_num = int(match.group(1))
                    video_lookup[(ep_num, camera_path)] = f"{camera_dir}/{file}"
    return video_lookup


def build_video_lookup(video_chunks: list[str], cameras: dict[str, str]) -> dict[tuple[int, str], str]:
    print(f"Building video lookup dict from {len(video_chunks)} chunks and {len(cameras)} cameras...")
    # Process chunks in parallel
    chunk_futures = [build_video_lookup_chunk.remote(chunk, cameras) for chunk in video_chunks]
    chunk_results = ray.get(chunk_futures)

    # Merge results
    video_lookup = {}
    for chunk_result in chunk_results:
        video_lookup.update(chunk_result)
    return video_lookup


class LeRobotConverter(BaseRoboticsConverter):
    def __init__(self, cfg):
        super().__init__(cfg)

        source_dir_contents = list_directory(cfg.source_episodes[0])
        self.has_videos = "videos" in source_dir_contents

        self.meta_episodes_path = resolve_path(cfg.source_episodes[0], "meta/episodes.jsonl")
        self.info_path = resolve_path(cfg.source_episodes[0], "meta/info.json")
        self.tasks_path = resolve_path(cfg.source_episodes[0], "meta/tasks.jsonl")
        self.data_path = resolve_path(cfg.source_episodes[0], "data")
        self.videos_path = resolve_path(cfg.source_episodes[0], "videos")

        self.fps = detect_fps(self.info_path)

        self.data_chunks = self.discover_chunks([self.data_path])
        if self.has_videos:
            self.video_chunks = self.discover_chunks([self.videos_path])
        else:
            self.video_chunks = []

        # Pre-build episode lookup. Maps from episode number to episode path.
        self.episode_lookup = build_episode_lookup(self.data_chunks)
        print(f"Built episode lookup with {len(self.episode_lookup)} episodes")

        # Pre-build image columns, cameras, and video lookup.
        self.image_columns, self.cameras, self.video_lookup = [], {}, {}
        if not self.has_videos:
            self.image_columns = discover_image_columns(self.data_chunks, "episode_{:06d}.parquet")
            if not self.image_columns or len(self.image_columns) == 0:
                raise ValueError("No image columns found in parquet files and none specified with --image_columns")
        else:
            # Discover cameras once and reuse it for all shards.
            self.cameras = discover_cameras(self.video_chunks)
            assert len(self.cameras) > 0, "No cameras found"
            print(f"Discovered cameras: {list(self.cameras.keys())}")

            if not cfg.camera_names:
                raise ValueError(
                    "camera_names must be specified for video-based LeRobot datasets. "
                    f"Available cameras from video directories: {list(self.cameras.keys())}"
                )

            unknown = set(cfg.camera_names) - set(self.cameras)
            if unknown:
                raise KeyError(f"Camera(s) {unknown} not found in discovered cameras {list(self.cameras.keys())}")

            # Pre-build video lookup once and reuse it for all shards.
            self.video_lookup = build_video_lookup(self.video_chunks, self.cameras)
            print(f"Built video lookup with {len(self.video_lookup)} video files")

        # Read episodes metadata
        self.entries = jsonl_load(self.meta_episodes_path)
        print(f"Loaded {len(self.entries)} episodes metadata")

    def get_language_instructions(self, sample_metadata) -> dict[str, str]:
        episode_index = sample_metadata["episode_index"]
        return {"original": self.entries[episode_index]["tasks"][0]}

    def discover_chunks(self, source_paths: list[str]) -> list[str]:
        """Discover all chunk directories in the base path."""
        chunk_dirs = []
        for source_path in source_paths:
            for item in list_directory(source_path):
                if item.startswith("chunk-"):
                    chunk_path = f"{source_path.rstrip('/')}/{item}"
                    chunk_dirs.append(chunk_path)
        chunk_dirs.sort()  # Sort to ensure consistent ordering
        return chunk_dirs

    def discover_episodes(self, source_paths: list[str], max_episodes_to_process: int = -1) -> list[str]:
        chunks = self.discover_chunks([os.path.join(source_paths[0], "data")])
        chunk_futures = [discover_episodes_chunk.remote(chunk) for chunk in chunks]
        chunk_results = ray.get(chunk_futures)

        all_episodes = []
        for chunk_result in chunk_results:
            all_episodes.extend(chunk_result)
        return all_episodes

    def get_episode_length(self, episode_data: Any) -> int:
        return len(episode_data)

    def load_episode_data(self, episode_path):
        if episode_path.startswith("s3"):
            with copy_to_temp_file(episode_path) as temp_parquet:
                df = pq.read_table(temp_parquet).to_pandas()
        else:
            df = pq.read_table(episode_path).to_pandas()
        return df

    def extract_camera_data(self, episode_data: Any):
        """
        Here, camera_data values are bytes (inline images) or numpy arrays (video frames).
        """
        camera_data = {}

        if self.has_videos:
            episode_index = int(episode_data["episode_index"].iloc[0])
            num_rows = len(episode_data)
            for camera_name in self.cfg.camera_names:
                video_path = self.video_lookup.get((episode_index, camera_name))
                if video_path is None:
                    print(f"Warning: No video found for episode {episode_index}, camera {camera_name}")
                    continue
                frames = decode_video_frames(video_path)
                if len(frames) != num_rows:
                    if len(frames) < num_rows:
                        raise ValueError(
                            f"Video {video_path} has {len(frames)} frames but parquet has {num_rows} rows. "
                            f"Video has fewer frames than expected — the dataset may be corrupted."
                        )
                    # Video has more frames than parquet rows, truncate to match
                    print(
                        f"Warning: Video has {len(frames)} frames but parquet has {num_rows} rows "
                        f"for episode {episode_index}, camera {camera_name}. Truncating video to match parquet."
                    )
                    frames = frames[:num_rows]
                camera_data[camera_name] = frames
        else:
            for camera_name in self.cfg.camera_names:
                images = episode_data[camera_name].to_list()
                if isinstance(images[0], dict) and "bytes" in images[0]:
                    images = [image["bytes"] for image in images]
                elif isinstance(images[0], bytes):
                    images = images
                else:
                    raise ValueError(f"Unsupported image data format in column {camera_name}")
                camera_data[camera_name] = images

        return camera_data

    def extract_lowdim_data(self, episode_data: Any):
        """
        Return a dictionary with lowdim keys as keys and lowdim data as values.
        """
        lowdim_cols = self.cfg.observation_keys + self.cfg.action_keys
        lowdim_data = {}
        for col in lowdim_cols:
            lowdim_data[col] = np.stack(episode_data[col].to_numpy())
        return lowdim_data

    def extract_intrinsics_extrinsics_data(self, episode_data: Any):
        return None, None

    def extract_metadata_data(self, episode_data: Any):
        exclude_keys = self.cfg.observation_keys + self.cfg.action_keys + list(self.cfg.camera_names)
        metadata_data = {}
        for key, value in episode_data.items():
            if key not in exclude_keys:
                metadata_data[key] = value.to_numpy()
        return metadata_data

    def extract_sample_data(
        self,
        anchor_timestep: int,
        episode_path: str,
        episode_length: int,
        camera_data: dict[str, np.ndarray],
        lowdim_data: dict[str, np.ndarray],
        intrinsics_data: dict[str, np.ndarray],
        extrinsics_data: dict[str, np.ndarray],
        metadata_data: dict[str, Any],
        statistics_ray_actor,
        logger_actor,
    ):
        logger_actor.increment_total_potential_samples.remote()

        # Calculate windows
        lowdim_start = anchor_timestep - self.cfg.past_lowdim_steps
        lowdim_end = anchor_timestep + self.cfg.future_lowdim_steps

        # Check padding
        past_padding = max(0, -lowdim_start)
        future_padding = max(0, lowdim_end - episode_length + 1)

        if past_padding > self.cfg.max_padding_left or future_padding > self.cfg.max_padding_right:
            logger_actor.increment_padding_samples_filtered.remote()
            return None, None, None, None, None

        valid_start = max(0, lowdim_start)
        valid_end = min(episode_length - 1, lowdim_end)

        # Check if robot is stationary (e.g. to filter pauses)
        if self.cfg.filter_still_samples and is_still_sample(
            lowdim_data, valid_start, valid_end, self.cfg.still_threshold
        ):
            logger_actor.increment_still_samples_filtered.remote()
            return None, None, None, None, None

        # Extract images
        sample_images = {}
        actual_image_timesteps = []

        for img_offset in self.cfg.image_indices:
            img_timestep = np.clip(anchor_timestep + img_offset, 0, episode_length - 1)
            actual_image_timesteps.append(int(img_timestep))

            for camera_name, camera_images in camera_data.items():
                key = f"{camera_name}_t{img_offset}"
                sample_images[key] = camera_images[img_timestep]

        # Process lowdim data (which includes actions)
        sample_lowdim = {}
        for key, data in lowdim_data.items():
            valid_data = data[valid_start : valid_end + 1]
            if past_padding > 0 or future_padding > 0:
                valid_data = self.pad_fn(valid_data, past_padding, future_padding)
            sample_lowdim[key] = valid_data

        # Create masks
        past_mask, future_mask = create_past_and_future_masks(
            anchor_timestep, self.cfg.past_lowdim_steps, self.cfg.future_lowdim_steps, episode_length
        )

        # Build stats_sample for batched statistics update (don't send immediately)
        # Must be done before modifying sample_lowdim with masks
        stats_sample = (
            None
            if statistics_ray_actor is None
            else {
                "lowdim": {k: v.copy() for k, v in sample_lowdim.items()},  # Copy before modifying
                "past_mask": past_mask,
                "future_mask": future_mask,
            }
        )

        # Add masks to sample_lowdim (after building stats_sample)
        sample_lowdim["past_mask"] = past_mask
        sample_lowdim["future_mask"] = future_mask

        sample_metadata = {
            "camera_names": list(camera_data.keys()),
            "anchor_relative_idx": int(self.cfg.past_lowdim_steps),
        }
        for key, value in metadata_data.items():
            if isinstance(value, (list, np.ndarray)):
                sample_metadata[key] = value[anchor_timestep]
            else:
                sample_metadata[key] = value
        language_instructions = self.get_language_instructions(sample_metadata)

        return sample_images, sample_lowdim, sample_metadata, language_instructions, stats_sample
