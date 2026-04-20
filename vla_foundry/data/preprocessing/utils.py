import io
import json
import re
import shutil
import tarfile
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import ray
from PIL import Image

from vla_foundry.aws.s3_io import download_fileobj_from_s3, upload_fileobj_to_s3
from vla_foundry.aws.s3_path import S3Path
from vla_foundry.aws.s3_utils import create_s3_client, is_s3_path
from vla_foundry.data.preprocessing.image_utils import (
    ImageResizingMethod,
    depth_image_to_bytes,
    image_to_bytes,
    rotate_image,
)
from vla_foundry.data.robotics.cv_utils import scale_intrinsics_3x3_for_resize_and_crop
from vla_foundry.file_utils import list_s3_directory_recursive


def upload_sample_to_s3(
    sample_data: dict[str, Any],
    output_dir: str,
    episode_path: str,
    episode_id: str,
    frame_idx: int,
    jpeg_quality: int = 95,
    resize_images_size: list[int] | None = None,
    image_resizing_method: ImageResizingMethod = ImageResizingMethod.CENTER_CROP,
    camera_rotations: dict[str, int] | None = None,
) -> str:
    """
    Package a single sample as a tar file and write it to S3 or local disk.

    Builds a tar file containing:
    - Camera images (JPEG for RGB, PNG for depth), optionally resized and rotated
    - Point maps as 3-channel uint16 TIFF files, if present
    - Low-dimensional data (state, actions, masks) as compressed NPZ
    - Metadata and language instructions as JSON
    - Rescaled camera intrinsics when original intrinsics are provided

    Args:
        sample_data: Dictionary built by BaseRoboticsConverter.process_episode() with keys:
            - "images": dict mapping "{camera_name}_t{offset}" keys to RGB numpy
              arrays (H, W, 3) uint8 or pre-encoded JPEG bytes
            - "lowdim": dict mapping field names to numpy arrays, including
              actions, state, masks, and optionally original/rescaled intrinsics
            - "metadata": SampleMetadata dataclass or equivalent dict with
              sample_id, camera_names, original_image_sizes, etc.
            - "language_instructions": dict mapping instruction type (e.g.
              "original", "randomized") to list of strings, or None
        output_dir: S3 URI (s3://bucket/prefix) or local directory path.
        episode_path: Original episode path, used to derive a unique ID for the tar filename.
        episode_id: Episode identifier included in the tar filename.
        frame_idx: Frame index within the episode, included in the tar filename.
        jpeg_quality: JPEG compression quality for RGB images (0-100).
        resize_images_size: Target [width, height] for image resizing. Required when
            images are numpy arrays, must be None when images are pre-encoded bytes.
        image_resizing_method: Resizing strategy (e.g. center crop) applied to numpy images.
        camera_rotations: Optional mapping of camera name to number of 90-degree
            counter-clockwise rotations to apply before encoding.

    Returns:
        Filename of the written tar archive.

    Raises:
        ValueError: If images are numpy arrays but resize_images_size is None,
            or if images are pre-encoded bytes but resize_images_size is set.
        AssertionError: If a camera name in the images is not found in
            metadata.camera_names, or if point map arrays have unexpected
            dtype or shape.
    """
    s3_client = create_s3_client() if is_s3_path(output_dir) else None
    tar_buffer = io.BytesIO()
    uuid_prefix = str(uuid.uuid4())

    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        original_image_sizes = {}
        # Convert images to bytes (JPEG for RGB, PNG for depth)
        for img_key, img_data in sample_data["images"].items():
            # Check if this is a depth image
            is_depth = "depth" in img_key
            camera_name_without_timestep = img_key.rsplit("_t", 1)[0]

            if not isinstance(img_data, bytes):
                # Apply camera-specific rotation if configured
                if camera_rotations and camera_name_without_timestep in camera_rotations:
                    k = camera_rotations[camera_name_without_timestep]
                    if k != 0:
                        img_data = rotate_image(img_data, k)

                if resize_images_size is None:
                    raise ValueError(
                        f"Image '{img_key}' is a numpy array but resize_images_size is not configured. "
                        "resize_images_size must be specified in preprocessing config when using numpy array images. "
                        "Set resize_images_size to your desired [width, height], e.g., [384, 384]."
                    )
                if is_depth:
                    # Depth images: PNG with uint16
                    image_bytes, original_image_size = depth_image_to_bytes(img_data, target_size=resize_images_size)
                    file_extension = "png"
                else:
                    # RGB images: JPEG
                    image_bytes, original_image_size = image_to_bytes(
                        img_data,
                        quality=jpeg_quality,
                        target_size=resize_images_size,
                        resize_method=image_resizing_method,
                    )
                    file_extension = "jpg"
            else:
                # Bytes passed directly - resize cannot be applied
                if resize_images_size is not None:
                    raise ValueError(
                        f"Image '{img_key}' is already encoded as bytes but resize_images_size={resize_images_size} "
                        "is configured. Converters must return numpy arrays for resizing to work. "
                        "Either return numpy arrays from the converter or set resize_images_size=null."
                    )
                image_bytes = img_data
                original_image_size = Image.open(io.BytesIO(img_data)).size
                file_extension = "jpg"  # Assume pre-encoded bytes are JPEG

            # Log original image sizes
            if isinstance(sample_data["metadata"], dict):
                assert camera_name_without_timestep in sample_data["metadata"].get("camera_names")
            else:
                assert camera_name_without_timestep in sample_data["metadata"].camera_names
            original_image_sizes[camera_name_without_timestep] = original_image_size

            tarinfo = tarfile.TarInfo(name=f"{uuid_prefix}.{img_key}.{file_extension}")
            tarinfo.size = len(image_bytes)
            tar.addfile(tarinfo, io.BytesIO(image_bytes))

        if isinstance(sample_data["metadata"], dict):
            sample_data["metadata"]["original_image_sizes"] = original_image_sizes
        else:
            sample_data["metadata"].original_image_sizes = original_image_sizes

        # Add the rescaled intrinsics into the sample_data
        sample_lowdim_data_with_rescaled_intrinsics = dict()
        if "lowdim" in sample_data and sample_data["lowdim"] is not None:
            for lowdim_key in sample_data["lowdim"]:
                # Check if the lowdim_key is of the form "original_intrinsics.{camera_name}"
                if bool(re.fullmatch(r"original_intrinsics\.[A-Za-z0-9_-]+", lowdim_key)):
                    camera_name = lowdim_key.split(".", 1)[1]
                    original_intrinsics = sample_data["lowdim"][lowdim_key]
                    assert camera_name in original_image_sizes, (
                        f"Camera name {camera_name} not found in original_image_sizes"
                    )

                    original_image_size = original_image_sizes[camera_name]
                    scaled_intrinsics = scale_intrinsics_3x3_for_resize_and_crop(
                        original_intrinsics, original_image_size, resize_images_size, image_resizing_method
                    )

                    # Add rescaled intrinsics.
                    sample_lowdim_data_with_rescaled_intrinsics[f"rescaled_intrinsics.{camera_name}"] = (
                        scaled_intrinsics
                    )
            sample_data["lowdim"].update(sample_lowdim_data_with_rescaled_intrinsics)

        for key, value in sample_data.items():
            data_buffer = io.BytesIO()
            if key == "images":  # Already added
                continue
            elif key in ["metadata", "language_instructions"]:
                # Save as JSON
                if isinstance(value, dict):
                    json_str = json.dumps(value, indent=2, default=str)
                elif value is None:  # e.g. language_instructions can be None
                    continue
                else:
                    json_str = json.dumps(asdict(value), indent=2, default=str)
                data_buffer.write(json_str.encode("utf-8"))
                data_buffer.seek(0)
                tarinfo = tarfile.TarInfo(name=f"{uuid_prefix}.{key}.json")
                tarinfo.size = len(data_buffer.getvalue())
                tar.addfile(tarinfo, data_buffer)
            else:
                # Everything else as NPZ
                if isinstance(value, dict):
                    np.savez_compressed(data_buffer, **value)
                else:
                    np.savez_compressed(data_buffer, data=value)
                data_buffer.seek(0)
                tarinfo = tarfile.TarInfo(name=f"{uuid_prefix}.{key}.npz")
                tarinfo.size = len(data_buffer.getvalue())
                tar.addfile(tarinfo, data_buffer)

    tar_buffer.seek(0)
    unique_id = extract_unique_id(episode_path)
    tar_filename = f"{unique_id}_{episode_id}_frame_{frame_idx}.tar"

    if is_s3_path(output_dir):
        # Upload to S3
        parsed = S3Path(s3_path=output_dir)
        s3_key = f"{parsed.key.rstrip('/')}/frames/{tar_filename}"
        s3_client.upload_fileobj(tar_buffer, parsed.bucket, s3_key)
        print(f"Uploaded s3://{parsed.bucket}/{s3_key}", flush=True)
    else:
        # Save to local filesystem
        local_path = Path(output_dir) / "frames" / tar_filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(tar_buffer.getvalue())
        print(f"Saved {local_path}", flush=True)

    return tar_filename


def extract_unique_id(episode_path: str) -> str:
    """Extract a deterministic unique ID from the episode path."""
    if "diffusion_spartan" in episode_path:
        # For diffusion_spartan, use the datetime as unique id
        return episode_path.split("/")[-3]
    else:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, episode_path))


def save_and_upload_dict(dict_data: dict, output_path: str, file_name: str):
    # Used to upload manifest.jsonl and stats.json (or save locally)
    body = "\n".join(json.dumps(record) for record in dict_data) if "jsonl" in file_name else json.dumps(dict_data)

    if is_s3_path(output_path):
        # Upload to S3
        parsed = S3Path(s3_path=output_path)
        s3_key = f"{parsed.key.rstrip('/')}/{file_name}"
        create_s3_client().put_object(
            Bucket=parsed.bucket,
            Key=s3_key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Uploaded {file_name} to s3://{parsed.bucket}/{s3_key}")
    else:
        # Save to local filesystem
        local_path = Path(output_path) / file_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"Saved {file_name} to {local_path}")


def save_and_upload_config(config, output_path: str, file_name: str):
    # Draccus dump to temp file then upload to s3 (or save locally)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as temp_file:
        draccus.dump(config, temp_file)
        temp_path = temp_file.name

    if is_s3_path(output_path):
        # Upload to S3
        parsed = S3Path(s3_path=output_path)
        s3_key = f"{parsed.key.rstrip('/')}/{file_name}"
        create_s3_client().upload_file(temp_path, parsed.bucket, s3_key)
        print(f"Uploaded {file_name} to s3://{parsed.bucket}/{s3_key}")
    else:
        # Save to local filesystem
        local_path = Path(output_path) / file_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(temp_path, local_path)
        print(f"Saved {file_name} to {local_path}")


def _download_tar_from_s3(s3_key: str, s3_client, bucket_name: str, s3_prefix: str):
    """Download a single tar file from S3 with retry logic."""
    full_key = f"{s3_prefix.rstrip('/')}/frames/{s3_key}"
    obj_buffer = download_fileobj_from_s3(bucket_name, full_key, s3_client=s3_client)
    return (s3_key, obj_buffer)


@ray.remote
def create_episode_shard(shard_files: list[str], episode_key: str, output_dir: str) -> str:
    """Download/read tar files and create an episode-based shard. Supports both S3 and local filesystem."""
    is_s3 = is_s3_path(output_dir)

    if is_s3:
        s3_client = create_s3_client()
        parsed = S3Path(s3_path=output_dir)
        bucket_name, s3_prefix = parsed.bucket, parsed.key
        download_tar = partial(_download_tar_from_s3, s3_client=s3_client, bucket_name=bucket_name, s3_prefix=s3_prefix)
    else:
        frames_dir = Path(output_dir) / "frames"

        def download_tar(tar_key):
            """Read a single tar file from local filesystem."""
            tar_path = frames_dir / tar_key
            with open(tar_path, "rb") as f:
                obj_buffer = io.BytesIO(f.read())
            return (tar_key, obj_buffer)

    # Download/read all tars in parallel
    downloaded_tars = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_tar, s3_key) for s3_key in shard_files]
        for future in as_completed(futures):
            s3_key, obj_buffer = future.result()
            downloaded_tars[s3_key] = obj_buffer

    # Sort files by frame index to maintain temporal order within episode
    def get_frame_idx(filename):
        # filename format: {unique_id}_{episode_id}_frame_{frame_idx}.tar
        return int(filename.rsplit("_frame_", 1)[1].replace(".tar", ""))

    sorted_files = sorted(shard_files, key=get_frame_idx)

    # Create shard by combining all downloaded tars
    shard_buffer = io.BytesIO()
    with tarfile.open(fileobj=shard_buffer, mode="w") as shard_tar:
        for s3_key in sorted_files:
            obj_buffer = downloaded_tars[s3_key]
            obj_buffer.seek(0)

            with tarfile.open(fileobj=obj_buffer, mode="r") as tar:
                for member in tar.getmembers():
                    shard_tar.addfile(member, tar.extractfile(member))

    # Save shard
    shard_buffer.seek(0)
    shard_key = f"episode_{episode_key}.tar"

    if is_s3:
        s3_client.upload_fileobj(shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/episodes/{shard_key}")
        print(f"Uploaded episode shard {shard_key} to s3://{bucket_name}/{s3_prefix.rstrip('/')}/episodes/{shard_key}")
    else:
        episodes_dir = Path(output_dir) / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        shard_path = episodes_dir / shard_key
        with open(shard_path, "wb") as f:
            f.write(shard_buffer.getvalue())
        print(f"Saved episode shard {shard_key} to {shard_path}")

    return (shard_key.rstrip(".tar"), len(shard_files))


@ray.remote
def create_shard(shard_files: list[str], shard_idx: int, output_dir: str) -> str:
    """Download tar files from S3 and create a shard. OPTIMIZED with parallel downloads."""
    is_s3 = is_s3_path(output_dir)

    if is_s3:
        s3_client = create_s3_client()
        parsed = S3Path(s3_path=output_dir)
        bucket_name, s3_prefix = parsed.bucket, parsed.key

        def read_tar(tar_key):
            """Download a single tar file from S3."""
            obj_buffer = io.BytesIO()
            full_key = f"{s3_prefix.rstrip('/')}/frames/{tar_key}"
            s3_client.download_fileobj(bucket_name, full_key, obj_buffer)
            obj_buffer.seek(0)
            return (tar_key, obj_buffer)
    else:
        frames_dir = Path(output_dir) / "frames"

        def read_tar(tar_key):
            """Read a single tar file from local filesystem."""
            tar_path = frames_dir / tar_key
            with open(tar_path, "rb") as f:
                obj_buffer = io.BytesIO(f.read())
            return (tar_key, obj_buffer)

    # Read all tars in parallel (use 5 threads. reduced concurrency to avoid S3 throttling)
    downloaded_tars = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(read_tar, tar_key) for tar_key in shard_files]
        for future in as_completed(futures):
            tar_key, obj_buffer = future.result()
            downloaded_tars[tar_key] = obj_buffer

    # Create shard by combining all tars
    shard_buffer = io.BytesIO()
    with tarfile.open(fileobj=shard_buffer, mode="w") as shard_tar:
        # Process in original order for consistency
        for tar_key in shard_files:
            obj_buffer = downloaded_tars[tar_key]
            obj_buffer.seek(0)

            # Extract contents and add to shard
            with tarfile.open(fileobj=obj_buffer, mode="r") as tar:
                for member in tar.getmembers():
                    shard_tar.addfile(member, tar.extractfile(member))

    # Save shard
    shard_buffer.seek(0)
    shard_name = f"shard_{shard_idx:06d}.tar"

    if is_s3:
        upload_fileobj_to_s3(
            shard_buffer, bucket_name, f"{s3_prefix.rstrip('/')}/shards/{shard_name}", s3_client=s3_client
        )
        print(f"Uploaded shard {shard_name} to s3://{bucket_name}/{s3_prefix.rstrip('/')}/shards/{shard_name}")
    else:
        shard_path = Path(output_dir) / "shards" / shard_name
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(shard_path, "wb") as f:
            f.write(shard_buffer.getvalue())
        print(f"Saved shard {shard_name} to {shard_path}")

    return (shard_name.rstrip(".tar"), len(shard_files))


def is_still_sample(lowdim_data: dict[str, np.ndarray], start_idx: int, end_idx: int, still_threshold: float) -> bool:
    """Check if sample is still by looking at action/position/pose keys."""
    recognized_patterns = ["action", "joint", "poses", "xyz", "actual"]
    movement_keys = [k for k in lowdim_data if any(x in k.lower() for x in recognized_patterns)]

    # If no recognized keys found, we can't determine stillness; don't filter
    if not movement_keys:
        return False

    for key in movement_keys:
        data = lowdim_data[key][start_idx : end_idx + 1]
        if len(data) > 1:
            # max() handles multi-dimensional arrays (e.g. 7-DoF joints)
            movement = np.std(data, axis=0).max()
            if movement > still_threshold:
                return False
    return True


@ray.remote
def copy_s3_object(source_bucket: str, source_key: str, dest_bucket: str, dest_key: str) -> str:
    """Copy a single S3 object from source to destination."""
    s3_client = create_s3_client()
    copy_source = {"Bucket": source_bucket, "Key": source_key}
    s3_client.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)
    return dest_key


def recursive_s3_copy(path1: str, path2: str) -> None:
    """
    Recursively copy all objects from path1 to path2 using Ray for parallelization.
    """
    # Parse source and destination paths
    source = S3Path(s3_path=path1)
    dest = S3Path(s3_path=path2)
    source_bucket, source_prefix = source.bucket, source.key.rstrip("/") + "/"
    dest_bucket, dest_prefix = dest.bucket, dest.key.rstrip("/") + "/"

    relative_paths = list(list_s3_directory_recursive(path1))
    print(f"Found {len(relative_paths)} objects to copy")
    print(f"Starting parallel copy from {path1} to {path2}")

    # Build copy tasks: (source_bucket, source_key, dest_bucket, dest_key)
    copy_tasks = []
    for relative_path in relative_paths:
        source_key = source_prefix + relative_path
        dest_key = dest_prefix + relative_path
        copy_tasks.append((source_bucket, source_key, dest_bucket, dest_key))

    # Launch Ray tasks in parallel for copying
    futures = [
        copy_s3_object.remote(src_bucket, src_key, dst_bucket, dst_key)
        for src_bucket, src_key, dst_bucket, dst_key in copy_tasks
    ]
    copied_keys = ray.get(futures)
    print(f"✅ Successfully copied {len(copied_keys)} objects from {path1} to {path2}")


def validate_pose_groups(pose_groups: list[dict[str, str]]):
    """
    Validate that pose groups are complete with required fields.

    Each pose group must contain "name", "position_key", and "rotation_key"
    entries. Used during converter initialization to catch configuration
    errors before processing begins.

    Args:
        pose_groups: List of pose group dicts, each requiring keys "name",
            "position_key", and "rotation_key". An empty list is valid and
            returns immediately.

    Raises:
        TypeError: If pose_groups is not a list.
        ValueError: If any pose group is missing required fields. The error
            message lists all invalid groups and their missing fields.
    """
    if not pose_groups:
        return

    if not isinstance(pose_groups, list):
        raise TypeError("pose_groups must be a list of dicts")

    errors = []
    valid_pose_groups = []

    for i, pose_group in enumerate(pose_groups):
        group_errors = []

        # Check required fields
        if "name" not in pose_group:
            group_errors.append("missing 'name' field")
        if "position_key" not in pose_group:
            group_errors.append("missing 'position_key' field")
        if "rotation_key" not in pose_group:
            group_errors.append("missing 'rotation_key' field")

        if group_errors:
            errors.append(f"Pose group {i}: {', '.join(group_errors)}")
            continue

        # Pose group has all required fields
        valid_pose_groups.append(pose_group)
        position_key = pose_group["position_key"]
        rotation_key = pose_group["rotation_key"]
        print(f"  ✅ Valid pose group: {pose_group['name']} ({position_key}, {rotation_key})")

    if errors:
        error_msg = "Pose group validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)

    print(f"✅ All {len(valid_pose_groups)} pose groups are valid")
