import json
import math

from vla_foundry.aws.s3_path import S3Path
from vla_foundry.aws.s3_utils import create_s3_client
from vla_foundry.data.preprocessing.hf_utils.hf_dataset_downloader import _download_and_upload_file
from vla_foundry.file_utils import list_s3_directory_recursive


def get_camera_names_from_s3(s3_client, bucket: str, prefix: str, chunk_num: int) -> list[str]:
    """Get camera names from the first chunk's video directory."""
    videos_prefix = f"{prefix}/videos/chunk-{chunk_num:03d}/"

    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=videos_prefix, Delimiter="/")

        camera_names = []
        for prefix_info in response.get("CommonPrefixes", []):
            folder_path = prefix_info["Prefix"]
            camera_name = folder_path.rstrip("/").split("/")[-1]
            camera_names.append(camera_name)

        return sorted(camera_names)
    except Exception as e:
        raise ValueError(f"Error getting camera names from {videos_prefix}: {e}") from e


def generate_expected_files(total_episodes: int, chunks_size: int, camera_names: list[str]):
    """Generate sets of expected data and video files."""
    expected_data_files, expected_video_files = set(), set()

    total_chunks = math.ceil(total_episodes / chunks_size)
    for chunk_idx in range(total_chunks):
        chunk_name = f"chunk-{chunk_idx:03d}"
        start_episode = chunk_idx * chunks_size
        end_episode = min((chunk_idx + 1) * chunks_size, total_episodes)

        # Generate expected data (parquet) files
        for episode_idx in range(start_episode, end_episode):
            data_file = f"data/{chunk_name}/episode_{episode_idx:06d}.parquet"
            expected_data_files.add(data_file)

        # Generate expected video (mp4) files for each camera
        for camera_name in camera_names:
            for episode_idx in range(start_episode, end_episode):
                video_file = f"videos/{chunk_name}/{camera_name}/episode_{episode_idx:06d}.mp4"
                expected_video_files.add(video_file)

    return expected_data_files, expected_video_files


def check_lerobot_complete(s3_path: str) -> list[str]:
    """
    Check if all files from a lerobot dataset are complete/downloaded correctly.
    Args:
        s3_path: S3 path to the dataset (e.g., "s3://your-bucket/your-path/hf_datasets/oxe_lerobot/toto_lerobot/")
    Returns:
        List of incomplete/missing files that were not downloaded correctly.
    """
    s3_client = create_s3_client()
    parsed = S3Path(s3_path=s3_path)
    bucket, prefix = parsed.bucket, parsed.key.rstrip("/")

    missing_files = []

    info_json_key = f"{prefix}/meta/info.json"
    try:
        response = s3_client.get_object(Bucket=bucket, Key=info_json_key)
        info_data = json.loads(response["Body"].read().decode("utf-8"))
    except Exception as e:
        raise FileNotFoundError(f"meta/info.json not found in {s3_path}") from e

    # Extract metadata
    total_episodes = info_data["total_episodes"]
    chunks_size = info_data.get("chunks_size", 1000)  # Default to 1000 if not specified
    camera_names = get_camera_names_from_s3(s3_client, bucket, prefix, 0)

    # Generate expected files and existing files. Pre-compute to avoid repetitive checks.
    expected_data_files, expected_video_files = generate_expected_files(total_episodes, chunks_size, camera_names)
    existing_objects = list_s3_directory_recursive(s3_path)

    # Check missing data (parquet) files
    for expected_file in expected_data_files:
        if expected_file not in existing_objects:
            missing_files.append(f"s3://{bucket}/{prefix}/{expected_file}")

    # Check missing video (mp4) files
    for expected_file in expected_video_files:
        if expected_file not in existing_objects:
            missing_files.append(f"s3://{bucket}/{prefix}/{expected_file}")

    return missing_files


def download_missing_files(
    missing_files: list[str],
    s3_path: str,
    hf_dataset: str,
    local_output_dir: str = "/tmp/lerobot_missing_files",
    preserve_structure: bool = True,
    max_retries: int = 20,
    backoff_factor: float = 3,
):
    """
    sample usage:
    download_missing_files(
        missing_files=check_lerobot_complete(f"s3://your-bucket/your-path/hf_datasets/droid_lerobot_fixed/"),
        s3_path="s3://your-bucket/your-path/hf_datasets/droid_lerobot_fixed/",
        hf_dataset="IPEC-COMMUNITY/droid_lerobot",
    )
    """
    parsed = S3Path(s3_path=s3_path)
    bucket, prefix = parsed.bucket, parsed.key
    for failed_file in missing_files:
        file_name = failed_file.removeprefix(s3_path).lstrip("/")
        url = f"https://huggingface.co/datasets/{hf_dataset}/resolve/main/{file_name}"
        _download_and_upload_file(
            url=url,
            relative_key=file_name,
            s3_bucket=bucket,
            s3_output_dir=prefix,
            local_output_dir=local_output_dir,
            preserve_structure=preserve_structure,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
