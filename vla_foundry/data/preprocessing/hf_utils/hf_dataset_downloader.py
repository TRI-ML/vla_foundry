import argparse
import contextlib
import logging
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import ray
import requests
from huggingface_hub import HfApi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vla_foundry.aws.s3_path import S3Path
from vla_foundry.aws.s3_utils import create_s3_client
from vla_foundry.file_utils import file_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote
def download_and_upload_file(*args, **kwargs):
    # This makes it usable outside of Ray setups
    return _download_and_upload_file(*args, **kwargs)


def _download_and_upload_file(
    url: str,
    relative_key: str,
    s3_bucket: str | None,
    s3_output_dir: str | None,
    local_output_dir: str,
    preserve_structure: bool,
    max_retries: int,
    backoff_factor: float,
) -> tuple[bool, dict | None]:
    """
    Ray remote function to download a file and optionally upload to S3
    Returns (success, error_info)
    """

    # Setup local paths
    temp_dir = Path(local_output_dir)
    filename = os.path.basename(urlparse(url).path)
    local_rel_path = relative_key if preserve_structure else filename
    temp_path = temp_dir / local_rel_path
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    # If exists, skip download and upload
    if file_exists(str(temp_path)) or file_exists(f"s3://{s3_bucket}/{s3_output_dir}/{local_rel_path}"):
        print(f"Skipping download and upload for {local_rel_path} because it already exists")
        return True, None

    # Setup retry configuration
    retry_config = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_config)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "hf-dataset-downloader/1.0",
        "Accept": "*/*",
    }

    try:
        # Download file
        with session.get(url, stream=True, timeout=120, headers=headers) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)

        print(f"Downloaded {local_rel_path}")

        # Upload to S3 if configured
        if s3_bucket is not None:
            s3_client = create_s3_client()
            s3_rel_key = relative_key if preserve_structure else filename
            s3_key = f"{s3_output_dir}/{s3_rel_key}"
            s3_client.upload_file(str(temp_path), s3_bucket, s3_key)
            print(f"Uploaded {s3_key} to S3")

            # Clean up local file after S3 upload
            with contextlib.suppress(OSError):
                os.remove(temp_path)

        return True, None

    except requests.HTTPError as http_error:
        error_info = {"url": url, "filename": local_rel_path, "error": str(http_error)}
        print(f"HTTP error downloading {url}: {http_error}")
        return False, error_info
    except requests.RequestException as req_error:
        error_info = {"url": url, "filename": local_rel_path, "error": str(req_error)}
        print(f"Network error downloading {url}: {req_error}")
        return False, error_info
    except Exception as unknown_error:
        error_info = {"url": url, "filename": local_rel_path, "error": str(unknown_error)}
        print(f"Unexpected error downloading {url}: {unknown_error}")
        return False, error_info


class DatasetDownloader:
    def __init__(
        self,
        s3_bucket: str = None,
        s3_output_dir: str = None,
        local_output_dir: str = None,
        preserve_structure: bool = False,
        max_retries: int = 10,
        backoff_factor: float = 3,
    ):
        """
        Initialize the downloader with S3 credentials and bucket
        """
        self.local_output_dir = local_output_dir
        self.s3_bucket = s3_bucket
        self.s3_output_dir = s3_output_dir
        self.preserve_structure = preserve_structure
        self.max_retries = max_retries
        self.temp_dir = Path(local_output_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.hf_api = HfApi()
        self.failed_files = []
        self.backoff_factor = backoff_factor

    def get_dataset_files(self, dataset_id: str, revision: str = "main", subfolder: str = "") -> list[tuple[str, str]]:
        """
        Get list of file URLs from a Hugging Face dataset

        Args:
            dataset_id: The Hugging Face dataset ID (e.g., 'mozilla-foundation/common_voice_11_0')
            revision: The git revision to use (default: "main")
            subfolder: Specific subfolder in the dataset to download from (optional)
        Returns:
            List of tuples: (download_url, relative_path_inside_dataset or relative to subfolder)
        """
        # List all files in the dataset
        files = self.hf_api.list_repo_files(repo_id=dataset_id, repo_type="dataset", revision=revision)
        if subfolder:
            files = [f for f in files if f.startswith(subfolder.rstrip("/"))]

        # Generate download URLs and relative keys
        urls: list[tuple[str, str]] = []
        base_prefix = subfolder.strip("/")
        for file_path in files:
            url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{file_path}"
            if base_prefix and file_path.startswith(base_prefix + "/"):
                relative_key = file_path[len(base_prefix) + 1 :]
            elif base_prefix and file_path == base_prefix:
                # File directly equals the subfolder name (rare, but handle)
                relative_key = os.path.basename(file_path)
            else:
                relative_key = file_path
            urls.append((url, relative_key))
        logger.info(f"Found {len(urls)} files in dataset {dataset_id}")
        return urls

    def log_failed_files(self, dataset_id: str) -> None:
        """Log failed files to a timestamped log file"""
        if not self.failed_files:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_safe = dataset_id.replace("/", "_")
        failed_log_path = self.temp_dir / f"failed_files_{dataset_safe}_{timestamp}.log"

        with open(failed_log_path, "w") as f:
            f.write(f"Failed files for dataset: {dataset_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total failed: {len(self.failed_files)}\n\n")

            for failed_file in self.failed_files:
                f.write(f"URL: {failed_file['url']}\n")
                f.write(f"Filename: {failed_file['filename']}\n")
                f.write(f"Error: {failed_file['error']}\n")
                f.write("-" * 80 + "\n")

        logger.info(f"Failed files logged to: {failed_log_path}")

    def download_dataset(self, dataset_id: str, revision: str = "main", subfolder: str = "") -> None:
        """
        Download an entire dataset in parallel and upload to S3
        """
        start_time = datetime.now()
        logger.info(
            f"Downloading dataset {dataset_id} (subfolder={subfolder}, preserve_structure={self.preserve_structure})"
        )
        # Get dataset URLs
        urls = self.get_dataset_files(dataset_id, revision, subfolder)

        # Download and upload files in parallel using Ray
        futures = []
        for url, relative_key in urls:
            future = download_and_upload_file.remote(
                url=url,
                relative_key=relative_key,
                s3_bucket=self.s3_bucket,
                s3_output_dir=self.s3_output_dir,
                local_output_dir=self.local_output_dir,
                preserve_structure=self.preserve_structure,
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
            )
            futures.append(future)

        results = ray.get(futures)

        # Process results and collect failed files
        successful = 0
        for success, error_info in results:
            if success:
                successful += 1
            elif error_info:
                self.failed_files.append(error_info)

        failed = len(urls) - successful
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Completed processing dataset {dataset_id}")
        logger.info(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Successfully processed {successful}/{len(urls)} files")
        logger.info(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Failed files: {failed}")
        logger.info(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Total duration: {duration}")

        # Log failed files to file if any
        if failed > 0:
            self.log_failed_files(dataset_id)

        # Cleanup temporary directory only in S3 mode
        if self.s3_bucket is not None and self.temp_dir.exists():
            with contextlib.suppress(OSError):
                self.temp_dir.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="default", help="AWS Profile")
    parser.add_argument("--dataset", type=str, required=True, help="Name of HF dataset to download")
    parser.add_argument("--mode", type=str, required=True, choices=["s3", "local"])

    # Path args
    parser.add_argument("--s3-output-path", type=str, help="S3 path in format 's3://bucket/path/to/dir'")
    parser.add_argument("--local-output-dir", type=str, required=True, help="Local path to save outputs to")
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help=(
            "If set, preserve the source dataset's internal folder structure when writing output folders."
            " If a subfolder is specified, the preserved structure is relative to that subfolder."
        ),
    )
    parser.add_argument("--subfolder", type=str, default="", help="Optional dataset subfolder to download from")

    # Downloader args
    parser.add_argument("--max-retries", type=int, default=20, help="Number of retries for failed downloads")
    parser.add_argument("--backoff-factor", type=float, default=3, help="Backoff factor for failed downloads")

    # Ray args
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address (default: auto)")
    parser.add_argument("--ray-num-cpus", type=int, default=None, help="Number of CPUs for Ray (default: auto-detect)")

    args = parser.parse_args()

    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
        logger.info(f"Connected to Ray cluster at {args.ray_address}")
    else:
        ray.init(address="auto", num_cpus=args.ray_num_cpus)
        logger.info(f"Started auto Ray cluster with num_cpus={args.ray_num_cpus}")

    if args.mode == "s3":
        assert args.s3_output_path is not None

        s3_path = S3Path(s3_path=args.s3_output_path)
        s3_bucket, s3_output_dir = s3_path.bucket, s3_path.key
        downloader = DatasetDownloader(
            s3_bucket=s3_bucket,
            s3_output_dir=s3_output_dir,
            local_output_dir=args.local_output_dir,
            preserve_structure=args.preserve_structure,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
        )
    elif args.mode == "local":
        downloader = DatasetDownloader(
            local_output_dir=args.local_output_dir,
            preserve_structure=args.preserve_structure,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
        )
    downloader.download_dataset(dataset_id=args.dataset, revision="main", subfolder=args.subfolder)

    ray.shutdown()

    # Retry failed downloads
    logger.info(f"Retrying {len(downloader.failed_files)} failed downloads")
    if downloader.failed_files:
        if args.mode == "s3":
            for failed_file in downloader.failed_files:
                _download_and_upload_file(
                    url=failed_file["url"],
                    relative_key=failed_file["filename"],
                    s3_bucket=downloader.s3_bucket,
                    s3_output_dir=downloader.s3_output_dir,
                    local_output_dir=downloader.local_output_dir,
                    preserve_structure=downloader.preserve_structure,
                    max_retries=downloader.max_retries,
                    backoff_factor=downloader.backoff_factor,
                )
        else:
            raise ValueError("Local mode not supported for retrying failed downloads")
