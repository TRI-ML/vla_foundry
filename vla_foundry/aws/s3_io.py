"""Low-level S3 I/O operations: upload, download, and listing with retry support."""

import io
import random
import time
from pathlib import Path

from botocore.client import BaseClient

from vla_foundry.aws.s3_utils import create_s3_client


def upload_fileobj_to_s3(
    buffer: io.BytesIO,
    bucket: str,
    key: str,
    s3_client: BaseClient | None = None,
    max_retries: int = 10,
) -> None:
    """Upload a BytesIO buffer to S3 with exponential backoff retry.

    Args:
        buffer: The in-memory buffer to upload. Seeks to start before each attempt.
        bucket: S3 bucket name.
        key: S3 object key.
        s3_client: Pre-existing S3 client, or None to create one via create_s3_client().
        max_retries: Maximum number of upload attempts.
    """
    if s3_client is None:
        s3_client = create_s3_client()

    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, bucket, key)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(f"S3 upload failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}")
            time.sleep(delay)


def download_fileobj_from_s3(
    bucket: str,
    key: str,
    s3_client: BaseClient | None = None,
    max_retries: int = 10,
) -> io.BytesIO:
    """Download an S3 object into a BytesIO buffer with exponential backoff retry.

    Args:
        bucket: S3 bucket name.
        key: S3 object key.
        s3_client: Pre-existing S3 client, or None to create one via create_s3_client().
        max_retries: Maximum number of download attempts.

    Returns:
        BytesIO buffer containing the downloaded object, seeked to position 0.
    """
    if s3_client is None:
        s3_client = create_s3_client()

    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            obj_buffer = io.BytesIO()
            s3_client.download_fileobj(bucket, key, obj_buffer)
            obj_buffer.seek(0)
            return obj_buffer
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(f"S3 download failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}")
            time.sleep(delay)


def upload_file_to_s3(
    local_path: str | Path,
    bucket: str,
    key: str,
    s3_client: BaseClient | None = None,
) -> None:
    """Upload a local file to S3.

    Args:
        local_path: Path to the local file to upload.
        bucket: S3 bucket name.
        key: S3 object key.
        s3_client: Pre-existing S3 client, or None to create one via create_s3_client().
    """
    if s3_client is None:
        s3_client = create_s3_client()
    s3_client.upload_file(str(local_path), bucket, key)


def put_object_to_s3(
    body: bytes | str,
    bucket: str,
    key: str,
    content_type: str | None = None,
    s3_client: BaseClient | None = None,
) -> None:
    """Upload bytes or a string directly to S3 via put_object.

    Args:
        body: The content to upload. Strings are encoded to UTF-8.
        bucket: S3 bucket name.
        key: S3 object key.
        content_type: MIME type (e.g. "application/json"), or None.
        s3_client: Pre-existing S3 client, or None to create one via create_s3_client().
    """
    if s3_client is None:
        s3_client = create_s3_client()
    if isinstance(body, str):
        body = body.encode("utf-8")

    kwargs = {"Bucket": bucket, "Key": key, "Body": body}
    if content_type:
        kwargs["ContentType"] = content_type
    s3_client.put_object(**kwargs)


def list_objects(
    bucket: str,
    prefix: str,
    s3_client: BaseClient | None = None,
    suffix_filter: str | None = None,
) -> list[str]:
    """List all S3 object keys under a prefix using pagination.

    Args:
        bucket: S3 bucket name.
        prefix: S3 key prefix to list under. A trailing "/" is appended if missing.
        s3_client: Pre-existing S3 client, or None to create one via create_s3_client().
        suffix_filter: If provided, only return keys ending with this suffix
            (e.g. ".parquet").

    Returns:
        List of full S3 keys matching the prefix (and optional suffix filter).
    """
    if s3_client is None:
        s3_client = create_s3_client()
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if suffix_filter is None or key.endswith(suffix_filter):
                keys.append(key)
    return keys


def list_objects_relative(
    bucket: str,
    prefix: str,
    s3_client: BaseClient | None = None,
) -> list[str]:
    """List all S3 objects under a prefix, returning paths relative to the prefix.

    Args:
        bucket: S3 bucket name.
        prefix: S3 key prefix. A trailing "/" is appended if missing.
        s3_client: Pre-existing S3 client, or None to create one via create_s3_client().

    Returns:
        List of relative key paths (with the prefix stripped).
    """
    if s3_client is None:
        s3_client = create_s3_client()
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    paginator = s3_client.get_paginator("list_objects_v2")
    relative_keys: list[str] = []

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative_key = key[len(prefix) :].lstrip("/")
                if relative_key:
                    relative_keys.append(relative_key)
    except Exception as e:
        print(f"Error listing s3://{bucket}/{prefix}: {e}")

    return relative_keys
