"""
S3Path: pathlib-like class for S3 URIs with optional I/O.

Combines pure path manipulation (bucket/key parsing, parent traversal, path
joining) with S3 I/O operations (exists, upload, download, list). The S3
client is created lazily on first I/O call, so pure path operations incur
no network or credential overhead.

Usage Examples:
    Pure path manipulation (no client created):
        >>> path = S3Path(s3_path="s3://my-bucket/data/file.txt")
        >>> path.bucket   # "my-bucket"
        >>> path.key      # "data/file.txt"
        >>> path.name     # "file.txt"
        >>> child = path.parent / "other.txt"

    I/O operations (client created lazily on first use):
        >>> path = S3Path(s3_path="s3://bucket/data/file.tar")
        >>> path.exists()
        >>> path.download_to_buffer()

    Explicit client reuse (recommended in loops):
        >>> from vla_foundry.aws.s3_utils import create_s3_client
        >>> client = create_s3_client()
        >>> for name in ["a.tar", "b.tar"]:
        ...     p = S3Path(s3_path=f"s3://bucket/shards/{name}", s3_client=client)
        ...     p.upload_fileobj(buffer)
"""

import io
from pathlib import Path

import botocore
from botocore.client import BaseClient

from vla_foundry.aws.s3_constants import DEFAULT_REGION, S3_PREFIX
from vla_foundry.aws.s3_io import (
    download_fileobj_from_s3,
    list_objects,
    list_objects_relative,
    put_object_to_s3,
    upload_file_to_s3,
    upload_fileobj_to_s3,
)
from vla_foundry.aws.s3_utils import create_s3_client


class S3Path:
    """Pathlib-inspired S3 path with lazy I/O support.

    Path manipulation (bucket, key, parent, /, name, etc.) works without
    creating an S3 client. I/O methods (exists, upload, download, list)
    create a client lazily on first use, or you can pass one explicitly.

    Segments layout: [S3_PREFIX, bucket, key_part1, key_part2, ...]
    e.g. "s3://my-bucket/data/file.txt" -> ["s3://", "my-bucket", "data", "file.txt"]

    Args:
        s3_path: Complete S3 URI starting with "s3://".
        bucket: S3 bucket name (without "s3://" prefix).
        key: S3 object key (path within the bucket).
        s3_client: boto3 S3 client, or None for lazy creation on first I/O call.
        region: AWS region (only used when lazily creating a client).

    Examples:
        >>> path = S3Path(s3_path="s3://my-bucket/data/file.txt")
        >>> path = S3Path(bucket="my-bucket", key="data/file.txt")
        >>> client = create_s3_client()
        >>> path = S3Path(s3_path="s3://bucket/key.txt", s3_client=client)
    """

    def __init__(
        self,
        *,
        s3_path: str | None = None,
        bucket: str | None = None,
        key: str | None = None,
        s3_client: BaseClient | None = None,
        region: str = DEFAULT_REGION,
    ) -> None:
        self._client = s3_client
        self._region = region

        if s3_path is not None:
            if bucket is not None or key is not None:
                raise ValueError("s3_path and (bucket, key) are mutually exclusive.")
            if not s3_path.startswith(S3_PREFIX):
                raise ValueError(f"{s3_path} isn't an S3 path.")

            self._trailing_slash = s3_path.endswith("/")
            path_without_prefix = s3_path.removeprefix(S3_PREFIX)

            if not path_without_prefix or path_without_prefix == "/":
                self._segments: list[str] = [S3_PREFIX]
                self._trailing_slash = True
            else:
                path_without_prefix = path_without_prefix.rstrip("/")
                self._segments = [S3_PREFIX] + path_without_prefix.split("/")

        elif bucket is not None:
            if bucket.startswith(S3_PREFIX):
                raise ValueError("bucket must NOT start with s3://")

            self._segments = [S3_PREFIX, bucket]
            if key:
                self._trailing_slash = key.endswith("/")
                key_parts = key.rstrip("/").split("/")
                self._segments.extend(key_parts)
            else:
                self._trailing_slash = False
        else:
            raise ValueError("Either s3_path or bucket must be non-None.")

    # --- Properties ---

    @property
    def key(self) -> str:
        """The S3 object key (everything after the bucket name)."""
        if len(self._segments) <= 2:
            return ""
        key_parts = "/".join(self._segments[2:])
        if self._trailing_slash:
            key_parts += "/"
        return key_parts

    @property
    def bucket(self) -> str:
        """The S3 bucket name without the s3:// prefix."""
        return self._segments[1] if len(self._segments) > 1 else ""

    @property
    def parent(self) -> "S3Path":
        """The immediate parent directory path."""
        if len(self._segments) <= 2:
            result = S3Path(s3_path=S3_PREFIX)
        else:
            result = S3Path._from_segments(self._segments[:-1], trailing_slash=True)
        result._client = self._client
        result._region = self._region
        return result

    @property
    def name(self) -> str:
        """The final component of the path."""
        return self._segments[-1] if len(self._segments) > 1 else ""

    @property
    def stem(self) -> str:
        """The final component without its extension."""
        return self.name.rsplit(".", 1)[0]

    @property
    def suffix(self) -> str:
        """The file extension including the dot."""
        segments = self.name.rsplit(".", 1)
        return "." + segments[1] if len(segments) > 1 else ""

    @property
    def client(self) -> BaseClient:
        """The boto3 S3 client (created lazily if not provided)."""
        return self._get_client()

    # --- Internal helpers ---

    @classmethod
    def _from_segments(cls, segments: list[str], trailing_slash: bool = False) -> "S3Path":
        instance = cls.__new__(cls)
        instance._segments = segments
        instance._trailing_slash = trailing_slash
        instance._client = None
        instance._region = DEFAULT_REGION
        return instance

    def _get_client(self) -> BaseClient:
        """Return the S3 client, creating one lazily if needed."""
        if self._client is None:
            self._client = create_s3_client(region=self._region)
        return self._client

    # --- Dunder methods ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, S3Path):
            return NotImplemented
        return self._segments == other._segments and self._trailing_slash == other._trailing_slash

    def __hash__(self) -> int:
        return hash((tuple(self._segments), self._trailing_slash))

    def __truediv__(self, relative_path: str) -> "S3Path":
        trailing_slash = relative_path.endswith("/")
        rel_parts = relative_path.rstrip("/").split("/")
        new_segments = self._segments + rel_parts
        result = S3Path._from_segments(new_segments, trailing_slash)
        result._client = self._client
        result._region = self._region
        return result

    def __str__(self) -> str:
        if len(self._segments) == 1:
            return S3_PREFIX
        path = self._segments[0] + "/".join(self._segments[1:])
        if self._trailing_slash:
            path += "/"
        return path

    def __repr__(self) -> str:
        return f'S3Path("{str(self)}")'

    # --- Path manipulation ---

    def removeprefix(self) -> Path:
        """Remove the S3 prefix and return a Path."""
        return Path(*self._segments[1:])

    def relative_to(self, other: "S3Path") -> Path:
        """Compute the relative path from another S3 path."""
        if len(self._segments) < len(other._segments):
            raise ValueError(f"{str(self)} is not relative to {str(other)}")
        if self._segments[: len(other._segments)] != other._segments:
            raise ValueError(f"{str(self)} is not relative to {str(other)}")
        return Path(*self._segments[len(other._segments) :])

    # --- I/O: Existence checks ---

    def is_file(self) -> bool:
        """Check if this path points to an existing S3 object."""
        try:
            self._get_client().head_object(Bucket=self.bucket, Key=self.key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def is_dir(self) -> bool:
        """Check if this path is a non-empty S3 prefix (directory)."""
        prefix = self.key
        if not prefix.endswith("/"):
            prefix += "/"
        response = self._get_client().list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            MaxKeys=1,
        )
        return "Contents" in response

    def exists(self) -> bool:
        """Check if this path exists as either a file or directory."""
        return self.is_file() or self.is_dir()

    # --- I/O: File-based operations ---

    def download_file(self, save_dir: str | Path) -> Path:
        """Download this S3 file to a local directory.

        Args:
            save_dir: Local directory where the file will be saved.

        Returns:
            Absolute path to the downloaded file.

        Raises:
            FileNotFoundError: If the S3 file doesn't exist.
        """
        if not self.is_file():
            raise FileNotFoundError(f"S3 file doesn't exist: {str(self)}")

        save_dir = Path(save_dir).expanduser().resolve()
        local_path = save_dir / self.name

        local_path.parent.mkdir(exist_ok=True, parents=True)
        self._get_client().download_file(self.bucket, self.key, str(local_path))
        return local_path

    def upload_from(self, local_filepath: str | Path) -> None:
        """Upload a local file to this S3 path.

        If this path ends with "/", the local filename is appended.

        Args:
            local_filepath: Path to the local file.

        Raises:
            FileNotFoundError: If local_filepath doesn't exist.
        """
        local_filepath = Path(local_filepath)
        if not local_filepath.is_file():
            raise FileNotFoundError(f"{local_filepath} doesn't exist.")

        target = self / local_filepath.name if self._trailing_slash else self

        upload_file_to_s3(local_filepath, target.bucket, target.key, s3_client=self._get_client())

    # --- I/O: Buffer-based operations ---

    def upload_fileobj(self, buffer: io.BytesIO, max_retries: int = 10) -> None:
        """Upload a BytesIO buffer to this S3 path with retry.

        Args:
            buffer: In-memory buffer to upload.
            max_retries: Maximum upload attempts with exponential backoff.
        """
        upload_fileobj_to_s3(buffer, self.bucket, self.key, s3_client=self._get_client(), max_retries=max_retries)

    def download_to_buffer(self, max_retries: int = 10) -> io.BytesIO:
        """Download this S3 object into a BytesIO buffer with retry.

        Returns:
            BytesIO buffer seeked to position 0.
        """
        return download_fileobj_from_s3(self.bucket, self.key, s3_client=self._get_client(), max_retries=max_retries)

    def put_object(self, body: bytes | str, content_type: str | None = None) -> None:
        """Upload bytes or a string directly to this S3 path.

        Args:
            body: Content to upload. Strings are encoded to UTF-8.
            content_type: MIME type (e.g. "application/json"), or None.
        """
        put_object_to_s3(body, self.bucket, self.key, content_type=content_type, s3_client=self._get_client())

    # --- I/O: Listing ---

    def list_objects(self, suffix_filter: str | None = None) -> list[str]:
        """List all S3 object keys under this path as a prefix.

        Args:
            suffix_filter: Only return keys ending with this suffix.

        Returns:
            List of full S3 keys under this prefix.
        """
        return list_objects(self.bucket, self.key, s3_client=self._get_client(), suffix_filter=suffix_filter)

    def list_objects_relative(self) -> list[str]:
        """List objects under this prefix, returning paths relative to it."""
        return list_objects_relative(self.bucket, self.key, s3_client=self._get_client())
