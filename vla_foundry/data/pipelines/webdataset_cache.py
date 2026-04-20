"""WebDataset file cache helpers."""

import os
from hashlib import blake2b
from urllib.parse import urlparse

import webdataset as wds
from webdataset.filters import pipelinefilter
from webdataset.handlers import reraise_exception
from webdataset.tariterators import group_by_keys

from vla_foundry.data.utils import _iter_tar_closing, tarfile_to_samples_closing

CACHE_DIR = os.environ.get("WDS_CACHE_DIR", "/tmp/wds_cache")
CACHE_SIZE_GB = int(os.environ.get("WDS_CACHE_SIZE_GB", "50"))
CACHE_VERBOSE = bool(int(os.environ.get("WDS_CACHE_VERBOSE", "0")))


def cache_url_to_name(url: str) -> str:
    """Map URL to a collision-safe cache filename."""
    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
        return wds.cache.url_to_cache_name(url)

    if parsed.scheme == "pipe":
        # Example: "aws s3 cp s3://bucket/path/shard_000001.tar -"
        basename = "shard"
        for token in parsed.path.split():
            if token.startswith("s3://"):
                basename = os.path.basename(urlparse(token).path.rstrip("/")) or "shard"
                break
    else:
        basename = os.path.basename(parsed.path.rstrip("/")) or "shard"

    digest = blake2b(url.encode("utf-8"), digest_size=12).hexdigest()
    return f"{digest}-{basename}"


def create_file_cache(
    cache_dir: str | None = None,
    cache_size_gb: int | None = None,
    cache_verbose: bool | None = None,
):
    """Create a WebDataset FileCache with optional overrides."""
    final_cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
    final_cache_size_gb = cache_size_gb if cache_size_gb is not None else CACHE_SIZE_GB
    final_cache_size_bytes = final_cache_size_gb * 1024 * 1024 * 1024
    final_cache_verbose = cache_verbose if cache_verbose is not None else CACHE_VERBOSE
    return wds.cache.FileCache(
        cache_dir=final_cache_dir,
        cache_size=final_cache_size_bytes,
        url_to_name=cache_url_to_name,
        # Avoid the default check_tar_format validator, which shells out to `file`.
        validator=None,
        verbose=final_cache_verbose,
    )


def cached_tarfile_samples(
    src,
    handler=reraise_exception,
    select_files=None,
    rename_files=None,
    file_cache=None,
):
    """Cache-backed tarfile_to_samples that closes each stream after reading.

    FileCache yields ``{"url": ..., "stream": ..., "local_path": ...}`` dicts.
    We feed these into ``_iter_tar_closing`` which handles tar iteration and
    closes each stream in a ``finally`` block to avoid leaking file descriptors.
    """
    if file_cache is None:
        raise ValueError("file_cache must be provided for cached_tarfile_samples")

    def _url_stream_pairs():
        for sample in file_cache(src):
            yield sample.get("url", ""), sample.get("stream")

    return group_by_keys(_iter_tar_closing(_url_stream_pairs(), handler), handler=handler)


cached_tarfile_to_samples = pipelinefilter(cached_tarfile_samples)


def get_tarfile_to_samples_stage(
    *,
    cache_cfg,
    handler=reraise_exception,
    select_files=None,
    rename_files=None,
):
    """Return either stream-closing or cache-backed tarfile-to-samples stage.

    Args:
        cache_cfg: A ``DatasetCacheParams`` instance (or any object with
            ``enabled``, ``cache_dir``, ``cache_size_gb``, ``cache_verbose``).
    """
    if not cache_cfg.enabled:
        return tarfile_to_samples_closing(handler=handler)
    file_cache = create_file_cache(
        cache_dir=cache_cfg.cache_dir,
        cache_size_gb=cache_cfg.cache_size_gb,
        cache_verbose=cache_cfg.cache_verbose,
    )
    return cached_tarfile_to_samples(
        handler=handler,
        select_files=select_files,
        rename_files=rename_files,
        file_cache=file_cache,
    )
