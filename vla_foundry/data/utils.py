import contextlib
import hashlib
import importlib
import logging
import random
import subprocess
import traceback
from collections.abc import Iterable, Sequence
from functools import lru_cache
from multiprocessing import Value
from urllib.parse import urlparse

import boto3
import webdataset as wds
from botocore.config import Config
from torch.utils.data import get_worker_info

from vla_foundry.file_utils import load_dataset_manifest, pt_load


def _parse_s3_url(url: str) -> tuple[str, str]:
    """Parse s3://bucket/key URL into (bucket, key)."""
    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URL: {url}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URL is missing object key: {url}")
    return parsed.netloc, key


@lru_cache(maxsize=1)
def _get_s3_client():
    """Reuse a single boto3 S3 client per process."""
    return boto3.client(
        "s3",
        config=Config(
            retries={"max_attempts": 5, "mode": "adaptive"},
            read_timeout=120,
        ),
    )


def _gopen_s3_boto(url, mode="rb", bufsize=8192, **kwargs):  # noqa: ARG001
    """WebDataset gopen handler for s3:// URLs backed by boto3."""
    if mode != "rb":
        raise ValueError(f"Unsupported mode for s3 gopen: {mode}")
    bucket, key = _parse_s3_url(url)
    response = _get_s3_client().get_object(Bucket=bucket, Key=key)
    return response["Body"]


def _install_webdataset_s3_gopen() -> None:
    """Register boto3-backed `s3://` reader with webdataset.gopen."""
    wds_gopen = importlib.import_module("webdataset.gopen")
    if getattr(wds_gopen, "_lbm_s3_gopen_patch", False):
        return
    wds_gopen.gopen_schemes["s3"] = _gopen_s3_boto
    wds_gopen._lbm_s3_gopen_patch = True


_install_webdataset_s3_gopen()


def _install_webdataset_patches() -> None:
    """Patch webdataset to handle broken pipes, fd leaks, and close streams."""

    wds_gopen = importlib.import_module("webdataset.gopen")

    if getattr(wds_gopen.Pipe, "_lbm_patched", False):
        return

    # --- Patch 1: ignore benign AWS broken-pipe exit codes + fix fd leak. ---
    original_init = wds_gopen.Pipe.__init__

    def patched_init(self, *args, ignore_status=None, **kwargs):  # type: ignore[override]
        # Pre-initialise so __del__/close never hit AttributeError.
        self.stream = None
        self.proc = None
        self.status = None

        ignore = list(ignore_status or [])
        if 1 not in ignore:
            ignore.append(1)
        cmd = args[0] if args else None
        if isinstance(cmd, str) and "aws s3 cp" in cmd and "stderr" not in kwargs:
            kwargs["stderr"] = subprocess.DEVNULL
        original_init(self, *args, ignore_status=ignore, **kwargs)

    wds_gopen.Pipe.__init__ = patched_init  # type: ignore[assignment]

    # --- Patch 2: make close() safe when __init__ failed partway. ---
    original_close = wds_gopen.Pipe.close

    def patched_close(self):
        if self.stream is None:
            # __init__ never finished — kill the process if it exists.
            if self.proc is not None:
                try:
                    self.proc.kill()
                    self.proc.wait()
                except Exception:
                    pass
            return
        original_close(self)

    wds_gopen.Pipe.close = patched_close

    wds_gopen.Pipe._lbm_patched = True


_install_webdataset_patches()


def _iter_tar_closing(url_stream_pairs, handler):
    """Iterate tar entries from (url, stream) pairs, closing each stream in a finally block.

    Yields individual file entries and empty-dict shard boundary markers suitable
    for consumption by ``group_by_keys``.
    """
    from webdataset.tariterators import tar_file_iterator

    for url, stream in url_stream_pairs:
        try:
            for s in tar_file_iterator(stream, handler=handler):
                if not (isinstance(s, dict) and "data" in s and "fname" in s):
                    raise ValueError(
                        f"Unexpected sample format from tar_file_iterator: "
                        f"type={type(s)}, keys={list(s.keys()) if isinstance(s, dict) else 'N/A'}"
                    )
                s["__url__"] = url
                yield s
            # Shard boundary marker (consumed by group_by_keys).
            yield {}
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()


class tarfile_to_samples_closing(wds.PipelineStage):
    """Drop-in replacement for wds.tarfile_to_samples that closes each shard stream after reading.

    The standard wds.tarfile_to_samples never explicitly closes Pipe streams,
    relying on GC / __del__.  When iterating over thousands of S3 shards the
    file-descriptor table fills up before the garbage collector runs.  This
    version closes each stream in a ``finally`` block immediately after the
    shard has been consumed.
    """

    def __init__(self, handler=None):
        self.handler = handler if handler is not None else log_and_continue

    def run(self, src):
        from webdataset.gopen import gopen
        from webdataset.tariterators import group_by_keys

        def _url_stream_pairs():
            for sample in src:
                url = sample["url"]
                yield url, gopen(url)

        return group_by_keys(_iter_tar_closing(_url_stream_pairs(), self.handler), handler=self.handler)


class SharedCheckpointCounter:
    """
    A process-safe counter that can be shared across dataloader workers.
    """

    def __init__(self, checkpoint_num: int = 0):
        """
        Args:
            checkpoint_num: Initial value for the counter.
        """
        self.shared_checkpoint_num = Value("i", checkpoint_num)

    def set_value(self, checkpoint_num: int):
        """Set the shared counter to a specific value."""
        self.shared_checkpoint_num.value = checkpoint_num

    def get_value(self):
        """Get the current value of the shared counter."""
        return self.shared_checkpoint_num.value


def log_and_continue(exn: BaseException) -> bool:
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    tb_str = "".join(traceback.format_tb(exn.__traceback__))
    logging.warning(f"Handling webdataset error ({repr(exn)}):\n{tb_str}Ignoring.")
    return True


def pytorch_worker_seed(increment: int = 0) -> int:
    """Get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # Favor using the worker's seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class deterministic_shuffle(wds.PipelineStage):
    """
    Deterministic shuffling stage for WebDataset pipelines.
    If `epoch` is an int, it is incremented locally each time
    `run()` is invoked, which may diverge across workers in multi-process
    settings. To keep workers aligned, pass a `SharedCheckpointCounter`.
    """

    def __init__(
        self,
        bufsize: int = 1000,
        initial: int = 100,
        seed: int = 0,
        epoch: int | SharedCheckpointCounter = -1,
    ) -> None:
        """
        Args:
            bufsize (int): Buffer size for shuffling.
            initial (int): Initial buffer size before yielding.
            seed: Seed for the random number generator.
            epoch: Epoch number.
        """
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src: Iterable) -> Iterable:
        """Yield items from `src` in a deterministic, buffered-shuffled order."""
        if isinstance(self.epoch, SharedCheckpointCounter):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
        # Otherwise, we use the seed + epoch to be deterministic AND the same across all nodes/workers in each epoch
        seed = pytorch_worker_seed(epoch) if self.seed < 0 else self.seed + epoch
        rng.seed(seed)
        return wds.filters._shuffle(src, self.bufsize, self.initial, rng)


def load_data_chunks(resume_from_checkpoint: str) -> tuple[list[int], int]:
    """
    Load dataloader cursor state from a checkpoint path.

    Args:
        resume_from_checkpoint: Path to a checkpoint file created by the
        training loop.
    """
    checkpoint = pt_load(resume_from_checkpoint, map_location="cpu")
    return checkpoint["curr_shard_idx_per_dataset"], checkpoint["samples_seen"]


def epochs_to_samples(manifest_paths: Sequence[str], num_epochs: int) -> int:
    """
    Compute total samples as `num_epochs * sum(num_sequences in manifests)`.

    Args:
        manifest_paths: Sequence of manifest file paths/URIs.
        num_epochs: Number of epochs to iterate over the combined dataset.
    """
    manifests = [load_dataset_manifest(path) for path in manifest_paths]
    num_samples = 0
    for m in manifests:
        num_samples += sum(i["num_sequences"] for i in m)
    return num_samples * num_epochs


def text_to_seed(text: str) -> int:
    """Convert an arbitrary string to a stable 32-bit seed via SHA-256."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32 - 1)
