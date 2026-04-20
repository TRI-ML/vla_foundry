import importlib
import importlib.util
import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import webdataset as wds


def _load_webdataset_cache_module():
    module_path = Path(__file__).resolve().parents[3] / "vla_foundry" / "data" / "pipelines" / "webdataset_cache.py"
    spec = importlib.util.spec_from_file_location("vla_foundry_test_webdataset_cache", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


webdataset_cache = _load_webdataset_cache_module()
cache_url_to_name = webdataset_cache.cache_url_to_name
cached_tarfile_to_samples = webdataset_cache.cached_tarfile_to_samples
get_tarfile_to_samples_stage = webdataset_cache.get_tarfile_to_samples_stage


def _make_tar_bytes(*entries):
    """Create tar bytes with the given (name, data) entries."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_cache_url_to_name_uses_full_url_identity():
    url_a = "s3://bucket-a/dataset-a/shards/shard_000001.tar"
    url_b = "s3://bucket-b/dataset-b/shards/shard_000001.tar"

    key_a = cache_url_to_name(url_a)
    key_b = cache_url_to_name(url_b)

    assert key_a != key_b
    assert "/" not in key_a
    assert "/" not in key_b
    assert key_a.endswith("shard_000001.tar")
    assert key_b.endswith("shard_000001.tar")


def test_cache_url_to_name_is_stable():
    url = "s3://bucket-a/dataset-a/shards/shard_000777.tar"

    assert cache_url_to_name(url) == cache_url_to_name(url)


def test_create_file_cache_disables_shell_validator(tmp_path):
    cache = webdataset_cache.create_file_cache(cache_dir=str(tmp_path), cache_size_gb=1)
    assert cache.validator is None


def test_file_cache_uses_local_copy_on_second_read(tmp_path, monkeypatch):
    calls = {"count": 0}

    def fake_s3_reader(url, mode="rb", bufsize=8192, **kwargs):
        calls["count"] += 1
        return io.BytesIO(b"fake shard bytes")

    wds_gopen = importlib.import_module("webdataset.gopen")

    monkeypatch.setitem(wds_gopen.gopen_schemes, "s3", fake_s3_reader)

    cache = wds.cache.FileCache(
        cache_dir=str(tmp_path),
        cache_size=int(1e9),
        url_to_name=cache_url_to_name,
        validator=None,
        verbose=False,
    )
    url = "s3://bucket-a/dataset-a/shards/shard_000001.tar"

    first = next(iter(cache([url])))
    first["stream"].read()
    first["stream"].close()

    second = next(iter(cache([url])))
    second["stream"].read()
    second["stream"].close()

    assert calls["count"] == 1


def test_cached_tarfile_to_samples_uses_cache_on_second_pass(tmp_path, monkeypatch):
    tar_bytes = _make_tar_bytes(("000001.txt", b"hello"))

    calls = {"count": 0}

    def fake_s3_reader(url, mode="rb", bufsize=8192, **kwargs):
        calls["count"] += 1
        return io.BytesIO(tar_bytes)

    wds_gopen = importlib.import_module("webdataset.gopen")
    monkeypatch.setitem(wds_gopen.gopen_schemes, "s3", fake_s3_reader)

    file_cache = wds.cache.FileCache(
        cache_dir=str(tmp_path),
        cache_size=int(1e9),
        url_to_name=cache_url_to_name,
        validator=None,
        verbose=False,
    )

    stage = cached_tarfile_to_samples(file_cache=file_cache)
    url = "s3://bucket-a/dataset-a/shards/shard_000001.tar"

    first = list(stage([{"url": url}]))
    second = list(stage([{"url": url}]))

    assert calls["count"] == 1
    assert first[0]["txt"] == b"hello"
    assert second[0]["txt"] == b"hello"


def test_cached_tarfile_to_samples_raises_without_file_cache():
    """cached_tarfile_samples must raise ValueError when file_cache is None."""
    stage = cached_tarfile_to_samples()
    with pytest.raises(ValueError, match="file_cache must be provided"):
        list(stage([{"url": "s3://bucket/shard.tar"}]))


def test_cached_tarfile_to_samples_closes_streams(tmp_path, monkeypatch):
    """Streams returned by FileCache must be closed after shard consumption."""
    tar_bytes = _make_tar_bytes(("000001.txt", b"data1"), ("000002.txt", b"data2"))

    def fake_s3_reader(url, mode="rb", bufsize=8192, **kwargs):
        return io.BytesIO(tar_bytes)

    wds_gopen = importlib.import_module("webdataset.gopen")
    monkeypatch.setitem(wds_gopen.gopen_schemes, "s3", fake_s3_reader)

    file_cache = wds.cache.FileCache(
        cache_dir=str(tmp_path),
        cache_size=int(1e9),
        url_to_name=cache_url_to_name,
        validator=None,
        verbose=False,
    )

    # Wrap file_cache to track that streams are closed after consumption.
    tracked_close_flags = []
    original_call = file_cache.__call__

    def tracking_file_cache(urls):
        for sample in original_call(urls):
            stream = sample.get("stream")
            if stream is not None:
                original_close = stream.close
                flag = {"closed": False}

                def make_tracking_close(orig, f):
                    def tracking_close():
                        f["closed"] = True
                        return orig()

                    return tracking_close

                stream.close = make_tracking_close(original_close, flag)
                tracked_close_flags.append(flag)
            yield sample

    stage = cached_tarfile_to_samples(file_cache=tracking_file_cache)

    urls = [
        "s3://bucket/shard_000001.tar",
        "s3://bucket/shard_000002.tar",
    ]
    samples = list(stage([{"url": u} for u in urls]))

    assert len(samples) > 0
    assert len(tracked_close_flags) > 0
    for flag in tracked_close_flags:
        assert flag["closed"], "Stream was not closed after shard consumption"


def test_get_tarfile_to_samples_stage_non_cache_closes_streams(tmp_path, monkeypatch):
    """When cache is disabled, get_tarfile_to_samples_stage should use
    tarfile_to_samples_closing which closes streams properly."""
    tar_bytes = _make_tar_bytes(("000001.txt", b"hello"))

    mock_streams = []

    def fake_gopen(url, *args, **kwargs):
        stream = MagicMock(wraps=io.BytesIO(tar_bytes))
        mock_streams.append(stream)
        return stream

    wds_gopen = importlib.import_module("webdataset.gopen")
    monkeypatch.setattr(wds_gopen, "gopen", fake_gopen)

    from types import SimpleNamespace

    cache_cfg = SimpleNamespace(enabled=False, cache_dir=None, cache_size_gb=None, cache_verbose=None)
    stage = get_tarfile_to_samples_stage(cache_cfg=cache_cfg)
    samples = list(stage.run([{"url": "s3://bucket/shard_000001.tar"}]))

    assert len(samples) > 0
    for stream in mock_streams:
        stream.close.assert_called()


def test_get_tarfile_to_samples_stage_cache_enabled(tmp_path, monkeypatch):
    """When cache is enabled, get_tarfile_to_samples_stage should use
    cached_tarfile_to_samples with a FileCache."""
    tar_bytes = _make_tar_bytes(("000001.txt", b"cached"))

    calls = {"count": 0}

    def fake_s3_reader(url, mode="rb", bufsize=8192, **kwargs):
        calls["count"] += 1
        return io.BytesIO(tar_bytes)

    wds_gopen = importlib.import_module("webdataset.gopen")
    monkeypatch.setitem(wds_gopen.gopen_schemes, "s3", fake_s3_reader)

    from types import SimpleNamespace

    cache_cfg = SimpleNamespace(enabled=True, cache_dir=str(tmp_path), cache_size_gb=1, cache_verbose=False)
    stage = get_tarfile_to_samples_stage(cache_cfg=cache_cfg)
    url = "s3://bucket/shard_000001.tar"

    first = list(stage([{"url": url}]))
    second = list(stage([{"url": url}]))

    # Only one remote fetch — second pass served from cache.
    assert calls["count"] == 1
    assert first[0]["txt"] == b"cached"
    assert second[0]["txt"] == b"cached"
