"""Tests for the S3Path class (pure path manipulation only, no AWS calls)."""

from pathlib import Path

import pytest

from vla_foundry.aws.s3_path import S3Path


class TestS3PathConstruction:
    """Tests for constructing S3Path from various inputs."""

    def test_from_s3_path_string(self):
        p = S3Path(s3_path="s3://my-bucket/data/file.txt")
        assert p.bucket == "my-bucket"
        assert p.key == "data/file.txt"

    def test_from_bucket_and_key(self):
        p = S3Path(bucket="my-bucket", key="data/file.txt")
        assert p.bucket == "my-bucket"
        assert p.key == "data/file.txt"

    def test_from_bucket_only(self):
        p = S3Path(bucket="my-bucket")
        assert p.bucket == "my-bucket"
        assert p.key == ""

    def test_from_s3_path_root(self):
        p = S3Path(s3_path="s3://")
        assert p.bucket == ""
        assert p.key == ""
        assert str(p) == "s3://"

    def test_from_s3_path_bucket_only(self):
        p = S3Path(s3_path="s3://my-bucket")
        assert p.bucket == "my-bucket"
        assert p.key == ""

    def test_from_s3_path_with_trailing_slash(self):
        p = S3Path(s3_path="s3://my-bucket/data/")
        assert p.bucket == "my-bucket"
        assert p.key == "data/"

    def test_invalid_no_args(self):
        with pytest.raises(ValueError, match="Either s3_path or bucket must be non-None"):
            S3Path()

    def test_invalid_not_s3_prefix(self):
        with pytest.raises(ValueError, match="isn't an S3 path"):
            S3Path(s3_path="http://example.com/file.txt")

    def test_invalid_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            S3Path(s3_path="s3://bucket/key", bucket="bucket")

    def test_invalid_bucket_with_prefix(self):
        with pytest.raises(ValueError, match="bucket must NOT start with s3://"):
            S3Path(bucket="s3://my-bucket")


class TestS3PathProperties:
    """Tests for S3Path property accessors."""

    def test_name(self):
        p = S3Path(s3_path="s3://bucket/data/file.txt")
        assert p.name == "file.txt"

    def test_name_bucket_only(self):
        p = S3Path(s3_path="s3://bucket")
        assert p.name == "bucket"

    def test_name_root(self):
        p = S3Path(s3_path="s3://")
        assert p.name == ""

    def test_stem(self):
        p = S3Path(s3_path="s3://bucket/data/file.txt")
        assert p.stem == "file"

    def test_stem_no_extension(self):
        p = S3Path(s3_path="s3://bucket/data/README")
        assert p.stem == "README"

    def test_stem_multiple_dots(self):
        p = S3Path(s3_path="s3://bucket/archive.tar.gz")
        assert p.stem == "archive.tar"

    def test_suffix(self):
        p = S3Path(s3_path="s3://bucket/data/file.txt")
        assert p.suffix == ".txt"

    def test_suffix_no_extension(self):
        p = S3Path(s3_path="s3://bucket/data/README")
        assert p.suffix == ""

    def test_suffix_multiple_dots(self):
        p = S3Path(s3_path="s3://bucket/archive.tar.gz")
        assert p.suffix == ".gz"


class TestS3PathParent:
    """Tests for the parent property."""

    def test_parent_of_file(self):
        p = S3Path(s3_path="s3://bucket/data/file.txt")
        parent = p.parent
        assert str(parent) == "s3://bucket/data/"
        assert parent.bucket == "bucket"
        assert parent.key == "data/"

    def test_parent_of_directory(self):
        p = S3Path(s3_path="s3://bucket/data/subdir/")
        parent = p.parent
        assert str(parent) == "s3://bucket/data/"

    def test_parent_of_bucket(self):
        p = S3Path(s3_path="s3://bucket")
        parent = p.parent
        assert str(parent) == "s3://"

    def test_parent_of_root(self):
        p = S3Path(s3_path="s3://")
        parent = p.parent
        assert str(parent) == "s3://"

    def test_chained_parent(self):
        p = S3Path(s3_path="s3://bucket/a/b/c.txt")
        assert str(p.parent.parent) == "s3://bucket/a/"
        assert str(p.parent.parent.parent) == "s3://bucket/"


class TestS3PathStr:
    """Tests for string representation."""

    def test_str_full_path(self):
        p = S3Path(s3_path="s3://bucket/data/file.txt")
        assert str(p) == "s3://bucket/data/file.txt"

    def test_str_with_trailing_slash(self):
        p = S3Path(s3_path="s3://bucket/data/")
        assert str(p) == "s3://bucket/data/"

    def test_str_root(self):
        p = S3Path(s3_path="s3://")
        assert str(p) == "s3://"

    def test_str_bucket_only(self):
        p = S3Path(s3_path="s3://bucket")
        assert str(p) == "s3://bucket"

    def test_repr(self):
        p = S3Path(s3_path="s3://bucket/key.txt")
        assert repr(p) == 'S3Path("s3://bucket/key.txt")'


class TestS3PathTruediv:
    """Tests for the / operator."""

    def test_join_simple(self):
        p = S3Path(s3_path="s3://bucket/data")
        child = p / "file.txt"
        assert str(child) == "s3://bucket/data/file.txt"

    def test_join_nested(self):
        p = S3Path(s3_path="s3://bucket")
        child = p / "a/b/c.txt"
        assert str(child) == "s3://bucket/a/b/c.txt"

    def test_join_preserves_trailing_slash(self):
        p = S3Path(s3_path="s3://bucket")
        child = p / "subdir/"
        assert str(child) == "s3://bucket/subdir/"

    def test_join_chained(self):
        p = S3Path(s3_path="s3://bucket")
        result = p / "a" / "b" / "c.txt"
        assert str(result) == "s3://bucket/a/b/c.txt"


class TestS3PathEquality:
    """Tests for __eq__ and __hash__."""

    def test_equal_from_same_string(self):
        a = S3Path(s3_path="s3://bucket/key.txt")
        b = S3Path(s3_path="s3://bucket/key.txt")
        assert a == b

    def test_equal_from_different_constructors(self):
        a = S3Path(s3_path="s3://bucket/key.txt")
        b = S3Path(bucket="bucket", key="key.txt")
        assert a == b

    def test_not_equal_different_key(self):
        a = S3Path(s3_path="s3://bucket/a.txt")
        b = S3Path(s3_path="s3://bucket/b.txt")
        assert a != b

    def test_not_equal_trailing_slash(self):
        a = S3Path(s3_path="s3://bucket/dir")
        b = S3Path(s3_path="s3://bucket/dir/")
        assert a != b

    def test_not_equal_to_string(self):
        p = S3Path(s3_path="s3://bucket/key.txt")
        assert p != "s3://bucket/key.txt"

    def test_hash_equal_objects(self):
        a = S3Path(s3_path="s3://bucket/key.txt")
        b = S3Path(s3_path="s3://bucket/key.txt")
        assert hash(a) == hash(b)

    def test_usable_in_set(self):
        a = S3Path(s3_path="s3://bucket/a.txt")
        b = S3Path(s3_path="s3://bucket/a.txt")
        c = S3Path(s3_path="s3://bucket/b.txt")
        assert len({a, b, c}) == 2


class TestS3PathManipulation:
    """Tests for path manipulation methods."""

    def test_removeprefix(self):
        p = S3Path(s3_path="s3://bucket/data/file.txt")
        result = p.removeprefix()
        assert isinstance(result, Path)
        assert str(result) == "bucket/data/file.txt"

    def test_relative_to(self):
        base = S3Path(s3_path="s3://bucket/data")
        child = S3Path(s3_path="s3://bucket/data/subdir/file.txt")
        result = child.relative_to(base)
        assert isinstance(result, Path)
        assert str(result) == "subdir/file.txt"

    def test_relative_to_not_relative(self):
        a = S3Path(s3_path="s3://bucket-a/data")
        b = S3Path(s3_path="s3://bucket-b/data/file.txt")
        with pytest.raises(ValueError, match="is not relative to"):
            b.relative_to(a)

    def test_relative_to_child_shorter(self):
        parent = S3Path(s3_path="s3://bucket/a/b/c")
        child = S3Path(s3_path="s3://bucket/a")
        with pytest.raises(ValueError, match="is not relative to"):
            child.relative_to(parent)
