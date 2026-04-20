"""Tests for S3 I/O functions using moto."""

import io
import tempfile

import boto3
import pytest
from moto import mock_aws

from vla_foundry.aws.s3_io import (
    download_fileobj_from_s3,
    list_objects,
    list_objects_relative,
    put_object_to_s3,
    upload_file_to_s3,
    upload_fileobj_to_s3,
)

BUCKET = "test-bucket"
REGION = "us-east-1"


@pytest.fixture
def s3_client():
    """Create a moto-backed S3 client with a test bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        yield client


class TestS3IO:
    """Functional tests for each S3 I/O function."""

    def test_upload_fileobj_to_s3(self, s3_client):
        buf = io.BytesIO(b"hello world")
        upload_fileobj_to_s3(buf, BUCKET, "data/file.bin", s3_client=s3_client)

        resp = s3_client.get_object(Bucket=BUCKET, Key="data/file.bin")
        assert resp["Body"].read() == b"hello world"

    def test_download_fileobj_from_s3(self, s3_client):
        s3_client.put_object(Bucket=BUCKET, Key="data/file.bin", Body=b"file contents")

        result = download_fileobj_from_s3(BUCKET, "data/file.bin", s3_client=s3_client)
        assert result.read() == b"file contents"
        assert result.tell() == len(b"file contents")

    def test_upload_file_to_s3(self, s3_client):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"local file content")
            f.flush()
            upload_file_to_s3(f.name, BUCKET, "data/file.txt", s3_client=s3_client)

        resp = s3_client.get_object(Bucket=BUCKET, Key="data/file.txt")
        assert resp["Body"].read() == b"local file content"

    def test_put_object_to_s3(self, s3_client):
        put_object_to_s3("hello", BUCKET, "key.txt", content_type="text/plain", s3_client=s3_client)

        resp = s3_client.get_object(Bucket=BUCKET, Key="key.txt")
        assert resp["Body"].read() == b"hello"
        assert resp["ContentType"] == "text/plain"

    def test_list_objects(self, s3_client):
        for key in ["data/a.parquet", "data/b.txt", "data/c.parquet"]:
            s3_client.put_object(Bucket=BUCKET, Key=key, Body=b"x")

        result = list_objects(BUCKET, "data", suffix_filter=".parquet", s3_client=s3_client)
        assert sorted(result) == ["data/a.parquet", "data/c.parquet"]

    def test_list_objects_relative(self, s3_client):
        for key in ["data/a.txt", "data/subdir/b.txt"]:
            s3_client.put_object(Bucket=BUCKET, Key=key, Body=b"x")

        result = list_objects_relative(BUCKET, "data", s3_client=s3_client)
        assert sorted(result) == ["a.txt", "subdir/b.txt"]
