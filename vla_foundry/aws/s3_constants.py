"""S3 constants shared across the AWS utility module."""

from botocore.config import Config

S3_PREFIX = "s3://"
DEFAULT_REGION = "us-west-2"
DEFAULT_S3_CONFIG = Config(
    max_pool_connections=50,
    retries={"mode": "adaptive", "max_attempts": 10},
)
