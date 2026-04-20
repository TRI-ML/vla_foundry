"""S3 client factory and general AWS helpers."""

import os

import boto3
import botocore
from botocore.config import Config

from vla_foundry.aws.s3_constants import DEFAULT_REGION, DEFAULT_S3_CONFIG, S3_PREFIX


def create_s3_client(
    session: boto3.session.Session | None = None,
    config: Config | None = None,
    region: str = DEFAULT_REGION,
) -> botocore.client.BaseClient:
    """Create a boto3 S3 client with sensible defaults.

    Args:
        session: A boto3 session. If None, a new session is created.
        config: A botocore Config for the client. If None, uses DEFAULT_S3_CONFIG
            (50 pool connections, adaptive retries with 10 max attempts).
        region: AWS region name (only used when session is None).

    Returns:
        A boto3 S3 client.
    """
    if session is None:
        session = boto3.session.Session(region_name=region)
    if config is None:
        config = DEFAULT_S3_CONFIG
    return session.client("s3", config=config)


def is_s3_path(path: str) -> bool:
    """Check whether a path is an S3 URI."""
    return path.startswith(S3_PREFIX)


def get_aws_credentials_env() -> dict[str, str]:
    """Extract current AWS credentials as environment variables.

    Useful for forwarding credentials to Ray workers or subprocesses
    to avoid reliance on IMDS, which can be flaky on worker nodes.

    Returns:
        Dict of environment variable names to values. Only includes
        credentials that are present in the current session.

    Raises:
        RuntimeError: If the AWS SSO token is expired.
    """
    env_vars: dict[str, str] = {}

    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials:
            frozen = credentials.get_frozen_credentials()
            if frozen.access_key:
                env_vars["AWS_ACCESS_KEY_ID"] = frozen.access_key
            if frozen.secret_key:
                env_vars["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
            if frozen.token:
                env_vars["AWS_SESSION_TOKEN"] = frozen.token

            aws_profile = os.environ.get("AWS_PROFILE")
            if aws_profile:
                env_vars["AWS_PROFILE"] = aws_profile
    except botocore.exceptions.UnauthorizedSSOTokenError as e:
        raise RuntimeError(
            "AWS SSO token is expired, but this run requires S3 access. "
            "Please run `aws sso login` (with your profile) and retry."
        ) from e

    return env_vars
