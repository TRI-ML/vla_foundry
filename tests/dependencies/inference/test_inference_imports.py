"""Tests for the `inference` dependency group.

Verifies that the inference extras (robot-gym, which bundles grpc-workspace)
install and import cleanly, and that the installed protobuf version is
compatible with the generated .pb2 files shipped inside the robot-gym wheel.

These tests run only in CI after `uv sync --group inference`.
"""

import inspect
import re


def test_robot_gym_import():
    """robot-gym provides the policy server / gRPC interface used by lbm_eval."""
    import robot_gym

    assert robot_gym is not None


def test_inference_policy_import():
    """Main inference policy module should import with inference deps installed."""
    from vla_foundry.inference.robotics import inference_policy

    assert inference_policy is not None


def test_grpc_workspace_can_be_imported():
    """grpc_workspace is bundled inside the robot-gym wheel; all proto modules must import."""
    import grpc_workspace
    from grpc_workspace.lbm_policy_client import LbmPolicyClient
    from grpc_workspace.lbm_policy_server import LbmPolicyServer
    from grpc_workspace.proto import GetPolicyMetadata_pb2, PolicyReset_pb2, PolicyStep_pb2, health_pb2

    assert grpc_workspace is not None
    assert LbmPolicyClient is not None
    assert LbmPolicyServer is not None
    assert PolicyStep_pb2 is not None
    assert PolicyReset_pb2 is not None
    assert GetPolicyMetadata_pb2 is not None
    assert health_pb2 is not None


def _get_generated_protobuf_version():
    """Read the protobuf version that the installed .pb2 files were generated with."""
    from grpc_workspace.proto import PolicyStep_pb2

    src_path = inspect.getsourcefile(PolicyStep_pb2)
    if src_path is None:
        return None
    with open(src_path) as f:
        match = re.search(r"# Protobuf Python Version: (\d+\.\d+\.\d+)", f.read())
    return match.group(1) if match else None


def test_protobuf_version_compatible_with_generated_code():
    """Installed protobuf must be new enough for the generated .pb2 files shipped in the wheel."""
    import google.protobuf

    installed_major = int(google.protobuf.__version__.split(".")[0])
    generated_version = _get_generated_protobuf_version()
    assert generated_version is not None, "Could not determine protobuf version from generated files"
    generated_major = int(generated_version.split(".")[0])

    # Protobuf 5+ generated code imports `google.protobuf.runtime_version` which doesn't
    # exist in protobuf 4. Older generated code runs on any version.
    assert installed_major >= generated_major, (
        f"Protobuf version mismatch: generated .pb2 files expect >= {generated_version} "
        f"but installed version is {google.protobuf.__version__}."
    )
