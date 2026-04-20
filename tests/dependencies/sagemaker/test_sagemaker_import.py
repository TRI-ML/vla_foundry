"""Tests for SageMaker import compatibility."""

from sagemaker.pytorch import PyTorch

import sagemaker


def test_sagemaker_import():
    """Test that sagemaker can be imported."""
    assert sagemaker is not None


def test_sagemaker_pytorch_estimator():
    """Test that PyTorch estimator can be imported."""
    assert PyTorch is not None
