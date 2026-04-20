"""
Pytest configuration for data validation tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests (may require external resources)")


def pytest_addoption(parser):
    """Add command line options for dataset configuration."""
    parser.addoption(
        "--dataset-path",
        action="store",
        default="tests/essential/test_assets/small_lbm_dataset/",
        help="Path to the dataset to validate (local path or S3 URL)",
    )
    parser.addoption(
        "--num-samples", action="store", type=int, default=5, help="Number of samples to validate from the dataset"
    )


@pytest.fixture
def dataset_config(request):
    """Fixture to provide dataset configuration from command line."""
    return {
        "path": request.config.getoption("--dataset-path"),
        "num_samples": request.config.getoption("--num-samples"),
    }
