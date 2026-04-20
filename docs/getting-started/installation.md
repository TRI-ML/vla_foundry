# Installation

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended package manager)
- CUDA-compatible GPU (for training)

## Install with uv (Recommended)

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment management. Once uv is installed:

```bash
# Clone the repository
git clone https://github.com/TRI-ML/vla_foundry.git
cd vla_foundry

# Create environment and install dependencies
uv sync
uv pip install -e .
```

## Running Scripts

The recommended workflow is to run scripts directly with `uv`:

```bash
uv run <script> <args>
```

Alternatively, activate the virtual environment:

```bash
source .venv/bin/activate
```

!!! note
    Even when using the activated venv, prefer `uv` for package and dependency management.

## Optional Dependency Groups

VLA Foundry organizes optional dependencies into groups for specific workflows:

=== "Preprocessing"

    ```bash
    uv sync --group=preprocessing
    ```
    Required for data preprocessing scripts (Ray, img2dataset, etc.)

=== "SageMaker"

    ```bash
    uv sync --group=sagemaker
    ```
    Required for launching training jobs on AWS SageMaker.

=== "Inference"

    ```bash
    uv sync --group=inference
    ```
    Required for model inference and deployment.

## Verify Installation

Run the essential test suite to verify everything is working:

```bash
uv run pytest tests/essential -v
```

!!! tip
    If tests fail with Hugging Face errors, see the [FAQ](../faq.md) for troubleshooting HF token setup.

## AWS SSO Setup

If you need access to S3 datasets, configure AWS SSO. See the [FAQ](../faq.md#setting-up-aws-sso) for detailed instructions.
