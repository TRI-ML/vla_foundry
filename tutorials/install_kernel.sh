#!/usr/bin/env bash
# Install a Jupyter kernel for VLA Foundry tutorials.
# Run from the repo root:  bash tutorials/install_kernel.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Sync the venv with required dependency groups
uv sync --group preprocessing --group tutorials --group dashboard --group visualization --quiet

# Register the kernel
.venv/bin/python -m ipykernel install --user --name vla_foundry --display-name "Python (vla_foundry)"

echo "Done. Select the 'Python (vla_foundry)' kernel in Jupyter."
