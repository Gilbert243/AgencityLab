#!/usr/bin/env bash
set -euo pipefail

# Build source and wheel distributions for AgencityLab.
# Expected tools:
# - python
# - build

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python -m pip install --upgrade build
python -m build
echo "Distribution artifacts generated in dist/"
