#!/usr/bin/env bash
set -euo pipefail

# Run the AgencityLab test suite.
# Expected tools:
# - python
# - pytest

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python -m pytest tests
