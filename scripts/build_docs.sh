#!/usr/bin/env bash
set -euo pipefail

# Build AgencityLab documentation.
# Expected tools:
# - python
# - sphinx-build (installed through the docs extra)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v sphinx-build >/dev/null 2>&1; then
  echo "sphinx-build is not available. Install the docs extras first."
  exit 1
fi

rm -rf docs/_build
sphinx-build -b html docs docs/_build/html
echo "Documentation generated in docs/_build/html"
