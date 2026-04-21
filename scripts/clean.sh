#!/usr/bin/env bash
set -euo pipefail

# Remove common build and cache directories.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

rm -rf build dist .pytest_cache docs/_build .ruff_cache
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
echo "Cleanup complete."
