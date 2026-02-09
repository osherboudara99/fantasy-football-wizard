#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${1:-3.11}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first, then re-run this script." >&2
  exit 1
fi

uv venv --python "$PYTHON_VERSION"
# shellcheck source=/dev/null
source .venv/Scripts/activate
uv sync

echo "Environment ready. Activate with: source .venv/bin/activate"
