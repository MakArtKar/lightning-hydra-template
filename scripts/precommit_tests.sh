#!/usr/bin/env bash
set -euo pipefail

# Prefer project .venv if present
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[precommit-tests] Running quick tests (not slow) with xdist..."
pytest -n auto -m "not slow" -q
