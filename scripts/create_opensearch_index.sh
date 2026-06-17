#!/usr/bin/env bash
# Wrapper called by Terraform null_resource.
# Uses the project venv when running locally; falls back to system python3 in CI
# (GitHub Actions installs requirements.txt globally before terraform apply).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -x "$SCRIPT_DIR/../.venv/bin/python3" ]]; then
  PYTHON="$SCRIPT_DIR/../.venv/bin/python3"
else
  PYTHON="python3"
fi

"$PYTHON" "$SCRIPT_DIR/create_opensearch_index.py"
