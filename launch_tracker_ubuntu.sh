#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv-cv-linux/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Tracker Ubuntu environment is missing. Run: bash setup_ubuntu.sh"
  exit 1
fi

cd "$ROOT/edge_tracker"
exec "$PYTHON" launcher_ui.py
