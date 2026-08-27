#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pinned upstream checkout supplying the TransReID ViT backbone that
# edge_tracker/reid_memory.py imports as `fastreid`. Not vendored into this repo.
FASTREID_REPO="https://github.com/JDAI-CV/fast-reid.git"
FASTREID_COMMIT="c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d"
FASTREID_DIR="$ROOT/edge_tracker/fast-reid"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "Then restart the shell and rerun this script." >&2
  exit 1
fi

sudo apt update
sudo apt install -y \
  python3-tk \
  libgl1 \
  libglib2.0-0 \
  mosquitto \
  mosquitto-clients \
  git

# uv creates each environment and installs from the committed uv.lock files, so
# both are reproducible. The CV worker resolves torch from NVIDIA's CUDA 12.8
# index (declared in edge_tracker/pyproject.toml) and MiVOLO from its pinned git
# commit, which were previously undocumented manual steps.
echo
echo "Syncing CV/GPU environment (.venv-cv-linux)..."
UV_PROJECT_ENVIRONMENT="$ROOT/.venv-cv-linux" uv sync --project "$ROOT/edge_tracker" --inexact

echo
echo "Syncing backend environment (backend/.venv-linux)..."
UV_PROJECT_ENVIRONMENT="$ROOT/backend/.venv-linux" uv sync --project "$ROOT/backend" --inexact

if [[ ! -d "$FASTREID_DIR/fastreid" ]]; then
  echo
  echo "Fetching fast-reid at pinned commit ${FASTREID_COMMIT:0:12}..."
  rm -rf "$FASTREID_DIR"
  git clone --quiet "$FASTREID_REPO" "$FASTREID_DIR"
  git -C "$FASTREID_DIR" checkout --quiet "$FASTREID_COMMIT"
else
  echo
  echo "fast-reid already present; leaving it untouched."
fi

NODE_MAJOR=0
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
fi
if ! command -v npm >/dev/null 2>&1 || [[ "$NODE_MAJOR" -lt 18 ]]; then
  NVM_DIRECTORY="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "$NVM_DIRECTORY/nvm.sh" ]]; then
    export NVM_DIR="$NVM_DIRECTORY"
    # shellcheck source=/dev/null
    source "$NVM_DIR/nvm.sh"
    nvm use 20 --silent >/dev/null
    NODE_MAJOR="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
  fi
fi
if ! command -v npm >/dev/null 2>&1 || [[ "$NODE_MAJOR" -lt 18 ]]; then
  echo "Node.js 20 is required before frontend setup. Install it with nvm, then rerun this script." >&2
  exit 1
fi

npm --prefix "$ROOT/frontend" install

echo
echo "Ubuntu dependencies installed."
"$ROOT/.venv-cv-linux/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
      f"available={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")
PY
echo
echo "Model weights (yolo26m.pt, yolo26n-cls.pt, transreid_msmt17.pth,"
echo "pose_landmarker_full.task, evacuation_mobilenet_v1.pth) are not fetched by this"
echo "script and must be present in edge_tracker/. start_ubuntu.sh verifies them."
echo
echo "Next: copy backend/.env.example to backend/.env, then run: bash start_ubuntu.sh"
