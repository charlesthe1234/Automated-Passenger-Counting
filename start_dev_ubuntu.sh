#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PYTHON="$PROJECT_ROOT/backend/.venv-linux/bin/python"

if [[ ! -x "$BACKEND_PYTHON" ]]; then
  echo "Backend Ubuntu environment is missing. Run: bash setup_ubuntu.sh" >&2
  exit 1
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
  fi
fi
command -v npm >/dev/null 2>&1 || { echo "npm is missing; activate Node 20 first." >&2; exit 1; }
NODE_MAJOR="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
[[ "$NODE_MAJOR" -ge 18 ]] || { echo "Node.js 18 or newer is required; Node 20 is recommended." >&2; exit 1; }

cleanup() {
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
  wait "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(
  cd "$PROJECT_ROOT/backend"
  exec "$BACKEND_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8000
) &
BACKEND_PID=$!

(
  cd "$PROJECT_ROOT/frontend"
  exec npm run dev
) &
FRONTEND_PID=$!

echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop both development services."
wait
