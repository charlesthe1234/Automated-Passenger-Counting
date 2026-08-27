#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PYTHON="$PROJECT_ROOT/backend/.venv-linux/bin/python"
CV_PYTHON="$PROJECT_ROOT/.venv-cv-linux/bin/python"
BACKEND_ENV="$PROJECT_ROOT/backend/.env"
FRONTEND_INDEX="$PROJECT_ROOT/frontend/dist/index.html"
FRONTEND_BUILD_STAMP="$PROJECT_ROOT/frontend/dist/.cag-production-build"
MOSQUITTO_CONFIG="$PROJECT_ROOT/mosquitto_server.conf"
SERVICE_LOG_DIR="$PROJECT_ROOT/LogEvidance"
MOSQUITTO_LOG="$SERVICE_LOG_DIR/mosquitto-server.log"
BACKEND_PID=""
BROKER_PID=""

fail() {
  echo "Startup failed: $*" >&2
  exit 1
}

port_listening() {
  local port_number="$1"
  [[ -n "$(ss -ltnH "( sport = :$port_number )" 2>/dev/null)" ]]
}

cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BROKER_PID" ]] && kill -0 "$BROKER_PID" 2>/dev/null; then
    kill "$BROKER_PID" 2>/dev/null || true
    wait "$BROKER_PID" 2>/dev/null || true
  fi
  exit "$exit_status"
}
trap cleanup EXIT INT TERM HUP

[[ -x "$BACKEND_PYTHON" ]] || fail "Backend environment missing. Run bash setup_ubuntu.sh"
[[ -x "$CV_PYTHON" ]] || fail "CV environment missing. Run bash setup_ubuntu.sh"
[[ -f "$BACKEND_ENV" ]] || fail "backend/.env is missing. Copy backend/.env.example and configure it."
[[ -f "$MOSQUITTO_CONFIG" ]] || fail "mosquitto_server.conf is missing."
[[ -f "$PROJECT_ROOT/edge_tracker/cv_worker.py" ]] || fail "edge_tracker/cv_worker.py is missing."
[[ -f "$PROJECT_ROOT/edge_tracker/yolo26m.pt" ]] || fail "YOLO model is missing."
[[ -f "$PROJECT_ROOT/edge_tracker/yolo26n-cls.pt" ]] || fail "BoT-SORT ReID model is missing."
[[ -f "$PROJECT_ROOT/edge_tracker/transreid_msmt17.pth" ]] || fail "TransReID checkpoint is missing."
[[ -f "$PROJECT_ROOT/edge_tracker/pose_landmarker_full.task" ]] || fail "MediaPipe model is missing."
[[ -f "$PROJECT_ROOT/edge_tracker/evacuation_mobilenet_v1.pth" ]] || fail "Role-classifier checkpoint is missing."
command -v ss >/dev/null 2>&1 || fail "The ss utility is required (package: iproute2)."
command -v curl >/dev/null 2>&1 || fail "curl is required."

(
  cd "$PROJECT_ROOT/backend"
  "$BACKEND_PYTHON" -c "from config import settings; assert settings.camera_source_map, 'No camera sources are configured'"
) || fail "Camera configuration is invalid. Set CAMERA_URLS in backend/.env."

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
command -v node >/dev/null 2>&1 || fail "Node.js is missing. Install/use Node 20 with nvm."
command -v npm >/dev/null 2>&1 || fail "npm is missing. Install/use Node 20 with nvm."
NODE_MAJOR="$(node --version | sed -E 's/^v([0-9]+).*/\1/')"
[[ "$NODE_MAJOR" -ge 18 ]] || fail "Node.js 18 or newer is required; Node 20 is recommended."

mkdir -p "$SERVICE_LOG_DIR"

if port_listening 8000; then
  fail "Port 8000 is already in use. Stop the existing backend before running this command."
fi

if port_listening 1883; then
  echo "MQTT broker: existing listener on port 1883"
else
  command -v mosquitto >/dev/null 2>&1 || fail "Mosquitto is not installed."
  mosquitto -c "$MOSQUITTO_CONFIG" >"$MOSQUITTO_LOG" 2>&1 &
  BROKER_PID=$!
  for _attempt in $(seq 1 50); do
    if port_listening 1883; then
      break
    fi
    if ! kill -0 "$BROKER_PID" 2>/dev/null; then
      fail "Mosquitto exited during startup. See $MOSQUITTO_LOG"
    fi
    sleep 0.1
  done
  port_listening 1883 || fail "Mosquitto did not open port 1883. See $MOSQUITTO_LOG"
  echo "MQTT broker: started by this command (PID $BROKER_PID)"
fi

FRONTEND_STALE=false
if [[ ! -f "$FRONTEND_INDEX" ]]; then
  FRONTEND_STALE=true
elif [[ ! -f "$FRONTEND_BUILD_STAMP" ]] || [[ "$(<"$FRONTEND_BUILD_STAMP")" != "same-origin-v1" ]]; then
  FRONTEND_STALE=true
elif [[ "$PROJECT_ROOT/frontend/package.json" -nt "$FRONTEND_INDEX" ]] || \
     [[ "$PROJECT_ROOT/frontend/package-lock.json" -nt "$FRONTEND_INDEX" ]] || \
     [[ -n "$(find "$PROJECT_ROOT/frontend/src" -type f -newer "$FRONTEND_INDEX" -print -quit)" ]]; then
  FRONTEND_STALE=true
fi

if [[ "$FRONTEND_STALE" == true ]]; then
  [[ -d "$PROJECT_ROOT/frontend/node_modules" ]] || fail "Frontend dependencies are missing. Run npm --prefix frontend install."
  echo "Frontend build: source changed; building production assets"
  VITE_API_URL=" " npm --prefix "$PROJECT_ROOT/frontend" run build
  printf '%s' "same-origin-v1" >"$FRONTEND_BUILD_STAMP"
else
  echo "Frontend build: up to date"
fi

(
  cd "$PROJECT_ROOT/backend"
  exec "$BACKEND_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8000
) &
BACKEND_PID=$!

HEALTH_READY=false
for _attempt in $(seq 1 150); do
  if curl --silent --fail --max-time 1 http://127.0.0.1:8000/health >/dev/null; then
    HEALTH_READY=true
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    fail "FastAPI exited during startup."
  fi
  sleep 0.2
done
[[ "$HEALTH_READY" == true ]] || fail "FastAPI health check timed out after 30 seconds."

LAN_ADDRESS="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "CAG Passenger Monitoring is online."
echo "Local dashboard: http://localhost:8000"
if [[ -n "$LAN_ADDRESS" ]]; then
  echo "LAN dashboard:   http://$LAN_ADDRESS:8000"
fi
echo "The dashboard will enable Start Session after CV models finish loading."
echo "Press Ctrl+C to stop services started by this command."

if command -v xdg-open >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
  xdg-open http://localhost:8000 >/dev/null 2>&1 &
fi

set +e
wait "$BACKEND_PID"
BACKEND_STATUS=$?
set -e
BACKEND_PID=""
exit "$BACKEND_STATUS"
