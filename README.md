# CAG Passenger Monitoring Dashboard V1.5

An Ubuntu-based FastAPI, React, MQTT, and multi-camera computer-vision system for tactical passenger monitoring and assistance evidence.

## Operator quick start

After completing [README_UBUNTU.md](README_UBUNTU.md) and configuring `backend/.env`, run from the repository root:

```bash
bash start_ubuntu.sh
```

Open `http://localhost:8000`. The dashboard reports model-loading progress and enables **Start Session** when the persistent CV worker is ready. Operators see only session status and Start/Stop controls; technical model and camera settings remain outside the user interface.

The command provides:

- one Mosquitto broker, reusing an existing port-1883 listener when present;
- FastAPI and the compiled React dashboard on port 8000;
- an idle CV worker that preloads the enabled YOLO, MediaPipe, TransReID, role, and MiVOLO models;
- safe single-session Start/Stop control;
- cleanup limited to processes started by this deployment command.

## Tester workflow

The detailed engineering launcher remains available independently:

```bash
bash launch_tracker_ubuntu.sh
```

It exposes camera, GPU, calibration, model, threshold, fusion, ReID, MQTT, map, logging, and recording settings. It and the dashboard worker share configuration construction and a runtime ownership lock, so they cannot consume the cameras and GPUs simultaneously.

Use `bash start_dev_ubuntu.sh` for Vite/FastAPI development.

## Configuration

Create the private configuration once:

```bash
cp backend/.env.example backend/.env
```

At minimum configure real camera sources:

```text
CAMERA_URLS=cam_1=rtsp://username:password@192.168.50.192:554/Streaming/Channels/101,cam_2=rtsp://username:password@192.168.50.81:554/Streaming/Channels/101
PRIMARY_CAMERA_ID=cam_1
MQTT_ENABLED=true
MQTT_HOST=localhost
MQTT_PORT=1883
```

Do not commit `backend/.env`. Camera credentials remain on the backend and are not returned to React or written unredacted to service logs.

The production CV preset is configured with `CV_*` variables documented in `backend/.env.example`. Current defaults preserve the tester launch behavior: two cameras when configured, YOLO on GPU 0, TransReID/MiVOLO on GPU 1, MediaPipe auto delegate, appearance ReID and demographics enabled, 480 cm map size, and 5-by-5 visual grid.

## Session states

```text
offline → loading → ready → starting → running
                       ↑                  ↓
                       └──── stopping ────┘

Unrecoverable worker/model/session error → failed
```

Only `ready` permits a new session. Repeated Start for the active run is idempotent, Stop while idle is safe, and conflicting transitions return HTTP 409.

## Control security

The dashboard requires a login. Every API route except the React shell,
`/health`, and `/api/auth/login` needs an authenticated session.

- **Start/Stop** requires an authenticated **admin** plus a CSRF token, on both
  loopback and LAN. Loopback is not a bypass, and the operator is never asked
  for a second shared token.
- **Passenger evidence** is served only from authenticated `/api/evidence/...`
  routes. The old unauthenticated `/uploads/...` static mounts are removed.
- **The CV worker** authenticates its HTTP uploads with `CV_SERVICE_TOKEN`
  (`X-CV-Service-Token`), scoped to the evacuee ingestion endpoints only.
- **MQTT ingestion is unchanged** and remains a broker-level concern.

The legacy `CV_CONTROL_TOKEN` path is disabled by default and survives only
behind `CV_CONTROL_LEGACY_TOKEN_ENABLED=true` for scripts that have not yet
migrated. React never requests, stores, or sends it.

Create the first administrator from `backend/`:

```bash
.venv-linux/bin/python -m auth.cli create-user <name> --role admin
```

Full procedure: [docs/deployment/login_auth_deployment.md](docs/deployment/login_auth_deployment.md).

## Relevant API

```text
GET  /health
GET  /api/cv/status
POST /api/cv/session/start
POST /api/cv/session/stop

GET  /api/status
GET  /api/cameras
GET  /api/cameras/{camera_id}/stream
GET  /api/metrics
GET  /api/metrics/trends
GET  /api/zones/status
GET  /api/tactical/latest
GET  /api/alerts
GET  /api/evacuees
GET  /api/evacuees/summary
GET  /api/reports/shift.csv
GET  /api/reports/shift.xlsx
```

The CV status contract includes `state`, `ready`, `running`, `run_id`, timestamps, worker PID, loading stage, safe error text, MQTT reachability, and whether the current request may control sessions.

## Data flow

The tracker publishes lightweight telemetry to:

```text
cag/metrics
cag/tactical
cag/alerts
```

FastAPI subscribes and stores metrics, alerts, tactical state, ReID galleries, and assistance evidence. Images and embeddings use FastAPI endpoints rather than MQTT. The dashboard tactical view renders the globally fused map; the camera selector affects only the live preview.

Passenger evidence is an assistance aid, not face recognition or proof of a person’s real identity. Staff must manually verify model-provided role, age, gender, and ReID results.

## Logs and recovery

- CV worker and state transitions: `LogEvidance/cv_service.jsonl`
- Mosquitto started by the script: `LogEvidance/mosquitto-server.log`
- Dashboard and technical test runs: paired `LogEvidance/*.console.log` and
  `LogEvidance/*.jsonl` files named from the run ID and start timestamp

See [README_UBUNTU.md](README_UBUNTU.md) for installation, LAN access, troubleshooting, and recovery commands.

## For developers

Start with [docs/architecture.md](docs/architecture.md). It explains how a camera
frame becomes a numbered person on the dashboard, maps every module to its job,
and gives a suggested reading order for the codebase.

### Tests

There is no pytest in either environment; both suites use `unittest`. Tests live
in a `tests/` package beside the code they exercise, and each suite is run from
its own directory. The `-t .` matters: the modules import each other flatly
(`from constants import ...`), so discovery needs the parent directory as the
top level to put it on `sys.path`.

```bash
# CV pipeline - about 560 tests, ~3 seconds. No GPU, cameras, or model
# checkpoints required; the models are stubbed.
cd edge_tracker
../.venv-cv-linux/bin/python -m unittest discover -s tests -t .

# Backend - about 173 tests, ~50 seconds
cd backend
.venv-linux/bin/python -m unittest discover -s tests -t .
```

Lint the first-party Python from the repository root:

```bash
uvx ruff check .
```
