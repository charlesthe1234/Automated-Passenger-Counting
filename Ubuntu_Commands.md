# Ubuntu operation guide

This repository uses two separate Ubuntu Python environments, both managed by
[uv](https://docs.astral.sh/uv/) and each described by its own `pyproject.toml`
and committed `uv.lock`:

- `.venv-cv-linux` (from `edge_tracker/`): Torch, CUDA, YOLO, MediaPipe, TransReID, role classification, and MiVOLO.
- `backend/.venv-linux` (from `backend/`): FastAPI, SQLite, camera previews, and MQTT ingestion.

They are deliberately separate: only the CV environment carries the multi-gigabyte
CUDA stack. Copied Windows virtual environments cannot run on Ubuntu.

## Initial setup

Install uv, the NVIDIA driver, reboot, and verify both cards with `nvidia-smi`. Then run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if uv is not installed
bash setup_ubuntu.sh
```

`setup_ubuntu.sh` creates both environments from their lock files, clones the
pinned `fast-reid` checkout, installs the frontend, and prints the resolved torch
and CUDA versions. The CUDA-enabled PyTorch build is resolved automatically from
NVIDIA's CUDA 12.8 wheel index declared in `edge_tracker/pyproject.toml`; it is no
longer a manual install step. Verify the GPU at any time with:

```bash
.venv-cv-linux/bin/python -c "import torch; print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

Model weights (`yolo26m.pt`, `transreid_msmt17.pth`, `pose_landmarker_full.task`,
`evacuation_mobilenet_v1.pth`) are not tracked in git and must already be present
in `edge_tracker/`; `start_ubuntu.sh` verifies them before launching.

## Changing dependencies

Edit the relevant `pyproject.toml`, then re-lock and sync that environment:

```bash
cd edge_tracker && UV_PROJECT_ENVIRONMENT=../.venv-cv-linux uv sync --inexact
cd backend     && UV_PROJECT_ENVIRONMENT=.venv-linux    uv sync --inexact
```

Both projects pin their transitive dependencies to the set the system was verified
on, via `constraint-dependencies`. Relax those pins only alongside a hardware test —
an unconstrained CV resolve jumps mediapipe, transformers, and OpenCV by a major
version. Commit the updated `uv.lock` with the change.

Copy `backend/.env.example` to `backend/.env` and configure `CAMERA_URLS`. Keep real RTSP credentials only in that private `.env` file.

## Normal operator startup

Run one command from the repository root:

```bash
bash start_ubuntu.sh
```

The script validates the environments and configuration, starts Mosquitto only if port 1883 is unused, builds React only when its production output is missing or stale, starts FastAPI, waits for `/health`, and opens:

```text
http://localhost:8000
```

The dashboard initially shows **Preparing computer vision**. It enables **Start Session** only after all configured models are loaded. **Stop Session** closes camera processing safely but keeps models loaded, so the next session starts faster.

Press `Ctrl+C` in the startup terminal to stop the backend, worker, and any Mosquitto process that this command created.

## Technical tester launcher

Testers can instead open the detailed Tkinter launcher:

```bash
bash launch_tracker_ubuntu.sh
```

It retains controls for camera selection, calibration, devices, model settings, thresholds, fusion, ReID, MQTT, map dimensions, grid dimensions, logging, and recording. Its camera defaults are read from `backend/.env`; credentials are redacted from the command preview.

The dashboard worker and technical launcher are independent entry points, but they intentionally cannot own the CV runtime simultaneously. Stop `start_ubuntu.sh` before using the technical launcher, or stop the technical tracker before starting the operator server.

## Frontend development

Use the separate development command when changing React code:

```bash
bash start_dev_ubuntu.sh
```

This serves Vite on `http://localhost:5173` and FastAPI on `http://localhost:8000`. Normal operators should use `start_ubuntu.sh`.

## LAN viewing and control

The LAN URL printed by the startup script opens a **login screen**. Viewers need
an account; nothing operational is readable while signed out.

- Staff can view Operations and Passenger Assistance.
- Admins additionally get CV Start/Stop and the Settings area for managing
  accounts. This works from the server PC and over the LAN with the same admin
  login plus CSRF — there is no separate operator access code any more.

Because HTTPS is not configured yet, LAN logins need this in `backend/.env`:

```text
AUTH_ALLOW_INSECURE_HTTP=true
```

⚠️ Passwords and session cookies then travel unencrypted. It is intended for a
controlled demonstration LAN only; set it back to `false` once TLS is in place.
Loopback login keeps working either way, so the server PC is never locked out.

The CV worker needs `CV_SERVICE_TOKEN` in the same file to upload evidence.
Full procedure: [docs/deployment/login_auth_deployment.md](docs/deployment/login_auth_deployment.md).

### First administrator

```bash
cd backend
.venv-linux/bin/python -m auth.cli create-user <name> --role admin
.venv-linux/bin/python -m auth.cli list-users
```

## Runs

Every field test is now an explicit **run**. Admins start one from the Operations
banner; all metrics, alerts, and evacuee evidence attach to that run ID. The
admin-only **Runs** tab lists past runs with counts and durations, exports a
report per run, and permanently deletes a single run without touching the others
or the database file.

Starting fresh means creating a new run, not deleting the database.

CV started from the launcher or a terminal still works. That data is accepted and
labelled **External / unmanaged** — Run Manager did not start it and cannot stop
it. This compatibility behaviour is controlled by `ALLOW_UNMANAGED_RUN_INGESTION`
(default `true`); startup logs a warning while it is enabled. Each manual launch
now defaults to a unique run ID like `debug_<user>_<timestamp>` instead of
reusing `field_test_001`.

The launcher's **Reset** button clears passenger data, runs, and uploaded
evidence, but **keeps operator accounts** and the schema — it no longer deletes
the database file.

Full procedure: [docs/deployment/run_manager_deployment.md](docs/deployment/run_manager_deployment.md).

## Recovery

- **Computer vision unavailable:** inspect `LogEvidance/cv_service.jsonl`, verify CUDA with the command above, and check that the configured model files exist. Restart `start_ubuntu.sh` after correcting the problem.
- **MQTT unavailable:** check `ss -ltn '( sport = :1883 )'` and `LogEvidance/mosquitto-server.log`.
- **Camera cannot open:** verify the camera IP, switch port, Ethernet route, and `CAMERA_URLS`. The worker deliberately does not log RTSP credentials.
- **Port 8000 already in use:** stop the earlier backend. The startup script refuses to replace an unknown process.
- **Technical launcher reports CV already owned:** stop the operator server/worker before starting the tester tracker.

Homography files remain valid only while camera placement, resolution, and crop remain unchanged. Desktop recording may require an Xorg session rather than Wayland.
