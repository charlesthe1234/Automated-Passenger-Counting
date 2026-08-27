from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import crud
from auth.dependencies import (
    AdminCsrfUser,
    CurrentUser,
    require_cv_service_or_admin,
)
from auth.router import admin_router, router as auth_router
from auth.sessions import purge_expired_sessions
from camera import CameraStreamer, camera_manager, mjpeg_frame_generator
from config import settings
from cv_api import router as cv_router
from cv_manager import cv_manager
from database import SessionLocal, get_db, init_db
from evacuees.router import router as evacuee_router
from evacuees.storage import ensure_upload_dir as ensure_evacuee_upload_dir
from evidence.router import (
    NO_STORE_HEADERS,
    evidence_url_for_observation,
    router as evidence_router,
)
from models import (
    MetricLogCreate,
    MetricLogRead,
    MetricTrendPointRead,
    PassengerObservationRead,
    PassengerObservationSummary,
    SystemAlertCreate,
    SystemAlertRead,
    TacticalStateCreate,
    TacticalStateRead,
    ZoneStatusRead,
)
from observation_storage import (
    clear_observation_images,
    ensure_upload_dir,
    save_observation_image,
)
from mqtt_bridge import mqtt_bridge
from reports.shift_report import REPORT_MIME_TYPE, generate_shift_report_csv, generate_shift_report_xlsx
from runs.router import router as runs_router
from runs.service import IngestionRejected, resolve_ingestion_run
from runs.write_guard import RunWriteConflict, immediate_write
from staff.router import router as staff_router
from tactical_state import tactical_store


logger = logging.getLogger(__name__)


def _warn_about_insecure_transport() -> None:
    if settings.auth_allow_insecure_http:
        logger.warning(
            "AUTH_ALLOW_INSECURE_HTTP is enabled: dashboard passwords and session "
            "cookies travel over unencrypted HTTP on this network. This is intended "
            "only for a controlled demonstration LAN. Set it back to false once "
            "HTTPS is configured."
        )
    if not settings.cv_service_token.get_secret_value():
        logger.warning(
            "CV_SERVICE_TOKEN is not set: the CV worker cannot upload identity "
            "metadata or gallery evidence until it is configured in backend/.env."
        )


async def _reconcile_runs_periodically(stop_event: asyncio.Event) -> None:
    """Keep run state truthful even when nobody is looking at the dashboard."""
    from runs.service import reconcile

    interval = settings.run_reconcile_interval_seconds
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            await asyncio.to_thread(_reconcile_once, reconcile)
        except Exception:  # pragma: no cover - a reconcile blip must not kill the task
            logger.warning("Run reconciliation pass failed.", exc_info=False)


def _reconcile_once(reconcile) -> None:
    """Take the write lock only when a transition is actually due.

    This task runs every couple of seconds for the lifetime of the process, so
    acquiring SQLite's exclusive lock unconditionally would contend with MQTT
    ingestion even when no run is in progress.
    """
    from runs.service import reconciliation_due

    try:
        with SessionLocal() as read_db:
            if not reconciliation_due(read_db):
                return
        with immediate_write() as db:
            reconcile(db)
    except RunWriteConflict:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Migrations must complete before MQTT, CV, or request serving begins.
    init_db()
    _warn_about_insecure_transport()
    with SessionLocal() as db:
        purge_expired_sessions(db)
        db.commit()
    ensure_upload_dir()
    ensure_evacuee_upload_dir()
    camera_manager.start_all()
    mqtt_bridge.start()
    cv_manager.start_worker()

    # Import late so run services see a fully initialised database.
    from runs.service import run_startup_tasks

    run_startup_tasks()
    stop_reconciler = asyncio.Event()
    reconciler = asyncio.create_task(_reconcile_runs_periodically(stop_reconciler))
    try:
        yield
    finally:
        stop_reconciler.set()
        reconciler.cancel()
        cv_manager.shutdown()
        mqtt_bridge.stop()
        camera_manager.stop_all()


app = FastAPI(title="CAG Passenger Monitoring API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Passenger evidence is intentionally NOT exposed through a static mount.
# It is served only by the authenticated, database-backed routes in
# evidence/router.py.
ensure_upload_dir()
ensure_evacuee_upload_dir()
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(evidence_router)
app.include_router(evacuee_router)
app.include_router(staff_router)
app.include_router(runs_router)
app.include_router(cv_router)

FRONTEND_DIST_PATH = settings.frontend_dist_path
FRONTEND_INDEX_PATH = FRONTEND_DIST_PATH / "index.html"
FRONTEND_ASSETS_PATH = FRONTEND_DIST_PATH / "assets"

if FRONTEND_ASSETS_PATH.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_PATH), name="frontend-assets")


DbSession = Annotated[Session, Depends(get_db)]


def get_camera_or_404(camera_id: str) -> CameraStreamer:
    streamer = camera_manager.get(camera_id)
    if streamer is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' is not configured.")
    return streamer


def guarded_ingest(run_id: str, writer):
    """Validate run policy and write, inside one serialized transaction.

    Shared by the HTTP fallback ingestion routes so they obey the same
    tombstone and active-run rules as MQTT ingestion.
    """
    try:
        with immediate_write() as db:
            resolve_ingestion_run(db, run_id)
            return writer(db)
    except IngestionRejected as reason:
        raise HTTPException(status_code=409, detail=f"Payload was not accepted: {reason}") from reason
    except RunWriteConflict as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/status")
def camera_status(_user: CurrentUser) -> dict:
    return camera_manager.primary().status()


@app.get("/api/cameras")
def camera_statuses(_user: CurrentUser) -> list[dict]:
    return camera_manager.all_status()


@app.get("/api/cameras/{camera_id}/status")
def camera_status_by_id(camera_id: str, _user: CurrentUser) -> dict:
    return get_camera_or_404(camera_id).status()


@app.get("/api/stream")
def stream_camera(_user: CurrentUser) -> StreamingResponse:
    return StreamingResponse(
        mjpeg_frame_generator(camera_manager.primary()),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=NO_STORE_HEADERS,
    )


@app.get("/api/cameras/{camera_id}/stream")
def stream_camera_by_id(camera_id: str, _user: CurrentUser) -> StreamingResponse:
    return StreamingResponse(
        mjpeg_frame_generator(get_camera_or_404(camera_id)),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=NO_STORE_HEADERS,
    )


@app.get("/api/metrics", response_model=list[MetricLogRead])
def get_metrics(
    db: DbSession, _user: CurrentUser, run_id: str | None = Query(default=None)
) -> list[MetricLogRead]:
    return crud.get_latest_metrics(db, run_id=run_id, limit=10)


@app.get("/api/metrics/trends", response_model=list[MetricTrendPointRead])
def get_metric_trends(
    db: DbSession,
    _user: CurrentUser,
    run_id: str | None = Query(default=None),
    minutes: int = Query(default=60, ge=1, le=1440),
) -> list[MetricTrendPointRead]:
    return crud.get_metric_trends(db, run_id=run_id, minutes=minutes)


@app.post(
    "/api/metrics",
    response_model=MetricLogRead,
    status_code=201,
    dependencies=[Depends(require_cv_service_or_admin)],
)
def post_metric(payload: MetricLogCreate) -> MetricLogRead:
    return guarded_ingest(payload.run_id, lambda db: crud.create_metric_log(db, payload))


@app.get("/api/zones/status", response_model=list[ZoneStatusRead])
def get_zone_status(
    db: DbSession, _user: CurrentUser, run_id: str | None = Query(default=None)
) -> list[ZoneStatusRead]:
    return crud.get_zone_status(db, capacities=settings.zone_capacity_map, run_id=run_id)


def _report_headers(filename: str) -> dict[str, str]:
    return {
        "Content-Disposition": f'attachment; filename="{filename}"',
        **NO_STORE_HEADERS,
    }


@app.get("/api/reports/shift.csv")
def get_shift_report_csv(
    db: DbSession, _user: CurrentUser, run_id: str | None = Query(default=None)
) -> Response:
    report = generate_shift_report_csv(db, capacities=settings.zone_capacity_map, run_id=run_id)
    return Response(
        content=report.content,
        media_type="text/csv; charset=utf-8",
        headers=_report_headers(report.filename),
    )


@app.get("/api/reports/shift.xlsx")
def get_shift_report_xlsx(
    db: DbSession, _user: CurrentUser, run_id: str | None = Query(default=None)
) -> Response:
    report = generate_shift_report_xlsx(db, capacities=settings.zone_capacity_map, run_id=run_id)
    return Response(
        content=report.content,
        media_type=REPORT_MIME_TYPE,
        headers=_report_headers(report.filename),
    )


@app.post(
    "/api/tactical",
    response_model=TacticalStateRead,
    status_code=201,
    dependencies=[Depends(require_cv_service_or_admin)],
)
def post_tactical_state(payload: TacticalStateCreate) -> TacticalStateRead:
    return tactical_store.update(payload)


@app.get("/api/tactical/latest", response_model=TacticalStateRead)
def get_latest_tactical_state(
    _user: CurrentUser,
    camera_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
) -> TacticalStateRead:
    return tactical_store.latest(camera_id=camera_id, run_id=run_id)


@app.get("/api/alerts", response_model=list[SystemAlertRead])
def get_alerts(
    db: DbSession, _user: CurrentUser, run_id: str | None = Query(default=None)
) -> list[SystemAlertRead]:
    return crud.get_latest_alerts(db, run_id=run_id, limit=5)


@app.post(
    "/api/alerts",
    response_model=SystemAlertRead,
    status_code=201,
    dependencies=[Depends(require_cv_service_or_admin)],
)
def post_alert(payload: SystemAlertCreate) -> SystemAlertRead:
    return guarded_ingest(payload.run_id, lambda db: crud.create_system_alert(db, payload))


@app.get("/api/observations/summary", response_model=PassengerObservationSummary)
def get_observation_summary(
    db: DbSession,
    _user: CurrentUser,
    run_id: str | None = Query(default=None),
) -> PassengerObservationSummary:
    return crud.get_observation_summary(db, run_id=run_id)


def _observation_payload(observation) -> dict:
    """Serve legacy observation images through the authenticated evidence route."""
    return {
        "id": observation.id,
        "timestamp": observation.timestamp,
        "run_id": observation.run_id,
        "camera_id": observation.camera_id,
        "track_id": observation.track_id,
        "age": observation.age,
        "gender": observation.gender,
        "age_confidence": observation.age_confidence,
        "gender_confidence": observation.gender_confidence,
        "image_url": evidence_url_for_observation(observation.id),
    }


@app.get("/api/observations", response_model=list[PassengerObservationRead])
def get_observations(
    db: DbSession,
    _user: CurrentUser,
    gender: str | None = Query(default=None),
    min_age: float | None = Query(default=None, ge=0, le=120),
    max_age: float | None = Query(default=None, ge=0, le=120),
    camera_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict]:
    if min_age is not None and max_age is not None and min_age > max_age:
        raise HTTPException(status_code=400, detail="min_age cannot be greater than max_age.")

    observations = crud.get_latest_observations(
        db,
        gender=gender,
        min_age=min_age,
        max_age=max_age,
        camera_id=camera_id,
        run_id=run_id,
        limit=limit,
    )
    return [_observation_payload(observation) for observation in observations]


@app.post(
    "/api/observations",
    response_model=PassengerObservationRead,
    status_code=201,
    dependencies=[Depends(require_cv_service_or_admin)],
)
async def post_observation(
    image: UploadFile = File(...),
    age: float = Form(..., ge=0, le=120),
    gender: str = Form(..., min_length=1, max_length=32),
    camera_id: str = Form(..., min_length=1, max_length=80),
    run_id: str = Form("default", max_length=80),
    track_id: str | None = Form(default=None, max_length=120),
    age_confidence: float | None = Form(default=None, ge=0, le=1),
    gender_confidence: float | None = Form(default=None, ge=0, le=1),
    timestamp: datetime | None = Form(default=None),
) -> PassengerObservationRead:
    image_path, image_url = await save_observation_image(image)
    try:
        observation = guarded_ingest(
            run_id,
            lambda db: crud.create_passenger_observation(
                db,
                timestamp=timestamp,
                run_id=run_id,
                camera_id=camera_id,
                track_id=track_id,
                age=age,
                gender=gender,
                age_confidence=age_confidence,
                gender_confidence=gender_confidence,
                image_path=str(image_path),
                image_url=image_url,
            ),
        )
    except Exception:
        image_path.unlink(missing_ok=True)
        raise
    return _observation_payload(observation)


@app.delete("/api/observations")
def delete_observations(db: DbSession, _admin: AdminCsrfUser) -> dict[str, int]:
    """Legacy high-risk endpoint: clears every observation row and image.

    Behaviour is unchanged from the pre-auth implementation and it is now
    restricted to an administrator session with CSRF. Run Manager replaces this
    with run-scoped deletion; do not surface it as a normal operator control.
    """
    deleted_rows = crud.clear_passenger_observations(db)
    db.commit()
    deleted_images = clear_observation_images()
    return {"deleted_rows": deleted_rows, "deleted_images": deleted_images}


def serve_frontend_index() -> FileResponse:
    if not FRONTEND_INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend build not found. Run `npm run build` in the frontend folder.",
        )
    return FileResponse(FRONTEND_INDEX_PATH)


@app.get("/", include_in_schema=False)
def frontend_root() -> FileResponse:
    return serve_frontend_index()


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_spa_fallback(full_path: str) -> FileResponse:
    reserved_paths = {"api", "assets", "health", "uploads"}
    # "uploads/" stays reserved so a stale bookmark to the removed static mount
    # returns 404 instead of silently rendering the SPA shell.
    reserved_prefixes = ("api/", "uploads/", "assets/")
    if full_path in reserved_paths or full_path.startswith(reserved_prefixes):
        raise HTTPException(status_code=404, detail="Not found")
    return serve_frontend_index()
