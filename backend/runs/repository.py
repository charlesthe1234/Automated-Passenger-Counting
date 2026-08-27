"""Database access for managed, legacy, and external runs."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from models import (
    EvacueeGalleryView,
    EvacueeIdentity,
    MetricLog,
    PassengerObservation,
    SystemAlert,
)
from timeutils import as_utc
from runs.models import (
    IN_PROGRESS_STATUSES,
    ORIGIN_EXTERNAL,
    ORIGIN_LEGACY,
    STATUS_ENDED,
    STATUS_EXTERNAL,
    DeletedRun,
    PendingFileDeletion,
    Run,
    RunEvent,
)

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

# Operational tables carrying a soft `run_id` string reference.
_RUN_SCOPED_MODELS = (MetricLog, SystemAlert, PassengerObservation, EvacueeIdentity)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_valid_run_id(run_id: str) -> bool:
    return bool(RUN_ID_PATTERN.match(run_id or ""))


# --- Lookups -------------------------------------------------------------


def get_by_run_id(db: Session, run_id: str) -> Run | None:
    return db.execute(select(Run).where(Run.run_id == run_id)).scalar_one_or_none()


def get_in_progress(db: Session) -> Run | None:
    return db.execute(
        select(Run).where(Run.status.in_(IN_PROGRESS_STATUSES))
    ).scalar_one_or_none()


def is_tombstoned(db: Session, run_id: str) -> bool:
    return db.get(DeletedRun, run_id) is not None


def get_recently_active_external(db: Session, *, window_seconds: int) -> list[Run]:
    """External runs that received data inside the activity window."""
    cutoff = utc_now() - timedelta(seconds=window_seconds)
    candidates = db.execute(
        select(Run)
        .where(Run.origin_type == ORIGIN_EXTERNAL)
        .where(Run.last_ingested_at.is_not(None))
        .order_by(Run.last_ingested_at.desc())
    ).scalars()
    return [run for run in candidates if (as_utc(run.last_ingested_at) or cutoff) >= cutoff]


def existing_run_ids_in_operational_tables(db: Session) -> set[str]:
    found: set[str] = set()
    for model in _RUN_SCOPED_MODELS:
        found.update(
            value
            for value in db.execute(select(model.run_id).distinct()).scalars()
            if value
        )
    return found


# --- Creation ------------------------------------------------------------


def generate_run_id(db: Session) -> str:
    """Collision-resistant, human-readable, unique across runs and tombstones."""
    taken = {row for row in db.execute(select(Run.run_id)).scalars()}
    taken.update(db.execute(select(DeletedRun.run_id)).scalars())
    taken.update(existing_run_ids_in_operational_tables(db))

    for _ in range(50):
        stamp = utc_now().strftime("%Y%m%d_%H%M%S")
        candidate = f"run_{stamp}_{secrets.token_hex(2)}"
        if candidate not in taken:
            return candidate
    raise RuntimeError("Could not generate an unused run ID.")


def create_run(
    db: Session,
    *,
    run_id: str,
    status: str,
    origin_type: str,
    name: str | None = None,
    description: str | None = None,
    is_demo: bool = False,
    created_by_user_id: int | None = None,
    requested_at: datetime | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    first_ingested_at: datetime | None = None,
    last_ingested_at: datetime | None = None,
) -> Run:
    now = utc_now()
    run = Run(
        run_id=run_id,
        name=name,
        description=description,
        status=status,
        is_demo=is_demo,
        origin_type=origin_type,
        created_by_user_id=created_by_user_id,
        requested_at=requested_at or now,
        started_at=started_at,
        ended_at=ended_at,
        first_ingested_at=first_ingested_at,
        last_ingested_at=last_ingested_at,
        status_changed_at=now,
        created_at=now,
        updated_at=now,
    )
    db.add(run)
    db.flush()
    return run


def append_event(
    db: Session,
    *,
    run_id: str,
    event_type: str,
    actor_user_id: int | None = None,
    safe_detail: str | None = None,
) -> RunEvent:
    event = RunEvent(
        run_id=run_id,
        event_type=event_type,
        actor_user_id=actor_user_id,
        timestamp=utc_now(),
        safe_detail=safe_detail,
    )
    db.add(event)
    return event


def set_status(
    db: Session,
    run: Run,
    status: str,
    *,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    failure_reason: str | None = None,
) -> Run:
    now = utc_now()
    run.status = status
    run.status_changed_at = now
    run.updated_at = now
    if started_at is not None:
        run.started_at = started_at
    if ended_at is not None:
        run.ended_at = ended_at
    if failure_reason is not None:
        run.failure_reason = failure_reason[:500]
    db.flush()
    return run


def touch_ingestion(db: Session, run: Run, *, moment: datetime | None = None) -> None:
    """Record that operational data was accepted for this run."""
    stamp = moment or utc_now()
    if run.first_ingested_at is None:
        run.first_ingested_at = stamp
    run.last_ingested_at = stamp
    run.updated_at = stamp


# --- Aggregate counts ----------------------------------------------------


def _grouped_counts(db: Session, model, *criteria) -> dict[str, int]:
    statement = select(model.run_id, func.count(model.id))
    if criteria:
        statement = statement.where(*criteria)
    rows = db.execute(statement.group_by(model.run_id)).all()
    return {run_id: count for run_id, count in rows if run_id}


def _gallery_counts(db: Session) -> dict[str, int]:
    """Gallery views belong to a run through their evacuee, not directly."""
    rows = db.execute(
        select(EvacueeIdentity.run_id, func.count(EvacueeGalleryView.id))
        .join(EvacueeGalleryView, EvacueeGalleryView.evacuee_id == EvacueeIdentity.id)
        .group_by(EvacueeIdentity.run_id)
    ).all()
    return {run_id: count for run_id, count in rows if run_id}


def _passenger_counts(db: Session) -> dict[str, tuple[int, int]]:
    """Peak and latest passenger count per run, in one pass each."""
    peaks = {
        run_id: int(peak or 0)
        for run_id, peak in db.execute(
            select(MetricLog.run_id, func.max(MetricLog.passenger_count)).group_by(MetricLog.run_id)
        ).all()
        if run_id
    }
    latest_ids = select(MetricLog.run_id, func.max(MetricLog.id).label("max_id")).group_by(
        MetricLog.run_id
    ).subquery()
    latest = {
        run_id: int(count or 0)
        for run_id, count in db.execute(
            select(MetricLog.run_id, MetricLog.passenger_count).join(
                latest_ids, MetricLog.id == latest_ids.c.max_id
            )
        ).all()
        if run_id
    }
    return {run_id: (peaks.get(run_id, 0), latest.get(run_id, 0)) for run_id in peaks}


def collect_summaries(db: Session) -> dict[str, dict]:
    """All per-run counts using grouped queries instead of one query per run."""
    metrics = _grouped_counts(db, MetricLog)
    alerts = _grouped_counts(db, SystemAlert)
    observations = _grouped_counts(db, PassengerObservation)
    evacuees = _grouped_counts(
        db,
        EvacueeIdentity,
        EvacueeIdentity.role == "evacuee",
    )
    galleries = _gallery_counts(db)
    passenger = _passenger_counts(db)

    run_ids = set(metrics) | set(alerts) | set(observations) | set(evacuees) | set(galleries)
    summaries = {}
    for run_id in run_ids:
        peak, latest = passenger.get(run_id, (0, 0))
        summaries[run_id] = {
            "metric_count": metrics.get(run_id, 0),
            "alert_count": alerts.get(run_id, 0),
            "observation_count": observations.get(run_id, 0),
            "evacuee_count": evacuees.get(run_id, 0),
            "gallery_view_count": galleries.get(run_id, 0),
            "peak_passenger_count": peak,
            "latest_passenger_count": latest,
        }
    return summaries


EMPTY_SUMMARY = {
    "metric_count": 0,
    "alert_count": 0,
    "observation_count": 0,
    "evacuee_count": 0,
    "gallery_view_count": 0,
    "peak_passenger_count": 0,
    "latest_passenger_count": 0,
}


def list_runs(
    db: Session,
    *,
    status: str | None = None,
    origin_type: str | None = None,
    is_demo: bool | None = None,
    limit: int = 25,
    offset: int = 0,
) -> tuple[list[Run], int]:
    statement = select(Run)
    count_statement = select(func.count(Run.id))
    if status:
        statement = statement.where(Run.status == status)
        count_statement = count_statement.where(Run.status == status)
    if origin_type:
        statement = statement.where(Run.origin_type == origin_type)
        count_statement = count_statement.where(Run.origin_type == origin_type)
    if is_demo is not None:
        statement = statement.where(Run.is_demo.is_(is_demo))
        count_statement = count_statement.where(Run.is_demo.is_(is_demo))

    total = int(db.execute(count_statement).scalar_one())
    ordering = func.coalesce(Run.started_at, Run.requested_at).desc()
    rows = list(
        db.execute(statement.order_by(ordering, Run.id.desc()).limit(limit).offset(offset)).scalars()
    )
    return rows, total


# --- Legacy backfill -----------------------------------------------------


def _legacy_bounds(db: Session) -> dict[str, tuple[datetime | None, datetime | None]]:
    bounds: dict[str, list[datetime | None]] = {}
    timestamped = (
        (MetricLog, MetricLog.timestamp),
        (SystemAlert, SystemAlert.timestamp),
        (PassengerObservation, PassengerObservation.timestamp),
        (EvacueeIdentity, EvacueeIdentity.first_seen_at),
        (EvacueeIdentity, EvacueeIdentity.last_seen_at),
    )
    for model, column in timestamped:
        rows = db.execute(
            select(model.run_id, func.min(column), func.max(column)).group_by(model.run_id)
        ).all()
        for run_id, earliest, latest in rows:
            if not run_id:
                continue
            current = bounds.setdefault(run_id, [None, None])
            earliest, latest = as_utc(earliest), as_utc(latest)
            if earliest and (current[0] is None or earliest < current[0]):
                current[0] = earliest
            if latest and (current[1] is None or latest > current[1]):
                current[1] = latest
    return {run_id: (values[0], values[1]) for run_id, values in bounds.items()}


def backfill_legacy_runs(db: Session) -> list[str]:
    """Import pre-existing run IDs as ended legacy runs. Safe to repeat."""
    known = {row for row in db.execute(select(Run.run_id)).scalars()}
    tombstoned = {row for row in db.execute(select(DeletedRun.run_id)).scalars()}
    discovered = existing_run_ids_in_operational_tables(db)
    missing = sorted(discovered - known - tombstoned)
    if not missing:
        return []

    bounds = _legacy_bounds(db)
    fallback = utc_now()
    imported = []
    for run_id in missing:
        earliest, latest = bounds.get(run_id, (None, None))
        earliest = earliest or fallback
        latest = latest or earliest
        create_run(
            db,
            run_id=run_id,
            status=STATUS_ENDED,
            origin_type=ORIGIN_LEGACY,
            name=f"Imported {run_id}",
            requested_at=earliest,
            started_at=earliest,
            ended_at=latest,
            first_ingested_at=earliest,
            last_ingested_at=latest,
        )
        append_event(
            db,
            run_id=run_id,
            event_type="legacy_imported",
            safe_detail="Imported from existing operational data.",
        )
        imported.append(run_id)
    return imported


def register_external_run(db: Session, run_id: str, *, moment: datetime | None = None) -> Run:
    """Create or refresh the run row for an externally launched pipeline.

    `ended_at` is deliberately never set: the backend does not control the
    external producer and cannot claim the run finished.
    """
    stamp = moment or utc_now()
    run = get_by_run_id(db, run_id)
    if run is None:
        run = create_run(
            db,
            run_id=run_id,
            status=STATUS_EXTERNAL,
            origin_type=ORIGIN_EXTERNAL,
            name=f"External {run_id}",
            requested_at=stamp,
            started_at=stamp,
            first_ingested_at=stamp,
            last_ingested_at=stamp,
        )
        append_event(
            db,
            run_id=run_id,
            event_type="external_created",
            safe_detail="Unmanaged CV data accepted under compatibility settings.",
        )
    else:
        touch_ingestion(db, run, moment=stamp)
    return run


# --- Deletion ------------------------------------------------------------


def collect_image_paths(db: Session, run_id: str) -> list[str]:
    """Every stored image path belonging to a run, de-duplicated."""
    gallery = db.execute(
        select(EvacueeGalleryView.image_path)
        .join(EvacueeIdentity, EvacueeGalleryView.evacuee_id == EvacueeIdentity.id)
        .where(EvacueeIdentity.run_id == run_id)
    ).scalars()
    observations = db.execute(
        select(PassengerObservation.image_path).where(PassengerObservation.run_id == run_id)
    ).scalars()
    return sorted({path for path in list(gallery) + list(observations) if path})


def delete_run_rows(db: Session, run_id: str) -> dict[str, int]:
    """Delete every run-scoped operational row. Caller owns the transaction."""
    identity_ids = list(
        db.execute(select(EvacueeIdentity.id).where(EvacueeIdentity.run_id == run_id)).scalars()
    )

    gallery_deleted = 0
    if identity_ids:
        gallery_deleted = int(
            db.execute(
                delete(EvacueeGalleryView).where(EvacueeGalleryView.evacuee_id.in_(identity_ids))
            ).rowcount
            or 0
        )

    alerts = int(db.execute(delete(SystemAlert).where(SystemAlert.run_id == run_id)).rowcount or 0)
    evacuees = int(
        db.execute(delete(EvacueeIdentity).where(EvacueeIdentity.run_id == run_id)).rowcount or 0
    )
    observations = int(
        db.execute(
            delete(PassengerObservation).where(PassengerObservation.run_id == run_id)
        ).rowcount
        or 0
    )
    metrics = int(db.execute(delete(MetricLog).where(MetricLog.run_id == run_id)).rowcount or 0)

    return {
        "deleted_gallery_views": gallery_deleted,
        "deleted_alerts": alerts,
        "deleted_evacuees": evacuees,
        "deleted_observations": observations,
        "deleted_metrics": metrics,
    }


def create_tombstone(db: Session, run_id: str, *, deleted_by_user_id: int | None) -> DeletedRun:
    tombstone = DeletedRun(
        run_id=run_id,
        deleted_at=utc_now(),
        deleted_by_user_id=deleted_by_user_id,
    )
    db.add(tombstone)
    db.flush()
    return tombstone


def record_pending_file_deletion(
    db: Session, *, run_id: str, storage_key: str, safe_error_code: str
) -> None:
    """Track an orphaned file so maintenance can find it later."""
    existing = db.execute(
        select(PendingFileDeletion)
        .where(PendingFileDeletion.run_id == run_id)
        .where(PendingFileDeletion.storage_key == storage_key)
        .where(PendingFileDeletion.completed_at.is_(None))
    ).scalar_one_or_none()

    now = utc_now()
    if existing is None:
        db.add(
            PendingFileDeletion(
                run_id=run_id,
                storage_key=storage_key,
                first_failed_at=now,
                last_attempt_at=now,
                attempt_count=1,
                safe_error_code=safe_error_code,
            )
        )
    else:
        existing.last_attempt_at = now
        existing.attempt_count += 1
        existing.safe_error_code = safe_error_code
