"""Run lifecycle, CV reconciliation, guarded ingestion, and safe deletion.

Run bookkeeping and the CV worker are separate systems. SQLite cannot roll back
a command already sent to an external process, so this module never claims
atomicity between them: it uses compensating state changes and reconciles the
two truths on a schedule.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

from config import settings
from cv_manager import CvTransitionError, cv_manager
from evacuees.storage import UPLOAD_DIR as EVACUEE_UPLOAD_DIR
from observation_storage import UPLOAD_DIR as OBSERVATION_UPLOAD_DIR
from runs import repository
from runs.models import (
    IN_PROGRESS_STATUSES,
    ORIGIN_MANAGED,
    STATUS_ACTIVE,
    STATUS_ENDED,
    STATUS_ENDING,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_STARTING,
)
from runs.write_guard import immediate_write

logger = logging.getLogger(__name__)

# Serializes start/end decisions inside this process; the partial unique index
# on `runs` is what actually guarantees a single in-progress run.
_lifecycle_lock = threading.Lock()


class RunConflictError(RuntimeError):
    """The request conflicts with current run or CV state (HTTP 409)."""


class RunNotFoundError(RuntimeError):
    """No such run (HTTP 404)."""


class RunValidationError(ValueError):
    """The request itself is invalid (HTTP 400)."""


class IngestionRejected(RuntimeError):
    """A run-scoped payload was not accepted."""


# --- Reconciliation ------------------------------------------------------


def _cv_snapshot() -> dict:
    status = cv_manager.status()
    return {
        "state": status.get("state"),
        "run_id": status.get("run_id"),
        "ready": bool(status.get("ready")),
    }


def _seconds_since(moment: datetime | None) -> float:
    if moment is None:
        return 0.0
    return (repository.utc_now() - (repository.as_utc(moment) or repository.utc_now())).total_seconds()


def decide_reconciliation(run, cv: dict) -> dict | None:
    """Pure decision: what should this run become, given the CV worker's state?

    Returning None means nothing needs to change. Keeping this separate from the
    write lets a read-only caller (the frequently polled `/api/runs/active`)
    determine that no transition is due without taking SQLite's write lock.
    """
    if run is None:
        return None

    cv_state = cv["state"]
    matches = cv["run_id"] == run.run_id
    now = repository.utc_now()

    if run.status == STATUS_STARTING:
        if cv_state == "running" and matches:
            return {"status": STATUS_ACTIVE, "event": "activated", "started_at": now}
        if cv_state in {"failed", "offline"}:
            return {
                "status": STATUS_FAILED, "event": "failed", "ended_at": now,
                "failure_reason": "The CV worker failed before the run became active.",
            }
        if _seconds_since(run.requested_at) > settings.run_start_timeout_seconds:
            return {
                "status": STATUS_FAILED, "event": "failed", "ended_at": now,
                "failure_reason": "The CV worker did not report running before the start timeout.",
                "stop_worker": True,
            }

    elif run.status == STATUS_ACTIVE:
        if cv_state in {"failed", "offline"}:
            return {
                "status": STATUS_INTERRUPTED, "event": "interrupted", "ended_at": now,
                "failure_reason": "The CV worker stopped unexpectedly during the run.",
            }
        if cv_state == "ready" and not matches:
            return {
                "status": STATUS_INTERRUPTED, "event": "interrupted", "ended_at": now,
                "failure_reason": "The CV worker is no longer running this run.",
            }

    elif run.status == STATUS_ENDING:
        if cv_state == "ready":
            return {"status": STATUS_ENDED, "event": "ended", "ended_at": now}
        if cv_state in {"failed", "offline"}:
            return {
                "status": STATUS_INTERRUPTED, "event": "interrupted", "ended_at": now,
                "failure_reason": "The CV worker failed while the run was stopping.",
            }
        if _seconds_since(run.status_changed_at) > settings.run_stop_timeout_seconds:
            return {
                "status": STATUS_INTERRUPTED, "event": "interrupted", "ended_at": now,
                "failure_reason": "The CV worker did not confirm it stopped before the timeout.",
            }

    return None


def reconciliation_due(db) -> bool:
    """Read-only check for whether reconcile() would change anything."""
    run = repository.get_in_progress(db)
    if run is None:
        return False
    return decide_reconciliation(run, _cv_snapshot()) is not None


def reconcile(db) -> bool:
    """Align the in-progress run with what the CV worker actually reports.

    A run is never marked active while CV reports a different run ID, and no run
    is left in a transitional state forever. Returns whether anything changed.
    """
    run = repository.get_in_progress(db)
    decision = decide_reconciliation(run, _cv_snapshot())
    if decision is None:
        return False

    repository.set_status(
        db, run, decision["status"],
        started_at=decision.get("started_at"),
        ended_at=decision.get("ended_at"),
        failure_reason=decision.get("failure_reason"),
    )
    repository.append_event(db, run_id=run.run_id, event_type=decision["event"])
    if decision.get("stop_worker"):
        _request_cv_stop_quietly()
    return True


def _request_cv_stop_quietly() -> None:
    try:
        cv_manager.stop_session()
    except CvTransitionError:
        pass
    except Exception:  # pragma: no cover - defensive
        logger.warning("Could not ask the CV worker to stop after a start timeout.")


def recover_after_restart() -> None:
    """Close runs abandoned by a backend restart instead of pretending they resumed."""
    with immediate_write() as db:
        run = repository.get_in_progress(db)
        if run is None:
            return
        cv = _cv_snapshot()
        if cv["state"] == "running" and cv["run_id"] == run.run_id:
            return
        repository.set_status(
            db, run, STATUS_INTERRUPTED,
            ended_at=repository.utc_now(),
            failure_reason="Server restarted before the run was closed.",
        )
        repository.append_event(db, run_id=run.run_id, event_type="interrupted")
        logger.warning("Run %s was marked interrupted after a backend restart.", run.run_id)


def run_startup_tasks() -> None:
    """Backfill legacy runs, then recover any abandoned in-progress run."""
    with immediate_write() as db:
        imported = repository.backfill_legacy_runs(db)
    if imported:
        logger.info("Imported %d pre-existing run(s) as legacy: %s", len(imported), ", ".join(imported))
    recover_after_restart()
    if settings.allow_unmanaged_run_ingestion:
        logger.warning(
            "ALLOW_UNMANAGED_RUN_INGESTION is enabled: CV data published outside Run "
            "Manager is accepted and recorded as an external/unmanaged run. Set it to "
            "false once every launch path starts through the dashboard."
        )


# --- Lifecycle -----------------------------------------------------------


def start_run(
    *,
    actor_user_id: int | None,
    run_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    is_demo: bool = False,
) -> dict:
    with _lifecycle_lock:
        cv = _cv_snapshot()
        if not cv["ready"] or cv["state"] != "ready":
            raise RunConflictError(
                f"The CV worker is not ready to start a run (currently {cv['state']})."
            )

        with immediate_write() as db:
            reconcile(db)
            if repository.get_in_progress(db) is not None:
                raise RunConflictError("Another run is already in progress.")

            chosen = (run_id or "").strip() or repository.generate_run_id(db)
            if not repository.is_valid_run_id(chosen):
                raise RunValidationError(
                    "Run ID may use only letters, numbers, underscore, and hyphen (max 80)."
                )
            if repository.is_tombstoned(db, chosen):
                raise RunConflictError("That run ID was permanently deleted and cannot be reused.")
            if repository.get_by_run_id(db, chosen) is not None:
                raise RunConflictError("That run ID already exists.")

            run = repository.create_run(
                db,
                run_id=chosen,
                status=STATUS_STARTING,
                origin_type=ORIGIN_MANAGED,
                name=(name or None),
                description=(description or None),
                is_demo=is_demo,
                created_by_user_id=actor_user_id,
            )
            repository.append_event(
                db, run_id=chosen, event_type="created", actor_user_id=actor_user_id
            )
            repository.append_event(
                db, run_id=chosen, event_type="start_requested", actor_user_id=actor_user_id
            )
            run_snapshot = serialize_run(run)

        # The run row is committed before the worker is touched, so a crash here
        # leaves a visible `starting` run that reconciliation will resolve.
        try:
            cv_manager.start_session(run_snapshot["run_id"])
        except (CvTransitionError, Exception) as error:
            with immediate_write() as db:
                failed = repository.get_by_run_id(db, run_snapshot["run_id"])
                if failed is not None:
                    repository.set_status(
                        db, failed, STATUS_FAILED,
                        ended_at=repository.utc_now(),
                        failure_reason=str(error)[:500],
                    )
                    repository.append_event(
                        db, run_id=failed.run_id, event_type="failed", actor_user_id=actor_user_id
                    )
                    run_snapshot = serialize_run(failed)
            raise RunConflictError(f"The CV worker rejected the start request: {error}") from error

        return run_snapshot


def end_run(*, run_id: str, actor_user_id: int | None) -> dict:
    with _lifecycle_lock:
        with immediate_write() as db:
            run = repository.get_by_run_id(db, run_id)
            if run is None:
                raise RunNotFoundError("That run was not found.")
            if run.status == STATUS_ENDED:
                return serialize_run(run)
            if run.status not in IN_PROGRESS_STATUSES:
                raise RunConflictError(f"A {run.status} run cannot be ended.")

            cv = _cv_snapshot()
            if cv["run_id"] and cv["run_id"] != run.run_id and cv["state"] == "running":
                raise RunConflictError("The CV worker is running a different run.")

            repository.set_status(db, run, STATUS_ENDING)
            repository.append_event(
                db, run_id=run.run_id, event_type="end_requested", actor_user_id=actor_user_id
            )
            snapshot = serialize_run(run)

        try:
            cv_manager.stop_session()
        except Exception as error:
            with immediate_write() as db:
                run = repository.get_by_run_id(db, run_id)
                if run is not None:
                    cv = _cv_snapshot()
                    truthful = STATUS_ACTIVE if cv["state"] == "running" else STATUS_FAILED
                    repository.set_status(
                        db, run, truthful,
                        ended_at=None if truthful == STATUS_ACTIVE else repository.utc_now(),
                        failure_reason=str(error)[:500],
                    )
                    snapshot = serialize_run(run)
            raise RunConflictError(f"The CV worker rejected the stop request: {error}") from error

        return snapshot


# --- Guarded ingestion ---------------------------------------------------


def resolve_ingestion_run(db, run_id: str) -> object:
    """Decide whether a run-scoped payload may be written, inside the caller's
    BEGIN IMMEDIATE transaction.

    Raises IngestionRejected when the payload must be dropped.
    """
    run_id = (run_id or "").strip()
    if not run_id:
        raise IngestionRejected("empty run id")

    if repository.is_tombstoned(db, run_id):
        raise IngestionRejected(f"run '{run_id}' was deleted")

    in_progress = repository.get_in_progress(db)
    if in_progress is not None:
        if in_progress.run_id != run_id:
            raise IngestionRejected(
                f"run id mismatch: managed run '{in_progress.run_id}' is active"
            )
        repository.touch_ingestion(db, in_progress)
        return in_progress

    existing = repository.get_by_run_id(db, run_id)
    if existing is not None:
        # Data for a known but not-in-progress run is kept, and only updates the
        # observed data interval. Its recorded lifecycle is left untouched.
        repository.touch_ingestion(db, existing)
        return existing

    if not settings.allow_unmanaged_run_ingestion:
        raise IngestionRejected(f"unknown run '{run_id}' and unmanaged ingestion is disabled")

    active_external = repository.get_recently_active_external(
        db, window_seconds=settings.external_run_active_window_seconds
    )
    for other in active_external:
        if other.run_id != run_id:
            raise IngestionRejected(
                f"external run '{other.run_id}' is already active; refusing a competing run"
            )
    return repository.register_external_run(db, run_id)


# --- Deletion ------------------------------------------------------------


_UPLOAD_ROOTS = (EVACUEE_UPLOAD_DIR, OBSERVATION_UPLOAD_DIR)


def _resolve_deletable_file(stored_path: str) -> tuple[Path, str] | None:
    """Return (absolute path, relative storage key) when safely inside a root."""
    for root in _UPLOAD_ROOTS:
        try:
            resolved_root = root.resolve(strict=True)
            candidate = Path(stored_path).resolve(strict=True)
            relative = candidate.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        if candidate.is_file():
            return candidate, str(relative)
    return None


def delete_run(*, run_id: str, confirm_run_id: str, actor_user_id: int | None) -> dict:
    if confirm_run_id != run_id:
        raise RunValidationError("The typed run ID does not match.")

    with immediate_write() as db:
        run = repository.get_by_run_id(db, run_id)
        if run is None:
            raise RunNotFoundError("That run was not found.")
        if run.status in IN_PROGRESS_STATUSES:
            raise RunConflictError("A run that is starting, active, or ending cannot be deleted.")

        # Trust the worker over the database row if they disagree.
        cv = _cv_snapshot()
        if cv["run_id"] == run_id and cv["state"] in {"starting", "running", "stopping"}:
            raise RunConflictError("The CV worker is still processing this run.")

        if run.origin_type == "external":
            quiet_since = _seconds_since(run.last_ingested_at)
            if run.last_ingested_at is not None and quiet_since < settings.external_run_active_window_seconds:
                raise RunConflictError(
                    "This external run is still receiving data. Stop the external CV "
                    "process and wait a few seconds before deleting it."
                )

        image_paths = repository.collect_image_paths(db, run_id)
        deletable = [entry for entry in (_resolve_deletable_file(p) for p in image_paths) if entry]

        counts = repository.delete_run_rows(db, run_id)
        repository.create_tombstone(db, run_id, deleted_by_user_id=actor_user_id)
        repository.append_event(
            db, run_id=run_id, event_type="deleted", actor_user_id=actor_user_id
        )
        db.delete(run)

    # Files are removed only after the database commit; SQLite can roll rows
    # back, but a deleted file cannot be restored.
    deleted_images = 0
    failures: list[tuple[str, str]] = []
    for absolute, storage_key in deletable:
        try:
            absolute.unlink()
            deleted_images += 1
        except FileNotFoundError:
            deleted_images += 1  # already gone counts as clean
        except OSError as error:
            failures.append((storage_key, type(error).__name__))

    if failures:
        with immediate_write() as db:
            for storage_key, code in failures:
                repository.record_pending_file_deletion(
                    db, run_id=run_id, storage_key=storage_key, safe_error_code=code
                )
            repository.append_event(
                db,
                run_id=run_id,
                event_type="file_cleanup_warning",
                safe_detail=f"{len(failures)} evidence file(s) could not be removed.",
            )

    _clear_in_memory_state(run_id)

    warnings = []
    if failures:
        warnings.append(
            f"{len(failures)} evidence file(s) could not be removed and were left for "
            "maintenance cleanup."
        )
    return {
        "run_id": run_id,
        **counts,
        "deleted_images": deleted_images,
        "file_cleanup_failures": len(failures),
        "pending_file_deletions": len(failures),
        "file_cleanup_warnings": warnings,
    }


def _clear_in_memory_state(run_id: str) -> None:
    """Drop cached live state for this run only."""
    from mqtt_bridge import mqtt_bridge
    from tactical_state import tactical_store

    try:
        tactical_store.clear_run(run_id)
    except Exception:  # pragma: no cover - cache clearing must never break deletion
        logger.warning("Could not clear tactical state for the deleted run.")
    try:
        mqtt_bridge.clear_run_cache(run_id)
    except Exception:  # pragma: no cover
        logger.warning("Could not clear MQTT caches for the deleted run.")


# --- Serialization -------------------------------------------------------


def serialize_run(run, summary: dict | None = None) -> dict:
    started = repository.as_utc(run.started_at)
    ended = repository.as_utc(run.ended_at)
    first_ingested = repository.as_utc(run.first_ingested_at)
    last_ingested = repository.as_utc(run.last_ingested_at)

    if started is not None:
        duration_end = ended or repository.utc_now()
        duration_seconds = max(0.0, (duration_end - started).total_seconds())
    elif first_ingested is not None and last_ingested is not None:
        duration_seconds = max(0.0, (last_ingested - first_ingested).total_seconds())
    else:
        duration_seconds = 0.0

    counts = summary or dict(repository.EMPTY_SUMMARY)
    return {
        "run_id": run.run_id,
        "name": run.name,
        "description": run.description,
        "status": run.status,
        "origin_type": run.origin_type,
        "is_demo": bool(run.is_demo),
        "created_by_user_id": run.created_by_user_id,
        "requested_at": repository.as_utc(run.requested_at),
        "started_at": started,
        "ended_at": ended,
        "first_ingested_at": first_ingested,
        "last_ingested_at": last_ingested,
        "failure_reason": run.failure_reason,
        "duration_seconds": duration_seconds,
        "is_in_progress": run.status in IN_PROGRESS_STATUSES,
        "can_delete": run.status not in IN_PROGRESS_STATUSES,
        "can_export": True,
        **counts,
    }
