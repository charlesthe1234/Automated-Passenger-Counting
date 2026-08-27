"""Run history, lifecycle, and deletion endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from auth.dependencies import AdminCsrfUser, CurrentUser
from config import settings
from database import get_db
from runs import repository, service
from runs.schemas import (
    RunDeleteRequest,
    RunDeleteResponse,
    RunListResponse,
    RunRead,
    RunStartRequest,
)
from runs.write_guard import RunWriteConflict, immediate_write

router = APIRouter(prefix="/api/runs", tags=["runs"])
DbSession = Annotated[Session, Depends(get_db)]


def _translate(error: Exception) -> HTTPException:
    if isinstance(error, service.RunNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, service.RunValidationError):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, service.RunConflictError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, RunWriteConflict):
        return HTTPException(status_code=503, detail=str(error))
    raise error


@router.get("", response_model=RunListResponse)
def list_runs(
    db: DbSession,
    _user: CurrentUser,
    status: str | None = Query(default=None),
    origin_type: str | None = Query(default=None),
    is_demo: bool | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> RunListResponse:
    rows, total = repository.list_runs(
        db, status=status, origin_type=origin_type, is_demo=is_demo, limit=limit, offset=offset
    )
    summaries = repository.collect_summaries(db)
    items = [
        RunRead(**service.serialize_run(run, summaries.get(run.run_id, dict(repository.EMPTY_SUMMARY))))
        for run in rows
    ]
    return RunListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/active", response_model=RunRead | None)
def get_active_run(db: DbSession, _user: CurrentUser) -> dict | None:
    # Every connected browser polls this endpoint. Taking SQLite's exclusive
    # write lock each time would serialise all viewers against each other and
    # against MQTT ingestion, so the transition is checked read-only first and
    # the lock is taken only when a state change is genuinely due.
    if service.reconciliation_due(db):
        try:
            with immediate_write() as write_db:
                service.reconcile(write_db)
        except RunWriteConflict:
            pass
        db.expire_all()

    run = repository.get_in_progress(db)
    if run is None:
        return None
    summaries = repository.collect_summaries(db)
    return service.serialize_run(run, summaries.get(run.run_id, dict(repository.EMPTY_SUMMARY)))


@router.get("/external-active", response_model=RunRead | None)
def get_external_active_run(db: DbSession, _user: CurrentUser, response: Response) -> dict | None:
    """Report observed external data activity only.

    This never claims Run Manager owns, started, or can stop that process.
    """
    active = repository.get_recently_active_external(
        db, window_seconds=settings.external_run_active_window_seconds
    )
    if not active:
        return None
    if len(active) > 1:
        # Two competing publishers is a degraded state, not something to
        # silently paper over by picking one.
        response.headers["X-Run-Conflict"] = "multiple-external-runs"
    summaries = repository.collect_summaries(db)
    run = active[0]
    return service.serialize_run(run, summaries.get(run.run_id, dict(repository.EMPTY_SUMMARY)))


@router.get("/{run_id}", response_model=RunRead)
def get_run(run_id: str, db: DbSession, _user: CurrentUser) -> dict:
    run = repository.get_by_run_id(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="That run was not found.")
    summaries = repository.collect_summaries(db)
    return service.serialize_run(run, summaries.get(run.run_id, dict(repository.EMPTY_SUMMARY)))


@router.post("/start", response_model=RunRead, status_code=202)
def start_run(payload: RunStartRequest, admin: AdminCsrfUser) -> dict:
    try:
        return service.start_run(
            actor_user_id=admin.id,
            run_id=payload.run_id,
            name=payload.name,
            description=payload.description,
            is_demo=payload.is_demo,
        )
    except Exception as error:
        raise _translate(error) from error


@router.post("/{run_id}/end", response_model=RunRead, status_code=202)
def end_run(run_id: str, admin: AdminCsrfUser) -> dict:
    try:
        return service.end_run(run_id=run_id, actor_user_id=admin.id)
    except Exception as error:
        raise _translate(error) from error


@router.delete("/{run_id}", response_model=RunDeleteResponse)
def delete_run(run_id: str, payload: RunDeleteRequest, admin: AdminCsrfUser) -> dict:
    try:
        return service.delete_run(
            run_id=run_id,
            confirm_run_id=payload.confirm_run_id,
            actor_user_id=admin.id,
        )
    except Exception as error:
        raise _translate(error) from error
