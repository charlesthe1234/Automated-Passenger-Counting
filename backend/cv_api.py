"""User-facing CV session API.

Start/Stop is authorized by an authenticated administrator session plus CSRF on
both loopback and LAN. The operator no longer supplies a second shared token;
the legacy token path survives only behind an explicit compatibility setting.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.dependencies import (
    AuthContext,
    optional_auth,
    require_auth,
    require_high_risk_control,
)
from auth.models import ROLE_ADMIN
from config import settings
from cv_manager import CvTransitionError, cv_manager


router = APIRouter(prefix="/api/cv", tags=["computer-vision"])


class CvSessionStart(BaseModel):
    run_id: str | None = Field(default=None, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")


class CvStatus(BaseModel):
    state: str
    ready: bool
    running: bool
    run_id: str | None
    started_at: str | None
    stopped_at: str | None
    pid: int | None
    loading_stage: str | None
    error: str | None
    mqtt_broker_reachable: bool
    control_allowed: bool
    control_mode: str


def _status_for_context(context: AuthContext | None) -> dict:
    """Report worker state plus whether this caller may operate the session."""
    status = cv_manager.status()
    status["control_allowed"] = context is not None and context.user.role == ROLE_ADMIN
    status["control_mode"] = (
        "legacy_token" if settings.cv_control_legacy_token_enabled else "admin_session"
    )
    return status


@router.get("/status", response_model=CvStatus)
def get_cv_status(context: Annotated[AuthContext, Depends(require_auth)]) -> dict:
    return _status_for_context(context)


@router.post(
    "/session/start",
    response_model=CvStatus,
    dependencies=[Depends(require_high_risk_control)],
)
def start_cv_session(
    payload: CvSessionStart,
    context: Annotated[AuthContext | None, Depends(optional_auth)],
) -> dict:
    """Compatibility route: delegates to Run Manager so a managed run always exists.

    It never calls `cv_manager` directly, so there is no supported path that
    starts CV without run bookkeeping and an audit trail.
    """
    from runs import service as run_service

    actor_id = context.user.id if context is not None else None
    try:
        run_service.start_run(
            actor_user_id=actor_id,
            run_id=payload.run_id,
            name="Legacy CV control" if actor_id is None else None,
        )
    except run_service.RunValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except run_service.RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_for_context(context)


@router.post(
    "/session/stop",
    response_model=CvStatus,
    dependencies=[Depends(require_high_risk_control)],
)
def stop_cv_session(
    context: Annotated[AuthContext | None, Depends(optional_auth)],
) -> dict:
    """Compatibility route: closes the managed run rather than only stopping CV."""
    from runs import service as run_service
    from runs.write_guard import immediate_write

    actor_id = context.user.id if context is not None else None
    with immediate_write() as db:
        from runs import repository

        active = repository.get_in_progress(db)
        active_run_id = active.run_id if active is not None else None

    if active_run_id is None:
        # No managed run to close; fall back to stopping the worker itself so a
        # stray session can still be halted.
        try:
            cv_manager.stop_session()
        except CvTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _status_for_context(context)

    try:
        run_service.end_run(run_id=active_run_id, actor_user_id=actor_id)
    except run_service.RunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status_for_context(context)
