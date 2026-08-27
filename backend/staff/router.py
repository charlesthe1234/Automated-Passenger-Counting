from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from auth.dependencies import CurrentUser
from database import get_db
from evacuees import repository
from models import EvacueeIdentityRead


router = APIRouter(prefix="/api/staff", tags=["Staff Review"])
DbSession = Annotated[Session, Depends(get_db)]


@router.get("", response_model=list[EvacueeIdentityRead])
def get_staff_detections(
    db: DbSession,
    _user: CurrentUser,
    run_id: str = Query(min_length=1, max_length=80),
    limit: int = Query(default=200, ge=1, le=200),
) -> list[dict]:
    """Return every CAG/SCDF prediction belonging to one selected run."""

    return repository.list_staff_identities(db, run_id=run_id, limit=limit)
