"""Database-backed, authenticated evidence image routes.

Passenger evidence is deliberately not served from a static mount. Every request
authenticates the browser user, looks the record up in SQLite, re-validates that
the stored file still resolves inside the configured upload root, and returns a
generic 404 that never leaks filesystem detail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth.dependencies import CurrentUser
from database import get_db
from evacuees.storage import UPLOAD_DIR as EVACUEE_UPLOAD_DIR
from models import EvacueeGalleryView, PassengerObservation
from observation_storage import UPLOAD_DIR as OBSERVATION_UPLOAD_DIR

router = APIRouter(prefix="/api/evidence", tags=["evidence"])
DbSession = Annotated[Session, Depends(get_db)]

# Sensitive images must not sit in a shared browser or proxy cache, and a
# cached copy must never be treated as proof of authorization after logout.
NO_STORE_HEADERS = {
    "Cache-Control": "private, no-store, max-age=0",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
}

# Content type is derived from trusted file inspection, never from a value the
# uploader supplied.
_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
)
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

_UNAVAILABLE = "That evidence image is unavailable."


def evidence_url_for_gallery_view(view_id: int) -> str:
    return f"/api/evidence/evacuees/{view_id}"


def evidence_url_for_observation(observation_id: int) -> str:
    return f"/api/evidence/observations/{observation_id}"


def _detect_media_type(path: Path) -> str | None:
    """Return a safe image content type, or None when the file is not an image."""
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        return None
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return None

    for prefix, media_type in _MAGIC_PREFIXES:
        if header.startswith(prefix):
            return media_type
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _resolve_within_root(root: Path, stored_path: str | None) -> Path | None:
    """Resolve a stored path and prove it stays inside the configured root.

    `resolve()` follows symlinks, so a link pointing outside the root resolves
    outside it and fails containment.
    """
    if not stored_path:
        return None
    try:
        resolved_root = root.resolve(strict=True)
        candidate = Path(stored_path).resolve(strict=True)
        candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if not candidate.is_file():
        return None
    return candidate


def _serve(root: Path, stored_path: str | None) -> FileResponse:
    resolved = _resolve_within_root(root, stored_path)
    if resolved is None:
        raise HTTPException(status_code=404, detail=_UNAVAILABLE)
    media_type = _detect_media_type(resolved)
    if media_type is None:
        raise HTTPException(status_code=404, detail=_UNAVAILABLE)
    return FileResponse(resolved, media_type=media_type, headers=NO_STORE_HEADERS)


@router.get("/evacuees/{view_id}")
def get_evacuee_evidence(view_id: int, _user: CurrentUser, db: DbSession) -> FileResponse:
    view = db.get(EvacueeGalleryView, view_id)
    if view is None:
        raise HTTPException(status_code=404, detail=_UNAVAILABLE)
    return _serve(EVACUEE_UPLOAD_DIR, view.image_path)


@router.get("/observations/{observation_id}")
def get_observation_evidence(
    observation_id: int, _user: CurrentUser, db: DbSession
) -> FileResponse:
    observation = db.get(PassengerObservation, observation_id)
    if observation is None:
        raise HTTPException(status_code=404, detail=_UNAVAILABLE)
    return _serve(OBSERVATION_UPLOAD_DIR, observation.image_path)
