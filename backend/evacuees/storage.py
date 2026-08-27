from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from config import settings


MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
UPLOAD_DIR = settings.evacuee_upload_path
PUBLIC_UPLOAD_PREFIX = "/uploads/evacuees"


def ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized[:100] or "default"


async def save_gallery_image(
    image: UploadFile,
    *,
    run_id: str,
    master_identity_id: int,
    view_type: str,
) -> tuple[Path, str]:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Gallery image must be JPEG, PNG, or WebP.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded gallery image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Gallery image must be 5 MB or smaller.")

    run_segment = _safe_segment(run_id)
    identity_segment = f"master_{int(master_identity_id):04d}"
    destination_dir = UPLOAD_DIR / run_segment / identity_segment
    destination_dir.mkdir(parents=True, exist_ok=True)

    suffix = ALLOWED_CONTENT_TYPES[image.content_type]
    filename = f"{_safe_segment(view_type)}_{uuid4().hex}{suffix}"
    destination = destination_dir / filename
    destination.write_bytes(image_bytes)
    public_url = f"{PUBLIC_UPLOAD_PREFIX}/{run_segment}/{identity_segment}/{filename}"
    return destination, public_url


def delete_gallery_image(image_path: str | Path | None) -> bool:
    if not image_path:
        return False

    root = UPLOAD_DIR.resolve()
    candidate = Path(image_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False

    if not candidate.is_file():
        return False
    candidate.unlink()

    parent = candidate.parent
    while parent != root:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return True
