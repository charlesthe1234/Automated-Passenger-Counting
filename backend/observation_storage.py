from __future__ import annotations

import shutil
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

UPLOAD_DIR = settings.observation_upload_path
PUBLIC_UPLOAD_PREFIX = "/uploads/observations"


def ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_observation_image(image: UploadFile) -> tuple[Path, str]:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Image must be JPEG, PNG, or WebP.")

    ensure_upload_dir()
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded image must be 5 MB or smaller.")

    suffix = ALLOWED_CONTENT_TYPES[image.content_type]
    filename = f"{uuid4().hex}{suffix}"
    destination = UPLOAD_DIR / filename
    destination.write_bytes(image_bytes)
    return destination, f"{PUBLIC_UPLOAD_PREFIX}/{filename}"


def clear_observation_images() -> int:
    if not UPLOAD_DIR.exists():
        return 0

    removed = 0
    for path in UPLOAD_DIR.iterdir():
        if path.is_file():
            path.unlink()
            removed += 1
        elif path.is_dir():
            shutil.rmtree(path)
            removed += 1
    return removed
