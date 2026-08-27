from __future__ import annotations

import base64
import binascii
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from auth.dependencies import AdminCsrfUser, CurrentUser, require_cv_service
from database import get_db
from evacuees import repository
from evacuees.storage import delete_gallery_image, save_gallery_image
from models import EvacueeIdentityRead, EvacueeIdentityUpsert, EvacueeSummary
from runs.service import IngestionRejected, resolve_ingestion_run
from runs.write_guard import RunWriteConflict, immediate_write


router = APIRouter(prefix="/api/evacuees", tags=["Passenger Assistance"])
DbSession = Annotated[Session, Depends(get_db)]


def _rejected(reason: IngestionRejected) -> HTTPException:
    """Run policy refused this evidence upload."""
    return HTTPException(status_code=409, detail=f"Evidence was not accepted: {reason}")


def _decode_feature(feature_b64: str | None, feature_dimension: int | None) -> bytes | None:
    if not feature_b64:
        return None
    if len(feature_b64) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="ReID feature payload is too large.")
    try:
        feature_blob = base64.b64decode(feature_b64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="ReID feature is not valid base64.") from exc
    if feature_dimension is not None and len(feature_blob) != feature_dimension * 4:
        raise HTTPException(
            status_code=400,
            detail="ReID feature byte length does not match its float32 dimension.",
        )
    return feature_blob


@router.get("", response_model=list[EvacueeIdentityRead])
def get_evacuees(
    db: DbSession,
    _user: CurrentUser,
    gender: str | None = Query(default=None),
    min_age: float | None = Query(default=None, ge=0, le=120),
    max_age: float | None = Query(default=None, ge=0, le=120),
    camera_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict]:
    if min_age is not None and max_age is not None and min_age > max_age:
        raise HTTPException(status_code=400, detail="min_age cannot be greater than max_age.")
    return repository.list_identities(
        db,
        gender=gender,
        min_age=min_age,
        max_age=max_age,
        camera_id=camera_id,
        run_id=run_id,
        status=status,
        limit=limit,
    )


@router.get("/summary", response_model=EvacueeSummary)
def get_evacuee_summary(
    db: DbSession,
    _user: CurrentUser,
    run_id: str | None = Query(default=None),
) -> dict[str, int]:
    return repository.get_summary(db, run_id=run_id)


@router.get("/reid-gallery", dependencies=[Depends(require_cv_service)])
def get_reid_gallery(
    db: DbSession,
    run_id: str = Query(default="default", min_length=1, max_length=80),
) -> dict:
    return repository.export_reid_gallery(db, run_id=run_id)


@router.get("/{evacuee_id}", response_model=EvacueeIdentityRead)
def get_evacuee(evacuee_id: int, db: DbSession, _user: CurrentUser) -> dict:
    identity = repository.get_identity(db, evacuee_id)
    if identity is None:
        raise HTTPException(status_code=404, detail="Evacuee identity was not found.")
    return identity


@router.put(
    "/by-master/{run_id}/{master_identity_id}",
    response_model=EvacueeIdentityRead,
    dependencies=[Depends(require_cv_service)],
)
def put_evacuee_identity(
    run_id: str,
    master_identity_id: int,
    payload: EvacueeIdentityUpsert,
) -> dict:
    if not run_id.strip() or len(run_id) > 80:
        raise HTTPException(status_code=400, detail="run_id must contain 1 to 80 characters.")
    if master_identity_id < 1:
        raise HTTPException(status_code=400, detail="master_identity_id must be positive.")

    # Run policy and the insert share one serialized transaction so a deleted
    # run cannot be recreated by a late upload.
    try:
        with immediate_write() as db:
            resolve_ingestion_run(db, run_id.strip())
            identity = repository.upsert_identity(
                db,
                run_id=run_id.strip(),
                master_identity_id=master_identity_id,
                payload=payload,
            )
            return repository.serialize_identities(db, [identity])[0]
    except IngestionRejected as reason:
        raise _rejected(reason) from reason
    except RunWriteConflict as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.put(
    "/by-master/{run_id}/{master_identity_id}/views/{view_type}",
    response_model=EvacueeIdentityRead,
    dependencies=[Depends(require_cv_service)],
)
async def put_evacuee_gallery_view(
    run_id: str,
    master_identity_id: int,
    view_type: str,
    image: UploadFile = File(...),
    feature_b64: str | None = Form(default=None),
    feature_dimension: int | None = Form(default=None, ge=1, le=65536),
    feature_space_id: str | None = Form(default=None, max_length=160),
    feature_source: str | None = Form(default=None, max_length=64),
    digest: str | None = Form(default=None, max_length=64),
    captured_at: datetime | None = Form(default=None),
    captured_frame: int | None = Form(default=None, ge=0),
    camera_id: str | None = Form(default=None, max_length=80),
    sharpness: float | None = Form(default=None, ge=0),
    detection_confidence: float | None = Form(default=None, ge=0, le=1),
) -> dict:
    normalized_view = view_type.strip().lower()
    if normalized_view not in repository.VIEW_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"view_type must be one of: {', '.join(repository.VIEW_TYPES)}.",
        )

    feature_blob = _decode_feature(feature_b64, feature_dimension)
    image_path, image_url = await save_gallery_image(
        image,
        run_id=run_id,
        master_identity_id=master_identity_id,
        view_type=normalized_view,
    )
    try:
        with immediate_write() as db:
            resolve_ingestion_run(db, run_id)
            identity = repository.get_by_master(
                db, run_id=run_id, master_identity_id=master_identity_id
            )
            if identity is None:
                identity = repository.upsert_identity(
                    db,
                    run_id=run_id,
                    master_identity_id=master_identity_id,
                    payload=EvacueeIdentityUpsert(last_camera_id=camera_id),
                )

            _, previous_image_path = repository.upsert_gallery_view(
                db,
                identity=identity,
                view_type=normalized_view,
                image_path=str(image_path),
                image_url=image_url,
                feature_blob=feature_blob,
                feature_dimension=feature_dimension,
                feature_space_id=feature_space_id,
                feature_source=feature_source,
                digest=digest,
                captured_at=captured_at,
                captured_frame=captured_frame,
                camera_id=camera_id,
                sharpness=sharpness,
                detection_confidence=detection_confidence,
            )
            refreshed = repository.get_by_master(
                db, run_id=run_id, master_identity_id=master_identity_id
            )
            payload = repository.serialize_identities(db, [refreshed])[0]
    except IngestionRejected as reason:
        delete_gallery_image(image_path)
        raise _rejected(reason) from reason
    except RunWriteConflict as error:
        delete_gallery_image(image_path)
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception:
        delete_gallery_image(image_path)
        raise

    if previous_image_path and previous_image_path != str(image_path):
        delete_gallery_image(previous_image_path)
    return payload


@router.delete("")
def delete_evacuees(
    db: DbSession,
    _admin: AdminCsrfUser,
    run_id: str | None = Query(default=None),
) -> dict[str, int]:
    deleted_rows, image_paths = repository.clear_identities(db, run_id=run_id)
    db.commit()
    deleted_images = sum(1 for path in image_paths if delete_gallery_image(path))
    return {"deleted_rows": deleted_rows, "deleted_images": deleted_images}
