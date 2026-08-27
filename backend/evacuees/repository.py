from __future__ import annotations

import base64
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import case, delete, desc, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from evidence.router import evidence_url_for_gallery_view
from models import (
    EvacueeGalleryView,
    EvacueeIdentity,
    EvacueeIdentityUpsert,
)


VIEW_TYPES = ("baseline", "front", "back", "left_side", "right_side")
DISPLAY_ORDER = {"front": 0, "left_side": 1, "right_side": 2, "back": 3, "baseline": 4}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _normalize_gender(value: str | None) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized or "unknown"


def get_by_master(db: Session, *, run_id: str, master_identity_id: int) -> EvacueeIdentity | None:
    statement = select(EvacueeIdentity).where(
        EvacueeIdentity.run_id == run_id,
        EvacueeIdentity.master_identity_id == master_identity_id,
    )
    return db.scalar(statement)


def upsert_identity(
    db: Session,
    *,
    run_id: str,
    master_identity_id: int,
    payload: EvacueeIdentityUpsert,
) -> EvacueeIdentity:
    now = _utc_now()
    normalized_age = payload.age if payload.age is not None and payload.age > 0 else None
    normalized_gender = _normalize_gender(payload.gender)
    insert_statement = sqlite_insert(EvacueeIdentity).values(
        run_id=run_id,
        master_identity_id=master_identity_id,
        role=payload.role.strip().lower(),
        role_confidence=payload.role_confidence,
        age=normalized_age,
        gender=normalized_gender,
        first_seen_at=payload.first_seen_at or now,
        last_seen_at=payload.last_seen_at or now,
        last_camera_id=payload.last_camera_id,
        current_status=payload.current_status.strip().lower(),
    )
    excluded = insert_statement.excluded
    statement = insert_statement.on_conflict_do_update(
        index_elements=[EvacueeIdentity.run_id, EvacueeIdentity.master_identity_id],
        set_={
            "role": excluded.role,
            "role_confidence": func.coalesce(excluded.role_confidence, EvacueeIdentity.role_confidence),
            "age": func.coalesce(excluded.age, EvacueeIdentity.age),
            "gender": case(
                (excluded.gender.in_(["unknown", "pending", "disabled"]), EvacueeIdentity.gender),
                else_=excluded.gender,
            ),
            "last_seen_at": excluded.last_seen_at,
            "last_camera_id": func.coalesce(excluded.last_camera_id, EvacueeIdentity.last_camera_id),
            "current_status": excluded.current_status,
        },
    )
    db.execute(statement)
    db.flush()
    identity = get_by_master(db, run_id=run_id, master_identity_id=master_identity_id)
    if identity is None:
        raise RuntimeError("Evacuee identity upsert did not return a row.")
    return identity


def upsert_gallery_view(
    db: Session,
    *,
    identity: EvacueeIdentity,
    view_type: str,
    image_path: str,
    image_url: str,
    feature_blob: bytes | None,
    feature_dimension: int | None,
    feature_space_id: str | None,
    feature_source: str | None,
    digest: str | None,
    captured_at: datetime | None,
    captured_frame: int | None,
    camera_id: str | None,
    sharpness: float | None,
    detection_confidence: float | None,
) -> tuple[EvacueeGalleryView, str | None]:
    statement = select(EvacueeGalleryView).where(
        EvacueeGalleryView.evacuee_id == identity.id,
        EvacueeGalleryView.view_type == view_type,
    )
    view = db.scalar(statement)
    previous_image_path = view.image_path if view is not None else None
    if view is None:
        view = EvacueeGalleryView(evacuee_id=identity.id, view_type=view_type)
        db.add(view)

    view.image_path = image_path
    view.image_url = image_url
    view.feature_blob = feature_blob
    view.feature_dimension = feature_dimension
    view.feature_space_id = feature_space_id
    view.feature_source = feature_source
    view.digest = digest
    view.captured_at = captured_at or _utc_now()
    view.captured_frame = captured_frame
    view.camera_id = camera_id
    view.sharpness = sharpness
    view.detection_confidence = detection_confidence

    if camera_id:
        identity.last_camera_id = camera_id
    identity.last_seen_at = max(_as_utc(identity.last_seen_at), _as_utc(view.captured_at))

    db.flush()
    db.refresh(view)
    return view, previous_image_path


def _view_payload(view: EvacueeGalleryView) -> dict:
    return {
        "id": view.id,
        "view_type": view.view_type,
        # Generated per response so evidence is only reachable through the
        # authenticated route, never through the stored static path.
        "image_url": evidence_url_for_gallery_view(view.id),
        "captured_at": _as_utc(view.captured_at),
        "captured_frame": view.captured_frame,
        "camera_id": view.camera_id,
        "sharpness": view.sharpness,
        "detection_confidence": view.detection_confidence,
    }


def _best_primary_view(views: list[EvacueeGalleryView]) -> EvacueeGalleryView | None:
    by_type = {view.view_type: view for view in views}
    if by_type.get("front") is not None:
        return by_type["front"]

    side_views = [view for view in views if view.view_type in {"left_side", "right_side"}]
    if side_views:
        return max(
            side_views,
            key=lambda item: (float(item.sharpness or 0), float(item.detection_confidence or 0)),
        )
    return by_type.get("baseline") or by_type.get("back")


def serialize_identities(db: Session, identities: list[EvacueeIdentity]) -> list[dict]:
    if not identities:
        return []

    identity_ids = [identity.id for identity in identities]
    views = list(
        db.scalars(
            select(EvacueeGalleryView).where(EvacueeGalleryView.evacuee_id.in_(identity_ids))
        ).all()
    )
    views_by_identity: dict[int, list[EvacueeGalleryView]] = defaultdict(list)
    for view in views:
        views_by_identity[view.evacuee_id].append(view)

    results = []
    for identity in identities:
        identity_views = sorted(
            views_by_identity.get(identity.id, []),
            key=lambda item: DISPLAY_ORDER.get(item.view_type, 99),
        )
        primary = _best_primary_view(identity_views)
        results.append(
            {
                "id": identity.id,
                "run_id": identity.run_id,
                "master_identity_id": identity.master_identity_id,
                "role": identity.role,
                "role_confidence": identity.role_confidence,
                "age": identity.age,
                "gender": identity.gender,
                "first_seen_at": _as_utc(identity.first_seen_at),
                "last_seen_at": _as_utc(identity.last_seen_at),
                "last_camera_id": identity.last_camera_id,
                "current_status": identity.current_status,
                "gallery_filled": len(identity_views),
                "gallery_total": len(VIEW_TYPES),
                "primary_view": _view_payload(primary) if primary else None,
                "views": [_view_payload(view) for view in identity_views],
            }
        )
    return results


def list_identities(
    db: Session,
    *,
    gender: str | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
    camera_id: str | None = None,
    run_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    statement = select(EvacueeIdentity).where(EvacueeIdentity.role == "evacuee")
    if gender:
        normalized_gender = _normalize_gender(gender)
        if normalized_gender == "unknown":
            statement = statement.where(EvacueeIdentity.gender.not_in(["male", "female"]))
        else:
            statement = statement.where(EvacueeIdentity.gender == normalized_gender)
    if min_age is not None:
        statement = statement.where(EvacueeIdentity.age >= min_age)
    if max_age is not None:
        statement = statement.where(EvacueeIdentity.age <= max_age)
    if camera_id:
        statement = statement.where(EvacueeIdentity.last_camera_id == camera_id)
    if run_id:
        statement = statement.where(EvacueeIdentity.run_id == run_id)
    if status:
        statement = statement.where(EvacueeIdentity.current_status == status.strip().lower())

    statement = statement.order_by(desc(EvacueeIdentity.last_seen_at), desc(EvacueeIdentity.id)).limit(limit)
    identities = list(db.scalars(statement).all())
    return serialize_identities(db, identities)


def list_staff_identities(
    db: Session,
    *,
    run_id: str,
    limit: int = 200,
) -> list[dict]:
    """List all model-predicted CAG and SCDF identities for one run."""

    statement = (
        select(EvacueeIdentity)
        .where(
            EvacueeIdentity.run_id == run_id,
            EvacueeIdentity.role.in_(("cag", "scdf")),
        )
        .order_by(desc(EvacueeIdentity.last_seen_at), desc(EvacueeIdentity.id))
        .limit(limit)
    )
    identities = list(db.scalars(statement).all())
    return serialize_identities(db, identities)


def get_identity(db: Session, evacuee_id: int) -> dict | None:
    identity = db.get(EvacueeIdentity, evacuee_id)
    if identity is None:
        return None
    return serialize_identities(db, [identity])[0]


def get_summary(db: Session, *, run_id: str | None = None) -> dict[str, int]:
    statement = select(
        func.count(EvacueeIdentity.id).label("total_analyzed"),
        func.coalesce(func.sum(case((EvacueeIdentity.gender == "male", 1), else_=0)), 0).label("males"),
        func.coalesce(func.sum(case((EvacueeIdentity.gender == "female", 1), else_=0)), 0).label("females"),
        func.coalesce(
            func.sum(case((EvacueeIdentity.gender.not_in(["male", "female"]), 1), else_=0)),
            0,
        ).label("unknown"),
        func.coalesce(func.sum(case((EvacueeIdentity.age < 18, 1), else_=0)), 0).label("minors"),
    ).where(EvacueeIdentity.role == "evacuee")
    if run_id:
        statement = statement.where(EvacueeIdentity.run_id == run_id)
    row = db.execute(statement).one()
    return {name: int(getattr(row, name) or 0) for name in ("total_analyzed", "males", "females", "unknown", "minors")}


def export_reid_gallery(db: Session, *, run_id: str) -> dict:
    identities = list(
        db.scalars(
            select(EvacueeIdentity)
            .where(EvacueeIdentity.run_id == run_id)
            .order_by(EvacueeIdentity.master_identity_id)
        ).all()
    )
    if not identities:
        return {"schema_version": 3, "run_id": run_id, "identities": {}}

    identity_ids = [identity.id for identity in identities]
    views = list(
        db.scalars(
            select(EvacueeGalleryView).where(EvacueeGalleryView.evacuee_id.in_(identity_ids))
        ).all()
    )
    views_by_identity: dict[int, list[EvacueeGalleryView]] = defaultdict(list)
    for view in views:
        views_by_identity[view.evacuee_id].append(view)

    payload: dict[str, dict] = {}
    for identity in identities:
        gallery = {view_type: None for view_type in VIEW_TYPES}
        for view in views_by_identity.get(identity.id, []):
            gallery[view.view_type] = {
                "feature_b64": base64.b64encode(view.feature_blob).decode("ascii") if view.feature_blob else None,
                "feature_source": view.feature_source,
                "feature_space_id": view.feature_space_id,
                "feature_dimension": view.feature_dimension,
                "image_url": view.image_url,
                "digest": view.digest,
                "captured_frame": view.captured_frame,
                "captured_at": _as_utc(view.captured_at).isoformat(),
                "camera_id": view.camera_id,
                "sharpness": view.sharpness,
                "detection_confidence": view.detection_confidence,
            }
        payload[str(identity.master_identity_id)] = {
            "role": identity.role,
            "role_confidence": identity.role_confidence or 0.0,
            "age": identity.age if identity.age is not None else "Unknown",
            "gender": identity.gender,
            "gallery": gallery,
            "hits": 0,
        }
    return {"schema_version": 3, "run_id": run_id, "identities": payload}


def clear_identities(db: Session, *, run_id: str | None = None) -> tuple[int, list[str]]:
    identity_statement = select(EvacueeIdentity)
    if run_id:
        identity_statement = identity_statement.where(EvacueeIdentity.run_id == run_id)
    identities = list(db.scalars(identity_statement).all())
    if not identities:
        return 0, []

    identity_ids = [identity.id for identity in identities]
    image_paths = list(
        db.scalars(
            select(EvacueeGalleryView.image_path).where(EvacueeGalleryView.evacuee_id.in_(identity_ids))
        ).all()
    )
    db.execute(delete(EvacueeGalleryView).where(EvacueeGalleryView.evacuee_id.in_(identity_ids)))
    db.execute(delete(EvacueeIdentity).where(EvacueeIdentity.id.in_(identity_ids)))
    db.flush()
    return len(identities), image_paths
