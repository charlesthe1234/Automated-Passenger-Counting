"""Request and response models for the run API.

No schema accepts an authoritative actor: `created_by_user_id` and every audit
actor are derived from the authenticated session.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from runs.models import RUN_DESCRIPTION_MAX_LENGTH, RUN_NAME_MAX_LENGTH


class RunStartRequest(BaseModel):
    run_id: str | None = Field(default=None, max_length=80)
    name: str | None = Field(default=None, max_length=RUN_NAME_MAX_LENGTH)
    description: str | None = Field(default=None, max_length=RUN_DESCRIPTION_MAX_LENGTH)
    is_demo: bool = False


class RunDeleteRequest(BaseModel):
    confirm_run_id: str = Field(min_length=1, max_length=80)


class RunRead(BaseModel):
    run_id: str
    name: str | None = None
    description: str | None = None
    status: str
    origin_type: str
    is_demo: bool = False
    created_by_user_id: int | None = None
    requested_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    first_ingested_at: datetime | None = None
    last_ingested_at: datetime | None = None
    failure_reason: str | None = None
    duration_seconds: float = 0.0
    is_in_progress: bool = False
    can_delete: bool = True
    can_export: bool = True
    metric_count: int = 0
    alert_count: int = 0
    observation_count: int = 0
    evacuee_count: int = 0
    gallery_view_count: int = 0
    peak_passenger_count: int = 0
    latest_passenger_count: int = 0


class RunListResponse(BaseModel):
    items: list[RunRead]
    total: int
    limit: int
    offset: int


class RunDeleteResponse(BaseModel):
    run_id: str
    deleted_metrics: int = 0
    deleted_alerts: int = 0
    deleted_observations: int = 0
    deleted_evacuees: int = 0
    deleted_gallery_views: int = 0
    deleted_images: int = 0
    file_cleanup_failures: int = 0
    pending_file_deletions: int = 0
    file_cleanup_warnings: list[str] = Field(default_factory=list)
