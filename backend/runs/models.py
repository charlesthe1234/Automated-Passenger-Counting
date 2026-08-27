"""SQLAlchemy tables making an operational run a first-class entity.

Existing operational tables keep their plain `run_id` strings as soft
references. No strict foreign key is retrofitted onto them, so data written
before this feature existed — or by an externally launched pipeline — stays
readable.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from database import Base

# Lifecycle states.
STATUS_STARTING = "starting"
STATUS_ACTIVE = "active"
STATUS_ENDING = "ending"
STATUS_ENDED = "ended"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
STATUS_EXTERNAL = "external"

# A run in one of these states owns the CV worker; only one may exist at a time.
IN_PROGRESS_STATUSES = (STATUS_STARTING, STATUS_ACTIVE, STATUS_ENDING)

ORIGIN_MANAGED = "managed"
ORIGIN_LEGACY = "legacy"
ORIGIN_EXTERNAL = "external"

RUN_NAME_MAX_LENGTH = 120
RUN_DESCRIPTION_MAX_LENGTH = 1000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(RUN_NAME_MAX_LENGTH), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    origin_type: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Set only once reconciliation confirms CV is running this exact run.
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Observed data interval, maintained for managed, legacy, and external runs.
    first_ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status_changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class RunEvent(Base):
    """Append-only lifecycle audit. Survives operational run deletion."""

    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )
    safe_detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class DeletedRun(Base):
    """Tombstone preventing a deleted run ID from being silently recreated."""

    __tablename__ = "deleted_runs"

    run_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )


class PendingFileDeletion(Base):
    """A sensitive file that could not be removed after its rows were deleted.

    Holds a validated relative storage key only, never an absolute path, so
    orphaned evidence stays discoverable for maintenance without leaking the
    server's filesystem layout.
    """

    __tablename__ = "pending_file_deletions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    first_failed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    safe_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
