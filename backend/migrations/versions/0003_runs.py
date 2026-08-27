"""Add managed runs, lifecycle audit, deletion tombstones, pending file cleanup.

Adds tables and indexes only. Existing operational rows keep their plain
`run_id` strings and are never rewritten; the idempotent legacy backfill that
imports them as `origin_type="legacy"` runs at application startup, not here,
so it can be re-run safely and tested independently.

Revision ID: 0003_runs
Revises: 0002_auth
Create Date: 2026-07-31

"""
from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_runs"
down_revision: str | None = "0002_auth"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Only one run may be starting/active/ending at any moment. A service-layer
# lock cannot protect against a second backend process or a bypass code path,
# so the constraint is enforced by the database as well. SQLite supports
# partial indexes; indexing a constant expression makes every in-progress row
# collide with every other one.
ONE_ACTIVE_RUN_INDEX = """
CREATE UNIQUE INDEX uq_runs_single_in_progress
ON runs ((1))
WHERE status IN ('starting', 'active', 'ending')
"""


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("is_demo", sa.Boolean(), nullable=False),
        sa.Column("origin_type", sa.String(length=16), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("first_ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_ingested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status_changed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_runs_id", "runs", ["id"])
    op.create_index("ix_runs_run_id", "runs", ["run_id"], unique=True)
    op.create_index("ix_runs_status", "runs", ["status"])
    op.create_index("ix_runs_origin_type", "runs", ["origin_type"])
    op.execute(ONE_ACTIVE_RUN_INDEX)

    op.create_table(
        "run_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("safe_detail", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_events_id", "run_events", ["id"])
    op.create_index("ix_run_events_run_id", "run_events", ["run_id"])
    op.create_index("ix_run_events_event_type", "run_events", ["event_type"])
    op.create_index("ix_run_events_timestamp", "run_events", ["timestamp"])

    op.create_table(
        "deleted_runs",
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_by_user_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["deleted_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("run_id"),
    )

    op.create_table(
        "pending_file_deletions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("first_failed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("safe_error_code", sa.String(length=64), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pending_file_deletions_id", "pending_file_deletions", ["id"])
    op.create_index("ix_pending_file_deletions_run_id", "pending_file_deletions", ["run_id"])


def downgrade() -> None:
    op.drop_table("pending_file_deletions")
    op.drop_table("deleted_runs")
    op.drop_table("run_events")
    op.execute("DROP INDEX IF EXISTS uq_runs_single_in_progress")
    op.drop_table("runs")
