"""Reviewed baseline: the operational schema as it existed before login/auth.

An existing pre-auth deployment is stamped at this revision after its database
and evidence directories have been backed up. A fresh database runs it for
real. Either way the next revision adds authentication on top.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-07-31

"""
from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "metric_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("passenger_count", sa.Integer(), nullable=False),
        sa.Column("zone_counts", sa.Text(), nullable=True),
        sa.Column("camera_online_count", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_metric_logs_id", "metric_logs", ["id"])
    op.create_index("ix_metric_logs_timestamp", "metric_logs", ["timestamp"])
    op.create_index("ix_metric_logs_run_id", "metric_logs", ["run_id"])

    op.create_table(
        "system_alerts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_system_alerts_id", "system_alerts", ["id"])
    op.create_index("ix_system_alerts_timestamp", "system_alerts", ["timestamp"])
    op.create_index("ix_system_alerts_run_id", "system_alerts", ["run_id"])
    op.create_index("ix_system_alerts_severity", "system_alerts", ["severity"])

    op.create_table(
        "passenger_observations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("camera_id", sa.String(length=80), nullable=False),
        sa.Column("track_id", sa.String(length=120), nullable=True),
        sa.Column("age", sa.Float(), nullable=False),
        sa.Column("gender", sa.String(length=32), nullable=False),
        sa.Column("age_confidence", sa.Float(), nullable=True),
        sa.Column("gender_confidence", sa.Float(), nullable=True),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("image_url", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_passenger_observations_id", "passenger_observations", ["id"])
    op.create_index("ix_passenger_observations_timestamp", "passenger_observations", ["timestamp"])
    op.create_index("ix_passenger_observations_run_id", "passenger_observations", ["run_id"])
    op.create_index("ix_passenger_observations_camera_id", "passenger_observations", ["camera_id"])
    op.create_index("ix_passenger_observations_track_id", "passenger_observations", ["track_id"])
    op.create_index("ix_passenger_observations_gender", "passenger_observations", ["gender"])

    op.create_table(
        "evacuee_identities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("master_identity_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("role_confidence", sa.Float(), nullable=True),
        sa.Column("age", sa.Float(), nullable=True),
        sa.Column("gender", sa.String(length=32), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_camera_id", sa.String(length=80), nullable=True),
        sa.Column("current_status", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "master_identity_id", name="uq_evacuee_run_master"),
    )
    op.create_index("ix_evacuee_identities_id", "evacuee_identities", ["id"])
    op.create_index("ix_evacuee_identities_run_id", "evacuee_identities", ["run_id"])
    op.create_index("ix_evacuee_identities_master_identity_id", "evacuee_identities", ["master_identity_id"])
    op.create_index("ix_evacuee_identities_role", "evacuee_identities", ["role"])
    op.create_index("ix_evacuee_identities_age", "evacuee_identities", ["age"])
    op.create_index("ix_evacuee_identities_gender", "evacuee_identities", ["gender"])
    op.create_index("ix_evacuee_identities_first_seen_at", "evacuee_identities", ["first_seen_at"])
    op.create_index("ix_evacuee_identities_last_seen_at", "evacuee_identities", ["last_seen_at"])
    op.create_index("ix_evacuee_identities_last_camera_id", "evacuee_identities", ["last_camera_id"])
    op.create_index("ix_evacuee_identities_current_status", "evacuee_identities", ["current_status"])

    op.create_table(
        "evacuee_gallery_views",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("evacuee_id", sa.Integer(), nullable=False),
        sa.Column("view_type", sa.String(length=24), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("image_url", sa.Text(), nullable=False),
        sa.Column("feature_blob", sa.LargeBinary(), nullable=True),
        sa.Column("feature_dimension", sa.Integer(), nullable=True),
        sa.Column("feature_space_id", sa.String(length=160), nullable=True),
        sa.Column("feature_source", sa.String(length=64), nullable=True),
        sa.Column("digest", sa.String(length=64), nullable=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("captured_frame", sa.Integer(), nullable=True),
        sa.Column("camera_id", sa.String(length=80), nullable=True),
        sa.Column("sharpness", sa.Float(), nullable=True),
        sa.Column("detection_confidence", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("evacuee_id", "view_type", name="uq_evacuee_gallery_view"),
        sa.ForeignKeyConstraint(["evacuee_id"], ["evacuee_identities.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_evacuee_gallery_views_id", "evacuee_gallery_views", ["id"])
    op.create_index("ix_evacuee_gallery_views_evacuee_id", "evacuee_gallery_views", ["evacuee_id"])
    op.create_index("ix_evacuee_gallery_views_view_type", "evacuee_gallery_views", ["view_type"])
    op.create_index("ix_evacuee_gallery_views_captured_at", "evacuee_gallery_views", ["captured_at"])
    op.create_index("ix_evacuee_gallery_views_camera_id", "evacuee_gallery_views", ["camera_id"])


def downgrade() -> None:
    op.drop_table("evacuee_gallery_views")
    op.drop_table("evacuee_identities")
    op.drop_table("passenger_observations")
    op.drop_table("system_alerts")
    op.drop_table("metric_logs")
