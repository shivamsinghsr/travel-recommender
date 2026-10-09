"""Add password hashes and the model_versions table

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-09
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("password_hash", sa.String(255), nullable=True))

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version", sa.String(40), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("artifact_path", sa.String(500), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("n_ratings", sa.Integer(), nullable=False),
        sa.Column("data_fingerprint", sa.String(32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_model_versions_is_active", "model_versions", ["is_active"])


def downgrade() -> None:
    op.drop_table("model_versions")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("password_hash")
