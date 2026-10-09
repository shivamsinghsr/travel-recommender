"""Initial schema: users, destinations, ratings

Revision ID: 0001
Revises:
Create Date: 2026-10-09
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("preferences", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "destinations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("state", sa.String(80), nullable=False),
        sa.Column("type", sa.String(40), nullable=False),
        sa.Column("best_months", sa.String(40), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
    )
    op.create_index("ix_destinations_state", "destinations", ["state"])
    op.create_index("ix_destinations_type", "destinations", ["type"])

    op.create_table(
        "ratings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("destination_id", sa.Integer(), sa.ForeignKey("destinations.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("review_text", sa.Text(), nullable=True),
        sa.Column("visited_on", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", "destination_id", name="uq_rating_user_destination"),
        sa.CheckConstraint("rating BETWEEN 1 AND 5", name="ck_rating_range"),
    )
    op.create_index("ix_ratings_user_id", "ratings", ["user_id"])
    op.create_index("ix_ratings_destination_id", "ratings", ["destination_id"])


def downgrade() -> None:
    op.drop_table("ratings")
    op.drop_table("destinations")
    op.drop_table("users")
