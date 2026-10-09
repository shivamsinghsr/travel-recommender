"""ORM models. The database enforces the rules v1's CSV files could not."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    preferences: Mapped[str] = mapped_column(String(200), default="")  # "Beach|Heritage"
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    ratings: Mapped[list[Rating]] = relationship(back_populates="user", cascade="all, delete-orphan")

    @property
    def preference_list(self) -> list[str]:
        return [p for p in self.preferences.split("|") if p]


class Destination(Base):
    __tablename__ = "destinations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    state: Mapped[str] = mapped_column(String(80), index=True)
    type: Mapped[str] = mapped_column(String(40), index=True)
    best_months: Mapped[str] = mapped_column(String(40))  # "10|11|12|1|2|3"
    description: Mapped[str] = mapped_column(Text, default="")

    @property
    def month_list(self) -> list[int]:
        return [int(m) for m in self.best_months.split("|") if m]


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "destination_id", name="uq_rating_user_destination"),
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_rating_range"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    destination_id: Mapped[int] = mapped_column(ForeignKey("destinations.id", ondelete="CASCADE"), index=True)
    rating: Mapped[int] = mapped_column(SmallInteger)
    review_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    visited_on: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="ratings")
    destination: Mapped[Destination] = relationship()


class ModelVersion(Base):
    """One row per trained model. Exactly one row is active; the API serves that one."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(40), unique=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    artifact_path: Mapped[str] = mapped_column(String(500))  # relative to MODELS_DIR
    config: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    n_ratings: Mapped[int] = mapped_column(Integer)
    data_fingerprint: Mapped[str] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
