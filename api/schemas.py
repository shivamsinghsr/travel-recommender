"""Request and response shapes. FastAPI validates against these automatically."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from recsys.catalog import TYPES

DestinationType = Literal[
    "Beach", "Hill Station", "Heritage", "Nature", "Adventure", "Spiritual", "Wildlife", "City"
]
assert set(DestinationType.__args__) == set(TYPES)  # keep the API enum in sync with the catalogue


class DestinationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    state: str
    type: str
    best_months: list[int]
    description: str

    @field_validator("best_months", mode="before")
    @classmethod
    def _split_months(cls, v: object) -> object:
        # stored as "10|11|12" in the database
        return [int(m) for m in v.split("|") if m] if isinstance(v, str) else v


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    preferences: list[DestinationType] = Field(default_factory=list, max_length=8)


class UserUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    preferences: list[DestinationType] | None = Field(default=None, max_length=8)


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    preferences: list[str]


class RatingIn(BaseModel):
    destination_id: int
    rating: int = Field(ge=1, le=5)
    review_text: str | None = Field(default=None, max_length=1000)
    visited_on: dt.date | None = None


class RatingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    destination_id: int
    rating: int
    review_text: str | None
    visited_on: dt.date | None


class RecommendationOut(BaseModel):
    destination: DestinationOut
    score: float = Field(description="Ranking score in [0, 1]")
    predicted_rating: float | None = Field(description="Expected rating on a 1-5 scale, if known")
    reason: str
    source: str


class RecommendationsOut(BaseModel):
    user_id: int
    strategy: Literal["popular", "content", "hybrid", "cf"]
    alpha: float = Field(description="Weight given to collaborative filtering (0-1)")
    items: list[RecommendationOut]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    ratings: int
    recommender: str
