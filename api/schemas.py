"""Request and response shapes. FastAPI validates against these automatically."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class RecommendationOut(BaseModel):
    destination: DestinationOut
    score: float = Field(description="Ranking score in [0, 1]")
    predicted_rating: float | None = Field(description="Expected rating on a 1-5 scale, if known")
    reason: str
    source: str


class RecommendationsOut(BaseModel):
    user_id: int
    strategy: str
    items: list[RecommendationOut]


class RatingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    destination_id: int
    rating: int
    review_text: str | None
    visited_on: dt.date | None


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    ratings: int
    recommender: str
