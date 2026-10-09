"""Result type shared by every recommender."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    destination_id: int
    score: float  # ranking score in [0, 1]; higher is better
    predicted_rating: float | None  # on the 1-5 scale, when the model can predict one
    reason: str  # one human-readable sentence
    source: str  # which component produced it: "cf", "content", "hybrid", "popular"
