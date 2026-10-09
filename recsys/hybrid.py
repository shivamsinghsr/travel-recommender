"""Hybrid recommender: collaborative filtering + content-based + popularity.

    alpha = min(n_ratings, 5) / 5
    score = alpha * CF + (1 - alpha) * (0.75 * content + 0.25 * popularity)

A brand-new user (alpha = 0) gets content-based results from their stated
preferences. Each rating shifts weight toward collaborative filtering, and
from five ratings on CF decides alone. Filters (type, state, month) are
applied *before* scoring, and places the user has rated are never returned.

The same algorithm runs in the browser (web/js/recommender.js); a parity
test keeps the two implementations in agreement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .catalog import Destination
from .content import content_scores, feature_matrix, user_profile
from .data import Interactions
from .popularity import popularity_scores
from .types import Recommendation

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")

CF_FULL_WEIGHT_AT = 5
CONTENT_SHARE = 0.75


class CFModel(Protocol):
    name: str

    def predict(self, ratings: Mapping[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class Filters:
    type: str | None = None
    state: str | None = None
    month: int | None = None

    def allows(self, d: Destination) -> bool:
        return ((self.type is None or d.type == self.type)
                and (self.state is None or d.state == self.state)
                and (self.month is None or d.in_season(self.month)))


@dataclass
class HybridResult:
    strategy: str  # "popular" | "content" | "hybrid" | "cf"
    alpha: float
    items: list[Recommendation]


class HybridRecommender:
    def __init__(self, destinations: Sequence[Destination], cf: CFModel, inter: Interactions) -> None:
        self.destinations = sorted(destinations, key=lambda d: d.id)
        ids = [d.id for d in self.destinations]
        if list(map(int, inter.item_ids)) != ids:
            raise ValueError("interaction columns must match the catalogue, sorted by id")
        self.cf = cf
        self.item_index = {d: j for j, d in enumerate(ids)}
        self.F = feature_matrix(self.destinations)
        self.popularity = popularity_scores(inter)
        self.global_mean = inter.global_mean()

    def recommend(
        self,
        ratings: Mapping[int, float],
        preferences: Sequence[str] = (),
        k: int = 5,
        filters: Filters | None = None,
    ) -> HybridResult:
        filters = filters or Filters()
        ratings = {int(d): float(r) for d, r in ratings.items() if int(d) in self.item_index}
        n = len(ratings)
        alpha = min(n, CF_FULL_WEIGHT_AT) / CF_FULL_WEIGHT_AT

        allowed = np.array([filters.allows(d) and d.id not in ratings for d in self.destinations])
        profile = user_profile(self.F, self.item_index, preferences, ratings)
        if profile is not None:
            other = CONTENT_SHARE * content_scores(self.F, profile) + (1 - CONTENT_SHARE) * self.popularity
        else:
            other = self.popularity

        pred = support = best = None
        if n:
            pred, support, best = self.cf.predict(ratings)
            score = alpha * (pred - 1.0) / 4.0 + (1 - alpha) * other
        else:
            score = other

        if n == 0:
            strategy = "content" if profile is not None else "popular"
        else:
            strategy = "cf" if alpha >= 1.0 else "hybrid"

        idx = np.flatnonzero(allowed)
        order = idx[np.argsort(-score[idx], kind="stable")][:k]
        prefs = set(preferences)
        items = []
        for j in order:
            d = self.destinations[j]
            reason = self._reason(d, j, ratings, prefs, filters, alpha, best)
            items.append(Recommendation(
                destination_id=d.id,
                score=float(np.clip(score[j], 0.0, 1.0)),
                predicted_rating=round(float(pred[j]), 2) if pred is not None else None,
                reason=reason,
                source=strategy,
            ))
        return HybridResult(strategy=strategy, alpha=alpha, items=items)

    def _reason(self, d, j, ratings, prefs, filters, alpha, best) -> str:
        if best is not None and alpha >= 0.4 and best[j] >= 0:
            src = self.destinations[int(best[j])]
            if ratings[src.id] >= 4:
                return f"Because you rated {src.name} {ratings[src.id]:g}/5"
        if d.type in prefs:
            return f"Matches your interest in {d.type} destinations"
        if filters.month is not None:
            return f"In season in {MONTHS[filters.month - 1]}"
        return "Consistently well rated by other travellers"

    def export(self) -> dict:
        """Everything the browser needs to run the same algorithm."""
        return {
            "destinations": [
                {"id": d.id, "name": d.name, "state": d.state, "type": d.type, "region": d.region,
                 "best_months": list(d.best_months), "description": d.description}
                for d in self.destinations
            ],
            # full precision, so the browser computes exactly what Python computes
            "features": self.F.tolist(),
            "popularity": self.popularity.tolist(),
            "global_mean": self.global_mean,
            "params": {"cf_full_weight_at": CF_FULL_WEIGHT_AT, "content_share": CONTENT_SHARE},
            "cf": self.cf.to_dict(),
        }
