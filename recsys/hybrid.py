"""Hybrid recommender: collaborative filtering + content-based + popularity.

    alpha = alpha_max * min(n_ratings, N) / N
    score = alpha * CF + (1 - alpha) * (c * content + (1 - c) * popularity)

The defaults (N = 5, c = 0.75, alpha_max = 1) are a starting point; the
training job (recsys/train.py) tunes alpha_max and c on held-out data.

A brand-new user (alpha = 0) gets content-based results from their stated
preferences. Each rating shifts weight toward collaborative filtering, up to
alpha_max once they have N ratings. Filters (type, state, month) are
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


def cf_from_dict(d: dict, item_ids: list[int]) -> CFModel:
    from .item_knn import ItemKNN
    from .mf import MatrixFactorization

    kinds = {ItemKNN.name: ItemKNN, MatrixFactorization.name: MatrixFactorization}
    if d["kind"] not in kinds:
        raise ValueError(f"unknown CF model kind {d['kind']!r}")
    return kinds[d["kind"]].from_dict(d, item_ids)


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
    def __init__(self, destinations: Sequence[Destination], cf: CFModel,
                 popularity: np.ndarray, global_mean: float,
                 cf_full_weight_at: int = CF_FULL_WEIGHT_AT, content_share: float = CONTENT_SHARE,
                 alpha_max: float = 1.0) -> None:
        self.cf_full_weight_at = int(cf_full_weight_at)
        self.content_share = float(content_share)
        self.alpha_max = float(alpha_max)
        self.destinations = sorted(destinations, key=lambda d: d.id)
        self.cf = cf
        self.item_index = {d.id: j for j, d in enumerate(self.destinations)}
        self.F = feature_matrix(self.destinations)
        self.popularity = np.asarray(popularity, dtype=float)
        self.global_mean = float(global_mean)
        if len(self.popularity) != len(self.destinations):
            raise ValueError("popularity must have one entry per destination")

    @classmethod
    def from_interactions(cls, destinations: Sequence[Destination], cf: CFModel,
                          inter: Interactions, **params) -> HybridRecommender:
        ids = sorted(d.id for d in destinations)
        if list(map(int, inter.item_ids)) != ids:
            raise ValueError("interaction columns must match the catalogue, sorted by id")
        return cls(destinations, cf, popularity_scores(inter), inter.global_mean(), **params)

    @classmethod
    def from_export(cls, payload: dict) -> HybridRecommender:
        """Rebuild from the dict produced by ``export()`` (a saved model artifact)."""
        dests = [
            Destination(id=d["id"], name=d["name"], state=d["state"], type=d["type"],
                        best_months=tuple(d["best_months"]), description=d.get("description", ""),
                        region=d["region"])
            for d in payload["destinations"]
        ]
        ids = [d.id for d in sorted(dests, key=lambda d: d.id)]
        return cls(dests, cf_from_dict(payload["cf"], ids), payload["popularity"], payload["global_mean"],
                   **payload["params"])

    def recommend(
        self,
        ratings: Mapping[int, float],
        preferences: Sequence[str] = (),
        k: int = 5,
        filters: Filters | None = None,
        force_alpha: float | None = None,
    ) -> HybridResult:
        """``force_alpha`` pins the CF weight; offline evaluation uses it to score each part alone."""
        filters = filters or Filters()
        ratings = {int(d): float(r) for d, r in ratings.items() if int(d) in self.item_index}
        n = len(ratings)
        alpha = self.alpha_max * min(n, self.cf_full_weight_at) / self.cf_full_weight_at
        if force_alpha is not None and n:
            alpha = float(force_alpha)

        allowed = np.array([filters.allows(d) and d.id not in ratings for d in self.destinations])
        profile = user_profile(self.F, self.item_index, preferences, ratings)
        if profile is not None:
            other = (self.content_share * content_scores(self.F, profile)
                     + (1 - self.content_share) * self.popularity)
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
        # Cite a rated place once CF carries real weight for this user (40% of its tuned maximum).
        if best is not None and alpha > 0 and alpha >= 0.4 * self.alpha_max and best[j] >= 0:
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
            "params": {"cf_full_weight_at": self.cf_full_weight_at, "content_share": self.content_share,
                       "alpha_max": self.alpha_max},
            "cf": self.cf.to_dict(),
        }
