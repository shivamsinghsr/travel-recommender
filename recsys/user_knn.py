"""User-based collaborative filtering (v1's idea, done correctly).

Differences from v1:
* users are found through ``Interactions.user_index``, never ``user_id - 1``;
* ratings are mean-centred per user, so a harsh rater and a generous rater
  with the same taste still look similar;
* similarity is shrunk by the number of co-rated places, so two users who
  share one place are not treated as twins;
* predictions use the k most similar users *who rated that place*, weighted
  by similarity, instead of a flat average over five fixed neighbours;
* places the user already rated are never recommended back;
* unknown users or users with no usable neighbours fall back to popularity.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from .data import Interactions
from .popularity import bayesian_average, popularity_scores
from .types import Recommendation


class UserKNN:
    def __init__(self, k: int = 30, shrinkage: float = 5.0, min_neighbours: int = 2) -> None:
        self.k = k
        self.shrinkage = shrinkage
        self.min_neighbours = min_neighbours

    def fit(self, inter: Interactions) -> UserKNN:
        self.inter = inter
        counts = inter.mask.sum(axis=1)
        sums = (inter.R * inter.mask).sum(axis=1)
        mu = inter.global_mean()
        self.user_mean = np.where(counts > 0, sums / np.maximum(counts, 1), mu)
        centred = np.where(inter.mask, inter.R - self.user_mean[:, None], 0.0)

        norms = np.linalg.norm(centred, axis=1)
        norms[norms == 0] = 1.0
        sim = (centred @ centred.T) / np.outer(norms, norms)
        overlap = inter.mask.astype(np.float64) @ inter.mask.T.astype(np.float64)
        sim *= overlap / (overlap + self.shrinkage)
        np.fill_diagonal(sim, 0.0)
        self.sim = sim
        self.centred = centred
        self.popularity = popularity_scores(inter)
        self.item_avg = bayesian_average(inter)
        return self

    def predict_row(self, user_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Predicted ratings for every item, plus how many neighbours backed each."""
        i = self.inter.user_index.get(int(user_id))
        if i is None:
            return None
        sims = self.sim[i]
        preds = np.full(self.inter.n_items, np.nan)
        support = np.zeros(self.inter.n_items, dtype=int)
        for j in range(self.inter.n_items):
            raters = np.flatnonzero(self.inter.mask[:, j] & (sims > 0))
            if raters.size == 0:
                continue
            top = raters[np.argsort(sims[raters])[::-1][: self.k]]
            w = sims[top]
            support[j] = top.size
            if top.size >= self.min_neighbours and w.sum() > 0:
                preds[j] = self.user_mean[i] + (w @ self.centred[top, j]) / w.sum()
        return np.clip(preds, 1.0, 5.0), support

    def recommend(
        self,
        user_id: int,
        k: int = 5,
        candidates: Collection[int] | None = None,
    ) -> list[Recommendation]:
        inter = self.inter
        allowed = np.ones(inter.n_items, dtype=bool)
        if candidates is not None:
            cand = set(int(c) for c in candidates)
            allowed = np.array([int(d) in cand for d in inter.item_ids])

        row = self.predict_row(user_id)
        if row is not None:
            i = inter.user_index[int(user_id)]
            allowed &= ~inter.mask[i]
            preds, support = row
            usable = allowed & ~np.isnan(preds)
            if usable.any():
                order = np.flatnonzero(usable)[np.argsort(-preds[usable], kind="stable")][:k]
                recs = [
                    Recommendation(
                        destination_id=int(inter.item_ids[j]),
                        score=float((preds[j] - 1.0) / 4.0),
                        predicted_rating=round(float(preds[j]), 2),
                        reason=f"Rated highly by {int(support[j])} travellers with tastes like yours",
                        source="cf",
                    )
                    for j in order
                ]
                if len(recs) == k:
                    return recs
                taken = {r.destination_id for r in recs}
                recs += self._popular(k - len(recs), allowed, exclude=taken)
                return recs
        return self._popular(k, allowed)

    def _popular(self, k: int, allowed: np.ndarray, exclude: Collection[int] = ()) -> list[Recommendation]:
        inter = self.inter
        ok = allowed & np.array([int(d) not in exclude for d in inter.item_ids])
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(-self.popularity[idx], kind="stable")][:k]
        return [
            Recommendation(
                destination_id=int(inter.item_ids[j]),
                score=float(self.popularity[j]),
                predicted_rating=round(float(self.item_avg[j]), 2),
                reason="Consistently well rated by other travellers",
                source="popular",
            )
            for j in order
        ]
