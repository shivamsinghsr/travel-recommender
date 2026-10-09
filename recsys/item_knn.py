"""Item-based collaborative filtering.

"People who rated X highly also rated Y highly." Item-item similarities are
computed once from all ratings; a prediction for any user then needs only
that user's own ratings. That makes it work for brand-new users after their
first rating, and lets the browser run it from a small exported matrix.

Prediction for place j, given the user's ratings {i: r_i}:

    baseline_j  = Bayesian-average rating of j
    dev_i       = r_i - baseline_i                      (how much the user liked i vs. everyone)
    offset      = sum(dev_i) / (n + 2)                  (generous/harsh rater, shrunk)
    pred_j      = baseline_j + offset
                  + sum_i s_ij (dev_i - offset) / (sum_i |s_ij| + 1)

using the ``k`` rated items most similar to j. The ``+ 1`` in the
denominator keeps a single weak neighbour from swinging the prediction.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .data import Interactions
from .popularity import bayesian_average


class ItemKNN:
    name = "item-knn"

    def __init__(self, k: int = 20, shrinkage: float = 10.0) -> None:
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, inter: Interactions) -> ItemKNN:
        counts = inter.mask.sum(axis=1)
        user_mean = (inter.R * inter.mask).sum(axis=1) / np.maximum(counts, 1)
        centred = np.where(inter.mask, inter.R - user_mean[:, None], 0.0)  # adjusted cosine
        norms = np.linalg.norm(centred, axis=0)
        norms[norms == 0] = 1.0
        sim = (centred.T @ centred) / np.outer(norms, norms)
        co = inter.mask.T.astype(np.float64) @ inter.mask.astype(np.float64)
        sim *= co / (co + self.shrinkage)
        np.fill_diagonal(sim, 0.0)
        self.sim = sim
        self.baseline = bayesian_average(inter)
        self.item_index = dict(inter.item_index)
        return self

    def predict(self, ratings: Mapping[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (predicted rating per item, support per item, best-supporting rated item per item).

        ``best`` holds a column index into the item axis, or -1 when nothing supports the prediction.
        """
        n_items = len(self.baseline)
        rated = [(self.item_index[d], float(r)) for d, r in ratings.items() if d in self.item_index]
        if not rated:
            return self.baseline.copy(), np.zeros(n_items), np.full(n_items, -1)
        cols = np.array([j for j, _ in rated])
        dev = np.array([r for _, r in rated]) - self.baseline[cols]
        offset = dev.sum() / (len(dev) + 2.0)
        resid = dev - offset

        S = self.sim[:, cols]  # n_items x n_rated
        if len(cols) > self.k:  # keep only the k most similar rated items per target
            keep = np.argsort(-np.abs(S), axis=1, kind="stable")[:, : self.k]
            mask = np.zeros_like(S, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            S = np.where(mask, S, 0.0)
        support = np.abs(S).sum(axis=1)
        pred = self.baseline + offset + (S @ resid) / (support + 1.0)
        contrib = S * resid
        best_local = np.argmax(contrib, axis=1)
        best = np.where(contrib[np.arange(n_items), best_local] > 0, cols[best_local], -1)
        return np.clip(pred, 1.0, 5.0), support, best

    @classmethod
    def from_dict(cls, d: dict, item_ids: list[int]) -> ItemKNN:
        m = cls(k=d["k"])
        m.sim = np.asarray(d["sim"], dtype=float)
        m.baseline = np.asarray(d["baseline"], dtype=float)
        m.item_index = {int(i): j for j, i in enumerate(item_ids)}
        return m

    def to_dict(self) -> dict:
        return {
            "kind": self.name,
            "k": self.k,
            "baseline": self.baseline.tolist(),
            "sim": self.sim.tolist(),
        }
