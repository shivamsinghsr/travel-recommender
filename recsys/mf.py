"""Biased matrix factorisation trained with alternating least squares (ALS).

Model:   r_ui ≈ mu + b_u + b_i + p_u · q_i

Each destination i gets a bias b_i and a small vector q_i of latent factors
(hidden dimensions such as "remote and rugged" vs "comfortable and urban"
that the model discovers on its own). Each user gets b_u and p_u.

Training alternates between solving every user's (p_u, b_u) with the items
fixed and every item's (q_i, b_i) with the users fixed; each step is a small
ridge regression with a closed-form solution.

Serving uses *fold-in*: given anyone's current ratings, solve for their
(p_u, b_u) on the spot with the trained item side held fixed. New users and
brand-new ratings are therefore scored immediately, without retraining.
Unlike neighbourhood methods, this still works when two users share no
rated places, because similarity flows through the latent factors.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .data import Interactions


def _ridge(A: np.ndarray, y: np.ndarray, reg: np.ndarray) -> np.ndarray:
    """Solve min ||A x - y||² + Σ reg_k x_k²."""
    return np.linalg.solve(A.T @ A + np.diag(reg), A.T @ y)


class MatrixFactorization:
    name = "mf"

    def __init__(self, factors: int = 16, reg: float = 5.0, bias_reg: float = 2.0,
                 iterations: int = 15, seed: int = 0) -> None:
        self.factors = factors
        self.reg = reg
        self.bias_reg = bias_reg
        self.iterations = iterations
        self.seed = seed

    @property
    def _reg_vec(self) -> np.ndarray:
        return np.r_[np.full(self.factors, self.reg), self.bias_reg]

    def fit(self, inter: Interactions) -> MatrixFactorization:
        rng = np.random.default_rng(self.seed)
        n_u, n_i, f = inter.n_users, inter.n_items, self.factors
        self.mu = inter.global_mean()
        P = rng.normal(0, 0.1, (n_u, f))
        Q = rng.normal(0, 0.1, (n_i, f))
        bu = np.zeros(n_u)
        bi = np.zeros(n_i)
        rows = [np.flatnonzero(inter.mask[u]) for u in range(n_u)]
        cols = [np.flatnonzero(inter.mask[:, i]) for i in range(n_i)]
        reg = self._reg_vec

        for _ in range(self.iterations):
            for u, items in enumerate(rows):
                if items.size == 0:
                    continue
                A = np.c_[Q[items], np.ones(items.size)]
                x = _ridge(A, inter.R[u, items] - self.mu - bi[items], reg)
                P[u], bu[u] = x[:f], x[f]
            for i, users in enumerate(cols):
                if users.size == 0:
                    continue
                A = np.c_[P[users], np.ones(users.size)]
                x = _ridge(A, inter.R[users, i] - self.mu - bu[users], reg)
                Q[i], bi[i] = x[:f], x[f]

        self.P, self.Q, self.user_bias, self.item_bias = P, Q, bu, bi
        self.item_index = dict(inter.item_index)
        self.user_index = dict(inter.user_index)
        return self

    def train_rmse(self, inter: Interactions) -> float:
        pred = self.mu + self.user_bias[:, None] + self.item_bias[None, :] + self.P @ self.Q.T
        err = (pred - inter.R)[inter.mask]
        return float(np.sqrt(np.mean(err**2)))

    def fold_in(self, ratings: Mapping[int, float]) -> tuple[np.ndarray, float] | None:
        rated = [(self.item_index[d], float(r)) for d, r in ratings.items() if d in self.item_index]
        if not rated:
            return None
        cols = np.array([j for j, _ in rated])
        y = np.array([r for _, r in rated]) - self.mu - self.item_bias[cols]
        A = np.c_[self.Q[cols], np.ones(len(cols))]
        x = _ridge(A, y, self._reg_vec)
        return x[: self.factors], float(x[self.factors])

    def predict(self, ratings: Mapping[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(predicted rating per item, support per item, best-explaining rated item per item)."""
        n_items = len(self.item_bias)
        base = self.mu + self.item_bias
        folded = self.fold_in(ratings)
        if folded is None:
            return np.clip(base, 1.0, 5.0), np.zeros(n_items), np.full(n_items, -1)
        p, b = folded
        pred = np.clip(base + b + self.Q @ p, 1.0, 5.0)

        # Explanation: the rated place most similar (in latent space) to each target that the
        # user liked more than expected.
        rated = [(self.item_index[d], float(r)) for d, r in ratings.items() if d in self.item_index]
        cols = np.array([j for j, _ in rated])
        resid = np.array([r for _, r in rated]) - (base[cols] + b)
        Qn = self.Q / np.maximum(np.linalg.norm(self.Q, axis=1, keepdims=True), 1e-12)
        sim = Qn @ Qn[cols].T  # n_items x n_rated
        contrib = sim * resid
        best_local = np.argmax(contrib, axis=1)
        best = np.where(contrib[np.arange(n_items), best_local] > 0, cols[best_local], -1)
        return pred, np.full(n_items, float(len(cols))), best

    def to_dict(self) -> dict:
        return {
            "kind": self.name,
            "factors": self.factors, "reg": self.reg, "bias_reg": self.bias_reg,
            "iterations": self.iterations,
            "mu": self.mu,
            "item_bias": self.item_bias.tolist(),
            "Q": self.Q.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict, item_ids: list[int]) -> MatrixFactorization:
        m = cls(factors=d["factors"], reg=d["reg"], bias_reg=d["bias_reg"], iterations=d["iterations"])
        m.mu = float(d["mu"])
        m.item_bias = np.asarray(d["item_bias"], dtype=float)
        m.Q = np.asarray(d["Q"], dtype=float)
        m.item_index = {int(i): j for j, i in enumerate(item_ids)}
        return m
