"""Ratings as a user x item matrix, with explicit ID <-> row mappings.

v1 indexed its similarity matrix with ``user_id - 1``, which silently picked
the wrong row whenever IDs had gaps and crashed for IDs past the last row.
Every lookup here goes through ``user_index`` / ``item_index`` instead.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .catalog import DATA_DIR


@dataclass(frozen=True)
class Rating:
    user_id: int
    destination_id: int
    rating: float
    visited_on: str = ""  # ISO date, used for time-based evaluation splits


@dataclass
class Interactions:
    user_ids: np.ndarray  # row -> user id
    item_ids: np.ndarray  # column -> destination id
    R: np.ndarray  # ratings, 0 where unknown
    mask: np.ndarray  # True where a rating exists

    def __post_init__(self) -> None:
        self.user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self.item_index = {int(d): j for j, d in enumerate(self.item_ids)}

    @classmethod
    def from_ratings(
        cls, ratings: Iterable[Rating], item_ids: Sequence[int] | None = None
    ) -> Interactions:
        ratings = list(ratings)
        users = sorted({r.user_id for r in ratings})
        items = sorted(set(item_ids) if item_ids is not None else {r.destination_id for r in ratings})
        u_idx = {u: i for i, u in enumerate(users)}
        i_idx = {d: j for j, d in enumerate(items)}
        R = np.zeros((len(users), len(items)), dtype=np.float64)
        mask = np.zeros_like(R, dtype=bool)
        for r in ratings:
            if r.destination_id not in i_idx:
                continue  # rating for a destination outside the catalogue
            i, j = u_idx[r.user_id], i_idx[r.destination_id]
            R[i, j] = r.rating
            mask[i, j] = True
        return cls(np.array(users, dtype=np.int64), np.array(items, dtype=np.int64), R, mask)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def global_mean(self) -> float:
        return float(self.R[self.mask].mean()) if self.mask.any() else 3.0

    def user_ratings(self, user_id: int) -> dict[int, float]:
        """destination_id -> rating for one user (empty if the user is unknown)."""
        i = self.user_index.get(int(user_id))
        if i is None:
            return {}
        cols = np.flatnonzero(self.mask[i])
        return {int(self.item_ids[j]): float(self.R[i, j]) for j in cols}


def load_ratings(path: Path | str | None = None) -> list[Rating]:
    path = Path(path) if path else DATA_DIR / "ratings.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            Rating(int(r["user_id"]), int(r["destination_id"]), float(r["rating"]), r.get("visited_on", ""))
            for r in csv.DictReader(fh)
        ]
