"""Content-based scoring from destination attributes.

Each destination becomes a vector: its type (strongest signal), its region
and the months it is in season. A user profile lives in the same space and
is built from stated preferences plus, once they exist, their ratings
(places rated above 3 pull the profile toward them, below 3 push it away).
Score = cosine similarity between profile and destination.

This works with zero ratings, which is exactly when collaborative filtering
cannot help (the cold-start problem).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .catalog import REGIONS, TYPES, Destination

TYPE_WEIGHT = 1.0
REGION_WEIGHT = 0.35
MONTH_WEIGHT = 0.35


def feature_matrix(destinations: Sequence[Destination]) -> np.ndarray:
    """One unit-length row per destination: [types | regions | months]."""
    n = len(destinations)
    F = np.zeros((n, len(TYPES) + len(REGIONS) + 12))
    for j, d in enumerate(destinations):
        F[j, TYPES.index(d.type)] = TYPE_WEIGHT
        F[j, len(TYPES) + REGIONS.index(d.region)] = REGION_WEIGHT
        months = np.zeros(12)
        months[[m - 1 for m in d.best_months]] = 1.0
        F[j, len(TYPES) + len(REGIONS):] = MONTH_WEIGHT * months / np.sqrt(months.sum())
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def user_profile(
    F: np.ndarray,
    item_index: Mapping[int, int],
    preferences: Sequence[str],
    ratings: Mapping[int, float],
) -> np.ndarray | None:
    """Profile vector in feature space, or None if nothing is known about the user."""
    profile = np.zeros(F.shape[1])
    prefs = [p for p in preferences if p in TYPES]
    if prefs:
        stated = np.zeros(F.shape[1])
        for p in prefs:
            stated[TYPES.index(p)] = 1.0
        profile += stated / np.linalg.norm(stated)

    rated = [(item_index[d], r) for d, r in ratings.items() if d in item_index]
    if rated:
        learned = sum((r - 3.0) * F[j] for j, r in rated)
        norm = np.linalg.norm(learned)
        if norm > 0:
            n = len(rated)
            profile += (n / (n + 3.0)) * learned / norm  # trust ratings more as they accumulate

    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else None


def content_scores(F: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Cosine similarity mapped from [-1, 1] to [0, 1]."""
    return (F @ profile + 1.0) / 2.0
