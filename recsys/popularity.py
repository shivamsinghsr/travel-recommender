"""Bayesian-average popularity: the fallback when nothing personal is known."""

from __future__ import annotations

import numpy as np

from .data import Interactions


def bayesian_average(inter: Interactions, prior_weight: float = 10.0) -> np.ndarray:
    """Per-item mean rating shrunk toward the global mean.

    An item with two 5-star ratings should not outrank one with two hundred
    4.6-star ratings; adding ``prior_weight`` pseudo-ratings at the global
    mean handles that.
    """
    counts = inter.mask.sum(axis=0)
    sums = (inter.R * inter.mask).sum(axis=0)
    mu = inter.global_mean()
    return (sums + prior_weight * mu) / (counts + prior_weight)


def popularity_scores(inter: Interactions, prior_weight: float = 10.0) -> np.ndarray:
    """Bayesian average rescaled to [0, 1] across the catalogue."""
    avg = bayesian_average(inter, prior_weight)
    lo, hi = float(avg.min()), float(avg.max())
    return (avg - lo) / (hi - lo) if hi > lo else np.full_like(avg, 0.5)
