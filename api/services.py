"""Keeps the fitted recommender in memory and refits it when ratings change."""

from __future__ import annotations

import threading
import time

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from recsys.catalog import STATE_REGION, Destination
from recsys.data import Interactions, Rating
from recsys.hybrid import Filters, HybridRecommender, HybridResult
from recsys.item_knn import ItemKNN

from . import models


def to_catalog(rows: list[models.Destination]) -> list[Destination]:
    return [
        Destination(
            id=d.id, name=d.name, state=d.state, type=d.type,
            best_months=tuple(d.month_list), description=d.description,
            region=STATE_REGION.get(d.state, "Central"),
        )
        for d in rows
    ]


class RecommenderService:
    """Refits item-item similarities from the database.

    A user's *own* ratings are always read fresh for each request, so a new
    rating changes that user's results immediately. The shared similarity
    matrix is refit when the ratings table changes, at most once every
    ``min_refresh_seconds``.
    """

    name = "hybrid(item-knn)"

    def __init__(self, min_refresh_seconds: float = 30.0) -> None:
        self._lock = threading.Lock()
        self.min_refresh_seconds = min_refresh_seconds
        self.model: HybridRecommender | None = None
        self._fingerprint: tuple[int, int] | None = None
        self._last_refresh = 0.0

    @staticmethod
    def fingerprint(session: Session) -> tuple[int, int]:
        n, total = session.execute(
            select(func.count(models.Rating.id), func.coalesce(func.sum(models.Rating.rating), 0))
        ).one()
        return int(n), int(total)

    def refresh(self, session: Session) -> None:
        dests = to_catalog(list(session.scalars(select(models.Destination).order_by(models.Destination.id))))
        rows = session.execute(
            select(models.Rating.user_id, models.Rating.destination_id, models.Rating.rating)
        ).all()
        inter = Interactions.from_ratings((Rating(u, d, float(r)) for u, d, r in rows), item_ids=[d.id for d in dests])
        model = HybridRecommender(dests, ItemKNN().fit(inter), inter)
        with self._lock:
            self.model = model
            self._fingerprint = self.fingerprint(session)
            self._last_refresh = time.monotonic()

    def ensure_fresh(self, session: Session, force: bool = False) -> None:
        if self.model is None:
            self.refresh(session)
            return
        if not force and time.monotonic() - self._last_refresh < self.min_refresh_seconds:
            return
        if force or self.fingerprint(session) != self._fingerprint:
            self.refresh(session)

    def recommend(
        self, ratings: dict[int, float], preferences: list[str], k: int, filters: Filters
    ) -> HybridResult:
        assert self.model is not None, "call refresh() first"
        return self.model.recommend(ratings, preferences, k=k, filters=filters)
