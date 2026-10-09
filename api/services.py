"""Keeps a fitted recommender in memory and rebuilds it from the database."""

from __future__ import annotations

import threading

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from recsys.data import Interactions, Rating
from recsys.types import Recommendation
from recsys.user_knn import UserKNN

from . import models


class RecommenderService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model: UserKNN | None = None
        self.n_ratings = 0

    def refresh(self, session: Session) -> None:
        """Re-read every rating from the database and refit."""
        item_ids = session.scalars(select(models.Destination.id)).all()
        rows = session.execute(
            select(models.Rating.user_id, models.Rating.destination_id, models.Rating.rating)
        ).all()
        inter = Interactions.from_ratings((Rating(u, d, float(r)) for u, d, r in rows), item_ids=item_ids)
        model = UserKNN().fit(inter)
        with self._lock:
            self.model, self.n_ratings = model, len(rows)

    def ensure_fresh(self, session: Session) -> None:
        count = session.scalar(select(func.count(models.Rating.id))) or 0
        if self.model is None or count != self.n_ratings:
            self.refresh(session)

    def recommend(self, user_id: int, k: int) -> list[Recommendation]:
        assert self.model is not None, "call refresh() first"
        return self.model.recommend(user_id, k=k)
