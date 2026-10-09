"""Serves the active trained model and swaps in newer versions without a restart.

Every ``check_seconds`` the store asks the database which row in
``model_versions`` is active. If that version differs from the one in
memory, it loads the artifact file and swaps it in atomically; requests in
flight keep using the model they started with.

Before the first training run there is no active version. The store then
falls back to fitting an item-based model from the live ratings table, so a
fresh deployment works immediately.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from recsys.hybrid import HybridRecommender

from . import models
from .services import RecommenderService

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Served:
    model: HybridRecommender
    version: str  # "live-fit" when no trained model is active
    config: dict | None
    metrics: dict | None


class ModelStore:
    def __init__(self, models_dir: Path, check_seconds: float = 900.0, fallback_refresh_seconds: float = 30.0):
        self.models_dir = Path(models_dir)
        self.check_seconds = check_seconds
        self._lock = threading.Lock()
        self._served: Served | None = None
        self._last_check = float("-inf")
        self._fallback = RecommenderService(min_refresh_seconds=fallback_refresh_seconds)

    def load_artifact(self, row: models.ModelVersion) -> Served:
        path = (self.models_dir / row.artifact_path).resolve()
        if not path.is_relative_to(self.models_dir.resolve()):
            raise ValueError(f"artifact path escapes MODELS_DIR: {row.artifact_path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Served(HybridRecommender.from_export(payload["model"]), row.version, row.config, row.metrics)

    def current(self, session: Session, force_check: bool = False) -> Served:
        now = time.monotonic()
        if force_check or now - self._last_check >= self.check_seconds or self._served is None:
            self._last_check = now
            row = session.scalar(select(models.ModelVersion).where(models.ModelVersion.is_active.is_(True))
                                 .order_by(models.ModelVersion.id.desc()).limit(1))
            if row is not None and (self._served is None or self._served.version != row.version):
                try:
                    served = self.load_artifact(row)
                    with self._lock:
                        self._served = served
                    log.info("serving model %s", row.version)
                except (OSError, ValueError, KeyError) as exc:
                    log.error("could not load model %s: %s", row.version, exc)
            elif row is None and self._served is not None and self._served.version != "live-fit":
                with self._lock:
                    self._served = None  # active model was withdrawn; fall back below

        served = self._served
        if served is not None and served.version != "live-fit":
            return served
        self._fallback.ensure_fresh(session)
        fallback = Served(self._fallback.model, "live-fit", None, None)
        with self._lock:
            self._served = fallback
        return fallback
