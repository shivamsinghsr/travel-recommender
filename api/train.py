"""Nightly training job against the production database.

    python -m api.train

Reads every rating from PostgreSQL, runs recsys.train (tune on validation,
report on test, gate against the active model), and if the candidate is at
least as good: writes the artifact to MODELS_DIR/<version>/model.json,
records it in ``model_versions`` and marks it active in one transaction.
Running API instances pick it up within MODEL_CHECK_SECONDS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from recsys.data import Rating
from recsys.train import Config, TrainingData, build_payload, run, summary_markdown, write_artifact

from . import models
from .config import get_settings
from .db import Database
from .services import to_catalog


def load_training_data(session: Session) -> TrainingData:
    dests = to_catalog(list(session.scalars(select(models.Destination).order_by(models.Destination.id))))
    rows = session.execute(select(
        models.Rating.user_id, models.Rating.destination_id, models.Rating.rating,
        models.Rating.visited_on, models.Rating.created_at)).all()
    ratings = [
        Rating(u, d, float(r), (visited or (created.date() if created else dt.date.today())).isoformat())
        for u, d, r, visited, created in rows
    ]
    prefs = {u.id: u.preference_list for u in session.scalars(select(models.User))}
    return TrainingData(dests, ratings, prefs)


def train(session: Session, models_dir, tolerance: float = 0.005, grids: dict | None = None) -> dict:
    active = session.scalar(select(models.ModelVersion).where(models.ModelVersion.is_active.is_(True)))
    baseline = Config.from_dict(active.config) if active else None
    data = load_training_data(session)
    if len(data.ratings) < 50:
        return {"published": False, "decision": "fewer than 50 ratings: not training"}

    result = run(data, baseline=baseline, tolerance=tolerance, grids=grids)
    print(summary_markdown(result))
    if not result.publish:
        return {"published": False, "decision": result.decision, "version": result.version}

    rel_path = f"{result.version}/model.json"
    write_artifact(build_payload(result), models_dir / rel_path)
    session.execute(update(models.ModelVersion).values(is_active=False))
    session.add(models.ModelVersion(
        version=result.version, artifact_path=rel_path, config=result.config.as_dict(),
        metrics=result.metrics_payload(), n_ratings=len(data.ratings),
        data_fingerprint=data.fingerprint(), is_active=True,
    ))
    session.commit()
    return {"published": True, "decision": result.decision, "version": result.version}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--if-missing", action="store_true", help="only train when no model is active yet")
    args = ap.parse_args()
    settings = get_settings()
    db = Database(settings.database_url)
    with db.SessionLocal() as s:
        if args.if_missing and s.scalar(select(models.ModelVersion.id).where(models.ModelVersion.is_active.is_(True))):
            print("a trained model is already active; skipping")
            return
        outcome = train(s, settings.models_dir, tolerance=args.tolerance)
    print(outcome["decision"])
    sys.exit(0)


if __name__ == "__main__":
    main()
