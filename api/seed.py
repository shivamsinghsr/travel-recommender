"""Migrate the database to the latest schema and load data/*.csv into it.

    python -m api.seed

Safe to run on every deploy: migrations are idempotent and the data load
does nothing once destinations exist.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from recsys.catalog import DATA_DIR, load_destinations

from . import models
from .config import ROOT, get_settings
from .db import Database


def migrate(database_url: str) -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["database_url"] = database_url
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")


def seed(session: Session, data_dir: Path = DATA_DIR) -> bool:
    if session.scalar(select(models.Destination.id).limit(1)) is not None:
        return False

    for d in load_destinations(data_dir / "destinations.csv"):
        session.add(models.Destination(
            id=d.id, name=d.name, state=d.state, type=d.type,
            best_months="|".join(map(str, d.best_months)), description=d.description,
        ))
    with open(data_dir / "users.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            session.add(models.User(
                id=int(row["id"]), name=row["name"], email=row["email"], preferences=row["preferences"],
            ))
    session.flush()
    with open(data_dir / "ratings.csv", newline="", encoding="utf-8") as fh:
        session.add_all(
            models.Rating(
                user_id=int(row["user_id"]),
                destination_id=int(row["destination_id"]),
                rating=int(row["rating"]),
                review_text=row["review_text"] or None,
                visited_on=dt.date.fromisoformat(row["visited_on"]) if row["visited_on"] else None,
            )
            for row in csv.DictReader(fh)
        )
    session.flush()
    if session.bind.dialect.name == "postgresql":
        # Explicit ids were inserted above; move the sequences past them so new rows don't collide.
        for table in ("users", "destinations"):
            session.execute(text(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), (SELECT MAX(id) FROM {table}))"
            ))
    session.commit()
    return True


def main() -> None:
    url = get_settings().database_url
    migrate(url)
    db = Database(url)
    with db.SessionLocal() as s:
        print("seeded" if seed(s) else "already seeded, nothing to do")


if __name__ == "__main__":
    main()
