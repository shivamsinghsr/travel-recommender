"""Load the CSVs in data/ into an empty database.

    python -m api.seed

Safe to run repeatedly: it does nothing if destinations already exist.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from recsys.catalog import DATA_DIR, load_destinations

from . import models
from .config import get_settings
from .db import Base, Database


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
    session.commit()
    return True


def main() -> None:
    db = Database(get_settings().database_url)
    Base.metadata.create_all(db.engine)
    with db.SessionLocal() as s:
        print("seeded" if seed(s) else "already seeded, nothing to do")


if __name__ == "__main__":
    main()
