"""Package the recommender for the static website.

    python -m recsys.export_web --out web/model

Writes ``model.json``: the destination catalogue, the content features, the
collaborative-filtering parameters, popularity, a few sample travellers to
try, and metadata. ``web/js/recommender.js`` runs the same hybrid algorithm
on it, entirely in the visitor's browser.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .catalog import DATA_DIR, TYPES, load_destinations
from .data import Interactions, load_ratings
from .hybrid import HybridRecommender
from .item_knn import ItemKNN

# Sample travellers to try in the demo: one per distinct taste profile.
PERSONA_TASTES = (("Adventure",), ("Beach", "Heritage"), ("Wildlife",), ("Spiritual",))


def pick_personas(users: list[dict], inter: Interactions, n_ratings=(8, 14)) -> list[dict]:
    personas, used = [], set()
    for taste in PERSONA_TASTES:
        for u in users:
            prefs = tuple(p for p in u["preferences"].split("|") if p)
            ratings = inter.user_ratings(int(u["id"]))
            if prefs == taste and n_ratings[0] <= len(ratings) <= n_ratings[1] and u["id"] not in used:
                used.add(u["id"])
                personas.append({
                    "id": int(u["id"]),
                    "name": u["name"],
                    "preferences": list(prefs),
                    "ratings": [{"destination_id": d, "rating": r} for d, r in ratings.items()],
                })
                break
    return personas


def build_hybrid(data_dir: Path = DATA_DIR) -> tuple[HybridRecommender, Interactions]:
    dests = load_destinations(data_dir / "destinations.csv")
    inter = Interactions.from_ratings(load_ratings(data_dir / "ratings.csv"), item_ids=[d.id for d in dests])
    return HybridRecommender(dests, ItemKNN().fit(inter), inter), inter


def export(out_dir: Path, data_dir: Path = DATA_DIR, extra_meta: dict | None = None,
           hybrid: HybridRecommender | None = None) -> Path:
    import csv

    if hybrid is None:
        hybrid, inter = build_hybrid(data_dir)
    else:
        inter = Interactions.from_ratings(load_ratings(data_dir / "ratings.csv"),
                                          item_ids=[d.id for d in hybrid.destinations])
    with open(data_dir / "users.csv", newline="", encoding="utf-8") as fh:
        users = list(csv.DictReader(fh))

    payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "cf_model": hybrid.cf.name,
            "n_users": inter.n_users,
            "n_ratings": int(inter.mask.sum()),
            "n_destinations": inter.n_items,
            "types": list(TYPES),
            **(extra_meta or {}),
        },
        "model": hybrid.export(),
        "personas": pick_personas(users, inter),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the model for the static website")
    ap.add_argument("--out", type=Path, default=Path("web/model"))
    args = ap.parse_args()
    path = export(args.out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
