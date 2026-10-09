"""Deterministic simulator for users and ratings.

The destinations are real; the people and their ratings are not. Each
simulated traveller has hidden tastes (type and region affinities, a few
latent factors, a harsh-or-generous rating bias). Their stated preferences
are a noisy view of those tastes, they mostly visit places they are drawn
to, and they rate each visit from the same hidden tastes plus noise.

That gives the data the structure real ratings have, so collaborative
filtering and content-based methods have something genuine to learn, and
offline evaluation numbers mean something.

Run:  python -m recsys.simulate            (writes data/users.csv, data/ratings.csv)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np

from .catalog import DATA_DIR, REGIONS, TYPES, Destination, load_destinations

FIRST_NAMES = [
    "Aarav", "Aditi", "Aditya", "Ananya", "Arjun", "Diya", "Ishaan", "Kavya",
    "Krishna", "Meera", "Neha", "Nikhil", "Priya", "Rahul", "Riya", "Rohan",
    "Saanvi", "Sahil", "Sneha", "Tanvi", "Varun", "Vihaan", "Zoya", "Kabir",
    "Ira", "Dev", "Pooja", "Siddharth", "Anika", "Yash", "Nandini", "Harsh",
    "Fatima", "Imran", "Gurpreet", "Simran", "Joseph", "Maria", "Lakshmi", "Suresh",
]
LAST_INITIALS = "ABCDEGHIJKMNPRSTVY"

REVIEWS = {
    5: ["Unforgettable, would go again tomorrow.", "Easily one of my best trips.",
        "Worth every rupee and every hour of travel."],
    4: ["Really enjoyed it, a few crowded spots.", "Great trip, plan more days than we did.",
        "Lovely place, food was a highlight."],
    3: ["Decent, but not what I expected.", "Fine for a short stop.",
        "Some parts were great, others forgettable."],
    2: ["Too crowded and overpriced for us.", "Weather spoiled most of the visit.",
        "Would not prioritise this again."],
    1: ["Disappointing from start to finish.", "Not for us at all.",
        "Skip it unless you really have to go."],
}

START = dt.date(2023, 1, 1)
END = dt.date(2026, 6, 30)


def _visit_date(rng: np.random.Generator, dest: Destination) -> dt.date:
    """A date between START and END, usually inside the destination's season."""
    year = int(rng.integers(START.year, END.year + 1))
    months = dest.best_months if rng.random() < 0.8 else tuple(range(1, 13))
    month = int(rng.choice(months))
    day = int(rng.integers(1, 29))
    d = dt.date(year, month, day)
    if d > END:
        d = dt.date(END.year - 1, month, day)
    return d


def simulate(
    destinations: list[Destination],
    n_users: int = 800,
    seed: int = 42,
    latent_dim: int = 4,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    n_items = len(destinations)
    type_idx = np.array([TYPES.index(d.type) for d in destinations])
    region_idx = np.array([REGIONS.index(d.region) for d in destinations])

    # Hidden item properties
    item_quality = rng.normal(0.0, 0.35, n_items)
    item_latent = rng.normal(0.0, 0.55, (n_items, latent_dim))
    fame = rng.gamma(2.0, 0.5, n_items)  # some places are simply visited more

    users: list[dict] = []
    ratings: list[dict] = []
    for uid in range(1, n_users + 1):
        # Hidden user tastes
        type_aff = rng.normal(0.0, 0.35, len(TYPES))
        n_loved = int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
        loved = rng.choice(len(TYPES), n_loved, replace=False)
        type_aff[loved] += rng.uniform(0.9, 1.5, n_loved)
        region_aff = rng.normal(0.0, 0.3, len(REGIONS))
        user_latent = rng.normal(0.0, 0.55, latent_dim)
        bias = rng.normal(0.0, 0.35)

        # Stated preferences: mostly the truly loved types, sometimes one is missed
        stated = [TYPES[i] for i in loved if rng.random() < 0.85]
        if not stated:
            stated = [TYPES[int(np.argmax(type_aff))]]

        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(list(LAST_INITIALS))}."
        users.append({
            "id": uid,
            "name": name,
            "email": f"traveller{uid}@example.com",
            "preferences": "|".join(sorted(stated, key=TYPES.index)),
        })

        affinity = (type_aff[type_idx] + region_aff[region_idx]
                    + item_latent @ user_latent + item_quality)
        visit_logit = 1.6 * affinity + np.log(fame)
        p = np.exp(visit_logit - visit_logit.max())
        p /= p.sum()
        n_visits = int(np.clip(rng.lognormal(2.0, 0.6), 3, 35))
        visited = rng.choice(n_items, n_visits, replace=False, p=p)

        for j in visited:
            raw = 2.75 + bias + affinity[j] + rng.normal(0.0, 0.55)
            rating = int(np.clip(np.rint(raw), 1, 5))
            dest = destinations[j]
            review = str(rng.choice(REVIEWS[rating])) if rng.random() < 0.3 else ""
            ratings.append({
                "user_id": uid,
                "destination_id": dest.id,
                "rating": rating,
                "visited_on": _visit_date(rng, dest).isoformat(),
                "review_text": review,
            })
    return users, ratings


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--users", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    dests = load_destinations(args.out / "destinations.csv")
    users, ratings = simulate(dests, n_users=args.users, seed=args.seed)
    write_csv(args.out / "users.csv", users)
    write_csv(args.out / "ratings.csv", ratings)
    print(f"wrote {len(users)} users and {len(ratings)} ratings to {args.out}")


if __name__ == "__main__":
    main()
