"""Destination catalogue: loading, and the fixed vocabularies used everywhere."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TYPES: tuple[str, ...] = (
    "Beach",
    "Hill Station",
    "Heritage",
    "Nature",
    "Adventure",
    "Spiritual",
    "Wildlife",
    "City",
)

REGIONS: tuple[str, ...] = ("North", "South", "East", "West", "Northeast", "Central")

STATE_REGION: dict[str, str] = {
    "Uttar Pradesh": "North",
    "Delhi": "North",
    "Punjab": "North",
    "Himachal Pradesh": "North",
    "Uttarakhand": "North",
    "Jammu and Kashmir": "North",
    "Ladakh": "North",
    "Rajasthan": "West",
    "Gujarat": "West",
    "Goa": "West",
    "Maharashtra": "West",
    "Dadra and Nagar Haveli and Daman and Diu": "West",
    "Kerala": "South",
    "Karnataka": "South",
    "Tamil Nadu": "South",
    "Puducherry": "South",
    "Telangana": "South",
    "Lakshadweep": "South",
    "West Bengal": "East",
    "Odisha": "East",
    "Bihar": "East",
    "Andaman and Nicobar Islands": "East",
    "Sikkim": "Northeast",
    "Meghalaya": "Northeast",
    "Assam": "Northeast",
    "Arunachal Pradesh": "Northeast",
    "Madhya Pradesh": "Central",
}


@dataclass(frozen=True)
class Destination:
    id: int
    name: str
    state: str
    type: str
    best_months: tuple[int, ...]
    description: str = ""
    region: str = field(default="")

    def in_season(self, month: int) -> bool:
        return month in self.best_months


def parse_months(value: str) -> tuple[int, ...]:
    months = tuple(int(m) for m in str(value).split("|") if m.strip())
    if not months or any(m < 1 or m > 12 for m in months):
        raise ValueError(f"invalid best_months value: {value!r}")
    return months


def load_destinations(path: Path | str | None = None) -> list[Destination]:
    """Read data/destinations.csv and validate every row."""
    path = Path(path) if path else DATA_DIR / "destinations.csv"
    out: list[Destination] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["type"] not in TYPES:
                raise ValueError(f"unknown type {row['type']!r} for {row['name']}")
            if row["state"] not in STATE_REGION:
                raise ValueError(f"unknown state {row['state']!r} for {row['name']}")
            out.append(
                Destination(
                    id=int(row["id"]),
                    name=row["name"],
                    state=row["state"],
                    type=row["type"],
                    best_months=parse_months(row["best_months"]),
                    description=row.get("description", ""),
                    region=STATE_REGION[row["state"]],
                )
            )
    ids = [d.id for d in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate destination ids")
    return out
