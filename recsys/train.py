"""Training job: tune, evaluate, gate, and package a model.

    python -m recsys.train --export-web web/model [--baseline live-model.json]

Steps
1. Split ratings by time: each user's latest 20% become the *test* set.
2. Split the remaining ratings by time again: the latest 20% of those become
   the *validation* set. Every candidate configuration (CF model type and
   size, and the hybrid blend weights) is trained on the rest and scored on
   validation; the best NDCG@5 wins. The test set is never used for choices.
3. Retrain the winner and several reference models on train+validation and
   report all of them on the test set.
4. Gate: if a baseline model is given (the one currently live), retrain *its*
   configuration on the same data and compare on the same test set. Publish
   only if the candidate is at least as good (within ``tolerance``).
5. Fit the winner on all ratings and write the artifact.

The API's ``python -m api.train`` runs the same job against PostgreSQL and
records each version in the ``model_versions`` table.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import DATA_DIR, TYPES, Destination, load_destinations
from .data import Interactions, Rating, load_ratings
from .hybrid import HybridRecommender
from .item_knn import ItemKNN
from .metrics import Scores, Split, evaluate, time_split
from .mf import MatrixFactorization
from .user_knn import UserKNN

ARTIFACT_FORMAT = 1
K = 5

CF_GRID: tuple[dict, ...] = (
    {"kind": "item-knn", "k": 20},
    {"kind": "mf", "factors": 8, "reg": 2.0},
    {"kind": "mf", "factors": 16, "reg": 5.0},
    {"kind": "mf", "factors": 24, "reg": 8.0},
)
ALPHA_MAX_GRID = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0)
CONTENT_SHARE_GRID = (0.75, 0.9, 1.0)


@dataclass
class TrainingData:
    destinations: list[Destination]
    ratings: list[Rating]
    preferences: dict[int, list[str]]
    users: list[dict] = field(default_factory=list)  # id, name, preferences (for demo personas)

    @property
    def item_ids(self) -> list[int]:
        return sorted(d.id for d in self.destinations)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for r in sorted(self.ratings, key=lambda r: (r.user_id, r.destination_id)):
            h.update(f"{r.user_id},{r.destination_id},{r.rating},{r.visited_on};".encode())
        return h.hexdigest()[:12]


@dataclass(frozen=True)
class Config:
    cf: Mapping
    alpha_max: float
    content_share: float
    cf_full_weight_at: int = 5

    def as_dict(self) -> dict:
        return {"cf": dict(self.cf), "alpha_max": self.alpha_max, "content_share": self.content_share,
                "cf_full_weight_at": self.cf_full_weight_at}

    @classmethod
    def from_dict(cls, d: Mapping) -> Config:
        return cls(cf=dict(d["cf"]), alpha_max=float(d["alpha_max"]), content_share=float(d["content_share"]),
                   cf_full_weight_at=int(d.get("cf_full_weight_at", 5)))

    def label(self) -> str:
        cf = self.cf["kind"] if self.cf["kind"] == "item-knn" else f"mf({self.cf['factors']}, reg {self.cf['reg']})"
        return f"hybrid {cf}, alpha_max {self.alpha_max}, content {self.content_share}"


PHASE2_DEFAULT = Config(cf={"kind": "item-knn", "k": 20}, alpha_max=1.0, content_share=0.75)


def load_csv_data(data_dir: Path = DATA_DIR) -> TrainingData:
    with open(data_dir / "users.csv", newline="", encoding="utf-8") as fh:
        users = list(csv.DictReader(fh))
    prefs = {int(u["id"]): [p for p in u["preferences"].split("|") if p] for u in users}
    return TrainingData(load_destinations(data_dir / "destinations.csv"), load_ratings(data_dir / "ratings.csv"),
                        prefs, users)


def make_cf(cfg: Mapping):
    if cfg["kind"] == "item-knn":
        return ItemKNN(k=int(cfg.get("k", 20)))
    if cfg["kind"] == "mf":
        return MatrixFactorization(factors=int(cfg["factors"]), reg=float(cfg["reg"]),
                                   bias_reg=float(cfg.get("bias_reg", 2.0)), iterations=int(cfg.get("iterations", 15)))
    raise ValueError(f"unknown CF kind {cfg['kind']!r}")


def fit(data: TrainingData, ratings: Sequence[Rating], config: Config) -> HybridRecommender:
    inter = Interactions.from_ratings(ratings, item_ids=data.item_ids)
    return HybridRecommender.from_interactions(
        data.destinations, make_cf(config.cf).fit(inter), inter,
        alpha_max=config.alpha_max, content_share=config.content_share, cf_full_weight_at=config.cf_full_weight_at,
    )


def score(data: TrainingData, split: Split, hybrid: HybridRecommender, force_alpha: float | None = None) -> Scores:
    def rec(uid: int, known: Mapping[int, float]) -> list[int]:
        res = hybrid.recommend(known, data.preferences.get(uid, []), k=K, force_alpha=force_alpha)
        return [r.destination_id for r in res.items]

    def pred(uid: int, known: Mapping[int, float]) -> dict[int, float]:
        p, _, _ = hybrid.cf.predict(known)
        return dict(zip(data.item_ids, p.tolist(), strict=True))

    return evaluate(split, rec, n_items=len(data.item_ids), k=K, predict=pred)


def tune(data: TrainingData, split: Split, cf_grid=CF_GRID, alpha_grid=ALPHA_MAX_GRID,
         content_grid=CONTENT_SHARE_GRID) -> tuple[Config, list[tuple[Config, Scores]]]:
    """Grid search on the validation split. CF models are fitted once per CF config."""
    results = []
    inter = Interactions.from_ratings(split.train, item_ids=data.item_ids)
    for cf_cfg in cf_grid:
        cf = make_cf(cf_cfg).fit(inter)
        for am, cs in itertools.product(alpha_grid, content_grid):
            cfg = Config(cf=cf_cfg, alpha_max=am, content_share=cs)
            hybrid = HybridRecommender.from_interactions(data.destinations, cf, inter, alpha_max=am, content_share=cs)
            results.append((cfg, score(data, split, hybrid)))
    best = max(results, key=lambda r: (round(r[1].ndcg, 6), r[1].precision))[0]
    return best, results


@dataclass
class TrainingResult:
    version: str
    config: Config
    test_metrics: dict[str, dict]
    candidate: Scores
    baseline: Scores | None
    publish: bool
    decision: str
    hybrid: HybridRecommender
    data: TrainingData
    seconds: float

    def metrics_payload(self) -> dict:
        return {
            "k": K,
            "split": "time-based: each user's latest 20% held out; tuned on a separate validation split",
            "test": self.test_metrics,
            "candidate": self.candidate.as_dict(),
            "baseline": self.baseline.as_dict() if self.baseline else None,
            "decision": self.decision,
        }


def run(data: TrainingData, baseline: Config | None = None, tolerance: float = 0.005,
        grids: dict | None = None) -> TrainingResult:
    t0 = time.time()
    test = time_split(data.ratings)
    validation = time_split(test.train)
    best, _ = tune(data, validation, **(grids or {}))

    # Reference models and the winner, trained on train+validation, scored on test.
    n_items = len(data.item_ids)
    metrics: dict[str, Scores] = {}
    inter = Interactions.from_ratings(test.train, item_ids=data.item_ids)
    pop = fit(data, test.train, PHASE2_DEFAULT)
    metrics["Most popular"] = evaluate(
        test, lambda u, k: [r.destination_id for r in pop.recommend(k, [], K).items], n_items, K)
    metrics["Content-based only"] = score(data, test, pop, force_alpha=0.0)
    user_knn = UserKNN().fit(inter)
    metrics["User-based CF (v1 idea, fixed)"] = evaluate(
        test, lambda u, k: [r.destination_id for r in user_knn.recommend(u, K)], n_items, K)
    metrics["Phase 2 hybrid (untuned)"] = score(data, test, fit(data, test.train, PHASE2_DEFAULT))
    candidate_model = fit(data, test.train, best)
    metrics["CF part alone (" + best.cf["kind"] + ")"] = score(data, test, candidate_model, force_alpha=1.0)
    candidate = score(data, test, candidate_model)
    metrics["Tuned hybrid (published)"] = candidate

    base_scores = None
    if baseline is not None:
        base_scores = score(data, test, fit(data, test.train, baseline))
    if base_scores is None:
        publish, decision = True, "no baseline model: publishing"
    elif candidate.ndcg >= base_scores.ndcg - tolerance:
        publish = True
        decision = f"candidate NDCG@5 {candidate.ndcg:.4f} vs live {base_scores.ndcg:.4f}: publishing"
    else:
        publish = False
        decision = f"candidate NDCG@5 {candidate.ndcg:.4f} worse than live {base_scores.ndcg:.4f}: keeping live model"

    final = fit(data, data.ratings, best)
    now = dt.datetime.now(dt.UTC)
    version = f"{now:%Y%m%d-%H%M}-{data.fingerprint()[:6]}"
    return TrainingResult(
        version=version, config=best, test_metrics={k: v.as_dict() for k, v in metrics.items()},
        candidate=candidate, baseline=base_scores, publish=publish, decision=decision, hybrid=final, data=data,
        seconds=time.time() - t0,
    )


def build_payload(result: TrainingResult, personas: list[dict] | None = None) -> dict:
    data = result.data
    return {
        "format": ARTIFACT_FORMAT,
        "meta": {
            "model_version": result.version,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "cf_model": result.config.label(),
            "n_users": len({r.user_id for r in data.ratings}),
            "n_ratings": len(data.ratings),
            "n_destinations": len(data.destinations),
            "data_fingerprint": data.fingerprint(),
            "types": list(TYPES),
            "training_seconds": round(result.seconds, 1),
        },
        "config": result.config.as_dict(),
        "metrics": result.metrics_payload(),
        "model": result.hybrid.export(),
        "personas": personas or [],
    }


def write_artifact(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return path


def summary_markdown(result: TrainingResult) -> str:
    lines = [f"### Model {result.version}", "", f"**Decision:** {result.decision}", "",
             f"Chosen configuration: {result.config.label()}", "",
             "| Model | Precision@5 | Recall@5 | NDCG@5 | Coverage |", "|---|---|---|---|---|"]
    for name, m in result.test_metrics.items():
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['ndcg']:.3f} | {m['coverage']:.2f} |")
    return "\n".join(lines) + "\n"


def _load_baseline(path: str | None) -> tuple[Config | None, Path | None]:
    if not path or not Path(path).is_file():
        return None, None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return Config.from_dict(payload["config"]), Path(path)
    except (KeyError, ValueError, json.JSONDecodeError):
        print(f"baseline {path} has no usable config; ignoring it")
        return None, None


def main() -> None:
    from .export_web import pick_personas

    ap = argparse.ArgumentParser(description="Train, evaluate and package the recommender (CSV data)")
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--export-web", type=Path, help="write model.json for the static site into this directory")
    ap.add_argument("--baseline", help="model.json of the live model; publish only if the new one is not worse")
    ap.add_argument("--tolerance", type=float, default=0.005)
    args = ap.parse_args()

    data = load_csv_data(args.data)
    baseline_cfg, baseline_path = _load_baseline(args.baseline)
    result = run(data, baseline_cfg, args.tolerance)
    md = summary_markdown(result)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)

    if args.export_web:
        target = args.export_web / "model.json"
        if result.publish:
            inter = Interactions.from_ratings(data.ratings, item_ids=data.item_ids)
            write_artifact(build_payload(result, pick_personas(data.users, inter)), target)
            print(f"wrote {target}")
        elif baseline_path is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(baseline_path, target)
            print(f"kept the live model: copied {baseline_path} to {target}")


if __name__ == "__main__":
    main()
