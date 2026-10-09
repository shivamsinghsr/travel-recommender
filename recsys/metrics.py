"""Offline evaluation with a time-based holdout.

For every user with at least ``min_ratings`` ratings, their most recent 20%
(at least one) are hidden. Models are trained on everything else and asked
for top-k recommendations from places the user had not yet rated. A hidden
place the user rated 4 or 5 counts as a hit.

Splitting by time, not at random, mirrors real use: the model must predict
where someone goes *next* from where they have been.

Metrics, averaged over users with at least one relevant hidden place:
  precision@k  share of the k recommendations that were hits
  recall@k     share of the user's hidden liked places that were recommended
  ndcg@k       like recall, but hits near the top count more
  coverage     share of the catalogue that appears in anyone's top k
  rmse         rating-prediction error on every hidden rating (CF models only)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .data import Rating

RELEVANT_AT = 4.0


@dataclass(frozen=True)
class Split:
    train: list[Rating]
    test: dict[int, dict[int, float]]  # user -> {destination: rating}


def time_split(ratings: Sequence[Rating], holdout: float = 0.2, min_ratings: int = 5) -> Split:
    by_user: dict[int, list[Rating]] = defaultdict(list)
    for r in ratings:
        by_user[r.user_id].append(r)
    train: list[Rating] = []
    test: dict[int, dict[int, float]] = {}
    for uid, rs in by_user.items():
        rs = sorted(rs, key=lambda r: (r.visited_on, r.destination_id))
        if len(rs) < min_ratings:
            train.extend(rs)
            continue
        n_test = max(1, int(round(len(rs) * holdout)))
        train.extend(rs[:-n_test])
        test[uid] = {r.destination_id: r.rating for r in rs[-n_test:]}
    return Split(train, test)


@dataclass(frozen=True)
class Scores:
    precision: float
    recall: float
    ndcg: float
    coverage: float
    rmse: float | None
    users: int

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


RecommendFn = Callable[[int, Mapping[int, float]], list[int]]  # (user, train ratings) -> ranked ids
PredictFn = Callable[[int, Mapping[int, float]], Mapping[int, float]]  # -> {dest: predicted rating}


def evaluate(
    split: Split,
    recommend: RecommendFn,
    n_items: int,
    k: int = 5,
    predict: PredictFn | None = None,
) -> Scores:
    train_by_user: dict[int, dict[int, float]] = defaultdict(dict)
    for r in split.train:
        train_by_user[r.user_id][r.destination_id] = r.rating

    precisions, recalls, ndcgs, sq_errors = [], [], [], []
    shown: set[int] = set()
    for uid, hidden in split.test.items():
        known = train_by_user.get(uid, {})
        if predict is not None:
            preds = predict(uid, known)
            sq_errors += [(preds[d] - r) ** 2 for d, r in hidden.items() if d in preds]
        relevant = {d for d, r in hidden.items() if r >= RELEVANT_AT}
        if not relevant:
            continue
        ranked = recommend(uid, known)[:k]
        shown.update(ranked)
        hits = sum(1 for d in ranked if d in relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))

    return Scores(
        precision=float(np.mean(precisions)) if precisions else 0.0,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
        coverage=len(shown) / n_items,
        rmse=float(np.sqrt(np.mean(sq_errors))) if sq_errors else None,
        users=len(precisions),
    )
