"""Phase 3: matrix factorisation, evaluation and the training job's publish gate."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recsys.data import Interactions, Rating, load_ratings
from recsys.hybrid import HybridRecommender
from recsys.metrics import ndcg_at_k, time_split
from recsys.mf import MatrixFactorization
from recsys.train import Config, build_payload, load_csv_data, run

SMALL_GRID = {
    "cf_grid": ({"kind": "item-knn", "k": 20}, {"kind": "mf", "factors": 8, "reg": 2.0}),
    "alpha_grid": (0.0, 0.3),
    "content_grid": (0.75,),
}


@pytest.fixture(scope="module")
def data():
    return load_csv_data()


@pytest.fixture(scope="module")
def result(data):
    return run(data, grids=SMALL_GRID)


def test_mf_fits_and_round_trips():
    inter = Interactions.from_ratings(load_ratings())
    mf = MatrixFactorization(factors=8, reg=2.0, iterations=10).fit(inter)
    assert mf.train_rmse(inter) < 0.85
    ratings = inter.user_ratings(int(inter.user_ids[0]))
    clone = MatrixFactorization.from_dict(json.loads(json.dumps(mf.to_dict())), list(map(int, inter.item_ids)))
    a, _, best_a = mf.predict(ratings)
    b, _, best_b = clone.predict(ratings)
    assert np.allclose(a, b) and (best_a == best_b).all()
    assert ((1 <= a) & (a <= 5)).all()


def test_mf_fold_in_tracks_tastes():
    inter = Interactions.from_ratings(load_ratings())
    mf = MatrixFactorization(factors=8, reg=2.0).fit(inter)
    low, _, _ = mf.predict({27: 1, 30: 1, 33: 1})  # dislikes Himalayan adventure
    high, _, _ = mf.predict({27: 5, 30: 5, 33: 5})  # loves it
    spiti_like = [mf.item_index[d] for d in (34, 29, 40)]  # other mountain-adventure places
    assert high[spiti_like].mean() > low[spiti_like].mean()


def test_time_split_holds_out_latest():
    ratings = [Rating(1, d, 4, f"2024-{d:02d}-01") for d in range(1, 11)] + [Rating(2, 1, 5, "2024-01-01")]
    split = time_split(ratings)
    assert set(split.test[1]) == {9, 10}  # the latest 20% of ten
    assert 2 not in split.test  # too few ratings to hold any out


def test_ndcg():
    assert ndcg_at_k([1, 2, 3], {1}, 3) == 1.0
    assert ndcg_at_k([2, 1, 3], {1}, 3) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k([2, 3], {1}, 2) == 0.0


def test_training_beats_baselines(result):
    test = result.test_metrics
    tuned = test["Tuned hybrid (published)"]["ndcg"]
    assert tuned > test["Most popular"]["ndcg"]
    assert tuned > test["Phase 2 hybrid (untuned)"]["ndcg"]
    assert tuned >= test["Content-based only"]["ndcg"] - 0.01
    assert result.publish


def test_gate_keeps_live_model_when_candidate_is_worse(data, result):
    # Same configuration as the candidate, but demand a large improvement: must refuse.
    strict = run(data, baseline=result.config, tolerance=-1.0, grids=SMALL_GRID)
    assert not strict.publish and "keeping live model" in strict.decision
    lenient = run(data, baseline=result.config, tolerance=0.005, grids=SMALL_GRID)
    assert lenient.publish


def test_artifact_rebuilds_identical_model(result):
    payload = json.loads(json.dumps(build_payload(result)))
    assert payload["format"] == 1 and payload["meta"]["model_version"] == result.version
    rebuilt = HybridRecommender.from_export(payload["model"])
    ratings = {10: 5, 19: 4, 27: 2}
    a = result.hybrid.recommend(ratings, ["Beach"], k=8)
    b = rebuilt.recommend(ratings, ["Beach"], k=8)
    assert [r.destination_id for r in a.items] == [r.destination_id for r in b.items]
    assert Config.from_dict(payload["config"]) == result.config
