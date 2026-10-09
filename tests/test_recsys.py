from __future__ import annotations

import numpy as np

from recsys.catalog import STATE_REGION, TYPES, load_destinations
from recsys.data import Interactions, Rating, load_ratings
from recsys.simulate import simulate
from recsys.user_knn import UserKNN


def test_catalogue_is_valid():
    dests = load_destinations()
    assert len(dests) >= 60
    assert {d.type for d in dests} == set(TYPES)
    assert all(d.state in STATE_REGION for d in dests)
    assert all(d.best_months for d in dests)


def test_simulation_is_deterministic():
    dests = load_destinations()
    a = simulate(dests, n_users=50, seed=7)
    b = simulate(dests, n_users=50, seed=7)
    assert a == b
    _, ratings = a
    assert all(1 <= r["rating"] <= 5 for r in ratings)


def test_id_mapping_handles_gaps():
    """v1 used user_id - 1 as the row; with gaps in IDs that picks the wrong user."""
    ratings = [Rating(1, 10, 5), Rating(5, 10, 2), Rating(5, 11, 4), Rating(9, 12, 3)]
    inter = Interactions.from_ratings(ratings)
    assert inter.user_ratings(5) == {10: 2.0, 11: 4.0}
    assert inter.user_ratings(9) == {12: 3.0}
    assert inter.user_ratings(4) == {}  # unknown user, no crash
    assert inter.user_ratings(10_000) == {}


def test_user_knn_never_recommends_rated_places():
    inter = Interactions.from_ratings(load_ratings())
    model = UserKNN().fit(inter)
    for uid in inter.user_ids[:25]:
        rated = set(inter.user_ratings(int(uid)))
        recs = model.recommend(int(uid), k=5)
        assert len(recs) == 5
        assert not rated & {r.destination_id for r in recs}
        assert all(0.0 <= r.score <= 1.0 for r in recs)


def test_unknown_user_gets_popular_places():
    inter = Interactions.from_ratings(load_ratings())
    recs = UserKNN().fit(inter).recommend(999_999, k=5)
    assert [r.source for r in recs] == ["popular"] * 5


def test_candidates_are_respected():
    inter = Interactions.from_ratings(load_ratings())
    model = UserKNN().fit(inter)
    allowed = {1, 2, 3, 4, 5, 6, 7, 8}
    uid = int(inter.user_ids[0])
    recs = model.recommend(uid, k=5, candidates=allowed)
    assert {r.destination_id for r in recs} <= allowed - set(inter.user_ratings(uid))


def test_similarity_is_symmetric_and_bounded():
    inter = Interactions.from_ratings(load_ratings())
    sim = UserKNN().fit(inter).sim
    assert np.allclose(sim, sim.T)
    assert sim.max() <= 1.0 + 1e-9 and sim.min() >= -1.0 - 1e-9
