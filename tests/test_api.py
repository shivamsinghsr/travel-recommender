"""Phase 1 behaviour, now behind sign-in."""

from __future__ import annotations


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["database"] is True
    assert body["ratings"] > 1000
    assert body["model_version"] == "live-fit"  # nothing trained yet
    assert body["cache"] == "memory"


def test_list_and_filter_destinations(client):
    assert len(client.get("/api/destinations").json()) == 66
    beaches = client.get("/api/destinations", params={"type": "Beach"}).json()
    assert beaches and all(d["type"] == "Beach" for d in beaches)
    july = client.get("/api/destinations", params={"month": 7}).json()
    assert july and all(7 in d["best_months"] for d in july)


def test_invalid_filters_are_rejected(client):
    assert client.get("/api/destinations", params={"type": "Moon"}).status_code == 422
    assert client.get("/api/destinations", params={"month": 13}).status_code == 422
    assert client.get("/api/destinations/9999").status_code == 404


def test_recommendations_exclude_visited(client, login):
    h = login(1)
    rated = {r["destination_id"] for r in client.get("/api/users/1/ratings", headers=h).json()}
    body = client.get("/api/users/1/recommendations", params={"k": 5}, headers=h).json()
    ids = [item["destination"]["id"] for item in body["items"]]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert not rated & set(ids)
    assert all(item["reason"] and 1 <= item["predicted_rating"] <= 5 for item in body["items"])


def test_high_user_id_does_not_crash(client, login):
    """v1 crashed with IndexError (HTTP 500) for user ids above the matrix size."""
    assert client.get("/api/users/800/recommendations", headers=login(800)).status_code == 200


def test_k_is_bounded(client, login):
    h = login(1)
    assert client.get("/api/users/1/recommendations", params={"k": 0}, headers=h).status_code == 422
    assert client.get("/api/users/1/recommendations", params={"k": 500}, headers=h).status_code == 422
