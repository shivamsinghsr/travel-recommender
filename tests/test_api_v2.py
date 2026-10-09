"""Phase 2: users, ratings write-back, hybrid recommendations, filters, migrations."""

from __future__ import annotations

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from api.db import Base


def _new_user(client, prefs=("Beach",), email="asha@example.com"):
    r = client.post("/api/users", json={"name": "Asha", "email": email, "preferences": list(prefs)})
    assert r.status_code == 201, r.text
    return r.json()


def test_migrations_match_models(app):
    with app.state.db.engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_create_user_validates_and_rejects_duplicates(client):
    user = _new_user(client)
    assert user["preferences"] == ["Beach"]
    assert client.post("/api/users", json={"name": "A", "email": "asha@example.com"}).status_code == 409
    assert client.post("/api/users", json={"name": "A", "email": "not-an-email"}).status_code == 422
    assert client.post("/api/users", json={"name": "A", "email": "b@example.com",
                                           "preferences": ["Moon"]}).status_code == 422


def test_new_user_gets_content_based_results(client):
    user = _new_user(client, prefs=["Beach"])
    body = client.get(f"/api/users/{user['id']}/recommendations").json()
    assert body["strategy"] == "content" and body["alpha"] == 0
    assert all(item["destination"]["type"] == "Beach" for item in body["items"])
    assert all(item["predicted_rating"] is None for item in body["items"])


def test_user_without_preferences_gets_popular(client):
    user = _new_user(client, prefs=[])
    assert client.get(f"/api/users/{user['id']}/recommendations").json()["strategy"] == "popular"


def test_rating_moves_user_toward_cf(client):
    uid = _new_user(client)["id"]
    for dest_id in (10, 11, 19):  # three beaches
        r = client.post(f"/api/users/{uid}/ratings", json={"destination_id": dest_id, "rating": 5})
        assert r.status_code == 201
    body = client.get(f"/api/users/{uid}/recommendations").json()
    assert body["strategy"] == "hybrid" and abs(body["alpha"] - 0.6) < 1e-9
    ids = {item["destination"]["id"] for item in body["items"]}
    assert not ids & {10, 11, 19}  # rated places never come back
    assert all(item["predicted_rating"] is not None for item in body["items"])


def test_rating_upsert_and_delete(client):
    uid = _new_user(client)["id"]
    assert client.post(f"/api/users/{uid}/ratings", json={"destination_id": 3, "rating": 2}).status_code == 201
    assert client.post(f"/api/users/{uid}/ratings", json={"destination_id": 3, "rating": 4}).status_code == 200
    ratings = client.get(f"/api/users/{uid}/ratings").json()
    assert [(r["destination_id"], r["rating"]) for r in ratings] == [(3, 4)]
    assert client.delete(f"/api/users/{uid}/ratings/3").status_code == 204
    assert client.get(f"/api/users/{uid}/ratings").json() == []
    assert client.delete(f"/api/users/{uid}/ratings/3").status_code == 404


def test_rating_validation(client):
    uid = _new_user(client)["id"]
    assert client.post(f"/api/users/{uid}/ratings", json={"destination_id": 3, "rating": 6}).status_code == 422
    assert client.post(f"/api/users/{uid}/ratings", json={"destination_id": 999, "rating": 4}).status_code == 422
    assert client.post("/api/users/99999/ratings", json={"destination_id": 3, "rating": 4}).status_code == 404


def test_filters_apply_to_recommendations(client):
    body = client.get("/api/users/1/recommendations", params={"type": "Wildlife", "month": 1, "k": 3}).json()
    assert body["items"]
    for item in body["items"]:
        assert item["destination"]["type"] == "Wildlife"
        assert 1 in item["destination"]["best_months"]


def test_frontend_config_points_at_api(client):
    r = client.get("/config.js")
    assert r.status_code == 200 and 'mode: "api"' in r.text


def test_update_preferences(client):
    uid = _new_user(client, prefs=["Beach"])["id"]
    r = client.patch(f"/api/users/{uid}", json={"preferences": ["Wildlife", "Wildlife", "City"]})
    assert r.status_code == 200 and r.json()["preferences"] == ["Wildlife", "City"]
    body = client.get(f"/api/users/{uid}/recommendations").json()
    assert {i["destination"]["type"] for i in body["items"]} <= {"Wildlife", "City"}
