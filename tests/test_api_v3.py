"""Phase 3: sign-in, authorisation, caching, trained-model hot swap."""

from __future__ import annotations

import os
import uuid

import fakeredis
import pytest

from api.cache import RedisCache
from api.config import Settings
from api.train import train

SMALL_GRID = {
    "cf_grid": ({"kind": "mf", "factors": 8, "reg": 2.0},),
    "alpha_grid": (0.0, 0.3),
    "content_grid": (0.75,),
}


def test_endpoints_require_sign_in(client, login):
    assert client.get("/api/users/1/recommendations").status_code == 401
    assert client.get("/api/users/1/recommendations",
                      headers={"Authorization": "Bearer not-a-token"}).status_code == 401
    other = login(2)
    assert client.get("/api/users/1/recommendations", headers=other).status_code == 403
    assert client.post("/api/users/1/ratings", json={"destination_id": 3, "rating": 5},
                       headers=other).status_code == 403


def test_tokens_signed_with_another_key_are_rejected(client):
    from api.security import create_token

    forged = create_token(1, "some-other-secret-that-is-long-enough-123", __import__("datetime").timedelta(hours=1))
    assert client.get("/api/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_login_errors_and_throttle(client, register):
    register(email="lock@example.com")
    bad = {"email": "lock@example.com", "password": "wrong password"}
    assert client.post("/api/auth/login", json=bad).status_code == 401
    unknown = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "whatever1"})
    assert unknown.status_code == 401 and unknown.json()["detail"] == "Email or password is incorrect"
    for _ in range(9):
        client.post("/api/auth/login", json=bad)
    assert client.post("/api/auth/login", json=bad).status_code == 429
    # even the right password is refused while throttled
    good = {"email": "lock@example.com", "password": "correct horse"}
    assert client.post("/api/auth/login", json=good).status_code == 429


def test_recommendations_are_cached_and_invalidated(client, register):
    uid, h = register()
    url = f"/api/users/{uid}/recommendations"
    first = client.get(url, headers=h)
    assert first.headers["X-Cache"] == "MISS"
    second = client.get(url, headers=h)
    assert second.headers["X-Cache"] == "HIT" and second.json() == first.json()
    top = first.json()["items"][0]["destination"]["id"]
    client.post(f"/api/users/{uid}/ratings", json={"destination_id": top, "rating": 2}, headers=h)
    third = client.get(url, headers=h)
    assert third.headers["X-Cache"] == "MISS"
    assert top not in [i["destination"]["id"] for i in third.json()["items"]]


def test_training_publishes_and_api_hot_swaps(app, client, login, settings):
    h = login(1)
    assert client.get("/api/users/1/recommendations", headers=h).headers["X-Model-Version"] == "live-fit"

    with app.state.db.SessionLocal() as s:
        outcome = train(s, settings.models_dir, grids=SMALL_GRID)
    assert outcome["published"], outcome
    version = outcome["version"]
    assert (settings.models_dir / version / "model.json").is_file()

    r = client.get("/api/users/1/recommendations", headers=h)
    assert r.status_code == 200 and r.headers["X-Model-Version"] == version
    info = client.get("/api/model").json()
    assert info["version"] == version and info["metrics"]["test"]
    assert client.get("/api/health").json()["model_version"] == version

    # A gate that demands a big improvement keeps the active model.
    with app.state.db.SessionLocal() as s:
        again = train(s, settings.models_dir, tolerance=-1.0, grids=SMALL_GRID)
    assert not again["published"]
    assert client.get("/api/model").json()["version"] == version


def test_production_requires_a_real_secret():
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(environment="production")
    assert Settings(environment="production", secret_key="x" * 40).environment == "production"


def test_redis_cache_round_trip():
    url = os.environ.get("TEST_REDIS_URL")
    cache = RedisCache(url) if url else RedisCache.__new__(RedisCache)
    if not url:
        cache.client = fakeredis.FakeRedis(decode_responses=True)
    assert cache.ping()
    key = f"test:{uuid.uuid4()}"  # unique, so a real shared Redis can't hold stale values
    cache.set(key, "v", ttl=60)
    assert cache.get(key) == "v"
    assert cache.incr(f"{key}:gen") == 1 and cache.incr(f"{key}:gen") == 2


def test_redis_outage_does_not_break_requests():
    cache = RedisCache("redis://127.0.0.1:1/0")  # nothing listens on port 1
    assert cache.get("x") is None and cache.ping() is False
    cache.set("x", "y", ttl=1)  # silently ignored
