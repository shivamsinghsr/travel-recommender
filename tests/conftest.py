from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import api.security
from api.config import Settings
from api.main import create_app
from api.seed import migrate, seed

# CI sets TEST_DATABASE_URL to a PostgreSQL service; locally the tests use a temporary SQLite file.
PG_URL = os.environ.get("TEST_DATABASE_URL")
DEMO_PASSWORD = "demo-password-123"


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    monkeypatch.setattr(api.security, "ITERATIONS", 1_000)  # production uses 600k


def _reset_postgres(url: str) -> None:
    from sqlalchemy import text

    from api.db import make_engine

    engine = make_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    engine.dispose()


@pytest.fixture()
def database_url(tmp_path) -> str:
    if PG_URL:
        _reset_postgres(PG_URL)
        return PG_URL
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture()
def settings(database_url, tmp_path) -> Settings:
    return Settings(database_url=database_url, refresh_seconds=0.0, model_check_seconds=0.0,
                    models_dir=tmp_path / "models", demo_password=DEMO_PASSWORD)


@pytest.fixture()
def app(settings):
    migrate(settings.database_url)
    application = create_app(settings)
    with application.state.db.SessionLocal() as s:
        seed(s, demo_password=DEMO_PASSWORD)
    yield application
    application.state.db.engine.dispose()


@pytest.fixture()
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def login(client):
    """login(user_id) -> auth headers for a seeded sample traveller."""
    def _login(user_id: int) -> dict:
        r = client.post("/api/auth/login", json={"email": f"traveller{user_id}@example.com",
                                                 "password": DEMO_PASSWORD})
        assert r.status_code == 200, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    return _login


@pytest.fixture()
def register(client):
    """register(prefs, email) -> (user_id, auth headers) for a brand-new account."""
    def _register(prefs=("Beach",), email="asha@example.com"):
        r = client.post("/api/auth/register", json={"name": "Asha", "email": email,
                                                    "password": "correct horse", "preferences": list(prefs)})
        assert r.status_code == 201, r.text
        body = r.json()
        return body["user"]["id"], {"Authorization": f"Bearer {body['access_token']}"}
    return _register
