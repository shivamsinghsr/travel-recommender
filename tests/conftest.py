from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import create_app
from api.seed import migrate, seed

# CI sets TEST_DATABASE_URL to a PostgreSQL service; locally the tests use a temporary SQLite file.
PG_URL = os.environ.get("TEST_DATABASE_URL")


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
    return Settings(database_url=database_url, refresh_seconds=0.0)


@pytest.fixture()
def app(settings):
    migrate(settings.database_url)
    application = create_app(settings)
    with application.state.db.SessionLocal() as s:
        seed(s)
    yield application
    application.state.db.engine.dispose()


@pytest.fixture()
def client(app):
    with TestClient(app) as c:
        yield c
