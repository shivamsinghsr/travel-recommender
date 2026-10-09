from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.db import Base
from api.main import create_app
from api.seed import seed


@pytest.fixture()
def app(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    application = create_app(settings)
    db = application.state.db
    Base.metadata.create_all(db.engine)
    with db.SessionLocal() as s:
        seed(s)
    return application


@pytest.fixture()
def client(app):
    with TestClient(app) as c:
        yield c
