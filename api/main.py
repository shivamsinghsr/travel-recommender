"""FastAPI application: routes for destinations and recommendations.

Run locally:
    python -m api.seed
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from . import models, schemas
from .config import Settings, get_settings
from .db import Database
from .services import RecommenderService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    db = Database(settings.database_url)
    service = RecommenderService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with db.SessionLocal() as s:
            service.refresh(s)
        yield
        db.engine.dispose()

    app = FastAPI(
        title="Travel Recommender API",
        version="1.0.0",
        description="Personalised Indian travel recommendations.",
        lifespan=lifespan,
    )
    app.state.db = db
    app.state.service = service
    app.state.settings = settings

    def get_session(request: Request) -> Iterator[Session]:
        yield from request.app.state.db.session()

    SessionDep = Annotated[Session, Depends(get_session)]

    @app.get("/api/health", response_model=schemas.HealthOut, tags=["meta"])
    def health(session: SessionDep) -> schemas.HealthOut:
        try:
            session.execute(text("SELECT 1"))
            n = session.scalar(select(func.count(models.Rating.id))) or 0
            ok = True
        except Exception:
            n, ok = 0, False
        return schemas.HealthOut(
            status="ok" if ok else "degraded", database=ok, ratings=n, recommender="user-knn"
        )

    @app.get("/api/destinations", response_model=list[schemas.DestinationOut], tags=["destinations"])
    def list_destinations(
        session: SessionDep,
        type: schemas.DestinationType | None = None,
        state: str | None = None,
        month: Annotated[int | None, Query(ge=1, le=12)] = None,
    ) -> list[models.Destination]:
        stmt = select(models.Destination).order_by(models.Destination.id)
        if type:
            stmt = stmt.where(models.Destination.type == type)
        if state:
            stmt = stmt.where(models.Destination.state == state)
        rows = session.scalars(stmt).all()
        if month:
            rows = [d for d in rows if month in d.month_list]
        return list(rows)

    @app.get("/api/destinations/{destination_id}", response_model=schemas.DestinationOut, tags=["destinations"])
    def get_destination(destination_id: int, session: SessionDep) -> models.Destination:
        dest = session.get(models.Destination, destination_id)
        if dest is None:
            raise HTTPException(404, f"No destination with id {destination_id}")
        return dest

    @app.get("/api/users/{user_id}/ratings", response_model=list[schemas.RatingOut], tags=["users"])
    def user_ratings(user_id: int, session: SessionDep) -> list[models.Rating]:
        if session.get(models.User, user_id) is None:
            raise HTTPException(404, f"No user with id {user_id}")
        stmt = (
            select(models.Rating)
            .where(models.Rating.user_id == user_id)
            .order_by(models.Rating.visited_on)
        )
        return list(session.scalars(stmt).all())

    @app.get(
        "/api/users/{user_id}/recommendations",
        response_model=schemas.RecommendationsOut,
        tags=["recommendations"],
    )
    def recommendations(
        user_id: int,
        session: SessionDep,
        k: Annotated[int, Query(ge=1, le=settings.max_k)] = settings.default_k,
    ) -> schemas.RecommendationsOut:
        if session.get(models.User, user_id) is None:
            raise HTTPException(404, f"No user with id {user_id}")
        service.ensure_fresh(session)
        recs = service.recommend(user_id, k)
        dests = {d.id: d for d in session.scalars(
            select(models.Destination).where(models.Destination.id.in_([r.destination_id for r in recs]))
        )}
        items = [
            schemas.RecommendationOut(
                destination=schemas.DestinationOut.model_validate(dests[r.destination_id]),
                score=round(r.score, 4),
                predicted_rating=r.predicted_rating,
                reason=r.reason,
                source=r.source,
            )
            for r in recs
        ]
        strategy = "popular" if all(r.source == "popular" for r in recs) else "user-knn"
        return schemas.RecommendationsOut(user_id=user_id, strategy=strategy, items=items)

    return app


app = create_app()
