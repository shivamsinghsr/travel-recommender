"""FastAPI application.

Run locally:
    python -m api.seed                # migrate + load data/*.csv
    uvicorn api.main:app --reload     # API at /api, docs at /docs, frontend at /
"""

from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from recsys.hybrid import Filters

from . import models, schemas
from .config import Settings, get_settings
from .db import Database
from .services import RecommenderService


def _user_out(u: models.User) -> schemas.UserOut:
    return schemas.UserOut(id=u.id, name=u.name, email=u.email, preferences=u.preference_list)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    db = Database(settings.database_url)
    service = RecommenderService(min_refresh_seconds=settings.refresh_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with db.SessionLocal() as s:
            service.refresh(s)
        yield
        db.engine.dispose()

    app = FastAPI(
        title="Travel Recommender API",
        version="2.0.0",
        description="Hybrid (collaborative + content-based) recommendations for Indian destinations.",
        lifespan=lifespan,
    )
    app.state.db, app.state.service, app.state.settings = db, service, settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def get_session(request: Request) -> Iterator[Session]:
        yield from request.app.state.db.session()

    SessionDep = Annotated[Session, Depends(get_session)]

    def get_user(user_id: int, session: Session) -> models.User:
        user = session.get(models.User, user_id)
        if user is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No user with id {user_id}")
        return user

    # ---- meta -------------------------------------------------------------
    @app.get("/api/health", response_model=schemas.HealthOut, tags=["meta"])
    def health(session: SessionDep) -> schemas.HealthOut:
        try:
            session.execute(text("SELECT 1"))
            n = session.scalar(select(func.count(models.Rating.id))) or 0
            ok = True
        except Exception:
            n, ok = 0, False
        return schemas.HealthOut(status="ok" if ok else "degraded", database=ok, ratings=n,
                                 recommender=service.name)

    # ---- destinations -----------------------------------------------------
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
        return [d for d in rows if month is None or month in d.month_list]

    @app.get("/api/destinations/{destination_id}", response_model=schemas.DestinationOut, tags=["destinations"])
    def get_destination(destination_id: int, session: SessionDep) -> models.Destination:
        dest = session.get(models.Destination, destination_id)
        if dest is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No destination with id {destination_id}")
        return dest

    # ---- users & ratings --------------------------------------------------
    @app.post("/api/users", response_model=schemas.UserOut, status_code=201, tags=["users"])
    def create_user(body: schemas.UserCreate, session: SessionDep) -> schemas.UserOut:
        user = models.User(name=body.name, email=body.email.lower(), preferences="|".join(body.preferences))
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists") from None
        return _user_out(user)

    @app.get("/api/users/{user_id}", response_model=schemas.UserOut, tags=["users"])
    def read_user(user_id: int, session: SessionDep) -> schemas.UserOut:
        return _user_out(get_user(user_id, session))

    @app.patch("/api/users/{user_id}", response_model=schemas.UserOut, tags=["users"])
    def update_user(user_id: int, body: schemas.UserUpdate, session: SessionDep) -> schemas.UserOut:
        user = get_user(user_id, session)
        if body.name is not None:
            user.name = body.name
        if body.preferences is not None:
            user.preferences = "|".join(dict.fromkeys(body.preferences))
        session.commit()
        return _user_out(user)

    @app.get("/api/users/{user_id}/ratings", response_model=list[schemas.RatingOut], tags=["ratings"])
    def user_ratings(user_id: int, session: SessionDep) -> list[models.Rating]:
        get_user(user_id, session)
        stmt = select(models.Rating).where(models.Rating.user_id == user_id).order_by(models.Rating.id)
        return list(session.scalars(stmt).all())

    @app.post("/api/users/{user_id}/ratings", response_model=schemas.RatingOut, tags=["ratings"])
    def upsert_rating(user_id: int, body: schemas.RatingIn, session: SessionDep, response: Response) -> models.Rating:
        get_user(user_id, session)
        if session.get(models.Destination, body.destination_id) is None:
            raise HTTPException(422, f"No destination with id {body.destination_id}")
        existing = session.scalar(select(models.Rating).where(
            models.Rating.user_id == user_id, models.Rating.destination_id == body.destination_id))
        if existing:
            existing.rating, existing.review_text = body.rating, body.review_text
            existing.visited_on = body.visited_on or existing.visited_on
            rating, response.status_code = existing, status.HTTP_200_OK
        else:
            rating = models.Rating(user_id=user_id, **body.model_dump())
            session.add(rating)
            response.status_code = status.HTTP_201_CREATED
        session.commit()
        return rating

    @app.delete("/api/users/{user_id}/ratings/{destination_id}", status_code=204, tags=["ratings"])
    def delete_rating(user_id: int, destination_id: int, session: SessionDep) -> Response:
        get_user(user_id, session)
        rating = session.scalar(select(models.Rating).where(
            models.Rating.user_id == user_id, models.Rating.destination_id == destination_id))
        if rating is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Rating not found")
        session.delete(rating)
        session.commit()
        return Response(status_code=204)

    # ---- recommendations ---------------------------------------------------
    @app.get("/api/users/{user_id}/recommendations", response_model=schemas.RecommendationsOut,
             tags=["recommendations"])
    def recommendations(
        user_id: int,
        session: SessionDep,
        k: Annotated[int, Query(ge=1, le=settings.max_k)] = settings.default_k,
        type: schemas.DestinationType | None = None,
        state: str | None = None,
        month: Annotated[int | None, Query(ge=1, le=12)] = None,
    ) -> schemas.RecommendationsOut:
        user = get_user(user_id, session)
        service.ensure_fresh(session)
        ratings = {r.destination_id: float(r.rating) for r in session.scalars(
            select(models.Rating).where(models.Rating.user_id == user_id))}
        result = service.recommend(ratings, user.preference_list, k, Filters(type=type, state=state, month=month))
        ids = [r.destination_id for r in result.items]
        dests = {d.id: d for d in session.scalars(select(models.Destination).where(models.Destination.id.in_(ids)))}
        return schemas.RecommendationsOut(
            user_id=user_id,
            strategy=result.strategy,
            alpha=result.alpha,
            items=[
                schemas.RecommendationOut(
                    destination=schemas.DestinationOut.model_validate(dests[r.destination_id]),
                    score=round(r.score, 4), predicted_rating=r.predicted_rating,
                    reason=r.reason, source=r.source,
                )
                for r in result.items
            ],
        )

    # ---- frontend ----------------------------------------------------------
    @app.get("/config.js", include_in_schema=False)
    def frontend_config() -> PlainTextResponse:
        # Served by the API, the frontend talks to this same origin instead of using the static demo model.
        return PlainTextResponse('window.TRAVEL_CONFIG = { mode: "api", apiBase: "" };',
                                 media_type="application/javascript")

    if settings.web_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")

    return app


app = create_app()
