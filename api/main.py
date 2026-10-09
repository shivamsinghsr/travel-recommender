"""FastAPI application.

Run locally:
    python -m api.seed                # migrate + load data/*.csv
    python -m api.train               # optional: train and activate a model
    uvicorn api.main:app --reload     # API at /api, docs at /docs, web app at /
"""

import datetime as dt
import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from recsys.hybrid import Filters

from . import models, schemas
from .cache import make_cache
from .config import Settings, get_settings
from .db import Database
from .model_store import ModelStore
from .security import create_token, hash_password, read_token, verify_password

log = logging.getLogger("travel")
bearer = HTTPBearer(auto_error=False)


def _user_out(u: models.User) -> schemas.UserOut:
    return schemas.UserOut(id=u.id, name=u.name, email=u.email, preferences=u.preference_list)


class LoginThrottle:
    """At most ``limit`` failed sign-ins per email in ``window`` seconds."""

    def __init__(self, limit: int = 10, window: float = 900.0) -> None:
        self.limit, self.window = limit, window
        self._fails: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, q: deque) -> None:
        cutoff = time.monotonic() - self.window
        while q and q[0] < cutoff:
            q.popleft()

    def blocked(self, key: str) -> bool:
        with self._lock:
            q = self._fails[key]
            self._trim(q)
            return len(q) >= self.limit

    def fail(self, key: str) -> None:
        with self._lock:
            self._fails[key].append(time.monotonic())

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    db = Database(settings.database_url)
    store = ModelStore(settings.models_dir, settings.model_check_seconds, settings.refresh_seconds)
    cache = make_cache(settings.redis_url)
    throttle = LoginThrottle()
    token_ttl = dt.timedelta(hours=settings.token_ttl_hours)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with db.SessionLocal() as s:
            served = store.current(s, force_check=True)
            log.info("serving model %s, cache backend %s", served.version, cache.backend)
        yield
        db.engine.dispose()

    app = FastAPI(
        title="Travel Recommender API",
        version="3.0.0",
        description="Hybrid (collaborative + content-based) recommendations for Indian destinations.",
        lifespan=lifespan,
    )
    app.state.db, app.state.store, app.state.cache, app.state.settings = db, store, cache, settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Cache", "X-Model-Version"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response

    def get_session(request: Request) -> Iterator[Session]:
        yield from request.app.state.db.session()

    SessionDep = Annotated[Session, Depends(get_session)]

    def current_user(
        session: SessionDep,
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> models.User:
        uid = read_token(creds.credentials, settings.secret_key) if creds else None
        user = session.get(models.User, uid) if uid is not None else None
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue",
                                headers={"WWW-Authenticate": "Bearer"})
        return user

    CurrentUser = Annotated[models.User, Depends(current_user)]

    def own(user_id: int, me: models.User) -> models.User:
        if me.id != user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "You can only access your own profile")
        return me

    def bump(user_id: int) -> None:
        cache.incr(f"recs-gen:{user_id}")

    def token_for(user: models.User) -> schemas.TokenOut:
        return schemas.TokenOut(access_token=create_token(user.id, settings.secret_key, token_ttl),
                                expires_in=int(token_ttl.total_seconds()), user=_user_out(user))

    # ---- meta -------------------------------------------------------------
    @app.get("/api/health", response_model=schemas.HealthOut, tags=["meta"])
    def health(session: SessionDep) -> schemas.HealthOut:
        try:
            session.execute(text("SELECT 1"))
            n = session.scalar(select(func.count(models.Rating.id))) or 0
            served = store.current(session)
            ok, version = True, served.version
        except Exception:
            n, ok, version = 0, False, "unavailable"
        cache_state = cache.backend if cache.ping() else f"{cache.backend} (unreachable)"
        return schemas.HealthOut(status="ok" if ok else "degraded", database=ok, cache=cache_state,
                                 ratings=n, model_version=version)

    @app.get("/api/model", response_model=schemas.ModelOut, tags=["meta"])
    def model_info(session: SessionDep) -> schemas.ModelOut:
        served = store.current(session)
        desc = ("Item-based model fitted from live ratings (no trained model is active yet)"
                if served.version == "live-fit" else "Trained by the nightly job")
        return schemas.ModelOut(version=served.version, description=desc, config=served.config,
                                metrics=served.metrics)

    # ---- auth -------------------------------------------------------------
    @app.post("/api/auth/register", response_model=schemas.TokenOut, status_code=201, tags=["auth"])
    def register(body: schemas.RegisterIn, session: SessionDep) -> schemas.TokenOut:
        user = models.User(name=body.name.strip(), email=body.email.lower(),
                           preferences="|".join(dict.fromkeys(body.preferences)),
                           password_hash=hash_password(body.password))
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists") from None
        return token_for(user)

    @app.post("/api/auth/login", response_model=schemas.TokenOut, tags=["auth"])
    def login(body: schemas.LoginIn, session: SessionDep) -> schemas.TokenOut:
        key = body.email.lower()
        if throttle.blocked(key):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                                "Too many failed sign-ins for this email. Try again in 15 minutes.")
        user = session.scalar(select(models.User).where(models.User.email == key))
        if user is None or not verify_password(body.password, user.password_hash):
            throttle.fail(key)
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Email or password is incorrect")
        throttle.reset(key)
        return token_for(user)

    @app.get("/api/me", response_model=schemas.UserOut, tags=["auth"])
    def me(user: CurrentUser) -> schemas.UserOut:
        return _user_out(user)

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

    # ---- users & ratings (signed-in user only) ------------------------------
    @app.get("/api/users/{user_id}", response_model=schemas.UserOut, tags=["users"])
    def read_user(user_id: int, me: CurrentUser) -> schemas.UserOut:
        return _user_out(own(user_id, me))

    @app.patch("/api/users/{user_id}", response_model=schemas.UserOut, tags=["users"])
    def update_user(user_id: int, body: schemas.UserUpdate, me: CurrentUser, session: SessionDep) -> schemas.UserOut:
        user = own(user_id, me)
        if body.name is not None:
            user.name = body.name
        if body.preferences is not None:
            user.preferences = "|".join(dict.fromkeys(body.preferences))
        session.commit()
        bump(user_id)
        return _user_out(user)

    @app.get("/api/users/{user_id}/ratings", response_model=list[schemas.RatingOut], tags=["ratings"])
    def user_ratings(user_id: int, me: CurrentUser, session: SessionDep) -> list[models.Rating]:
        own(user_id, me)
        stmt = select(models.Rating).where(models.Rating.user_id == user_id).order_by(models.Rating.id)
        return list(session.scalars(stmt).all())

    @app.post("/api/users/{user_id}/ratings", response_model=schemas.RatingOut, tags=["ratings"])
    def upsert_rating(user_id: int, body: schemas.RatingIn, me: CurrentUser, session: SessionDep,
                      response: Response) -> models.Rating:
        own(user_id, me)
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
        bump(user_id)
        return rating

    @app.delete("/api/users/{user_id}/ratings/{destination_id}", status_code=204, tags=["ratings"])
    def delete_rating(user_id: int, destination_id: int, me: CurrentUser, session: SessionDep) -> Response:
        own(user_id, me)
        rating = session.scalar(select(models.Rating).where(
            models.Rating.user_id == user_id, models.Rating.destination_id == destination_id))
        if rating is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Rating not found")
        session.delete(rating)
        session.commit()
        bump(user_id)
        return Response(status_code=204)

    # ---- recommendations ---------------------------------------------------
    @app.get("/api/users/{user_id}/recommendations", response_model=schemas.RecommendationsOut,
             tags=["recommendations"])
    def recommendations(
        user_id: int,
        me: CurrentUser,
        session: SessionDep,
        response: Response,
        k: Annotated[int, Query(ge=1, le=settings.max_k)] = settings.default_k,
        type: schemas.DestinationType | None = None,
        state: str | None = None,
        month: Annotated[int | None, Query(ge=1, le=12)] = None,
    ) -> schemas.RecommendationsOut:
        user = own(user_id, me)
        served = store.current(session)
        gen = cache.get(f"recs-gen:{user_id}") or "0"
        key = f"recs:{served.version}:{user_id}:{gen}:{k}:{type}:{state}:{month}"
        response.headers["X-Model-Version"] = served.version
        if (hit := cache.get(key)) is not None:
            response.headers["X-Cache"] = "HIT"
            return schemas.RecommendationsOut.model_validate_json(hit)

        ratings = {r.destination_id: float(r.rating) for r in session.scalars(
            select(models.Rating).where(models.Rating.user_id == user_id).order_by(models.Rating.id))}
        result = served.model.recommend(ratings, user.preference_list, k=k,
                                        filters=Filters(type=type, state=state, month=month))
        ids = [r.destination_id for r in result.items]
        dests = {d.id: d for d in session.scalars(select(models.Destination).where(models.Destination.id.in_(ids)))}
        out = schemas.RecommendationsOut(
            user_id=user_id, strategy=result.strategy, alpha=round(result.alpha, 4), model_version=served.version,
            items=[
                schemas.RecommendationOut(
                    destination=schemas.DestinationOut.model_validate(dests[r.destination_id]),
                    score=round(r.score, 4), predicted_rating=r.predicted_rating, reason=r.reason, source=r.source,
                )
                for r in result.items
            ],
        )
        cache.set(key, out.model_dump_json(), settings.cache_ttl_seconds)
        response.headers["X-Cache"] = "MISS"
        return out

    # ---- frontend ----------------------------------------------------------
    @app.get("/config.js", include_in_schema=False)
    def frontend_config() -> PlainTextResponse:
        # Served by the API, the web app talks to this same origin instead of using the static demo model.
        cfg = {"mode": "api", "apiBase": ""}
        return PlainTextResponse(f"window.TRAVEL_CONFIG = {json.dumps(cfg)};", media_type="application/javascript")

    if settings.web_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")

    return app


app = create_app()
