"""Settings read from environment variables (or a local .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DEV_SECRET = "dev-only-secret-change-me-in-production-0123456789"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"  # set to "production" when deployed
    database_url: str = "sqlite:///./travel.db"
    redis_url: str | None = None  # e.g. redis://redis:6379/0; in-process cache when unset
    secret_key: str = DEV_SECRET  # signs access tokens; must be set in production
    token_ttl_hours: int = 24 * 7
    demo_password: str | None = None  # if set, seeded sample travellers can sign in with it

    default_k: int = 5
    max_k: int = 20
    cache_ttl_seconds: int = 3600
    # Comma-separated list, e.g. "https://you.github.io,http://localhost:5173"
    cors_origins: str = "*"
    web_dir: Path = ROOT / "web"
    models_dir: Path = ROOT / "models"
    model_check_seconds: float = 900.0
    refresh_seconds: float = 30.0

    @model_validator(mode="after")
    def _production_needs_a_secret(self) -> Settings:
        if self.environment == "production" and (self.secret_key == DEV_SECRET or len(self.secret_key) < 32):
            raise ValueError("SECRET_KEY must be set to a random value of 32+ characters in production")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
