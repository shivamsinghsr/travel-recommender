"""Settings read from environment variables (or a local .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./travel.db"
    default_k: int = 5
    max_k: int = 20
    # Comma-separated list, e.g. "https://you.github.io,http://localhost:5173"
    cors_origins: str = "*"
    # Directory with the static frontend; served at "/" when it exists
    web_dir: Path = ROOT / "web"
    refresh_seconds: float = 30.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
