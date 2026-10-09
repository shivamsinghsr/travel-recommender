"""Settings read from environment variables (or a local .env file)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./travel.db"
    default_k: int = 5
    max_k: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
