"""Application configuration from environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration."""

    url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/chestxpert",
        description="Async database URL",
    )
    sync_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/chestxpert",
        description="Sync database URL for Alembic",
    )
    echo: bool = Field(default=False, description="Log SQL queries")
    pool_size: int = Field(default=20, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=50)
    pool_timeout: int = Field(default=30, ge=1)


class JWTSettings(BaseSettings):
    """JWT authentication settings."""

    secret_key: str = Field(default="dev-secret-change-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=1)


class RateLimitSettings(BaseSettings):
    """API rate limiting configuration."""

    requests_per_minute: int = Field(default=60, ge=1)
    burst: int = Field(default=10, ge=1)


class AppSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        case_sensitive=False,
    )

    # App metadata
    app_name: str = "ChestXpert API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)

    # Paths
    image_storage_path: Path = Field(
        default=Path("/data/nih-chest-xrays/images"),
        description="Root path for NIH chest X-ray images",
    )
    csv_data_path: Path = Field(
        default=Path("/data/nih-chest-xrays"),
        description="Path to NIH CSV metadata files",
    )

    # Model
    default_model_version: str = Field(default="densenet121-v1.0")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Sub-modules
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)


@lru_cache
def get_settings() -> AppSettings:
    """Get cached application settings (singleton)."""
    return AppSettings()


# Convenience accessors
settings = get_settings()
db_settings = settings.database
jwt_settings = settings.jwt
rl_settings = settings.rate_limit
