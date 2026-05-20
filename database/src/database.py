"""Database session and connection management."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import db_settings


# ============================================================
# Async Engine (FastAPI)
# ============================================================

async_engine = create_async_engine(
    db_settings.url,
    echo=db_settings.echo,
    pool_size=db_settings.pool_size,
    max_overflow=db_settings.max_overflow,
    pool_timeout=db_settings.pool_timeout,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ============================================================
# Sync Engine (Alembic / Scripts)
# ============================================================

sync_engine = create_engine(
    db_settings.sync_url,
    echo=db_settings.echo,
    pool_pre_ping=True,
    isolation_level="READ COMMITTED",
)

SyncSessionLocal = sessionmaker(
    sync_engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ============================================================
# Base Model
# ============================================================

class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    pass


# ============================================================
# Session Dependencies
# ============================================================

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_async_session_context() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for session (non-FastAPI use)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_sync_session() -> Session:
    """Get sync session for scripts/migration."""
    return SyncSessionLocal()


# ============================================================
# Utility
# ============================================================

async def init_db() -> None:
    """Create all tables (dev only, use Alembic for production)."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await async_engine.dispose()
