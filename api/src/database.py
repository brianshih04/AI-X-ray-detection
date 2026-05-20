"""SQLAlchemy database engine & session factory."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from src.config import settings


class Base(DeclarativeBase):
    pass


# Lazy engine initialization — avoid importing psycopg2 at module level
# when running tests with SQLite
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def init_engine(url: str | None = None):
    """Initialize engine with explicit URL (used by tests to override)."""
    global _engine, _SessionLocal
    _engine = create_engine(url or settings.DATABASE_URL, pool_pre_ping=True)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def get_db():
    """FastAPI dependency that yields a DB session."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()
