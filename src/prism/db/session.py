"""Engine / session helpers and schema bootstrap."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from prism.config import get_settings
from prism.db.models import Base

_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create extensions + tables. Stage 0 bootstrap (Alembic later)."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.bm25_dir.mkdir(parents=True, exist_ok=True)
    try:
        from prism.rag.index.vector_ann import ensure_vector_index

        ensure_vector_index(engine, settings.vector_search)
    except Exception:
        # Index build can fail on empty/odd DBs — search still works via seq scan.
        pass
