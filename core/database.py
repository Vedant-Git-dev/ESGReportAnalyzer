"""
core/database.py
SQLAlchemy 2.x engine + session factory.
All agents import get_db() or engine from here — never create their own.
"""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.config import get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def _build_engine():
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,          # detect stale connections
        pool_size=5,
        max_overflow=10,
        echo=False,                  # set True for SQL debug
        future=True,
    )
    logger.info("database.engine_created", url=settings.database_url.split("@")[-1])
    return engine


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context-manager session. Use as:
        with get_db() as db:
            db.query(...)
    Rolls back on exception, always closes.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Quick health check — returns True if DB is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("database.connection_failed", error=str(exc))
        return False


def create_all_tables() -> None:
    """Create all tables defined in models. Safe to call multiple times."""
    from models import db_models  # noqa: F401 — side-effect import registers models
    Base.metadata.create_all(bind=engine)
    logger.info("database.tables_created")