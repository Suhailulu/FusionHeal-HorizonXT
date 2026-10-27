from __future__ import annotations
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import get_settings

settings = get_settings()

# For SQLite, enable check_same_thread=False for FastAPI concurrency
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency to yield an isolated database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(target_engine=None) -> None:
    """Initialize database tables and run safe column additions."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)
    try:
        from sqlalchemy import text
        with eng.begin() as conn:
            for table_name, table in Base.metadata.tables.items():
                try:
                    res = conn.execute(text(f"PRAGMA table_info({table_name})"))
                    existing_cols = {row[1] for row in res.fetchall()}
                    if existing_cols:
                        for col in table.columns:
                            if col.name not in existing_cols:
                                col_type = col.type.compile(eng.dialect)
                                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}"))
                except Exception:
                    pass
    except Exception:
        pass


# Run initial migration on import
init_db()


