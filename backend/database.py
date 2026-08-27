from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _database_path() -> Path:
    configured = Path(settings.sqlite_db_path)
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parent / configured


DATABASE_PATH = _database_path()
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)


@event.listens_for(engine, "connect")
def _enforce_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    """SQLite ignores ON DELETE clauses unless this pragma is set per connection.

    `evacuee_gallery_views.evacuee_id` has always declared ON DELETE CASCADE but
    never enforced it. Authentication attribution and the later run/alert
    features rely on real ON DELETE behaviour, so it is enabled here for every
    application connection.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def enable_wal_mode() -> None:
    with engine.connect() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL"))
        connection.execute(text("PRAGMA synchronous=NORMAL"))
        connection.commit()


ALEMBIC_INI_PATH = Path(__file__).resolve().parent / "alembic.ini"


def _alembic_config():
    from alembic.config import Config

    config = Config(str(ALEMBIC_INI_PATH))
    config.set_main_option("script_location", str(ALEMBIC_INI_PATH.parent / "migrations"))
    config.set_main_option("sqlalchemy.url", SQLALCHEMY_DATABASE_URL)
    return config


def current_revision() -> str | None:
    """Return the Alembic revision stamped on the database, if any."""
    with engine.connect() as connection:
        has_table = connection.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'")
        ).fetchone()
        if not has_table:
            return None
        row = connection.execute(text("SELECT version_num FROM alembic_version")).fetchone()
        return row[0] if row else None


def _has_legacy_unstamped_schema() -> bool:
    """True when operational tables exist but no Alembic revision is recorded."""
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='metric_logs'")
        ).fetchone()
    return row is not None


def run_migrations() -> None:
    """Upgrade the database to head before any subsystem is allowed to start.

    An operator-facing error is raised instead of silently creating tables, so a
    database in an unrecognised state is never written to by a newer backend.
    """
    from alembic import command

    if _has_legacy_unstamped_schema() and current_revision() is None:
        raise RuntimeError(
            "This database predates the migration system and is not stamped with an "
            "Alembic revision.\n"
            "Back up the database and the configured upload directories first, then run "
            "the documented baseline procedure from inside the backend directory:\n"
            "    .venv-linux/bin/python -m alembic stamp 0001_baseline\n"
            "    .venv-linux/bin/python -m alembic upgrade head\n"
            "See docs/deployment/login_auth_deployment.md before continuing."
        )

    command.upgrade(_alembic_config(), "head")
    logger.info("Database schema is at revision %s", current_revision())


def init_db() -> None:
    import models  # noqa: F401 - registers SQLAlchemy models with Base metadata.
    import auth.models  # noqa: F401 - registers the authentication tables.

    run_migrations()
    enable_wal_mode()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
