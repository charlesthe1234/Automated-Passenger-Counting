"""Alembic environment for the CAG Passenger Monitoring backend.

The backend runs as flat modules with `backend/` on sys.path (uvicorn is
started as `main:app` from inside that directory), so this file mirrors that
layout instead of importing a `backend.` package.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import SQLALCHEMY_DATABASE_URL, Base  # noqa: E402
import models  # noqa: F401,E402 - registers the operational tables on Base.
import auth.models  # noqa: F401,E402 - registers the authentication tables on Base.
import runs.models  # noqa: F401,E402 - registers the run-management tables on Base.

config = context.config
config.set_main_option("sqlalchemy.url", SQLALCHEMY_DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _configure(**kwargs) -> None:
    context.configure(
        target_metadata=target_metadata,
        compare_type=True,
        # SQLite cannot ALTER most columns in place; batch mode rewrites the
        # table instead so later feature migrations can add columns safely.
        render_as_batch=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    _configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
