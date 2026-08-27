"""One serialized SQLite write path for run-scoped ingestion and deletion.

Checking a tombstone and then inserting a row are two statements. Under SQLite's
default deferred transactions the check runs in a read transaction that another
connection can delete underneath, so a late MQTT payload could resurrect a run
that was just deleted. Every run-scoped write therefore goes through
`immediate_write()`, which issues `BEGIN IMMEDIATE` and takes the write lock
before the first read.

A dedicated engine is used so that disabling pysqlite's implicit transaction
handling does not alter the behaviour of ordinary request sessions.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from config import settings
from database import SQLALCHEMY_DATABASE_URL

logger = logging.getLogger(__name__)

_BUSY_TIMEOUT_SECONDS = settings.sqlite_busy_timeout_ms / 1000.0

write_engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        # Autocommit at the driver level: pysqlite must not start its own
        # transaction, so the "begin" hook below can issue BEGIN IMMEDIATE.
        "isolation_level": None,
        "timeout": _BUSY_TIMEOUT_SECONDS,
    },
    poolclass=NullPool,
    future=True,
)


@event.listens_for(write_engine, "connect")
def _configure_connection(dbapi_connection, connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={settings.sqlite_busy_timeout_ms}")
    finally:
        cursor.close()


@event.listens_for(write_engine, "begin")
def _begin_immediate(connection) -> None:
    connection.exec_driver_sql("BEGIN IMMEDIATE")


class RunWriteConflict(RuntimeError):
    """The write lock could not be acquired within the configured retries."""


def _is_lock_contention(error: OperationalError) -> bool:
    message = str(error.orig).lower()
    return "locked" in message or "busy" in message


def _acquire_write_transaction():
    """Take the SQLite write lock, retrying only while it is contended.

    Retrying happens strictly before any caller work runs. Once the caller has
    started writing, a failure must surface rather than silently re-executing
    the body.
    """
    attempts = settings.sqlite_write_lock_retry_count + 1
    last_error: OperationalError | None = None

    for attempt in range(attempts):
        connection = write_engine.connect()
        try:
            return connection, connection.begin()
        except OperationalError as error:
            connection.close()
            last_error = error
            if not _is_lock_contention(error):
                raise
            if attempt < attempts - 1:
                time.sleep(0.05 * (attempt + 1))

    raise RunWriteConflict(
        "The database stayed locked across every retry; try again shortly."
    ) from last_error


@contextmanager
def immediate_write():
    """Yield a Session inside a single BEGIN IMMEDIATE transaction.

    Integrity, foreign-key, and uniqueness failures are real errors and are
    never retried.
    """
    connection, transaction = _acquire_write_transaction()
    session = Session(bind=connection, future=True)
    try:
        yield session
        session.flush()
        transaction.commit()
    except Exception:
        transaction.rollback()
        raise
    finally:
        session.close()
        connection.close()
