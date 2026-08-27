"""Datetime helpers shared across backend subsystems."""

from __future__ import annotations

from datetime import datetime, timezone


def as_utc(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; treat stored values as UTC.

    Every column in this application stores UTC, but the SQLite driver hands
    back tz-naive objects. Comparing one of those against an aware `now()`
    raises, and passing one to an API response silently claims local time. Both
    auth session expiry and run lifecycle timing depend on this, so it lives in
    one place rather than being reimplemented per subsystem.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
