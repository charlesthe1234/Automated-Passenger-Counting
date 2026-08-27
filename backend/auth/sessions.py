"""Server-side session issue, validation, and revocation.

Only hashes of the session and CSRF tokens are persisted. The raw session token
exists solely inside the HttpOnly cookie; the raw CSRF token is returned to
React once and held in memory there.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, or_, select, update
from sqlalchemy.orm import Session

from auth.models import AuthEvent, AuthSession, User
from timeutils import as_utc

SESSION_COOKIE_NAME = "cag_session"
CSRF_HEADER_NAME = "X-CSRF-Token"
# 256 bits of entropy, URL-safe.
TOKEN_BYTES = 32
# Avoid writing to SQLite on every single polled request.
LAST_SEEN_REFRESH_SECONDS = 300


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IssuedSession:
    session: AuthSession
    session_token: str
    csrf_token: str


def record_event(
    db: Session,
    *,
    event_type: str,
    user_id: int | None = None,
    client_label: str | None = None,
) -> AuthEvent:
    """Append one audit row. Never receives secrets."""
    event = AuthEvent(
        timestamp=utc_now(),
        user_id=user_id,
        event_type=event_type,
        client_label=(client_label or None),
    )
    db.add(event)
    return event


def create_session(
    db: Session,
    *,
    user: User,
    idle_minutes: int,
    absolute_hours: int,
    client_label: str | None = None,
) -> IssuedSession:
    now = utc_now()
    session_token = generate_token()
    csrf_token = generate_token()
    auth_session = AuthSession(
        token_hash=hash_token(session_token),
        user_id=user.id,
        csrf_token_hash=hash_token(csrf_token),
        created_at=now,
        last_seen_at=now,
        idle_expires_at=now + timedelta(minutes=idle_minutes),
        absolute_expires_at=now + timedelta(hours=absolute_hours),
        client_label=client_label,
    )
    db.add(auth_session)
    db.flush()
    return IssuedSession(session=auth_session, session_token=session_token, csrf_token=csrf_token)


def load_valid_session(db: Session, session_token: str | None) -> AuthSession | None:
    """Return a live session, or None when missing, revoked, or expired."""
    if not session_token:
        return None
    auth_session = db.execute(
        select(AuthSession).where(AuthSession.token_hash == hash_token(session_token))
    ).scalar_one_or_none()
    if auth_session is None or auth_session.revoked_at is not None:
        return None

    now = utc_now()
    if as_utc(auth_session.idle_expires_at) <= now:
        return None
    if as_utc(auth_session.absolute_expires_at) <= now:
        return None
    return auth_session


def touch_session(db: Session, auth_session: AuthSession, *, idle_minutes: int) -> None:
    """Extend the idle window, writing at most once per refresh interval."""
    now = utc_now()
    last_seen = as_utc(auth_session.last_seen_at)
    if last_seen is not None and (now - last_seen).total_seconds() < LAST_SEEN_REFRESH_SECONDS:
        return
    auth_session.last_seen_at = now
    auth_session.idle_expires_at = now + timedelta(minutes=idle_minutes)


def verify_csrf(auth_session: AuthSession, supplied_token: str | None) -> bool:
    if not supplied_token:
        return False
    return hmac.compare_digest(auth_session.csrf_token_hash, hash_token(supplied_token))


def rotate_csrf(db: Session, auth_session: AuthSession) -> str:
    csrf_token = generate_token()
    auth_session.csrf_token_hash = hash_token(csrf_token)
    db.flush()
    return csrf_token


def revoke_session(db: Session, auth_session: AuthSession) -> None:
    if auth_session.revoked_at is None:
        auth_session.revoked_at = utc_now()


def revoke_sessions_for_user(db: Session, user_id: int) -> int:
    """Defence in depth for disable and role changes."""
    result = db.execute(
        update(AuthSession)
        .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=utc_now())
    )
    return int(result.rowcount or 0)


def purge_expired_sessions(db: Session) -> int:
    """Remove rows that can no longer authenticate anyone."""
    now = utc_now()
    result = db.execute(
        delete(AuthSession).where(
            or_(
                AuthSession.absolute_expires_at <= now,
                AuthSession.idle_expires_at <= now,
                AuthSession.revoked_at.is_not(None),
            )
        )
    )
    return int(result.rowcount or 0)
