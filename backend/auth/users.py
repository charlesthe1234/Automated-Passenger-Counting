"""User account operations shared by the admin API and the recovery CLI."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from auth import sessions as session_service
from auth.models import ROLE_ADMIN, ROLE_STAFF, VALID_ROLES, User
from auth.password import (
    CredentialError,
    hash_password,
    normalize_display_name,
    normalize_username,
)


class UserExistsError(ValueError):
    """Raised when a username is already taken."""


class LastAdminError(ValueError):
    """Raised when an action would remove the final active administrator."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_by_username(db: Session, username: str) -> User | None:
    normalized = (username or "").strip().lower()
    if not normalized:
        return None
    return db.execute(select(User).where(User.username == normalized)).scalar_one_or_none()


def list_users(db: Session) -> list[User]:
    return list(db.execute(select(User).order_by(User.username)).scalars())


def count_active_admins(db: Session, *, excluding_user_id: int | None = None) -> int:
    statement = select(func.count(User.id)).where(
        User.role == ROLE_ADMIN, User.is_active.is_(True)
    )
    if excluding_user_id is not None:
        statement = statement.where(User.id != excluding_user_id)
    return int(db.execute(statement).scalar_one())


def create_user(
    db: Session,
    *,
    username: str,
    display_name: str,
    password: str,
    role: str = ROLE_STAFF,
    allow_demo_minimum: bool = False,
) -> User:
    normalized_username = normalize_username(username)
    normalized_display_name = normalize_display_name(display_name)
    if role not in VALID_ROLES:
        raise CredentialError(f"Role must be one of: {', '.join(VALID_ROLES)}.")
    if get_by_username(db, normalized_username) is not None:
        raise UserExistsError(f"An account named '{normalized_username}' already exists.")

    now = utc_now()
    user = User(
        username=normalized_username,
        display_name=normalized_display_name,
        password_hash=hash_password(password, allow_demo_minimum=allow_demo_minimum),
        role=role,
        is_active=True,
        must_change_password=False,
        created_at=now,
        updated_at=now,
    )
    db.add(user)
    db.flush()
    return user


def set_password(db: Session, user: User, password: str) -> None:
    """Replace a password completely and end that user's existing sessions."""
    user.password_hash = hash_password(password)
    user.updated_at = utc_now()
    session_service.revoke_sessions_for_user(db, user.id)
    db.flush()


def set_role(db: Session, user: User, role: str) -> None:
    if role not in VALID_ROLES:
        raise CredentialError(f"Role must be one of: {', '.join(VALID_ROLES)}.")
    if (
        user.role == ROLE_ADMIN
        and role != ROLE_ADMIN
        and user.is_active
        and count_active_admins(db, excluding_user_id=user.id) == 0
    ):
        raise LastAdminError("The final active administrator cannot be demoted.")
    user.role = role
    user.updated_at = utc_now()
    # Existing sessions must not retain the previous role's authority.
    session_service.revoke_sessions_for_user(db, user.id)
    db.flush()


def set_active(db: Session, user: User, is_active: bool) -> None:
    if (
        not is_active
        and user.role == ROLE_ADMIN
        and user.is_active
        and count_active_admins(db, excluding_user_id=user.id) == 0
    ):
        raise LastAdminError("The final active administrator cannot be disabled.")
    user.is_active = is_active
    user.updated_at = utc_now()
    if not is_active:
        session_service.revoke_sessions_for_user(db, user.id)
    db.flush()


def record_login(db: Session, user: User) -> None:
    user.last_login_at = utc_now()
    db.flush()
