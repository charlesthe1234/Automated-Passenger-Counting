"""Browser authentication and admin account-management endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from auth import sessions as session_service
from auth import users as user_service
from auth.dependencies import (
    AuthContext,
    client_label,
    is_loopback_request,
    is_secure_request,
    require_admin,
    require_admin_csrf,
    require_auth,
    require_csrf,
    session_cookie_kwargs,
)
from auth.models import User
from auth.password import CredentialError
from auth.rate_limit import login_rate_limiter
from auth.schemas import (
    CsrfResponse,
    LoginRequest,
    LoginResponse,
    MeResponse,
    PasswordResetRequest,
    UserAdminRead,
    UserCreateRequest,
    UserRead,
    UserUpdateRequest,
)
from config import settings
from database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["authentication"])
admin_router = APIRouter(prefix="/api/admin", tags=["administration"])

DbSession = Annotated[Session, Depends(get_db)]

GENERIC_LOGIN_ERROR = "Invalid username or password."


def _rate_limit_key(request: Request, username: str) -> str:
    from auth.dependencies import client_host

    return f"{client_host(request)}|{username.strip().lower()}"


def _require_transport_security(request: Request) -> None:
    """Refuse password login over plain HTTP unless explicitly permitted."""
    if is_secure_request(request) or is_loopback_request(request):
        return
    if settings.auth_allow_insecure_http:
        logger.warning(
            "Accepting a password login over unencrypted HTTP because "
            "AUTH_ALLOW_INSECURE_HTTP is enabled. Credentials are not confidential "
            "on this network."
        )
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "Login over plain HTTP is disabled. Enable HTTPS, or set "
            "AUTH_ALLOW_INSECURE_HTTP=true for a controlled demonstration network."
        ),
    )


@router.post("/api/auth/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: DbSession,
) -> LoginResponse:
    _require_transport_security(request)

    label = client_label(request)
    key = _rate_limit_key(request, payload.username)
    retry_after = login_rate_limiter.retry_after_seconds(key)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail="Too many failed sign-in attempts. Try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )

    user = user_service.get_by_username(db, payload.username)
    from auth.password import verify_password

    # Verification runs even for an unknown username so the response does not
    # reveal which accounts exist.
    password_ok = verify_password(user.password_hash if user else "", payload.password)

    if user is None or not password_ok or not user.is_active:
        login_rate_limiter.register_failure(key)
        session_service.record_event(
            db,
            event_type="login_failure",
            user_id=user.id if user is not None else None,
            client_label=label,
        )
        db.commit()
        raise HTTPException(status_code=401, detail=GENERIC_LOGIN_ERROR)

    login_rate_limiter.register_success(key)
    issued = session_service.create_session(
        db,
        user=user,
        idle_minutes=settings.auth_session_idle_minutes,
        absolute_hours=settings.auth_session_absolute_hours,
        client_label=label,
    )
    user_service.record_login(db, user)
    session_service.record_event(
        db, event_type="login_success", user_id=user.id, client_label=label
    )
    db.commit()

    response.set_cookie(value=issued.session_token, **session_cookie_kwargs(request))
    return LoginResponse(user=UserRead.model_validate(user), csrf_token=issued.csrf_token)


@router.get("/api/auth/me", response_model=MeResponse)
def read_me(
    context: Annotated[AuthContext, Depends(require_auth)],
    db: DbSession,
) -> MeResponse:
    # A page refresh loses the in-memory CSRF token, so a fresh one is issued.
    csrf_token = session_service.rotate_csrf(db, context.session)
    db.commit()
    return MeResponse(user=UserRead.model_validate(context.user), csrf_token=csrf_token)


@router.get("/api/auth/csrf", response_model=CsrfResponse)
def read_csrf(
    context: Annotated[AuthContext, Depends(require_auth)],
    db: DbSession,
) -> CsrfResponse:
    csrf_token = session_service.rotate_csrf(db, context.session)
    db.commit()
    return CsrfResponse(csrf_token=csrf_token)


@router.post("/api/auth/logout")
def logout(
    context: Annotated[AuthContext, Depends(require_csrf)],
    request: Request,
    response: Response,
    db: DbSession,
) -> dict[str, bool]:
    session_service.revoke_session(db, context.session)
    session_service.record_event(
        db,
        event_type="logout",
        user_id=context.user.id,
        client_label=client_label(request),
    )
    db.commit()
    response.delete_cookie(
        key=session_service.SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=is_secure_request(request),
    )
    return {"ok": True}


# --- Admin account management -------------------------------------------


def _user_or_404(db: Session, user_id: int) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="That account was not found.")
    return user


@admin_router.get("/users", response_model=list[UserAdminRead])
def list_users(
    _admin: Annotated[User, Depends(require_admin)],
    db: DbSession,
) -> list[User]:
    return user_service.list_users(db)


@admin_router.post("/users", response_model=UserAdminRead, status_code=201)
def create_user(
    payload: UserCreateRequest,
    admin: Annotated[User, Depends(require_admin_csrf)],
    request: Request,
    db: DbSession,
) -> User:
    try:
        user = user_service.create_user(
            db,
            username=payload.username,
            display_name=payload.display_name,
            password=payload.password,
            role=payload.role,
        )
    except user_service.UserExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CredentialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_service.record_event(
        db, event_type="user_created", user_id=admin.id, client_label=client_label(request)
    )
    db.commit()
    db.refresh(user)
    return user


@admin_router.patch("/users/{user_id}", response_model=UserAdminRead)
def update_user(
    user_id: int,
    payload: UserUpdateRequest,
    admin: Annotated[User, Depends(require_admin_csrf)],
    request: Request,
    db: DbSession,
) -> User:
    user = _user_or_404(db, user_id)
    label = client_label(request)

    try:
        if payload.role is not None and payload.role != user.role:
            user_service.set_role(db, user, payload.role)
            session_service.record_event(
                db, event_type="role_changed", user_id=admin.id, client_label=label
            )
        if payload.is_active is not None and payload.is_active != user.is_active:
            user_service.set_active(db, user, payload.is_active)
            session_service.record_event(
                db,
                event_type="user_enabled" if payload.is_active else "user_disabled",
                user_id=admin.id,
                client_label=label,
            )
    except user_service.LastAdminError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CredentialError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.commit()
    db.refresh(user)
    return user


@admin_router.post("/users/{user_id}/reset-password", response_model=UserAdminRead)
def reset_password(
    user_id: int,
    payload: PasswordResetRequest,
    admin: Annotated[User, Depends(require_admin_csrf)],
    request: Request,
    db: DbSession,
) -> User:
    user = _user_or_404(db, user_id)
    try:
        user_service.set_password(db, user, payload.password)
    except CredentialError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_service.record_event(
        db, event_type="password_reset", user_id=admin.id, client_label=client_label(request)
    )
    db.commit()
    db.refresh(user)
    return user
