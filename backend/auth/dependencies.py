"""Reusable authentication and authorization dependencies.

Authorization always re-reads the live `User` row for the request. Role and
active state are never cached inside the session, so disabling or demoting an
operator takes effect on their very next request.
"""

from __future__ import annotations

import hmac
import ipaddress
from dataclasses import dataclass
from typing import Annotated, Callable

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from auth import sessions as session_service
from auth.models import ROLE_ADMIN, ROLE_STAFF, AuthSession, User
from config import settings
from database import get_db

DbSession = Annotated[Session, Depends(get_db)]

# Clears the session cookie alongside a 401 so an expired session does not keep
# being replayed by the browser.
_CLEAR_COOKIE_HEADER = {
    "Set-Cookie": (
        f"{session_service.SESSION_COOKIE_NAME}=; Path=/; Max-Age=0; "
        "HttpOnly; SameSite=Lax"
    )
}


@dataclass(frozen=True)
class AuthContext:
    user: User
    session: AuthSession


# --- Request inspection --------------------------------------------------


def client_host(request: Request) -> str:
    """Best-effort client address.

    Forwarded headers are honoured only when the deployment explicitly declares
    that FastAPI sits behind a trusted proxy, so a LAN client cannot spoof
    loopback by sending its own X-Forwarded-For.
    """
    if settings.auth_trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client is not None else ""


def is_loopback_request(request: Request) -> bool:
    host = client_host(request)
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_secure_request(request: Request) -> bool:
    if settings.auth_trust_proxy_headers:
        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
        if forwarded_proto:
            return forwarded_proto.lower() == "https"
    return request.url.scheme == "https"


def client_label(request: Request) -> str | None:
    """Privacy-safe label stored on sessions and audit events."""
    host = client_host(request)
    return host[:120] if host else None


def same_origin(request: Request) -> str | None:
    host_header = request.headers.get("host")
    if not host_header:
        return None
    scheme = "https" if is_secure_request(request) else request.url.scheme
    return f"{scheme}://{host_header}"


def trusted_origins(request: Request) -> set[str]:
    origins = {origin.rstrip("/") for origin in settings.cors_origin_list}
    current = same_origin(request)
    if current:
        origins.add(current.rstrip("/"))
    return origins


# --- Session resolution --------------------------------------------------


def optional_auth(request: Request, db: DbSession) -> AuthContext | None:
    """Resolve the caller's session, or None when unauthenticated."""
    token = request.cookies.get(session_service.SESSION_COOKIE_NAME)
    auth_session = session_service.load_valid_session(db, token)
    if auth_session is None:
        return None

    user = db.get(User, auth_session.user_id)
    if user is None or not user.is_active:
        # A disabled account must not keep operating through an old cookie.
        session_service.revoke_session(db, auth_session)
        db.commit()
        return None

    session_service.touch_session(
        db, auth_session, idle_minutes=settings.auth_session_idle_minutes
    )
    db.commit()
    return AuthContext(user=user, session=auth_session)


def require_auth(context: Annotated[AuthContext | None, Depends(optional_auth)]) -> AuthContext:
    if context is None:
        raise HTTPException(
            status_code=401,
            detail="Sign in to continue.",
            headers=_CLEAR_COOKIE_HEADER,
        )
    return context


def require_user(context: Annotated[AuthContext, Depends(require_auth)]) -> User:
    return context.user


def require_role(*allowed_roles: str) -> Callable[[AuthContext], User]:
    """Build a dependency enforcing one of the supplied roles."""

    def dependency(context: Annotated[AuthContext, Depends(require_auth)]) -> User:
        if context.user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail="Your account does not have permission for this action.",
            )
        return context.user

    return dependency


require_staff_or_admin = require_role(ROLE_STAFF, ROLE_ADMIN)
require_admin = require_role(ROLE_ADMIN)


# --- CSRF ----------------------------------------------------------------


def require_csrf(
    request: Request,
    context: Annotated[AuthContext, Depends(require_auth)],
) -> AuthContext:
    """Validate the CSRF request token for cookie-authenticated writes."""
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") not in trusted_origins(request):
        raise HTTPException(status_code=403, detail="Request origin is not trusted.")

    supplied = request.headers.get(session_service.CSRF_HEADER_NAME)
    if not session_service.verify_csrf(context.session, supplied):
        raise HTTPException(
            status_code=403,
            detail="Missing or invalid CSRF token. Refresh the page and try again.",
        )
    return context


def require_admin_csrf(
    context: Annotated[AuthContext, Depends(require_csrf)],
) -> User:
    if context.user.role != ROLE_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Your account does not have permission for this action.",
        )
    return context.user


def require_staff_csrf(
    context: Annotated[AuthContext, Depends(require_csrf)],
) -> User:
    if context.user.role not in (ROLE_STAFF, ROLE_ADMIN):
        raise HTTPException(
            status_code=403,
            detail="Your account does not have permission for this action.",
        )
    return context.user


# --- Service authentication ---------------------------------------------


SERVICE_TOKEN_HEADER = "X-CV-Service-Token"


def has_valid_service_token(request: Request) -> bool:
    expected = settings.cv_service_token.get_secret_value()
    supplied = request.headers.get(SERVICE_TOKEN_HEADER)
    if not expected or not supplied:
        return False
    return hmac.compare_digest(expected, supplied)


def require_cv_service(request: Request) -> None:
    """Least-privilege machine authentication for the headless CV worker.

    Deliberately grants no session, no role, and no CSRF exemption elsewhere.
    """
    if has_valid_service_token(request):
        return
    if not settings.cv_service_token.get_secret_value():
        raise HTTPException(
            status_code=503,
            detail="CV service authentication is not configured on this server.",
        )
    raise HTTPException(status_code=401, detail="A valid CV service token is required.")


def require_cv_service_or_admin(
    request: Request,
    context: Annotated[AuthContext | None, Depends(optional_auth)],
) -> None:
    """Ingestion routes reachable by the CV worker or by an admin operator.

    The service token path carries no cookie and therefore no CSRF; the browser
    path is cookie-authenticated and must present a valid CSRF token.
    """
    if has_valid_service_token(request):
        return
    if context is None:
        raise HTTPException(
            status_code=401,
            detail="A valid CV service token or an administrator session is required.",
            headers=_CLEAR_COOKIE_HEADER,
        )
    if context.user.role != ROLE_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Your account does not have permission for this action.",
        )
    supplied = request.headers.get(session_service.CSRF_HEADER_NAME)
    if not session_service.verify_csrf(context.session, supplied):
        raise HTTPException(
            status_code=403,
            detail="Missing or invalid CSRF token. Refresh the page and try again.",
        )


# --- High-risk control ---------------------------------------------------


def require_high_risk_control(
    request: Request,
    context: Annotated[AuthContext | None, Depends(optional_auth)],
) -> User | None:
    """Authorize CV Start/Stop and other high-risk dashboard controls.

    An authenticated admin plus CSRF is sufficient on both loopback and LAN; no
    second shared token is requested from the operator. The legacy operator
    token remains available only when a deployment explicitly re-enables the
    transitional compatibility path for scripts that have not migrated yet.
    """
    if context is not None:
        if context.user.role != ROLE_ADMIN:
            raise HTTPException(
                status_code=403,
                detail="Only an administrator can control the CV session.",
            )
        supplied = request.headers.get(session_service.CSRF_HEADER_NAME)
        if not session_service.verify_csrf(context.session, supplied):
            raise HTTPException(
                status_code=403,
                detail="Missing or invalid CSRF token. Refresh the page and try again.",
            )
        return context.user

    if settings.cv_control_legacy_token_enabled:
        expected = settings.cv_control_token.get_secret_value()
        supplied = request.headers.get("X-Operator-Token")
        if expected and supplied and hmac.compare_digest(expected, supplied):
            return None

    raise HTTPException(
        status_code=401,
        detail="Sign in as an administrator to control the CV session.",
        headers=_CLEAR_COOKIE_HEADER,
    )


# --- Cookie helpers ------------------------------------------------------


def session_cookie_kwargs(request: Request) -> dict:
    return {
        "key": session_service.SESSION_COOKIE_NAME,
        "httponly": True,
        "samesite": "lax",
        "secure": is_secure_request(request),
        "path": "/",
        "max_age": settings.auth_session_absolute_hours * 3600,
    }


CurrentUser = Annotated[User, Depends(require_user)]
CurrentAuth = Annotated[AuthContext, Depends(require_auth)]
AdminUser = Annotated[User, Depends(require_admin)]
AdminCsrfUser = Annotated[User, Depends(require_admin_csrf)]
StaffCsrfUser = Annotated[User, Depends(require_staff_csrf)]
