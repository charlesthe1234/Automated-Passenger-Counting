"""Request and response models for the authentication API.

No schema here exposes a password hash, a session token, or a CSRF hash, and no
schema accepts an authoritative actor name from the client.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from auth.models import ROLE_ADMIN, ROLE_STAFF
from auth.password import (
    DISPLAY_NAME_MAX_LENGTH,
    PASSWORD_MAX_LENGTH,
    USERNAME_MAX_LENGTH,
)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=USERNAME_MAX_LENGTH)
    password: str = Field(min_length=1, max_length=PASSWORD_MAX_LENGTH)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str
    role: str


class UserAdminRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None


class LoginResponse(BaseModel):
    user: UserRead
    csrf_token: str


class MeResponse(BaseModel):
    user: UserRead
    csrf_token: str


class CsrfResponse(BaseModel):
    csrf_token: str


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=USERNAME_MAX_LENGTH)
    display_name: str = Field(min_length=1, max_length=DISPLAY_NAME_MAX_LENGTH)
    password: str = Field(min_length=1, max_length=PASSWORD_MAX_LENGTH)
    role: str = Field(default=ROLE_STAFF, pattern=f"^({ROLE_STAFF}|{ROLE_ADMIN})$")


class UserUpdateRequest(BaseModel):
    role: str | None = Field(default=None, pattern=f"^({ROLE_STAFF}|{ROLE_ADMIN})$")
    is_active: bool | None = None


class PasswordResetRequest(BaseModel):
    password: str = Field(min_length=1, max_length=PASSWORD_MAX_LENGTH)
