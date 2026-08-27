"""Argon2id password hashing and credential validation limits."""

from __future__ import annotations

import re

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# argon2-cffi defaults to Argon2id. There is deliberately no fallback to a
# weaker algorithm: if this import or verification fails, login fails.
_hasher = PasswordHasher()

USERNAME_PATTERN = re.compile(r"^[a-z0-9._-]+$")
USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 50
DISPLAY_NAME_MIN_LENGTH = 1
DISPLAY_NAME_MAX_LENGTH = 100
PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128
# The agreed controlled-demo password is shorter than the deployment minimum.
# Only the explicitly gated demo seed path may use this relaxed floor.
DEMO_PASSWORD_MIN_LENGTH = 8


class CredentialError(ValueError):
    """Raised when a supplied credential fails validation."""


def normalize_username(raw_username: str) -> str:
    username = (raw_username or "").strip().lower()
    if not USERNAME_MIN_LENGTH <= len(username) <= USERNAME_MAX_LENGTH:
        raise CredentialError(
            f"Username must be {USERNAME_MIN_LENGTH} to {USERNAME_MAX_LENGTH} characters."
        )
    if not USERNAME_PATTERN.match(username):
        raise CredentialError(
            "Username may contain only letters, digits, dot, underscore, and hyphen."
        )
    return username


def normalize_display_name(raw_display_name: str) -> str:
    display_name = (raw_display_name or "").strip()
    if not DISPLAY_NAME_MIN_LENGTH <= len(display_name) <= DISPLAY_NAME_MAX_LENGTH:
        raise CredentialError(
            f"Display name must be {DISPLAY_NAME_MIN_LENGTH} to "
            f"{DISPLAY_NAME_MAX_LENGTH} characters."
        )
    return display_name


def validate_password(password: str, *, allow_demo_minimum: bool = False) -> str:
    """Check length limits before any hashing work is performed."""
    if password is None:
        raise CredentialError("A password is required.")
    minimum = DEMO_PASSWORD_MIN_LENGTH if allow_demo_minimum else PASSWORD_MIN_LENGTH
    if len(password) > PASSWORD_MAX_LENGTH:
        raise CredentialError(f"Password must be {PASSWORD_MAX_LENGTH} characters or fewer.")
    if len(password) < minimum:
        raise CredentialError(f"Password must be at least {minimum} characters.")
    return password


def hash_password(password: str, *, allow_demo_minimum: bool = False) -> str:
    validate_password(password, allow_demo_minimum=allow_demo_minimum)
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Verify a candidate password. Never raises for an ordinary mismatch."""
    if not password_hash or password is None:
        return False
    if len(password) > PASSWORD_MAX_LENGTH:
        # Refuse to spend hashing time on oversized input.
        return False
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
