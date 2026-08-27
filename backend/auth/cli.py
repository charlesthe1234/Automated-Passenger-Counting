"""Account bootstrap and recovery CLI.

Run from inside the backend directory so the flat module layout resolves:

    .venv-linux/bin/python -m auth.cli list-users

Every password is entered through a hidden interactive prompt. Passwords are
never accepted as command-line arguments, never echoed, and never logged.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from auth import users as user_service
from auth.models import ROLE_ADMIN, ROLE_STAFF, User
from auth.password import CredentialError
from config import settings
from database import SessionLocal, current_revision

DEMO_USERNAME = "denn"
DEMO_DISPLAY_NAME = "Denn"


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def _require_schema_ready() -> None:
    """Refuse to run against a database that has not been migrated yet."""
    from alembic.script import ScriptDirectory

    from database import _alembic_config

    head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
    revision = current_revision()
    if revision == head:
        return
    raise SystemExit(
        "The database schema is not at the expected revision "
        f"(found {revision or 'none'}, expected {head}).\n"
        "Back up the database and uploads, then run from the backend directory:\n"
        "    .venv-linux/bin/python -m alembic upgrade head"
    )


def _prompt_password(prompt: str = "Password: ", *, confirm: bool = True) -> str:
    password = getpass.getpass(prompt)
    if not password:
        raise SystemExit("No password was entered.")
    if confirm:
        again = getpass.getpass("Confirm password: ")
        if password != again:
            raise SystemExit("The passwords did not match.")
    return password


def _print_user(user: User) -> None:
    state = "active" if user.is_active else "disabled"
    last_login = user.last_login_at.isoformat() if user.last_login_at else "never"
    print(f"  {user.username:20} {user.display_name:24} {user.role:6} {state:9} last login: {last_login}")


def command_create_user(args: argparse.Namespace) -> int:
    _require_schema_ready()
    password = _prompt_password()
    with SessionLocal() as db:
        try:
            user = user_service.create_user(
                db,
                username=args.username,
                display_name=args.display_name or args.username,
                password=password,
                role=args.role,
            )
        except (user_service.UserExistsError, CredentialError) as exc:
            return _fail(str(exc))
        db.commit()
        print(f"Created {user.role} account '{user.username}'.")
    return 0


def command_list_users(_args: argparse.Namespace) -> int:
    _require_schema_ready()
    with SessionLocal() as db:
        accounts = user_service.list_users(db)
        if not accounts:
            print("No accounts exist yet. Create the first administrator with create-user.")
            return 0
        print(f"{len(accounts)} account(s):")
        for user in accounts:
            _print_user(user)
    return 0


def command_reset_password(args: argparse.Namespace) -> int:
    _require_schema_ready()
    with SessionLocal() as db:
        user = user_service.get_by_username(db, args.username)
        if user is None:
            return _fail(f"No account named '{args.username}' exists.")
        password = _prompt_password()
        try:
            user_service.set_password(db, user, password)
        except CredentialError as exc:
            return _fail(str(exc))
        db.commit()
        print(f"Password replaced for '{user.username}'. Active sessions were revoked.")
    return 0


def _set_active(username: str, is_active: bool) -> int:
    _require_schema_ready()
    with SessionLocal() as db:
        user = user_service.get_by_username(db, username)
        if user is None:
            return _fail(f"No account named '{username}' exists.")
        try:
            user_service.set_active(db, user, is_active)
        except user_service.LastAdminError as exc:
            return _fail(str(exc))
        db.commit()
        print(f"Account '{user.username}' is now {'active' if is_active else 'disabled'}.")
    return 0


def command_enable_user(args: argparse.Namespace) -> int:
    return _set_active(args.username, True)


def command_disable_user(args: argparse.Namespace) -> int:
    return _set_active(args.username, False)


def command_seed_demo_users(args: argparse.Namespace) -> int:
    """Create the single agreed demonstration administrator.

    Refuses to run unless every demo gate is present. The password is typed
    twice at a hidden prompt and stored only as an Argon2id hash.
    """
    if not settings.is_demo_env:
        return _fail("Demo seeding requires APP_ENV=demo.")
    if not settings.allow_demo_account_seeding:
        return _fail("Demo seeding requires ALLOW_DEMO_ACCOUNT_SEEDING=true.")
    if not args.confirm_insecure_demo:
        return _fail("Demo seeding requires the --confirm-insecure-demo flag.")

    _require_schema_ready()
    print(
        "This creates the agreed demonstration administrator with a known weak\n"
        "password. It is unsuitable for deployment. Enter the demo password twice."
    )
    password = _prompt_password()

    with SessionLocal() as db:
        existing = user_service.get_by_username(db, DEMO_USERNAME)
        if existing is not None:
            # Never silently overwrite an existing account or reset its password.
            print(
                f"Account '{DEMO_USERNAME}' already exists; nothing was changed. "
                "Use reset-password if you intend to replace its password."
            )
            return 0
        try:
            user = user_service.create_user(
                db,
                username=DEMO_USERNAME,
                display_name=DEMO_DISPLAY_NAME,
                password=password,
                role=ROLE_ADMIN,
                allow_demo_minimum=True,
            )
        except (user_service.UserExistsError, CredentialError) as exc:
            return _fail(str(exc))
        db.commit()
        print(
            f"Created demonstration administrator '{user.username}'.\n"
            "Create individual named staff accounts through Admin Settings so that\n"
            "operator actions remain attributable."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auth.cli",
        description="CAG dashboard account bootstrap and recovery.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-user", help="Create one account interactively.")
    create.add_argument("username")
    create.add_argument("--display-name", default=None)
    create.add_argument("--role", choices=[ROLE_STAFF, ROLE_ADMIN], default=ROLE_STAFF)
    create.set_defaults(handler=command_create_user)

    listing = subparsers.add_parser("list-users", help="List accounts without password data.")
    listing.set_defaults(handler=command_list_users)

    reset = subparsers.add_parser("reset-password", help="Replace an account password.")
    reset.add_argument("username")
    reset.set_defaults(handler=command_reset_password)

    enable = subparsers.add_parser("enable-user", help="Re-enable a disabled account.")
    enable.add_argument("username")
    enable.set_defaults(handler=command_enable_user)

    disable = subparsers.add_parser("disable-user", help="Disable an account.")
    disable.add_argument("username")
    disable.set_defaults(handler=command_disable_user)

    demo = subparsers.add_parser(
        "seed-demo-users",
        help="Create the gated demonstration administrator.",
    )
    demo.add_argument("--confirm-insecure-demo", action="store_true")
    demo.set_defaults(handler=command_seed_demo_users)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
