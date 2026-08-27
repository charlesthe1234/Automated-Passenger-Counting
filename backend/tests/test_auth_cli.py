"""Bootstrap and recovery CLI tests, including the gated demo seed."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from tests import BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))

_TMP = tempfile.mkdtemp(prefix="cag-auth-cli-")
os.environ.update(
    {
        "SQLITE_DB_PATH": os.path.join(_TMP, "test.db"),
        "OBSERVATION_UPLOAD_DIR": os.path.join(_TMP, "observations"),
        "EVACUEE_UPLOAD_DIR": os.path.join(_TMP, "evacuees"),
        "CAMERA_URL": "",
        "CAMERA_URLS": "",
        "CV_ENABLED": "false",
        "MQTT_ENABLED": "false",
    }
)

from auth import cli  # noqa: E402
from auth import users as user_service  # noqa: E402
from auth.models import AuthEvent, AuthSession, User  # noqa: E402
from auth.password import verify_password  # noqa: E402
from config import settings  # noqa: E402
from database import SessionLocal, init_db  # noqa: E402

DEMO_PASSWORD = "P@ssword1"


def assert_isolated_database():
    """Refuse to run if the resolved database is the operator's real one.

    The SQLite path is resolved once at import time and test modules share a
    process, so a bad import order could otherwise point a test at a live
    deployment. Failing loudly beats migrating or writing rows into it.
    """
    import database

    resolved = str(database.DATABASE_PATH)
    if not resolved.startswith(tempfile.gettempdir()):
        raise AssertionError(
            f"Refusing to run: tests resolved to the real database at {resolved}. "
            "Set SQLITE_DB_PATH to a temporary path before importing backend modules."
        )



def run_cli(argv, password=None):
    """Invoke the CLI, feeding hidden prompts and capturing output."""
    out, err = io.StringIO(), io.StringIO()
    prompts = [password, password] if password is not None else []
    with patch("auth.cli.getpass.getpass", side_effect=prompts or Exception("unexpected prompt")):
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
    return code, out.getvalue() + err.getvalue()


class CliTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_isolated_database()
        init_db()

    def setUp(self):
        with SessionLocal() as db:
            for model in (AuthEvent, AuthSession, User):
                db.query(model).delete()
            db.commit()


class CreateUserTests(CliTestCase):
    def test_create_user_stores_only_an_argon2id_hash(self):
        code, _ = run_cli(["create-user", "alice", "--role", "admin"], password="AlicesPassword1")
        self.assertEqual(code, 0)
        with SessionLocal() as db:
            user = user_service.get_by_username(db, "alice")
            self.assertEqual(user.role, "admin")
            self.assertTrue(user.password_hash.startswith("$argon2id$"))
            self.assertNotIn("AlicesPassword1", user.password_hash)
            self.assertTrue(verify_password(user.password_hash, "AlicesPassword1"))

    def test_password_is_never_echoed_in_output(self):
        _, output = run_cli(["create-user", "bob"], password="BobsPassword1")
        self.assertNotIn("BobsPassword1", output)

    def test_mismatched_confirmation_is_rejected(self):
        out, err = io.StringIO(), io.StringIO()
        with patch("auth.cli.getpass.getpass", side_effect=["Password1234", "Different1234"]):
            with redirect_stdout(out), redirect_stderr(err):
                with self.assertRaises(SystemExit):
                    cli.main(["create-user", "carol"])
        with SessionLocal() as db:
            self.assertIsNone(user_service.get_by_username(db, "carol"))

    def test_duplicate_username_fails_without_overwriting(self):
        run_cli(["create-user", "dave"], password="DavesPassword1")
        code, _ = run_cli(["create-user", "dave"], password="DifferentPass1")
        self.assertEqual(code, 1)
        with SessionLocal() as db:
            user = user_service.get_by_username(db, "dave")
            self.assertTrue(verify_password(user.password_hash, "DavesPassword1"))

    def test_short_password_is_rejected_for_normal_accounts(self):
        code, _ = run_cli(["create-user", "erin"], password=DEMO_PASSWORD)
        self.assertEqual(code, 1)
        with SessionLocal() as db:
            self.assertIsNone(user_service.get_by_username(db, "erin"))


class ListAndRecoveryTests(CliTestCase):
    def test_list_users_never_prints_a_password_hash(self):
        run_cli(["create-user", "frank"], password="FranksPassword1")
        code, output = run_cli(["list-users"])
        self.assertEqual(code, 0)
        self.assertIn("frank", output)
        self.assertNotIn("argon2", output)

    def test_reset_password_replaces_the_credential(self):
        run_cli(["create-user", "grace"], password="GracesPassword1")
        code, _ = run_cli(["reset-password", "grace"], password="NewGracePass1")
        self.assertEqual(code, 0)
        with SessionLocal() as db:
            user = user_service.get_by_username(db, "grace")
            self.assertFalse(verify_password(user.password_hash, "GracesPassword1"))
            self.assertTrue(verify_password(user.password_hash, "NewGracePass1"))

    def test_disable_and_enable_round_trip(self):
        run_cli(["create-user", "heidi", "--role", "admin"], password="HeidisPassword1")
        run_cli(["create-user", "ivan", "--role", "admin"], password="IvansPassword1")
        self.assertEqual(run_cli(["disable-user", "ivan"])[0], 0)
        with SessionLocal() as db:
            self.assertFalse(user_service.get_by_username(db, "ivan").is_active)
        self.assertEqual(run_cli(["enable-user", "ivan"])[0], 0)
        with SessionLocal() as db:
            self.assertTrue(user_service.get_by_username(db, "ivan").is_active)

    def test_final_admin_cannot_be_disabled_from_the_cli(self):
        run_cli(["create-user", "judy", "--role", "admin"], password="JudysPassword1")
        code, output = run_cli(["disable-user", "judy"])
        self.assertEqual(code, 1)
        self.assertIn("final active administrator", output)


class DemoSeedGateTests(CliTestCase):
    def seed(self, password=DEMO_PASSWORD, confirm=True):
        argv = ["seed-demo-users"]
        if confirm:
            argv.append("--confirm-insecure-demo")
        return run_cli(argv, password=password if confirm else None)

    def test_refuses_without_demo_app_env(self):
        with patch.object(settings, "app_env", "production"), patch.object(
            settings, "allow_demo_account_seeding", True
        ):
            code, output = run_cli(["seed-demo-users", "--confirm-insecure-demo"])
        self.assertEqual(code, 1)
        self.assertIn("APP_ENV=demo", output)

    def test_refuses_without_the_seeding_flag(self):
        with patch.object(settings, "app_env", "demo"), patch.object(
            settings, "allow_demo_account_seeding", False
        ):
            code, output = run_cli(["seed-demo-users", "--confirm-insecure-demo"])
        self.assertEqual(code, 1)
        self.assertIn("ALLOW_DEMO_ACCOUNT_SEEDING=true", output)

    def test_refuses_without_the_confirmation_flag(self):
        with patch.object(settings, "app_env", "demo"), patch.object(
            settings, "allow_demo_account_seeding", True
        ):
            code, output = run_cli(["seed-demo-users"])
        self.assertEqual(code, 1)
        self.assertIn("--confirm-insecure-demo", output)

    def test_seeds_only_denn_as_admin_when_all_gates_pass(self):
        with patch.object(settings, "app_env", "demo"), patch.object(
            settings, "allow_demo_account_seeding", True
        ):
            code, output = self.seed()
        self.assertEqual(code, 0)
        self.assertNotIn(DEMO_PASSWORD, output)
        with SessionLocal() as db:
            accounts = user_service.list_users(db)
            self.assertEqual([user.username for user in accounts], ["denn"])
            self.assertEqual(accounts[0].role, "admin")
            self.assertTrue(verify_password(accounts[0].password_hash, DEMO_PASSWORD))
            self.assertTrue(accounts[0].password_hash.startswith("$argon2id$"))

    def test_rerun_never_overwrites_an_existing_account_password(self):
        with patch.object(settings, "app_env", "demo"), patch.object(
            settings, "allow_demo_account_seeding", True
        ):
            self.seed()
            code, output = self.seed(password="SomethingElse1")
        self.assertEqual(code, 0)
        self.assertIn("already exists", output)
        with SessionLocal() as db:
            user = user_service.get_by_username(db, "denn")
            # The original demo password still works; the rerun changed nothing.
            self.assertTrue(verify_password(user.password_hash, DEMO_PASSWORD))


if __name__ == "__main__":
    unittest.main()
