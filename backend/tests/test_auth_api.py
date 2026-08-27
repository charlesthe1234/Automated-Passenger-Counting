"""Login, session lifecycle, CSRF, and admin account-management tests."""

import os
import sys
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

from tests import BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))

_TMP = tempfile.mkdtemp(prefix="cag-auth-api-")
os.environ.update(
    {
        "SQLITE_DB_PATH": os.path.join(_TMP, "test.db"),
        "OBSERVATION_UPLOAD_DIR": os.path.join(_TMP, "observations"),
        "EVACUEE_UPLOAD_DIR": os.path.join(_TMP, "evacuees"),
        "CAMERA_URL": "",
        "CAMERA_URLS": "",
        "CV_ENABLED": "false",
        "MQTT_ENABLED": "false",
        "CV_SERVICE_TOKEN": "unit-test-service-token",
        # TestClient is treated as a non-loopback plain-HTTP client.
        "AUTH_ALLOW_INSECURE_HTTP": "true",
        "CORS_ORIGINS": "http://localhost:5173",
    }
)

from starlette.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from auth import sessions as session_service  # noqa: E402
from auth import users as user_service  # noqa: E402
from auth.models import AuthEvent, AuthSession, User  # noqa: E402
from auth.rate_limit import login_rate_limiter  # noqa: E402
from config import settings  # noqa: E402
from database import SessionLocal, init_db  # noqa: E402

ADMIN_PASSWORD = "AdminPassword1"
STAFF_PASSWORD = "StaffPassword1"


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


class AuthTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_isolated_database()
        # Test modules share one process and `settings` is built once, so the
        # values this module depends on are pinned here rather than relying on
        # which module happened to import config first.
        cls._pinned = [
            patch.object(settings, "auth_allow_insecure_http", True),
            patch.object(settings, "cors_origins", "http://localhost:5173"),
            patch.object(settings, "auth_trust_proxy_headers", False),
        ]
        for pin in cls._pinned:
            pin.start()
        init_db()

    @classmethod
    def tearDownClass(cls):
        for pin in cls._pinned:
            pin.stop()

    def setUp(self):
        login_rate_limiter.reset()
        self.client = TestClient(main.app)
        with SessionLocal() as db:
            for model in (AuthEvent, AuthSession, User):
                db.query(model).delete()
            db.commit()
            user_service.create_user(
                db,
                username="denn",
                display_name="Denn",
                password=ADMIN_PASSWORD,
                role="admin",
            )
            user_service.create_user(
                db,
                username="staffer",
                display_name="Staffer",
                password=STAFF_PASSWORD,
                role="staff",
            )
            db.commit()

    def tearDown(self):
        self.client.close()

    def login(self, username="denn", password=ADMIN_PASSWORD, client=None):
        target = client or self.client
        return target.post(
            "/api/auth/login", json={"username": username, "password": password}
        )


class LoginTests(AuthTestCase):
    def test_correct_password_creates_httponly_session_cookie(self):
        response = self.login()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["user"]["username"], "denn")
        self.assertEqual(body["user"]["role"], "admin")
        self.assertTrue(body["csrf_token"])

        cookie_header = response.headers.get("set-cookie", "")
        self.assertIn("cag_session=", cookie_header)
        self.assertIn("HttpOnly", cookie_header)
        self.assertIn("SameSite=lax", cookie_header.replace("SameSite=Lax", "SameSite=lax"))

    def test_login_response_never_exposes_password_hash(self):
        body = self.login().json()
        self.assertNotIn("password_hash", str(body))

    def test_raw_session_token_is_not_stored_in_sqlite(self):
        self.login()
        raw_token = self.client.cookies.get("cag_session")
        self.assertTrue(raw_token)
        with SessionLocal() as db:
            stored = db.query(AuthSession).one()
            self.assertNotEqual(stored.token_hash, raw_token)
            self.assertEqual(stored.token_hash, session_service.hash_token(raw_token))

    def test_wrong_password_is_rejected_with_generic_error(self):
        response = self.login(password="not-the-password")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid username or password.")

    def test_unknown_username_gives_the_same_generic_error(self):
        response = self.login(username="ghost", password="whatever123")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid username or password.")

    def test_disabled_user_cannot_log_in(self):
        with SessionLocal() as db:
            user = user_service.get_by_username(db, "staffer")
            user_service.set_active(db, user, False)
            db.commit()
        self.assertEqual(self.login("staffer", STAFF_PASSWORD).status_code, 401)

    def test_rate_limiter_blocks_repeated_failures(self):
        for _ in range(5):
            self.assertEqual(self.login(password="wrong-password").status_code, 401)
        blocked = self.login(password="wrong-password")
        self.assertEqual(blocked.status_code, 429)
        # A correct password is still blocked while the penalty is active.
        self.assertEqual(self.login().status_code, 429)

    def test_plain_http_login_is_rejected_when_insecure_http_is_disabled(self):
        with patch.object(settings, "auth_allow_insecure_http", False):
            response = self.login()
        self.assertEqual(response.status_code, 400)
        self.assertIn("AUTH_ALLOW_INSECURE_HTTP", response.json()["detail"])

    def test_forwarded_headers_cannot_fake_loopback_or_https(self):
        with patch.object(settings, "auth_allow_insecure_http", False):
            response = self.client.post(
                "/api/auth/login",
                json={"username": "denn", "password": ADMIN_PASSWORD},
                headers={"X-Forwarded-For": "127.0.0.1", "X-Forwarded-Proto": "https"},
            )
        self.assertEqual(response.status_code, 400)

    def test_successful_and_failed_logins_are_audited(self):
        self.login(password="wrong-password")
        self.login()
        with SessionLocal() as db:
            kinds = [event.event_type for event in db.query(AuthEvent).all()]
        self.assertIn("login_failure", kinds)
        self.assertIn("login_success", kinds)


class SessionLifecycleTests(AuthTestCase):
    def test_me_requires_authentication(self):
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_session_survives_repeated_requests(self):
        self.login()
        for _ in range(3):
            self.assertEqual(self.client.get("/api/auth/me").status_code, 200)

    def test_logout_revokes_the_session_and_replay_fails(self):
        csrf = self.login().json()["csrf_token"]
        raw_cookie = self.client.cookies.get("cag_session")

        logout = self.client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

        replay = TestClient(main.app, cookies={"cag_session": raw_cookie})
        self.assertEqual(replay.get("/api/auth/me").status_code, 401)
        replay.close()

    def test_idle_expiry_ends_the_session(self):
        self.login()
        with SessionLocal() as db:
            auth_session = db.query(AuthSession).one()
            auth_session.idle_expires_at = session_service.utc_now() - timedelta(seconds=1)
            db.commit()
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_absolute_expiry_ends_the_session(self):
        self.login()
        with SessionLocal() as db:
            auth_session = db.query(AuthSession).one()
            auth_session.absolute_expires_at = session_service.utc_now() - timedelta(seconds=1)
            db.commit()
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_disabling_a_user_revokes_live_sessions_immediately(self):
        self.login("staffer", STAFF_PASSWORD)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
        with SessionLocal() as db:
            user = user_service.get_by_username(db, "staffer")
            user_service.set_active(db, user, False)
            db.commit()
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_demoting_an_admin_removes_admin_authority_from_existing_session(self):
        with SessionLocal() as db:
            user_service.create_user(
                db,
                username="second",
                display_name="Second Admin",
                password=ADMIN_PASSWORD,
                role="admin",
            )
            db.commit()
        csrf = self.login("second", ADMIN_PASSWORD).json()["csrf_token"]
        self.assertEqual(self.client.get("/api/admin/users").status_code, 200)

        with SessionLocal() as db:
            user = user_service.get_by_username(db, "second")
            user_service.set_role(db, user, "staff")
            db.commit()

        # The role change also revokes sessions, so the old cookie is dead.
        response = self.client.get("/api/admin/users")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            self.client.post("/api/admin/users", json={}, headers={"X-CSRF-Token": csrf}).status_code,
            401,
        )

    def test_expired_session_response_clears_the_cookie(self):
        self.login()
        with SessionLocal() as db:
            auth_session = db.query(AuthSession).one()
            auth_session.revoked_at = session_service.utc_now()
            db.commit()
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 401)
        self.assertIn("cag_session=", response.headers.get("set-cookie", ""))


class CsrfTests(AuthTestCase):
    def test_unsafe_request_without_csrf_is_rejected(self):
        self.login()
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 403)

    def test_unsafe_request_with_wrong_csrf_is_rejected(self):
        self.login()
        response = self.client.post("/api/auth/logout", headers={"X-CSRF-Token": "nope"})
        self.assertEqual(response.status_code, 403)

    def test_valid_csrf_succeeds(self):
        csrf = self.login().json()["csrf_token"]
        self.assertEqual(
            self.client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf}).status_code, 200
        )

    def test_untrusted_origin_is_rejected(self):
        csrf = self.login().json()["csrf_token"]
        response = self.client.post(
            "/api/auth/logout",
            headers={"X-CSRF-Token": csrf, "Origin": "http://evil.example"},
        )
        self.assertEqual(response.status_code, 403)

    def test_configured_dev_origin_is_accepted(self):
        csrf = self.login().json()["csrf_token"]
        response = self.client.post(
            "/api/auth/logout",
            headers={"X-CSRF-Token": csrf, "Origin": "http://localhost:5173"},
        )
        self.assertEqual(response.status_code, 200)

    def test_csrf_endpoint_issues_a_usable_token(self):
        self.login()
        csrf = self.client.get("/api/auth/csrf").json()["csrf_token"]
        self.assertEqual(
            self.client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf}).status_code, 200
        )


class AdminUserManagementTests(AuthTestCase):
    def admin_headers(self):
        csrf = self.login().json()["csrf_token"]
        return {"X-CSRF-Token": csrf}

    def test_staff_cannot_reach_admin_routes(self):
        self.login("staffer", STAFF_PASSWORD)
        self.assertEqual(self.client.get("/api/admin/users").status_code, 403)

    def test_unauthenticated_cannot_reach_admin_routes(self):
        self.assertEqual(self.client.get("/api/admin/users").status_code, 401)

    def test_admin_can_list_users_without_password_data(self):
        self.login()
        response = self.client.get("/api/admin/users")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("password", response.text.lower())

    def test_admin_can_create_a_staff_account_that_can_log_in(self):
        headers = self.admin_headers()
        response = self.client.post(
            "/api/admin/users",
            json={
                "username": "newstaff",
                "display_name": "New Staff",
                "password": "BrandNewPass1",
                "role": "staff",
            },
            headers=headers,
        )
        self.assertEqual(response.status_code, 201)
        fresh = TestClient(main.app)
        self.assertEqual(self.login("newstaff", "BrandNewPass1", client=fresh).status_code, 200)
        fresh.close()

    def test_creating_a_user_requires_csrf(self):
        self.login()
        response = self.client.post(
            "/api/admin/users",
            json={
                "username": "nocsrf",
                "display_name": "No CSRF",
                "password": "BrandNewPass1",
                "role": "staff",
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_duplicate_username_is_rejected(self):
        headers = self.admin_headers()
        response = self.client.post(
            "/api/admin/users",
            json={
                "username": "staffer",
                "display_name": "Duplicate",
                "password": "BrandNewPass1",
                "role": "staff",
            },
            headers=headers,
        )
        self.assertEqual(response.status_code, 409)

    def test_short_password_is_rejected(self):
        headers = self.admin_headers()
        response = self.client.post(
            "/api/admin/users",
            json={
                "username": "shorty",
                "display_name": "Shorty",
                "password": "P@ssword1",
                "role": "staff",
            },
            headers=headers,
        )
        self.assertEqual(response.status_code, 400)

    def test_final_active_admin_cannot_be_disabled(self):
        headers = self.admin_headers()
        with SessionLocal() as db:
            admin_id = user_service.get_by_username(db, "denn").id
        response = self.client.patch(
            f"/api/admin/users/{admin_id}", json={"is_active": False}, headers=headers
        )
        self.assertEqual(response.status_code, 409)

    def test_final_active_admin_cannot_be_demoted(self):
        headers = self.admin_headers()
        with SessionLocal() as db:
            admin_id = user_service.get_by_username(db, "denn").id
        response = self.client.patch(
            f"/api/admin/users/{admin_id}", json={"role": "staff"}, headers=headers
        )
        self.assertEqual(response.status_code, 409)

    def test_admin_can_reset_a_password_and_old_password_stops_working(self):
        headers = self.admin_headers()
        with SessionLocal() as db:
            staff_id = user_service.get_by_username(db, "staffer").id
        response = self.client.post(
            f"/api/admin/users/{staff_id}/reset-password",
            json={"password": "ReplacementPass1"},
            headers=headers,
        )
        self.assertEqual(response.status_code, 200)

        fresh = TestClient(main.app)
        self.assertEqual(self.login("staffer", STAFF_PASSWORD, client=fresh).status_code, 401)
        login_rate_limiter.reset()
        self.assertEqual(self.login("staffer", "ReplacementPass1", client=fresh).status_code, 200)
        fresh.close()

    def test_admin_actions_record_the_acting_admin(self):
        headers = self.admin_headers()
        self.client.post(
            "/api/admin/users",
            json={
                "username": "audited",
                "display_name": "Audited",
                "password": "BrandNewPass1",
                "role": "staff",
            },
            headers=headers,
        )
        with SessionLocal() as db:
            admin_id = user_service.get_by_username(db, "denn").id
            event = (
                db.query(AuthEvent)
                .filter(AuthEvent.event_type == "user_created")
                .one()
            )
            self.assertEqual(event.user_id, admin_id)


if __name__ == "__main__":
    unittest.main()
