"""Route-protection matrix, CV service authentication, and evidence delivery."""

import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from tests import BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))

_TMP = tempfile.mkdtemp(prefix="cag-auth-protect-")
SERVICE_TOKEN = "unit-test-service-token"
os.environ.update(
    {
        "SQLITE_DB_PATH": os.path.join(_TMP, "test.db"),
        "OBSERVATION_UPLOAD_DIR": os.path.join(_TMP, "observations"),
        "EVACUEE_UPLOAD_DIR": os.path.join(_TMP, "evacuees"),
        "CAMERA_URL": "",
        "CAMERA_URLS": "",
        "CV_ENABLED": "false",
        "MQTT_ENABLED": "false",
        "CV_SERVICE_TOKEN": SERVICE_TOKEN,
        "AUTH_ALLOW_INSECURE_HTTP": "true",
        "CORS_ORIGINS": "http://localhost:5173",
    }
)

from pydantic import SecretStr  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from auth import users as user_service  # noqa: E402
from auth.models import AuthEvent, AuthSession, User  # noqa: E402
from auth.rate_limit import login_rate_limiter  # noqa: E402
from config import settings  # noqa: E402
from database import SessionLocal, init_db  # noqa: E402
from models import EvacueeGalleryView, EvacueeIdentity  # noqa: E402

# Upload roots are module-level constants resolved at import time, and test
# modules share one process, so the import order decides whose configuration
# wins. These are redirected explicitly below so a test can never write into
# the operator's real evidence directory.
EVACUEE_UPLOAD_DIR = Path(_TMP) / "evacuees"
OBSERVATION_UPLOAD_DIR = Path(_TMP) / "observations"

ADMIN_PASSWORD = "AdminPassword1"
STAFF_PASSWORD = "StaffPassword1"
SERVICE_HEADER = {"X-CV-Service-Token": SERVICE_TOKEN}


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



def tiny_png() -> bytes:
    """A real 1x1 PNG so magic-byte inspection succeeds."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixel = zlib.compress(b"\x00\xff\xff\xff")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixel) + chunk(b"IEND", b"")


class ProtectionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_isolated_database()
        # Test modules share one process and `settings` is built once, so the
        # values this module depends on are pinned here rather than relying on
        # which module happened to import config first.
        import evacuees.storage
        import evidence.router
        import observation_storage

        EVACUEE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OBSERVATION_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls._pinned = [
            patch.object(settings, "auth_allow_insecure_http", True),
            patch.object(settings, "cors_origins", "http://localhost:5173"),
            patch.object(settings, "cv_service_token", SecretStr(SERVICE_TOKEN)),
            patch.object(settings, "cv_control_legacy_token_enabled", False),
            patch.object(evacuees.storage, "UPLOAD_DIR", EVACUEE_UPLOAD_DIR),
            patch.object(observation_storage, "UPLOAD_DIR", OBSERVATION_UPLOAD_DIR),
            patch.object(evidence.router, "EVACUEE_UPLOAD_DIR", EVACUEE_UPLOAD_DIR),
            patch.object(evidence.router, "OBSERVATION_UPLOAD_DIR", OBSERVATION_UPLOAD_DIR),
        ]
        for pin in cls._pinned:
            pin.start()

        # Fail loudly rather than ever writing evidence into the real tree.
        real_root = Path(__file__).resolve().parent / "uploads"
        for redirected in (evacuees.storage.UPLOAD_DIR, evidence.router.EVACUEE_UPLOAD_DIR):
            assert not str(redirected).startswith(str(real_root)), (
                f"Test upload root {redirected} is inside the real evidence directory."
            )
        init_db()

    @classmethod
    def tearDownClass(cls):
        for pin in cls._pinned:
            pin.stop()

    def setUp(self):
        login_rate_limiter.reset()
        self.client = TestClient(main.app)
        with SessionLocal() as db:
            for model in (EvacueeGalleryView, EvacueeIdentity, AuthEvent, AuthSession, User):
                db.query(model).delete()
            db.commit()
            user_service.create_user(
                db, username="denn", display_name="Denn", password=ADMIN_PASSWORD, role="admin"
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

    def login(self, username="denn", password=ADMIN_PASSWORD) -> str:
        response = self.client.post(
            "/api/auth/login", json={"username": username, "password": password}
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["csrf_token"]

    def seed_gallery_view(self) -> int:
        image_dir = EVACUEE_UPLOAD_DIR / "field_test_001" / "master_0012"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / "front_unittest.png"
        image_path.write_bytes(tiny_png())
        with SessionLocal() as db:
            identity = EvacueeIdentity(run_id="field_test_001", master_identity_id=12)
            db.add(identity)
            db.flush()
            view = EvacueeGalleryView(
                evacuee_id=identity.id,
                view_type="front",
                image_path=str(image_path),
                image_url="/uploads/evacuees/field_test_001/master_0012/front_unittest.png",
            )
            db.add(view)
            db.commit()
            return view.id


class PublicSurfaceTests(ProtectionTestCase):
    def test_health_is_public(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_login_endpoint_is_public(self):
        response = self.client.post(
            "/api/auth/login", json={"username": "denn", "password": "wrong-password"}
        )
        self.assertEqual(response.status_code, 401)

    def test_static_upload_mounts_no_longer_exist(self):
        for path in (
            "/uploads/evacuees/field_test_001/master_0012/front_unittest.png",
            "/uploads/observations/anything.jpg",
        ):
            self.assertEqual(self.client.get(path).status_code, 404, path)


class UnauthenticatedAccessTests(ProtectionTestCase):
    PROTECTED_GETS = [
        "/api/status",
        "/api/cameras",
        "/api/cameras/cam_1/status",
        "/api/stream",
        "/api/cameras/cam_1/stream",
        "/api/metrics",
        "/api/metrics/trends",
        "/api/zones/status",
        "/api/tactical/latest",
        "/api/alerts",
        "/api/observations",
        "/api/observations/summary",
        "/api/reports/shift.csv",
        "/api/reports/shift.xlsx",
        "/api/evacuees",
        "/api/evacuees/summary",
        "/api/evacuees/1",
        "/api/staff?run_id=field_test_001",
        "/api/cv/status",
        "/api/admin/users",
        "/api/evidence/evacuees/1",
        "/api/evidence/observations/1",
    ]

    def test_every_protected_get_returns_401_without_a_session(self):
        for path in self.PROTECTED_GETS:
            self.assertEqual(self.client.get(path).status_code, 401, path)

    def test_unauthenticated_writes_are_rejected(self):
        self.assertEqual(self.client.post("/api/metrics", json={}).status_code, 401)
        self.assertEqual(self.client.post("/api/tactical", json={}).status_code, 401)
        self.assertEqual(self.client.post("/api/alerts", json={}).status_code, 401)
        self.assertEqual(self.client.delete("/api/observations").status_code, 401)
        self.assertEqual(self.client.delete("/api/evacuees").status_code, 401)
        self.assertEqual(self.client.post("/api/cv/session/start", json={}).status_code, 401)
        self.assertEqual(self.client.post("/api/cv/session/stop").status_code, 401)


class AuthenticatedBrowserAccessTests(ProtectionTestCase):
    def test_staff_can_read_operational_data(self):
        self.login("staffer", STAFF_PASSWORD)
        for path in (
            "/api/metrics",
            "/api/alerts",
            "/api/zones/status",
            "/api/evacuees",
            "/api/staff?run_id=field_test_001",
        ):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_reports_are_downloadable_and_not_cacheable(self):
        self.login("staffer", STAFF_PASSWORD)
        for path in ("/api/reports/shift.csv", "/api/reports/shift.xlsx"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertEqual(response.headers["cache-control"], "private, no-store, max-age=0")
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")
            self.assertIn("attachment", response.headers["content-disposition"])


class StaffReviewTests(ProtectionTestCase):
    def test_listing_contains_only_cag_and_scdf_from_selected_run(self):
        with SessionLocal() as db:
            db.add_all(
                [
                    EvacueeIdentity(
                        run_id="field_test_001",
                        master_identity_id=1,
                        role="evacuee",
                        role_confidence=0.99,
                    ),
                    EvacueeIdentity(
                        run_id="field_test_001",
                        master_identity_id=2,
                        role="cag",
                        role_confidence=0.82,
                        last_camera_id="cam_1",
                    ),
                    EvacueeIdentity(
                        run_id="field_test_001",
                        master_identity_id=3,
                        role="scdf",
                        role_confidence=0.91,
                        last_camera_id="cam_2",
                    ),
                    EvacueeIdentity(
                        run_id="older_run",
                        master_identity_id=4,
                        role="cag",
                        role_confidence=0.88,
                    ),
                ]
            )
            db.commit()

        self.login("staffer", STAFF_PASSWORD)
        response = self.client.get("/api/staff?run_id=field_test_001")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual({item["master_identity_id"] for item in payload}, {2, 3})
        self.assertEqual({item["role"] for item in payload}, {"cag", "scdf"})
        confidences = {item["master_identity_id"]: item["role_confidence"] for item in payload}
        self.assertEqual(confidences, {2: 0.82, 3: 0.91})


class LegacyObservationDeletionTests(ProtectionTestCase):
    """D4: the legacy global wipe keeps its behaviour but is admin-only now."""

    def test_unauthenticated_cannot_call_the_global_wipe(self):
        self.assertEqual(self.client.delete("/api/observations").status_code, 401)

    def test_ordinary_staff_cannot_call_the_global_wipe(self):
        csrf = self.login("staffer", STAFF_PASSWORD)
        response = self.client.delete("/api/observations", headers={"X-CSRF-Token": csrf})
        self.assertEqual(response.status_code, 403)

    def test_staff_cannot_bypass_it_with_a_missing_csrf_token(self):
        self.login("staffer", STAFF_PASSWORD)
        self.assertEqual(self.client.delete("/api/observations").status_code, 403)

    def test_admin_with_csrf_can_still_call_it(self):
        csrf = self.login()
        response = self.client.delete("/api/observations", headers={"X-CSRF-Token": csrf})
        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted_rows", response.json())

    def test_admin_without_csrf_is_rejected(self):
        self.login()
        self.assertEqual(self.client.delete("/api/observations").status_code, 403)


class CvServiceAuthTests(ProtectionTestCase):
    def test_valid_service_token_can_read_the_reid_gallery(self):
        response = self.client.get(
            "/api/evacuees/reid-gallery", params={"run_id": "field_test_001"}, headers=SERVICE_HEADER
        )
        self.assertEqual(response.status_code, 200)

    def test_missing_service_token_is_rejected(self):
        self.assertEqual(self.client.get("/api/evacuees/reid-gallery").status_code, 401)

    def test_invalid_service_token_is_rejected(self):
        response = self.client.get(
            "/api/evacuees/reid-gallery", headers={"X-CV-Service-Token": "wrong"}
        )
        self.assertEqual(response.status_code, 401)

    def test_service_token_can_upsert_identity_metadata(self):
        response = self.client.put(
            "/api/evacuees/by-master/field_test_001/12",
            json={"role": "evacuee", "gender": "female", "age": 29.0},
            headers=SERVICE_HEADER,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["master_identity_id"], 12)

    def test_service_token_can_upload_a_gallery_view_end_to_end(self):
        self.client.put(
            "/api/evacuees/by-master/field_test_001/12",
            json={"role": "evacuee"},
            headers=SERVICE_HEADER,
        )
        response = self.client.put(
            "/api/evacuees/by-master/field_test_001/12/views/front",
            files={"image": ("front.png", tiny_png(), "image/png")},
            data={"camera_id": "cam_1", "digest": "abc123"},
            headers=SERVICE_HEADER,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["gallery_filled"], 1)
        # Evidence is advertised through the authenticated route, not /uploads.
        self.assertTrue(payload["primary_view"]["image_url"].startswith("/api/evidence/evacuees/"))

    def test_service_token_cannot_perform_admin_actions(self):
        self.assertEqual(self.client.get("/api/admin/users", headers=SERVICE_HEADER).status_code, 401)
        self.assertEqual(
            self.client.post("/api/cv/session/start", json={}, headers=SERVICE_HEADER).status_code,
            401,
        )
        self.assertEqual(
            self.client.post("/api/cv/session/stop", headers=SERVICE_HEADER).status_code, 401
        )
        self.assertEqual(
            self.client.delete("/api/evacuees", headers=SERVICE_HEADER).status_code, 401
        )
        self.assertEqual(
            self.client.delete("/api/observations", headers=SERVICE_HEADER).status_code, 401
        )

    def test_service_token_cannot_read_browser_data(self):
        self.assertEqual(self.client.get("/api/evacuees", headers=SERVICE_HEADER).status_code, 401)
        self.assertEqual(self.client.get("/api/metrics", headers=SERVICE_HEADER).status_code, 401)

    def test_service_token_may_post_fallback_ingestion(self):
        response = self.client.post(
            "/api/metrics",
            json={"run_id": "field_test_001", "passenger_count": 4},
            headers=SERVICE_HEADER,
        )
        self.assertEqual(response.status_code, 201)

    def test_service_ingestion_does_not_require_csrf(self):
        response = self.client.post(
            "/api/alerts",
            json={"run_id": "field_test_001", "severity": "warning", "message": "test"},
            headers=SERVICE_HEADER,
        )
        self.assertEqual(response.status_code, 201)


class CvControlTests(ProtectionTestCase):
    def test_staff_cannot_control_the_cv_session(self):
        csrf = self.login("staffer", STAFF_PASSWORD)
        response = self.client.post(
            "/api/cv/session/start", json={}, headers={"X-CSRF-Token": csrf}
        )
        self.assertEqual(response.status_code, 403)

    def test_admin_without_csrf_cannot_control_the_cv_session(self):
        self.login()
        self.assertEqual(self.client.post("/api/cv/session/start", json={}).status_code, 403)

    def test_admin_with_csrf_is_authorized_and_no_operator_token_is_requested(self):
        csrf = self.login()
        response = self.client.post(
            "/api/cv/session/start", json={}, headers={"X-CSRF-Token": csrf}
        )
        # CV is disabled in tests, so authorization passes and the manager
        # rejects the transition; the point is that it is not a 401/403.
        self.assertNotIn(response.status_code, (401, 403))

    def test_legacy_operator_token_is_disabled_by_default(self):
        with patch.object(settings, "cv_control_token", settings.cv_control_token.__class__("legacy")):
            response = self.client.post(
                "/api/cv/session/start", json={}, headers={"X-Operator-Token": "legacy"}
            )
        self.assertEqual(response.status_code, 401)

    def test_legacy_operator_token_works_only_when_explicitly_enabled(self):
        token_type = settings.cv_control_token.__class__
        with patch.object(settings, "cv_control_legacy_token_enabled", True), patch.object(
            settings, "cv_control_token", token_type("legacy")
        ):
            response = self.client.post(
                "/api/cv/session/start", json={}, headers={"X-Operator-Token": "legacy"}
            )
        self.assertNotIn(response.status_code, (401, 403))

    def test_cv_status_reports_admin_session_control_mode(self):
        self.login()
        body = self.client.get("/api/cv/status").json()
        self.assertEqual(body["control_mode"], "admin_session")
        self.assertTrue(body["control_allowed"])

    def test_cv_status_does_not_allow_control_for_staff(self):
        self.login("staffer", STAFF_PASSWORD)
        body = self.client.get("/api/cv/status").json()
        self.assertFalse(body["control_allowed"])


class EvidenceRouteTests(ProtectionTestCase):
    def test_evidence_requires_authentication(self):
        view_id = self.seed_gallery_view()
        self.assertEqual(self.client.get(f"/api/evidence/evacuees/{view_id}").status_code, 401)

    def test_authenticated_staff_can_load_evidence_with_no_store_headers(self):
        view_id = self.seed_gallery_view()
        self.login("staffer", STAFF_PASSWORD)
        response = self.client.get(f"/api/evidence/evacuees/{view_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["cache-control"], "private, no-store, max-age=0")
        self.assertEqual(response.headers["pragma"], "no-cache")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.content[:8], b"\x89PNG\r\n\x1a\n")

    def test_evidence_is_denied_again_after_logout(self):
        view_id = self.seed_gallery_view()
        csrf = self.login("staffer", STAFF_PASSWORD)
        self.assertEqual(self.client.get(f"/api/evidence/evacuees/{view_id}").status_code, 200)
        self.client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
        self.assertEqual(self.client.get(f"/api/evidence/evacuees/{view_id}").status_code, 401)

    def test_unknown_view_returns_a_generic_404(self):
        self.login()
        response = self.client.get("/api/evidence/evacuees/999999")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "That evidence image is unavailable.")

    def test_path_outside_the_upload_root_is_refused(self):
        view_id = self.seed_gallery_view()
        outside = Path(_TMP) / "outside_secret.png"
        outside.write_bytes(tiny_png())
        with SessionLocal() as db:
            view = db.get(EvacueeGalleryView, view_id)
            view.image_path = str(outside)
            db.commit()
        self.login()
        response = self.client.get(f"/api/evidence/evacuees/{view_id}")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(str(outside), response.text)

    def test_symlink_escaping_the_upload_root_is_refused(self):
        view_id = self.seed_gallery_view()
        secret = Path(_TMP) / "symlink_secret.png"
        secret.write_bytes(tiny_png())
        link = EVACUEE_UPLOAD_DIR / "field_test_001" / "master_0012" / "escape.png"
        link.symlink_to(secret)
        with SessionLocal() as db:
            view = db.get(EvacueeGalleryView, view_id)
            view.image_path = str(link)
            db.commit()
        self.login()
        self.assertEqual(self.client.get(f"/api/evidence/evacuees/{view_id}").status_code, 404)

    def test_non_image_content_is_refused(self):
        view_id = self.seed_gallery_view()
        with SessionLocal() as db:
            view = db.get(EvacueeGalleryView, view_id)
            disguised = Path(view.image_path)
            disguised.write_bytes(b"#!/bin/sh\necho not-an-image\n")
            db.commit()
        self.login()
        self.assertEqual(self.client.get(f"/api/evidence/evacuees/{view_id}").status_code, 404)

    def test_missing_file_returns_generic_404_without_filesystem_detail(self):
        view_id = self.seed_gallery_view()
        with SessionLocal() as db:
            view = db.get(EvacueeGalleryView, view_id)
            Path(view.image_path).unlink()
            db.commit()
        self.login()
        response = self.client.get(f"/api/evidence/evacuees/{view_id}")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(str(EVACUEE_UPLOAD_DIR), response.text)


class ForeignKeyEnforcementTests(ProtectionTestCase):
    def test_gallery_views_cascade_when_an_identity_is_deleted(self):
        self.seed_gallery_view()
        with SessionLocal() as db:
            self.assertEqual(db.query(EvacueeGalleryView).count(), 1)
            db.query(EvacueeIdentity).delete()
            db.commit()
            # PRAGMA foreign_keys=ON makes the declared ON DELETE CASCADE real.
            self.assertEqual(db.query(EvacueeGalleryView).count(), 0)


if __name__ == "__main__":
    unittest.main()
