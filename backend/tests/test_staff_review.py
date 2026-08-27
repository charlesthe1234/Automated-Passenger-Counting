"""Focused tests for the Staff Review read path without the HTTP client shim."""

import unittest
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic_core import PydanticUndefined

from auth.dependencies import require_user
from evacuees.repository import list_staff_identities
from main import app
from models import EvacueeGalleryView, EvacueeIdentity
from staff.router import router


class StaffReviewRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        EvacueeIdentity.__table__.create(self.engine)
        EvacueeGalleryView.__table__.create(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def tearDown(self):
        self.engine.dispose()

    def test_only_selected_run_cag_and_scdf_are_returned_recent_first(self):
        now = datetime.now(timezone.utc)
        with self.Session() as db:
            db.add_all(
                [
                    EvacueeIdentity(
                        run_id="field_test_001",
                        master_identity_id=1,
                        role="evacuee",
                        role_confidence=0.99,
                        last_seen_at=now,
                    ),
                    EvacueeIdentity(
                        run_id="field_test_001",
                        master_identity_id=2,
                        role="cag",
                        role_confidence=0.82,
                        last_seen_at=now - timedelta(seconds=10),
                    ),
                    EvacueeIdentity(
                        run_id="field_test_001",
                        master_identity_id=3,
                        role="scdf",
                        role_confidence=0.91,
                        last_seen_at=now,
                    ),
                    EvacueeIdentity(
                        run_id="older_run",
                        master_identity_id=4,
                        role="cag",
                        role_confidence=0.88,
                        last_seen_at=now,
                    ),
                ]
            )
            db.commit()

            payload = list_staff_identities(db, run_id="field_test_001")

        self.assertEqual([item["master_identity_id"] for item in payload], [3, 2])
        self.assertEqual([item["role"] for item in payload], ["scdf", "cag"])
        self.assertEqual([item["role_confidence"] for item in payload], [0.91, 0.82])

    def test_staff_route_requires_an_authenticated_user(self):
        route = next(route for route in router.routes if getattr(route, "path", None) == "/api/staff")
        dependency_calls = {dependency.call for dependency in route.dependant.dependencies}
        self.assertIn(require_user, dependency_calls)
        run_id = next(field for field in route.dependant.query_params if field.name == "run_id")
        self.assertIs(run_id.default, PydanticUndefined)

    def test_staff_route_is_registered_on_the_application(self):
        self.assertIn("/api/staff", app.openapi()["paths"])


if __name__ == "__main__":
    unittest.main()
