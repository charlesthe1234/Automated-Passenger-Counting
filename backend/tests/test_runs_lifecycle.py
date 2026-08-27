"""Run lifecycle, reconciliation, backfill, and external-run tests."""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tests import BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))

_TMP = tempfile.mkdtemp(prefix="cag-runs-life-")
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

import models  # noqa: E402
from config import settings  # noqa: E402
from database import SessionLocal, init_db  # noqa: E402
from auth import users as user_service  # noqa: E402
from auth.models import User  # noqa: E402
from runs import repository, service  # noqa: E402
from runs.models import (  # noqa: E402
    DeletedRun,
    PendingFileDeletion,
    Run,
    RunEvent,
)
from runs.write_guard import immediate_write  # noqa: E402


def assert_isolated_database():
    import database

    resolved = str(database.DATABASE_PATH)
    if not resolved.startswith(tempfile.gettempdir()):
        raise AssertionError(f"Refusing to run against the real database at {resolved}.")


class FakeCv:
    """Stands in for the CV worker so lifecycle transitions are deterministic."""

    def __init__(self, state="ready", run_id=None):
        self.state = state
        self.run_id = run_id
        self.start_calls = []
        self.stop_calls = 0
        self.raise_on_start = None
        self.raise_on_stop = None

    def status(self):
        return {"state": self.state, "run_id": self.run_id, "ready": self.state == "ready"}

    def start_session(self, run_id=None):
        if self.raise_on_start:
            raise self.raise_on_start
        self.start_calls.append(run_id)
        self.state = "running"
        self.run_id = run_id

    def stop_session(self):
        if self.raise_on_stop:
            raise self.raise_on_stop
        self.stop_calls += 1
        self.state = "ready"


class RunTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_isolated_database()
        init_db()

    def setUp(self):
        self.cv = FakeCv()
        self.cv_patch = patch.object(service, "cv_manager", self.cv)
        self.cv_patch.start()
        with SessionLocal() as db:
            for model in (
                PendingFileDeletion, DeletedRun, RunEvent, Run,
                models.EvacueeGalleryView, models.EvacueeIdentity,
                models.PassengerObservation, models.MetricLog, models.SystemAlert,
                User,
            ):
                db.query(model).delete()
            db.commit()
            # Run attribution is a real foreign key, so the actor must exist.
            self.admin_id = user_service.create_user(
                db, username="denn", display_name="Denn",
                password="AdminPassword1", role="admin",
            ).id
            self.other_admin_id = user_service.create_user(
                db, username="second", display_name="Second",
                password="AdminPassword1", role="admin",
            ).id
            db.commit()

    def tearDown(self):
        self.cv_patch.stop()

    def start(self, **kwargs):
        return service.start_run(actor_user_id=self.admin_id, **kwargs)

    def reconcile(self):
        with immediate_write() as db:
            service.reconcile(db)

    def get(self, run_id):
        with SessionLocal() as db:
            run = repository.get_by_run_id(db, run_id)
            return None if run is None else service.serialize_run(run)


class StartAndEndTests(RunTestCase):
    def test_generated_run_id_is_unique_and_well_formed(self):
        run = self.start()
        self.assertTrue(repository.is_valid_run_id(run["run_id"]))
        self.assertTrue(run["run_id"].startswith("run_"))
        self.assertEqual(run["status"], "starting")
        self.assertEqual(run["origin_type"], "managed")

    def test_operator_supplied_run_id_is_accepted(self):
        run = self.start(run_id="morning_drill", name="Morning drill")
        self.assertEqual(run["run_id"], "morning_drill")
        self.assertEqual(run["name"], "Morning drill")

    def test_invalid_run_id_is_rejected(self):
        with self.assertRaises(service.RunValidationError):
            self.start(run_id="bad id with spaces")

    def test_duplicate_run_id_is_rejected(self):
        self.start(run_id="dup")
        self.cv.state = "ready"
        with self.assertRaises(service.RunConflictError):
            self.start(run_id="dup")

    def test_second_in_progress_run_is_rejected(self):
        self.start(run_id="first")
        self.cv.state = "ready"  # pretend CV is free even though a run is open
        with self.assertRaises(service.RunConflictError):
            self.start(run_id="second")

    def test_start_requires_a_ready_worker(self):
        self.cv.state = "loading"
        with self.assertRaises(service.RunConflictError):
            self.start()

    def test_started_at_stays_null_until_cv_confirms(self):
        run = self.start(run_id="pending")
        self.assertIsNone(run["started_at"])
        self.assertIsNotNone(run["requested_at"])

        self.reconcile()
        activated = self.get("pending")
        self.assertEqual(activated["status"], "active")
        self.assertIsNotNone(activated["started_at"])

    def test_run_is_never_activated_for_a_different_cv_run(self):
        self.start(run_id="mine")
        self.cv.run_id = "someone_else"
        self.reconcile()
        self.assertEqual(self.get("mine")["status"], "starting")

    def test_immediate_cv_start_failure_marks_the_run_failed(self):
        self.cv.raise_on_start = RuntimeError("worker refused")
        with self.assertRaises(service.RunConflictError):
            self.start(run_id="doomed")
        failed = self.get("doomed")
        self.assertEqual(failed["status"], "failed")
        self.assertIsNotNone(failed["ended_at"])
        self.assertIn("worker refused", failed["failure_reason"])

    def test_end_moves_through_ending_to_ended(self):
        self.start(run_id="closeme")
        self.reconcile()
        ending = service.end_run(run_id="closeme", actor_user_id=self.admin_id)
        self.assertEqual(ending["status"], "ending")
        self.reconcile()
        self.assertEqual(self.get("closeme")["status"], "ended")

    def test_ending_an_already_ended_run_is_idempotent(self):
        self.start(run_id="idem")
        self.reconcile()
        service.end_run(run_id="idem", actor_user_id=self.admin_id)
        self.reconcile()
        again = service.end_run(run_id="idem", actor_user_id=self.admin_id)
        self.assertEqual(again["status"], "ended")

    def test_ending_an_unknown_run_is_404(self):
        with self.assertRaises(service.RunNotFoundError):
            service.end_run(run_id="ghost", actor_user_id=self.admin_id)

    def test_creator_comes_from_the_session(self):
        run = service.start_run(actor_user_id=self.other_admin_id, run_id="attributed")
        self.assertEqual(run["created_by_user_id"], self.other_admin_id)


class ReconciliationTests(RunTestCase):
    def test_worker_failure_during_start_marks_failed(self):
        self.start(run_id="r1")
        self.cv.state = "failed"
        self.reconcile()
        self.assertEqual(self.get("r1")["status"], "failed")

    def test_worker_failure_during_active_marks_interrupted(self):
        self.start(run_id="r2")
        self.reconcile()
        self.cv.state = "failed"
        self.reconcile()
        self.assertEqual(self.get("r2")["status"], "interrupted")

    def test_start_timeout_fails_the_run(self):
        self.start(run_id="slow")
        self.cv.state = "loading"
        with SessionLocal() as db:
            run = repository.get_by_run_id(db, "slow")
            run.requested_at = datetime.now(timezone.utc) - timedelta(seconds=999)
            db.commit()
        self.reconcile()
        self.assertEqual(self.get("slow")["status"], "failed")

    def test_stop_timeout_marks_interrupted(self):
        self.start(run_id="stuck")
        self.reconcile()
        service.end_run(run_id="stuck", actor_user_id=self.admin_id)
        self.cv.state = "running"  # worker never confirmed it stopped
        with SessionLocal() as db:
            run = repository.get_by_run_id(db, "stuck")
            run.status_changed_at = datetime.now(timezone.utc) - timedelta(seconds=999)
            db.commit()
        self.reconcile()
        self.assertEqual(self.get("stuck")["status"], "interrupted")

    def test_restart_recovery_marks_abandoned_run_interrupted(self):
        self.start(run_id="abandoned")
        self.reconcile()
        self.cv.state = "ready"
        self.cv.run_id = None
        service.recover_after_restart()
        recovered = self.get("abandoned")
        self.assertEqual(recovered["status"], "interrupted")
        self.assertIn("Server restarted", recovered["failure_reason"])

    def test_restart_recovery_leaves_a_genuinely_running_run_alone(self):
        self.start(run_id="survivor")
        self.reconcile()
        service.recover_after_restart()
        self.assertEqual(self.get("survivor")["status"], "active")


class ReconciliationCostTests(RunTestCase):
    """The active-run endpoint is polled by every browser, so the common path
    must not take SQLite's exclusive write lock."""

    def test_no_transition_is_due_when_no_run_is_in_progress(self):
        with SessionLocal() as db:
            self.assertFalse(service.reconciliation_due(db))

    def test_no_transition_is_due_while_an_active_run_is_healthy(self):
        self.start(run_id="steady")
        self.reconcile()
        with SessionLocal() as db:
            self.assertFalse(service.reconciliation_due(db))

    def test_a_transition_is_due_when_cv_confirms_a_starting_run(self):
        self.start(run_id="pending2")
        with SessionLocal() as db:
            self.assertTrue(service.reconciliation_due(db))

    def test_a_transition_is_due_when_the_worker_fails(self):
        self.start(run_id="dying")
        self.reconcile()
        self.cv.state = "failed"
        with SessionLocal() as db:
            self.assertTrue(service.reconciliation_due(db))

    def test_polling_the_read_only_check_never_takes_the_write_lock(self):
        import runs.write_guard as write_guard

        self.start(run_id="polled")
        self.reconcile()

        acquisitions = {"count": 0}
        original = write_guard._acquire_write_transaction

        def counting():
            acquisitions["count"] += 1
            return original()

        with patch.object(write_guard, "_acquire_write_transaction", counting):
            with SessionLocal() as db:
                for _ in range(25):
                    service.reconciliation_due(db)
        self.assertEqual(acquisitions["count"], 0)

    def test_reconcile_reports_whether_it_changed_anything(self):
        self.start(run_id="reports")
        with immediate_write() as db:
            self.assertTrue(service.reconcile(db))
        with immediate_write() as db:
            self.assertFalse(service.reconcile(db))


class RunSummaryTests(RunTestCase):
    def test_evacuee_count_excludes_cag_and_scdf_staff(self):
        with SessionLocal() as db:
            db.add_all(
                [
                    models.EvacueeIdentity(
                        run_id="role_count",
                        master_identity_id=1,
                        role="evacuee",
                    ),
                    models.EvacueeIdentity(
                        run_id="role_count",
                        master_identity_id=2,
                        role="evacuee",
                    ),
                    models.EvacueeIdentity(
                        run_id="role_count",
                        master_identity_id=3,
                        role="cag",
                    ),
                    models.EvacueeIdentity(
                        run_id="role_count",
                        master_identity_id=4,
                        role="scdf",
                    ),
                ]
            )
            db.commit()

            summaries = repository.collect_summaries(db)

        self.assertEqual(summaries["role_count"]["evacuee_count"], 2)


class LegacyBackfillTests(RunTestCase):
    def seed_legacy(self, run_id="field_test_001"):
        now = datetime.now(timezone.utc)
        with SessionLocal() as db:
            db.add(models.MetricLog(timestamp=now - timedelta(hours=3), run_id=run_id, passenger_count=4))
            db.add(models.MetricLog(timestamp=now - timedelta(hours=1), run_id=run_id, passenger_count=9))
            db.add(models.SystemAlert(timestamp=now - timedelta(hours=2), run_id=run_id,
                                      severity="warning", message="legacy"))
            db.commit()

    def test_existing_run_ids_are_imported_once(self):
        self.seed_legacy()
        with immediate_write() as db:
            imported = repository.backfill_legacy_runs(db)
        self.assertEqual(imported, ["field_test_001"])

        run = self.get("field_test_001")
        self.assertEqual(run["origin_type"], "legacy")
        self.assertEqual(run["status"], "ended")
        self.assertEqual(run["name"], "Imported field_test_001")

    def test_backfill_is_idempotent(self):
        self.seed_legacy()
        for _ in range(3):
            with immediate_write() as db:
                repository.backfill_legacy_runs(db)
        with SessionLocal() as db:
            self.assertEqual(db.query(Run).filter(Run.run_id == "field_test_001").count(), 1)

    def test_derived_times_match_the_underlying_data(self):
        self.seed_legacy()
        with immediate_write() as db:
            repository.backfill_legacy_runs(db)
        run = self.get("field_test_001")
        span = (run["ended_at"] - run["started_at"]).total_seconds()
        self.assertGreater(span, 3000)  # roughly the 1h..3h spread

    def test_legacy_rows_are_not_modified(self):
        self.seed_legacy()
        with SessionLocal() as db:
            before = [(m.id, m.passenger_count) for m in db.query(models.MetricLog).all()]
        with immediate_write() as db:
            repository.backfill_legacy_runs(db)
        with SessionLocal() as db:
            after = [(m.id, m.passenger_count) for m in db.query(models.MetricLog).all()]
        self.assertEqual(before, after)

    def test_tombstoned_run_ids_are_never_reimported(self):
        self.seed_legacy()
        with immediate_write() as db:
            repository.create_tombstone(db, "field_test_001", deleted_by_user_id=None)
        with immediate_write() as db:
            imported = repository.backfill_legacy_runs(db)
        self.assertEqual(imported, [])


class IngestionPolicyTests(RunTestCase):
    def resolve(self, run_id):
        with immediate_write() as db:
            return service.resolve_ingestion_run(db, run_id).run_id

    def test_payload_matching_the_managed_run_is_accepted(self):
        self.start(run_id="managed_run")
        self.assertEqual(self.resolve("managed_run"), "managed_run")

    def test_mismatched_run_id_is_rejected_while_a_run_is_active(self):
        self.start(run_id="managed_run")
        with self.assertRaises(service.IngestionRejected):
            self.resolve("something_else")

    def test_tombstoned_run_is_always_rejected(self):
        with immediate_write() as db:
            repository.create_tombstone(db, "gone", deleted_by_user_id=None)
        with self.assertRaises(service.IngestionRejected):
            self.resolve("gone")

    def test_unknown_run_becomes_external_under_the_compatibility_default(self):
        self.assertTrue(settings.allow_unmanaged_run_ingestion)
        self.assertEqual(self.resolve("debug_20260731_1200"), "debug_20260731_1200")
        run = self.get("debug_20260731_1200")
        self.assertEqual(run["origin_type"], "external")
        self.assertEqual(run["status"], "external")
        self.assertIsNone(run["ended_at"])  # never claims a confirmed end

    def test_external_run_is_created_exactly_once(self):
        for _ in range(4):
            self.resolve("debug_repeat")
        with SessionLocal() as db:
            self.assertEqual(db.query(Run).filter(Run.run_id == "debug_repeat").count(), 1)
            events = db.query(RunEvent).filter(
                RunEvent.run_id == "debug_repeat", RunEvent.event_type == "external_created"
            ).count()
        self.assertEqual(events, 1)

    def test_competing_external_run_is_rejected_within_the_activity_window(self):
        self.resolve("debug_one")
        with self.assertRaises(service.IngestionRejected):
            self.resolve("debug_two")

    def test_unknown_run_is_rejected_in_strict_mode(self):
        with patch.object(settings, "allow_unmanaged_run_ingestion", False):
            with self.assertRaises(service.IngestionRejected):
                self.resolve("unmanaged")

    def test_external_data_is_never_labelled_an_ended_legacy_run(self):
        self.resolve("debug_label")
        run = self.get("debug_label")
        self.assertNotEqual(run["origin_type"], "legacy")
        self.assertNotEqual(run["status"], "ended")


if __name__ == "__main__":
    unittest.main()
