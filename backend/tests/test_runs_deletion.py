"""Run deletion: scoping, file cleanup, tombstones, and concurrency."""

import os
import struct
import sys
import tempfile
import threading
import unittest
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tests import BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))

_TMP = tempfile.mkdtemp(prefix="cag-runs-del-")
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
from auth import users as user_service  # noqa: E402
from auth.models import User  # noqa: E402
from database import SessionLocal, init_db  # noqa: E402
from runs import repository, service  # noqa: E402
from runs.models import DeletedRun, PendingFileDeletion, Run, RunEvent  # noqa: E402
from runs.write_guard import immediate_write  # noqa: E402
from tactical_state import tactical_store  # noqa: E402

EVACUEE_ROOT = Path(_TMP) / "evacuees"
OBSERVATION_ROOT = Path(_TMP) / "observations"


def assert_isolated_database():
    import database

    resolved = str(database.DATABASE_PATH)
    if not resolved.startswith(tempfile.gettempdir()):
        raise AssertionError(f"Refusing to run against the real database at {resolved}.")


def tiny_png() -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


class FakeCv:
    def __init__(self):
        self.state = "ready"
        self.run_id = None

    def status(self):
        return {"state": self.state, "run_id": self.run_id, "ready": self.state == "ready"}

    def start_session(self, run_id=None):
        self.state, self.run_id = "running", run_id

    def stop_session(self):
        self.state = "ready"


class DeletionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_isolated_database()
        import evacuees.storage
        import observation_storage

        EVACUEE_ROOT.mkdir(parents=True, exist_ok=True)
        OBSERVATION_ROOT.mkdir(parents=True, exist_ok=True)
        cls._pinned = [
            patch.object(evacuees.storage, "UPLOAD_DIR", EVACUEE_ROOT),
            patch.object(observation_storage, "UPLOAD_DIR", OBSERVATION_ROOT),
            patch.object(service, "_UPLOAD_ROOTS", (EVACUEE_ROOT, OBSERVATION_ROOT)),
        ]
        for pin in cls._pinned:
            pin.start()
        init_db()

    @classmethod
    def tearDownClass(cls):
        for pin in cls._pinned:
            pin.stop()

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
            self.admin_id = user_service.create_user(
                db, username="denn", display_name="Denn",
                password="AdminPassword1", role="admin",
            ).id
            db.commit()

    def tearDown(self):
        self.cv_patch.stop()

    def seed_run(self, run_id, *, images=2, status="ended", origin="managed"):
        """Create a fully populated run with real image files on disk."""
        now = datetime.now(timezone.utc)
        paths = []
        with immediate_write() as db:
            repository.create_run(
                db, run_id=run_id, status=status, origin_type=origin,
                requested_at=now, started_at=now,
                ended_at=now if status == "ended" else None,
                first_ingested_at=now, last_ingested_at=now,
            )
            db.add(models.MetricLog(timestamp=now, run_id=run_id, passenger_count=5))
            db.add(models.SystemAlert(timestamp=now, run_id=run_id, severity="info", message="x"))

            identity = models.EvacueeIdentity(run_id=run_id, master_identity_id=1)
            db.add(identity)
            db.flush()

            for index in range(images):
                folder = EVACUEE_ROOT / run_id / "master_0001"
                folder.mkdir(parents=True, exist_ok=True)
                image_path = folder / f"front_{index}.png"
                image_path.write_bytes(tiny_png())
                paths.append(image_path)
                db.add(models.EvacueeGalleryView(
                    evacuee_id=identity.id, view_type=("front" if index == 0 else "back"),
                    image_path=str(image_path), image_url="/legacy",
                ))

            observation_path = OBSERVATION_ROOT / f"{run_id}_obs.png"
            observation_path.write_bytes(tiny_png())
            paths.append(observation_path)
            db.add(models.PassengerObservation(
                timestamp=now, run_id=run_id, camera_id="cam_1", age=30.0, gender="male",
                image_path=str(observation_path), image_url="/legacy",
            ))
        return paths

    def delete(self, run_id, confirm=None):
        return service.delete_run(
            run_id=run_id,
            confirm_run_id=run_id if confirm is None else confirm,
            actor_user_id=self.admin_id,
        )


class DeletionScopeTests(DeletionTestCase):
    def test_wrong_confirmation_is_rejected(self):
        self.seed_run("run_a")
        with self.assertRaises(service.RunValidationError):
            self.delete("run_a", confirm="run_A")

    def test_unknown_run_is_404(self):
        with self.assertRaises(service.RunNotFoundError):
            self.delete("nope")

    def test_in_progress_run_cannot_be_deleted(self):
        self.seed_run("live", status="active")
        with self.assertRaises(service.RunConflictError):
            self.delete("live")

    def test_cv_state_prevents_deleting_a_truly_running_run(self):
        self.seed_run("ghost_row", status="ended")
        self.cv.state, self.cv.run_id = "running", "ghost_row"
        with self.assertRaises(service.RunConflictError):
            self.delete("ghost_row")

    def test_deleting_one_run_leaves_another_untouched(self):
        keep_paths = self.seed_run("keep")
        self.seed_run("drop")
        self.delete("drop")

        with SessionLocal() as db:
            self.assertIsNotNone(repository.get_by_run_id(db, "keep"))
            self.assertIsNone(repository.get_by_run_id(db, "drop"))
            self.assertEqual(db.query(models.MetricLog).filter_by(run_id="keep").count(), 1)
            self.assertEqual(db.query(models.MetricLog).filter_by(run_id="drop").count(), 0)
        for path in keep_paths:
            self.assertTrue(path.exists(), f"{path} should have survived")

    def test_all_run_scoped_rows_are_removed(self):
        self.seed_run("purge")
        summary = self.delete("purge")
        self.assertEqual(summary["deleted_metrics"], 1)
        self.assertEqual(summary["deleted_alerts"], 1)
        self.assertEqual(summary["deleted_observations"], 1)
        self.assertEqual(summary["deleted_evacuees"], 1)
        self.assertEqual(summary["deleted_gallery_views"], 2)

    def test_gallery_rows_are_found_through_identities(self):
        self.seed_run("via_identity")
        self.delete("via_identity")
        with SessionLocal() as db:
            self.assertEqual(db.query(models.EvacueeGalleryView).count(), 0)


class FileCleanupTests(DeletionTestCase):
    def test_evidence_files_are_removed_after_commit(self):
        paths = self.seed_run("withfiles")
        summary = self.delete("withfiles")
        self.assertEqual(summary["deleted_images"], len(paths))
        for path in paths:
            self.assertFalse(path.exists())

    def test_missing_file_counts_as_clean_and_never_rolls_back(self):
        paths = self.seed_run("halfgone")
        paths[0].unlink()
        summary = self.delete("halfgone")
        self.assertEqual(summary["file_cleanup_failures"], 0)
        with SessionLocal() as db:
            self.assertIsNone(repository.get_by_run_id(db, "halfgone"))

    def test_failed_cleanup_is_reported_and_recorded_without_absolute_paths(self):
        self.seed_run("stubborn")

        real_unlink = Path.unlink

        def refuse(self, *args, **kwargs):
            if self.suffix == ".png":
                raise PermissionError("read-only")
            return real_unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", refuse):
            summary = self.delete("stubborn")

        self.assertGreater(summary["file_cleanup_failures"], 0)
        self.assertTrue(summary["file_cleanup_warnings"])
        with SessionLocal() as db:
            pending = db.query(PendingFileDeletion).all()
            self.assertTrue(pending)
            for record in pending:
                self.assertFalse(record.storage_key.startswith("/"))
                self.assertNotIn(str(EVACUEE_ROOT), record.storage_key)
                self.assertEqual(record.safe_error_code, "PermissionError")
            # The database deletion still succeeded.
            self.assertIsNone(repository.get_by_run_id(db, "stubborn"))

    def test_a_file_outside_the_upload_root_is_never_deleted(self):
        outside = Path(_TMP) / "not_evidence.png"
        outside.write_bytes(tiny_png())
        self.seed_run("sneaky")
        with immediate_write() as db:
            view = db.query(models.EvacueeGalleryView).first()
            view.image_path = str(outside)
        self.delete("sneaky")
        self.assertTrue(outside.exists(), "a path outside the upload root must be left alone")


class TombstoneTests(DeletionTestCase):
    def test_tombstone_and_audit_event_survive_deletion(self):
        self.seed_run("audited")
        self.delete("audited")
        with SessionLocal() as db:
            self.assertIsNotNone(db.get(DeletedRun, "audited"))
            events = [e.event_type for e in db.query(RunEvent).filter_by(run_id="audited")]
            self.assertIn("deleted", events)

    def test_late_payload_cannot_recreate_a_deleted_run(self):
        self.seed_run("late")
        self.delete("late")
        with self.assertRaises(service.IngestionRejected):
            with immediate_write() as db:
                service.resolve_ingestion_run(db, "late")
        with SessionLocal() as db:
            self.assertIsNone(repository.get_by_run_id(db, "late"))

    def test_deleted_run_id_cannot_be_reused_for_a_new_run(self):
        self.seed_run("reused")
        self.delete("reused")
        with self.assertRaises(service.RunConflictError):
            service.start_run(actor_user_id=self.admin_id, run_id="reused")

    def test_sqlite_files_are_never_removed(self):
        import database

        self.seed_run("keepdb")
        self.delete("keepdb")
        self.assertTrue(Path(database.DATABASE_PATH).exists())

    def test_matching_tactical_cache_is_cleared_and_others_kept(self):
        from models import TacticalStateCreate

        for run_id in ("cached", "other"):
            tactical_store.update(TacticalStateCreate(
                timestamp=int(datetime.now(timezone.utc).timestamp()),
                camera_id="fused", run_id=run_id, people_count=1,
                positions_cm=[{"x": 10.0, "y": 10.0, "area": "inside"}],
                map_size_cm=480, outside_context_cm=700,
            ))
        self.seed_run("cached")
        self.delete("cached")
        self.assertFalse(tactical_store.latest(run_id="cached").has_data)
        self.assertTrue(tactical_store.latest(run_id="other").has_data)


class ExternalRunDeletionTests(DeletionTestCase):
    def test_recently_active_external_run_cannot_be_deleted(self):
        self.seed_run("busy_external", status="external", origin="external")
        with self.assertRaises(service.RunConflictError) as context:
            self.delete("busy_external")
        self.assertIn("Stop the external CV process", str(context.exception))

    def test_quiet_external_run_can_be_deleted(self):
        self.seed_run("quiet_external", status="external", origin="external")
        with immediate_write() as db:
            run = repository.get_by_run_id(db, "quiet_external")
            run.last_ingested_at = datetime.now(timezone.utc) - timedelta(seconds=600)
        summary = self.delete("quiet_external")
        self.assertEqual(summary["run_id"], "quiet_external")


class ConcurrencyTests(DeletionTestCase):
    def test_deletion_and_ingestion_cannot_interleave(self):
        """A concurrent write must not resurrect a run being deleted."""
        self.seed_run("racy")
        errors = []
        barrier = threading.Barrier(2, timeout=10)

        def deleter():
            try:
                barrier.wait()
                self.delete("racy")
            except Exception as error:  # pragma: no cover - surfaced via errors
                errors.append(("delete", error))

        def ingester():
            try:
                barrier.wait()
                for _ in range(20):
                    try:
                        with immediate_write() as db:
                            service.resolve_ingestion_run(db, "racy")
                            db.add(models.MetricLog(
                                timestamp=datetime.now(timezone.utc),
                                run_id="racy", passenger_count=1,
                            ))
                    except service.IngestionRejected:
                        return  # tombstone won; correct outcome
            except Exception as error:  # pragma: no cover
                errors.append(("ingest", error))

        threads = [threading.Thread(target=deleter), threading.Thread(target=ingester)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        with SessionLocal() as db:
            # Whatever the interleaving, the run must be gone and stay gone.
            self.assertIsNone(repository.get_by_run_id(db, "racy"))
            self.assertIsNotNone(db.get(DeletedRun, "racy"))
            self.assertEqual(db.query(models.MetricLog).filter_by(run_id="racy").count(), 0)


if __name__ == "__main__":
    unittest.main()
