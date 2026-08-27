"""Migration tests.

Each case runs in its own subprocess against a throwaway database, because the
configured SQLite path is resolved once per process. Nothing here touches the
operator's real database.
"""

import re
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests import BACKEND_DIR
PYTHON = str(BACKEND_DIR / ".venv-linux" / "bin" / "python")


def run_python(code: str, db_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(BACKEND_DIR),
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "SQLITE_DB_PATH": str(db_path),
            "OBSERVATION_UPLOAD_DIR": str(db_path.parent / "observations"),
            "EVACUEE_UPLOAD_DIR": str(db_path.parent / "evacuees"),
            "CAMERA_URL": "",
            "CAMERA_URLS": "",
            "CV_ENABLED": "false",
            "MQTT_ENABLED": "false",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )


def alembic(args: list[str], db_path: Path) -> subprocess.CompletedProcess:
    return run_python(
        f"import sys; sys.argv=['alembic']+{args!r};"
        "from alembic.config import main; main(argv=sys.argv[1:])",
        db_path,
    )


def head_revision() -> str:
    """Derive head from the migration scripts so new revisions do not break tests."""
    import subprocess
    result = subprocess.run(
        [PYTHON, "-c",
         "from alembic.config import Config; from alembic.script import ScriptDirectory;"
         "c=Config('alembic.ini'); c.set_main_option('script_location','migrations');"
         "print(ScriptDirectory.from_config(c).get_current_head())"],
        cwd=str(BACKEND_DIR), capture_output=True, text=True, timeout=60,
    )
    return result.stdout.strip()


def schema_of(db_path: Path) -> dict:
    connection = sqlite3.connect(db_path)
    try:
        return {
            (kind, name): re.sub(r"\s+", " ", sql or "").strip()
            for kind, name, sql in connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
            )
        }
    finally:
        connection.close()


SEED_LEGACY_DATABASE = """
import database, models
from datetime import datetime, timezone
database.Base.metadata.create_all(bind=database.engine)
session = database.SessionLocal()
now = datetime.now(timezone.utc)
session.add(models.MetricLog(timestamp=now, run_id="field_test_001", passenger_count=7,
                             zone_counts='{"cam_1":4}', camera_online_count=2))
session.add(models.SystemAlert(timestamp=now, run_id="field_test_001", severity="warning",
                               message="cam_1 visibility reduced"))
session.add(models.PassengerObservation(timestamp=now, run_id="field_test_001", camera_id="cam_1",
                                        age=34.0, gender="male", image_path="/x/a.jpg",
                                        image_url="/uploads/observations/a.jpg"))
identity = models.EvacueeIdentity(run_id="field_test_001", master_identity_id=12,
                                  role="evacuee", gender="female", age=29.0)
session.add(identity); session.flush()
session.add(models.EvacueeGalleryView(evacuee_id=identity.id, view_type="front",
                                      image_path="/x/f.png", image_url="/legacy/f.png",
                                      feature_dimension=768))
session.commit(); session.close()
"""


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cag-migration-"))

    def test_fresh_database_upgrades_to_head(self):
        db_path = self.tmp / "fresh.db"
        result = alembic(["upgrade", "head"], db_path)
        self.assertEqual(result.returncode, 0, result.stderr)

        connection = sqlite3.connect(db_path)
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        connection.close()

        self.assertEqual(revision, head_revision())
        for table in (
            "metric_logs", "system_alerts", "passenger_observations",
            "evacuee_identities", "evacuee_gallery_views",
            "users", "auth_sessions", "auth_events",
            "runs", "run_events", "deleted_runs", "pending_file_deletions",
        ):
            self.assertIn(table, tables)

    def test_baseline_matches_the_pre_auth_create_all_schema(self):
        """A stamped existing database and a fresh one must be identical."""
        legacy_path = self.tmp / "legacy.db"
        baseline_path = self.tmp / "baseline.db"

        seeded = run_python(SEED_LEGACY_DATABASE, legacy_path)
        self.assertEqual(seeded.returncode, 0, seeded.stderr)
        upgraded = alembic(["upgrade", "0001_baseline"], baseline_path)
        self.assertEqual(upgraded.returncode, 0, upgraded.stderr)

        self.assertEqual(schema_of(legacy_path), schema_of(baseline_path))

    def test_populated_database_survives_stamp_and_upgrade(self):
        db_path = self.tmp / "populated.db"
        self.assertEqual(run_python(SEED_LEGACY_DATABASE, db_path).returncode, 0)

        stamped = alembic(["stamp", "0001_baseline"], db_path)
        self.assertEqual(stamped.returncode, 0, stamped.stderr)
        upgraded = alembic(["upgrade", "head"], db_path)
        self.assertEqual(upgraded.returncode, 0, upgraded.stderr)

        connection = sqlite3.connect(db_path)
        try:
            for table in (
                "metric_logs", "system_alerts", "passenger_observations",
                "evacuee_identities", "evacuee_gallery_views",
            ):
                count = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                self.assertEqual(count, 1, f"{table} lost rows during upgrade")
            self.assertEqual(
                connection.execute("SELECT image_url FROM evacuee_gallery_views").fetchone()[0],
                "/legacy/f.png",
            )
            self.assertEqual(
                connection.execute("SELECT version_num FROM alembic_version").fetchone()[0],
                head_revision(),
            )
        finally:
            connection.close()

    def test_startup_refuses_an_unstamped_legacy_database(self):
        db_path = self.tmp / "unstamped.db"
        self.assertEqual(run_python(SEED_LEGACY_DATABASE, db_path).returncode, 0)

        result = run_python("import database; database.run_migrations()", db_path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not stamped with an Alembic revision", result.stderr)
        self.assertIn("alembic stamp 0001_baseline", result.stderr)

    def test_downgrade_removes_only_the_auth_tables(self):
        db_path = self.tmp / "downgrade.db"
        self.assertEqual(alembic(["upgrade", "head"], db_path).returncode, 0)
        result = alembic(["downgrade", "0001_baseline"], db_path)
        self.assertEqual(result.returncode, 0, result.stderr)

        connection = sqlite3.connect(db_path)
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        connection.close()
        self.assertNotIn("users", tables)
        self.assertNotIn("auth_sessions", tables)
        self.assertIn("metric_logs", tables)
        self.assertIn("evacuee_gallery_views", tables)


if __name__ == "__main__":
    unittest.main()
