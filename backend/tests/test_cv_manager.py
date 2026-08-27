import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr


from tests import BACKEND_DIR, TESTS_DIR
sys.path.insert(0, str(BACKEND_DIR))

from cv_manager import CvManager, CvTransitionError


FAKE_WORKER = TESTS_DIR / "fake_cv_worker.py"


def wait_for_state(manager, expected, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = manager.status()["state"]
        if state == expected:
            return manager.status()
        time.sleep(0.01)
    raise AssertionError(f"CV manager did not reach {expected!r}; got {manager.status()!r}")


class CvManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        directory = Path(self.temporary.name)
        self.command_log = directory / "commands.log"
        self.settings = SimpleNamespace(
            cv_enabled=True,
            cv_worker_python_path=Path(sys.executable),
            cv_worker_script_path=FAKE_WORKER,
            cv_worker_log_path=directory / "cv.jsonl",
            mqtt_host="127.0.0.1",
            mqtt_port=9,
            # The manager passes this to the worker environment so the CV
            # process can authenticate its HTTP evidence uploads.
            cv_service_token=SecretStr("unit-test-service-token"),
        )
        self.manager = None

    def tearDown(self):
        if self.manager is not None:
            self.manager.shutdown(timeout=1.0)
        self.temporary.cleanup()

    def start_manager(self, mode="ready"):
        environment = {
            "FAKE_CV_MODE": mode,
            "FAKE_CV_COMMAND_LOG": str(self.command_log),
        }
        self.environment_patch = patch.dict(os.environ, environment)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        self.manager = CvManager(self.settings)
        self.manager.start_worker()
        return self.manager

    def test_state_transitions_and_graceful_shutdown(self):
        manager = self.start_manager()
        wait_for_state(manager, "ready")
        manager.start_session("run_001")
        running = wait_for_state(manager, "running")
        self.assertTrue(running["running"])
        self.assertEqual(running["run_id"], "run_001")
        manager.stop_session()
        ready = wait_for_state(manager, "ready")
        self.assertFalse(ready["running"])
        manager.shutdown(timeout=1.0)
        self.assertEqual(manager.status()["state"], "offline")

    def test_duplicate_start_does_not_send_second_command(self):
        manager = self.start_manager()
        wait_for_state(manager, "ready")
        manager.start_session("same_run")
        wait_for_state(manager, "running")
        manager.start_session("same_run")
        time.sleep(0.05)
        commands = self.command_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(commands.count("start"), 1)
        with self.assertRaises(CvTransitionError):
            manager.start_session("different_run")

    def test_stop_when_idle_is_safe(self):
        manager = self.start_manager()
        wait_for_state(manager, "ready")
        self.assertEqual(manager.stop_session()["state"], "ready")

    def test_worker_startup_failure_is_reported(self):
        manager = self.start_manager("startup_failure")
        status = wait_for_state(manager, "failed")
        self.assertIn("exited unexpectedly", status["error"])

    def test_model_loading_failure_is_preserved(self):
        manager = self.start_manager("model_failure")
        status = wait_for_state(manager, "failed")
        self.assertEqual(status["error"], "Fake model failed to load")

    def test_rtsp_credentials_are_redacted_from_manager_text(self):
        safe = CvManager._safe_log_text(
            "failed rtsp://admin:camera-secret@192.0.2.10/Streaming/Channels/101"
        )
        self.assertNotIn("camera-secret", safe)
        self.assertIn("rtsp://<credentials>@192.0.2.10", safe)


if __name__ == "__main__":
    unittest.main()
