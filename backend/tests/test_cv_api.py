"""Unit tests for the CV session handlers.

Authorization moved to the shared auth dependencies when login landed: CV
Start/Stop requires an authenticated admin session plus CSRF, covered end to end
by `CvControlTests` in test_auth_protection.py.

Run Manager then made these compatibility routes delegate to the run service
rather than driving `cv_manager` directly, so that no supported path starts CV
without managed-run bookkeeping. These tests cover the status contract and that
delegation.
"""

import sys
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from fastapi import HTTPException

from tests import BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))

import cv_api
from cv_api import CvSessionStart, CvStatus
from cv_manager import CvTransitionError


@dataclass
class FakeUser:
    role: str
    id: int = 1


@dataclass
class FakeContext:
    user: FakeUser


ADMIN_CONTEXT = FakeContext(user=FakeUser(role="admin"))
STAFF_CONTEXT = FakeContext(user=FakeUser(role="staff"))


class FakeManager:
    def __init__(self):
        self.start_calls = []
        self.stop_calls = 0
        self.raise_on_start = None
        self.raise_on_stop = None

    def status(self):
        return {
            "state": "ready",
            "ready": True,
            "running": False,
            "run_id": None,
            "started_at": None,
            "stopped_at": None,
            "pid": 123,
            "loading_stage": "Complete",
            "error": None,
            "mqtt_broker_reachable": True,
        }

    def start_session(self, run_id=None):
        if self.raise_on_start:
            raise self.raise_on_start
        self.start_calls.append(run_id)

    def stop_session(self):
        if self.raise_on_stop:
            raise self.raise_on_stop
        self.stop_calls += 1


class CvApiTests(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        self.manager_patch = patch.object(cv_api, "cv_manager", self.manager)
        self.manager_patch.start()

    def tearDown(self):
        self.manager_patch.stop()

    def test_status_contract(self):
        validated = CvStatus(**cv_api.get_cv_status(ADMIN_CONTEXT))
        self.assertTrue(validated.ready)
        self.assertTrue(validated.control_allowed)
        self.assertEqual(validated.control_mode, "admin_session")

    def test_status_reports_no_control_for_staff(self):
        validated = CvStatus(**cv_api.get_cv_status(STAFF_CONTEXT))
        self.assertFalse(validated.control_allowed)

    def test_start_delegates_to_run_manager_not_cv_manager(self):
        """The legacy route must create a managed run, never bypass bookkeeping."""
        with patch("runs.service.start_run") as start_run:
            cv_api.start_cv_session(CvSessionStart(run_id="field_test_1"), ADMIN_CONTEXT)
        start_run.assert_called_once()
        self.assertEqual(start_run.call_args.kwargs["run_id"], "field_test_1")
        self.assertEqual(start_run.call_args.kwargs["actor_user_id"], ADMIN_CONTEXT.user.id)
        # It must not have driven the worker directly.
        self.assertEqual(self.manager.start_calls, [])

    def test_start_without_a_run_id_lets_run_manager_generate_one(self):
        with patch("runs.service.start_run") as start_run:
            cv_api.start_cv_session(CvSessionStart(), ADMIN_CONTEXT)
        self.assertIsNone(start_run.call_args.kwargs["run_id"])

    def test_start_conflict_becomes_409(self):
        from runs.service import RunConflictError

        with patch("runs.service.start_run", side_effect=RunConflictError("already running")):
            with self.assertRaises(HTTPException) as context:
                cv_api.start_cv_session(CvSessionStart(run_id="field_test_1"), ADMIN_CONTEXT)
        self.assertEqual(context.exception.status_code, 409)

    def test_start_validation_error_becomes_400(self):
        from runs.service import RunValidationError

        with patch("runs.service.start_run", side_effect=RunValidationError("bad id")):
            with self.assertRaises(HTTPException) as context:
                cv_api.start_cv_session(CvSessionStart(run_id="ok_id"), ADMIN_CONTEXT)
        self.assertEqual(context.exception.status_code, 400)

    def test_stop_closes_the_managed_run_when_one_is_in_progress(self):
        class FakeRun:
            run_id = "field_test_1"

        with patch("runs.repository.get_in_progress", return_value=FakeRun()), \
             patch("runs.service.end_run") as end_run:
            cv_api.stop_cv_session(ADMIN_CONTEXT)
        end_run.assert_called_once()
        self.assertEqual(end_run.call_args.kwargs["run_id"], "field_test_1")
        self.assertEqual(self.manager.stop_calls, 0)

    def test_stop_falls_back_to_the_worker_when_no_managed_run_exists(self):
        with patch("runs.repository.get_in_progress", return_value=None):
            cv_api.stop_cv_session(ADMIN_CONTEXT)
        self.assertEqual(self.manager.stop_calls, 1)

    def test_stop_transition_error_becomes_409(self):
        self.manager.raise_on_stop = CvTransitionError("No session is running.")
        with patch("runs.repository.get_in_progress", return_value=None):
            with self.assertRaises(HTTPException) as context:
                cv_api.stop_cv_session(ADMIN_CONTEXT)
        self.assertEqual(context.exception.status_code, 409)

    def test_legacy_token_caller_has_no_session_and_no_control_flag(self):
        # require_high_risk_control returns None for the transitional token path.
        validated = CvStatus(**cv_api.get_cv_status(None))
        self.assertFalse(validated.control_allowed)

    def test_control_mode_reflects_the_legacy_compatibility_setting(self):
        with patch.object(cv_api.settings, "cv_control_legacy_token_enabled", True):
            validated = CvStatus(**cv_api.get_cv_status(ADMIN_CONTEXT))
        self.assertEqual(validated.control_mode, "legacy_token")


if __name__ == "__main__":
    unittest.main()
