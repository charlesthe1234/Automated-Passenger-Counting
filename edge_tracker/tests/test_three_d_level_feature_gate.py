"""Tests that the 3D Level Detection checkbox fully controls the feature.

The acceptance rule is that unticking one checkbox removes every trace of the
experimental geometry from a run -- not that its output is merely hidden.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


from launch_config import build_tracker_arguments, default_launch_values, is_truthy
from tests.test_three_d_level import synthetic_camera, write_calibration


def tracker_arguments(**overrides):
    values = default_launch_values()
    values.update(
        {
            "camera_mode": "both",
            "source_1": "rtsp://example/1",
            "source_2": "rtsp://example/2",
            "camera_id_1": "cam_1",
            "camera_id_2": "cam_2",
        }
    )
    values.update(overrides)
    return build_tracker_arguments(values)


class LauncherConfigurationTests(unittest.TestCase):
    def test_flag_is_absent_by_default(self):
        self.assertNotIn("--enable-3d-level-detection", tracker_arguments())

    def test_unticked_checkbox_passes_no_flag_and_no_calibration_paths(self):
        arguments = tracker_arguments(enable_3d_level_detection=False)
        self.assertNotIn("--enable-3d-level-detection", arguments)
        self.assertNotIn("--elevated-matrix", arguments)
        self.assertNotIn("--elevated-matrix-2", arguments)

    def test_ticked_checkbox_passes_the_flag_and_both_calibration_paths(self):
        arguments = tracker_arguments(
            enable_3d_level_detection=True,
            elevated_matrix_1="ground_up_1.json",
            elevated_matrix_2="ground_up_2.json",
        )
        self.assertIn("--enable-3d-level-detection", arguments)
        self.assertEqual(arguments[arguments.index("--elevated-matrix") + 1], "ground_up_1.json")
        self.assertEqual(arguments[arguments.index("--elevated-matrix-2") + 1], "ground_up_2.json")

    def test_string_values_from_the_environment_are_accepted(self):
        for raw in ("true", "TRUE", "1", "yes", "on"):
            self.assertIn("--enable-3d-level-detection", tracker_arguments(enable_3d_level_detection=raw))
        for raw in ("false", "0", "no", "off", ""):
            self.assertNotIn(
                "--enable-3d-level-detection", tracker_arguments(enable_3d_level_detection=raw)
            )

    def test_environment_default_is_off(self):
        self.assertFalse(is_truthy(default_launch_values().get("enable_3d_level_detection")))

    def test_toggling_between_launches_does_not_retain_stale_state(self):
        """Two consecutive builds must reflect only their own setting."""
        enabled = tracker_arguments(enable_3d_level_detection=True)
        disabled = tracker_arguments(enable_3d_level_detection=False)
        again = tracker_arguments(enable_3d_level_detection=True)
        self.assertIn("--enable-3d-level-detection", enabled)
        self.assertNotIn("--enable-3d-level-detection", disabled)
        self.assertIn("--enable-3d-level-detection", again)


class PositionConfidenceGatingCheckboxTests(unittest.TestCase):
    """The legacy-location-rule checkbox must reach the tracker, and only then."""

    def test_flag_is_absent_by_default(self):
        self.assertNotIn("--disable-position-confidence-gating", tracker_arguments())

    def test_unticked_keeps_the_confidence_rule_active(self):
        arguments = tracker_arguments(disable_position_confidence_gating=False)
        self.assertNotIn("--disable-position-confidence-gating", arguments)

    def test_ticked_passes_the_flag(self):
        arguments = tracker_arguments(disable_position_confidence_gating=True)
        self.assertIn("--disable-position-confidence-gating", arguments)

    def test_environment_strings_are_honoured(self):
        for raw in ("true", "1", "YES", "on"):
            self.assertIn(
                "--disable-position-confidence-gating",
                tracker_arguments(disable_position_confidence_gating=raw),
            )
        for raw in ("false", "0", "no", ""):
            self.assertNotIn(
                "--disable-position-confidence-gating",
                tracker_arguments(disable_position_confidence_gating=raw),
            )

    def test_the_two_experimental_checkboxes_are_independent(self):
        only_legacy = tracker_arguments(disable_position_confidence_gating=True)
        self.assertNotIn("--enable-3d-level-detection", only_legacy)
        only_three_d = tracker_arguments(enable_3d_level_detection=True)
        self.assertNotIn("--disable-position-confidence-gating", only_three_d)


class TrackerGateTests(unittest.TestCase):
    """The tracker-side gate, exercised without importing heavy CV modules."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        projection = synthetic_camera()
        write_calibration(self.root / "ground.json", projection, 0.0)
        write_calibration(self.root / "elevated.json", projection, 170.0)

    @staticmethod
    def _args(**overrides):
        defaults = {
            "enable_3d_level_detection": False,
            "elevated_matrix": "elevated.json",
            "elevated_matrix_2": "elevated2.json",
            "three_d_max_lean_degrees": 6.0,
        }
        defaults.update(overrides)
        return type("Args", (), defaults)()

    @staticmethod
    def _contexts(root, count=1):
        return [
            type("Ctx", (), {"camera_id": f"cam_{index + 1}", "matrix_path": root / "ground.json"})()
            for index in range(count)
        ]

    def _build(self, args, contexts):
        from main_tracker import build_three_d_level_estimator

        return build_three_d_level_estimator(contexts, args)

    def test_disabled_returns_none_and_never_imports_the_module(self):
        """A disabled run must not even load the experimental geometry."""
        for name in [key for key in sys.modules if key == "three_d_level"]:
            del sys.modules[name]
        estimator = self._build(self._args(enable_3d_level_detection=False), self._contexts(self.root))
        self.assertIsNone(estimator)
        self.assertNotIn("three_d_level", sys.modules)

    def test_disabled_does_not_open_any_calibration_file(self):
        opened = []
        real_open = Path.open

        def tracking_open(self, *args, **kwargs):
            opened.append(str(self))
            return real_open(self, *args, **kwargs)

        with mock.patch.object(Path, "open", tracking_open), \
             mock.patch.object(Path, "read_text", lambda self, **kw: opened.append(str(self)) or ""):
            self._build(self._args(enable_3d_level_detection=False), self._contexts(self.root))
        self.assertEqual([path for path in opened if "elevated" in path], [])

    def test_enabled_with_valid_calibration_returns_a_ready_estimator(self):
        estimator = self._build(
            self._args(enable_3d_level_detection=True, elevated_matrix=str(self.root / "elevated.json")),
            self._contexts(self.root),
        )
        self.assertIsNotNone(estimator)
        self.assertTrue(estimator.ready)

    def test_missing_calibration_disables_the_feature_without_raising(self):
        """Standard 2D tracking must survive a broken experimental calibration."""
        estimator = self._build(
            self._args(enable_3d_level_detection=True, elevated_matrix=str(self.root / "absent.json")),
            self._contexts(self.root),
        )
        self.assertIsNone(estimator)

    def test_malformed_calibration_disables_the_feature_without_raising(self):
        (self.root / "broken.json").write_text("{oops", encoding="utf-8")
        estimator = self._build(
            self._args(enable_3d_level_detection=True, elevated_matrix=str(self.root / "broken.json")),
            self._contexts(self.root),
        )
        self.assertIsNone(estimator)

    def test_second_camera_without_calibration_disables_the_feature(self):
        estimator = self._build(
            self._args(
                enable_3d_level_detection=True,
                elevated_matrix=str(self.root / "elevated.json"),
                elevated_matrix_2=str(self.root / "absent2.json"),
            ),
            self._contexts(self.root, count=2),
        )
        self.assertIsNone(estimator)


class LauncherStatusTests(unittest.TestCase):
    """Launcher validation logic, exercised without opening a window.

    The methods are called unbound against a duck-typed stand-in so the checks
    run on a headless machine, where no X display is available.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        projection = synthetic_camera()
        write_calibration(self.root / "ground.json", projection, 0.0)
        write_calibration(self.root / "elevated.json", projection, 170.0)

    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

        def set(self, value):
            self._value = value

    class _Label:
        def __init__(self):
            self.text = ""

        def configure(self, text=None, **_kwargs):
            self.text = text

        def cget(self, _key):
            return self.text

    def _fake_launcher(self, enabled=True, elevated="elevated.json"):
        import launcher_ui

        fake = mock.Mock()
        fake.enable_3d_level_detection = self._Var(enabled)
        fake.camera_id_1 = self._Var("cam_1")
        fake.matrix_1 = self._Var("ground.json")
        fake._selected_cameras = lambda: ["1"]
        fake.three_d_status = self._Label()
        fake._three_d_calibration_problem = (
            lambda: launcher_ui.LauncherApp._three_d_calibration_problem(fake)
        )
        patcher = mock.patch.object(launcher_ui, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        values_patcher = mock.patch.object(
            launcher_ui, "default_launch_values", lambda: {"elevated_matrix_1": elevated}
        )
        values_patcher.start()
        self.addCleanup(values_patcher.stop)
        return launcher_ui, fake

    def test_disabled_reports_no_problem_and_shows_disabled(self):
        launcher_ui, fake = self._fake_launcher(enabled=False)
        self.assertIsNone(launcher_ui.LauncherApp._three_d_calibration_problem(fake))
        launcher_ui.LauncherApp._refresh_three_d_status(fake)
        self.assertEqual(fake.three_d_status.cget("text"), "3D Level Detection: Disabled")

    def test_valid_calibration_shows_enabled(self):
        launcher_ui, fake = self._fake_launcher(enabled=True)
        self.assertIsNone(launcher_ui.LauncherApp._three_d_calibration_problem(fake))
        launcher_ui.LauncherApp._refresh_three_d_status(fake)
        self.assertEqual(fake.three_d_status.cget("text"), "3D Level Detection: Enabled")

    def test_missing_calibration_produces_the_specified_operator_message(self):
        launcher_ui, fake = self._fake_launcher(enabled=True, elevated="absent.json")
        problem = launcher_ui.LauncherApp._three_d_calibration_problem(fake)
        self.assertIsNotNone(problem)
        self.assertIn("3D Level Detection could not start", problem)
        self.assertIn("cam_1", problem)
        self.assertIn("standard 2D ground-plane tracking", problem)
        launcher_ui.LauncherApp._refresh_three_d_status(fake)
        self.assertEqual(
            fake.three_d_status.cget("text"), "3D Level Detection: Unavailable - calibration error"
        )

    def test_status_label_missing_during_window_build_is_tolerated(self):
        launcher_ui, fake = self._fake_launcher(enabled=False)
        fake.three_d_status = None
        launcher_ui.LauncherApp._refresh_three_d_status(fake)  # must not raise


class ShadowModeTests(unittest.TestCase):
    """The shadow hook must observe and log without changing any position."""

    def test_no_estimator_means_the_hook_does_nothing(self):
        from pose_engine import run_three_d_level_shadow

        self.assertIsNone(
            run_three_d_level_shadow(None, {}, "mediapipe", "cam_1", 1, 1, "confirmed", 0, None)
        )

    def test_hook_never_raises_into_the_tracking_loop(self):
        from pose_engine import run_three_d_level_shadow

        class Exploding:
            ready = True

            def learn_landmark_heights(self, _observation):
                raise RuntimeError("experimental geometry failed")

        result = run_three_d_level_shadow(
            Exploding(),
            {"metrology_landmarks": {"nose": (10.0, 20.0)}, "strict_foot_point": (10.0, 90.0)},
            "mediapipe",
            "cam_1",
            1,
            1,
            "confirmed",
            0,
            None,
        )
        self.assertIsNone(result)

    def test_only_a_strict_direct_foot_point_may_teach_a_height(self):
        from pose_engine import run_three_d_level_shadow

        seen = {}

        class Recording:
            ready = True

            def learn_landmark_heights(self, observation):
                seen.update(observation)
                return {}

            def estimate_ground_position(self, _observation, _state=None):
                from three_d_level import UNAVAILABLE

                return UNAVAILABLE, []

            def known_heights(self, _identity_id):
                return {}

        landmarks = {"metrology_landmarks": {"nose": (10.0, 20.0)}, "strict_foot_point": None}
        run_three_d_level_shadow(
            Recording(), landmarks, "anatomical_ratio", "cam_1", 1, 1, "confirmed", 0, None
        )
        self.assertFalse(seen["foot_is_direct"])

        landmarks["strict_foot_point"] = (10.0, 90.0)
        run_three_d_level_shadow(
            Recording(), landmarks, "mediapipe", "cam_1", 1, 1, "confirmed", 0, None
        )
        self.assertTrue(seen["foot_is_direct"])

    def test_unconfirmed_identity_is_not_treated_as_stable(self):
        from pose_engine import run_three_d_level_shadow

        seen = {}

        class Recording:
            ready = True

            def learn_landmark_heights(self, observation):
                seen.update(observation)
                return {}

            def estimate_ground_position(self, _observation, _state=None):
                from three_d_level import UNAVAILABLE

                return UNAVAILABLE, []

            def known_heights(self, _identity_id):
                return {}

        run_three_d_level_shadow(
            Recording(),
            {"metrology_landmarks": {"nose": (10.0, 20.0)}, "strict_foot_point": (10.0, 90.0)},
            "mediapipe",
            "cam_1",
            1,
            1,
            "provisional",
            0,
            None,
        )
        self.assertFalse(seen["identity_stable"])


if __name__ == "__main__":
    unittest.main()
