import sys
import unittest


from tests import EDGE_TRACKER_DIR

sys.path.insert(0, str(EDGE_TRACKER_DIR))

from launch_config import (
    build_tracker_arguments,
    dashboard_launch_values,
    default_launch_values,
    redact_tracker_arguments,
)


class LaunchConfigTests(unittest.TestCase):
    def test_dashboard_permanently_disables_map_motion_filter(self):
        values = dashboard_launch_values(
            "managed_run",
            env={"CAMERA_URLS": "cam_1=0,cam_2=1"},
        )
        arguments = build_tracker_arguments(values)

        self.assertEqual(values["run_id"], "managed_run")
        self.assertIn("--disable-map-motion-filter", arguments)

    def test_launcher_default_keeps_speed_hold_checkbox_unticked(self):
        values = default_launch_values({"CAMERA_URLS": "cam_1=0"})
        arguments = build_tracker_arguments(values)

        self.assertNotIn("--disable-map-motion-filter", arguments)

    def test_operator_preset_preserves_current_two_camera_defaults(self):
        values = default_launch_values(
            {
                "CAMERA_URLS": (
                    "cam_1=rtsp://user:secret@192.0.2.1/live,"
                    "cam_2=rtsp://user:secret@192.0.2.2/live"
                )
            }
        )
        arguments = build_tracker_arguments(values)
        self.assertIn("--source-2", arguments)
        self.assertIn("--use-mediapipe-feet", arguments)
        self.assertIn("--use-appearance-reid", arguments)
        self.assertEqual(arguments[arguments.index("--device") + 1], "0")
        self.assertEqual(arguments[arguments.index("--device-2") + 1], "1")
        self.assertEqual(arguments[arguments.index("--reid-device") + 1], "cuda:1")
        self.assertEqual(
            arguments[arguments.index("--mqtt-publish-interval") + 1], "0.2"
        )

    def test_indexed_mediapipe_gpu_is_forwarded(self):
        values = default_launch_values(
            {
                "CAMERA_URLS": "cam_1=0,cam_2=1",
                "CV_MEDIAPIPE_DELEGATE": "gpu:0",
            }
        )
        arguments = build_tracker_arguments(values)
        self.assertEqual(
            arguments[arguments.index("--mediapipe-delegate") + 1], "gpu:0"
        )

    def test_legacy_yolo_device_remains_the_fallback_for_both_cameras(self):
        values = default_launch_values(
            {
                "CAMERA_URLS": "cam_1=0,cam_2=1",
                "CV_YOLO_DEVICE": "cpu",
            }
        )
        arguments = build_tracker_arguments(values)
        self.assertEqual(arguments[arguments.index("--device") + 1], "cpu")
        self.assertEqual(arguments[arguments.index("--device-2") + 1], "cpu")

    def test_camera_two_only_uses_its_own_device_as_primary(self):
        values = default_launch_values({"CAMERA_URLS": "cam_1=0,cam_2=1"})
        values["camera_mode"] = "camera_2"
        values["yolo_device_2"] = "1"
        arguments = build_tracker_arguments(values)
        self.assertEqual(arguments[arguments.index("--device") + 1], "1")
        self.assertNotIn("--device-2", arguments)

    def test_missing_camera_configuration_is_rejected(self):
        values = default_launch_values({})
        with self.assertRaisesRegex(ValueError, "Camera 1 source"):
            build_tracker_arguments(values)

    def test_credentials_are_redacted_from_display_arguments(self):
        arguments = [
            "--source",
            "rtsp://admin:secret@192.0.2.1/live",
            "--mqtt-password",
            "broker-secret",
        ]
        safe = redact_tracker_arguments(arguments)
        self.assertNotIn("secret", " ".join(safe))
        self.assertIn("<credentials>@192.0.2.1", safe[1])


if __name__ == "__main__":
    unittest.main()
