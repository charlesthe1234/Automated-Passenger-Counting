import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from main_tracker import process_camera_frame


class FakeCapture:
    def __init__(self, success=True, sequence=8):
        self.success = success
        self.sequence = sequence
        self.prepare_calls = 0

    def read_with_metadata(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8) if self.success else None
        return self.success, frame, 10.0, self.sequence

    def prepare_frame(self, frame):
        self.prepare_calls += 1
        return frame + 1


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def track(self, _frame, **kwargs):
        self.kwargs = kwargs
        # Deliberately has no plot() method: accepted boxes must be drawn by
        # the application so a suppressed shadow can never leak through an
        # Ultralytics pre-rendered frame.
        return [SimpleNamespace()]


class CameraSequenceHandlingTests(unittest.TestCase):
    def make_context(self, success=True):
        return SimpleNamespace(
            cap=FakeCapture(success=success),
            last_capture_sequence=8,
            tactical_points=[(1.0, 2.0)],
            tactical_observations=[{"camera_id": "cam_1", "point": (1.0, 2.0)}],
            raw_frame=None,
            annotated_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        )

    def test_duplicate_cached_frame_preserves_last_observations(self):
        context = self.make_context(success=True)
        self.assertTrue(process_camera_frame(context, 0.4, "cpu"))
        self.assertEqual(context.tactical_points, [(1.0, 2.0)])
        self.assertEqual(len(context.tactical_observations), 1)
        self.assertEqual(context.cap.prepare_calls, 0)

    def test_failed_capture_clears_stale_observations(self):
        context = self.make_context(success=False)
        self.assertFalse(process_camera_frame(context, 0.4, "cpu"))
        self.assertEqual(context.tactical_points, [])
        self.assertEqual(context.tactical_observations, [])
        self.assertEqual(context.cap.prepare_calls, 0)

    def test_tracker_controls_are_forwarded_and_shadow_is_not_published(self):
        context = SimpleNamespace(
            cap=FakeCapture(success=True, sequence=9),
            last_capture_sequence=8,
            tactical_points=[(1.0, 2.0)],
            tactical_observations=[{"camera_id": "cam_1", "point": (1.0, 2.0)}],
            raw_frame=None,
            annotated_frame=None,
            frame_index=0,
            _last_frame_time=None,
            fps=0.0,
            model=FakeModel(),
            use_mediapipe_feet=False,
            pose_estimator=None,
            anatomical_ratio_memory={},
            anatomical_anchor_memory={},
            last_foot_memory={},
            appearance_memory=None,
            camera_id="cam_1",
            homography=np.eye(3, dtype=np.float32),
            map_motion_memory={},
        )
        suppressed_point = {"suppressed": True}

        with patch("main_tracker.get_standing_points", return_value=[suppressed_point]):
            self.assertTrue(
                process_camera_frame(
                    context,
                    0.4,
                    "cpu",
                    nms_iou=0.61,
                    tracker_config="custom_tracker.yaml",
                )
            )

        self.assertEqual(context.model.kwargs["iou"], 0.61)
        self.assertEqual(context.model.kwargs["tracker"], "custom_tracker.yaml")
        self.assertEqual(context.model.kwargs["device"], "cpu")
        self.assertEqual(context.tactical_points, [])
        self.assertEqual(context.tactical_observations, [])
        self.assertIsNotNone(context.annotated_frame)
        self.assertEqual(context.cap.prepare_calls, 1)
        self.assertTrue(np.all(context.raw_frame == 1))


if __name__ == "__main__":
    unittest.main()
