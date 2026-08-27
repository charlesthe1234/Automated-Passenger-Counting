"""Prove the experimental feature cannot affect an ordinary run.

The acceptance rule the operator cares about: with the checkbox unticked, the
system must behave exactly as it did before 3D Level Detection existed, and must
not require any elevated-plane calibration.

These tests exercise the real get_standing_points pipeline rather than asserting
the property in prose.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import ground_point
import pose_engine
from tests.test_three_d_level import synthetic_camera, write_calibration


class _Boxes:
    def __init__(self, boxes, track_ids, confidences):
        self.xyxy = _Tensor(np.asarray(boxes, dtype=float))
        self.id = _Tensor(np.asarray(track_ids)) if track_ids is not None else None
        self.conf = _Tensor(np.asarray(confidences, dtype=float))

    def __len__(self):
        return len(self.xyxy.value)


class _Tensor:
    """Minimal stand-in for the torch tensors Ultralytics returns."""

    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def detach(self):
        return self

    def astype(self, dtype):
        return self.value.astype(dtype)

    def tolist(self):
        return self.value.tolist()

    def __getitem__(self, item):
        return _Tensor(self.value[item])

    def __len__(self):
        return len(self.value)


class _Result:
    def __init__(self, boxes, track_ids, confidences):
        self.boxes = _Boxes(boxes, track_ids, confidences)
        self.keypoints = None


class _Landmark:
    def __init__(self, x, y, visibility=0.99):
        self.x, self.y, self.visibility = x, y, visibility
        self.z = 0.0


class _PoseResult:
    def __init__(self, landmarks):
        self.pose_landmarks = [landmarks]


class _PoseEstimator:
    """Returns a fixed, fully visible skeleton so runs are deterministic."""

    def detect(self, crop):
        landmarks = [_Landmark(0.5, 0.5) for _ in range(33)]
        placement = {
            0: (0.50, 0.08),    # nose
            11: (0.40, 0.22), 12: (0.60, 0.22),   # shoulders
            23: (0.43, 0.52), 24: (0.57, 0.52),   # hips
            25: (0.44, 0.72), 26: (0.56, 0.72),   # knees
            27: (0.45, 0.93), 28: (0.55, 0.93),   # ankles
            29: (0.44, 0.96), 30: (0.56, 0.96),   # heels
            31: (0.46, 0.99), 32: (0.54, 0.99),   # toes
        }
        for index, (x, y) in placement.items():
            landmarks[index] = _Landmark(x, y)
        return _PoseResult(landmarks)


def run_pipeline(three_d_estimator):
    """Run the real get_standing_points and return its output."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = _Result(
        boxes=[[700.0, 300.0, 900.0, 900.0], [1200.0, 350.0, 1380.0, 920.0]],
        track_ids=[4, 9],
        confidences=[0.93, 0.88],
    )
    return pose_engine.get_standing_points(
        result,
        frame,
        pose_estimator=_PoseEstimator(),
        anatomical_ratio_memory={},
        anatomical_anchor_memory={},
        last_foot_memory={},
        frame_index=12,
        camera_id="cam_1",
        observation_time=1234.5,
        use_mediapipe_feet=True,
        map_projector=lambda point: (point[0] * 0.25, point[1] * 0.25),
        map_size_cm=480.0,
        three_d_estimator=three_d_estimator,
    )


class DisabledPathTests(unittest.TestCase):
    def test_pipeline_output_is_identical_with_and_without_the_feature(self):
        """Shadow mode must not alter a single field of the production result."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        projection = synthetic_camera()
        write_calibration(root / "ground.json", projection, 0.0)
        write_calibration(root / "elevated.json", projection, 170.0)
        from three_d_level import ThreeDLevelPositionEstimator

        estimator = ThreeDLevelPositionEstimator().initialize(
            {"cam_1": (root / "ground.json", root / "elevated.json")}
        )

        without = run_pipeline(None)
        with_feature = run_pipeline(estimator)

        self.assertTrue(without, "the pipeline produced no standing points to compare")
        self.assertEqual(len(without), len(with_feature))
        for plain, experimental in zip(without, with_feature):
            self.assertEqual(plain, experimental)

    def test_disabled_run_does_no_metrology_landmark_work(self):
        """The extra landmark extraction must not run on an ordinary frame."""
        with mock.patch.object(
            ground_point, "extract_metrology_landmarks", side_effect=AssertionError("called")
        ) as spy:
            run_pipeline(None)
        spy.assert_not_called()

    def test_enabled_run_does_collect_metrology_landmarks(self):
        """Confirms the previous test is meaningful rather than vacuous."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        projection = synthetic_camera()
        write_calibration(root / "ground.json", projection, 0.0)
        write_calibration(root / "elevated.json", projection, 170.0)
        from three_d_level import ThreeDLevelPositionEstimator

        estimator = ThreeDLevelPositionEstimator().initialize(
            {"cam_1": (root / "ground.json", root / "elevated.json")}
        )
        with mock.patch.object(
            ground_point, "extract_metrology_landmarks", wraps=ground_point.extract_metrology_landmarks
        ) as spy:
            run_pipeline(estimator)
        self.assertTrue(spy.called)

    def test_disabled_run_never_touches_the_experimental_module(self):
        sys.modules.pop("three_d_level", None)
        run_pipeline(None)
        self.assertNotIn("three_d_level", sys.modules)

    def test_disabled_run_needs_no_elevated_calibration_file(self):
        """No elevated calibration exists here at all; the run must still work."""
        for name in ("homography_matrix_elevated.json", "homography_matrix_2_elevated.json"):
            self.assertFalse(
                (Path(pose_engine.__file__).parent / name).exists(),
                f"{name} exists, so this test is not proving anything",
            )
        self.assertTrue(run_pipeline(None))


if __name__ == "__main__":
    unittest.main()
