import unittest
from types import SimpleNamespace

from human_orientation import get_human_orientation
from reid_crop_quality import assess_reid_body_completeness


def landmark(x=0.0, y=0.0, z=0.0, visibility=1.0):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def pose(left_shoulder, right_shoulder, left_hip, right_hip):
    landmarks = [landmark() for _ in range(33)]
    landmarks[11] = left_shoulder
    landmarks[12] = right_shoulder
    landmarks[23] = left_hip
    landmarks[24] = right_hip
    return landmarks


def complete_body_pose():
    landmarks = [landmark(visibility=0.0) for _ in range(33)]
    landmarks[0] = landmark(0.50, 0.08)
    landmarks[11] = landmark(0.42, 0.25)
    landmarks[23] = landmark(0.45, 0.50)
    landmarks[25] = landmark(0.46, 0.70)
    landmarks[27] = landmark(0.47, 0.92)
    return landmarks


class HumanOrientationTests(unittest.TestCase):
    def test_front_and_back_use_strict_parallel_gate(self):
        front = pose(
            landmark(0.8, 0.2), landmark(0.2, 0.2),
            landmark(0.7, 0.5), landmark(0.3, 0.5),
        )
        back = pose(
            landmark(0.2, 0.2), landmark(0.8, 0.2),
            landmark(0.3, 0.5), landmark(0.7, 0.5),
        )
        self.assertEqual(get_human_orientation(front), "front")
        self.assertEqual(get_human_orientation(back), "back")

    def test_side_views_require_narrow_shoulders_before_depth_is_trusted(self):
        left = pose(
            landmark(0.48, 0.2, -0.9), landmark(0.52, 0.2, 0.0),
            landmark(0.48, 0.5), landmark(0.52, 0.5),
        )
        right = pose(
            landmark(0.48, 0.2, 0.9), landmark(0.52, 0.2, 0.0),
            landmark(0.48, 0.5), landmark(0.52, 0.5),
        )
        self.assertEqual(get_human_orientation(left), "left_side")
        self.assertEqual(get_human_orientation(right), "right_side")

    def test_transition_and_low_visibility_are_rejected(self):
        transition = pose(
            landmark(0.35, 0.2, -0.9), landmark(0.65, 0.2, 0.0),
            landmark(0.4, 0.5), landmark(0.6, 0.5),
        )
        hidden = pose(
            landmark(0.8, 0.2, visibility=0.2), landmark(0.2, 0.2),
            landmark(0.7, 0.5), landmark(0.3, 0.5),
        )
        self.assertIsNone(get_human_orientation(transition))
        self.assertIsNone(get_human_orientation(hidden))

    def test_zero_depth_extreme_side_view_is_ambiguous(self):
        ambiguous = pose(
            landmark(0.501, 0.2, 0.0), landmark(0.499, 0.2, 0.0),
            landmark(0.50, 0.5), landmark(0.50, 0.5),
        )
        self.assertIsNone(get_human_orientation(ambiguous))


class ReIDBodyCompletenessTests(unittest.TestCase):
    def test_one_visible_side_is_enough_for_a_complete_body(self):
        complete, missing = assess_reid_body_completeness(complete_body_pose())

        self.assertTrue(complete)
        self.assertEqual(missing, ())

    def test_missing_head_and_shoulders_are_reported(self):
        landmarks = complete_body_pose()
        landmarks[0] = landmark(0.50, 0.08, visibility=0.1)
        landmarks[11] = landmark(0.42, 0.25, visibility=0.1)

        complete, missing = assess_reid_body_completeness(landmarks)

        self.assertFalse(complete)
        self.assertIn("head", missing)
        self.assertIn("shoulders", missing)

    def test_landmark_outside_saved_reid_crop_is_missing(self):
        landmarks = complete_body_pose()

        complete, missing = assess_reid_body_completeness(
            landmarks,
            normalized_bounds=(0.0, 0.10, 1.0, 1.0),
        )

        self.assertFalse(complete)
        self.assertIn("head", missing)

    def test_none_and_short_landmark_sets_are_safe(self):
        for landmarks in (None, [landmark()]):
            complete, missing = assess_reid_body_completeness(landmarks)
            self.assertFalse(complete)
            self.assertTrue(missing)

    def test_front_and_side_views_require_nose_and_one_eye(self):
        landmarks = complete_body_pose()
        complete, missing = assess_reid_body_completeness(landmarks, orientation="front")
        self.assertFalse(complete)
        self.assertIn("head", missing)

        landmarks[2] = landmark(0.48, 0.06)
        complete, missing = assess_reid_body_completeness(landmarks, orientation="left_side")
        self.assertTrue(complete)
        self.assertEqual(missing, ())

    def test_complete_body_can_be_close_to_crop_edges(self):
        landmarks = complete_body_pose()
        landmarks[0] = landmark(0.50, 0.01)
        landmarks[2] = landmark(0.48, 0.01)
        landmarks[27] = landmark(0.47, 0.99)
        complete, missing = assess_reid_body_completeness(landmarks, orientation="front")
        self.assertTrue(complete)
        self.assertEqual(missing, ())

    def test_detection_touching_vertical_frame_boundary_is_rejected(self):
        complete, missing = assess_reid_body_completeness(
            complete_body_pose(),
            touches_vertical_frame_boundary=True,
        )
        self.assertFalse(complete)
        self.assertIn("frame_boundary", missing)

    def test_body_completeness_debug_details_explain_landmark_decision(self):
        landmarks = complete_body_pose()
        details = {}

        complete, missing = assess_reid_body_completeness(
            landmarks,
            debug_details=details,
        )

        self.assertTrue(complete)
        self.assertEqual(missing, ())
        self.assertTrue(details["body_complete"])
        self.assertIn("left_ankle", details["landmarks"])
        self.assertIn("visibility", details["landmarks"]["left_ankle"])
        self.assertTrue(details["landmarks"]["left_ankle"]["within_saved_crop"])


if __name__ == "__main__":
    unittest.main()
