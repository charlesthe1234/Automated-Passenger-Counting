"""Regression tests for locating a face inside a ReID crop from pose landmarks."""

import unittest

from constants import MIN_MEDIAPIPE_VISIBILITY
from face_region import estimate_face_box, normalized_box_to_pixels


class _Landmark:
    def __init__(self, x, y, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.visibility = float(visibility)


def _landmarks(**overrides):
    """A 33-landmark pose with everything invisible except what is named.

    Indices follow MediaPipe: 0 nose, 2/5 eyes, 7/8 ears, 9/10 mouth corners.
    """
    points = [_Landmark(0.0, 0.0, 0.0) for _ in range(33)]
    names = {
        "nose": 0,
        "left_eye": 2,
        "right_eye": 5,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
    }
    for name, value in overrides.items():
        x, y = value
        points[names[name]] = _Landmark(x, y, 1.0)
    return points


def _centre_and_size(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), (x2 - x1, y2 - y1)


class FaceBoxFromLandmarksTests(unittest.TestCase):
    def test_a_frontal_head_is_sized_from_the_ear_span(self):
        # Ears 0.10 apart on a square crop: head width 0.115, box 1.5x that.
        landmarks = _landmarks(
            nose=(0.50, 0.20),
            left_eye=(0.47, 0.19),
            right_eye=(0.53, 0.19),
            left_ear=(0.45, 0.20),
            right_ear=(0.55, 0.20),
        )
        box = estimate_face_box(landmarks, crop_width=100, crop_height=100)
        self.assertIsNotNone(box)
        centre, (width, height) = _centre_and_size(box)
        self.assertAlmostEqual(width, 0.10 * 1.15 * 1.5, places=5)
        self.assertAlmostEqual(height, width, places=5)
        self.assertAlmostEqual(centre[0], 0.50, places=5)
        self.assertAlmostEqual(centre[1], 0.20, places=5)

    def test_the_box_is_square_in_pixels_not_in_fractions(self):
        # A person-shaped crop is far taller than it is wide, so an equal span
        # of x and y fractions would be a tall rectangle over the face.
        landmarks = _landmarks(
            nose=(0.50, 0.10),
            left_ear=(0.40, 0.10),
            right_ear=(0.60, 0.10),
        )
        box = estimate_face_box(landmarks, crop_width=100, crop_height=300)
        self.assertIsNotNone(box)
        _, (width, height) = _centre_and_size(box)
        self.assertAlmostEqual(width * 100, height * 300, places=3)

    def test_eyes_alone_still_size_a_box(self):
        landmarks = _landmarks(nose=(0.50, 0.20), left_eye=(0.46, 0.19), right_eye=(0.54, 0.19))
        box = estimate_face_box(landmarks, crop_width=100, crop_height=100)
        self.assertIsNotNone(box)
        _, (width, _height) = _centre_and_size(box)
        self.assertAlmostEqual(width, 0.08 * 2.85 * 1.5, places=5)

    def test_a_profile_view_sizes_from_one_eye_and_one_ear(self):
        # Turned away: the far eye and far ear are both hidden, which is
        # exactly the case the frontal cascade this replaces could not read.
        landmarks = _landmarks(nose=(0.50, 0.20), left_eye=(0.47, 0.19), left_ear=(0.40, 0.20))
        box = estimate_face_box(landmarks, crop_width=100, crop_height=100)
        self.assertIsNotNone(box)
        _, (width, _height) = _centre_and_size(box)
        self.assertGreater(width, 0.0)

    def test_mouth_corners_are_the_last_resort(self):
        landmarks = _landmarks(mouth_left=(0.48, 0.22), mouth_right=(0.52, 0.22))
        box = estimate_face_box(landmarks, crop_width=100, crop_height=100)
        self.assertIsNotNone(box)
        centre, _size = _centre_and_size(box)
        # No nose, so the box centres on what is visible.
        self.assertAlmostEqual(centre[0], 0.50, places=5)

    def test_a_nose_alone_cannot_size_a_box(self):
        # One point gives a position but no scale, and a guessed scale would
        # hand MiVOLO either a crop of one nostril or half the torso.
        box = estimate_face_box(_landmarks(nose=(0.5, 0.2)), crop_width=100, crop_height=100)
        self.assertIsNone(box)

    def test_no_visible_head_landmarks_returns_none(self):
        self.assertIsNone(estimate_face_box(_landmarks(), crop_width=100, crop_height=100))
        self.assertIsNone(estimate_face_box(None, crop_width=100, crop_height=100))
        self.assertIsNone(estimate_face_box([], crop_width=100, crop_height=100))

    def test_landmarks_below_the_visibility_floor_are_ignored(self):
        landmarks = _landmarks(nose=(0.50, 0.20), left_ear=(0.45, 0.20), right_ear=(0.55, 0.20))
        for index in (7, 8):
            landmarks[index].visibility = MIN_MEDIAPIPE_VISIBILITY - 0.01
        self.assertIsNone(estimate_face_box(landmarks, crop_width=100, crop_height=100))

    def test_a_non_finite_landmark_is_ignored(self):
        landmarks = _landmarks(nose=(0.50, 0.20), left_ear=(0.45, 0.20), right_ear=(0.55, 0.20))
        landmarks[7] = _Landmark(float("nan"), 0.20, 1.0)
        self.assertIsNone(estimate_face_box(landmarks, crop_width=100, crop_height=100))


class FaceBoxCoordinateSpaceTests(unittest.TestCase):
    """The landmarks are measured on the pose crop, which is not the saved crop."""

    def test_the_box_is_remapped_onto_the_saved_reid_crop(self):
        # The pose crop is padded wider than the ReID crop, so the ReID crop
        # occupies the middle half of it.  A face at the pose crop's centre
        # must land at the ReID crop's centre, not at 0.5 of the wrong image.
        landmarks = _landmarks(
            nose=(0.50, 0.30),
            left_ear=(0.45, 0.30),
            right_ear=(0.55, 0.30),
        )
        bounds = (0.25, 0.20, 0.75, 0.80)
        box = estimate_face_box(
            landmarks,
            normalized_bounds=bounds,
            crop_width=100,
            crop_height=100,
        )
        self.assertIsNotNone(box)
        centre, (width, _height) = _centre_and_size(box)
        self.assertAlmostEqual(centre[0], (0.50 - 0.25) / 0.50, places=5)
        self.assertAlmostEqual(centre[1], (0.30 - 0.20) / 0.60, places=5)
        # Half the width of the pose crop means twice the fraction of the
        # ReID crop.
        self.assertAlmostEqual(width, 0.10 * 1.15 * 1.5 / 0.50, places=5)

    def test_degenerate_bounds_are_refused(self):
        landmarks = _landmarks(nose=(0.5, 0.3), left_ear=(0.45, 0.3), right_ear=(0.55, 0.3))
        self.assertIsNone(
            estimate_face_box(landmarks, normalized_bounds=(0.5, 0.0, 0.5, 1.0))
        )

    def test_a_face_at_the_crop_edge_is_clamped_not_discarded(self):
        # Half a face at the edge of the crop still carries real signal.
        landmarks = _landmarks(nose=(0.02, 0.05), left_ear=(0.0, 0.05), right_ear=(0.06, 0.05))
        box = estimate_face_box(landmarks, crop_width=100, crop_height=100)
        self.assertIsNotNone(box)
        self.assertEqual(box[0], 0.0)
        self.assertEqual(box[1], 0.0)
        self.assertGreater(box[2], 0.0)

    def test_a_face_entirely_outside_the_saved_crop_returns_none(self):
        # MediaPipe found a head in the pose crop's padding, outside the part
        # actually saved as the ReID crop.  There is no face in the image the
        # demographics worker will hold, so there is no box to report.
        landmarks = _landmarks(nose=(0.05, 0.05), left_ear=(0.03, 0.05), right_ear=(0.07, 0.05))
        box = estimate_face_box(
            landmarks,
            normalized_bounds=(0.50, 0.50, 1.0, 1.0),
            crop_width=100,
            crop_height=100,
        )
        self.assertIsNone(box)


class NormalizedBoxToPixelsTests(unittest.TestCase):
    def test_a_box_converts_to_inclusive_exclusive_pixel_bounds(self):
        self.assertEqual(normalized_box_to_pixels((0.25, 0.5, 0.75, 1.0), 100, 200), (25, 100, 75, 200))

    def test_a_box_that_rounds_away_to_nothing_returns_none(self):
        self.assertIsNone(normalized_box_to_pixels((0.5, 0.5, 0.5001, 0.5001), 10, 10))

    def test_a_missing_box_or_empty_crop_returns_none(self):
        self.assertIsNone(normalized_box_to_pixels(None, 10, 10))
        self.assertIsNone(normalized_box_to_pixels((0.0, 0.0, 1.0, 1.0), 0, 10))


if __name__ == "__main__":
    unittest.main()
