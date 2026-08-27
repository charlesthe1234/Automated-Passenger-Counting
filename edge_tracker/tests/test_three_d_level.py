"""Tests for the experimental two-plane metrology module.

The geometry is checked against a synthetic camera whose projection matrix is
known exactly, so every claim about recovered heights and ground positions is
verified rather than assumed.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from three_d_level import (
    CalibrationError,
    PositionEstimate,
    ThreeDLevelPositionEstimator,
    build_plane_calibration,
    expected_vertical_direction,
    ground_from_landmark,
    height_from_pixels,
    lean_degrees,
)


def synthetic_camera(tilt_degrees=58.0, centre=(120.0, -260.0, 320.0), yaw_degrees=0.0):
    """A tent-height camera looking down at the calibrated floor square."""
    intrinsics = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    tilt = np.deg2rad(tilt_degrees)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt), -np.sin(tilt)],
            [0.0, np.sin(tilt), np.cos(tilt)],
        ]
    )
    if yaw_degrees:
        yaw = np.deg2rad(yaw_degrees)
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
        ) @ rotation
    position = np.asarray(centre, dtype=float)
    return intrinsics @ np.hstack([rotation, (-rotation @ position).reshape(3, 1)])


def project(projection, x, y, z):
    homogeneous = projection @ np.array([x, y, z, 1.0])
    return homogeneous[:2] / homogeneous[2]


def plane_world_to_image(projection, height):
    return projection[:, [0, 1, 3]] + height * np.outer(projection[:, 2], (0.0, 0.0, 1.0))


def write_calibration(path, projection, height_cm, map_size_cm=480.0, scale=1.0, plane_height_key=True):
    """Save a calibration in the project's image->world form at an arbitrary scale."""
    world_to_image = plane_world_to_image(projection, height_cm)
    image_to_world = np.linalg.inv(world_to_image) * scale
    payload = {"matrix": image_to_world.tolist(), "map_size_cm": map_size_cm}
    if plane_height_key and height_cm > 0:
        payload["plane_height_cm"] = height_cm
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    return payload


class TwoPlaneGeometryTests(unittest.TestCase):
    def setUp(self):
        self.projection = synthetic_camera()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.elevated_height = 170.0
        # Deliberately different projective scales: raw OpenCV output is only
        # defined up to scale and must not be differenced directly.
        write_calibration(self.root / "ground.json", self.projection, 0.0, scale=0.37)
        write_calibration(
            self.root / "elevated.json", self.projection, self.elevated_height, scale=4.1
        )
        self.calibration = build_plane_calibration(
            "cam_1", self.root / "ground.json", self.root / "elevated.json"
        )

    def test_scale_reconciliation_makes_the_two_calibrations_consistent(self):
        self.assertLess(self.calibration.residual, 1e-9)

    def test_recovers_the_true_vertical_vanishing_point(self):
        recovered = self.calibration.vertical_vanishing_point
        truth = self.projection[:, 2]
        np.testing.assert_allclose(recovered / recovered[2], truth / truth[2], atol=1e-6)

    def test_homography_at_arbitrary_height_matches_the_true_projection(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            x, y, z = rng.uniform(0, 480), rng.uniform(0, 480), rng.uniform(0, 200)
            expected = project(self.projection, x, y, z)
            homography = self.calibration.homography_at_height(z)
            got = homography @ np.array([x, y, 1.0])
            np.testing.assert_allclose(got[:2] / got[2], expected, atol=1e-6)

    def test_recovers_landmark_height_from_foot_and_landmark_pixels(self):
        rng = np.random.default_rng(5)
        for _ in range(200):
            x, y, z = rng.uniform(0, 480), rng.uniform(0, 480), rng.uniform(40, 180)
            foot = project(self.projection, x, y, 0.0)
            landmark = project(self.projection, x, y, z)
            self.assertAlmostEqual(height_from_pixels(self.calibration, foot, landmark), z, places=5)

    def test_recovers_ground_position_from_a_landmark_at_known_height(self):
        rng = np.random.default_rng(9)
        for _ in range(200):
            x, y, z = rng.uniform(0, 480), rng.uniform(0, 480), rng.uniform(20, 180)
            landmark = project(self.projection, x, y, z)
            point = ground_from_landmark(self.calibration, landmark, z)
            np.testing.assert_allclose(point, (x, y), atol=1e-6)

    def test_mismeasured_plane_height_biases_heights_but_not_ground_position(self):
        """A wrong elevated height cancels out because learning and applying share it."""
        write_calibration(
            self.root / "wrong.json", self.projection, self.elevated_height, scale=1.0
        )
        payload = json.loads((self.root / "wrong.json").read_text())
        payload["plane_height_cm"] = self.elevated_height + 8.5  # mis-measured tape
        (self.root / "wrong.json").write_text(json.dumps(payload), encoding="utf-8")
        wrong = build_plane_calibration("cam_1", self.root / "ground.json", self.root / "wrong.json")

        x, y, true_height = 240.0, 180.0, 92.0
        foot = project(self.projection, x, y, 0.0)
        landmark = project(self.projection, x, y, true_height)

        learned = height_from_pixels(wrong, foot, landmark)
        self.assertGreater(abs(learned - true_height), 3.0)  # height is biased

        recovered = ground_from_landmark(wrong, landmark, learned)
        np.testing.assert_allclose(recovered, (x, y), atol=1e-6)  # position is not

    def test_expected_vertical_direction_is_not_the_image_y_axis(self):
        """Perspective tilts world-vertical differently across the frame."""
        left = expected_vertical_direction(self.calibration, (200.0, 900.0))
        right = expected_vertical_direction(self.calibration, (1700.0, 900.0))
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertGreater(float(np.linalg.norm(left - right)), 0.05)

    def test_upright_person_reads_as_no_lean(self):
        x, y = 240.0, 200.0
        foot = project(self.projection, x, y, 0.0)
        head = project(self.projection, x, y, 165.0)
        self.assertLess(lean_degrees(self.calibration, foot, head), 1e-3)

    def test_lean_across_the_viewing_azimuth_is_detected(self):
        """Sideways lean corrupts the measured height, so it must be visible."""
        foot = project(self.projection, 240.0, 200.0, 0.0)
        head = project(self.projection, 275.0, 200.0, 160.0)  # 35 cm sideways
        self.assertGreater(lean_degrees(self.calibration, foot, head), 5.0)
        corrupted = height_from_pixels(self.calibration, foot, head)
        self.assertGreater(abs(corrupted - 160.0), 30.0)

    def test_lean_along_the_viewing_azimuth_is_invisible_but_also_harmless(self):
        """Documents a real blind spot, and why it does not matter.

        Lean toward or away from the camera barely changes the angle between the
        torso and the projected vertical, so the gate cannot see it. It also
        barely changes the recovered height, so there is nothing to catch.
        """
        foot = project(self.projection, 240.0, 200.0, 0.0)
        for offset in (35.0, -35.0):
            head = project(self.projection, 240.0, 200.0 + offset, 160.0)
            self.assertLess(lean_degrees(self.calibration, foot, head), 3.0)
            self.assertAlmostEqual(height_from_pixels(self.calibration, foot, head), 160.0, delta=2.0)


class CalibrationValidationTests(unittest.TestCase):
    def setUp(self):
        self.projection = synthetic_camera()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_calibration(self.root / "ground.json", self.projection, 0.0)

    def test_missing_file_reports_clearly(self):
        with self.assertRaises(CalibrationError) as caught:
            build_plane_calibration("cam_2", self.root / "ground.json", self.root / "absent.json")
        self.assertIn("not found", str(caught.exception))

    def test_malformed_json_reports_clearly(self):
        (self.root / "bad.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(CalibrationError):
            build_plane_calibration("cam_2", self.root / "ground.json", self.root / "bad.json")

    def test_missing_plane_height_is_rejected(self):
        write_calibration(
            self.root / "noheight.json", self.projection, 170.0, plane_height_key=False
        )
        with self.assertRaises(CalibrationError) as caught:
            build_plane_calibration("cam_2", self.root / "ground.json", self.root / "noheight.json")
        self.assertIn("plane_height_cm", str(caught.exception))

    def test_mismatched_map_size_is_rejected(self):
        write_calibration(self.root / "othermap.json", self.projection, 170.0, map_size_cm=600.0)
        with self.assertRaises(CalibrationError) as caught:
            build_plane_calibration("cam_2", self.root / "ground.json", self.root / "othermap.json")
        self.assertIn("world coordinate system", str(caught.exception))

    def test_camera_moved_between_calibrations_is_rejected(self):
        moved = synthetic_camera(yaw_degrees=1.5)
        write_calibration(self.root / "moved.json", moved, 170.0)
        with self.assertRaises(CalibrationError) as caught:
            build_plane_calibration("cam_2", self.root / "ground.json", self.root / "moved.json")
        self.assertIn("residual", str(caught.exception))

    def test_small_camera_movement_raises_the_residual(self):
        """The residual must grow with movement so a threshold is meaningful."""
        residuals = []
        for yaw in (0.0, 0.05, 0.2):
            moved = synthetic_camera(yaw_degrees=yaw)
            path = self.root / f"yaw_{yaw}.json"
            write_calibration(path, moved, 170.0)
            calibration = build_plane_calibration(
                "cam_2", self.root / "ground.json", path, residual_tolerance=1.0
            )
            residuals.append(calibration.residual)
        self.assertLess(residuals[0], residuals[1])
        self.assertLess(residuals[1], residuals[2])


class HeightLearningTests(unittest.TestCase):
    def setUp(self):
        self.projection = synthetic_camera()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_calibration(self.root / "ground.json", self.projection, 0.0)
        write_calibration(self.root / "elevated.json", self.projection, 170.0)
        self.estimator = ThreeDLevelPositionEstimator(min_height_samples=5)
        self.estimator.initialize(
            {"cam_1": (self.root / "ground.json", self.root / "elevated.json")}
        )
        self.truth = {"knee_centre": 48.0, "hip_centre": 92.0, "shoulder_centre": 140.0, "nose": 162.0}

    def observation(self, x=240.0, y=200.0, direct=True, stable=True, identity_id=7, lean_cm=0.0):
        landmarks = {
            name: tuple(project(self.projection, x + lean_cm * (height / 162.0), y, height))
            for name, height in self.truth.items()
        }
        return {
            "camera_id": "cam_1",
            "identity_id": identity_id,
            "foot_pixel": tuple(project(self.projection, x, y, 0.0)),
            "landmarks": landmarks,
            "foot_is_direct": direct,
            "identity_stable": stable,
        }

    def test_learns_metric_heights_from_direct_foot_frames(self):
        learned = {}
        for step in range(12):
            learned = self.estimator.learn_landmark_heights(
                self.observation(x=200.0 + 8 * step, y=180.0 + 5 * step)
            )
        for name, height in self.truth.items():
            self.assertAlmostEqual(learned[name], height, delta=0.5)

    def test_does_not_learn_without_a_direct_foot_observation(self):
        for _ in range(12):
            self.estimator.learn_landmark_heights(self.observation(direct=False))
        self.assertEqual(self.estimator.known_heights(7), {})

    def test_does_not_learn_from_an_unstable_identity(self):
        for _ in range(12):
            self.estimator.learn_landmark_heights(self.observation(stable=False))
        self.assertEqual(self.estimator.known_heights(7), {})

    def test_does_not_learn_while_bending(self):
        for _ in range(12):
            self.estimator.learn_landmark_heights(self.observation(lean_cm=45.0))
        self.assertEqual(self.estimator.known_heights(7), {})

    def test_ordinary_walking_lean_still_converges_on_the_true_height(self):
        """Walkers lean both ways, so the median window absorbs it.

        Rejecting every frame with a few degrees of lean would discard most
        usable samples, so the gate is deliberately loose and the robust
        estimator does the work instead.
        """
        learned = {}
        for step in range(40):
            sway = 14.0 * np.sin(step * 0.7)
            learned = self.estimator.learn_landmark_heights(
                self.observation(x=200.0 + 5 * step, lean_cm=sway)
            )
        # The knee, being closest to the ground, is barely affected.
        self.assertAlmostEqual(learned["knee_centre"], self.truth["knee_centre"], delta=3.0)
        # Residual bias grows with landmark height. This is the whole reason the
        # estimator prefers the lowest available landmark, so assert the ordering
        # rather than a tolerance that would silently drift.
        errors = [abs(learned[name] - self.truth[name]) for name in
                  ("knee_centre", "hip_centre", "shoulder_centre", "nose")]
        self.assertEqual(errors, sorted(errors))

    def test_implausible_heights_are_discarded(self):
        """A wrong identity association must not poison the stored height."""
        for _ in range(12):
            self.estimator.learn_landmark_heights(self.observation())
        before = self.estimator.known_heights(7)["nose"]
        bad = self.observation()
        bad["landmarks"]["nose"] = tuple(project(self.projection, 240.0, 200.0, 400.0))
        for _ in range(10):
            self.estimator.learn_landmark_heights(bad)
        self.assertAlmostEqual(self.estimator.known_heights(7)["nose"], before, delta=0.5)

    def test_estimates_ground_position_once_heights_are_known(self):
        for step in range(12):
            self.estimator.learn_landmark_heights(self.observation(x=200.0 + 8 * step))
        hidden = self.observation(x=300.0, y=260.0)
        hidden["foot_pixel"] = None  # feet now occluded
        best, candidates = self.estimator.estimate_ground_position(hidden, identity_state="confirmed")
        self.assertIsNotNone(best.point_cm)
        np.testing.assert_allclose(best.point_cm, (300.0, 260.0), atol=1.5)
        self.assertEqual(len(candidates), 4)

    def test_returns_unavailable_when_no_height_has_been_learned(self):
        best, candidates = self.estimator.estimate_ground_position(self.observation(identity_id=99))
        self.assertIsNone(best.point_cm)
        self.assertEqual(best.source, "unavailable")
        self.assertEqual(best.evidence, "none")
        self.assertEqual(candidates, [])

    def test_lowest_landmark_is_preferred(self):
        for step in range(12):
            self.estimator.learn_landmark_heights(self.observation(x=200.0 + 8 * step))
        best, _ = self.estimator.estimate_ground_position(self.observation(), identity_state="confirmed")
        self.assertEqual(best.landmark, "knee_centre")
        self.assertEqual(best.source, "learned_knee_height")

    def test_unconfirmed_identity_lowers_confidence(self):
        for step in range(12):
            self.estimator.learn_landmark_heights(self.observation(x=200.0 + 8 * step))
        confirmed, _ = self.estimator.estimate_ground_position(
            self.observation(), identity_state="confirmed"
        )
        provisional, _ = self.estimator.estimate_ground_position(
            self.observation(), identity_state="provisional"
        )
        self.assertLess(provisional.confidence, confirmed.confidence)

    def test_forgetting_an_identity_clears_its_heights(self):
        for step in range(12):
            self.estimator.learn_landmark_heights(self.observation(x=200.0 + 8 * step))
        self.estimator.forget_identity(7)
        self.assertEqual(self.estimator.known_heights(7), {})

    def test_close_releases_all_state(self):
        for step in range(12):
            self.estimator.learn_landmark_heights(self.observation(x=200.0 + 8 * step))
        self.estimator.close()
        self.assertFalse(self.estimator.ready)
        self.assertEqual(self.estimator.heights, {})
        self.assertEqual(self.estimator.known_heights(7), {})


class EvidencePolicyTests(unittest.TestCase):
    def test_direct_feet_are_hard_evidence(self):
        estimate = PositionEstimate((10.0, 20.0), "direct_feet", 0.95, 8.0)
        self.assertEqual(estimate.evidence, "hard")

    def test_extrapolated_sources_are_never_hard(self):
        for source in ("anatomical_ratio", "last_seen", "box_bottom", "learned_nose_height"):
            self.assertEqual(PositionEstimate((0.0, 0.0), source, 0.5, 40.0).evidence, "soft")

    def test_unavailable_carries_no_evidence(self):
        self.assertEqual(PositionEstimate(None, "unavailable", 0.0, float("inf")).evidence, "none")


if __name__ == "__main__":
    unittest.main()
