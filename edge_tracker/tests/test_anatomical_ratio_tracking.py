"""Anatomical-ratio foot estimation: the fallback used when feet are occluded.

When a person walks behind someone else their feet vanish, so the tactical map
point is reconstructed from the head-to-shoulder anchor and a learned body
ratio.  These tests pin that reconstruction, the quality gates that decide when
the ratio may be learned, and the physics hold that rejects a teleporting foot.

Position memories are addressed through ``resolve_memory_key``, which namespaces
entries as ``("track", id)`` or ``("identity", id)``.  Tests use that helper
rather than raw ids: seeding a raw id would leave the assertion passing
vacuously, because the production write would land on a different key.

An earlier test here covered synchronous single-crop ID reuse in
AppearanceIdentityMemory.  That behaviour was replaced by the asynchronous
five-crop intake, and its successor is
``test_reid_intake_lifecycle.test_new_local_id_reuses_existing_master_through_all_slot_matcher``.
"""

import unittest

import numpy as np

from core_math import (
    calculate_anatomical_ratio,
    estimate_virtual_foot_from_ratio,
    extrapolate_fourth_corner,
    resolve_memory_key,
    store_anatomical_ratio,
)
from ground_point import estimate_mediapipe_foot_point


class FakeLandmark:
    def __init__(self, x=0.0, y=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.visibility = visibility


class FakePoseResult:
    def __init__(self, landmarks):
        self.pose_landmarks = [landmarks]


class FakeEmptyPoseResult:
    pose_landmarks = []


class FakePoseEstimator:
    def __init__(self, results):
        self.results = list(results)

    def detect(self, _crop):
        return self.results.pop(0)


def make_landmarks(crop_width, crop_height, points):
    landmarks = [FakeLandmark() for _ in range(33)]
    for index, point in points.items():
        if len(point) == 3:
            x, y, visibility = point
        else:
            x, y = point
            visibility = 0.95
        landmarks[index] = FakeLandmark(
            x=x / crop_width,
            y=y / crop_height,
            visibility=visibility,
        )
    return landmarks


class AnatomicalRatioTrackingTest(unittest.TestCase):
    def test_ratio_can_reconstruct_scaled_virtual_foot(self):
        initial_nose = np.array([100.0, 50.0])
        initial_shoulder = np.array([100.0, 70.0])
        initial_foot = np.array([100.0, 200.0])

        ratio = calculate_anatomical_ratio(initial_nose, initial_shoulder, initial_foot)

        current_nose = np.array([250.0, 80.0])
        current_shoulder = np.array([250.0, 90.0])
        virtual_foot = estimate_virtual_foot_from_ratio(current_nose, current_shoulder, ratio)

        self.assertAlmostEqual(ratio, 20.0 / 130.0)
        self.assertTrue(np.allclose(virtual_foot, np.array([250.0, 155.0])))

    def test_virtual_foot_projects_perpendicular_to_shoulder_line(self):
        nose = np.array([100.0, 50.0])
        shoulder = np.array([100.0, 70.0])
        left_shoulder = np.array([90.0, 75.0])
        right_shoulder = np.array([110.0, 65.0])
        ratio = 20.0 / 130.0

        virtual_foot = estimate_virtual_foot_from_ratio(
            nose,
            shoulder,
            ratio,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
        )

        shoulder_vector = right_shoulder - left_shoulder
        projected_vector = virtual_foot - shoulder
        self.assertGreater(virtual_foot[1], shoulder[1])
        self.assertNotAlmostEqual(virtual_foot[0], shoulder[0])
        self.assertAlmostEqual(float(np.dot(shoulder_vector, projected_vector)), 0.0)

    def test_invalid_short_anchor_does_not_update_ratio(self):
        nose = np.array([100.0, 50.0])
        shoulder = np.array([100.0, 52.0])
        foot = np.array([100.0, 200.0])

        self.assertIsNone(calculate_anatomical_ratio(nose, shoulder, foot))

    def test_ratio_above_strict_upper_bound_is_rejected(self):
        nose = np.array([100.0, 50.0])
        shoulder = np.array([100.0, 70.0])
        foot = np.array([100.0, 150.0])

        self.assertIsNone(calculate_anatomical_ratio(nose, shoulder, foot))

    def test_ratio_storage_uses_ema_instead_of_overwrite(self):
        memory = {}
        store_anatomical_ratio(memory, 1, 0.10)
        store_anatomical_ratio(memory, 1, 0.20)

        self.assertAlmostEqual(memory[resolve_memory_key(1)], 0.11)

    def test_memory_keys_are_namespaced_by_track_or_identity(self):
        """Pin the key encoding once, so the other tests can use the helper."""
        self.assertEqual(resolve_memory_key(7), ("track", 7))
        self.assertEqual(resolve_memory_key(7, identity_id=3), ("identity", 3))
        self.assertIsNone(resolve_memory_key(None))

        memory = {}
        store_anatomical_ratio(memory, 7, 0.10)
        store_anatomical_ratio(memory, 7, 0.10, identity_id=3)

        # A track and an identity that happen to share a number must not
        # collide, which is why raw ids are no longer used as keys.
        self.assertEqual(sorted(memory, key=repr), [("identity", 3), ("track", 7)])

    def test_mediapipe_ratio_update_uses_slow_ema_when_head_straight_and_feet_clear(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {resolve_memory_key(5): 0.10}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=5,
            frame_index=1,
        )

        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertAlmostEqual(
            memory[resolve_memory_key(5)], 0.10 * 0.98 + (20.0 / 130.0) * 0.02
        )

    def test_low_confidence_feet_do_not_update_ratio_memory(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {resolve_memory_key(6): 0.10}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0, 0.60),
                29: (50.0, 200.0, 0.60),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=6,
            frame_index=1,
        )

        # Ankles at 0.60 clear MIN_MEDIAPIPE_VISIBILITY but not the stricter
        # MIN_INITIAL_FOOT_VISIBILITY: MediaPipe placed a foot it is not
        # confident about.  A ratio learned from frames where the feet *were*
        # clearly visible beats it, and is certainly not rewritten by it.
        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 270.0])))
        self.assertAlmostEqual(memory[resolve_memory_key(6)], 0.10)

    def test_low_confidence_feet_are_still_used_when_no_ratio_is_learned_yet(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0, 0.60),
                29: (50.0, 200.0, 0.60),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=6,
            frame_index=1,
        )

        # An uncertain measurement still beats nothing at all: with no ratio
        # to fall back on, dropping this person would take them out of the
        # cross-camera matcher entirely.
        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertEqual(memory, {})

    def test_mediapipe_clear_view_stores_ratio_then_occlusion_uses_track_memory(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])

        clear_view_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        occluded_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 80.0),
                11: (50.0, 90.0),
                12: (50.0, 90.0),
            },
        )
        estimator = FakePoseEstimator([
            FakePoseResult(clear_view_landmarks),
            FakePoseResult(occluded_landmarks),
        ])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=42,
            frame_index=1,
        )
        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertAlmostEqual(memory[resolve_memory_key(42)], 20.0 / 130.0)
        self.assertTrue(
            np.allclose(
                last_foot_memory[resolve_memory_key(42)]["point"],
                np.array([50.0, 200.0]),
            )
        )

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=42,
            frame_index=2,
        )
        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 155.0])))

    def test_pose_dropout_reuses_recent_foot_for_same_track_id(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])

        clear_view_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 50.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        estimator = FakePoseEstimator([
            FakePoseResult(clear_view_landmarks),
            FakeEmptyPoseResult(),
        ])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=7,
            frame_index=10,
            pose_dropout_ttl_frames=30,
        )
        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=7,
            frame_index=20,
            pose_dropout_ttl_frames=30,
        )
        self.assertEqual(method, "last_seen")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))

    def test_missing_ankle_does_not_count_as_real_foot(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {resolve_memory_key(9): 20.0 / 130.0}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 80.0),
                11: (50.0, 90.0),
                12: (50.0, 90.0),
                29: (50.0, 190.0),
                31: (50.0, 200.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=9,
            frame_index=1,
        )

        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 155.0])))

    def test_missing_ankle_does_not_fallback_to_box_bottom(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 80.0),
                11: (50.0, 90.0),
                12: (50.0, 90.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=99,
            frame_index=1,
        )

        self.assertIsNone(foot)
        self.assertEqual(method, "no_visible_ankle")
        self.assertNotIn(99, memory)

    def test_looking_down_uses_virtual_foot_and_freezes_ratio_memory(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {resolve_memory_key(8): 20.0 / 130.0}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 80.0),
                2: (45.0, 50.0),
                5: (55.0, 50.0),
                11: (50.0, 100.0),
                12: (50.0, 100.0),
                27: (50.0, 190.0),
                29: (50.0, 210.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=8,
            frame_index=1,
        )

        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 230.0])))
        # A head-down pose must never be used to relearn the ratio.
        self.assertAlmostEqual(memory[resolve_memory_key(8)], 20.0 / 130.0)

    def test_head_down_first_sighting_never_learns_a_ratio(self):
        """Feet visible but head down, and nothing learned yet.

        The earlier head-down test returns through the stored-ratio branch, so
        it never reaches the gate that decides whether a ratio may be learned.
        A foreshortened head-to-shoulder anchor would bake in a permanently
        wrong body scale, so the real foot is used and nothing is stored.
        """
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 80.0),    # nose well below the eye line
                2: (45.0, 50.0),
                5: (55.0, 50.0),
                11: (50.0, 95.0),
                12: (50.0, 95.0),
                27: (50.0, 215.0),  # feet fully visible and sharp
                29: (50.0, 222.0),
                31: (50.0, 228.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=77,
            frame_index=1,
        )

        # The measured foot is trustworthy; only the ratio update is refused.
        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 225.0])))
        # 15/130 would sit inside the valid band, so only the head-pitch gate
        # stops it from being learned.
        self.assertEqual(memory, {})

    def test_looking_down_uses_saved_head_up_anchor_instead_of_shrunken_anchor(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        ratio_memory = {resolve_memory_key(12): 20.0 / 130.0}
        anchor_memory = {resolve_memory_key(12): 20.0}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 220.0])
        landmarks = make_landmarks(
            crop_width=112,
            crop_height=237,
            points={
                0: (50.0, 94.0),
                2: (45.0, 70.0),
                5: (55.0, 70.0),
                11: (50.0, 100.0),
                12: (50.0, 100.0),
                27: (50.0, 190.0),
                29: (50.0, 210.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=ratio_memory,
            anatomical_anchor_memory=anchor_memory,
            last_foot_memory=last_foot_memory,
            track_id=12,
            frame_index=1,
        )

        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 230.0])))
        # The shrunken head-down anchor must not overwrite the head-up one.
        self.assertAlmostEqual(anchor_memory[resolve_memory_key(12)], 20.0)

    def test_impossible_jump_holds_previous_foot_point(self):
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {
            resolve_memory_key(10): {
                "point": np.array([50.0, 200.0]),
                "frame_index": 1,
                "owner_track_id": 10,
            }
        }
        box = np.array([280.0, 0.0, 380.0, 240.0])
        landmarks = make_landmarks(
            crop_width=124,
            crop_height=259,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        estimator = FakePoseEstimator([FakePoseResult(landmarks)])

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=10,
            frame_index=2,
            max_foot_jump_pixels_per_frame=80.0,
        )

        self.assertEqual(method, "physics_hold")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertTrue(
            np.allclose(
                last_foot_memory[resolve_memory_key(10)]["point"],
                np.array([50.0, 200.0]),
            )
        )

    def test_ratio_is_stored_by_identity_when_track_changes(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        memory = {}
        last_foot_memory = {}
        box = np.array([0.0, 0.0, 100.0, 200.0])

        clear_view_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 50.0),
                2: (45.0, 47.0),
                5: (55.0, 47.0),
                11: (50.0, 70.0),
                12: (50.0, 70.0),
                27: (50.0, 190.0),
                29: (50.0, 200.0),
            },
        )
        occluded_landmarks = make_landmarks(
            crop_width=112,
            crop_height=216,
            points={
                0: (50.0, 80.0),
                11: (50.0, 90.0),
                12: (50.0, 90.0),
            },
        )
        estimator = FakePoseEstimator([
            FakePoseResult(clear_view_landmarks),
            FakePoseResult(occluded_landmarks),
        ])
        # The master ID is supplied directly. Obtaining one through the real
        # ReID memory would test the intake pipeline, not this behaviour, and
        # that pipeline is already covered by test_reid_intake_lifecycle.
        identity_id = 4321

        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=1,
            identity_id=identity_id,
            frame_index=1,
        )
        self.assertEqual(method, "mediapipe")
        self.assertTrue(np.allclose(foot, np.array([50.0, 200.0])))
        self.assertAlmostEqual(
            memory[resolve_memory_key(1, identity_id=identity_id)], 20.0 / 130.0
        )
        self.assertNotIn(resolve_memory_key(1), memory)

        # BoT-SORT renumbers the same person from track 1 to track 99. The
        # identity-keyed memory is what lets them keep their learned ratio, so
        # the occluded frame can still reconstruct a foot point.
        foot, method = estimate_mediapipe_foot_point(
            frame,
            box,
            estimator,
            anatomical_ratio_memory=memory,
            last_foot_memory=last_foot_memory,
            track_id=99,
            identity_id=identity_id,
            frame_index=2,
        )
        self.assertEqual(method, "anatomical_ratio")
        self.assertTrue(np.allclose(foot, np.array([50.0, 155.0])))

    def test_extrapolate_fourth_corner_uses_visible_edge_lines(self):
        known_corners = {
            "top_left": (10.0, 10.0),
            "top_right": (110.0, 10.0),
            "bottom_right": (120.0, 90.0),
        }
        edge_points = {
            "edge_a": (60.0, 90.0),
            "edge_b": (8.0, 50.0),
        }

        corner = extrapolate_fourth_corner(known_corners, edge_points, missing_corner="bottom_left")

        self.assertTrue(np.allclose(corner, np.array([6.0, 90.0])))

    def test_extrapolate_fourth_corner_accepts_edge_points_in_either_order(self):
        known_corners = {
            "top_left": (10.0, 10.0),
            "top_right": (110.0, 10.0),
            "bottom_right": (120.0, 90.0),
        }
        edge_points = {
            "edge_a": (8.0, 50.0),
            "edge_b": (60.0, 90.0),
        }

        corner = extrapolate_fourth_corner(known_corners, edge_points, missing_corner="bottom_left")

        self.assertTrue(np.allclose(corner, np.array([6.0, 90.0])))

    def test_extrapolate_fourth_corner_uses_explicit_clockwise_edges(self):
        known_corners = {
            "top_left": (10.0, 10.0),
            "top_right": (110.0, 10.0),
            "bottom_left": (10.0, 90.0),
        }
        edge_points = {
            "edge_from_prev": (112.5, 55.0),
            "edge_from_next": (62.5, 95.0),
        }

        corner = extrapolate_fourth_corner(
            known_corners,
            edge_points,
            missing_corner="bottom_right",
            allow_swapped_edges=False,
        )

        self.assertTrue(np.allclose(corner, np.array([115.0, 100.0])))

    def test_extrapolate_fourth_corner_falls_back_to_parallelogram_math(self):
        known_corners = {
            "top_left": (10.0, 10.0),
            "top_right": (110.0, 10.0),
            "bottom_right": (110.0, 90.0),
        }

        corner = extrapolate_fourth_corner(known_corners, missing_corner="bottom_left")

        self.assertTrue(np.allclose(corner, np.array([10.0, 90.0])))

if __name__ == "__main__":
    unittest.main()

