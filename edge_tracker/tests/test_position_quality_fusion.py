"""Position quality: who the map believes, and how long a pairing survives.

Two consumers want opposite things from a ground point.  The tactical map wants
the most accurate one and may take it from whichever camera sees best.  The
cross-camera matcher wants an *independent* point per camera, because a point
borrowed from the other camera would make its own output its input.  These
tests pin both halves, plus the patience that keeps one distorted frame from
splitting a settled pair into two people.
"""

import unittest

from constants import (
    DEFAULT_MAP_POSITION_EMA_ALPHA,
    DEFAULT_POSITION_SPLIT_FRAMES,
    POSITION_QUALITY_HARD,
    POSITION_QUALITY_NONE,
    POSITION_QUALITY_SOFT,
    POSITION_QUALITY_STALE,
)
from core_math import (
    classify_position_quality,
    is_soft_position,
    position_quality_weight,
    update_map_motion,
)
from camera_fusion import fuse_camera_points


def observation(camera, track, identity, point, quality=POSITION_QUALITY_HARD, captured_at=10.0):
    return {
        "camera_id": camera,
        "local_track_id": track,
        "identity_id": identity,
        "reid_confirmed": True,
        "point": point,
        "position_quality": quality,
        "captured_at": captured_at,
    }


def fuse(left, right, pair_memory, max_distance_cm=50.0):
    return fuse_camera_points(
        {"cam_1": [left], "cam_2": [right]},
        max_distance_cm=max_distance_cm,
        require_reid=True,
        pair_memory=pair_memory,
    )


def settle_pair(pair_memory, frames=2):
    """Fuse an agreeing pair often enough that it counts as established."""
    for _ in range(frames):
        fuse(
            observation("cam_1", 1, 8, (100.0, 100.0)),
            observation("cam_2", 7, 8, (104.0, 100.0)),
            pair_memory,
        )


class PositionQualityGradingTests(unittest.TestCase):
    def test_measured_foot_in_a_clean_box_is_hard(self):
        self.assertEqual(
            classify_position_quality("mediapipe", box_evidence="hard"),
            POSITION_QUALITY_HARD,
        )

    def test_body_ratio_estimate_is_never_hard(self):
        self.assertEqual(
            classify_position_quality("anatomical_ratio", box_evidence="hard"),
            POSITION_QUALITY_SOFT,
        )

    def test_measured_foot_in_a_clipped_box_is_downgraded(self):
        self.assertEqual(
            classify_position_quality("mediapipe", box_evidence="soft"),
            POSITION_QUALITY_SOFT,
        )

    def test_low_visibility_foot_is_downgraded_even_in_a_clean_box(self):
        # Nothing else notices this case: MediaPipe returned a foot and the box
        # is unobstructed, so the point looks like a measurement.
        self.assertEqual(
            classify_position_quality("mediapipe", box_evidence="hard", strict_foot_visible=False),
            POSITION_QUALITY_SOFT,
        )

    def test_a_held_or_remembered_point_is_stale(self):
        self.assertEqual(classify_position_quality("physics_hold"), POSITION_QUALITY_STALE)
        self.assertEqual(classify_position_quality("last_seen"), POSITION_QUALITY_STALE)

    def test_only_hard_may_break_a_pairing(self):
        self.assertFalse(is_soft_position(POSITION_QUALITY_HARD))
        self.assertTrue(is_soft_position(POSITION_QUALITY_SOFT))
        self.assertTrue(is_soft_position(POSITION_QUALITY_STALE))
        # An ungraded point is treated as unreliable rather than trusted.
        self.assertTrue(is_soft_position(None))

    def test_hard_outweighs_soft_by_an_order_of_magnitude(self):
        self.assertGreaterEqual(
            position_quality_weight(POSITION_QUALITY_HARD),
            10.0 * position_quality_weight(POSITION_QUALITY_SOFT),
        )
        self.assertEqual(position_quality_weight(POSITION_QUALITY_NONE), 0.0)


class WeightedCenterTests(unittest.TestCase):
    def test_clear_camera_effectively_owns_the_dot(self):
        fused = fuse(
            observation("cam_1", 1, 8, (100.0, 100.0), POSITION_QUALITY_HARD),
            observation("cam_2", 7, 8, (140.0, 100.0), POSITION_QUALITY_SOFT),
            {},
        )

        self.assertEqual(len(fused), 1)
        # The plain mean would sit at 120: halfway to a camera that cannot see
        # the feet, which is the tug-of-war this replaces.
        self.assertLess(fused[0]["center"][0], 105.0)

    def test_two_clear_cameras_still_average(self):
        fused = fuse(
            observation("cam_1", 1, 8, (100.0, 100.0), POSITION_QUALITY_HARD),
            observation("cam_2", 7, 8, (140.0, 100.0), POSITION_QUALITY_HARD),
            {},
        )

        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0]["center"][0], 120.0)

    def test_ungraded_observations_keep_the_plain_mean(self):
        left = observation("cam_1", 1, 8, (100.0, 100.0))
        right = observation("cam_2", 7, 8, (140.0, 100.0))
        left.pop("position_quality")
        right.pop("position_quality")

        fused = fuse(left, right, {})

        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0]["center"][0], 120.0)


class FootlessObservationTests(unittest.TestCase):
    """A hidden foot must not delete the person from the matcher."""

    def test_a_person_without_a_ground_point_is_still_counted(self):
        fused = fuse_camera_points(
            {"cam_1": [observation("cam_1", 1, 8, None, POSITION_QUALITY_NONE)]},
            max_distance_cm=50.0,
            require_reid=True,
        )

        self.assertEqual(len(fused), 1)
        self.assertIsNone(fused[0]["center"])

    def test_shared_identity_pairs_without_geometry_and_borrows_the_position(self):
        fused = fuse(
            observation("cam_1", 1, 8, None, POSITION_QUALITY_NONE),
            observation("cam_2", 7, 8, (100.0, 100.0), POSITION_QUALITY_HARD),
            {},
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(set(fused[0]["sources"]), {"cam_1", "cam_2"})
        # Display may borrow across cameras; the matcher above never does.
        self.assertEqual(fused[0]["center"], (100.0, 100.0))

    def test_strangers_are_not_paired_merely_because_one_lost_its_feet(self):
        fused = fuse(
            observation("cam_1", 1, 8, None, POSITION_QUALITY_NONE),
            observation("cam_2", 7, 9, (100.0, 100.0), POSITION_QUALITY_HARD),
            {},
        )

        self.assertEqual(len(fused), 2)

    def test_a_camera_that_can_see_the_person_is_the_preferred_partner(self):
        blind = observation("cam_1", 1, 8, None, POSITION_QUALITY_NONE)
        seeing = observation("cam_1", 2, 8, (100.0, 100.0), POSITION_QUALITY_HARD)

        fused = fuse_camera_points(
            {
                "cam_1": [blind, seeing],
                "cam_2": [observation("cam_2", 7, 8, (104.0, 100.0), POSITION_QUALITY_HARD)],
            },
            max_distance_cm=50.0,
            require_reid=True,
            pair_memory={},
        )

        paired = [person for person in fused if len(person["sources"]) > 1]
        self.assertEqual(len(paired), 1)
        self.assertEqual(
            {observation["local_track_id"] for observation in paired[0]["observations"]},
            {2, 7},
        )


class PairPersistenceTests(unittest.TestCase):
    """One distorted frame must not put a second dot on the map."""

    def test_an_unsettled_pair_still_splits_immediately(self):
        fused = fuse(
            observation("cam_1", 1, 8, (100.0, 100.0)),
            observation("cam_2", 7, 8, (180.0, 100.0)),
            {},
        )

        self.assertEqual(len(fused), 2)

    def test_a_settled_pair_survives_a_single_wobble(self):
        pair_memory = {}
        settle_pair(pair_memory)

        fused = fuse(
            observation("cam_1", 1, 8, (100.0, 100.0)),
            observation("cam_2", 7, 8, (180.0, 100.0)),
            pair_memory,
        )

        self.assertEqual(len(fused), 1)

    def test_a_settled_pair_splits_once_the_separation_persists(self):
        pair_memory = {}
        settle_pair(pair_memory)

        counts = []
        for _ in range(DEFAULT_POSITION_SPLIT_FRAMES):
            counts.append(
                len(
                    fuse(
                        observation("cam_1", 1, 8, (100.0, 100.0)),
                        observation("cam_2", 7, 8, (180.0, 100.0)),
                        pair_memory,
                    )
                )
            )

        self.assertEqual(counts[-1], 2)
        self.assertEqual(set(counts[:-1]), {1})

    def test_a_frame_back_in_range_clears_the_streak(self):
        pair_memory = {}
        settle_pair(pair_memory)
        for _ in range(DEFAULT_POSITION_SPLIT_FRAMES - 1):
            fuse(
                observation("cam_1", 1, 8, (100.0, 100.0)),
                observation("cam_2", 7, 8, (180.0, 100.0)),
                pair_memory,
            )

        settle_pair(pair_memory, frames=1)
        fused = fuse(
            observation("cam_1", 1, 8, (100.0, 100.0)),
            observation("cam_2", 7, 8, (180.0, 100.0)),
            pair_memory,
        )

        self.assertEqual(len(fused), 1)

    def test_a_soft_point_sustains_a_pairing_without_ever_counting_against_it(self):
        pair_memory = {}
        settle_pair(pair_memory)

        for _ in range(DEFAULT_POSITION_SPLIT_FRAMES * 3):
            fused = fuse(
                observation("cam_1", 1, 8, (100.0, 100.0), POSITION_QUALITY_HARD),
                observation("cam_2", 7, 8, (180.0, 100.0), POSITION_QUALITY_SOFT),
                pair_memory,
            )
            self.assertEqual(len(fused), 1)

    def test_patience_is_bounded_by_the_widened_gate(self):
        pair_memory = {}
        settle_pair(pair_memory)

        fused = fuse(
            observation("cam_1", 1, 8, (100.0, 100.0), POSITION_QUALITY_HARD),
            observation("cam_2", 7, 8, (1000.0, 100.0), POSITION_QUALITY_SOFT),
            pair_memory,
        )

        self.assertEqual(len(fused), 2)

    def test_pairing_history_does_not_outlive_the_people_in_it(self):
        pair_memory = {}
        settle_pair(pair_memory)
        self.assertTrue(pair_memory)

        for _ in range(200):
            fuse_camera_points(
                {"cam_1": [], "cam_2": []},
                max_distance_cm=50.0,
                require_reid=True,
                pair_memory=pair_memory,
            )

        self.assertEqual(pair_memory, {})


class MotionFilterQualityTests(unittest.TestCase):
    def test_a_soft_point_moves_the_smoothed_position_less_than_a_hard_one(self):
        def step(quality):
            memory = {}
            update_map_motion(memory, ("identity", 1), (100.0, 100.0), 1.0)
            point, _speed, _status = update_map_motion(
                memory, ("identity", 1), (200.0, 100.0), 2.0, quality=quality
            )
            return point[0]

        hard = step(POSITION_QUALITY_HARD)
        soft = step(POSITION_QUALITY_SOFT)

        self.assertAlmostEqual(hard, 100.0 + 100.0 * DEFAULT_MAP_POSITION_EMA_ALPHA)
        self.assertLess(soft, hard)
        self.assertGreater(soft, 100.0)

    def test_a_stale_point_does_not_move_the_smoothed_position_at_all(self):
        memory = {}
        update_map_motion(memory, ("identity", 1), (100.0, 100.0), 1.0)
        point, _speed, _status = update_map_motion(
            memory, ("identity", 1), (200.0, 100.0), 2.0, quality=POSITION_QUALITY_STALE
        )

        self.assertAlmostEqual(point[0], 100.0)


if __name__ == "__main__":
    unittest.main()
