"""One physical person should produce one dot, even when the names disagree.

Normal fusion applies every identity rule and refuses to combine observations
whose masters conflict.  That is correct for identity, but it leaves the map
drawing one person twice whenever two cameras see the same body and the
appearance layer has temporarily labelled them differently -- which the
recorded run did roughly 198 times, every one of them with the two points
inside the normal fusion distance.

These tests pin the presentation-level answer: geometry alone may decide that
two dots are one person, colour tells the operator which layer decided what,
and none of it is allowed to touch the identity state underneath.
"""

import unittest

from constants import (
    DEFAULT_DISPLAY_DUPLICATE_DISTANCE_CM,
    POSITION_QUALITY_HARD,
    POSITION_QUALITY_NONE,
    POSITION_QUALITY_SOFT,
    POSITION_QUALITY_STALE,
)
from camera_fusion import suppress_display_duplicates
from fused_person import display_position_quality
from tactical_render import (
    MAP_COLOR_DARK_GREEN,
    MAP_COLOR_LIGHT_GREEN,
    MAP_COLOR_STALE,
    MAP_COLOR_YELLOW,
    fused_person_style,
    per_camera_point_style,
)


def person(camera, track, identity, point, quality=POSITION_QUALITY_HARD, reason=None, state="confirmed"):
    """One display person as fuse_camera_points emits it for a single camera."""
    observation = {
        "camera_id": camera,
        "local_track_id": track,
        "identity_id": identity,
        "identity_state": state,
        "point": point,
        "position_quality": quality,
        "position_quality_reason": reason,
    }
    return {
        "center": point,
        "points": [point],
        "sources": [camera],
        "observations": [observation],
        "identity_id": identity,
        "temporary_group_id": None,
        "identity_state": state,
        "role": None,
    }


def fused_person(identity, point, cameras=("cam_1", "cam_2")):
    observations = [
        {
            "camera_id": camera,
            "local_track_id": index,
            "identity_id": identity,
            "point": point,
            "position_quality": POSITION_QUALITY_HARD,
        }
        for index, camera in enumerate(cameras)
    ]
    return {
        "center": point,
        "points": [point] * len(cameras),
        "sources": list(cameras),
        "observations": observations,
        "identity_id": identity,
        "temporary_group_id": None,
        "identity_state": "confirmed",
        "role": None,
    }


class CameraAuthorityTests(unittest.TestCase):
    """The camera that can see the feet owns the dot."""

    def test_hard_cam_1_beats_soft_cam_2_at_twenty_centimetres(self):
        left = person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person(
            "cam_2", 16, 1, (120.0, 100.0), POSITION_QUALITY_SOFT, "box_clipped_by_frame_bottom"
        )

        display = suppress_display_duplicates([left, right])

        self.assertEqual(len(display), 1)
        self.assertEqual(display[0]["identity_id"], 2)
        self.assertEqual(display[0]["center"], (100.0, 100.0))
        self.assertEqual(display[0]["sources"], ["cam_1"])

    def test_hard_cam_2_beats_soft_cam_1(self):
        left = person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_SOFT)
        right = person("cam_2", 16, 1, (120.0, 100.0), POSITION_QUALITY_HARD)

        display = suppress_display_duplicates([left, right])

        self.assertEqual(len(display), 1)
        self.assertEqual(display[0]["identity_id"], 1)
        self.assertEqual(display[0]["center"], (120.0, 100.0))

    def test_hard_beats_stale(self):
        left = person("cam_1", 1, 5, (100.0, 100.0), POSITION_QUALITY_STALE)
        right = person("cam_2", 2, 5, (110.0, 100.0), POSITION_QUALITY_HARD)

        display = suppress_display_duplicates([left, right])

        self.assertEqual(len(display), 1)
        self.assertEqual(display[0]["center"], (110.0, 100.0))

    def test_a_missing_identity_never_outranks_a_named_one_at_equal_quality(self):
        left = person("cam_1", 18, 1, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 25, None, (115.0, 100.0), POSITION_QUALITY_HARD, state=None)

        display = suppress_display_duplicates([left, right])

        self.assertEqual(len(display), 1)
        self.assertEqual(display[0]["identity_id"], 1)


class SuppressionBoundaryTests(unittest.TestCase):
    """The rule must never reach beyond the case it was built for."""

    def test_two_detections_from_the_same_camera_are_never_collapsed(self):
        left = person("cam_1", 1, 1, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_1", 2, 2, (120.0, 100.0), POSITION_QUALITY_SOFT)

        display = suppress_display_duplicates([left, right])

        self.assertEqual(len(display), 2)

    def test_a_fused_pair_is_never_collapsed_against_its_own_camera(self):
        pair = fused_person(1, (100.0, 100.0))
        stray = person("cam_1", 9, 4, (110.0, 100.0), POSITION_QUALITY_SOFT)

        display = suppress_display_duplicates([pair, stray])

        self.assertEqual(len(display), 2)

    def test_observations_outside_the_duplicate_radius_stay_separate(self):
        left = person("cam_1", 1, 1, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 2, 2, (145.0, 100.0), POSITION_QUALITY_SOFT)

        display = suppress_display_duplicates([left, right])

        self.assertEqual(len(display), 2)

    def test_the_radius_is_tighter_than_the_normal_fusion_limit(self):
        # 45 cm would fuse normally, but two people can stand 45 cm apart.
        self.assertLess(DEFAULT_DISPLAY_DUPLICATE_DISTANCE_CM, 50.0)
        left = person("cam_1", 1, 1, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 2, 2, (145.0, 100.0), POSITION_QUALITY_SOFT)

        self.assertEqual(len(suppress_display_duplicates([left, right])), 2)

    def test_a_person_without_a_position_is_never_a_duplicate(self):
        left = person("cam_1", 1, 1, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 2, 2, None, POSITION_QUALITY_SOFT)
        right["center"] = None

        display = suppress_display_duplicates([left, right])

        self.assertEqual(len(display), 2)

    def test_suppression_can_be_disabled(self):
        left = person("cam_1", 1, 1, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 2, 2, (110.0, 100.0), POSITION_QUALITY_SOFT)

        self.assertEqual(len(suppress_display_duplicates([left, right], 0.0)), 2)


class SameMasterDuplicateTests(unittest.TestCase):
    """One master ID must never occupy two dots, however far apart they drift.

    Observed in run debug_mpstudent_20260807_180148, cycle 606: cam_1 held a
    stale point for ID 4 while cam_2 still measured them, 82 cm apart.  Every
    distance gate declined the pair -- correctly, by its own rule -- so camera
    authority was never asked, and the map drew ID 4 twice.  It happened in 468
    of 1711 cycles across six different masters.
    """

    def test_stale_loses_to_hard_far_beyond_every_distance_gate(self):
        stale = person("cam_1", 46, 4, (299.0, 326.0), POSITION_QUALITY_STALE, "physics_hold")
        live = person("cam_2", 37, 4, (267.0, 402.0), POSITION_QUALITY_HARD, "mediapipe")

        display = suppress_display_duplicates([stale, live])

        self.assertEqual(len(display), 1)
        self.assertEqual(display[0]["center"], (267.0, 402.0))
        self.assertEqual(display[0]["sources"], ["cam_2"])
        self.assertEqual(display[0]["suppressed_duplicates"][0]["basis"], "same_master")

    def test_stale_loses_to_soft(self):
        stale = person("cam_1", 46, 4, (299.0, 326.0), POSITION_QUALITY_STALE)
        soft = person("cam_2", 37, 4, (394.0, 395.0), POSITION_QUALITY_SOFT)

        display = suppress_display_duplicates([stale, soft])

        self.assertEqual(len(display), 1)
        self.assertEqual(display[0]["center"], (394.0, 395.0))

    def test_a_positioned_side_beats_a_positionless_one(self):
        blind = person("cam_1", 46, 4, None, POSITION_QUALITY_NONE)
        blind["center"] = None
        live = person("cam_2", 37, 4, (267.0, 402.0), POSITION_QUALITY_HARD)

        display = suppress_display_duplicates([blind, live])

        self.assertEqual(len(display), 1)
        self.assertEqual(display[0]["center"], (267.0, 402.0))

    def test_different_masters_far_apart_still_stay_separate(self):
        """Only a *shared* master collapses without geometry."""
        left = person("cam_1", 46, 4, (299.0, 326.0), POSITION_QUALITY_STALE)
        right = person("cam_2", 37, 5, (267.0, 402.0), POSITION_QUALITY_HARD)

        self.assertEqual(len(suppress_display_duplicates([left, right])), 2)

    def test_one_camera_holding_a_master_twice_is_left_visible(self):
        """An identity fault must not be hidden by the display layer."""
        first = person("cam_1", 17, 4, (100.0, 100.0), POSITION_QUALITY_HARD)
        second = person("cam_1", 18, 4, (300.0, 300.0), POSITION_QUALITY_STALE)

        self.assertEqual(len(suppress_display_duplicates([first, second])), 2)

    def test_same_master_collapse_survives_the_proximity_rule_being_disabled(self):
        stale = person("cam_1", 46, 4, (299.0, 326.0), POSITION_QUALITY_STALE)
        live = person("cam_2", 37, 4, (267.0, 402.0), POSITION_QUALITY_HARD)

        display = suppress_display_duplicates([stale, live], duplicate_distance_cm=0.0)

        self.assertEqual(len(display), 1)
        self.assertEqual(display[0]["center"], (267.0, 402.0))

    def test_equal_quality_same_master_is_order_independent(self):
        left = person("cam_1", 46, 4, (299.0, 326.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 37, 4, (267.0, 402.0), POSITION_QUALITY_HARD)

        forward = suppress_display_duplicates([left, right])
        backward = suppress_display_duplicates([right, left])

        self.assertEqual(len(forward), 1)
        self.assertEqual(forward[0]["center"], backward[0]["center"])

    def test_the_dropped_side_keeps_its_diagnostics(self):
        stale = person("cam_1", 46, 4, (299.0, 326.0), POSITION_QUALITY_STALE, "physics_hold")
        live = person("cam_2", 37, 4, (267.0, 402.0), POSITION_QUALITY_HARD)

        dropped = suppress_display_duplicates([stale, live])[0]["suppressed_duplicates"][0]

        self.assertEqual(dropped["camera_id"], "cam_1")
        self.assertEqual(dropped["local_track_id"], 46)
        self.assertEqual(dropped["identity_id"], 4)
        self.assertEqual(dropped["position_quality"], POSITION_QUALITY_STALE)
        self.assertEqual(dropped["position_quality_reason"], "physics_hold")
        self.assertAlmostEqual(dropped["distance_cm"], 82.4, places=0)

    def test_nothing_upstream_is_mutated(self):
        stale = person("cam_1", 46, 4, (299.0, 326.0), POSITION_QUALITY_STALE)
        live = person("cam_2", 37, 4, (267.0, 402.0), POSITION_QUALITY_HARD)
        originals = [dict(stale), dict(live)]
        observations = [dict(stale["observations"][0]), dict(live["observations"][0])]

        suppress_display_duplicates([stale, live])

        self.assertEqual(stale, originals[0])
        self.assertEqual(live, originals[1])
        self.assertEqual(stale["observations"][0], observations[0])
        self.assertEqual(live["observations"][0], observations[1])

    def test_the_survivor_is_still_drawn_as_a_confirmed_master(self):
        stale = person("cam_1", 46, 4, (299.0, 326.0), POSITION_QUALITY_STALE)
        live = person("cam_2", 37, 4, (267.0, 402.0), POSITION_QUALITY_HARD)

        display = suppress_display_duplicates([stale, live])

        self.assertEqual(display[0]["identity_id"], 4)
        self.assertEqual(fused_person_style(display[0]), MAP_COLOR_LIGHT_GREEN)


class ConservativeTieBreakTests(unittest.TestCase):
    def test_equal_evidence_with_conflicting_identities_keeps_both(self):
        """Choosing here would invent a fact the location layer does not have."""
        left = person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 16, 1, (115.0, 100.0), POSITION_QUALITY_HARD)

        display = suppress_display_duplicates([left, right])

        self.assertEqual(len(display), 2)
        self.assertEqual({p["identity_id"] for p in display}, {1, 2})

    def test_equal_evidence_same_master_collapses_deterministically(self):
        left = person("cam_1", 17, 7, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 16, 7, (115.0, 100.0), POSITION_QUALITY_HARD)

        forward = suppress_display_duplicates([left, right])
        backward = suppress_display_duplicates([right, left])

        self.assertEqual(len(forward), 1)
        self.assertEqual(len(backward), 1)
        # List order must not decide the survivor.
        self.assertEqual(forward[0]["center"], backward[0]["center"])
        self.assertEqual(forward[0]["sources"], backward[0]["sources"])

    def test_two_anonymous_detections_collapse_deterministically(self):
        left = person("cam_1", 3, None, (100.0, 100.0), POSITION_QUALITY_HARD, state=None)
        right = person("cam_2", 4, None, (115.0, 100.0), POSITION_QUALITY_HARD, state=None)

        forward = suppress_display_duplicates([left, right])
        backward = suppress_display_duplicates([right, left])

        self.assertEqual(len(forward), 1)
        self.assertEqual(forward[0]["center"], backward[0]["center"])


class DiagnosticsTests(unittest.TestCase):
    def test_the_suppressed_observation_stays_available_for_debugging(self):
        left = person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person(
            "cam_2", 16, 1, (120.0, 100.0), POSITION_QUALITY_SOFT, "feet_occluded_by_other_detection"
        )

        display = suppress_display_duplicates([left, right])
        dropped = display[0]["suppressed_duplicates"]

        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["camera_id"], "cam_2")
        self.assertEqual(dropped[0]["local_track_id"], 16)
        self.assertEqual(dropped[0]["identity_id"], 1)
        self.assertEqual(dropped[0]["position_quality"], POSITION_QUALITY_SOFT)
        self.assertEqual(
            dropped[0]["position_quality_reason"], "feet_occluded_by_other_detection"
        )
        self.assertEqual(dropped[0]["reason"], "better_position_evidence")
        self.assertAlmostEqual(dropped[0]["distance_cm"], 20.0)

    def test_nothing_upstream_is_mutated(self):
        """The identity layer has already run; this stage must be inert."""
        left = person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 16, 1, (120.0, 100.0), POSITION_QUALITY_SOFT)
        originals = [dict(left), dict(right)]
        original_observations = [dict(left["observations"][0]), dict(right["observations"][0])]

        suppress_display_duplicates([left, right])

        self.assertEqual(left, originals[0])
        self.assertEqual(right, originals[1])
        self.assertEqual(left["observations"][0], original_observations[0])
        self.assertEqual(right["observations"][0], original_observations[1])
        # Master IDs in particular must survive untouched.
        self.assertEqual(left["identity_id"], 2)
        self.assertEqual(right["identity_id"], 1)


class FusedMapColourTests(unittest.TestCase):
    """Colour on the combined map means identity and fusion status."""

    def test_confirmed_master_from_both_cameras_is_dark_green(self):
        self.assertEqual(fused_person_style(fused_person(1, (10.0, 10.0))), MAP_COLOR_DARK_GREEN)

    def test_confirmed_master_from_one_camera_is_light_green(self):
        single = person("cam_1", 1, 2, (10.0, 10.0))
        self.assertEqual(fused_person_style(single), MAP_COLOR_LIGHT_GREEN)

    def test_a_master_seen_by_one_camera_never_reverts_to_yellow(self):
        single = person("cam_1", 1, 4, (10.0, 10.0))
        self.assertNotEqual(fused_person_style(single), MAP_COLOR_YELLOW)

    def test_a_local_provisional_track_is_yellow(self):
        local = person("cam_1", 17, None, (10.0, 10.0), state=None)
        self.assertEqual(fused_person_style(local), MAP_COLOR_YELLOW)

    def test_an_analyzing_temporary_group_is_yellow(self):
        analyzing = person("cam_1", 17, None, (10.0, 10.0), state="analyzing")
        analyzing["temporary_group_id"] = "tmp_1"
        self.assertEqual(fused_person_style(analyzing), MAP_COLOR_YELLOW)

    def test_surviving_a_suppression_keeps_the_master_green(self):
        left = person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD)
        right = person("cam_2", 16, 1, (120.0, 100.0), POSITION_QUALITY_SOFT)

        display = suppress_display_duplicates([left, right])

        self.assertEqual(fused_person_style(display[0]), MAP_COLOR_LIGHT_GREEN)


class PerCameraMapColourTests(unittest.TestCase):
    """Colour on a per-camera map means position quality, nothing else."""

    def test_hard_is_dark_green_and_filled(self):
        colour, thickness = per_camera_point_style({"position_quality": POSITION_QUALITY_HARD})
        self.assertEqual(colour, MAP_COLOR_DARK_GREEN)
        self.assertEqual(thickness, -1)

    def test_soft_is_light_green_and_filled(self):
        colour, thickness = per_camera_point_style({"position_quality": POSITION_QUALITY_SOFT})
        self.assertEqual(colour, MAP_COLOR_LIGHT_GREEN)
        self.assertEqual(thickness, -1)

    def test_stale_is_faded_and_hollow(self):
        colour, thickness = per_camera_point_style({"position_quality": POSITION_QUALITY_STALE})
        self.assertEqual(colour, MAP_COLOR_STALE)
        self.assertGreater(thickness, 0, "a stale point must not look like a live measurement")

    def test_a_provisional_track_is_yellow_whatever_its_quality(self):
        for quality in (POSITION_QUALITY_HARD, POSITION_QUALITY_SOFT, POSITION_QUALITY_STALE):
            colour, _ = per_camera_point_style(
                {"position_quality": quality, "provisional": True}
            )
            self.assertEqual(colour, MAP_COLOR_YELLOW)

    def test_the_two_maps_disagree_on_purpose(self):
        """A soft hard-ID dot is light green on one map, dark green on the other."""
        soft_single = person("cam_1", 1, 2, (10.0, 10.0), POSITION_QUALITY_SOFT)
        hard_pair = fused_person(2, (10.0, 10.0))
        self.assertEqual(
            per_camera_point_style({"position_quality": POSITION_QUALITY_HARD})[0],
            MAP_COLOR_DARK_GREEN,
        )
        self.assertEqual(fused_person_style(soft_single), MAP_COLOR_LIGHT_GREEN)
        self.assertEqual(fused_person_style(hard_pair), MAP_COLOR_DARK_GREEN)


class DisplayQualityTests(unittest.TestCase):
    def test_a_pair_reports_its_best_camera(self):
        pair = fused_person(1, (10.0, 10.0))
        pair["observations"][1]["position_quality"] = POSITION_QUALITY_SOFT
        self.assertEqual(display_position_quality(pair), POSITION_QUALITY_HARD)

    def test_a_person_with_no_positioned_observation_is_none(self):
        empty = person("cam_1", 1, 1, None)
        empty["center"] = None
        empty["observations"][0]["point"] = None
        self.assertEqual(display_position_quality(empty), "none")


class RecordedScenarioTests(unittest.TestCase):
    """The t=146-153 s case from debug_mpstudent_20260806_212438.

    cam_1 sees Mikail and Haoran clearly. cam_2's view of both is occluded or
    clipped, and its identity layer disagrees about who they are. cam_2 does
    see Denn clearly, and that pair fuses normally. The map showed five dots
    for three people; it should show three.
    """

    def test_five_dots_for_three_people_becomes_three(self):
        haoran_cam_1 = person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD)
        haoran_cam_2 = person(
            "cam_2", 16, 1, (124.0, 100.0), POSITION_QUALITY_SOFT, "box_clipped_by_frame_bottom"
        )
        mikail_cam_1 = person("cam_1", 18, 1, (300.0, 200.0), POSITION_QUALITY_HARD)
        mikail_cam_2 = person(
            "cam_2",
            25,
            None,
            (315.0, 200.0),
            POSITION_QUALITY_SOFT,
            "feet_occluded_by_other_detection",
            state=None,
        )
        denn = fused_person(3, (50.0, 400.0))

        display = suppress_display_duplicates(
            [haoran_cam_1, haoran_cam_2, mikail_cam_1, mikail_cam_2, denn]
        )

        self.assertEqual(len(display), 3)
        centers = {p["center"] for p in display}
        self.assertIn((100.0, 100.0), centers, "Haoran's position must come from cam_1")
        self.assertIn((300.0, 200.0), centers, "Mikail's position must come from cam_1")
        self.assertIn((50.0, 400.0), centers, "Denn stays fused across both cameras")
        # cam_2's degraded observations are recorded, not deleted.
        dropped_cameras = {
            d["camera_id"] for p in display for d in p.get("suppressed_duplicates", ())
        }
        self.assertEqual(dropped_cameras, {"cam_2"})


if __name__ == "__main__":
    unittest.main()
