"""A ground point that was inferred must not veto a confident appearance match.

The failure this prevents, observed in the field: two cameras see one person,
their box-bottom ground points disagree by more than the fusion limit because one
box is clipped or its feet are hidden behind someone else, the physical gate
therefore refuses the correct master, and one person is issued a second master
identity -- while TransReID was reporting a 0.106 distance for the same pair.
"""

import unittest

import numpy as np

from constants import (
    DEFAULT_NEW_MATCH_POSITION_SPLIT_FRAMES,
    DEFAULT_POSITION_SPLIT_FRAMES,
)
from ground_point import classify_box_bottom_evidence
from reid_memory import AppearanceIdentityMemory


class BoxBottomEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_clean_standalone_detection_is_hard_evidence(self):
        evidence, reason = classify_box_bottom_evidence(self.frame, (800, 300, 950, 900))
        self.assertEqual(evidence, "hard")
        self.assertIsNone(reason)

    def test_box_clipped_by_the_frame_bottom_is_soft(self):
        evidence, reason = classify_box_bottom_evidence(self.frame, (800, 300, 950, 1079))
        self.assertEqual(evidence, "soft")
        self.assertEqual(reason, "box_clipped_by_frame_bottom")

    def test_box_clipped_by_a_side_is_soft(self):
        evidence, reason = classify_box_bottom_evidence(self.frame, (0, 300, 150, 900))
        self.assertEqual(evidence, "soft")
        self.assertEqual(reason, "box_clipped_by_frame_side")

    def test_feet_hidden_behind_another_person_is_soft(self):
        subject = (800, 300, 950, 900)
        blocker = (780, 700, 980, 1000)  # standing in front, covering the feet
        evidence, reason = classify_box_bottom_evidence(self.frame, subject, [subject, blocker])
        self.assertEqual(evidence, "soft")
        self.assertEqual(reason, "feet_occluded_by_other_detection")

    def test_someone_overlapping_only_the_head_does_not_soften_it(self):
        subject = (800, 300, 950, 900)
        neighbour = (780, 280, 980, 400)  # overlaps the head, not the feet
        evidence, _reason = classify_box_bottom_evidence(self.frame, subject, [subject, neighbour])
        self.assertEqual(evidence, "hard")

    def test_a_missing_box_is_never_hard_evidence(self):
        self.assertEqual(classify_box_bottom_evidence(self.frame, None)[0], "soft")


class PhysicalGateEvidenceTests(unittest.TestCase):
    """Exercises the real gate, not a reimplementation of it."""

    def _memory(self, **kwargs):
        options = dict(
            cross_camera_fusion_distance_cm=50.0,
            cross_camera_max_skew_seconds=0.35,
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        options.update(kwargs)
        memory = AppearanceIdentityMemory(**options)
        self.addCleanup(memory.close, drain=False)
        return memory

    def _stage_conflict(self, memory, incoming_evidence, other_evidence):
        """One master seen by two cameras, 120 cm apart -- far past the 50 cm limit."""
        left_key = ("cam_1", 1)
        right_key = ("cam_2", 2)
        memory.identities[7] = memory._new_record()
        memory._remember_position_evidence(left_key, other_evidence)
        memory._record_master_observation_locked(7, left_key, (100.0, 100.0), 1000.0)
        memory._remember_position_evidence(right_key, incoming_evidence)
        return right_key

    def _disagree(self, memory, key, times, established_binding=False):
        return [
            memory._physical_match_allowed_locked(
                7,
                "cam_2",
                (220.0, 100.0),
                1000.05,
                track_key=key,
                established_binding=established_binding,
            )
            for _ in range(times)
        ]

    def test_two_measured_points_still_reject_as_before(self):
        memory = self._memory()
        key = self._stage_conflict(memory, "hard", "hard")
        allowed = memory._physical_match_allowed_locked(
            7, "cam_2", (220.0, 100.0), 1000.05, track_key=key
        )
        self.assertFalse(allowed, "a genuine two-measurement conflict must still be rejected")

    def test_an_inferred_incoming_point_may_not_veto(self):
        memory = self._memory()
        key = self._stage_conflict(memory, "soft", "hard")
        allowed = memory._physical_match_allowed_locked(
            7, "cam_2", (220.0, 100.0), 1000.05, track_key=key
        )
        self.assertTrue(allowed, "a clipped-box position must not reject the correct master")

    def test_an_inferred_stored_point_may_not_veto(self):
        memory = self._memory()
        key = self._stage_conflict(memory, "hard", "soft")
        allowed = memory._physical_match_allowed_locked(
            7, "cam_2", (220.0, 100.0), 1000.05, track_key=key
        )
        self.assertTrue(allowed)

    def test_soft_disagreement_does_not_accumulate_a_violation_streak(self):
        """Otherwise repeated soft frames would eventually reject anyway."""
        memory = self._memory()
        key = self._stage_conflict(memory, "soft", "hard")
        for _ in range(10):
            memory._physical_match_allowed_locked(
                7, "cam_2", (220.0, 100.0), 1000.05, track_key=key
            )
        self.assertEqual(memory.physical_violation_counts, {})

    def test_the_old_behaviour_is_reproducible_for_comparison(self):
        memory = self._memory(position_confidence_gating=False)
        key = self._stage_conflict(memory, "soft", "hard")
        allowed = memory._physical_match_allowed_locked(
            7, "cam_2", (220.0, 100.0), 1000.05, track_key=key
        )
        self.assertFalse(allowed, "the switch must restore the previous rule exactly")

    def test_agreeing_positions_are_allowed_regardless_of_evidence(self):
        for incoming in ("hard", "soft"):
            memory = self._memory()
            key = self._stage_conflict(memory, incoming, "hard")
            self.assertTrue(
                memory._physical_match_allowed_locked(
                    7, "cam_2", (110.0, 105.0), 1000.05, track_key=key
                )
            )

    def test_callers_that_supply_no_evidence_keep_the_previous_rules(self):
        """Backwards compatibility for any path that never sets evidence."""
        memory = self._memory()
        memory.identities[7] = memory._new_record()
        memory._record_master_observation_locked(7, ("cam_1", 1), (100.0, 100.0), 1000.0)
        allowed = memory._physical_match_allowed_locked(
            7, "cam_2", (220.0, 100.0), 1000.05, track_key=("cam_2", 2)
        )
        self.assertFalse(allowed)

    def test_stored_observation_carries_its_evidence(self):
        memory = self._memory()
        memory._remember_position_evidence(("cam_1", 1), "soft", "box_clipped_by_frame_bottom")
        memory._record_master_observation_locked(7, ("cam_1", 1), (10.0, 10.0), 1000.0)
        stored = memory.recent_master_observations[7]["cam_1"]
        self.assertEqual(stored["evidence"], "soft")
        self.assertEqual(stored["evidence_reason"], "box_clipped_by_frame_bottom")

    def test_unknown_evidence_strings_are_treated_as_soft(self):
        memory = self._memory()
        memory._remember_position_evidence(("cam_1", 1), "probably fine")
        self.assertEqual(memory._position_evidence(("cam_1", 1)), "soft")


class EstablishedBindingPatienceTests(unittest.TestCase):
    """A binding that has been working survives a run of bad geometry.

    The evidence grade above only catches unreliability a camera can detect --
    a clipped box, an occluded foot.  It cannot catch a perfectly visible foot
    projected through a grazing-angle homography, where a two-pixel wobble is
    half a metre on the map and the point is graded hard because nothing knows
    better.  Only consistency separates that from two people walking apart.
    """

    _memory = PhysicalGateEvidenceTests._memory
    _stage_conflict = PhysicalGateEvidenceTests._stage_conflict
    _disagree = PhysicalGateEvidenceTests._disagree

    def test_a_single_measured_disagreement_no_longer_breaks_a_binding(self):
        memory = self._memory()
        key = self._stage_conflict(memory, "hard", "hard")

        self.assertEqual(
            self._disagree(memory, key, DEFAULT_POSITION_SPLIT_FRAMES - 1, True),
            [True] * (DEFAULT_POSITION_SPLIT_FRAMES - 1),
        )

    def test_a_persistent_disagreement_still_breaks_it(self):
        memory = self._memory()
        key = self._stage_conflict(memory, "hard", "hard")

        results = self._disagree(memory, key, DEFAULT_POSITION_SPLIT_FRAMES, True)

        self.assertFalse(results[-1])

    def test_one_agreeing_frame_clears_the_streak(self):
        memory = self._memory()
        key = self._stage_conflict(memory, "hard", "hard")
        self._disagree(memory, key, DEFAULT_POSITION_SPLIT_FRAMES - 1, True)

        memory._physical_match_allowed_locked(
            7, "cam_2", (110.0, 100.0), 1000.05, track_key=key, established_binding=True
        )

        # The streak restarts rather than resuming, so wobbles minutes apart
        # never add up to evidence that was never simultaneous.
        self.assertEqual(
            self._disagree(memory, key, DEFAULT_POSITION_SPLIT_FRAMES - 1, True),
            [True] * (DEFAULT_POSITION_SPLIT_FRAMES - 1),
        )

    def test_soft_frames_never_accumulate_toward_the_break(self):
        memory = self._memory()
        key = self._stage_conflict(memory, "soft", "hard")

        results = self._disagree(memory, key, DEFAULT_POSITION_SPLIT_FRAMES * 3, True)

        self.assertEqual(results, [True] * (DEFAULT_POSITION_SPLIT_FRAMES * 3))
        self.assertEqual(memory.physical_violation_counts, {})


class NewMatchPatienceTests(unittest.TestCase):
    """Joining a master is not judged more harshly than keeping one.

    Refusing a join mints a duplicate that nothing later undoes, so a reading
    barely over the limit -- the homography's error bar -- buys a second look
    rather than a second ID.  A reading several times over is a different
    person and is still refused on sight.
    """

    _memory = PhysicalGateEvidenceTests._memory

    def stage(self, memory, other_point):
        """One master already seen on cam_1 at (100, 100)."""
        memory.identities[7] = memory._new_record()
        memory._remember_position_evidence(("cam_1", 1), "hard")
        memory._record_master_observation_locked(7, ("cam_1", 1), (100.0, 100.0), 1000.0)
        key = ("cam_2", 2)
        memory._remember_position_evidence(key, "hard")
        return lambda: memory._physical_match_allowed_locked(
            7,
            "cam_2",
            other_point,
            1000.05,
            track_key=key,
        )

    def test_a_marginally_over_reading_is_forgiven_once(self):
        memory = self._memory()
        # 62.4 cm against a 50 cm limit: the disagreement that cost a real ID.
        attempt = self.stage(memory, (162.4, 100.0))

        self.assertTrue(attempt(), "a quarter over the limit is not another person")

    def test_a_marginally_over_reading_that_persists_is_still_refused(self):
        memory = self._memory()
        attempt = self.stage(memory, (162.4, 100.0))

        attempt()

        self.assertFalse(attempt(), "patience is a second look, not a free pass")

    def test_a_grossly_over_reading_is_refused_on_sight(self):
        memory = self._memory()
        # 120 cm: past the slack ratio, so no patience is spent on it.
        attempt = self.stage(memory, (220.0, 100.0))

        self.assertFalse(attempt())

    def test_patience_never_exceeds_what_an_established_binding_gets(self):
        memory = self._memory()

        self.assertLessEqual(
            DEFAULT_NEW_MATCH_POSITION_SPLIT_FRAMES,
            DEFAULT_POSITION_SPLIT_FRAMES,
            "a binding that has been working must never be the more fragile one",
        )
        del memory


if __name__ == "__main__":
    unittest.main()
