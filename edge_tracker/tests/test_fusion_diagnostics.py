"""Diagnostics must explain a bad frame without changing what happened in it.

The question these logs exist to answer is: at this cycle, what did each camera
see, which cross-camera pairings were possible, why was each one taken or
refused, and what finally reached the map.  Answering it previously meant
joining several event types on timestamps.

Every test here holds one of two lines: the log became richer, and the run
became no different.
"""

import copy
import unittest
from unittest.mock import patch

from constants import POSITION_QUALITY_HARD, POSITION_QUALITY_SOFT
from identity_debug import (
    configure_identity_debug,
    identity_debug_detail_enabled,
    identity_debug_enabled,
    state_changed,
)
from camera_fusion import (
    fuse_camera_points,
    suppress_display_duplicates,
)
from fusion_diagnostics import (
    _candidate_flags,
    _identity_status,
    _physical_pairing_status,
    candidate_diagnostic_key,
    log_fusion_cycle_summary,
    observation_diagnostic_key,
)


def observation(camera, track, identity, point, quality=POSITION_QUALITY_HARD, reason=None,
                captured_at=10.0, confirmed=True, state="confirmed"):
    return {
        "camera_id": camera,
        "local_track_id": track,
        "identity_id": identity,
        "identity_state": state,
        "reid_confirmed": confirmed,
        "point": point,
        "position_quality": quality,
        "position_quality_reason": reason,
        "captured_at": captured_at,
        "inside_tactical_map": True,
    }


def person(camera, track, identity, point, quality=POSITION_QUALITY_HARD, reason=None):
    obs = observation(camera, track, identity, point, quality, reason)
    return {
        "center": point,
        "points": [point],
        "sources": [camera],
        "observations": [obs],
        "identity_id": identity,
        "temporary_group_id": None,
        "identity_state": "confirmed",
        "role": None,
    }


class DebugGateTests(unittest.TestCase):
    def tearDown(self):
        configure_identity_debug(False, None)

    def test_gate_is_closed_by_default(self):
        configure_identity_debug(False, None)
        self.assertFalse(identity_debug_enabled())
        self.assertFalse(identity_debug_detail_enabled())

    def test_detail_requires_debug_logging_to_be_on_at_all(self):
        configure_identity_debug(False, None, detail=True)
        self.assertFalse(identity_debug_detail_enabled())
        configure_identity_debug(True, None, detail=True)
        self.assertTrue(identity_debug_detail_enabled())

    def test_state_changed_reports_only_transitions(self):
        configure_identity_debug(True, None)
        self.assertTrue(state_changed("scope", "k", ("a",)))
        self.assertFalse(state_changed("scope", "k", ("a",)))
        self.assertFalse(state_changed("scope", "k", ("a",)))
        self.assertTrue(state_changed("scope", "k", ("b",)))
        # Changing back is a transition too -- a time throttle would hide it.
        self.assertTrue(state_changed("scope", "k", ("a",)))

    def test_state_changed_records_nothing_while_disabled(self):
        configure_identity_debug(False, None)
        self.assertFalse(state_changed("scope", "k", ("a",)))
        configure_identity_debug(True, None)
        # Nothing was remembered while off, so the first live call is a change.
        self.assertTrue(state_changed("scope", "k", ("a",)))


class ZeroBehaviourChangeTests(unittest.TestCase):
    """The same input must produce the same result with logging on or off."""

    def tearDown(self):
        configure_identity_debug(False, None)

    def _scenario(self):
        return {
            "cam_1": [
                observation("cam_1", 17, 2, (100.0, 100.0)),
                observation("cam_1", 18, 1, (300.0, 200.0)),
            ],
            "cam_2": [
                observation("cam_2", 16, 1, (124.0, 100.0), POSITION_QUALITY_SOFT, "box_clipped_by_frame_bottom"),
                observation("cam_2", 25, None, (315.0, 200.0), POSITION_QUALITY_SOFT,
                            "feet_occluded_by_other_detection", state=None),
            ],
        }

    def _run(self):
        return fuse_camera_points(
            copy.deepcopy(self._scenario()), 50.0, require_reid=True, pair_memory={}, fusion_cycle_id=7
        )

    def _shape(self, people):
        """Order-independent fingerprint of the fused result."""
        return sorted(
            repr((p.get("identity_id"), tuple(sorted(p["sources"])), p.get("center")))
            for p in people
        )

    def test_fused_output_is_identical_with_logging_on_and_off(self):
        configure_identity_debug(False, None)
        off = self._shape(self._run())
        configure_identity_debug(True, None)
        on = self._shape(self._run())
        configure_identity_debug(True, None, detail=True)
        detailed = self._shape(self._run())

        self.assertEqual(off, on)
        self.assertEqual(off, detailed)

    def test_display_suppression_is_identical_with_logging_on_and_off(self):
        def run():
            people = [
                person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD),
                person("cam_2", 16, 1, (120.0, 100.0), POSITION_QUALITY_SOFT),
            ]
            return self._shape(suppress_display_duplicates(people, fusion_cycle_id=3))

        configure_identity_debug(False, None)
        off = run()
        configure_identity_debug(True, None, detail=True)
        on = run()
        self.assertEqual(off, on)

    def test_logging_does_not_mutate_observations(self):
        configure_identity_debug(True, None, detail=True)
        scenario = self._scenario()
        pristine = copy.deepcopy(scenario)

        fuse_camera_points(scenario, 50.0, require_reid=True, pair_memory={}, fusion_cycle_id=1)

        self.assertEqual(scenario, pristine)

    def test_candidate_eligibility_is_unchanged_by_logging(self):
        """Same accepted pairings, not merely the same count."""
        configure_identity_debug(False, None)
        off = {tuple(sorted(p["sources"])) for p in self._run()}
        configure_identity_debug(True, None, detail=True)
        on = {tuple(sorted(p["sources"])) for p in self._run()}
        self.assertEqual(off, on)


class CandidateDiagnosticTests(unittest.TestCase):
    def setUp(self):
        configure_identity_debug(True, None)
        self.addCleanup(configure_identity_debug, False, None)

    def _decisions(self, camera_observations, **kwargs):
        with patch("camera_fusion.identity_event") as event:
            fuse_camera_points(camera_observations, 50.0, require_reid=True,
                               pair_memory={}, fusion_cycle_id=42, **kwargs)
        return [
            call.kwargs
            for call in event.call_args_list
            if call.args == ("cross_camera_association_decision",)
        ]

    def test_close_different_master_is_flagged_and_fully_described(self):
        decisions = self._decisions({
            "cam_1": [observation("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD)],
            "cam_2": [observation("cam_2", 16, 1, (124.0, 100.0), POSITION_QUALITY_SOFT,
                                  "box_clipped_by_frame_bottom")],
        })

        self.assertEqual(len(decisions), 1)
        record = decisions[0]
        self.assertIn("close_different_master", record["flags"])
        self.assertIn("possible_duplicate_master", record["flags"])
        self.assertEqual(record["identity_status"], "different_master")
        self.assertEqual(record["physical_pairing_status"], "spatially_compatible")
        self.assertEqual(record["rejection_reason"], "different_master")
        self.assertTrue(record["passed_distance_gate"])
        self.assertFalse(record["passed_identity_gate"])
        self.assertFalse(record["selected"])
        self.assertEqual(record["fusion_cycle_id"], 42)
        self.assertEqual(record["candidate_id"], "cam_1#17~cam_2#16")
        self.assertAlmostEqual(record["distance_cm"], 24.0)
        self.assertEqual(record["left_quality"], POSITION_QUALITY_HARD)
        self.assertEqual(record["right_quality"], POSITION_QUALITY_SOFT)
        self.assertEqual(record["right_quality_reason"], "box_clipped_by_frame_bottom")

    def test_hard_hard_identity_conflict_is_separately_flagged(self):
        decisions = self._decisions({
            "cam_1": [observation("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD)],
            "cam_2": [observation("cam_2", 16, 1, (115.0, 100.0), POSITION_QUALITY_HARD)],
        })

        flags = decisions[0]["flags"]
        self.assertIn("close_hard_hard_identity_conflict", flags)
        self.assertIn("close_different_master", flags)

    def test_missing_identity_is_flagged(self):
        decisions = self._decisions({
            "cam_1": [observation("cam_1", 18, 1, (300.0, 200.0), POSITION_QUALITY_HARD)],
            "cam_2": [observation("cam_2", 25, None, (315.0, 200.0), POSITION_QUALITY_HARD, state=None)],
        })

        self.assertIn("close_identity_missing", decisions[0]["flags"])
        self.assertEqual(decisions[0]["identity_status"], "identity_missing_one")

    def test_physical_and_identity_status_are_reported_independently(self):
        """Spatially fine, identity broken -- the log must say both."""
        decisions = self._decisions({
            "cam_1": [observation("cam_1", 1, 2, (100.0, 100.0))],
            "cam_2": [observation("cam_2", 2, 1, (110.0, 100.0))],
        })
        record = decisions[0]
        self.assertEqual(record["physical_pairing_status"], "spatially_compatible")
        self.assertEqual(record["identity_status"], "different_master")

    def test_uneventful_candidates_are_not_logged_without_detail_mode(self):
        decisions = self._decisions({
            "cam_1": [observation("cam_1", 1, 8, (100.0, 100.0))],
            "cam_2": [observation("cam_2", 2, 8, (104.0, 100.0))],
        })
        self.assertEqual(decisions, [])

    def test_detail_mode_records_the_whole_candidate_matrix(self):
        configure_identity_debug(True, None, detail=True)
        decisions = self._decisions({
            "cam_1": [observation("cam_1", 1, 8, (100.0, 100.0))],
            "cam_2": [observation("cam_2", 2, 8, (104.0, 100.0))],
        })
        self.assertEqual(len(decisions), 1)
        record = decisions[0]
        self.assertTrue(record["selected"])
        self.assertTrue(record["eligible_for_assignment"])
        self.assertEqual(record["identity_status"], "same_master")
        self.assertIsNotNone(record["association_cost"])
        self.assertIsNotNone(record["geometric_cost"])

    def test_nothing_is_logged_while_debugging_is_off(self):
        configure_identity_debug(False, None)
        decisions = self._decisions({
            "cam_1": [observation("cam_1", 17, 2, (100.0, 100.0))],
            "cam_2": [observation("cam_2", 16, 1, (110.0, 100.0))],
        })
        self.assertEqual(decisions, [])


class CycleSummaryTests(unittest.TestCase):
    def setUp(self):
        configure_identity_debug(True, None)
        self.addCleanup(configure_identity_debug, False, None)

    def test_summary_reports_counts_either_side_of_suppression(self):
        before = [
            person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD),
            person("cam_2", 16, 1, (120.0, 100.0), POSITION_QUALITY_SOFT),
        ]
        stats = {}
        after = suppress_display_duplicates(before, fusion_cycle_id=9, stats=stats)

        with patch("fusion_diagnostics.identity_event") as event:
            log_fusion_cycle_summary(
                9,
                {"cam_1": [before[0]["observations"][0]], "cam_2": [before[1]["observations"][0]]},
                before,
                after,
                unresolved_count=stats["unresolved"],
            )

        summary = next(
            call.kwargs for call in event.call_args_list
            if call.args == ("fusion_cycle_summary",)
        )
        self.assertEqual(summary["fusion_cycle_id"], 9)
        self.assertEqual(summary["fused_count_before_suppression"], 2)
        self.assertEqual(summary["fused_count_after_suppression"], 1)
        self.assertEqual(summary["suppressed_duplicate_count"], 1)
        self.assertEqual(summary["unresolved_duplicate_count"], 0)
        self.assertEqual(summary["camera_observation_counts"], {"cam_1": 1, "cam_2": 1})
        self.assertEqual(summary["displayed_ids"], [2])
        self.assertEqual(summary["people"][0]["authority_camera"], "cam_1")

    def test_unresolved_conflicts_are_counted(self):
        both_hard = [
            person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD),
            person("cam_2", 16, 1, (115.0, 100.0), POSITION_QUALITY_HARD),
        ]
        stats = {}
        after = suppress_display_duplicates(both_hard, fusion_cycle_id=4, stats=stats)

        self.assertEqual(len(after), 2)
        self.assertEqual(stats["unresolved"], 1)
        self.assertEqual(stats["suppressed"], 0)

    def test_summary_is_silent_while_debugging_is_off(self):
        configure_identity_debug(False, None)
        with patch("fusion_diagnostics.identity_event") as event:
            log_fusion_cycle_summary(1, {}, [], [])
        event.assert_not_called()


class SuppressionTraceabilityTests(unittest.TestCase):
    def setUp(self):
        configure_identity_debug(True, None)
        self.addCleanup(configure_identity_debug, False, None)

    def test_suppression_carries_the_same_cycle_and_candidate_id(self):
        people = [
            person("cam_1", 17, 2, (100.0, 100.0), POSITION_QUALITY_HARD),
            person("cam_2", 16, 1, (120.0, 100.0), POSITION_QUALITY_SOFT, "feet_occluded_by_other_detection"),
        ]
        with patch("camera_fusion.identity_event") as event:
            suppress_display_duplicates(people, fusion_cycle_id=11)

        record = next(
            call.kwargs for call in event.call_args_list
            if call.args == ("cross_camera_display_duplicate_suppressed",)
        )
        self.assertEqual(record["fusion_cycle_id"], 11)
        # The same name the association stage used, so the two join directly.
        self.assertEqual(record["candidate_id"], "cam_1#17~cam_2#16")
        self.assertEqual(
            record["candidate_id"],
            candidate_diagnostic_key(people[0]["observations"][0], people[1]["observations"][0]),
        )
        self.assertEqual(record["kept_master"], 2)
        self.assertEqual(record["suppressed_master"], 1)
        self.assertTrue(record["identity_disagreement"])


class DiagnosticKeyTests(unittest.TestCase):
    def test_keys_are_deterministic_and_identity_independent(self):
        left = observation("cam_1", 17, 2, (0.0, 0.0))
        right = observation("cam_2", 16, 1, (0.0, 0.0))
        key = candidate_diagnostic_key(left, right)

        self.assertEqual(observation_diagnostic_key(left), "cam_1#17")
        self.assertEqual(key, "cam_1#17~cam_2#16")
        # The master is what a reader is investigating, so it must not rename
        # the candidate midway through the investigation.
        left["identity_id"] = 99
        right["identity_id"] = None
        self.assertEqual(candidate_diagnostic_key(left, right), key)


class StatusVocabularyTests(unittest.TestCase):
    def test_identity_status_names_every_case(self):
        def obs(identity, group=None):
            o = observation("cam", 1, identity, (0.0, 0.0))
            o["temporary_group_id"] = group
            return o

        self.assertEqual(_identity_status(obs(1), obs(1)), "same_master")
        self.assertEqual(_identity_status(obs(1), obs(2)), "different_master")
        self.assertEqual(_identity_status(obs(None), obs(None)), "identity_missing_both")
        self.assertEqual(_identity_status(obs(1), obs(None)), "identity_missing_one")
        self.assertEqual(
            _identity_status(obs(None, "tmp_1"), obs(None, "tmp_1")), "same_temporary_group"
        )

    def test_physical_status_is_independent_of_identity(self):
        self.assertEqual(_physical_pairing_status(None, 50.0, 125.0, False), "no_shared_geometry")
        self.assertEqual(_physical_pairing_status(10.0, 50.0, 125.0, False), "spatially_compatible")
        self.assertEqual(_physical_pairing_status(10.0, 50.0, 125.0, True), "established_pair")
        self.assertEqual(_physical_pairing_status(80.0, 50.0, 125.0, True), "within_tolerance_band")
        self.assertEqual(_physical_pairing_status(200.0, 50.0, 125.0, True), "spatially_incompatible")

    def test_flags_stay_empty_for_an_ordinary_candidate(self):
        record = {
            "distance_cm": 10.0,
            "distance_limit_cm": 50.0,
            "identity_status": "same_master",
            "left_quality": POSITION_QUALITY_HARD,
            "right_quality": POSITION_QUALITY_HARD,
        }
        self.assertEqual(_candidate_flags(record), [])


if __name__ == "__main__":
    unittest.main()
