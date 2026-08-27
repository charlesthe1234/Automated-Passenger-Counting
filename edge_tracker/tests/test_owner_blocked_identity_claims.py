"""A master another box is holding must still be compared against.

One ID can only sit on one box per camera, so a master already held is removed
from the candidate list before intake compares anything.  That is correct while
the holder is the right person.  When two people cross and the tracker hands
one of them the other's box, the holder is an impostor -- and the man whose
identity it is gets compared only against the people he is not, finds no match,
and is issued a duplicate ID for someone who already exists.

The periodic audit does eventually notice and move the impostor off.  In the
run that motivated this it had already flagged the error four seconds earlier
and was one cycle from repairing it, but intake asked its question 234ms before
that cycle completed.  Waiting on the audit is not a fix; the answer has to be
available at the moment the question is asked.

Scoring a held master cannot bind it -- the single-owner rule still stands.  It
only makes the match visible, so the holder can be put on trial for it.
"""

import unittest
from unittest import mock

import numpy as np

from reid_memory import AppearanceIdentityMemory


def make_slot(feature, camera_id="cam_1", frame_index=1):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    feature = feature / np.linalg.norm(feature)
    return {
        "feature": feature,
        "feature_source": "transreid",
        "feature_space_id": "test-feature-space",
        "feature_dimension": int(feature.size),
        "image_path": None,
        "digest": None,
        "captured_frame": int(frame_index),
        "captured_at": float(frame_index),
        "camera_id": camera_id,
        "sharpness": 200.0,
        "detection_confidence": 0.95,
    }


class OwnerBlockedScoringTests(unittest.TestCase):
    """The match has to be measured before anything can act on it."""

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        for identity_id, feature in ((1, (1.0, 0.0, 0.0)), (2, (0.0, 1.0, 0.0))):
            record = self.memory._new_record(identity_state="confirmed")
            record["gallery"]["baseline"] = make_slot(feature)
            self.memory.identities[identity_id] = record

    def tearDown(self):
        self.memory.close(drain=False)

    def match(self, feature, excluded, collect=True):
        blocked = [] if collect else None
        with self.memory._lock:
            result = self.memory._matching_identity_locked(
                self.memory._normalize_feature(np.asarray(feature, dtype=np.float32)),
                query_feature_space_id="test-feature-space",
                excluded_identity_ids=excluded,
                owner_blocked_matches=blocked,
            )
        return result, blocked

    def test_held_master_is_scored_but_never_returned(self):
        (identity_id, _slot, _distance), blocked = self.match((1.0, 0.02, 0.0), {1})

        self.assertIsNone(identity_id, "a held master must not be bound")
        self.assertEqual([m["identity_id"] for m in blocked], [1])
        self.assertLess(blocked[0]["distance"], 0.05)

    def test_a_weak_resemblance_is_recorded_but_will_not_contest(self):
        # Every blocked master is scored so a duplicate ID can be explained
        # afterwards; the contest applies its own bar to the same list.  Keeping
        # only contest-worthy scores hid exactly the near-misses that matter.
        _result, blocked = self.match((0.0, 1.0, 0.0), {1})

        self.assertEqual([m["identity_id"] for m in blocked], [1])
        self.assertGreater(blocked[0]["distance"], self.memory.strong_match_distance)

    def test_behaviour_is_unchanged_when_the_caller_does_not_ask(self):
        (identity_id, _slot, _distance), blocked = self.match(
            (1.0, 0.02, 0.0), {1}, collect=False
        )
        self.assertIsNone(identity_id)
        self.assertIsNone(blocked)

    def test_an_eligible_master_still_wins_normally(self):
        (identity_id, _slot, distance), _blocked = self.match((0.0, 1.0, 0.02), {1})
        self.assertEqual(identity_id, 2)
        self.assertLess(distance, 0.05)


class OwnerBlockedContestTests(unittest.TestCase):
    """Who gets challenged, and who is left alone."""

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0, 0.0))
        self.memory.identities[2] = record
        self.incumbent = ("cam_1", 14)
        self.challenger = ("cam_1", 17)
        self.memory.track_to_identity[self.incumbent] = 2
        self.memory.visible_track_keys_by_camera["cam_1"] = {
            self.incumbent,
            self.challenger,
        }
        # Two people standing apart, not one person detected twice.
        self.memory.track_boxes[self.incumbent] = (10, 10, 60, 210)
        self.memory.track_boxes[self.challenger] = (300, 10, 350, 210)
        self.task = {
            "track_key": self.challenger,
            "frame_index": 100,
            "generation": 1,
            "samples": [
                {
                    "crop": np.zeros((64, 32, 3), dtype=np.uint8),
                    "frame_index": 100 + i,
                    "sharpness": 200.0,
                    "detection_confidence": 0.9,
                }
                for i in range(5)
            ],
        }

    def tearDown(self):
        self.memory.close(drain=False)

    def contest(self, distance=0.05):
        with self.memory._lock:
            return self.memory._contest_owner_blocked_master_locked(
                [{"identity_id": 2, "matched_slot": "baseline", "distance": distance}],
                self.challenger,
                "cam_1",
                self.task,
                "transreid",
                "test-feature-space",
            )

    def test_the_holder_is_challenged_for_a_strong_match(self):
        self.assertTrue(self.contest())
        self.assertIn(2, self.memory.physical_conflicts)
        conflict = self.memory.physical_conflicts[2]
        self.assertEqual(conflict["challenger_key"], self.challenger)
        self.assertIn(self.incumbent, conflict["candidates"])

    def test_a_duplicate_box_never_contests_the_original(self):
        # Near-total overlap is one person detected twice; the shadow
        # machinery owns that case and a ghost must not unseat its own track.
        self.memory.track_boxes[self.challenger] = (12, 12, 62, 212)
        self.assertFalse(self.contest())
        self.assertNotIn(2, self.memory.physical_conflicts)

    def test_a_lost_contest_is_not_refought(self):
        # Otherwise two people the model cannot separate contest, lose, and
        # contest again forever, and the newcomer never receives any ID.
        self.memory.physical_conflict_rejections[self.challenger] = {2}
        self.assertFalse(self.contest())

    def test_an_ambiguous_holder_is_not_put_on_trial(self):
        # With two live holders there is no single incumbent to compare
        # against, so the arbiter would have nothing to decide between.
        other = ("cam_1", 21)
        self.memory.track_to_identity[other] = 2
        self.memory.visible_track_keys_by_camera["cam_1"].add(other)
        self.assertFalse(self.contest())

    def test_a_weak_match_leaves_the_holder_alone(self):
        self.assertFalse(self.contest(distance=0.9))


if __name__ == "__main__":
    unittest.main()


class BlockedMasterReportingTests(unittest.TestCase):
    """The score of the master nobody looked at has to survive the decision.

    These numbers were computed and discarded unless they happened to start a
    contest, so the one question a duplicate ID raises afterwards -- how close
    was the right answer that was skipped -- had no answer in the log.
    """

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        self.key = ("cam_2", 34)

    def tearDown(self):
        self.memory.close(drain=False)

    def report(self, matches):
        events = []
        with mock.patch(
            "reid_memory.identity_event",
            side_effect=lambda name, **fields: events.append((name, fields)),
        ):
            with self.memory._lock:
                self.memory._report_owner_blocked_matches_locked(
                    matches, self.key, "cam_2", "normal_intake_match", 100
                )
        return events

    def test_a_near_miss_that_never_contested_is_still_recorded(self):
        events = self.report(
            [{"identity_id": 1, "matched_slot": "front", "distance": 0.23}]
        )

        self.assertEqual(len(events), 1)
        name, fields = events[0]
        self.assertEqual(name, "owner_blocked_masters_scored")
        self.assertEqual(fields["scored"][0]["master_id"], 1)
        self.assertAlmostEqual(fields["scored"][0]["distance"], 0.23)
        self.assertFalse(
            fields["scored"][0]["would_contest"],
            "0.23 sits above the contest bar -- which is the fact worth logging",
        )

    def test_scores_are_ordered_closest_first(self):
        _name, fields = self.report(
            [
                {"identity_id": 3, "matched_slot": "back", "distance": 0.40},
                {"identity_id": 1, "matched_slot": "front", "distance": 0.12},
            ]
        )[0]
        self.assertEqual([s["master_id"] for s in fields["scored"]], [1, 3])
        self.assertTrue(fields["scored"][0]["would_contest"])

    def test_nothing_is_logged_when_no_master_was_blocked(self):
        self.assertEqual(self.report([]), [])


class ProvisionalMemberChallengeTests(unittest.TestCase):
    """A group member holds a placeholder, not an identity, and may contest.

    Most identities are created by two boxes pairing up, so this is the path
    that matters.  It refused every contest, because the check asked whether
    the challenger already held anything and a group member holds its group's
    negative token.  Two boxes of a man who already had ID 3 recognised it at
    0.147 and 0.135, could not challenge for it, and were issued ID 6.
    """

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0, 0.0))
        self.memory.identities[3] = record
        self.incumbent = ("cam_1", 90)
        self.challenger = ("cam_1", 98)
        self.memory.track_to_identity[self.incumbent] = 3
        self.memory.visible_track_keys_by_camera["cam_1"] = {
            self.incumbent,
            self.challenger,
        }
        self.memory.track_boxes[self.incumbent] = (10, 10, 60, 210)
        self.memory.track_boxes[self.challenger] = (300, 10, 350, 210)
        self.task = {
            "track_key": self.challenger,
            "frame_index": 100,
            "generation": 1,
            "samples": [
                {
                    "crop": np.zeros((64, 32, 3), dtype=np.uint8),
                    "frame_index": 100 + i,
                    "sharpness": 200.0,
                    "detection_confidence": 0.9,
                }
                for i in range(5)
            ],
        }

    def tearDown(self):
        self.memory.close(drain=False)

    def contest(self):
        with self.memory._lock:
            return self.memory._contest_owner_blocked_master_locked(
                [{"identity_id": 3, "matched_slot": "baseline", "distance": 0.147}],
                self.challenger,
                "cam_1",
                self.task,
                "transreid",
                "test-feature-space",
            )

    def test_a_member_of_a_temporary_group_may_still_contest(self):
        self.memory.track_to_identity[self.challenger] = -13
        self.assertTrue(self.contest())
        self.assertIn(3, self.memory.physical_conflicts)

    def test_an_unbound_track_may_still_contest(self):
        self.assertTrue(self.contest())

    def test_a_track_that_already_owns_a_master_may_not(self):
        # It has an identity of its own, so it has nothing to win and must not
        # unseat anybody.
        other = self.memory._new_record(identity_state="confirmed")
        self.memory.identities[9] = other
        self.memory.track_to_identity[self.challenger] = 9
        self.assertFalse(self.contest())
