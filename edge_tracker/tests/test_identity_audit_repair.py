"""Periodic re-checking of settled bindings.

When two people cross, the local tracker can hand each of them the other's box.
Nothing in the identity layer notices: both tracks stay bound, both keep their
master, and the tactical map shows two IDs sitting on the wrong bodies until
one of them leaves.  The audit re-compares every settled binding against all
galleries on an interval so that swap repairs itself.

Repair has to be harder than detection.  Two people in similar clothing sit
near the matching threshold, and a repair that acts on a single lucky frame
would trade their identities back and forth every cycle -- worse than the swap
it set out to fix.  A rival master therefore has to win by a margin, and win
repeatedly, before anything moves.
"""

import time
import unittest

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


class StubExtractor:
    """Returns whatever the test says the current crop looks like."""

    def __init__(self, feature):
        self.feature = np.asarray(feature, dtype=np.float32)
        self.calls = 0

    def set(self, feature):
        self.feature = np.asarray(feature, dtype=np.float32)

    def extract_many_aligned(self, crops):
        self.calls += len(crops)
        return [self.feature.copy() for _ in crops]

    @staticmethod
    def feature_space_id(_dimension):
        return "test-feature-space"


class IdentityAuditTests(unittest.TestCase):
    def setUp(self):
        self.extractor = StubExtractor((1.0, 0.0, 0.0))
        self.memory = AppearanceIdentityMemory(
            reid_extractor=self.extractor,
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            identity_audit_rounds=2,
            start_worker=False,
        )
        # Two settled people: master 1 looks like x, master 2 looks like y.
        for identity_id, feature in ((1, (1.0, 0.0, 0.0)), (2, (0.0, 1.0, 0.0))):
            record = self.memory._new_record(identity_state="confirmed")
            record["gallery"]["baseline"] = make_slot(feature)
            self.memory.identities[identity_id] = record
        self.key = ("cam_1", 10)
        self.memory.track_to_identity[self.key] = 1

    def tearDown(self):
        self.memory.close(drain=False)

    def audit(self):
        self.memory._process_identity_audit_task(
            {
                "type": "identity_audit",
                "track_key": self.key,
                "identity_id": self.memory.track_to_identity.get(self.key),
                "camera_id": "cam_1",
                "frame_index": 1,
                "crop": np.zeros((8, 8, 3), dtype=np.uint8),
            }
        )

    def test_correct_binding_is_left_alone(self):
        self.audit()
        self.audit()
        self.assertEqual(self.memory.track_to_identity[self.key], 1)

    def test_swapped_track_is_repaired_after_repeated_agreement(self):
        # The box now sits on the other person.
        self.extractor.set((0.0, 1.0, 0.0))

        self.audit()
        self.assertEqual(
            self.memory.track_to_identity[self.key],
            1,
            "one contradicting frame must not move a binding",
        )

        self.audit()
        self.assertEqual(self.memory.track_to_identity[self.key], 2)
        self.assertIn(self.key, self.memory.identities[2]["member_track_keys"])
        self.assertEqual(
            self.memory.assignment_metadata(10, camera_id="cam_1")["confirmation_reason"],
            "identity_audit",
        )

    def test_a_single_stray_frame_cannot_accumulate(self):
        # Alternating verdicts are what two similar-looking people produce.
        # The tally must reset, or they eventually trade identities.
        for _ in range(6):
            self.extractor.set((0.0, 1.0, 0.0))
            self.audit()
            self.extractor.set((1.0, 0.0, 0.0))
            self.audit()
        self.assertEqual(self.memory.track_to_identity[self.key], 1)

    def test_repair_never_gives_one_master_two_visible_owners(self):
        other_key = ("cam_1", 11)
        self.memory.track_to_identity[other_key] = 2
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.key, other_key}
        self.extractor.set((0.0, 1.0, 0.0))

        self.audit()
        self.audit()

        self.assertEqual(self.memory.track_to_identity[self.key], 1)
        self.assertEqual(self.memory.track_to_identity[other_key], 2)

    def test_rebinding_clears_the_tally(self):
        self.extractor.set((0.0, 1.0, 0.0))
        self.audit()
        with self.memory._lock:
            self.memory._clear_local_binding_locked(self.key)
        self.assertNotIn(self.key, self.memory.identity_audit_state)


class AuditSchedulingTests(unittest.TestCase):
    """The audit must not run on crops that cannot answer the question."""

    def setUp(self):
        self.extractor = StubExtractor((1.0, 0.0, 0.0))
        self.memory = AppearanceIdentityMemory(
            reid_extractor=self.extractor,
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            blur_threshold=1.0,
            identity_audit_interval_seconds=5.0,
            start_worker=False,
        )
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0, 0.0))
        self.memory.identities[1] = record
        self.key = ("cam_1", 10)
        self.memory.track_to_identity[self.key] = 1
        self.crop = np.full((64, 32, 3), 255, dtype=np.uint8)
        self.crop[::2] = 0  # sharp enough to clear the blur gate

    def tearDown(self):
        self.memory.close(drain=False)

    def schedule(self, now, body_complete=True):
        with self.memory._lock:
            self.memory._schedule_identity_audit_locked(
                self.key,
                1,
                self.crop,
                frame_index=int(now),
                detection_confidence=0.9,
                now=now,
                body_complete=body_complete,
            )
        return self.memory._task_queue.qsize()

    def test_first_sighting_starts_the_clock_without_auditing(self):
        self.assertEqual(self.schedule(100.0), 0)

    def test_audit_runs_once_the_interval_elapses(self):
        self.schedule(100.0)
        self.assertEqual(self.schedule(104.0), 0)
        self.assertEqual(self.schedule(106.0), 1)

    def test_partial_body_is_never_audited(self):
        self.schedule(100.0)
        self.assertEqual(self.schedule(200.0, body_complete=False), 0)

    def test_master_under_physical_conflict_is_not_audited(self):
        self.schedule(100.0)
        self.memory.physical_conflicts[1] = {"token": 1, "candidates": {}}
        self.assertEqual(self.schedule(200.0), 0)


class AuditYieldsToContestTests(unittest.TestCase):
    """Two repair paths must not pull on the same track at once.

    A contest weighs two claimants against each other with several crops
    apiece; the audit judges one track on one crop.  When the audit reassigned
    a claimant in the middle of an arbitration, the contest was left waiting on
    a track that had been moved to another master -- 35 seconds of wrong IDs in
    one recorded huddle, ending only when the audit happened to put it back.

    Standing aside indefinitely is the worse failure, so a contest that has run
    far past any reasonable length releases its claim on the audit.
    """

    def setUp(self):
        self.extractor = StubExtractor((0.0, 1.0, 0.0))
        self.memory = AppearanceIdentityMemory(
            reid_extractor=self.extractor,
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            identity_audit_rounds=1,  # one strike, so a move is one audit away
            start_worker=False,
        )
        for identity_id, feature in ((1, (1.0, 0.0, 0.0)), (2, (0.0, 1.0, 0.0))):
            record = self.memory._new_record(identity_state="confirmed")
            record["gallery"]["baseline"] = make_slot(feature)
            self.memory.identities[identity_id] = record
        self.key = ("cam_2", 16)
        self.memory.track_to_identity[self.key] = 1

    def tearDown(self):
        self.memory.close(drain=False)

    def start_contest(self, age_seconds=0.0, include_track=True):
        self.memory.physical_conflicts[1] = {
            "token": 99,
            "candidates": {
                (self.key if include_track else ("cam_1", 90)): [],
                ("cam_1", 18): [],
            },
            "last_frames": {},
            "submitted": False,
            "attempts": 0,
            "started_frame": 0,
            "started_monotonic": time.monotonic() - age_seconds,
        }

    def audit(self):
        self.memory._process_identity_audit_task(
            {
                "type": "identity_audit",
                "track_key": self.key,
                "identity_id": self.memory.track_to_identity.get(self.key),
                "camera_id": "cam_2",
                "frame_index": 1,
                "crop": np.zeros((8, 8, 3), dtype=np.uint8),
            }
        )

    def test_without_a_contest_the_audit_moves_the_track(self):
        self.audit()
        self.assertEqual(self.memory.track_to_identity[self.key], 2)

    def test_a_live_contest_freezes_the_binding(self):
        self.start_contest()
        self.audit()
        self.assertEqual(
            self.memory.track_to_identity[self.key],
            1,
            "the contest must be allowed to finish with both claimants present",
        )

    def test_the_audit_does_not_bank_a_strike_while_yielding(self):
        self.start_contest()
        self.audit()
        self.assertFalse(self.memory.identity_audit_state.get(self.key, {}).get("rivals"))

    def test_a_contest_between_other_tracks_is_none_of_its_business(self):
        self.start_contest(include_track=False)
        self.audit()
        self.assertEqual(self.memory.track_to_identity[self.key], 2)

    def test_a_stalled_contest_stops_blocking_repairs(self):
        # A huddle can starve arbitration of clean crops for as long as it
        # lasts; the audit must not be muted for that whole time.
        self.start_contest(age_seconds=10_000.0)
        self.audit()
        self.assertEqual(self.memory.track_to_identity[self.key], 2)


if __name__ == "__main__":
    unittest.main()


class WithdrawContributionsTests(unittest.TestCase):
    """A repair must take back what the track left behind.

    Moving the track and leaving its photographs where they were is how one
    stray crop of Haoran ended up permanently in Mikail's gallery.  Two of
    those in a single huddle blended the two identities into an average of
    both, after which no rival could win by a margin and the audit was left
    with nothing to tell them apart.
    """

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        self.key = ("cam_2", 16)
        self.record = self.memory._new_record(identity_state="confirmed")
        self.memory.identities[1] = self.record

    def tearDown(self):
        self.memory.close(drain=False)

    def slot(self, contributor=None, captured_at=1.0, sharpness=200.0):
        s = make_slot((1.0, 0.0, 0.0))
        s["contributed_by_track_key"] = contributor
        s["captured_at"] = captured_at
        s["sharpness"] = sharpness
        return s

    def withdraw(self, since=None):
        with self.memory._lock:
            return self.memory._withdraw_track_contributions_locked(self.key, 1, since)

    def test_a_crop_from_the_departing_track_is_taken_back(self):
        self.record["gallery"]["baseline"] = self.slot(("cam_1", 3))
        self.record["gallery"]["left_side"] = self.slot(self.key)

        self.assertTrue(self.withdraw())
        self.assertIsNone(self.record["gallery"]["left_side"])

    def test_other_tracks_contributions_are_left_alone(self):
        keeper = self.slot(("cam_1", 3))
        self.record["gallery"]["baseline"] = keeper
        self.record["gallery"]["front"] = self.slot(("cam_1", 3))

        self.assertEqual(self.withdraw(), ())
        self.assertIs(self.record["gallery"]["baseline"], keeper)

    def test_crops_taken_before_the_last_passing_audit_are_kept(self):
        # A passing audit vouched for the binding at that moment, so what came
        # before it is trusted; only what came after is suspect.
        self.record["gallery"]["baseline"] = self.slot(("cam_1", 3))
        self.record["gallery"]["front"] = self.slot(self.key, captured_at=5.0)
        self.record["gallery"]["back"] = self.slot(self.key, captured_at=25.0)

        self.withdraw(since=20.0)

        self.assertIsNotNone(self.record["gallery"]["front"])
        self.assertIsNone(self.record["gallery"]["back"])

    def test_per_camera_views_and_baselines_are_withdrawn_too(self):
        self.record["gallery"]["baseline"] = self.slot(("cam_1", 3))
        self.record["camera_baselines"] = {"cam_2": self.slot(self.key)}
        self.record["camera_views"] = {"cam_2": {"front": self.slot(self.key)}}

        self.withdraw()

        self.assertNotIn("cam_2", self.record["camera_baselines"])
        self.assertIsNone(self.record["camera_views"]["cam_2"]["front"])

    def test_an_identity_is_never_left_without_a_baseline(self):
        # Every later comparison is measured against the baseline, so losing it
        # while other views survive would strand the identity.
        self.record["gallery"]["baseline"] = self.slot(self.key)
        self.record["gallery"]["front"] = self.slot(("cam_1", 3), sharpness=900.0)

        self.withdraw()

        self.assertIsNotNone(self.record["gallery"]["baseline"])
        self.assertEqual(self.record["gallery"]["baseline"]["sharpness"], 900.0)

    def test_unattributed_crops_are_not_touched(self):
        # Galleries restored from a previous run carry no contributor, and a
        # withdrawal must never guess.
        self.record["gallery"]["baseline"] = self.slot(None)
        self.assertEqual(self.withdraw(), ())
        self.assertIsNotNone(self.record["gallery"]["baseline"])
