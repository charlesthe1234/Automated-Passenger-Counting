import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from constants import DEFAULT_POSITION_SPLIT_FRAMES
from reid_memory import AppearanceIdentityMemory
from tests.test_reid_intake_lifecycle import (
    CountingBatchExtractor,
    process_tracks,
    sharp_frame,
)


class StaleGenerationExtractor:
    def __init__(self):
        self.calls = 0
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def extract_many_aligned(self, crops):
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            if not self.release_first.wait(timeout=2.0):
                raise RuntimeError("test gate timed out")
            feature = np.array([1.0, 0.0], dtype=np.float32)
        else:
            feature = np.array([0.0, 1.0], dtype=np.float32)
        return [feature.copy() for _ in crops]


class SecondBatchGateExtractor:
    def __init__(self):
        self.calls = 0
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def extract_many_aligned(self, crops):
        self.calls += 1
        if self.calls == 2:
            self.second_started.set()
            if not self.release_second.wait(timeout=2.0):
                raise RuntimeError("test gate timed out")
        feature = np.array([1.0, 0.0], dtype=np.float32)
        return [feature.copy() for _ in crops]


class FailOnceExtractor:
    def __init__(self, always_fail=False):
        self.calls = 0
        self.always_fail = always_fail

    def extract_many_aligned(self, crops):
        self.calls += 1
        if self.always_fail or self.calls == 1:
            raise RuntimeError("intentional extractor failure")
        feature = np.array([1.0, 0.0], dtype=np.float32)
        return [feature.copy() for _ in crops]


class FakeRoleClassifier:
    def __init__(self, role, confidence):
        self.role = role
        self.confidence = confidence

    def predict(self, _crop):
        return self.role, self.confidence


class ReIDSafetyEdgeTests(unittest.TestCase):
    def setUp(self):
        self.memories = []

    def tearDown(self):
        for memory in self.memories:
            memory.close(drain=False, timeout=0.2)

    def make_memory(self, extractor, **kwargs):
        memory = AppearanceIdentityMemory(
            reid_extractor=extractor,
            intake_frames=5,
            intake_delay_seconds=0.0,
            intake_timeout_seconds=1.0,
            blur_threshold=1.0,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            **kwargs,
        )
        self.memories.append(memory)
        return memory

    def test_stale_generation_cannot_bind_reused_local_track(self):
        extractor = StaleGenerationExtractor()
        memory = self.make_memory(extractor, ttl_frames=2)
        crop = sharp_frame()[10:100, 10:60]
        try:
            for frame_index in range(1, 6):
                memory.assign(7, crop, frame_index, camera_id="cam_1", observed_at=float(frame_index))
            self.assertTrue(extractor.first_started.wait(timeout=1.0))

            for frame_index in range(100, 105):
                memory.assign(7, crop, frame_index, camera_id="cam_1", observed_at=float(frame_index))
            extractor.release_first.set()
            self.assertTrue(memory.wait_for_idle(timeout=2.0))

            identity_id = memory.assign(7, crop, 105, camera_id="cam_1", observed_at=105.0)[0]
            self.assertEqual(identity_id, 1)
            self.assertEqual(len(memory.identities), 1)
            self.assertEqual(memory.next_identity_id, 2)
            baseline = memory.identities[1]["gallery"]["baseline"]["feature"]
            self.assertTrue(np.allclose(baseline, np.array([0.0, 1.0], dtype=np.float32)))
        finally:
            extractor.release_first.set()

    def test_mapped_cross_camera_teleport_waits_for_both_claimants(self):
        memory = self.make_memory(
            CountingBatchExtractor(),
            cross_camera_fusion_distance_cm=50.0,
            cross_camera_max_skew_seconds=0.35,
        )
        crop = sharp_frame()[10:100, 10:60]

        for frame_index in range(1, 6):
            memory.assign(
                1, crop, frame_index, camera_id="cam_1",
                observed_at=9.99 + frame_index * 0.01, map_point=(0.0, 0.0),
            )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(memory.assign(1, crop, 6, camera_id="cam_1", observed_at=10.05, map_point=(0.0, 0.0))[0], 1)

        for frame_index in range(1, 6):
            memory.assign(
                7, crop, frame_index, camera_id="cam_2",
                observed_at=10.05 + frame_index * 0.01, map_point=(5.0, 0.0),
            )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(memory.assign(7, crop, 6, camera_id="cam_2", observed_at=10.11, map_point=(5.0, 0.0))[0], 1)

        # Arbitration opens only once the teleport persists: a single frame of
        # it is indistinguishable from a distorted ground point.
        teleport_end = 7 + DEFAULT_POSITION_SPLIT_FRAMES - 1
        for frame_index in range(7, teleport_end + 1):
            held = memory.assign(
                7, crop, frame_index, camera_id="cam_2",
                observed_at=10.15 + (frame_index - 7) * 0.01, map_point=(200.0, 0.0),
            )[0]
        self.assertEqual(held, 1)
        self.assertEqual(memory.lookup(7, camera_id="cam_2"), 1)
        self.assertIn(1, memory.physical_conflicts)

        for frame_index in range(teleport_end + 1, teleport_end + 5):
            memory.assign(
                7, crop, frame_index, camera_id="cam_2",
                observed_at=10.15 + (frame_index - 7) * 0.01, map_point=(200.0, 0.0),
            )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(
            memory.assign(
                7, crop, teleport_end + 5, camera_id="cam_2",
                observed_at=10.15 + (teleport_end - 2) * 0.01, map_point=(200.0, 0.0),
            )[0],
            1,
        )
        self.assertEqual(len(memory.identities), 1)

    def test_submission_time_same_camera_peer_remains_excluded(self):
        extractor = SecondBatchGateExtractor()
        memory = self.make_memory(extractor)
        try:
            for frame_index in range(1, 6):
                process_tracks(memory, frame_index, [10, 20], camera_id="cam_1")
            self.assertTrue(extractor.second_started.wait(timeout=1.0))
            self.assertEqual(memory.lookup(10, camera_id="cam_1"), 1)

            memory.mapped_identity_ids([20], camera_id="cam_1", frame_index=6)
            extractor.release_second.set()
            self.assertTrue(memory.wait_for_idle(timeout=2.0))
            self.assertEqual(memory.lookup(10, camera_id="cam_1"), 1)
            self.assertEqual(memory.lookup(20, camera_id="cam_1"), 2)
        finally:
            extractor.release_second.set()

    def test_failed_intake_uses_backoff_and_bounded_fresh_samples(self):
        extractor = FailOnceExtractor()
        memory = self.make_memory(extractor, intake_retry_frames=3)
        for frame_index in range(1, 6):
            process_tracks(memory, frame_index, [1])
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(extractor.calls, 1)

        for frame_index in range(6, 11):
            process_tracks(memory, frame_index, [1])
            state = memory.pending_intake.get(("cam_1", 1))
            self.assertLessEqual(len(state["samples"]), 5)
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(extractor.calls, 2)
        self.assertEqual(process_tracks(memory, 11, [1])[0]["identity_id"], 1)

    def test_always_failing_extractor_never_retries_every_frame(self):
        extractor = FailOnceExtractor(always_fail=True)
        memory = self.make_memory(extractor, intake_retry_frames=3, max_retry_frames=12)
        for frame_index in range(1, 31):
            process_tracks(memory, frame_index, [1])
            memory.wait_for_idle(timeout=0.2)
            state = memory.pending_intake.get(("cam_1", 1))
            if state is not None:
                self.assertLessEqual(len(state["samples"]), 5)
        self.assertLessEqual(extractor.calls, 4)
        self.assertEqual(memory.identities, {})

    def test_feature_spaces_are_a_hard_matching_boundary(self):
        memory = self.make_memory(CountingBatchExtractor())
        with memory._lock:
            record = memory._new_record()
            record["gallery"]["baseline"] = {
                "feature": np.array([1.0, 0.0], dtype=np.float32),
                "feature_space_id": "transreid-space",
            }
            memory.identities[1] = record
        query = np.array([1.0, 0.0], dtype=np.float32)
        self.assertIsNone(memory.find_matching_identity(query, feature_space_id="color-space")[0])
        self.assertEqual(memory.find_matching_identity(query, feature_space_id="transreid-space")[0], 1)

    def test_color_fallback_binding_is_never_transreid_confirmed(self):
        memory = self.make_memory(None)
        for frame_index in range(1, 6):
            process_tracks(memory, frame_index, [1])
        self.assertTrue(memory.wait_for_idle())
        point = process_tracks(memory, 6, [1])[0]
        self.assertEqual(point["identity_id"], 1)
        self.assertFalse(point["reid_confirmed"])

    def test_low_confidence_staff_role_is_kept_when_nothing_beats_it(self):
        memory = self.make_memory(CountingBatchExtractor())
        memory.enable_role_classification = True
        memory._role_classifier = FakeRoleClassifier("scdf", 0.34)
        for frame_index in range(1, 6):
            process_tracks(memory, frame_index, [1])
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(memory.identities[1]["role"], "scdf")
        self.assertEqual(memory.identities[1]["age"], "N/A")

    def test_a_more_confident_vote_overrules_a_doubtful_one(self):
        memory = self.make_memory(None)
        record = memory._new_record("evacuee", 0.4152)
        self.assertTrue(memory._apply_role_vote_locked(record, "cag", 0.9876))
        self.assertEqual(record["role"], "cag")
        self.assertAlmostEqual(record["role_confidence"], 0.9876)

    def test_a_doubtful_vote_cannot_overrule_a_confident_one(self):
        memory = self.make_memory(None)
        record = memory._new_record("cag", 0.9876)
        self.assertFalse(memory._apply_role_vote_locked(record, "evacuee", 0.4152))
        self.assertEqual(record["role"], "cag")
        self.assertAlmostEqual(record["role_confidence"], 0.9876)

    def test_the_first_vote_wins_on_an_unclassified_record(self):
        memory = self.make_memory(None)
        record = memory._new_record(identity_state="provisional")
        self.assertFalse(record["role_classified"])
        self.assertTrue(memory._apply_role_vote_locked(record, "cag", 0.4803))
        self.assertEqual(record["role"], "cag")
        self.assertTrue(record["role_classified"])

    def test_a_staff_vote_drops_the_stashed_demographics_crops(self):
        memory = self.make_memory(None)
        record = memory._new_record("evacuee", 0.4152)
        record["pending_demographics_crops"] = [sharp_frame()]
        memory._apply_role_vote_locked(record, "cag", 0.9876)
        self.assertNotIn("pending_demographics_crops", record)
        self.assertEqual(record["age"], "N/A")
        self.assertEqual(record["gender"], "N/A")

    def test_evidence_failure_does_not_commit_an_unpaired_vector(self):
        memory = AppearanceIdentityMemory(
            reid_extractor=CountingBatchExtractor(),
            intake_frames=5,
            intake_delay_seconds=0.0,
            blur_threshold=1.0,
            evidence_dir=Path("never_written_evidence"),
            enable_role_classification=False,
            enable_demographics=False,
        )
        self.memories.append(memory)
        with mock.patch.object(
            memory,
            "_send_evidence_task",
            return_value={"ok": False, "error": "intentional evidence failure"},
        ):
            for frame_index in range(1, 6):
                process_tracks(memory, frame_index, [1])
            self.assertTrue(memory.wait_for_idle())
        self.assertEqual(memory.identities, {})


if __name__ == "__main__":
    unittest.main()
