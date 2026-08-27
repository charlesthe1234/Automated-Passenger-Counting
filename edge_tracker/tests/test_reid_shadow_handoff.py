import threading
import unittest

import numpy as np

from pose_engine import get_standing_points
from reid_memory import AppearanceIdentityMemory


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBoxes:
    def __init__(self, detections):
        self.xyxy = FakeTensor([item[1] for item in detections])
        self.id = FakeTensor([item[0] for item in detections])
        self.conf = FakeTensor([item[2] for item in detections])

    def __len__(self):
        return len(self.id.value)


class FakeResult:
    def __init__(self, detections):
        self.boxes = FakeBoxes(detections)
        self.keypoints = None


class SequenceBatchExtractor:
    """Return one deterministic feature per submitted intake batch."""

    def __init__(self, *features):
        self.features = [np.asarray(feature, dtype=np.float32) for feature in features]
        self.batch_sizes = []
        self._lock = threading.Lock()

    def extract_many_aligned(self, crops):
        with self._lock:
            call_index = len(self.batch_sizes)
            self.batch_sizes.append(len(crops))
        feature = self.features[min(call_index, len(self.features) - 1)]
        return [feature.copy() for _crop in crops]


class GatedSecondBatchExtractor(SequenceBatchExtractor):
    """Hold a handoff comparison until the candidate has disappeared."""

    def __init__(self, feature=(1.0, 0.0, 0.0)):
        super().__init__(feature, feature)
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def extract_many_aligned(self, crops):
        with self._lock:
            call_number = len(self.batch_sizes) + 1
        if call_number == 2:
            self.second_started.set()
            if not self.release_second.wait(timeout=2.0):
                raise RuntimeError("test gate timed out")
        return super().extract_many_aligned(crops)


CANONICAL_BOX = (20, 10, 70, 110)
GHOST_BOX = (21, 11, 71, 111)
SEPARATE_BOX = (90, 10, 140, 110)


def sharp_frame():
    yy, xx = np.indices((130, 170))
    checker = ((xx // 4 + yy // 4) % 2 * 255).astype(np.uint8)
    return np.repeat(checker[:, :, None], 3, axis=2)


def detection(track_id, box, confidence=0.95):
    return int(track_id), tuple(box), float(confidence)


def process(memory, frame_index, detections):
    return get_standing_points(
        FakeResult(detections),
        sharp_frame(),
        frame_index=frame_index,
        appearance_memory=memory,
        camera_id="cam_1",
        observation_time=float(frame_index) / 30.0,
        use_mediapipe_feet=False,
    )


class ReIDShadowHandoffTests(unittest.TestCase):
    def setUp(self):
        self.memories = []

    def tearDown(self):
        for memory in self.memories:
            memory.close(drain=False, timeout=0.2)

    def make_memory(self, extractor):
        memory = AppearanceIdentityMemory(
            reid_extractor=extractor,
            intake_frames=5,
            intake_delay_seconds=0.0,
            intake_timeout_seconds=1.0,
            blur_threshold=1.0,
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
        )
        self.memories.append(memory)
        return memory

    def establish_master_one(self, memory):
        for frame_index in range(1, 6):
            process(memory, frame_index, [detection(10, CANONICAL_BOX)])
        self.assertTrue(memory.wait_for_idle(timeout=2.0))
        point = process(memory, 6, [detection(10, CANONICAL_BOX)])[0]
        self.assertEqual(point["identity_id"], 1)

    def test_one_frame_overlapping_ghost_is_suppressed_without_intake(self):
        extractor = SequenceBatchExtractor((1.0, 0.0, 0.0))
        memory = self.make_memory(extractor)
        self.establish_master_one(memory)

        points = process(
            memory,
            7,
            [
                detection(10, CANONICAL_BOX),
                detection(20, GHOST_BOX, confidence=0.80),
            ],
        )
        self.assertFalse(points[0]["suppressed"])
        self.assertTrue(points[1]["suppressed"])
        self.assertTrue(memory.is_track_suppressed(20, camera_id="cam_1"))
        self.assertEqual(memory.pending_count(20, camera_id="cam_1"), 0)

        process(memory, 8, [detection(10, CANONICAL_BOX)])
        self.assertIsNone(memory.lookup(20, camera_id="cam_1"))
        self.assertEqual(memory.pending_count(20, camera_id="cam_1"), 0)
        self.assertEqual(set(memory.identities), {1})
        self.assertEqual(extractor.batch_sizes, [5])

    def test_replacement_track_hands_off_to_the_existing_master(self):
        extractor = SequenceBatchExtractor(
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        memory = self.make_memory(extractor)
        self.establish_master_one(memory)

        process(
            memory,
            7,
            [detection(10, CANONICAL_BOX), detection(20, GHOST_BOX)],
        )
        self.assertTrue(memory.is_track_suppressed(20, camera_id="cam_1"))

        # ByteTrack drops the canonical local ID while its overlapping
        # replacement survives. The replacement still has to provide the
        # normal five-crop evidence before ownership can transfer.
        for frame_index in range(8, 13):
            process(memory, frame_index, [detection(20, GHOST_BOX)])
        self.assertTrue(memory.wait_for_idle(timeout=2.0))
        point = process(memory, 13, [detection(20, GHOST_BOX)])[0]

        self.assertEqual(point["identity_id"], 1)
        self.assertTrue(point["reidentified"])
        self.assertFalse(point["suppressed"])
        self.assertEqual(memory.lookup(20, camera_id="cam_1"), 1)
        self.assertIsNone(memory.lookup(10, camera_id="cam_1"))
        self.assertEqual(set(memory.identities), {1})
        self.assertEqual(extractor.batch_sizes, [5, 5])

    def test_overlapping_real_people_that_separate_remain_distinct(self):
        extractor = SequenceBatchExtractor(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        memory = self.make_memory(extractor)
        self.establish_master_one(memory)

        # Geometry may nominate track 20 as a shadow for one ambiguous frame.
        process(
            memory,
            7,
            [detection(10, CANONICAL_BOX), detection(20, GHOST_BOX)],
        )
        self.assertTrue(memory.is_track_suppressed(20, camera_id="cam_1"))

        # Independent motion must release it into normal intake. Its different
        # appearance then creates a second master instead of stealing ID 1.
        # Two consecutive separated frames are required to reject the shadow
        # hypothesis, after which the normal five-crop intake starts.
        for frame_index in range(8, 14):
            process(
                memory,
                frame_index,
                [detection(10, CANONICAL_BOX), detection(20, SEPARATE_BOX)],
            )
        self.assertTrue(memory.wait_for_idle(timeout=2.0))
        points = process(
            memory,
            14,
            [detection(10, CANONICAL_BOX), detection(20, SEPARATE_BOX)],
        )

        self.assertFalse(memory.is_track_suppressed(20, camera_id="cam_1"))
        self.assertEqual([point["identity_id"] for point in points], [1, 2])
        self.assertEqual(set(memory.identities), {1, 2})
        self.assertEqual(extractor.batch_sizes, [5, 5])

    def test_persistent_overlap_gets_one_appearance_veto_and_can_be_real(self):
        extractor = SequenceBatchExtractor(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        memory = self.make_memory(extractor)
        self.establish_master_one(memory)

        for frame_index in range(7, 15):
            points = process(
                memory,
                frame_index,
                [detection(10, CANONICAL_BOX), detection(20, GHOST_BOX)],
            )
            if frame_index <= 9:
                self.assertTrue(points[1]["suppressed"])
        self.assertTrue(memory.wait_for_idle(timeout=2.0))

        points = process(
            memory,
            15,
            [detection(10, CANONICAL_BOX), detection(20, GHOST_BOX)],
        )
        self.assertEqual([point["identity_id"] for point in points], [1, 2])
        self.assertFalse(points[1]["suppressed"])
        self.assertEqual(extractor.batch_sizes, [5, 5])

    def test_persistent_matching_shadow_is_verified_once_then_hands_off(self):
        extractor = SequenceBatchExtractor(
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        memory = self.make_memory(extractor)
        self.establish_master_one(memory)

        for frame_index in range(7, 15):
            process(
                memory,
                frame_index,
                [detection(10, CANONICAL_BOX), detection(20, GHOST_BOX)],
            )
        self.assertTrue(memory.wait_for_idle(timeout=2.0))
        self.assertTrue(memory.is_track_suppressed(20, camera_id="cam_1"))
        self.assertEqual(set(memory.identities), {1})
        self.assertEqual(extractor.batch_sizes, [5, 5])

        # Once appearance has confirmed the duplicate, losing the old local
        # track transfers ownership immediately without a third GPU batch.
        point = process(memory, 15, [detection(20, GHOST_BOX)])[0]
        self.assertEqual(point["identity_id"], 1)
        self.assertTrue(point["reidentified"])
        self.assertIsNone(memory.lookup(10, camera_id="cam_1"))
        self.assertEqual(extractor.batch_sizes, [5, 5])

    def test_duplicate_during_original_intake_joins_provisional_group(self):
        extractor = SequenceBatchExtractor((1.0, 0.0, 0.0))
        memory = self.make_memory(extractor)

        process(memory, 1, [detection(10, CANONICAL_BOX)])
        process(memory, 2, [detection(10, CANONICAL_BOX)])
        for frame_index in range(3, 6):
            points = process(
                memory,
                frame_index,
                [detection(10, CANONICAL_BOX), detection(20, GHOST_BOX)],
            )
            self.assertTrue(points[1]["suppressed"])
            self.assertEqual(memory.pending_count(20, camera_id="cam_1"), 0)
        self.assertTrue(memory.wait_for_idle(timeout=2.0))
        self.assertEqual(set(memory.identities), {1})
        self.assertEqual(extractor.batch_sizes, [5])

        process(memory, 6, [detection(10, CANONICAL_BOX)])
        self.assertIsNone(memory.lookup(20, camera_id="cam_1"))
        self.assertEqual(set(memory.identities), {1})

    def test_one_frame_geometry_wobble_does_not_release_the_shadow(self):
        extractor = SequenceBatchExtractor((1.0, 0.0, 0.0))
        memory = self.make_memory(extractor)
        self.establish_master_one(memory)

        process(
            memory,
            7,
            [detection(10, CANONICAL_BOX), detection(20, GHOST_BOX)],
        )
        points = process(
            memory,
            8,
            [detection(10, CANONICAL_BOX), detection(20, SEPARATE_BOX)],
        )
        self.assertTrue(points[1]["suppressed"])
        points = process(
            memory,
            9,
            [detection(10, CANONICAL_BOX), detection(20, GHOST_BOX)],
        )
        self.assertTrue(points[1]["suppressed"])
        process(memory, 10, [detection(10, CANONICAL_BOX)])

        self.assertEqual(memory.pending_count(20, camera_id="cam_1"), 0)
        self.assertEqual(set(memory.identities), {1})
        self.assertEqual(extractor.batch_sizes, [5])

    def test_ordinary_reid_clears_a_nonvisible_same_camera_owner(self):
        extractor = SequenceBatchExtractor(
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        memory = self.make_memory(extractor)
        self.establish_master_one(memory)

        # No simultaneous overlap is available here, so this follows the
        # ordinary gallery matcher instead of the targeted shadow path.
        for frame_index in range(7, 12):
            process(memory, frame_index, [detection(99, CANONICAL_BOX)])
        self.assertTrue(memory.wait_for_idle(timeout=2.0))
        point = process(memory, 12, [detection(99, CANONICAL_BOX)])[0]

        self.assertEqual(point["identity_id"], 1)
        self.assertIsNone(memory.lookup(10, camera_id="cam_1"))
        self.assertEqual(memory.lookup(99, camera_id="cam_1"), 1)
        self.assertEqual(set(memory.identities), {1})

    def test_disappeared_handoff_candidate_cannot_commit_stale_worker_result(self):
        extractor = GatedSecondBatchExtractor()
        memory = self.make_memory(extractor)
        try:
            self.establish_master_one(memory)
            process(
                memory,
                7,
                [detection(10, CANONICAL_BOX), detection(20, GHOST_BOX)],
            )
            for frame_index in range(8, 13):
                process(memory, frame_index, [detection(20, GHOST_BOX)])
            self.assertTrue(extractor.second_started.wait(timeout=1.0))

            # Removing every detection must invalidate both the intake
            # generation and the stored handoff candidacy before GPU work
            # returns.
            process(memory, 13, [])
            extractor.release_second.set()
            self.assertTrue(memory.wait_for_idle(timeout=2.0))

            self.assertIsNone(memory.lookup(20, camera_id="cam_1"))
            self.assertEqual(memory.pending_count(20, camera_id="cam_1"), 0)
            self.assertEqual(set(memory.identities), {1})
            self.assertEqual(memory.next_identity_id, 2)
        finally:
            extractor.release_second.set()


if __name__ == "__main__":
    unittest.main()
