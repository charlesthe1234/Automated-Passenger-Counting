"""Regression tests for which crops reach MiVOLO, and when it is run again.

The intake burst is five consecutive frames of one camera, ranked for nothing
but the single baseline photo.  These cover the separate ranking demographics
does for itself, and the re-estimate a much closer view triggers.
"""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from constants import (
    DEFAULT_DEMOGRAPHICS_CROP_COUNT,
    DEFAULT_DEMOGRAPHICS_MAX_REFRESHES,
    DEFAULT_DEMOGRAPHICS_REFRESH_QUALITY_RATIO,
)
from demographics import DemographicsReading
from reid_memory import AppearanceIdentityMemory
from tests.test_reid_intake_lifecycle import CountingBatchExtractor, process_tracks


class _RecordingEngine:
    """Stands in for MiVOLO, keeping whatever crop sets it was handed."""

    def __init__(self, reading):
        self.reading = reading
        self.batches = []

    def analyze_batch(self, candidates):
        self.batches.append(list(candidates))
        return self.reading


class _FailingEngine:
    def analyze_batch(self, _candidates):
        raise RuntimeError("intentional demographics failure")


def crop(height=300, width=100):
    """A textured crop, so measured sharpness is a real number rather than 0."""
    yy, xx = np.indices((height, width))
    checker = ((xx // 4 + yy // 4) % 2 * 255).astype(np.uint8)
    return np.repeat(checker[:, :, None], 3, axis=2)


def sample(sharpness=100.0, face_box=None, height=300, width=100, **extra):
    return {
        "crop": crop(height, width),
        "sharpness": sharpness,
        "face_box": face_box,
        "body_bounds": (0.02, 0.02, 0.98, 0.91),
        "occluder_boxes": (),
        **extra,
    }


def face_box_of_width(fraction, top=0.02):
    """A face box spanning ``fraction`` of the crop width, square in a 300x100."""
    return (0.5 - fraction / 2.0, top, 0.5 + fraction / 2.0, top + fraction / 3.0)


class DemographicsRankingTests(unittest.TestCase):
    def test_a_crop_with_a_face_always_outranks_one_without(self):
        # However sharp a faceless crop is, it cannot tell the model an age.
        faceless = sample(sharpness=100000.0)
        with_face = sample(sharpness=1.0, face_box=face_box_of_width(0.4))
        self.assertGreater(
            AppearanceIdentityMemory._demographics_quality(with_face),
            AppearanceIdentityMemory._demographics_quality(faceless),
        )

    def test_within_the_face_tier_a_bigger_sharper_face_wins(self):
        small = sample(sharpness=100.0, face_box=face_box_of_width(0.2))
        large = sample(sharpness=100.0, face_box=face_box_of_width(0.6))
        self.assertGreater(
            AppearanceIdentityMemory._demographics_quality(large),
            AppearanceIdentityMemory._demographics_quality(small),
        )
        blurred = sample(sharpness=10.0, face_box=face_box_of_width(0.6))
        self.assertGreater(
            AppearanceIdentityMemory._demographics_quality(large),
            AppearanceIdentityMemory._demographics_quality(blurred),
        )

    def test_an_empty_crop_scores_nothing(self):
        self.assertEqual(
            AppearanceIdentityMemory._demographics_quality({"crop": None}),
            (0, 0.0),
        )

    def test_candidates_come_back_best_first_and_carry_their_framing(self):
        samples = [
            sample(sharpness=50.0),
            sample(sharpness=100.0, face_box=face_box_of_width(0.2)),
            sample(sharpness=100.0, face_box=face_box_of_width(0.6)),
        ]
        candidates = AppearanceIdentityMemory._demographics_candidates(samples)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]["face_box"], face_box_of_width(0.6))
        self.assertEqual(candidates[1]["face_box"], face_box_of_width(0.2))
        self.assertIsNone(candidates[2]["face_box"])
        self.assertEqual(candidates[0]["body_bounds"], (0.02, 0.02, 0.98, 0.91))

    def test_the_candidate_list_is_capped(self):
        samples = [sample(sharpness=float(index)) for index in range(1, 12)]
        candidates = AppearanceIdentityMemory._demographics_candidates(samples)
        self.assertEqual(len(candidates), DEFAULT_DEMOGRAPHICS_CROP_COUNT)

    def test_candidates_copy_their_crops_away_from_the_intake_buffer(self):
        # The intake buffer is reused frame to frame, so a candidate holding a
        # view into it would silently become someone else's pixels.
        original = sample()
        expected = original["crop"].copy()
        candidate = AppearanceIdentityMemory._demographics_candidates([original])[0]
        original["crop"][:] = 0
        self.assertTrue((candidate["crop"] == expected).all())

    def test_samples_without_a_crop_are_skipped(self):
        candidates = AppearanceIdentityMemory._demographics_candidates(
            [{"crop": None, "sharpness": 900.0}, sample()]
        )
        self.assertEqual(len(candidates), 1)

    def test_pool_quality_counts_only_face_bearing_crops(self):
        faceless = AppearanceIdentityMemory._demographics_candidates(
            [sample(sharpness=100000.0)]
        )
        self.assertEqual(AppearanceIdentityMemory._demographics_pool_quality(faceless), 0.0)
        with_face = AppearanceIdentityMemory._demographics_candidates(
            [sample(sharpness=1.0, face_box=face_box_of_width(0.4))]
        )
        self.assertGreater(AppearanceIdentityMemory._demographics_pool_quality(with_face), 0.0)


class DemographicsPoolTests(unittest.TestCase):
    def test_a_better_crop_displaces_the_worst_and_the_pool_stays_capped(self):
        pool = AppearanceIdentityMemory._demographics_candidates(
            [sample(sharpness=float(index)) for index in range(1, 6)]
        )
        merged = AppearanceIdentityMemory._merge_demographics_pool(
            pool,
            [sample(sharpness=10.0, face_box=face_box_of_width(0.5))],
        )
        self.assertEqual(len(merged), DEFAULT_DEMOGRAPHICS_CROP_COUNT)
        self.assertEqual(merged[0]["face_box"], face_box_of_width(0.5))

    def test_earlier_good_crops_are_kept_so_a_re_estimate_still_votes(self):
        # A single excellent look should improve the answer, not replace a
        # five-crop consensus with a one-crop guess.
        pool = AppearanceIdentityMemory._demographics_candidates(
            [sample(sharpness=100.0, face_box=face_box_of_width(0.3)) for _ in range(5)]
        )
        merged = AppearanceIdentityMemory._merge_demographics_pool(
            pool,
            [sample(sharpness=100.0, face_box=face_box_of_width(0.9))],
        )
        self.assertEqual(len(merged), DEFAULT_DEMOGRAPHICS_CROP_COUNT)
        self.assertEqual(sum(1 for entry in merged if entry["face_box"] is not None), 5)


class DemographicsRefreshTests(unittest.TestCase):
    def setUp(self):
        self.memories = []

    def tearDown(self):
        for memory in self.memories:
            memory.close(drain=False, timeout=0.2)

    def make_memory(self, enable_demographics=True):
        memory = AppearanceIdentityMemory(
            reid_extractor=CountingBatchExtractor(),
            intake_frames=5,
            intake_delay_seconds=0.0,
            blur_threshold=1.0,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=enable_demographics,
            demographics_engine=_RecordingEngine(
                DemographicsReading(30, "male", 0.9, 5, 5)
            ),
        )
        self.memories.append(memory)
        return memory

    def settled_record(self, memory, identity_id=1, quality=None):
        """A confirmed evacuee whose first estimate has already come back."""
        record = memory._new_record("evacuee", 0.9)
        record["age"] = 30
        record["gender"] = "male"
        pool = AppearanceIdentityMemory._demographics_candidates(
            [sample(sharpness=100.0, face_box=face_box_of_width(0.2))]
        )
        record["demographics_crop_pool"] = pool
        record["demographics_quality"] = (
            AppearanceIdentityMemory._demographics_pool_quality(pool)
            if quality is None
            else quality
        )
        memory.identities[identity_id] = record
        return record

    def test_a_far_better_face_triggers_a_re_estimate(self):
        memory = self.make_memory()
        record = self.settled_record(memory)
        before = record["demographics_quality"]
        pool = memory._consider_demographics_refresh_locked(
            1,
            [sample(sharpness=100.0, face_box=face_box_of_width(0.9))],
        )
        self.assertIsNotNone(pool)
        self.assertGreater(
            record["demographics_quality"],
            before * DEFAULT_DEMOGRAPHICS_REFRESH_QUALITY_RATIO,
        )
        self.assertEqual(record["demographics_refreshes"], 1)

    def test_a_marginally_better_face_is_not_worth_the_gpu(self):
        memory = self.make_memory()
        record = self.settled_record(memory)
        self.assertIsNone(
            memory._consider_demographics_refresh_locked(
                1,
                [sample(sharpness=105.0, face_box=face_box_of_width(0.21))],
            )
        )
        self.assertNotIn("demographics_refreshes", record)

    def test_the_pool_still_absorbs_a_crop_that_does_not_trigger_a_re_estimate(self):
        # The improvement is banked so a later crop is measured against the
        # best seen, not against the original intake.
        memory = self.make_memory()
        record = self.settled_record(memory)
        memory._consider_demographics_refresh_locked(
            1,
            [sample(sharpness=105.0, face_box=face_box_of_width(0.21))],
        )
        self.assertEqual(record["demographics_crop_pool"][0]["face_box"], face_box_of_width(0.21))

    def test_a_faceless_crop_never_triggers_a_re_estimate(self):
        memory = self.make_memory()
        self.settled_record(memory)
        self.assertIsNone(
            memory._consider_demographics_refresh_locked(1, [sample(sharpness=1e9)])
        )

    def test_the_first_face_seen_re_estimates_an_answer_taken_without_one(self):
        memory = self.make_memory()
        record = self.settled_record(memory, quality=0.0)
        record["demographics_crop_pool"] = AppearanceIdentityMemory._demographics_candidates(
            [sample(sharpness=500.0)]
        )
        self.assertIsNotNone(
            memory._consider_demographics_refresh_locked(
                1,
                [sample(sharpness=10.0, face_box=face_box_of_width(0.3))],
            )
        )

    def test_re_estimates_are_capped(self):
        memory = self.make_memory()
        record = self.settled_record(memory)
        record["demographics_refreshes"] = DEFAULT_DEMOGRAPHICS_MAX_REFRESHES
        self.assertIsNone(
            memory._consider_demographics_refresh_locked(
                1,
                [sample(sharpness=1000.0, face_box=face_box_of_width(0.95))],
            )
        )

    def test_an_estimate_still_in_flight_is_never_raced(self):
        memory = self.make_memory()
        record = self.settled_record(memory)
        record["age"] = "Pending"
        self.assertIsNone(
            memory._consider_demographics_refresh_locked(
                1,
                [sample(sharpness=1000.0, face_box=face_box_of_width(0.95))],
            )
        )

    def test_crops_awaiting_a_first_estimate_block_a_re_estimate(self):
        memory = self.make_memory()
        record = self.settled_record(memory)
        record["pending_demographics_crops"] = record["demographics_crop_pool"]
        self.assertIsNone(
            memory._consider_demographics_refresh_locked(
                1,
                [sample(sharpness=1000.0, face_box=face_box_of_width(0.95))],
            )
        )

    def test_staff_and_unconfirmed_identities_are_left_alone(self):
        memory = self.make_memory()
        record = self.settled_record(memory)
        good = [sample(sharpness=1000.0, face_box=face_box_of_width(0.95))]

        record["role"] = "cag"
        self.assertIsNone(memory._consider_demographics_refresh_locked(1, good))

        record["role"] = "evacuee"
        record["identity_state"] = "provisional"
        self.assertIsNone(memory._consider_demographics_refresh_locked(1, good))

        self.assertIsNone(memory._consider_demographics_refresh_locked(999, good))

    def test_nothing_runs_when_demographics_are_switched_off(self):
        memory = self.make_memory(enable_demographics=False)
        self.settled_record(memory)
        self.assertIsNone(
            memory._consider_demographics_refresh_locked(
                1,
                [sample(sharpness=1000.0, face_box=face_box_of_width(0.95))],
            )
        )

    def test_a_staff_vote_drops_the_whole_demographics_pool(self):
        memory = self.make_memory()
        record = self.settled_record(memory)
        record["pending_demographics_crops"] = record["demographics_crop_pool"]
        memory._apply_role_vote_locked(record, "cag", 0.99)
        self.assertNotIn("pending_demographics_crops", record)
        self.assertNotIn("demographics_crop_pool", record)
        self.assertNotIn("demographics_quality", record)
        self.assertEqual(record["age"], "N/A")


class RefreshThroughAssignTests(unittest.TestCase):
    """The re-estimate has to fire from the live per-frame path, not just in
    isolation: a mapped track only carries a face box on the frames where a
    semantic probe already ran MediaPipe."""

    def setUp(self):
        self.memories = []

    def tearDown(self):
        for memory in self.memories:
            memory.close(drain=False, timeout=0.2)

    def settled_memory(self):
        engine = _RecordingEngine(DemographicsReading(30, "male", 0.9, 5, 0))
        memory = AppearanceIdentityMemory(
            reid_extractor=CountingBatchExtractor(),
            intake_frames=5,
            intake_delay_seconds=0.0,
            blur_threshold=1.0,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=True,
            demographics_engine=engine,
        )
        self.memories.append(memory)
        for frame_index in range(1, 6):
            process_tracks(memory, frame_index, [1])
        self.assertTrue(memory.wait_for_idle())
        # The intake crops carried no face box, so the first answer was taken
        # body-only -- the case a later close-up should overturn.
        self.assertEqual(memory.identities[1]["age"], 30)
        self.assertEqual(len(engine.batches), 1)
        return memory, engine

    def test_a_frame_carrying_a_good_face_box_queues_a_re_estimate(self):
        memory, engine = self.settled_memory()
        engine.reading = DemographicsReading(52, "female", 0.95, 5, 1)
        memory.assign(
            1,
            crop(),
            6,
            camera_id="cam_1",
            detection_confidence=0.95,
            observed_at=6.0,
            intake_face_box=face_box_of_width(0.5),
            intake_body_bounds=(0.02, 0.02, 0.98, 0.91),
        )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(len(engine.batches), 2)
        self.assertEqual(memory.identities[1]["age"], 52)
        self.assertEqual(memory.identities[1]["gender"], "female")
        self.assertEqual(memory.identities[1]["demographics_refreshes"], 1)

    def test_a_frame_without_a_face_box_costs_nothing(self):
        memory, engine = self.settled_memory()
        memory.assign(
            1,
            crop(),
            6,
            camera_id="cam_1",
            detection_confidence=0.95,
            observed_at=6.0,
        )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(len(engine.batches), 1)
        self.assertEqual(memory.identities[1]["age"], 30)


class DemographicsPersistenceTests(unittest.TestCase):
    """Crops are working state and must never reach the saved record.

    A record is deep-copied for every backend save and pickled whole for every
    local one, so five crops on it would be megabytes written per save.
    """

    def setUp(self):
        self.memories = []

    def tearDown(self):
        for memory in self.memories:
            memory.close(drain=False, timeout=0.2)

    def make_memory(self, db_path):
        memory = AppearanceIdentityMemory(
            reid_extractor=CountingBatchExtractor(),
            intake_frames=5,
            intake_delay_seconds=0.0,
            blur_threshold=1.0,
            db_path=db_path,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=True,
            demographics_engine=_RecordingEngine(
                DemographicsReading(30, "male", 0.9, 5, 5)
            ),
        )
        self.memories.append(memory)
        return memory

    def test_the_crop_pool_is_stripped_from_the_pickled_database(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "gallery.pkl"
            memory = self.make_memory(db_path)
            for frame_index in range(1, 6):
                process_tracks(memory, frame_index, [1])
            self.assertTrue(memory.wait_for_idle())

            record = memory.identities[1]
            record["demographics_crop_pool"] = (
                AppearanceIdentityMemory._demographics_candidates([sample()])
            )
            record["pending_demographics_crops"] = record["demographics_crop_pool"]
            memory.save_database(1)

            with db_path.open("rb") as handle:
                payload = pickle.load(handle)
            saved = payload["identities"][1]
            self.assertNotIn("demographics_crop_pool", saved)
            self.assertNotIn("pending_demographics_crops", saved)
            # Everything else about the identity survives.
            self.assertEqual(saved["role"], record["role"])
            self.assertIsNotNone(saved["gallery"]["baseline"])

    def test_stripping_leaves_the_live_record_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = self.make_memory(Path(directory) / "gallery.pkl")
            for frame_index in range(1, 6):
                process_tracks(memory, frame_index, [1])
            self.assertTrue(memory.wait_for_idle())

            record = memory.identities[1]
            pool = AppearanceIdentityMemory._demographics_candidates([sample()])
            record["demographics_crop_pool"] = pool
            memory.save_database(1)
            self.assertIs(memory.identities[1]["demographics_crop_pool"], pool)

    def test_a_record_holding_no_crops_is_passed_through_untouched(self):
        record = {"role": "evacuee", "age": 30}
        self.assertIs(AppearanceIdentityMemory._without_transient_crops(record), record)

    def test_the_pool_is_released_once_no_re_estimate_can_use_it(self):
        memory = self.make_memory(None)
        record = memory._new_record("evacuee", 0.9)
        record["age"] = 30
        record["demographics_refreshes"] = DEFAULT_DEMOGRAPHICS_MAX_REFRESHES - 1
        record["demographics_crop_pool"] = (
            AppearanceIdentityMemory._demographics_candidates(
                [sample(sharpness=1.0, face_box=face_box_of_width(0.1))]
            )
        )
        record["demographics_quality"] = AppearanceIdentityMemory._demographics_pool_quality(
            record["demographics_crop_pool"]
        )
        memory.identities[3] = record

        queued = memory._consider_demographics_refresh_locked(
            3,
            [sample(sharpness=500.0, face_box=face_box_of_width(0.9))],
        )
        # The queued set still holds its own reference to the crops.
        self.assertTrue(queued)
        self.assertNotIn("demographics_crop_pool", record)


class DemographicsWorkerTests(unittest.TestCase):
    """The reading the engine returns has to land on the record."""

    def setUp(self):
        self.memories = []

    def tearDown(self):
        for memory in self.memories:
            memory.close(drain=False, timeout=0.2)

    def make_memory(self, engine):
        memory = AppearanceIdentityMemory(
            reid_extractor=CountingBatchExtractor(),
            intake_frames=5,
            intake_delay_seconds=0.0,
            blur_threshold=1.0,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=True,
            demographics_engine=engine,
        )
        self.memories.append(memory)
        return memory

    def test_a_full_intake_writes_the_reading_onto_the_record(self):
        engine = _RecordingEngine(DemographicsReading(41, "female", 0.87, 5, 4))
        memory = self.make_memory(engine)
        for frame_index in range(1, 6):
            process_tracks(memory, frame_index, [1])
        self.assertTrue(memory.wait_for_idle())

        record = memory.identities[1]
        self.assertEqual(record["age"], 41)
        self.assertEqual(record["gender"], "female")
        self.assertAlmostEqual(record["demographics_confidence"], 0.87)

    def test_the_engine_is_handed_framed_candidates_not_bare_crops(self):
        engine = _RecordingEngine(DemographicsReading(30, "male", 0.9, 5, 5))
        memory = self.make_memory(engine)
        for frame_index in range(1, 6):
            process_tracks(memory, frame_index, [1])
        self.assertTrue(memory.wait_for_idle())

        self.assertEqual(len(engine.batches), 1)
        candidate = engine.batches[0][0]
        self.assertIsInstance(candidate, dict)
        self.assertIn("face_box", candidate)
        self.assertIn("body_bounds", candidate)
        self.assertIn("occluder_boxes", candidate)

    def test_a_failed_first_estimate_leaves_the_record_unknown(self):
        memory = self.make_memory(_FailingEngine())
        for frame_index in range(1, 6):
            process_tracks(memory, frame_index, [1])
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(memory.identities[1]["age"], "Unknown")

    def test_a_failed_re_estimate_leaves_the_earlier_answer_standing(self):
        # Losing a refresh must not turn a good reading into "Unknown".
        memory = self.make_memory(_FailingEngine())
        record = memory._new_record("evacuee", 0.9)
        record["age"] = 34
        record["gender"] = "male"
        memory.identities[7] = record
        memory._queue_demographics(
            7,
            AppearanceIdentityMemory._demographics_candidates([sample()]),
            "closer_view",
        )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(record["age"], 34)
        self.assertEqual(record["gender"], "male")


if __name__ == "__main__":
    unittest.main()
