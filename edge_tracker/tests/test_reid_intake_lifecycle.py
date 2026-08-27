import unittest
import time
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from pose_engine import get_standing_points
from reid_memory import AppearanceIdentityMemory


class CountingBatchExtractor:
    def __init__(self, feature=(1.0, 0.0, 0.0)):
        self.feature = np.asarray(feature, dtype=np.float32)
        self.batch_sizes = []

    def extract_many_aligned(self, crops):
        self.batch_sizes.append(len(crops))
        return [self.feature.copy() for _ in crops]

    def extract_many(self, crops):
        self.batch_sizes.append(len(crops))
        return [self.feature.copy() for _ in crops]

    def extract(self, _crop):
        raise AssertionError("The async lifecycle should use aligned batch extraction")


class SlowBatchExtractor(CountingBatchExtractor):
    def extract_many_aligned(self, crops):
        time.sleep(0.2)
        return super().extract_many_aligned(crops)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBoxes:
    def __init__(self, boxes, track_ids, confidences=None):
        self.xyxy = FakeTensor(boxes)
        self.id = FakeTensor(track_ids)
        self.conf = FakeTensor(confidences if confidences is not None else [0.95] * len(track_ids))

    def __len__(self):
        return len(self.xyxy.value)


class FakeResult:
    def __init__(self, track_ids, boxes=None):
        boxes = boxes if boxes is not None else [[10, 10, 60, 100] for _ in track_ids]
        self.boxes = FakeBoxes(boxes, track_ids)
        self.keypoints = None


def sharp_frame():
    yy, xx = np.indices((120, 90))
    checker = ((xx // 4 + yy // 4) % 2 * 255).astype(np.uint8)
    return np.repeat(checker[:, :, None], 3, axis=2)


def process_tracks(
    memory,
    frame_index,
    track_ids,
    camera_id="cam_1",
    map_point=None,
    boxes=None,
    map_size_cm=None,
):
    return get_standing_points(
        FakeResult(track_ids, boxes=boxes),
        sharp_frame(),
        frame_index=frame_index,
        appearance_memory=memory,
        camera_id=camera_id,
        observation_time=float(frame_index),
        map_projector=(None if map_point is None else lambda _image_point: map_point),
        map_size_cm=map_size_cm,
    )


class ReIDIntakeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.memories = []

    def tearDown(self):
        for memory in self.memories:
            memory.close(drain=True)

    def make_memory(self, feature=(1.0, 0.0, 0.0)):
        extractor = CountingBatchExtractor(feature)
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
            # These tests pin how much GPU work intake and the semantic slots
            # cost.  The periodic audit is a separate budget with its own
            # tests, and here it would only obscure the counts being asserted.
            identity_audit_interval_seconds=0.0,
        )
        self.memories.append(memory)
        return memory, extractor

    def complete_intake(self, memory, track_id, camera_id="cam_1", start_frame=1, map_point=None):
        last_point = None
        for frame_index in range(start_frame, start_frame + 5):
            last_point = process_tracks(memory, frame_index, [track_id], camera_id=camera_id, map_point=map_point)[0]
        self.assertIsNone(last_point["identity_id"])
        self.assertTrue(memory.wait_for_idle())
        return process_tracks(memory, start_frame + 5, [track_id], camera_id=camera_id, map_point=map_point)[0]

    def test_new_track_runs_one_five_crop_batch_then_lookup_only(self):
        memory, extractor = self.make_memory()

        for frame_index in range(1, 5):
            point = process_tracks(memory, frame_index, [10])[0]
            self.assertIsNone(point["identity_id"])
            self.assertEqual(extractor.batch_sizes, [])

        point = process_tracks(memory, 5, [10])[0]
        self.assertIsNone(point["identity_id"])
        self.assertTrue(memory.wait_for_idle())
        point = process_tracks(memory, 6, [10])[0]
        self.assertEqual(point["identity_id"], 1)
        self.assertEqual(extractor.batch_sizes, [5])

        for frame_index in range(7, 21):
            self.assertEqual(process_tracks(memory, frame_index, [10])[0]["identity_id"], 1)
        self.assertEqual(extractor.batch_sizes, [5])
        self.assertEqual(memory.track_to_identity[("cam_1", 10)], 1)

    def test_new_track_waits_outside_tactical_map_before_reid_intake(self):
        memory, extractor = self.make_memory()

        for frame_index in range(1, 8):
            point = process_tracks(
                memory,
                frame_index,
                [10],
                map_point=(481.0, 240.0),
                map_size_cm=480,
            )[0]
            self.assertFalse(point["inside_tactical_map"])
            self.assertIsNone(point["identity_id"])

        self.assertEqual(memory.pending_count(10, camera_id="cam_1"), 0)
        self.assertEqual(extractor.batch_sizes, [])

        for frame_index in range(8, 13):
            point = process_tracks(
                memory,
                frame_index,
                [10],
                map_point=(480.0, 240.0),
                map_size_cm=480,
            )[0]
            self.assertTrue(point["inside_tactical_map"])
            self.assertIsNone(point["identity_id"])

        self.assertTrue(memory.wait_for_idle())
        point = process_tracks(
            memory,
            13,
            [10],
            map_point=(480.0, 240.0),
            map_size_cm=480,
        )[0]
        self.assertEqual(point["identity_id"], 1)
        self.assertEqual(extractor.batch_sizes, [5])

    def test_incomplete_body_frames_do_not_enter_initial_reid_batch(self):
        memory, extractor = self.make_memory()
        crop = sharp_frame()

        for frame_index in range(1, 7):
            identity_id, _, _ = memory.assign(
                10,
                crop,
                frame_index,
                camera_id="cam_1",
                intake_body_complete=False,
                intake_missing_regions=("head", "shoulders"),
            )
            self.assertIsNone(identity_id)

        self.assertEqual(memory.pending_count(10, camera_id="cam_1"), 0)
        self.assertEqual(extractor.batch_sizes, [])

        for frame_index in range(7, 12):
            memory.assign(
                10,
                crop,
                frame_index,
                camera_id="cam_1",
                intake_body_complete=True,
            )

        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(extractor.batch_sizes, [5])
        self.assertEqual(
            memory.assign(
                10,
                crop,
                12,
                camera_id="cam_1",
                intake_body_complete=False,
                intake_missing_regions=("head",),
            )[0],
            1,
        )

    def test_debug_events_explain_accepted_crops_and_baseline_selection(self):
        memory, _extractor = self.make_memory()
        crop = sharp_frame()
        body_details = {
            "body_complete": True,
            "landmarks": {
                "left_ankle": {
                    "visibility": 0.95,
                    "x": 0.45,
                    "y": 0.90,
                    "within_saved_crop": True,
                    "usable": True,
                }
            },
        }

        with patch("reid_memory.identity_event") as event:
            for frame_index in range(1, 6):
                memory.assign(
                    10,
                    crop,
                    frame_index,
                    camera_id="cam_1",
                    detection_confidence=0.95,
                    orientation="front",
                    observed_at=float(frame_index),
                    map_point=(100.0, 120.0),
                    intake_body_complete=True,
                    intake_body_details=body_details,
                )
            self.assertTrue(memory.wait_for_idle())

        names = [call.args[0] for call in event.call_args_list]
        self.assertEqual(names.count("intake_crop_accepted"), 5)
        self.assertIn("intake_batch_submitted", names)
        self.assertIn("intake_baseline_candidate_selected", names)
        self.assertIn("baseline_selected", names)
        baseline_call = next(
            call for call in event.call_args_list if call.args[0] == "baseline_selected"
        )
        selected = baseline_call.kwargs["selected_sample"]
        self.assertEqual(selected["crop_shape"], crop.shape)
        self.assertTrue(selected["body_complete"])
        self.assertEqual(
            selected["body_details"]["landmarks"]["left_ankle"]["visibility"],
            0.95,
        )
        self.assertFalse(baseline_call.kwargs["console"])

    def test_new_master_allocation_waits_for_cross_camera_hold_release(self):
        memory, extractor = self.make_memory()
        crop = sharp_frame()
        left_key = ("cam_1", 10)
        right_key = ("cam_2", 20)
        token = (left_key, right_key)
        memory.hold_new_master_creation(left_key, right_key, token)

        for frame_index in range(1, 6):
            memory.assign(
                10,
                crop,
                frame_index,
                camera_id="cam_1",
                observed_at=float(frame_index),
                intake_body_complete=True,
            )
        self.assertTrue(memory.wait_for_idle())
        self.assertNotIn(left_key, memory.track_to_identity)
        self.assertTrue(memory.pending_intake[left_key]["deferred_by_new_master_hold"])

        memory.release_new_master_hold(
            left_key,
            right_key,
            token,
            "grace_expired",
        )
        memory.assign(
            10,
            crop,
            6,
            camera_id="cam_1",
            observed_at=6.0,
            intake_body_complete=True,
        )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(
            memory.assign(10, crop, 7, camera_id="cam_1", observed_at=7.0)[0],
            1,
        )
        self.assertEqual(extractor.batch_sizes, [5, 5])

    def test_hold_does_not_block_a_confident_match_to_an_existing_master(self):
        memory, extractor = self.make_memory()
        self.assertEqual(self.complete_intake(memory, 10)["identity_id"], 1)

        left_key = ("cam_1", 10)
        right_key = ("cam_2", 20)
        token = (left_key, right_key)
        held_keys = memory.hold_new_master_creation(left_key, right_key, token)
        self.assertEqual(held_keys, (right_key,))

        matched = self.complete_intake(
            memory,
            20,
            camera_id="cam_2",
            start_frame=7,
        )
        self.assertEqual(matched["identity_id"], 1)
        self.assertTrue(matched["reidentified"])
        self.assertEqual(extractor.batch_sizes, [5, 5])

    def test_fifth_crop_does_not_block_on_slow_transreid(self):
        extractor = SlowBatchExtractor()
        memory = AppearanceIdentityMemory(
            reid_extractor=extractor,
            intake_frames=5,
            intake_delay_seconds=0.0,
            blur_threshold=1.0,
            enable_role_classification=False,
            enable_demographics=False,
        )
        self.memories.append(memory)
        for frame_index in range(1, 5):
            process_tracks(memory, frame_index, [1])
        started = time.monotonic()
        process_tracks(memory, 5, [1])
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.1)
        self.assertTrue(memory.wait_for_idle(timeout=2.0))
        self.assertEqual(process_tracks(memory, 6, [1])[0]["identity_id"], 1)

    def test_new_local_id_reuses_existing_master_through_all_slot_matcher(self):
        memory, extractor = self.make_memory()
        self.assertEqual(self.complete_intake(memory, 10)["identity_id"], 1)

        final_point = self.complete_intake(memory, 99, start_frame=7)
        self.assertEqual(final_point["identity_id"], 1)
        self.assertTrue(final_point["reidentified"])
        self.assertEqual(extractor.batch_sizes, [5, 5])

    def test_active_master_is_excluded_only_inside_the_same_camera(self):
        memory, extractor = self.make_memory()
        self.assertEqual(self.complete_intake(memory, 10)["identity_id"], 1)

        # Keep this fixture focused on same-camera master exclusion. These
        # people must not overlap enough to trigger the gallery crop gate.
        distinct_person_boxes = [[2, 10, 37, 100], [52, 10, 87, 100]]
        for frame_index in range(7, 12):
            process_tracks(
                memory,
                frame_index,
                [10, 20],
                camera_id="cam_1",
                boxes=distinct_person_boxes,
            )
        self.assertTrue(memory.wait_for_idle())
        # This extractor returns one feature for everybody, so the newcomer is
        # indistinguishable from the incumbent and first challenges it for
        # master 1 rather than being handed a duplicate outright.  It loses,
        # and only then earns its own master -- a few frames later than a
        # clearly different-looking person would.
        for frame_index in range(12, 20):
            points = process_tracks(
                memory,
                frame_index,
                [10, 20],
                camera_id="cam_1",
                boxes=distinct_person_boxes,
            )
            self.assertTrue(memory.wait_for_idle())
            if points[1]["identity_id"] is not None:
                break
        self.assertEqual(points[0]["identity_id"], 1)
        self.assertEqual(points[1]["identity_id"], 2)

        # The same local number in a second camera is a separate local key,
        # but it may legitimately map to the same global master.
        for frame_index in range(1, 6):
            process_tracks(memory, frame_index, [10], camera_id="cam_2")
        self.assertTrue(memory.wait_for_idle())
        cam_2_point = process_tracks(memory, 6, [10], camera_id="cam_2")[0]
        self.assertEqual(cam_2_point["identity_id"], 1)
        # Three five-crop intakes, plus the contest the look-alike newcomer
        # triggered: one six-crop arbitration batch (three crops from each
        # claimant) and the re-run intake it was deferred into.  That is the
        # price of not handing out a duplicate ID, and it is only paid when a
        # newcomer strongly matches a master another live box is holding.
        self.assertEqual(extractor.batch_sizes, [5, 5, 6, 5, 5])

    def test_physics_can_veto_impossible_cross_camera_master_reuse(self):
        extractor = CountingBatchExtractor()
        memory = AppearanceIdentityMemory(
            reid_extractor=extractor,
            intake_frames=5,
            intake_delay_seconds=0.0,
            blur_threshold=1.0,
            cross_camera_fusion_distance_cm=50.0,
            cross_camera_max_skew_seconds=0.35,
            enable_role_classification=False,
            enable_demographics=False,
        )
        self.memories.append(memory)
        crop = sharp_frame()[10:100, 10:60]
        for frame_index in range(1, 6):
            memory.assign(
                1,
                crop,
                frame_index,
                camera_id="cam_1",
                detection_confidence=0.95,
                observed_at=5.9 + frame_index * 0.01,
                map_point=(0.0, 0.0),
            )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(
            memory.assign(
                1,
                crop,
                6,
                camera_id="cam_1",
                detection_confidence=0.95,
                observed_at=6.0,
                map_point=(0.0, 0.0),
            )[0],
            1,
        )

        # Use capture times inside the skew window so simultaneous physics is
        # relevant. Direct calls make that timing explicit.
        for frame_index in range(1, 6):
            memory.assign(
                7,
                crop,
                frame_index,
                camera_id="cam_2",
                detection_confidence=0.95,
                observed_at=6.0 + frame_index * 0.01,
                map_point=(200.0, 0.0),
            )
        self.assertTrue(memory.wait_for_idle())
        # A strong appearance match at an impossible location now stays
        # analysing while the current owner is checked with fresh crops. It
        # must not steal the identity or create a duplicate before that check.
        far_identity = memory.assign(
            7,
            crop,
            6,
            camera_id="cam_2",
            detection_confidence=0.95,
            observed_at=6.1,
            map_point=(200.0, 0.0),
        )[0]
        self.assertIsNone(far_identity)
        self.assertNotIn(2, memory.identities)

        # Both tracks look equally close to Master 1, so the established owner
        # keeps it. The challenger is then released to create Master 2.
        for frame_index in range(7, 10):
            memory.assign(
                1,
                crop,
                frame_index,
                camera_id="cam_1",
                detection_confidence=0.95,
                observed_at=6.04 + frame_index * 0.01,
                map_point=(0.0, 0.0),
            )
        self.assertTrue(memory.wait_for_idle())
        memory.assign(
            7,
            crop,
            7,
            camera_id="cam_2",
            detection_confidence=0.95,
            observed_at=6.14,
            map_point=(200.0, 0.0),
        )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(memory.lookup(7, camera_id="cam_2"), 2)

        for frame_index in range(1, 6):
            memory.assign(
                9,
                crop,
                frame_index,
                camera_id="cam_3",
                detection_confidence=0.95,
                observed_at=6.15 + frame_index * 0.01,
                map_point=(5.0, 0.0),
            )
        self.assertTrue(memory.wait_for_idle())
        close_identity = memory.assign(
            9,
            crop,
            6,
            camera_id="cam_3",
            detection_confidence=0.95,
            observed_at=6.21,
            map_point=(5.0, 0.0),
        )[0]
        self.assertEqual(close_identity, 1)

    def test_fresh_identity_has_exactly_five_named_slots(self):
        memory, _extractor = self.make_memory()
        self.complete_intake(memory, 1)
        gallery = memory.identities[1]["gallery"]
        self.assertEqual(set(gallery), {"baseline", "front", "back", "left_side", "right_side"})
        self.assertIsNotNone(gallery["baseline"])
        self.assertTrue(all(gallery[name] is None for name in ("front", "back", "left_side", "right_side")))

    def test_semantic_inference_runs_once_only_for_a_missing_clear_slot(self):
        memory, extractor = self.make_memory()
        self.complete_intake(memory, 1)
        crop = sharp_frame()[10:100, 10:60]

        memory.assign(
            1,
            crop,
            31,
            camera_id="cam_1",
            detection_confidence=0.95,
            orientation="left_side",
            observed_at=31.0,
        )
        self.assertTrue(memory.wait_for_idle())
        self.assertIsNotNone(memory.identities[1]["gallery"]["left_side"])
        self.assertEqual(extractor.batch_sizes, [5, 1])

        memory.assign(
            1,
            crop,
            70,
            camera_id="cam_1",
            detection_confidence=0.95,
            orientation="left_side",
            observed_at=70.0,
        )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(extractor.batch_sizes, [5, 1])

    def test_full_gallery_stops_after_at_most_four_semantic_calls(self):
        memory, extractor = self.make_memory()
        self.complete_intake(memory, 1)
        crop = sharp_frame()[10:100, 10:60]
        for frame_index, orientation in zip(
            (31, 61, 91, 121),
            ("front", "back", "left_side", "right_side"),
        ):
            memory.assign(
                1,
                crop,
                frame_index,
                camera_id="cam_1",
                detection_confidence=0.95,
                orientation=orientation,
                observed_at=float(frame_index),
            )
            self.assertTrue(memory.wait_for_idle())
        self.assertEqual(memory.gallery_status(1), (5, 5))
        self.assertEqual(extractor.batch_sizes, [5, 1, 1, 1, 1])

        memory.assign(
            1,
            crop,
            200,
            camera_id="cam_1",
            detection_confidence=0.95,
            orientation="front",
            observed_at=200.0,
        )
        self.assertTrue(memory.wait_for_idle())
        self.assertEqual(extractor.batch_sizes, [5, 1, 1, 1, 1])

    def test_match_uses_a_semantic_slot_when_baseline_disagrees(self):
        memory, _extractor = self.make_memory()
        with memory._lock:
            record = memory._new_record()
            record["gallery"]["baseline"] = {"feature": np.array([0.0, 1.0, 0.0], dtype=np.float32)}
            record["gallery"]["back"] = {"feature": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
            memory.identities[7] = record
            memory.next_identity_id = 8

        matched, similarity, distance = memory.find_matching_identity(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(matched, 7)
        self.assertAlmostEqual(similarity, 1.0)
        self.assertAlmostEqual(distance, 0.0)

    def test_cosine_distance_threshold_is_strictly_below_point_35(self):
        memory, _extractor = self.make_memory()
        memory.distance_threshold = 0.35
        with memory._lock:
            record = memory._new_record()
            record["gallery"]["baseline"] = {"feature": np.array([1.0, 0.0], dtype=np.float32)}
            memory.identities[1] = record

        accepted = np.array([0.651, np.sqrt(1.0 - 0.651**2)], dtype=np.float32)
        boundary = np.array([0.65, np.sqrt(1.0 - 0.65**2)], dtype=np.float32)
        self.assertEqual(memory.find_matching_identity(accepted)[0], 1)
        self.assertIsNone(memory.find_matching_identity(boundary)[0])

    def test_fresh_database_and_raw_evidence_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database_path = root / "gallery.pkl"
            evidence_dir = root / "evidence"
            extractor = CountingBatchExtractor()
            memory = AppearanceIdentityMemory(
                reid_extractor=extractor,
                intake_frames=5,
                intake_delay_seconds=0.0,
                blur_threshold=1.0,
                db_path=database_path,
                evidence_dir=evidence_dir,
                enable_role_classification=False,
                enable_demographics=False,
            )
            self.memories.append(memory)
            self.complete_intake(memory, 1)
            memory.close(drain=True)
            self.memories.remove(memory)

            with database_path.open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["schema_version"], 3)

            baseline = payload["identities"][1]["gallery"]["baseline"]
            saved_crop = cv2.imread(baseline["image_path"], cv2.IMREAD_COLOR)
            self.assertIsNotNone(saved_crop)
            self.assertEqual(baseline["digest"], AppearanceIdentityMemory._crop_digest(saved_crop))

            reloaded = AppearanceIdentityMemory(
                reid_extractor=extractor,
                db_path=database_path,
                evidence_dir=evidence_dir,
                enable_role_classification=False,
                enable_demographics=False,
                start_worker=False,
            )
            self.memories.append(reloaded)
            self.assertEqual(reloaded.gallery_status(1), (1, 5))
            self.assertAlmostEqual(float(np.linalg.norm(reloaded.identities[1]["gallery"]["baseline"]["feature"])), 1.0)

    def test_camera_two_slots_keep_vectors_without_writing_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database_path = root / "gallery.pkl"
            evidence_dir = root / "evidence"
            memory = AppearanceIdentityMemory(
                reid_extractor=CountingBatchExtractor(),
                db_path=database_path,
                evidence_dir=evidence_dir,
                evidence_camera_ids={"cam_1"},
                enable_role_classification=False,
                enable_demographics=False,
                start_worker=False,
            )
            self.memories.append(memory)
            feature = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            base_sample = {
                "crop": sharp_frame(),
                "frame_index": 1,
                "observed_at": 1.0,
                "sharpness": 100.0,
                "detection_confidence": 0.95,
            }
            cam_1_slot, cam_1_task = memory._make_slot(
                1,
                "baseline",
                feature,
                {**base_sample, "camera_id": "cam_1"},
                "test",
                "test-space",
            )
            cam_2_slot, cam_2_task = memory._make_slot(
                1,
                "front",
                feature,
                {**base_sample, "camera_id": "cam_2", "frame_index": 2},
                "test",
                "test-space",
            )
            cam_1_angle_slot, duplicate_cam_1_task = memory._make_slot(
                1,
                "back",
                feature,
                {**base_sample, "camera_id": "cam_1"},
                "test",
                "test-space",
            )
            with memory._lock:
                record = memory._new_record()
                record["gallery"]["baseline"] = cam_1_slot
                record["gallery"]["front"] = cam_2_slot
                record["gallery"]["back"] = cam_1_angle_slot
                memory.identities[1] = record
                memory._queue_evidence_save(cam_1_task)
                memory._queue_evidence_save(cam_2_task)
            self.assertTrue(memory.wait_for_idle())
            memory.save_database(1)

            self.assertTrue(Path(cam_1_slot["image_path"]).is_file())
            self.assertTrue(cam_1_slot["evidence_expected"])
            self.assertIsNone(duplicate_cam_1_task)
            self.assertEqual(cam_1_angle_slot["image_path"], cam_1_slot["image_path"])
            self.assertEqual(cam_1_angle_slot["digest"], cam_1_slot["digest"])
            self.assertIsNone(cam_2_slot["image_path"])
            self.assertIsNone(cam_2_slot["digest"])
            self.assertFalse(cam_2_slot["evidence_expected"])
            self.assertIsNotNone(cam_2_slot["feature"])

            memory.close(drain=True)
            self.memories.remove(memory)
            reloaded = AppearanceIdentityMemory(
                reid_extractor=CountingBatchExtractor(),
                db_path=database_path,
                evidence_dir=evidence_dir,
                evidence_camera_ids={"cam_1"},
                enable_role_classification=False,
                enable_demographics=False,
                start_worker=False,
            )
            self.memories.append(reloaded)
            reloaded_cam_2_slot = reloaded.identities[1]["gallery"]["front"]
            self.assertIsNone(reloaded_cam_2_slot["image_path"])
            self.assertIsNotNone(reloaded_cam_2_slot["feature"])


if __name__ == "__main__":
    unittest.main()
