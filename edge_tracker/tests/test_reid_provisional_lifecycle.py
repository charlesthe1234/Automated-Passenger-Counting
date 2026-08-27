import unittest

import numpy as np

from constants import DEFAULT_POSITION_SPLIT_FRAMES, REID_SEMANTIC_SLOTS
from cross_camera_provisional import CrossCameraProvisionalCoordinator
from reid_memory import AppearanceIdentityMemory


class FixedExtractor:
    def __init__(self, feature):
        self.feature = np.asarray(feature, dtype=np.float32)
        self.batch_sizes = []

    def extract_many_aligned(self, crops):
        self.batch_sizes.append(len(crops))
        return [self.feature.copy() for _ in crops]

    @staticmethod
    def feature_space_id(_dimension):
        return "test-feature-space"


def make_slot(feature, camera_id, frame_index=1):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    feature /= np.linalg.norm(feature)
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


class ProvisionalIdentityLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            provisional_location_confirm_frames=3,
            start_worker=False,
        )
        self.left_key = ("cam_1", 10)
        self.right_key = ("cam_2", 20)

    def tearDown(self):
        self.memory.close(drain=False)

    def create_pair(self):
        return self.memory.create_provisional_pair(self.left_key, self.right_key)

    def seed_angle(self, identity_id, camera_id, orientation, feature):
        with self.memory._lock:
            record = self.memory.identities[identity_id]
            camera_gallery = record["camera_views"].setdefault(
                camera_id,
                {slot_name: None for slot_name in REID_SEMANTIC_SLOTS},
            )
            camera_gallery[orientation] = make_slot(feature, camera_id)
            record.setdefault("global_reid_checked_track_keys", set()).update(
                key
                for key in record.get("member_track_keys", ())
                if key[0] == camera_id
            )

    def process_delayed_location_assigned_task(
        self,
        extractor_feature,
        task_provisional_identity_id=None,
    ):
        self.memory.close(drain=False)
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor(extractor_feature),
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        identity_id = 7
        track_key = ("cam_2", 18)
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        record["location_managed"] = True
        record["member_track_keys"].add(track_key)
        record["confirmation_reason"] = "stable_location"
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[track_key] = identity_id
        self.memory.pending_intake[track_key] = {
            "submitted": True,
            "generation": 1,
        }

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        task = {
            "type": "intake",
            "track_key": track_key,
            "camera_id": "cam_2",
            "frame_index": 10,
            "samples": [
                {
                    "crop": crop,
                    "frame_index": 10,
                    "camera_id": "cam_2",
                    "observed_at": 1.0,
                    "sharpness": 200.0,
                    "area": float(crop.shape[0] * crop.shape[1]),
                    "detection_confidence": 0.95,
                    "orientation": "front",
                    "map_point": (0.0, 0.0),
                }
            ],
            # This is the stale snapshot that previously prevented ID 7
            # from being compared and caused a new ID to be created.
            "excluded_identity_ids": {identity_id},
            "same_camera_peer_keys": set(),
            "generation": 1,
            "handoff_identity_id": None,
            "handoff_from_key": None,
            "provisional_identity_id": task_provisional_identity_id,
        }
        self.memory._process_intake_task(task)
        return identity_id, track_key

    def test_create_provisional_pair_uses_unnumbered_temporary_group(self):
        temporary_id = self.create_pair()

        self.assertLess(temporary_id, 0)
        self.assertEqual(self.memory.identity_state(temporary_id), "provisional")
        self.assertEqual(self.memory.lookup_track_key(self.left_key), temporary_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), temporary_id)
        self.assertIsNone(self.memory.lookup(10, camera_id="cam_1"))
        self.assertEqual(self.memory.temporary_group(10, camera_id="cam_1"), "tmp_1")
        self.assertEqual(
            self.memory.create_provisional_pair(self.right_key, self.left_key),
            temporary_id,
        )
        self.assertEqual(self.memory.next_identity_id, 1)

    def test_camera_specific_same_angle_slots_can_coexist(self):
        identity_id = self.create_pair()
        self.seed_angle(identity_id, "cam_1", "front", (1.0, 0.0))
        self.seed_angle(identity_id, "cam_2", "front", (0.0, 1.0))

        record = self.memory.identities[identity_id]
        cam_1_front = record["camera_views"]["cam_1"]["front"]
        cam_2_front = record["camera_views"]["cam_2"]["front"]

        self.assertIsNot(cam_1_front, cam_2_front)
        np.testing.assert_array_equal(cam_1_front["feature"], np.array([1.0, 0.0]))
        np.testing.assert_array_equal(cam_2_front["feature"], np.array([0.0, 1.0]))

    def test_matching_same_angle_allocates_master_after_global_search(self):
        temporary_id = self.create_pair()
        self.seed_angle(temporary_id, "cam_1", "front", (1.0, 0.0))
        self.seed_angle(temporary_id, "cam_2", "front", (0.99, 0.10))

        with self.memory._lock:
            master_id = self.memory._evaluate_provisional_evidence_locked(temporary_id)

        self.assertEqual(master_id, 1)
        self.assertNotIn(temporary_id, self.memory.identities)
        self.assertEqual(self.memory.identity_state(master_id), "confirmed")
        self.assertEqual(self.memory.lookup_track_key(self.left_key), master_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), master_id)
        self.assertEqual(
            self.memory.identities[master_id]["confirmation_reason"],
            "same_angle_reid",
        )
        self.assertIsNotNone(self.memory.identities[master_id]["gallery"]["baseline"])
        self.assertTrue(
            self.memory.assignment_metadata(10, camera_id="cam_1")["appearance_confirmed"]
        )

    def test_opposite_angles_are_inconclusive_and_remain_provisional(self):
        identity_id = self.create_pair()
        self.seed_angle(identity_id, "cam_1", "front", (1.0, 0.0))
        self.seed_angle(identity_id, "cam_2", "back", (1.0, 0.0))

        with self.memory._lock:
            promoted = self.memory._evaluate_provisional_evidence_locked(identity_id)

        self.assertFalse(promoted)
        self.assertEqual(self.memory.identity_state(identity_id), "provisional")
        self.assertEqual(self.memory.identities[identity_id]["reid_comparisons"], {})
        self.assertEqual(self.memory.lookup_track_key(self.left_key), identity_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)

    def test_high_same_angle_distance_challenges_but_does_not_split(self):
        identity_id = self.create_pair()
        self.seed_angle(identity_id, "cam_1", "left_side", (1.0, 0.0))
        self.seed_angle(identity_id, "cam_2", "left_side", (0.0, 1.0))

        with self.memory._lock:
            promoted = self.memory._evaluate_provisional_evidence_locked(identity_id)

        self.assertFalse(promoted)
        self.assertEqual(self.memory.identity_state(identity_id), "challenged")
        self.assertEqual(self.memory.lookup_track_key(self.left_key), identity_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)
        self.assertEqual(
            self.memory.assignment_metadata(10, camera_id="cam_1")["identity_state"],
            "challenged",
        )
        self.assertAlmostEqual(
            self.memory.identities[identity_id]["reid_comparisons"][
                "cam_1:cam_2:left_side"
            ],
            1.0,
        )

    def test_stable_location_allocates_master_only_after_both_global_checks(self):
        temporary_id = self.create_pair()
        record = self.memory.identities[temporary_id]
        record["camera_baselines"]["cam_1"] = make_slot((1.0, 0.0), "cam_1")

        state = self.memory.note_location_match(temporary_id, pair_streak=3, observations=())
        self.assertEqual(state, "provisional")

        # One person seen from two angles: close, but not the identical crop.
        # Orthogonal baselines here would now read as two different people and
        # be vetoed before promotion.
        record["camera_baselines"]["cam_2"] = make_slot((1.0, 0.5), "cam_2")
        record["global_reid_checked_track_keys"].update(
            (self.left_key, self.right_key)
        )
        state = self.memory.note_location_match(temporary_id, pair_streak=3, observations=())

        self.assertEqual(state, "confirmed")
        master_id = 1
        self.assertNotIn(temporary_id, self.memory.identities)
        self.assertEqual(self.memory.identity_state(master_id), "confirmed")
        self.assertEqual(record["confirmation_reason"], "stable_location")
        self.assertEqual(self.memory.lookup_track_key(self.left_key), master_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), master_id)
        self.assertFalse(
            self.memory.assignment_metadata(10, camera_id="cam_1")["appearance_confirmed"]
        )

    def test_disagreeing_baselines_veto_the_stable_location_shortcut(self):
        temporary_id = self.create_pair()
        record = self.memory.identities[temporary_id]
        # Two people walking together while mutually occluded: location agrees
        # perfectly, appearance does not.
        record["camera_baselines"]["cam_1"] = make_slot((1.0, 0.0), "cam_1")
        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")
        record["global_reid_checked_track_keys"].update(
            (self.left_key, self.right_key)
        )

        state = self.memory.note_location_match(temporary_id, pair_streak=9, observations=())

        self.assertEqual(state, "provisional")
        self.assertIn(temporary_id, self.memory.identities)
        self.assertEqual(self.memory.identity_state(temporary_id), "provisional")

    def test_baseline_veto_defers_rather_than_challenges(self):
        temporary_id = self.create_pair()
        record = self.memory.identities[temporary_id]
        record["camera_baselines"]["cam_1"] = make_slot((1.0, 0.0), "cam_1")
        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")
        record["global_reid_checked_track_keys"].update(
            (self.left_key, self.right_key)
        )
        self.memory.note_location_match(temporary_id, pair_streak=9, observations=())

        # A sharper baseline arrives and the pair now agrees.  The earlier veto
        # must not have left a sticky state that blocks this promotion.
        record["camera_baselines"]["cam_2"] = make_slot((1.0, 0.5), "cam_2")
        state = self.memory.note_location_match(temporary_id, pair_streak=9, observations=())

        self.assertEqual(state, "confirmed")
        self.assertEqual(record["confirmation_reason"], "stable_location")

    def test_baseline_veto_does_not_block_same_angle_reid_promotion(self):
        identity_id = self.create_pair()
        record = self.memory.identities[identity_id]
        # Cross-angle baselines disagree, but a matching-angle comparison is
        # the stronger evidence and must still be able to promote.
        record["camera_baselines"]["cam_1"] = make_slot((1.0, 0.0), "cam_1")
        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")
        record["global_reid_checked_track_keys"].update(
            (self.left_key, self.right_key)
        )
        self.seed_angle(identity_id, "cam_1", "front", (1.0, 0.0))
        self.seed_angle(identity_id, "cam_2", "front", (1.0, 0.02))

        with self.memory._lock:
            promoted = self.memory._evaluate_provisional_evidence_locked(identity_id)

        self.assertTrue(promoted)
        self.assertEqual(record["confirmation_reason"], "same_angle_reid")

    def test_incomparable_baselines_do_not_veto(self):
        temporary_id = self.create_pair()
        record = self.memory.identities[temporary_id]
        left = make_slot((1.0, 0.0), "cam_1")
        right = make_slot((0.0, 1.0), "cam_2")
        # A feature-space change must never strand a pair permanently.
        right["feature_space_id"] = "other-feature-space"
        record["camera_baselines"]["cam_1"] = left
        record["camera_baselines"]["cam_2"] = right
        record["global_reid_checked_track_keys"].update(
            (self.left_key, self.right_key)
        )

        state = self.memory.note_location_match(temporary_id, pair_streak=9, observations=())

        self.assertEqual(state, "confirmed")

    def test_location_cannot_promote_before_global_reid_check(self):
        identity_id = self.create_pair()
        record = self.memory.identities[identity_id]
        record["camera_baselines"]["cam_1"] = make_slot((1.0, 0.0), "cam_1")
        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")

        state = self.memory.note_location_match(
            identity_id,
            pair_streak=3,
            observations=(),
        )

        self.assertEqual(state, "provisional")
        self.assertEqual(self.memory.identity_state(identity_id), "provisional")

    def test_provisional_global_reid_reuses_existing_master_for_whole_pair(self):
        self.memory.close(drain=False)
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor((1.0, 0.0)),
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        existing_id = 1
        existing = self.memory._new_record(identity_state="confirmed")
        existing["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[existing_id] = existing
        self.memory.next_identity_id = 2

        provisional_id = self.create_pair()
        self.assertLess(provisional_id, 0)
        self.assertEqual(self.memory.next_identity_id, 2)

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.memory.assign(
            10,
            crop,
            frame_index=1,
            excluded_identity_ids={existing_id},
            camera_id="cam_1",
            detection_confidence=0.95,
            orientation="front",
            observed_at=1.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertNotIn(provisional_id, self.memory.identities)
        self.assertEqual(self.memory.lookup_track_key(self.left_key), existing_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), existing_id)
        self.assertEqual(self.memory.identity_state(existing_id), "confirmed")
        self.assertTrue(
            self.memory.assignment_metadata(10, camera_id="cam_1")[
                "appearance_confirmed"
            ]
        )

    def test_later_camera_track_attaches_provisionally_to_existing_master(self):
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id
        self.memory.track_binding_metadata[self.left_key] = {
            "identity_state": "confirmed",
            "appearance_confirmed": True,
        }

        attached_id = self.memory.create_provisional_pair(self.left_key, self.right_key)

        self.assertEqual(attached_id, identity_id)
        self.assertEqual(self.memory.track_identity_state(self.left_key), "confirmed")
        self.assertEqual(self.memory.track_identity_state(self.right_key), "provisional")
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)
        self.assertEqual(self.memory.next_identity_id, 8)

        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")
        self.memory.note_location_match(identity_id, pair_streak=3, observations=(
            {"camera_id": "cam_1", "local_track_id": 10, "point": (0.0, 0.0), "captured_at": 1.0},
            {"camera_id": "cam_2", "local_track_id": 20, "point": (0.0, 0.0), "captured_at": 1.0},
        ))

        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)

    def test_a_master_keeps_its_stored_views_when_a_member_attaches(self):
        # These crops used to be erased so a newcomer could not be confirmed on
        # them.  Erasing four slots on every re-attachment is what let a swapped
        # track refill them with the wrong person, so they are kept and the
        # confirmation is scoped instead.
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        record["gallery"]["front"] = make_slot((1.0, 0.0), "cam_1")
        record["camera_views"]["cam_2"] = {
            slot_name: None for slot_name in REID_SEMANTIC_SLOTS
        }
        stored = make_slot((1.0, 0.0), "cam_2")
        record["camera_views"]["cam_2"]["front"] = stored
        record["camera_baselines"] = {"cam_2": make_slot((1.0, 0.0), "cam_2")}
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id

        self.memory.create_provisional_pair(self.left_key, self.right_key)

        self.assertIs(record["camera_views"]["cam_2"]["front"], stored)
        self.assertIsNotNone(record["camera_baselines"].get("cam_2"))

    def test_a_member_of_an_existing_master_is_never_confirmed_across_cameras(self):
        # Comparing the two cameras' stored views answers "are these two fresh
        # tracks one person?".  For somebody who already holds a master ID the
        # answer must come from that gallery, or a newcomer that supplied no
        # crop of its own is confirmed by the view already sitting in its slot.
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        record["gallery"]["front"] = make_slot((1.0, 0.0), "cam_1")
        record["camera_views"]["cam_2"] = {
            slot_name: None for slot_name in REID_SEMANTIC_SLOTS
        }
        record["camera_views"]["cam_2"]["front"] = make_slot((1.0, 0.0), "cam_2")
        record["pending_member_keys"] = {self.right_key}
        self.memory.identities[identity_id] = record
        self.memory.track_to_identity[self.left_key] = identity_id
        self.memory.track_to_identity[self.right_key] = identity_id

        with self.memory._lock:
            confirmed = self.memory._evaluate_provisional_evidence_locked(identity_id)

        self.assertFalse(confirmed)
        self.assertIn(self.right_key, record["pending_member_keys"])

    def test_a_brand_new_pair_still_uses_the_cross_camera_check(self):
        # The route is not removed -- it is the only evidence available when
        # neither track has a record to be compared against.
        identity_id = self.create_pair()
        self.seed_angle(identity_id, "cam_1", "front", (1.0, 0.0))
        self.seed_angle(identity_id, "cam_2", "front", (0.99, 0.10))

        with self.memory._lock:
            promoted = self.memory._evaluate_provisional_evidence_locked(identity_id)

        self.assertTrue(promoted)

    def test_existing_master_history_cannot_skip_new_member_location_streak(self):
        self.memory.provisional_location_confirm_frames = 4
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        record["location_match_frames"] = 99
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id

        self.memory.create_provisional_pair(self.left_key, self.right_key)
        record["camera_baselines"]["cam_2"] = make_slot((0.0, 1.0), "cam_2")

        self.memory.note_location_match(identity_id, pair_streak=3, observations=(
            {"camera_id": "cam_1", "local_track_id": 10, "point": (0.0, 0.0), "captured_at": 1.0},
            {"camera_id": "cam_2", "local_track_id": 20, "point": (0.0, 0.0), "captured_at": 1.0},
        ))
        self.assertEqual(self.memory.track_identity_state(self.right_key), "provisional")

        self.memory.note_location_match(identity_id, pair_streak=4, observations=(
            {"camera_id": "cam_1", "local_track_id": 10, "point": (0.0, 0.0), "captured_at": 1.1},
            {"camera_id": "cam_2", "local_track_id": 20, "point": (0.0, 0.0), "captured_at": 1.1},
        ))
        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")

    def test_attached_track_intake_cannot_create_a_second_master(self):
        self.memory.close(drain=False)
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor((0.0, 1.0)),
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id
        self.memory.create_provisional_pair(self.left_key, self.right_key)

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        returned_id, _similarity, _reidentified = self.memory.assign(
            20,
            crop,
            frame_index=1,
            camera_id="cam_2",
            detection_confidence=0.95,
            orientation="back",
            observed_at=1.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(returned_id, identity_id)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)
        self.assertEqual(set(self.memory.identities), {identity_id})
        self.assertTrue(
            self.memory.assignment_metadata(20, camera_id="cam_2")[
                "provisional_intake_complete"
            ]
        )

    def test_revoked_provisional_binding_returns_to_intake_without_key_error(self):
        identity_id = self.create_pair()
        self.memory._physical_match_allowed_locked = lambda *_args, **_kwargs: False
        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        returned_id, _similarity, _reidentified = self.memory.assign(
            self.left_key[1],
            crop,
            frame_index=1,
            camera_id=self.left_key[0],
            detection_confidence=0.95,
            orientation="front",
            observed_at=1.0,
            map_point=(200.0, 0.0),
            intake_body_complete=True,
        )

        self.assertIsNone(returned_id)
        self.assertIsNone(self.memory.lookup_track_key(self.left_key))
        self.assertIn(identity_id, self.memory.identities)

    def test_split_survivor_rejoins_new_master_and_runs_fresh_intake(self):
        self.memory.close(drain=False)
        extractor = FixedExtractor((1.0, 0.0))
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=extractor,
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            cross_camera_fusion_distance_cm=50.0,
            enable_role_classification=False,
            enable_demographics=False,
            provisional_location_confirm_frames=12,
            provisional_split_recovery_seconds=10.0,
            start_worker=True,
        )
        temporary_id = self.create_pair()
        for track_key in (self.left_key, self.right_key):
            self.memory.track_binding_metadata[track_key][
                "provisional_intake_complete"
            ] = True

        # Reproduce the asymmetric split: cam 1 is released while cam 2 keeps
        # the completed tmp_1 intake that used to strand it forever.
        self.memory._physical_match_allowed_locked = (
            lambda _identity_id, camera_id, *_args, **_kwargs: camera_id != "cam_1"
        )
        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.memory.assign(
            self.left_key[1],
            crop,
            frame_index=1,
            camera_id=self.left_key[0],
            detection_confidence=0.95,
            orientation="front",
            observed_at=1.0,
            map_point=(100.0, 100.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        master_id = self.memory.lookup_track_key(self.left_key)
        self.assertEqual(master_id, 1)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), temporary_id)
        self.assertTrue(
            self.memory.can_recover_provisional_pair(self.left_key, self.right_key)
        )

        coordinator = CrossCameraProvisionalCoordinator(
            self.memory,
            max_distance_cm=50.0,
            max_skew_seconds=0.35,
            required_pair_frames=3,
            location_confirm_frames=12,
        )
        for frame_index in range(2, 5):
            observations = {
                "cam_1": [
                    {
                        "camera_id": "cam_1",
                        "local_track_id": self.left_key[1],
                        "point": (100.0, 100.0),
                        "captured_at": float(frame_index),
                    }
                ],
                "cam_2": [
                    {
                        "camera_id": "cam_2",
                        "local_track_id": self.right_key[1],
                        "point": (102.0, 100.0),
                        "captured_at": float(frame_index),
                    }
                ],
            }
            coordinator.update(observations)

        self.assertNotIn(temporary_id, self.memory.identities)
        self.assertEqual(self.memory.lookup_track_key(self.right_key), master_id)
        self.assertEqual(self.memory.track_identity_state(self.right_key), "provisional")
        self.assertFalse(
            self.memory.assignment_metadata(
                self.right_key[1],
                camera_id=self.right_key[0],
            )["provisional_intake_complete"]
        )

        self.memory.assign(
            self.right_key[1],
            crop,
            frame_index=5,
            camera_id=self.right_key[0],
            detection_confidence=0.95,
            orientation="front",
            observed_at=5.0,
            map_point=(102.0, 100.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(self.memory.lookup_track_key(self.right_key), master_id)
        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")
        self.assertTrue(
            self.memory.assignment_metadata(
                self.right_key[1],
                camera_id=self.right_key[0],
            )["provisional_intake_complete"]
        )
        self.assertEqual(extractor.batch_sizes, [1, 1])

    def test_split_recovery_timeout_releases_survivor_to_normal_intake(self):
        self.memory.close(drain=False)
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            provisional_split_recovery_seconds=0.0,
            start_worker=False,
        )
        temporary_id = self.create_pair()
        for track_key in (self.left_key, self.right_key):
            self.memory.track_binding_metadata[track_key][
                "provisional_intake_complete"
            ] = True
        self.memory._physical_match_allowed_locked = (
            lambda _identity_id, camera_id, *_args, **_kwargs: camera_id != "cam_1"
        )
        crop = np.full((64, 32, 3), 127, dtype=np.uint8)

        self.memory.assign(
            self.left_key[1],
            crop,
            frame_index=1,
            camera_id=self.left_key[0],
            observed_at=1.0,
            map_point=(100.0, 100.0),
            intake_body_complete=True,
        )
        self.memory.assign(
            self.right_key[1],
            crop,
            frame_index=2,
            camera_id=self.right_key[0],
            observed_at=2.0,
            map_point=(200.0, 200.0),
            intake_body_complete=True,
        )

        self.assertNotIn(temporary_id, self.memory.identities)
        self.assertIsNone(self.memory.lookup_track_key(self.right_key))
        self.assertIn(self.right_key, self.memory.pending_intake)

    def test_attached_track_global_reid_confirms_the_existing_master(self):
        self.memory.close(drain=False)
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor((1.0, 0.0)),
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id
        self.memory.create_provisional_pair(self.left_key, self.right_key)

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.memory.assign(
            20,
            crop,
            frame_index=1,
            camera_id="cam_2",
            detection_confidence=0.95,
            orientation="front",
            observed_at=1.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(set(self.memory.identities), {identity_id})
        self.assertEqual(self.memory.lookup_track_key(self.right_key), identity_id)
        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")
        self.assertTrue(
            self.memory.assignment_metadata(20, camera_id="cam_2")[
                "appearance_confirmed"
            ]
        )
        # Crops gathered while the track was still a guess are withheld, not
        # delivered into the identity on confirmation.  What the member
        # contributes from here on arrives through the ordinary path.
        self.assertNotIn(self.right_key, self.memory.pending_member_evidence)
        record = self.memory.identities[identity_id]
        self.assertNotIn("cam_2", record["camera_baselines"])
        self.assertIsNone(
            record.get("camera_views", {}).get("cam_2", {}).get("front")
        )

    def test_borderline_location_match_needs_two_batches_before_commit(self):
        self.memory.close(drain=False)
        borderline_feature = (0.76, float(np.sqrt(1.0 - 0.76**2)))
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor(borderline_feature),
            distance_threshold=0.27,
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        identity_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[identity_id] = record
        self.memory.next_identity_id = 8
        self.memory.track_to_identity[self.left_key] = identity_id
        self.memory.create_provisional_pair(self.left_key, self.right_key)

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.memory.assign(
            20,
            crop,
            frame_index=1,
            camera_id="cam_2",
            detection_confidence=0.95,
            orientation="front",
            observed_at=1.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(self.memory.track_identity_state(self.right_key), "provisional")
        self.assertNotIn("cam_2", record["camera_baselines"])
        self.assertIn(self.right_key, self.memory.pending_member_evidence)

        self.memory._process_provisional_semantic_task(
            {
                "type": "provisional_semantic",
                "identity_id": identity_id,
                "track_key": self.right_key,
                "slot_name": "back",
                "sample": {
                    "crop": crop,
                    "frame_index": 50,
                    "camera_id": "cam_2",
                    "observed_at": 1.5,
                    "sharpness": 200.0,
                    "area": float(crop.shape[0] * crop.shape[1]),
                    "detection_confidence": 0.95,
                    "orientation": "back",
                    "map_point": (0.0, 0.0),
                },
            }
        )
        # The crop is held in staging, not written into the identity.  cam_2
        # may have no stored views at all now that attaching no longer creates
        # an empty set for the camera.
        self.assertIsNone(
            record.get("camera_views", {}).get("cam_2", {}).get("back")
        )
        self.assertIsNotNone(
            self.memory.pending_member_evidence[self.right_key]["views"]["back"]
        )

        self.memory.assign(
            20,
            crop,
            frame_index=100,
            camera_id="cam_2",
            detection_confidence=0.95,
            orientation="front",
            observed_at=2.0,
            map_point=(0.0, 0.0),
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        # Two batches are still required before the member is confirmed; what
        # changed is that its staged crops are withheld rather than written
        # into the identity once it is.
        self.assertEqual(self.memory.track_identity_state(self.right_key), "confirmed")
        self.assertNotIn(self.right_key, self.memory.pending_member_evidence)
        self.assertNotIn("cam_2", record["camera_baselines"])
        self.assertIsNone(record.get("camera_views", {}).get("cam_2", {}).get("front"))

    def test_borderline_global_match_does_not_create_or_bind_on_first_batch(self):
        self.memory.close(drain=False)
        borderline_feature = (0.76, float(np.sqrt(1.0 - 0.76**2)))
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            reid_extractor=FixedExtractor(borderline_feature),
            distance_threshold=0.27,
            intake_frames=1,
            intake_delay_seconds=0.0,
            blur_threshold=0.0,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        existing_id = 7
        record = self.memory._new_record(identity_state="confirmed")
        record["gallery"]["baseline"] = make_slot((1.0, 0.0), "cam_1")
        self.memory.identities[existing_id] = record
        self.memory.next_identity_id = 8

        checker = np.indices((64, 32)).sum(axis=0) % 2
        crop = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        self.memory.assign(
            30,
            crop,
            frame_index=1,
            camera_id="cam_2",
            detection_confidence=0.95,
            observed_at=1.0,
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertIsNone(self.memory.lookup_track_key(("cam_2", 30)))
        self.assertEqual(set(self.memory.identities), {existing_id})

        self.memory.assign(
            30,
            crop,
            frame_index=100,
            camera_id="cam_2",
            detection_confidence=0.95,
            observed_at=2.0,
            intake_body_complete=True,
        )
        self.assertTrue(self.memory.wait_for_idle())

        self.assertEqual(self.memory.lookup_track_key(("cam_2", 30)), existing_id)
        self.assertEqual(set(self.memory.identities), {existing_id})

    def test_delayed_intake_confirms_latest_location_assignment(self):
        identity_id, track_key = self.process_delayed_location_assigned_task(
            (1.0, 0.0)
        )

        self.assertEqual(set(self.memory.identities), {identity_id})
        self.assertEqual(self.memory.lookup_track_key(track_key), identity_id)
        self.assertEqual(self.memory.track_identity_state(track_key), "confirmed")
        self.assertTrue(
            self.memory.assignment_metadata(18, camera_id="cam_2")[
                "appearance_confirmed"
            ]
        )

    def test_delayed_intake_mismatch_challenges_without_creating_new_id(self):
        identity_id, track_key = self.process_delayed_location_assigned_task(
            (0.0, 1.0),
            task_provisional_identity_id=99,
        )

        self.assertEqual(set(self.memory.identities), {identity_id})
        self.assertEqual(self.memory.lookup_track_key(track_key), identity_id)
        self.assertEqual(self.memory.track_identity_state(track_key), "challenged")
        metadata = self.memory.assignment_metadata(18, camera_id="cam_2")
        self.assertFalse(metadata["appearance_confirmed"])
        self.assertAlmostEqual(metadata["distance"], 1.0)
        record = self.memory.identities[identity_id]
        self.assertNotIn("cam_2", record["camera_baselines"])
        self.assertNotIn("cam_2", record["camera_views"])
        self.assertNotIn(track_key, self.memory.pending_member_evidence)

    def test_physical_gate_requires_a_consistent_run_of_bad_samples(self):
        identity_id = self.create_pair()
        self.memory.cross_camera_fusion_distance_cm = 50.0
        with self.memory._lock:
            self.memory.recent_master_observations[identity_id] = {
                "cam_1": {
                    "track_key": self.left_key,
                    "map_point": (0.0, 0.0),
                    "observed_at": 1.0,
                }
            }
            results = [
                self.memory._physical_match_allowed_locked(
                    identity_id,
                    "cam_2",
                    (100.0, 0.0),
                    1.0,
                    track_key=self.right_key,
                    established_binding=True,
                )
                for _ in range(DEFAULT_POSITION_SPLIT_FRAMES)
            ]

        expected = [True] * (DEFAULT_POSITION_SPLIT_FRAMES - 1) + [False]
        self.assertEqual(results, expected)


if __name__ == "__main__":
    unittest.main()
