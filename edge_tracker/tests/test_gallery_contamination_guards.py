"""Guards that stop one person's crop landing in another person's gallery.

Three separate paths allowed a location-only guess to become permanent:

* an unpromoted group wrote its crops to the evidence folder immediately, so
  two people who merely walked close together shared a folder;
* the stable-location fallback confirmed a member that appearance had already
  argued against; and
* refusing a merge left the group free to be promoted into a second master for
  a person who already had one.

These tests cover all three, and pin the behaviour that must survive: a group
that appearance vouches for still gets its crops, and a member appearance never
judged is still confirmable by location alone.
"""

import unittest
from unittest import mock

import numpy as np

from reid_memory import AppearanceIdentityMemory


def make_slot(camera_id, seed=1.0, frame_index=1):
    feature = np.full(8, float(seed), dtype=np.float32)
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


class MemoryTestCase(unittest.TestCase):
    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            provisional_location_confirm_frames=3,
            start_worker=False,
        )
        # Capture evidence writes instead of spawning the writer subprocess.
        self.saved = []
        self.memory._queue_evidence_save = self.saved.append

    def tearDown(self):
        self.memory.close(drain=False)

    def make_provisional(self, identity_id=-1, member_keys=()):
        record = self.memory._new_record(identity_state="provisional")
        record["member_track_keys"] = set(member_keys)
        record["global_reid_checked_track_keys"] = set(member_keys)
        self.memory.identities[identity_id] = record
        for key in member_keys:
            self.memory.track_to_identity[key] = identity_id
        return record


class DeferredEvidenceTests(MemoryTestCase):
    """An unpromoted group is a geometric guess, not a person."""

    def test_unpromoted_group_writes_no_image(self):
        record = self.make_provisional()
        self.memory._defer_provisional_evidence_locked(
            record, {"identity_id": -1, "slot_name": "baseline_cam_1",
                     "crop": None, "output_path": "/ev/Temporary_0001/a.png"}
        )
        self.assertEqual(self.saved, [])
        self.assertEqual(len(record["deferred_evidence_tasks"]), 1)

    def test_promotion_releases_images_into_the_master_folder(self):
        record = self.make_provisional(member_keys=[("cam_1", 1)])
        record["camera_baselines"] = {"cam_1": make_slot("cam_1")}
        record["camera_baselines"]["cam_1"]["image_path"] = "/ev/Temporary_0001/a.png"
        self.memory._defer_provisional_evidence_locked(
            record, {"identity_id": -1, "slot_name": "baseline_cam_1",
                     "crop": None, "output_path": "/ev/Temporary_0001/a.png"}
        )

        with self.memory._lock:
            master_id = self.memory._promote_provisional_locked(-1, "stable_location")

        self.assertIsNotNone(master_id)
        self.assertEqual(len(self.saved), 1)
        # Re-addressed from the temporary folder to the master's folder...
        self.assertIn(f"Master_{master_id:04d}", self.saved[0]["output_path"])
        self.assertEqual(self.saved[0]["identity_id"], master_id)
        # ...and the slot must follow, or the saved digest never finds it.
        self.assertEqual(
            self.memory.identities[master_id]["camera_baselines"]["cam_1"]["image_path"],
            self.saved[0]["output_path"],
        )

    def test_group_torn_down_before_promotion_never_writes(self):
        record = self.make_provisional()
        self.memory._defer_provisional_evidence_locked(
            record, {"identity_id": -1, "slot_name": "baseline_cam_1",
                     "crop": None, "output_path": "/ev/Temporary_0001/a.png"}
        )
        self.memory.identities.pop(-1)
        self.assertEqual(self.saved, [])


class StableLocationOverrideTests(MemoryTestCase):
    """Standing in the right place cannot overrule an appearance rejection."""

    def setUp(self):
        super().setUp()
        self.key = ("cam_1", 11)
        self.record = self.memory._new_record()
        self.record["pending_member_keys"] = {self.key}
        self.record["gallery"]["baseline"] = make_slot("cam_2")
        self.memory.identities[1] = self.record
        self.memory.track_to_identity[self.key] = 1

    def _confirm(self, reason):
        with self.memory._lock:
            return self.memory._confirm_pending_members_locked(
                1, reason, appearance_confirmed=(reason == "global_reid")
            )

    def test_rejected_member_is_not_confirmed_by_location(self):
        self.record["appearance_rejected_member_keys"] = {self.key}
        self.assertFalse(self._confirm("stable_location"))
        self.assertIn(self.key, self.record["pending_member_keys"])

    def test_unjudged_member_is_still_confirmed_by_location(self):
        self.assertTrue(self._confirm("stable_location"))
        self.assertNotIn(self.key, self.record["pending_member_keys"])

    def test_challenged_member_is_not_confirmed_by_location(self):
        self.record["pending_member_keys"] = set()
        self.record["challenged_member_keys"] = {self.key}
        self.record["appearance_rejected_member_keys"] = {self.key}
        self.assertFalse(self._confirm("stable_location"))
        self.assertIn(self.key, self.record["challenged_member_keys"])

    def test_positive_reid_can_still_confirm_a_rejected_member(self):
        self.record["appearance_rejected_member_keys"] = {self.key}
        self.assertTrue(self._confirm("global_reid"))


class BlockedMergePromotionTests(MemoryTestCase):
    """Refusing a merge must not manufacture a second master instead."""

    def setUp(self):
        super().setUp()
        self.incumbent = ("cam_1", 10)
        self.member = ("cam_1", 54)
        self.memory.identities[5] = self.memory._new_record()
        self.memory.track_to_identity[self.incumbent] = 5

        self.record = self.make_provisional(identity_id=-10, member_keys=[self.member])
        self.record["camera_baselines"] = {"cam_1": make_slot("cam_1")}
        self.record["merge_blocked_by_master"] = 5

    def _promote(self):
        with self.memory._lock:
            return self.memory._promote_provisional_locked(-10, "stable_location")

    def test_no_new_master_while_the_incumbent_is_visible(self):
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.incumbent, self.member}
        self.assertIsNone(self._promote())
        self.assertIn(-10, self.memory.identities)

    def test_promotion_resumes_once_the_incumbent_disappears(self):
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.member}
        master_id = self._promote()
        self.assertIsNotNone(master_id)
        self.assertGreater(master_id, 0)
        self.assertNotIn("merge_blocked_by_master", self.memory.identities[master_id])

    def test_unblocked_group_promotes_normally(self):
        self.record.pop("merge_blocked_by_master")
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.incumbent, self.member}
        self.assertIsNotNone(self._promote())


class SealedGalleryTests(MemoryTestCase):
    """A complete gallery stops accepting replacements.

    Overwrites were decided on sharpness alone, and sharpness cannot tell one
    man from another.  During a swap a crisp crop of the wrong person beat the
    softer crop of the right one and took the slot -- the baseline included.
    Two swaps in one huddle blended two identities into an average of both,
    after which no rival could win by a margin and the audit had nothing left
    to separate them with.
    """

    def setUp(self):
        super().setUp()
        self.record = self.memory._new_record(identity_state="confirmed")
        self.memory.identities[1] = self.record

    def fill(self, *slot_names):
        for name in slot_names:
            self.record["gallery"][name] = make_slot("cam_1")

    def sealed(self):
        with self.memory._lock:
            return self.memory._gallery_is_sealed_locked(self.record)

    def test_a_partial_gallery_is_not_sealed(self):
        self.fill("baseline", "front", "back", "left_side")
        self.assertFalse(self.sealed(), "one slot short must still accept crops")

    def test_baseline_and_four_sides_seals_it(self):
        self.fill("baseline", "front", "back", "left_side", "right_side")
        self.assertTrue(self.sealed())

    def test_a_sharper_crop_cannot_take_a_filled_slot(self):
        existing = make_slot("cam_1")
        sharper = make_slot("cam_1")
        sharper["sharpness"] = 99999.0
        self.assertFalse(self.memory._slot_admits(sharper, existing, sealed=True))
        # ...but quality still decides while the gallery is still filling.
        self.assertTrue(self.memory._slot_admits(sharper, existing, sealed=False))

    def test_empty_slots_still_fill_on_a_sealed_identity(self):
        self.assertTrue(self.memory._slot_admits(make_slot("cam_2"), None, sealed=True))

    def test_a_merge_cannot_overwrite_an_established_gallery(self):
        self.fill("baseline", "front", "back", "left_side", "right_side")
        original = make_slot("cam_1")
        self.record["camera_baselines"] = {"cam_1": original}
        intruder = make_slot("cam_1")
        intruder["sharpness"] = 99999.0

        with self.memory._lock:
            sealed = self.memory._gallery_is_sealed_locked(self.record)
            admitted = self.memory._slot_admits(intruder, original, sealed)

        self.assertFalse(admitted)
        self.assertIs(self.record["camera_baselines"]["cam_1"], original)


class DisputedTrackTests(MemoryTestCase):
    """A track the audit has caught matching another master contributes nothing."""

    def setUp(self):
        super().setUp()
        self.record = self.memory._new_record(identity_state="confirmed")
        self.record["gallery"]["baseline"] = make_slot("cam_1")
        self.memory.identities[1] = self.record
        self.key = ("cam_1", 7)

    def reject(self):
        with self.memory._lock:
            return self.memory._gallery_admission_rejected_locked(
                1,
                self.record,
                make_slot("cam_1"),
                "right_side",
                "master_gallery",
                track_key=self.key,
            )

    def test_a_flagged_track_may_not_write(self):
        self.memory.identity_audit_state[self.key] = {"next_due": 0.0, "rivals": {2: 1}}
        self.assertTrue(self.reject())

    def test_an_unflagged_track_writes_normally(self):
        self.memory.identity_audit_state[self.key] = {"next_due": 0.0, "rivals": {}}
        self.assertFalse(self.reject())

    def test_a_cleared_dispute_restores_the_track(self):
        self.memory.identity_audit_state[self.key] = {"next_due": 0.0, "rivals": {2: 1}}
        self.assertTrue(self.reject())
        self.memory.identity_audit_state[self.key]["rivals"] = {}
        self.assertFalse(self.reject())


def directional_slot(camera_id, vector, frame_index=1, sharpness=200.0):
    feature = np.asarray(vector, dtype=np.float32).reshape(-1)
    feature = feature / np.linalg.norm(feature)
    slot = make_slot(camera_id, frame_index=frame_index)
    slot["feature"] = feature
    slot["feature_dimension"] = int(feature.size)
    slot["sharpness"] = float(sharpness)
    return slot


class GalleryAdmissionTests(MemoryTestCase):
    """A swapped local track yields a flawless crop of the wrong person.

    Sharpness, confidence and body completeness all pass on that crop, so the
    only thing that can refuse it is a comparison against what the identity
    already looks like.
    """

    def setUp(self):
        super().setUp()
        self.record = self.memory._new_record(identity_state="confirmed")
        baseline = directional_slot("cam_1", (1.0, 0.0, 0.0))
        baseline["image_path"] = "/evidence/Master_0007/Slot_baseline_cam_1_frame_1.png"
        self.record["gallery"]["baseline"] = baseline
        self.memory.identities[7] = self.record

    def reject(self, slot, slot_name="left_side"):
        with self.memory._lock:
            return self.memory._gallery_admission_rejected_locked(
                7,
                self.record,
                slot,
                slot_name,
                "master_gallery",
            )

    def test_wrong_person_crop_is_refused(self):
        # Clean, sharp, complete -- and somebody else entirely.
        self.assertTrue(self.reject(directional_slot("cam_1", (0.0, 1.0, 0.0), sharpness=9000.0)))

    def test_genuine_new_angle_is_admitted(self):
        self.assertFalse(self.reject(directional_slot("cam_1", (1.0, 0.35, 0.0))))

    def test_first_view_has_nothing_to_disagree_with(self):
        empty = self.memory._new_record(identity_state="confirmed")
        self.memory.identities[8] = empty
        with self.memory._lock:
            rejected = self.memory._gallery_admission_rejected_locked(
                8,
                empty,
                directional_slot("cam_1", (0.0, 1.0, 0.0)),
                "front",
                "master_gallery",
            )
        self.assertFalse(rejected)

    def test_agreeing_with_most_of_the_gallery_admits_a_new_angle(self):
        # A genuine new angle is not identical to what is stored, and need not
        # be: it has to satisfy the gallery as a whole, not every member of it.
        self.record["gallery"]["front"] = directional_slot("cam_1", (1.0, 0.30, 0.0))
        self.record["camera_views"] = {
            "cam_2": {"left_side": directional_slot("cam_2", (1.0, 0.45, 0.0))}
        }
        self.assertFalse(self.reject(directional_slot("cam_2", (1.0, 0.38, 0.0)), "back"))

    def test_one_agreement_no_longer_carries_a_lone_dissenter(self):
        # This used to be admitted: matching a single stored view was enough.
        # Measured crops of one person agree with their gallery broadly, and
        # letting one friendly angle speak for the rest is what admitted a
        # second man's photographs.
        self.record["camera_views"] = {
            "cam_2": {"left_side": directional_slot("cam_2", (0.0, 1.0, 0.0))}
        }
        self.assertTrue(self.reject(directional_slot("cam_2", (0.0, 1.0, 0.2)), "back"))

    def test_the_deciding_photo_is_named_in_the_record(self):
        # A wrong admission can only be diagnosed by opening the crop that
        # vouched for it, so the decision has to name that file.
        events = []
        with mock.patch(
            "reid_memory.identity_event",
            side_effect=lambda name, **fields: events.append((name, fields)),
        ):
            self.reject(directional_slot("cam_1", (1.0, 0.35, 0.0)))

        name, fields = events[-1]
        self.assertEqual(name, "gallery_admission_accepted")
        self.assertEqual(fields["closest_stored_view"], "gallery:baseline")
        self.assertEqual(
            fields["closest_stored_view_image_path"],
            "/evidence/Master_0007/Slot_baseline_cam_1_frame_1.png",
        )
        self.assertEqual(
            [row["stored_view"] for row in fields["comparisons"]],
            ["gallery:baseline"],
        )
        self.assertIn("image_path", fields["comparisons"][0])

    def test_one_forgiving_view_cannot_vouch_for_a_stranger(self):
        # The closest stored view used to decide alone.  The crop that put
        # Haoran into Mikail's gallery sat 0.26 from one photo and 0.32-0.41
        # from the other nine, and that single number admitted it.
        self.record["gallery"]["front"] = directional_slot("cam_1", (0.0, 1.0, 0.0))
        self.record["gallery"]["back"] = directional_slot("cam_1", (0.0, 0.0, 1.0))
        self.record["camera_views"] = {}

        events = []
        with mock.patch(
            "reid_memory.identity_event",
            side_effect=lambda name, **fields: events.append((name, fields)),
        ):
            self.reject(directional_slot("cam_1", (1.0, 0.0, 0.0)))

        _name, fields = events[-1]
        self.assertLess(fields["best_distance"], 0.01, "one view agrees exactly")
        self.assertGreater(
            fields["verdict_distance"],
            fields["best_distance"],
            "the verdict must reflect the gallery, not its friendliest member",
        )

    def test_incomparable_feature_space_is_not_judged(self):
        slot = directional_slot("cam_1", (0.0, 1.0, 0.0))
        slot["feature_space_id"] = "other-feature-space"
        self.assertFalse(self.reject(slot))


if __name__ == "__main__":
    unittest.main()
