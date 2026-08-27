"""Guards that stop one master ID from being served to two live local tracks.

Two separate defects allowed a single person to become several master IDs:

* the location-driven binding paths did not enforce the single-owner rule that
  the ReID paths already enforce, so two visible tracks in one camera could end
  up holding the same master; and
* every identity-keyed position memory is shared, so once that happened both
  tracks read and wrote one slot and their map points collapsed onto one dot.

These tests cover both guards, and equally importantly they pin the behaviour
that must *not* change: a person whose local track is renumbered still inherits
their previous foot position and motion state across frames.
"""

import unittest

import numpy as np

from core_math import (
    recall_recent_foot_point,
    reject_impossible_foot_jump,
    remember_foot_point,
    update_map_motion,
)
from reid_memory import AppearanceIdentityMemory


TRACK_A = 29
TRACK_B = 3
OWNER_A = ("cam_1", TRACK_A)
OWNER_B = ("cam_1", TRACK_B)


class MapMotionOwnerConflictTests(unittest.TestCase):
    """update_map_motion is what produced the exactly-identical coordinates.

    Every detection in a frame carries one capture timestamp, so a second track
    sharing the identity used to hit the dt==0 branch and be handed the first
    track's smoothed point verbatim.
    """

    def setUp(self):
        self.memory = {}
        self.key = ("identity", 1)

    def _seed_owner_a(self):
        update_map_motion(self.memory, self.key, (100.0, 100.0), 1.0, owner=OWNER_A)
        return update_map_motion(self.memory, self.key, (110.0, 100.0), 2.0, owner=OWNER_A)

    def test_second_owner_in_same_frame_keeps_its_own_measurement(self):
        self._seed_owner_a()
        point, speed, status = update_map_motion(
            self.memory, self.key, (300.0, 300.0), 2.0, owner=OWNER_B
        )
        self.assertEqual(status, "owner_conflict")
        self.assertEqual(point, (300.0, 300.0))
        self.assertIsNone(speed)

    def test_second_owner_does_not_overwrite_the_first(self):
        (expected_x, expected_y), _speed, _status = self._seed_owner_a()
        update_map_motion(self.memory, self.key, (300.0, 300.0), 2.0, owner=OWNER_B)
        stored = self.memory[self.key]
        self.assertEqual(stored["owner"], OWNER_A)
        self.assertAlmostEqual(float(stored["point"][0]), expected_x)
        self.assertAlmostEqual(float(stored["point"][1]), expected_y)

    def test_same_owner_repeating_a_timestamp_is_unchanged(self):
        self._seed_owner_a()
        point, _speed, status = update_map_motion(
            self.memory, self.key, (999.0, 999.0), 2.0, owner=OWNER_A
        )
        self.assertEqual(status, "same_time")
        self.assertNotEqual(point, (999.0, 999.0))

    def test_owner_unaware_callers_keep_legacy_behaviour(self):
        update_map_motion(self.memory, self.key, (100.0, 100.0), 1.0)
        update_map_motion(self.memory, self.key, (110.0, 100.0), 2.0)
        point, _speed, status = update_map_motion(self.memory, self.key, (999.0, 999.0), 2.0)
        self.assertEqual(status, "same_time")
        self.assertNotEqual(point, (999.0, 999.0))

    def test_a_renumbered_track_still_inherits_motion_state(self):
        """The identity key exists so smoothing survives a track renumber."""
        self._seed_owner_a()
        point, _speed, status = update_map_motion(
            self.memory, self.key, (120.0, 100.0), 3.0, owner=OWNER_B
        )
        self.assertEqual(status, "smooth")
        self.assertNotEqual(point, (120.0, 100.0))  # smoothed against A's history


class FootMemoryOwnerConflictTests(unittest.TestCase):
    def setUp(self):
        self.memory = {}
        remember_foot_point(self.memory, TRACK_A, (100.0, 200.0), frame_index=5, identity_id=1)

    def test_other_track_in_same_frame_is_refused(self):
        self.assertIsNone(
            recall_recent_foot_point(self.memory, TRACK_B, 5, 30, identity_id=1)
        )

    def test_owning_track_in_same_frame_still_recalls(self):
        recalled = recall_recent_foot_point(self.memory, TRACK_A, 5, 30, identity_id=1)
        self.assertIsNotNone(recalled)
        np.testing.assert_allclose(recalled, [100.0, 200.0])

    def test_renumbered_track_inherits_across_frames(self):
        """Carry-over is the whole point of the identity key; keep it working."""
        recalled = recall_recent_foot_point(self.memory, TRACK_B, 6, 30, identity_id=1)
        self.assertIsNotNone(recalled)
        np.testing.assert_allclose(recalled, [100.0, 200.0])

    def test_other_track_keeps_its_own_measured_foot(self):
        self.assertIsNone(
            reject_impossible_foot_jump(self.memory, TRACK_B, (900.0, 900.0), 5, 120.0, identity_id=1)
        )

    def test_real_teleport_is_still_rejected_for_the_owner(self):
        held = reject_impossible_foot_jump(
            self.memory, TRACK_A, (900.0, 900.0), 6, 120.0, identity_id=1
        )
        self.assertIsNotNone(held)
        np.testing.assert_allclose(held, [100.0, 200.0])


class SingleOwnerBindingTests(unittest.TestCase):
    """Location-driven binding must respect the ReID paths' single-owner rule."""

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        self.incumbent = ("cam_1", 10)
        self.newcomer = ("cam_1", 11)
        self.partner = ("cam_2", 20)
        self.memory.identities[1] = self.memory._new_record()
        self.memory.track_to_identity[self.incumbent] = 1
        self.memory.track_to_identity[self.partner] = 1
        self.memory.visible_track_keys_by_camera["cam_2"] = {self.partner}

    def tearDown(self):
        self.memory.close(drain=False)

    def _set_incumbent_visible(self, visible):
        self.memory.visible_track_keys_by_camera["cam_1"] = (
            {self.incumbent, self.newcomer} if visible else {self.newcomer}
        )

    def test_pair_declined_while_incumbent_is_on_screen(self):
        self._set_incumbent_visible(True)
        self.assertIsNone(self.memory.create_provisional_pair(self.newcomer, self.partner))

    def test_declined_pair_leaves_no_partial_state(self):
        self._set_incumbent_visible(True)
        self.memory.create_provisional_pair(self.newcomer, self.partner)
        self.assertNotIn(self.newcomer, self.memory.track_to_identity)
        record = self.memory.identities[1]
        self.assertNotIn(self.newcomer, record["member_track_keys"])
        self.assertNotIn(self.newcomer, record["pending_member_keys"])

    def test_pair_allowed_once_the_incumbent_track_is_gone(self):
        self._set_incumbent_visible(False)
        self.assertEqual(self.memory.create_provisional_pair(self.newcomer, self.partner), 1)
        self.assertEqual(self.memory.track_to_identity[self.newcomer], 1)

    def test_unbound_pair_still_forms_a_temporary_group(self):
        self._set_incumbent_visible(True)
        fresh_left, fresh_right = ("cam_1", 50), ("cam_2", 60)
        identity_id = self.memory.create_provisional_pair(fresh_left, fresh_right)
        self.assertIsNotNone(identity_id)
        self.assertLess(identity_id, 0)


class MergeSingleOwnerTests(unittest.TestCase):
    """The merge path only retired *non-visible* owners, so it could double-bind."""

    PROVISIONAL_ID = -1

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        self.incumbent = ("cam_1", 10)
        self.newcomer = ("cam_1", 11)
        self.memory.identities[1] = self.memory._new_record()
        self.memory.track_to_identity[self.incumbent] = 1

        provisional = self.memory._new_record(identity_state="provisional")
        provisional["member_track_keys"] = {self.newcomer}
        self.memory.identities[self.PROVISIONAL_ID] = provisional
        self.memory.track_to_identity[self.newcomer] = self.PROVISIONAL_ID

    def tearDown(self):
        self.memory.close(drain=False)

    def _merge(self):
        with self.memory._lock:
            return self.memory._merge_provisional_into_confirmed_locked(
                self.PROVISIONAL_ID, 1, self.newcomer, "baseline", 0.05, "transreid", "fs-test"
            )

    def test_merge_declined_while_incumbent_is_on_screen(self):
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.incumbent, self.newcomer}
        self.assertIsNone(self._merge())
        self.assertEqual(self.memory.track_to_identity[self.newcomer], self.PROVISIONAL_ID)
        self.assertIn(self.PROVISIONAL_ID, self.memory.identities)

    def test_merge_proceeds_once_the_incumbent_track_is_gone(self):
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.newcomer}
        self.assertEqual(self._merge(), 1)
        self.assertEqual(self.memory.track_to_identity[self.newcomer], 1)
        self.assertNotIn(self.PROVISIONAL_ID, self.memory.identities)


if __name__ == "__main__":
    unittest.main()
