"""A track that never comes back must not hold its group hostage.

Promotion requires every bound member of a group to have completed its photo
check.  The only cleanup that removed a stale member ran over the tracks a
camera could currently see, so a track that vanished for good was never
examined: its binding survived, it stayed on its group's member list, and the
roll call waited forever on photos it could not deliver.  Both cameras kept a
live box on the person and both kept showing "analysing".

Occlusion is what triggers it.  A person hidden behind a crowd loses their
track number, gets re-acquired under a new one, and the abandoned number is
left behind still bound -- so the very situation that makes identification hard
also creates the thing that makes it impossible.
"""

import unittest

import numpy as np

from reid_memory import AppearanceIdentityMemory


def make_slot(feature, camera_id, frame_index=1):
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


class AbandonedBindingTests(unittest.TestCase):
    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            provisional_location_confirm_frames=3,
            track_abandon_frames=45,
            start_worker=False,
        )
        self.live = ("cam_2", 19)
        self.orphan = ("cam_2", 16)
        self.memory.track_to_identity[self.orphan] = -4
        self.memory.track_to_identity[self.live] = -4
        self.memory.track_last_seen[self.orphan] = (718, 718.0)
        self.memory.track_last_seen[self.live] = (1000, 1000.0)

    def tearDown(self):
        self.memory.close(drain=False)

    def observe(self, frame_index):
        return self.memory.observe_tracks(
            [19],
            None,
            frame_index=frame_index,
            camera_id="cam_2",
        )

    def test_vanished_track_is_struck_off(self):
        self.observe(1000)
        self.assertNotIn(self.orphan, self.memory.track_to_identity)
        self.assertIn(self.live, self.memory.track_to_identity)

    def test_briefly_missing_track_keeps_its_binding(self):
        # Still inside the tracker's own buffer -- it may yet come back under
        # the same number, so the identity layer must not give up first.
        self.memory.track_last_seen[self.orphan] = (990, 990.0)
        self.observe(1000)
        self.assertIn(self.orphan, self.memory.track_to_identity)

    def test_abandoning_removes_the_key_from_its_group(self):
        record = self.memory._new_record(identity_state="provisional")
        record["member_track_keys"] = {self.orphan, self.live}
        self.memory.identities[-4] = record

        self.observe(1000)

        self.assertNotIn(self.orphan, record["member_track_keys"])
        self.assertIn(self.live, record["member_track_keys"])

    def test_group_stranded_by_a_vanished_member_can_promote_again(self):
        record = self.memory._new_record(identity_state="provisional")
        record["member_track_keys"] = {self.orphan, self.live}
        # The live track finished its checks; the vanished one never did.
        record["global_reid_checked_track_keys"] = {self.live}
        record["camera_baselines"] = {
            "cam_1": make_slot((1.0, 0.0), "cam_1"),
            "cam_2": make_slot((1.0, 0.5), "cam_2"),
        }
        self.memory.identities[-4] = record

        with self.memory._lock:
            self.assertFalse(
                self.memory._provisional_global_reid_complete_locked(-4),
                "the orphan should block promotion before the sweep runs",
            )

        self.observe(1000)

        with self.memory._lock:
            self.assertTrue(self.memory._provisional_global_reid_complete_locked(-4))

    def test_sweep_leaves_other_cameras_alone(self):
        other = ("cam_1", 18)
        self.memory.track_to_identity[other] = -4
        self.memory.track_last_seen[other] = (700, 700.0)

        self.observe(1000)

        self.assertIn(other, self.memory.track_to_identity)


class RollCallVisibilityTests(unittest.TestCase):
    """The backstop: never wait on a member the camera cannot see."""

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            track_abandon_frames=0,  # sweep disabled: test the backstop alone
            start_worker=False,
        )
        self.present = ("cam_1", 18)
        self.absent = ("cam_2", 16)
        record = self.memory._new_record(identity_state="provisional")
        record["member_track_keys"] = {self.present, self.absent}
        record["global_reid_checked_track_keys"] = {self.present}
        self.memory.identities[-4] = record
        self.memory.track_to_identity[self.present] = -4
        self.memory.track_to_identity[self.absent] = -4

    def tearDown(self):
        self.memory.close(drain=False)

    def complete(self):
        with self.memory._lock:
            return self.memory._provisional_global_reid_complete_locked(-4)

    def test_unreported_cameras_are_not_treated_as_absent(self):
        self.assertFalse(self.complete())

    def test_member_its_camera_cannot_see_is_not_waited_on(self):
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.present}
        self.memory.visible_track_keys_by_camera["cam_2"] = set()
        self.assertTrue(self.complete())

    def test_visible_unchecked_member_still_blocks(self):
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.present}
        self.memory.visible_track_keys_by_camera["cam_2"] = {self.absent}
        self.assertFalse(self.complete())

    def test_a_group_nobody_can_see_does_not_promote(self):
        self.memory.visible_track_keys_by_camera["cam_1"] = set()
        self.memory.visible_track_keys_by_camera["cam_2"] = set()
        self.assertFalse(self.complete())


if __name__ == "__main__":
    unittest.main()
