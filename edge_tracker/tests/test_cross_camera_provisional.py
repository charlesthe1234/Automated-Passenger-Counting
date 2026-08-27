import unittest
from unittest.mock import patch

from cross_camera_provisional import CrossCameraProvisionalCoordinator
from identity_debug import configure_identity_debug


def observation(camera_id, track_id, point, captured_at, identity_id=None):
    return {
        "camera_id": camera_id,
        "local_track_id": track_id,
        "identity_id": identity_id,
        "identity_state": None,
        "reid_confirmed": False,
        "point": point,
        "captured_at": captured_at,
    }


def camera_pair(
    left_track=1,
    right_track=7,
    left_point=(100.0, 100.0),
    right_point=(105.0, 102.0),
    captured_at=10.0,
):
    return {
        "cam_1": [observation("cam_1", left_track, left_point, captured_at)],
        "cam_2": [observation("cam_2", right_track, right_point, captured_at)],
    }


class FakeMemory:
    def __init__(self):
        self.bindings = {}
        self.states = {}
        self.next_identity_id = 1
        self.created_pairs = []
        self.location_matches = []
        self.applied_holds = []
        self.released_holds = []

    def lookup_track_key(self, track_key):
        return self.bindings.get(track_key)

    def create_provisional_pair(self, left_key, right_key):
        identity_id = self.next_identity_id
        self.next_identity_id += 1
        self.bindings[left_key] = identity_id
        self.bindings[right_key] = identity_id
        self.states[identity_id] = "provisional"
        self.created_pairs.append((left_key, right_key, identity_id))
        return identity_id

    def identity_state(self, identity_id):
        return self.states.get(identity_id)

    def note_location_match(self, identity_id, streak, observations):
        track_keys = tuple(
            (str(item["camera_id"]), int(item["local_track_id"]))
            for item in observations
        )
        self.location_matches.append((identity_id, streak, track_keys))

    def bind_shared(self, identity_id, *track_keys, state="provisional"):
        for track_key in track_keys:
            self.bindings[track_key] = identity_id
        self.states[identity_id] = state
        self.next_identity_id = max(self.next_identity_id, identity_id + 1)

    def hold_new_master_creation(self, left_key, right_key, hold_token):
        held = tuple(
            key for key in (left_key, right_key) if self.bindings.get(key) is None
        )
        self.applied_holds.append((hold_token, held))
        return held

    def release_new_master_hold(self, left_key, right_key, hold_token, reason):
        self.released_holds.append((hold_token, reason))
        return ()


class CrossCameraProvisionalCoordinatorTests(unittest.TestCase):
    def make_coordinator(self, memory):
        return CrossCameraProvisionalCoordinator(
            memory,
            max_distance_cm=50.0,
            max_skew_seconds=0.35,
            required_pair_frames=3,
            location_confirm_frames=12,
        )

    def test_three_consecutive_location_matches_create_one_provisional_id(self):
        memory = FakeMemory()
        coordinator = self.make_coordinator(memory)

        for frame_number in range(1, 4):
            observations = camera_pair(captured_at=10.0 + frame_number * 0.1)
            coordinator.update(observations)
            if frame_number < 3:
                self.assertIsNone(observations["cam_1"][0]["identity_id"])
                self.assertIsNone(observations["cam_2"][0]["identity_id"])

        left = observations["cam_1"][0]
        right = observations["cam_2"][0]
        self.assertEqual(left["identity_id"], right["identity_id"])
        self.assertEqual(left["identity_state"], "provisional")
        self.assertEqual(right["identity_state"], "provisional")
        self.assertFalse(left["reid_confirmed"])
        self.assertFalse(right["reid_confirmed"])
        self.assertEqual(len(memory.created_pairs), 1)

    def test_one_location_match_does_not_create_a_provisional_id(self):
        memory = FakeMemory()
        coordinator = self.make_coordinator(memory)
        observations = camera_pair()

        coordinator.update(observations)

        self.assertIsNone(observations["cam_1"][0]["identity_id"])
        self.assertIsNone(observations["cam_2"][0]["identity_id"])
        self.assertEqual(memory.created_pairs, [])

    def test_two_by_two_pairing_is_one_to_one_and_uses_nearest_locations(self):
        memory = FakeMemory()
        coordinator = self.make_coordinator(memory)

        for frame_number in range(1, 4):
            captured_at = 20.0 + frame_number * 0.1
            observations = {
                "cam_1": [
                    observation("cam_1", 1, (0.0, 0.0), captured_at),
                    observation("cam_1", 2, (100.0, 0.0), captured_at),
                ],
                "cam_2": [
                    observation("cam_2", 11, (98.0, 0.0), captured_at),
                    observation("cam_2", 12, (2.0, 0.0), captured_at),
                ],
            }
            coordinator.update(observations)

        identities = {
            (item["camera_id"], item["local_track_id"]): item["identity_id"]
            for camera_observations in observations.values()
            for item in camera_observations
        }
        self.assertEqual(identities[("cam_1", 1)], identities[("cam_2", 12)])
        self.assertEqual(identities[("cam_1", 2)], identities[("cam_2", 11)])
        self.assertNotEqual(identities[("cam_1", 1)], identities[("cam_1", 2)])
        self.assertEqual(len(memory.created_pairs), 2)

        created_track_pairs = {
            frozenset((left_key, right_key))
            for left_key, right_key, _identity_id in memory.created_pairs
        }
        self.assertEqual(
            created_track_pairs,
            {
                frozenset((("cam_1", 1), ("cam_2", 12))),
                frozenset((("cam_1", 2), ("cam_2", 11))),
            },
        )

    def test_shared_id_remains_stable_after_the_pair_is_created(self):
        memory = FakeMemory()
        coordinator = self.make_coordinator(memory)

        for frame_number in range(1, 4):
            observations = camera_pair(captured_at=30.0 + frame_number * 0.1)
            coordinator.update(observations)
        original_identity_id = observations["cam_1"][0]["identity_id"]

        later = camera_pair(
            left_point=(108.0, 104.0),
            right_point=(111.0, 106.0),
            captured_at=30.4,
        )
        coordinator.update(later)

        self.assertEqual(later["cam_1"][0]["identity_id"], original_identity_id)
        self.assertEqual(later["cam_2"][0]["identity_id"], original_identity_id)
        self.assertEqual(len(memory.created_pairs), 1)

    def test_existing_shared_identity_notifies_memory_of_location_match(self):
        memory = FakeMemory()
        memory.bind_shared(42, ("cam_1", 1), ("cam_2", 7))
        coordinator = self.make_coordinator(memory)
        observations = camera_pair(captured_at=40.0)

        coordinator.update(observations)

        self.assertEqual(observations["cam_1"][0]["identity_id"], 42)
        self.assertEqual(observations["cam_2"][0]["identity_id"], 42)
        self.assertEqual(observations["cam_1"][0]["identity_state"], "provisional")
        self.assertEqual(memory.created_pairs, [])
        self.assertEqual(
            memory.location_matches,
            [(42, 1, (("cam_1", 1), ("cam_2", 7)))],
        )

    def test_ambiguous_nearby_people_do_not_create_location_ids(self):
        memory = FakeMemory()
        coordinator = self.make_coordinator(memory)

        for frame_number in range(1, 4):
            captured_at = 50.0 + frame_number * 0.1
            observations = {
                "cam_1": [
                    observation("cam_1", 1, (0.0, 0.0), captured_at),
                    observation("cam_1", 2, (10.0, 0.0), captured_at),
                ],
                "cam_2": [
                    observation("cam_2", 11, (4.0, 0.0), captured_at),
                    observation("cam_2", 12, (6.0, 0.0), captured_at),
                ],
            }
            coordinator.update(observations)

        self.assertEqual(memory.created_pairs, [])

    def test_inconsistent_cross_camera_motion_resets_pair_streak(self):
        memory = FakeMemory()
        coordinator = self.make_coordinator(memory)
        positions = [((0.0, 0.0), (0.0, 0.0)), ((1.0, 0.0), (1.0, 0.0)), ((21.0, 0.0), (-19.0, 0.0))]

        for frame_number, (left_point, right_point) in enumerate(positions, start=1):
            observations = camera_pair(
                left_point=left_point,
                right_point=right_point,
                captured_at=60.0 + frame_number * 0.1,
            )
            coordinator.update(observations)

        self.assertEqual(memory.created_pairs, [])
        self.assertEqual(memory.released_holds[-1][1], "movement_disagreement")

    def test_debug_events_record_streak_progress_and_exact_reset_reason(self):
        memory = FakeMemory()
        coordinator = self.make_coordinator(memory)
        # Pair evaluations are now assembled only when debug logging is on, so
        # the gate has to be open for the coordinator to emit anything at all.
        configure_identity_debug(True, None)
        self.addCleanup(configure_identity_debug, False, None)

        with patch("cross_camera_provisional.identity_event") as event:
            first = camera_pair(captured_at=70.0)
            first["cam_1"][0]["frame_index"] = 593
            first["cam_2"][0]["frame_index"] = 593
            coordinator.update(first)

            failed = camera_pair(
                left_point=(0.0, 0.0),
                right_point=(61.0, 0.0),
                captured_at=70.1,
            )
            failed["cam_1"][0]["frame_index"] = 594
            failed["cam_2"][0]["frame_index"] = 594
            coordinator.update(failed)

        evaluations = [
            call.kwargs
            for call in event.call_args_list
            if call.args == ("provisional_pair_evaluated",)
        ]
        self.assertEqual(len(evaluations), 2)
        self.assertTrue(evaluations[0]["accepted"])
        self.assertEqual(evaluations[0]["streak_before"], 0)
        self.assertEqual(evaluations[0]["streak_after"], 1)
        self.assertEqual(evaluations[0]["left_frame_index"], 593)
        self.assertFalse(evaluations[1]["accepted"])
        self.assertEqual(evaluations[1]["reason"], "distance")
        self.assertEqual(evaluations[1]["streak_before"], 1)
        self.assertEqual(evaluations[1]["streak_after"], 0)
        self.assertEqual(evaluations[1]["left_frame_index"], 594)
        self.assertFalse(evaluations[1]["console"])

    def test_distance_failure_uses_grace_then_recovers_into_provisional_group(self):
        memory = FakeMemory()
        coordinator = CrossCameraProvisionalCoordinator(
            memory,
            max_distance_cm=50.0,
            max_skew_seconds=0.35,
            required_pair_frames=3,
            location_confirm_frames=12,
            hold_grace_frames=5,
            hold_max_frames=12,
        )

        coordinator.update(camera_pair(captured_at=80.1))
        self.assertEqual(len(memory.applied_holds), 1)

        for update in range(2, 7):
            coordinator.update(
                camera_pair(
                    left_point=(0.0, 0.0),
                    right_point=(61.0, 0.0),
                    captured_at=80.0 + update * 0.1,
                )
            )
        self.assertEqual(memory.released_holds, [])

        for update in range(7, 10):
            observations = camera_pair(captured_at=80.0 + update * 0.1)
            coordinator.update(observations)

        self.assertEqual(len(memory.created_pairs), 1)
        self.assertEqual(
            memory.released_holds[-1][1],
            "provisional_group_established",
        )

    def test_hold_releases_after_grace_expires_without_recovery(self):
        memory = FakeMemory()
        coordinator = CrossCameraProvisionalCoordinator(
            memory,
            max_distance_cm=50.0,
            max_skew_seconds=0.35,
            required_pair_frames=3,
            location_confirm_frames=12,
            hold_grace_frames=5,
            hold_max_frames=12,
        )

        coordinator.update(camera_pair(captured_at=90.1))
        for update in range(2, 9):
            coordinator.update(
                camera_pair(
                    left_point=(0.0, 0.0),
                    right_point=(61.0, 0.0),
                    captured_at=90.0 + update * 0.1,
                )
            )

        self.assertEqual(memory.created_pairs, [])
        self.assertEqual(memory.released_holds[-1][1], "grace_expired")


if __name__ == "__main__":
    unittest.main()
