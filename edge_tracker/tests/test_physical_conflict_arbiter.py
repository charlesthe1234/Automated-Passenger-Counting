"""Appearance arbitration for incompatible cross-camera master claims.

When both cameras temporarily bind the same master to people at impossible
locations, camera scheduling order must not decide which binding survives.
Both claims stay held until clean post-overlap crops give ReID a decisive
winner.
"""

import time
import unittest
from unittest import mock

import numpy as np

from constants import (
    DEFAULT_PHYSICAL_CONFLICT_REID_MARGIN,
    DEFAULT_POSITION_SPLIT_FRAMES,
)
from reid_memory import AppearanceIdentityMemory


MASTER_ID = 1
RIGHTFUL_TRACK = ("cam_1", 11)
WRONG_TRACK = ("cam_2", 22)


class MarkerExtractor:
    """Return one of two orthogonal features from a crop marker pixel."""

    def extract_many_aligned(self, crops):
        return [
            np.asarray(
                [1.0, 0.0] if int(crop[0, 0, 0]) > 127 else [0.0, 1.0],
                dtype=np.float32,
            )
            for crop in crops
        ]


def sharp_crop(matches_master):
    yy, xx = np.indices((80, 40))
    checker = ((xx // 2 + yy // 2) % 2 * 255).astype(np.uint8)
    crop = np.repeat(checker[:, :, None], 3, axis=2)
    crop[0, 0, 0] = 255 if matches_master else 0
    return crop


class SwapExtractor:
    """Encode Mik and Denn as two distinct deterministic appearances."""

    def extract_many_aligned(self, crops):
        return [
            np.asarray(
                [1.0, 0.0] if int(crop[0, 0, 0]) < 127 else [0.0, 1.0],
                dtype=np.float32,
            )
            for crop in crops
        ]


def person_crop(person):
    crop = sharp_crop(matches_master=(person == "denn"))
    crop[0, 0, 0] = 0 if person == "mik" else 255
    return crop


class PhysicalConflictArbiterTests(unittest.TestCase):
    def make_memory(self):
        memory = AppearanceIdentityMemory(
            reid_extractor=MarkerExtractor(),
            distance_threshold=0.30,
            cross_camera_fusion_distance_cm=50.0,
            physical_conflict_reid_frames=3,
            physical_conflict_reid_margin=0.05,
            blur_threshold=1.0,
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        record = memory._new_record()
        feature = np.asarray([1.0, 0.0], dtype=np.float32)
        feature_space_id = memory._feature_space_id("transreid", feature)
        record["gallery"]["baseline"] = {
            "feature": feature,
            "feature_source": "transreid",
            "feature_space_id": feature_space_id,
            "feature_dimension": 2,
        }
        record["member_track_keys"] = {RIGHTFUL_TRACK, WRONG_TRACK}
        memory.identities[MASTER_ID] = record
        memory.track_to_identity[RIGHTFUL_TRACK] = MASTER_ID
        memory.track_to_identity[WRONG_TRACK] = MASTER_ID
        memory.visible_track_keys_by_camera["cam_1"] = {RIGHTFUL_TRACK}
        memory.visible_track_keys_by_camera["cam_2"] = {WRONG_TRACK}
        memory.recent_master_observations[MASTER_ID] = {
            "cam_1": {
                "track_key": RIGHTFUL_TRACK,
                "map_point": (0.0, 0.0),
                "observed_at": 1.0,
            },
            "cam_2": {
                "track_key": WRONG_TRACK,
                "map_point": (200.0, 0.0),
                "observed_at": 1.0,
            },
        }
        self.addCleanup(memory.close, drain=False)
        return memory

    @staticmethod
    def assign(memory, key, frame_index, crop, map_point):
        return memory.assign(
            key[1],
            crop,
            frame_index,
            camera_id=key[0],
            detection_confidence=0.95,
            observed_at=1.0 + frame_index * 0.01,
            map_point=map_point,
            intake_body_complete=True,
        )

    def start_conflict(self, memory):
        """Drive the disagreement until arbitration opens, and say when.

        Arbitration waits for the cameras to disagree consistently.  One frame
        apart is indistinguishable from a foot point wobbling at a grazing
        angle, so it no longer opens a conflict on its own.  Callers continue
        from the returned frame index rather than assuming frame 1.
        """
        for frame_index in range(1, DEFAULT_POSITION_SPLIT_FRAMES + 1):
            result = self.assign(
                memory,
                WRONG_TRACK,
                frame_index,
                sharp_crop(False),
                (200.0, 0.0),
            )
            self.assertEqual(result[0], MASTER_ID)
        self.assertIn(MASTER_ID, memory.physical_conflicts)
        return DEFAULT_POSITION_SPLIT_FRAMES

    def test_single_position_disagreement_does_not_open_a_conflict(self):
        memory = self.make_memory()

        result = self.assign(memory, WRONG_TRACK, 1, sharp_crop(False), (200.0, 0.0))

        self.assertEqual(result[0], MASTER_ID)
        self.assertEqual(memory.physical_conflicts, {})

    def test_a_frame_back_in_range_clears_the_disagreement_streak(self):
        memory = self.make_memory()

        for frame_index in range(1, DEFAULT_POSITION_SPLIT_FRAMES):
            self.assign(memory, WRONG_TRACK, frame_index, sharp_crop(False), (200.0, 0.0))
        # Agreeing once resets the count, so the streak has to start over
        # rather than resuming where a minutes-old wobble left it.
        self.assign(
            memory,
            WRONG_TRACK,
            DEFAULT_POSITION_SPLIT_FRAMES,
            sharp_crop(False),
            (10.0, 0.0),
        )
        self.assign(
            memory,
            WRONG_TRACK,
            DEFAULT_POSITION_SPLIT_FRAMES + 1,
            sharp_crop(False),
            (200.0, 0.0),
        )

        self.assertEqual(memory.physical_conflicts, {})

    def test_first_physical_failure_holds_both_bindings(self):
        memory = self.make_memory()

        self.start_conflict(memory)

        self.assertEqual(memory.track_to_identity[RIGHTFUL_TRACK], MASTER_ID)
        self.assertEqual(memory.track_to_identity[WRONG_TRACK], MASTER_ID)


class ConnectedSwapRecoveryTests(unittest.TestCase):
    """A loser must wait for the other half of a two-way swap to resolve."""

    MIK_ID = 2
    DENN_ID = 3
    CAM1_MIK = ("cam_1", 38)
    CAM1_DENN = ("cam_1", 18)
    CAM2_DENN_WITH_MIK_ID = ("cam_2", 34)
    CAM2_MIK_WITH_DENN_ID = ("cam_2", 36)

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            reid_extractor=SwapExtractor(),
            intake_frames=5,
            intake_delay_seconds=0.0,
            distance_threshold=0.30,
            cross_camera_fusion_distance_cm=50.0,
            physical_conflict_reid_frames=3,
            physical_conflict_reid_margin=0.05,
            physical_conflict_recovery_grace_frames=15,
            physical_conflict_recovery_max_frames=450,
            blur_threshold=1.0,
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        self.addCleanup(self.memory.close, drain=False)
        self.memory.next_identity_id = 4
        feature_space_id = self.memory._feature_space_id(
            "transreid", np.asarray([1.0, 0.0], dtype=np.float32)
        )
        for identity_id, feature, members in (
            (
                self.MIK_ID,
                np.asarray([1.0, 0.0], dtype=np.float32),
                {self.CAM1_MIK, self.CAM2_DENN_WITH_MIK_ID},
            ),
            (
                self.DENN_ID,
                np.asarray([0.0, 1.0], dtype=np.float32),
                {self.CAM1_DENN, self.CAM2_MIK_WITH_DENN_ID},
            ),
        ):
            record = self.memory._new_record()
            record["gallery"]["baseline"] = {
                "feature": feature,
                "feature_source": "transreid",
                "feature_space_id": feature_space_id,
                "feature_dimension": 2,
            }
            record["member_track_keys"] = set(members)
            self.memory.identities[identity_id] = record
            for key in members:
                self.memory.track_to_identity[key] = identity_id

        self.memory.visible_track_keys_by_camera["cam_1"] = {
            self.CAM1_MIK,
            self.CAM1_DENN,
        }
        self.memory.visible_track_keys_by_camera["cam_2"] = {
            self.CAM2_DENN_WITH_MIK_ID,
            self.CAM2_MIK_WITH_DENN_ID,
        }
        self.memory.recent_master_observations = {
            self.MIK_ID: {
                "cam_1": {
                    "track_key": self.CAM1_MIK,
                    "map_point": (200.0, 0.0),
                    "observed_at": 1.0,
                },
                "cam_2": {
                    "track_key": self.CAM2_DENN_WITH_MIK_ID,
                    "map_point": (0.0, 0.0),
                    "observed_at": 1.0,
                },
            },
            self.DENN_ID: {
                "cam_1": {
                    "track_key": self.CAM1_DENN,
                    "map_point": (0.0, 0.0),
                    "observed_at": 1.0,
                },
                "cam_2": {
                    "track_key": self.CAM2_MIK_WITH_DENN_ID,
                    "map_point": (200.0, 0.0),
                    "observed_at": 1.0,
                },
            },
        }

    def assign(self, key, frame_index, person, point, excluded=()):
        return self.memory.assign(
            key[1],
            person_crop(person),
            frame_index,
            excluded_identity_ids=set(excluded),
            camera_id=key[0],
            detection_confidence=0.95,
            observed_at=1.0 + frame_index * 0.01,
            map_point=point,
            intake_body_complete=True,
        )

    def resolve_conflict(self, master_id, rightful_key, rightful_person, wrong_key, wrong_person, start_frame):
        # Arbitration only opens once the two cameras have disagreed for
        # several consecutive frames, so the wrong claim has to be repeated.
        for frame_index in range(start_frame, start_frame + DEFAULT_POSITION_SPLIT_FRAMES):
            self.assign(
                wrong_key,
                frame_index,
                wrong_person,
                (200.0, 0.0) if wrong_person == "mik" else (0.0, 0.0),
            )
        self.assertIn(master_id, self.memory.physical_conflicts)
        start_frame += DEFAULT_POSITION_SPLIT_FRAMES - 1
        for frame_index in range(start_frame + 1, start_frame + 4):
            self.assign(
                rightful_key,
                frame_index,
                rightful_person,
                (200.0, 0.0) if rightful_person == "mik" else (0.0, 0.0),
            )
            self.assign(
                wrong_key,
                frame_index,
                wrong_person,
                (200.0, 0.0) if wrong_person == "mik" else (0.0, 0.0),
            )
        self.assertTrue(self.memory.wait_for_idle())
        # Callers continue the scenario from here rather than from a fixed
        # frame, so the length of the disagreement streak stays in one place.
        return start_frame + 3

    def test_mik_waits_for_id2_conflict_then_recovers_without_id4(self):
        # Correct ID 3 first: cam2's Mik loses the ID swapped from Denn.
        resolved = self.resolve_conflict(
            self.DENN_ID,
            self.CAM1_DENN,
            "denn",
            self.CAM2_MIK_WITH_DENN_ID,
            "mik",
            1,
        )
        self.assertNotIn(self.CAM2_MIK_WITH_DENN_ID, self.memory.track_to_identity)
        self.assertIn(
            self.CAM2_MIK_WITH_DENN_ID,
            self.memory.physical_conflict_recovery_holds,
        )

        # The related ID 2 conflict begins while cam2 still wrongly gives it
        # to Denn. This must link to Mik's recovery hold.
        for frame_index in range(resolved + 1, resolved + 1 + DEFAULT_POSITION_SPLIT_FRAMES):
            self.assign(
                self.CAM2_DENN_WITH_MIK_ID,
                frame_index,
                "denn",
                (0.0, 0.0),
            )
        second = resolved + DEFAULT_POSITION_SPLIT_FRAMES
        self.assertIn(self.MIK_ID, self.memory.physical_conflicts)
        recovery = self.memory.physical_conflict_recovery_holds[
            self.CAM2_MIK_WITH_DENN_ID
        ]
        self.assertTrue(recovery["related_conflict_tokens"])

        # Mik finishes his five-crop analysis, but ID 2 is still occupied by
        # cam2/Denn. The old behaviour created Master 4 here.
        for frame_index in range(second + 1, second + 6):
            self.assign(
                self.CAM2_MIK_WITH_DENN_ID,
                frame_index,
                "mik",
                (200.0, 0.0),
                excluded={self.MIK_ID, self.DENN_ID},
            )
        self.assertTrue(self.memory.wait_for_idle())
        self.assertNotIn(self.CAM2_MIK_WITH_DENN_ID, self.memory.track_to_identity)
        self.assertNotIn(4, self.memory.identities)
        self.assertTrue(
            self.memory.pending_intake[self.CAM2_MIK_WITH_DENN_ID][
                "deferred_by_physical_conflict_hold"
            ]
        )

        # Once the ID 2 conflict removes Denn, the saved intake is retried and
        # Mik is allowed to reclaim ID 2 instead of creating ID 4.
        for frame_index in range(second + 6, second + 9):
            self.assign(self.CAM1_MIK, frame_index, "mik", (200.0, 0.0))
            self.assign(
                self.CAM2_DENN_WITH_MIK_ID,
                frame_index,
                "denn",
                (0.0, 0.0),
            )
        self.assertTrue(self.memory.wait_for_idle())
        self.assertNotIn(self.MIK_ID, self.memory.physical_conflicts)
        self.assertNotIn(
            self.CAM2_MIK_WITH_DENN_ID,
            self.memory.physical_conflict_recovery_holds,
        )

        self.assign(
            self.CAM2_MIK_WITH_DENN_ID,
            second + 9,
            "mik",
            (200.0, 0.0),
            excluded={self.DENN_ID},
        )
        self.assertTrue(self.memory.wait_for_idle())
        self.assertEqual(
            self.memory.lookup(
                self.CAM2_MIK_WITH_DENN_ID[1],
                camera_id=self.CAM2_MIK_WITH_DENN_ID[0],
            ),
            self.MIK_ID,
        )
        self.assertNotIn(4, self.memory.identities)

    def test_recovery_grace_expires_when_no_related_conflict_appears(self):
        resolved = self.resolve_conflict(
            self.DENN_ID,
            self.CAM1_DENN,
            "denn",
            self.CAM2_MIK_WITH_DENN_ID,
            "mik",
            1,
        )

        for frame_index in range(resolved + 2, resolved + 7):
            self.assign(
                self.CAM2_MIK_WITH_DENN_ID,
                frame_index,
                "mik",
                (200.0, 0.0),
                excluded={self.MIK_ID, self.DENN_ID},
            )
        self.assertTrue(self.memory.wait_for_idle())
        self.assertNotIn(4, self.memory.identities)

        self.assign(
            self.CAM2_MIK_WITH_DENN_ID,
            resolved + 15,
            "mik",
            (200.0, 0.0),
            excluded={self.MIK_ID, self.DENN_ID},
        )
        self.assertTrue(self.memory.wait_for_idle())
        self.assertEqual(
            self.memory.lookup(
                self.CAM2_MIK_WITH_DENN_ID[1],
                camera_id=self.CAM2_MIK_WITH_DENN_ID[0],
            ),
            4,
        )


class ContestedIdentityClaimTests(unittest.TestCase):
    """A strong newcomer may challenge a physically conflicting wrong owner."""

    CHALLENGER = ("cam_1", 20)
    INCUMBENT = ("cam_2", 19)

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            reid_extractor=MarkerExtractor(),
            intake_frames=5,
            intake_delay_seconds=0.0,
            distance_threshold=0.30,
            cross_camera_fusion_distance_cm=50.0,
            physical_conflict_reid_frames=3,
            physical_conflict_reid_margin=0.05,
            blur_threshold=1.0,
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        self.addCleanup(self.memory.close, drain=False)
        feature = np.asarray([1.0, 0.0], dtype=np.float32)
        feature_space_id = self.memory._feature_space_id("transreid", feature)
        record = self.memory._new_record()
        record["gallery"]["baseline"] = {
            "feature": feature,
            "feature_source": "transreid",
            "feature_space_id": feature_space_id,
            "feature_dimension": 2,
        }
        record["member_track_keys"] = {self.INCUMBENT}
        self.memory.identities[MASTER_ID] = record
        self.memory.next_identity_id = 2
        self.memory.track_to_identity[self.INCUMBENT] = MASTER_ID
        self.memory.visible_track_keys_by_camera["cam_1"] = {self.CHALLENGER}
        self.memory.visible_track_keys_by_camera["cam_2"] = {self.INCUMBENT}

    def assign(self, key, frame_index, matches_master, point):
        return self.memory.assign(
            key[1],
            sharp_crop(matches_master),
            frame_index,
            camera_id=key[0],
            detection_confidence=0.95,
            observed_at=1.0 + frame_index * 0.01,
            map_point=point,
            intake_body_complete=True,
        )

    def start_challenge(self, challenger_matches=True):
        for frame_index in range(1, 6):
            # Refresh the wrong owner's current position before each newcomer
            # crop, reproducing the cross-camera state at the ID-3 incident.
            self.assign(self.INCUMBENT, frame_index, False, (0.0, 0.0))
            self.assign(
                self.CHALLENGER,
                frame_index,
                challenger_matches,
                (300.0, 0.0),
            )
        self.assertTrue(self.memory.wait_for_idle())

    def test_stronger_challenger_recovers_master_without_creating_duplicate(self):
        self.start_challenge()

        conflict = self.memory.physical_conflicts[MASTER_ID]
        self.assertEqual(conflict["challenger_key"], self.CHALLENGER)
        self.assertNotIn(2, self.memory.identities)
        self.assertIsNone(
            self.memory.lookup(
                self.CHALLENGER[1],
                camera_id=self.CHALLENGER[0],
            )
        )

        for frame_index in range(6, 9):
            self.assign(self.INCUMBENT, frame_index, False, (0.0, 0.0))
        self.assertTrue(self.memory.wait_for_idle())

        self.assertNotIn(MASTER_ID, self.memory.physical_conflicts)
        self.assertEqual(
            self.memory.lookup(
                self.CHALLENGER[1],
                camera_id=self.CHALLENGER[0],
            ),
            MASTER_ID,
        )
        self.assertNotIn(self.INCUMBENT, self.memory.track_to_identity)
        self.assertIn(
            MASTER_ID,
            self.memory.physical_conflict_rejections[self.INCUMBENT],
        )
        self.assertNotIn(2, self.memory.identities)
        self.assertEqual(
            self.memory.track_binding_metadata[self.CHALLENGER][
                "conflict_resolution"
            ],
            "appearance",
        )

    def install_location_hold(self, age_seconds):
        """A hold whose second candidate is a track that never appears.

        It can therefore never reach a verdict, which is the whole point: the
        real ID-4 incident stalled on a candidate refused for blur on every
        frame it was ever offered.
        """

        self.memory.physical_conflicts[MASTER_ID] = {
            "token": 99,
            "candidates": {self.INCUMBENT: [], ("cam_2", 98): []},
            "last_frames": {},
            "submitted": False,
            "attempts": 0,
            "started_frame": 0,
            "started_monotonic": time.monotonic() - age_seconds,
        }

    def test_stalled_location_hold_gives_way_to_a_strong_challenger(self):
        self.install_location_hold(
            self.memory.physical_conflict_stall_seconds + 30.0,
        )

        self.start_challenge()

        conflict = self.memory.physical_conflicts[MASTER_ID]
        self.assertEqual(
            conflict.get("challenger_key"),
            self.CHALLENGER,
            "the challenger must be able to take over a hold that cannot finish",
        )
        self.assertNotEqual(conflict["token"], 99, "the stalled hold must be replaced")
        self.assertNotIn(2, self.memory.identities, "and no duplicate is minted")

    def test_young_location_hold_still_blocks_a_challenger(self):
        self.install_location_hold(0.0)

        self.start_challenge()

        conflict = self.memory.physical_conflicts[MASTER_ID]
        self.assertEqual(
            conflict["token"],
            99,
            "a hold that may yet resolve keeps its turn",
        )
        self.assertIsNone(conflict.get("challenger_key"))

    def test_challenger_holding_a_group_placeholder_can_still_win(self):
        self.start_challenge()

        # The pairing path raises its contests from tracks that already belong
        # to a temporary group, so by the time a verdict is computed the
        # challenger holds that group's negative placeholder rather than
        # nothing at all.  Requiring "holds nothing" here discarded every such
        # contest as stale.
        group_id = -1
        group = self.memory._new_record(identity_state="provisional")
        group["member_track_keys"] = {self.CHALLENGER}
        self.memory.identities[group_id] = group
        self.memory.track_to_identity[self.CHALLENGER] = group_id

        for frame_index in range(6, 9):
            self.assign(self.INCUMBENT, frame_index, False, (0.0, 0.0))
        self.assertTrue(self.memory.wait_for_idle())

        self.assertNotIn(
            MASTER_ID,
            self.memory.physical_conflicts,
            "the contest must reach a verdict rather than be dropped as stale",
        )
        self.assertEqual(
            self.memory.track_to_identity.get(self.CHALLENGER),
            MASTER_ID,
            "the challenger takes back the master it matched",
        )
        self.assertNotIn(
            self.CHALLENGER,
            group["member_track_keys"],
            "and stops being listed as a member of the group it left",
        )
        self.assertNotIn(2, self.memory.identities, "no duplicate is minted")

    def test_weak_newcomer_does_not_challenge_current_owner(self):
        self.start_challenge(challenger_matches=False)

        self.assertNotIn(MASTER_ID, self.memory.physical_conflicts)
        self.assertEqual(self.memory.track_to_identity[self.INCUMBENT], MASTER_ID)
        self.assertEqual(
            self.memory.lookup(
                self.CHALLENGER[1],
                camera_id=self.CHALLENGER[0],
            ),
            2,
        )

    def test_equally_strong_owner_is_not_displaced(self):
        self.start_challenge()

        for frame_index in range(6, 9):
            self.assign(self.INCUMBENT, frame_index, True, (0.0, 0.0))
        self.assertTrue(self.memory.wait_for_idle())

        self.assertNotIn(MASTER_ID, self.memory.physical_conflicts)
        self.assertEqual(self.memory.track_to_identity[self.INCUMBENT], MASTER_ID)
        self.assertNotIn(self.CHALLENGER, self.memory.track_to_identity)
        self.assign(self.CHALLENGER, 9, True, (300.0, 0.0))
        self.assertTrue(self.memory.wait_for_idle())
        self.assertEqual(
            self.memory.lookup(
                self.CHALLENGER[1],
                camera_id=self.CHALLENGER[0],
            ),
            2,
        )


class PhysicalConflictArbiterAdditionalTests(unittest.TestCase):
    make_memory = PhysicalConflictArbiterTests.make_memory
    assign = staticmethod(PhysicalConflictArbiterTests.assign)
    start_conflict = PhysicalConflictArbiterTests.start_conflict

    def test_missing_clean_crop_waits_without_revoking_either_claim(self):
        memory = self.make_memory()
        started = self.start_conflict(memory)

        for frame_index in range(started + 1, started + 7):
            self.assign(memory, RIGHTFUL_TRACK, frame_index, None, (0.0, 0.0))
            self.assign(
                memory,
                WRONG_TRACK,
                frame_index,
                sharp_crop(False),
                (200.0, 0.0),
            )

        self.assertIn(MASTER_ID, memory.physical_conflicts)
        self.assertEqual(memory.track_to_identity[RIGHTFUL_TRACK], MASTER_ID)
        self.assertEqual(memory.track_to_identity[WRONG_TRACK], MASTER_ID)

    def test_clear_appearance_winner_is_independent_of_camera_call_order(self):
        for order in ((RIGHTFUL_TRACK, WRONG_TRACK), (WRONG_TRACK, RIGHTFUL_TRACK)):
            with self.subTest(order=order):
                memory = self.make_memory()
                started = self.start_conflict(memory)
                for frame_index in range(started + 1, started + 4):
                    for key in order:
                        self.assign(
                            memory,
                            key,
                            frame_index,
                            sharp_crop(key == RIGHTFUL_TRACK),
                            (0.0, 0.0) if key == RIGHTFUL_TRACK else (200.0, 0.0),
                        )

                self.assertNotIn(MASTER_ID, memory.physical_conflicts)
                self.assertEqual(memory.track_to_identity[RIGHTFUL_TRACK], MASTER_ID)
                self.assertNotIn(WRONG_TRACK, memory.track_to_identity)
                self.assertIn(
                    MASTER_ID,
                    memory.physical_conflict_rejections[WRONG_TRACK],
                )
                self.assertEqual(
                    memory.track_binding_metadata[RIGHTFUL_TRACK]["conflict_resolution"],
                    "appearance",
                )

    def test_tied_appearance_is_inconclusive_and_revokes_neither(self):
        memory = self.make_memory()
        started = self.start_conflict(memory)
        conflict = memory.physical_conflicts[MASTER_ID]
        conflict["candidates"] = {key: [] for key in conflict["candidates"]}
        conflict["last_frames"] = {}

        for frame_index in range(started + 1, started + 4):
            self.assign(memory, RIGHTFUL_TRACK, frame_index, sharp_crop(True), (0.0, 0.0))
            self.assign(memory, WRONG_TRACK, frame_index, sharp_crop(True), (200.0, 0.0))

        conflict = memory.physical_conflicts[MASTER_ID]
        self.assertEqual(conflict["attempts"], 1)
        self.assertTrue(all(not samples for samples in conflict["candidates"].values()))
        self.assertEqual(memory.track_to_identity[RIGHTFUL_TRACK], MASTER_ID)
        self.assertEqual(memory.track_to_identity[WRONG_TRACK], MASTER_ID)

    def test_location_recovery_cancels_hold_without_revocation(self):
        memory = self.make_memory()
        started = self.start_conflict(memory)

        self.assign(memory, WRONG_TRACK, started + 1, sharp_crop(False), (10.0, 0.0))

        self.assertNotIn(MASTER_ID, memory.physical_conflicts)
        self.assertEqual(memory.track_to_identity[RIGHTFUL_TRACK], MASTER_ID)
        self.assertEqual(memory.track_to_identity[WRONG_TRACK], MASTER_ID)


class ContestSurvivesLocationRecoveryTests(unittest.TestCase):
    """Geometry agreeing does not answer a claim raised on appearance.

    The swap that makes a contest necessary usually drags *every* one of the
    master's owner tracks onto one body.  The master then stops being in two
    places and reads as perfectly healthy, which is precisely when its real
    owner is outside contesting it.  Cancelling on that reading killed 50
    consecutive claims in one huddle, each roughly 29ms old -- none of them
    alive for the three frames needed to collect the incumbent crops that
    decide who keeps the ID.
    """

    make_memory = PhysicalConflictArbiterTests.make_memory

    def arrange(self, challenger_key):
        memory = self.make_memory()
        # Both owner tracks now report the same spot: the swap has already
        # happened and the impostor is wearing the master on both cameras.
        memory.recent_master_observations[MASTER_ID]["cam_2"]["map_point"] = (0.0, 0.0)
        state = {
            "token": 7,
            "candidates": {RIGHTFUL_TRACK: [], WRONG_TRACK: []},
            "last_frames": {},
            "submitted": False,
            "attempts": 0,
            "started_frame": 1,
            "started_monotonic": time.monotonic(),
        }
        if challenger_key is not None:
            state["challenger_key"] = challenger_key
        memory.physical_conflicts[MASTER_ID] = state
        return memory

    def recheck_position(self, memory):
        with memory._lock:
            return memory._physical_match_allowed_locked(
                MASTER_ID,
                "cam_1",
                (0.0, 0.0),
                1.0,
                track_key=RIGHTFUL_TRACK,
            )

    def test_challenger_contest_outlives_agreeing_positions(self):
        memory = self.arrange(challenger_key=RIGHTFUL_TRACK)

        self.assertTrue(self.recheck_position(memory))

        self.assertIn(
            MASTER_ID,
            memory.physical_conflicts,
            "an appearance contest must not be cancelled by recovered geometry",
        )
        self.assertEqual(
            memory.physical_conflicts[MASTER_ID]["token"],
            7,
            "the contest must survive intact, not restart under a fresh token",
        )

    def test_location_born_hold_still_cancels_on_agreeing_positions(self):
        memory = self.arrange(challenger_key=None)

        self.assertTrue(self.recheck_position(memory))

        self.assertNotIn(
            MASTER_ID,
            memory.physical_conflicts,
            "a hold opened by geometry is still answered by geometry",
        )


class GradedExtractor:
    """Turn a crop's marker byte into a feature at a chosen cosine distance.

    The two-feature MarkerExtractor can only express "identical" or "opposite",
    which cannot reproduce the case that matters here: a loser that is still a
    good match for the master it just lost.
    """

    @staticmethod
    def crop(distance):
        # Textured, so it clears the blur gate; the distance rides in one pixel.
        yy, xx = np.indices((80, 40))
        checker = ((xx // 2 + yy // 2) % 2 * 255).astype(np.uint8)
        crop = np.repeat(checker[:, :, None], 3, axis=2)
        crop[0, 0, 0] = int(round(distance * 255))
        return crop

    def extract_many_aligned(self, crops):
        features = []
        for crop in crops:
            cosine = 1.0 - int(crop[0, 0, 0]) / 255.0
            features.append(
                np.asarray(
                    [cosine, float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))],
                    dtype=np.float32,
                )
            )
        return features


class LoserBarScopeTests(unittest.TestCase):
    """Coming second in a geometry arbitration is not proof of being someone else.

    The bar is permanent and shuts both doors -- the master can no longer be
    matched, and can no longer be contested.  That is right for a challenger
    that attacked and lost, which would otherwise attack forever.  Applied to a
    track that never contested anything it barred two men from their own IDs at
    0.181 and 0.229, both inside the 0.30 match threshold.
    """

    def make_memory(self):
        memory = AppearanceIdentityMemory(
            reid_extractor=GradedExtractor(),
            distance_threshold=0.30,
            cross_camera_fusion_distance_cm=50.0,
            physical_conflict_reid_frames=3,
            physical_conflict_reid_margin=0.15,
            blur_threshold=1.0,
            db_path=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=True,
        )
        self.addCleanup(memory.close, drain=False)
        record = memory._new_record()
        feature = np.asarray([1.0, 0.0], dtype=np.float32)
        record["gallery"]["baseline"] = {
            "feature": feature,
            "feature_source": "transreid",
            "feature_space_id": memory._feature_space_id("transreid", feature),
            "feature_dimension": 2,
        }
        record["member_track_keys"] = {RIGHTFUL_TRACK, WRONG_TRACK}
        memory.identities[MASTER_ID] = record
        memory.track_to_identity[RIGHTFUL_TRACK] = MASTER_ID
        memory.track_to_identity[WRONG_TRACK] = MASTER_ID
        memory.visible_track_keys_by_camera["cam_1"] = {RIGHTFUL_TRACK}
        memory.visible_track_keys_by_camera["cam_2"] = {WRONG_TRACK}
        memory.recent_master_observations[MASTER_ID] = {
            "cam_1": {
                "track_key": RIGHTFUL_TRACK,
                "map_point": (0.0, 0.0),
                "observed_at": 1.0,
            },
            "cam_2": {
                "track_key": WRONG_TRACK,
                "map_point": (200.0, 0.0),
                "observed_at": 1.0,
            },
        }
        return memory

    assign = staticmethod(PhysicalConflictArbiterTests.assign)

    def arbitrate(self, winner_distance, loser_distance):
        """Drive a real location arbitration to a verdict and report the bar."""
        memory = self.make_memory()
        for frame_index in range(1, DEFAULT_POSITION_SPLIT_FRAMES + 4):
            self.assign(
                memory,
                WRONG_TRACK,
                frame_index,
                GradedExtractor.crop(loser_distance),
                (200.0, 0.0),
            )
            self.assign(
                memory,
                RIGHTFUL_TRACK,
                frame_index,
                GradedExtractor.crop(winner_distance),
                (0.0, 0.0),
            )
        self.assertTrue(memory.wait_for_idle())
        self.assertNotIn(
            MASTER_ID,
            memory.physical_conflicts,
            "the arbitration must have reached a verdict for this to mean anything",
        )
        return MASTER_ID in memory.physical_conflict_rejections.get(WRONG_TRACK, ())

    def test_close_loser_of_a_location_hold_keeps_its_claim(self):
        # Decisive (margin 0.23) but the loser is still inside the 0.30 match
        # threshold, which is the shape that cost two men their own IDs.
        self.assertFalse(
            self.arbitrate(winner_distance=0.02, loser_distance=0.25),
            "a loser that still matches has not been shown to be someone else",
        )

    def test_loser_that_no_longer_matches_is_barred(self):
        self.assertTrue(self.arbitrate(winner_distance=0.02, loser_distance=0.62))


class ArbitrationMarginTests(unittest.TestCase):
    """The margin has to separate a verdict from a coin toss."""

    def test_default_margin_clears_every_verdict_that_stole_an_identity(self):
        # Both recorded sessions: the verdicts that evicted the real owner
        # separated by these margins, and must now read as inconclusive.
        for observed in (0.068, 0.104):
            self.assertGreater(DEFAULT_PHYSICAL_CONFLICT_REID_MARGIN, observed)

    def test_default_margin_still_admits_every_verdict_that_was_right(self):
        for observed in (0.341, 0.406, 0.509, 0.516):
            self.assertLess(DEFAULT_PHYSICAL_CONFLICT_REID_MARGIN, observed)


class PromotionWaitsForContestTests(unittest.TestCase):
    """A group must not number itself while its own member is contesting.

    Promotion counts how long two boxes have agreed about where they are; a
    contest compares the person against the master itself.  Left to race, the
    weaker evidence won and issued a duplicate for someone who already had an
    ID.
    """

    make_memory = PhysicalConflictArbiterTests.make_memory
    GROUP_ID = -1

    def arrange(self):
        memory = self.make_memory()
        group = memory._new_record(identity_state="provisional")
        group["member_track_keys"] = {RIGHTFUL_TRACK}
        memory.identities[self.GROUP_ID] = group
        memory.physical_conflicts[MASTER_ID] = {
            "token": 4,
            "candidates": {RIGHTFUL_TRACK: [], WRONG_TRACK: []},
            "last_frames": {},
            "submitted": False,
            "attempts": 0,
            "started_frame": 1,
            "started_monotonic": time.monotonic(),
            "challenger_key": RIGHTFUL_TRACK,
        }
        return memory

    def contest(self, memory):
        with memory._lock:
            return memory._member_contest_in_flight_locked(self.GROUP_ID)

    def test_live_contest_by_a_member_holds_the_promotion(self):
        memory = self.arrange()

        self.assertIsNotNone(self.contest(memory))
        events = []
        with mock.patch(
            "reid_memory.identity_event",
            side_effect=lambda name, **fields: events.append((name, fields)),
        ):
            with memory._lock:
                promoted = memory._promote_provisional_locked(
                    self.GROUP_ID,
                    "stable_location",
                )

        self.assertIsNone(promoted, "the group must wait for the verdict")
        # Promotion refuses for several reasons; this asserts it refused for
        # *this* one, and refused before spending work on the others.
        self.assertEqual(
            [(name, fields["reason"]) for name, fields in events],
            [("provisional_promotion_deferred", "member_contesting_master")],
        )
        self.assertEqual(events[0][1]["contested_master_id"], MASTER_ID)
        self.assertEqual(events[0][1]["contest_token"], 4)

    def test_contest_past_patience_no_longer_holds_the_promotion(self):
        memory = self.arrange()
        memory.physical_conflicts[MASTER_ID]["started_monotonic"] = time.monotonic() - (
            memory.identity_audit_contest_patience_seconds + 1.0
        )

        self.assertIsNone(
            self.contest(memory),
            "a contest that never concludes delays a number, never withholds it",
        )

    def test_contest_raised_by_an_outsider_does_not_hold_the_promotion(self):
        memory = self.arrange()
        memory.physical_conflicts[MASTER_ID]["challenger_key"] = ("cam_2", 77)

        self.assertIsNone(
            self.contest(memory),
            "only a contest this group's own track raised is its business",
        )

    def test_location_hold_does_not_hold_the_promotion(self):
        memory = self.arrange()
        memory.physical_conflicts[MASTER_ID].pop("challenger_key")

        self.assertIsNone(
            self.contest(memory),
            "a geometry hold decides nothing about who this group is",
        )


class ArbiterBlurRelaxationTests(unittest.TestCase):
    """A camera too soft for the gallery's bar must not deadlock the arbiter.

    Arbitration stores nothing and only ranks two candidates, and its winner
    still has to clear the distance margin.  Refusing its crops on the gallery's
    terms let one contest sit at 3/3 against 0/3 for four seconds while the
    master it locked was denied to the man matching it at 0.047.
    """

    make_memory = PhysicalConflictArbiterTests.make_memory

    def hold_aged(self, memory, age_seconds):
        memory.physical_conflicts[MASTER_ID] = {
            "token": 1,
            "candidates": {RIGHTFUL_TRACK: [], WRONG_TRACK: []},
            "last_frames": {},
            "submitted": False,
            "attempts": 0,
            "started_frame": 1,
            "started_monotonic": time.monotonic() - age_seconds,
        }

    def offer_blurry_crop(self, memory, frame_index):
        # A flat field has no edges at all, so it scores zero sharpness and is
        # refused by any threshold above zero.
        crop = np.zeros((80, 40, 3), dtype=np.uint8)
        with memory._lock:
            memory._collect_physical_conflict_sample_locked(
                MASTER_ID,
                RIGHTFUL_TRACK,
                crop,
                frame_index,
                0.95,
                1.0,
                (0.0, 0.0),
                True,
            )
        return memory.physical_conflicts[MASTER_ID]["candidates"][RIGHTFUL_TRACK]

    def test_blurry_crop_is_refused_while_the_contest_is_young(self):
        memory = self.make_memory()
        self.hold_aged(memory, 0.0)

        self.assertEqual(
            self.offer_blurry_crop(memory, 10),
            [],
            "a sharp crop may still arrive; the gate holds at first",
        )

    def test_blurry_crop_is_accepted_once_the_gate_times_out(self):
        memory = self.make_memory()
        self.hold_aged(memory, memory.physical_conflict_blur_timeout_seconds + 0.5)

        samples = self.offer_blurry_crop(memory, 10)

        self.assertEqual(
            len(samples),
            1,
            "a starved contest must take what it can get rather than deadlock",
        )

    def test_relaxation_never_precedes_the_stall_write_off(self):
        memory = self.make_memory()

        self.assertLessEqual(
            memory.physical_conflict_blur_timeout_seconds,
            memory.physical_conflict_stall_seconds,
            "a hold must get its relaxed crops before it is written off",
        )


class StalledLocationHoldTests(unittest.TestCase):
    """A hold that can no longer conclude must not keep the master to itself."""

    make_memory = PhysicalConflictArbiterTests.make_memory

    def hold(self, memory, age_seconds, starved=True, challenger_key=None):
        full = [{"crop": None}] * memory.physical_conflict_reid_frames
        state = {
            "token": 1,
            "candidates": {
                RIGHTFUL_TRACK: list(full),
                WRONG_TRACK: [] if starved else list(full),
            },
            "last_frames": {},
            "submitted": False,
            "attempts": 0,
            "started_frame": 1,
            "started_monotonic": time.monotonic() - age_seconds,
        }
        if challenger_key is not None:
            state["challenger_key"] = challenger_key
        return state

    def is_stalled(self, memory, **kwargs):
        with memory._lock:
            return memory._location_hold_is_stalled_locked(self.hold(memory, **kwargs))

    def test_starved_hold_past_the_stall_window_stands_aside(self):
        memory = self.make_memory()

        self.assertTrue(
            self.is_stalled(
                memory,
                age_seconds=memory.physical_conflict_stall_seconds + 1.0,
            )
        )

    def test_young_hold_keeps_the_master(self):
        memory = self.make_memory()

        self.assertFalse(
            self.is_stalled(memory, age_seconds=0.0),
            "a hold that may still resolve on its own is not stalled",
        )

    def test_hold_with_every_candidate_supplied_is_not_stalled(self):
        memory = self.make_memory()

        self.assertFalse(
            self.is_stalled(
                memory,
                age_seconds=memory.physical_conflict_stall_seconds + 1.0,
                starved=False,
            ),
            "a hold holding full evidence is waiting on the worker, not starved",
        )

    def test_appearance_contest_is_never_treated_as_stalled(self):
        memory = self.make_memory()

        self.assertFalse(
            self.is_stalled(
                memory,
                age_seconds=memory.physical_conflict_stall_seconds + 1.0,
                challenger_key=RIGHTFUL_TRACK,
            ),
            "restarting a contest would throw away the evidence it has",
        )


if __name__ == "__main__":
    unittest.main()
