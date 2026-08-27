"""Folding a group into an existing master asks the whole gallery.

This is the largest commitment the system makes: it hands one person's
photographs to another identity, permanently, and the owner cannot get them
back.  It turned on a single closest-slot comparison, and one man merged into
another's identity on 0.28993 against a 0.30 bar.  That was not one unlucky
frame either -- the borderline guard had already made him win three separate
intake batches, all at 0.26-0.29.

Measured across the target's whole gallery the same pair sat at 0.378, while
every genuinely different pair in that session ran 0.53-0.79 and the one real
duplicate ran 0.279.  Nothing was missing but the question.
"""

import unittest

import numpy as np

from reid_memory import AppearanceIdentityMemory


def slot(feature):
    f = np.asarray(feature, dtype=np.float32).reshape(-1)
    f = f / np.linalg.norm(f)
    return {
        "feature": f,
        "feature_source": "transreid",
        "feature_space_id": "test-feature-space",
        "feature_dimension": int(f.size),
        "image_path": None,
        "digest": None,
        "captured_frame": 1,
        "captured_at": 1.0,
        "camera_id": "cam_1",
        "sharpness": 200.0,
        "detection_confidence": 0.95,
    }


def spread(base, offsets):
    """A gallery of one person: alike, but not identical."""
    return [slot((1.0, o, 0.0)) if base == "x" else slot((o, 1.0, 0.0)) for o in offsets]


class MergeAgreementTests(unittest.TestCase):
    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        self.target = self.memory._new_record(identity_state="confirmed")
        self.memory.identities[2] = self.target
        self.group = self.memory._new_record(identity_state="provisional")
        self.memory.identities[-7] = self.group

    def tearDown(self):
        self.memory.close(drain=False)

    def rejected(self, query):
        with self.memory._lock:
            return self.memory._merge_agreement_rejected_locked(
                -7,
                2,
                self.memory._normalize_feature(np.asarray(query, dtype=np.float32)),
                "test-feature-space",
                "right_side",
                0.29,
            )

    def test_one_close_slot_no_longer_carries_the_merge(self):
        # The shape of the real failure: the group matches one slot well and
        # the rest of the gallery not at all.
        for name, s in zip(("baseline", "front", "back", "left_side"),
                           spread("x", (0.0, 0.1, 0.15, 0.2))):
            self.target["gallery"][name] = s
        self.target["gallery"]["right_side"] = slot((0.0, 1.0, 0.0))
        self.assertTrue(self.rejected((0.02, 1.0, 0.0)))

    def test_a_group_that_agrees_broadly_still_merges(self):
        for name, s in zip(("baseline", "front", "back", "left_side", "right_side"),
                           spread("x", (0.0, 0.1, 0.15, 0.2, 0.25))):
            self.target["gallery"][name] = s
        self.group["camera_baselines"] = {"cam_1": slot((1.0, 0.12, 0.0))}
        self.assertFalse(self.rejected((1.0, 0.08, 0.0)))

    def test_the_groups_own_crops_are_weighed_not_just_the_query(self):
        for name, s in zip(("baseline", "front"), spread("x", (0.0, 0.1))):
            self.target["gallery"][name] = s
        # The query alone would agree; the group's stored crops do not.
        self.group["camera_views"] = {
            "cam_2": {"front": slot((0.0, 1.0, 0.0)), "back": slot((0.0, 1.0, 0.1))}
        }
        self.assertTrue(self.rejected((1.0, 0.05, 0.0)))

    def test_nothing_comparable_leaves_the_original_guarantee_alone(self):
        # A feature-space change must not silently block every merge.
        s = slot((1.0, 0.0, 0.0))
        s["feature_space_id"] = "other-space"
        self.target["gallery"]["baseline"] = s
        self.assertFalse(self.rejected((1.0, 0.0, 0.0)))


class MeasuredSeparationTests(unittest.TestCase):
    """The threshold has to sit between the two labelled cases from the run."""

    def setUp(self):
        self.memory = AppearanceIdentityMemory(
            db_path=None,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )

    def tearDown(self):
        self.memory.close(drain=False)

    def test_the_limit_admits_a_real_duplicate_and_refuses_the_bad_merge(self):
        limit = self.memory.provisional_merge_distance
        self.assertGreater(limit, 0.279, "the one confirmed duplicate must still merge")
        self.assertLessEqual(limit, 0.378, "the one confirmed wrong merge must be refused")

    def test_the_limit_clears_one_persons_own_viewpoint_spread(self):
        # A clean identity's own photographs spanned a median of 0.343.  A
        # limit under that refuses to reunite anyone with a wide spread, which
        # manufactures the duplicate IDs this exists to prevent.
        self.assertGreaterEqual(self.memory.provisional_merge_distance, 0.343)


if __name__ == "__main__":
    unittest.main()
