import unittest

import yaml

from constants import DEFAULT_TRACKER_CONFIG_PATH

from tests import EDGE_TRACKER_DIR

FALLBACK_TRACKER_CONFIG_PATH = "bytetrack_ghost_resistant.yaml"


def load_tracker_config(name):
    return yaml.safe_load((EDGE_TRACKER_DIR / name).read_text(encoding="utf-8"))


class TrackerConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_tracker_config(DEFAULT_TRACKER_CONFIG_PATH)

    def test_default_tracker_is_ocsort(self):
        self.assertEqual(self.config["tracker_type"], "ocsort")

    def test_detection_thresholds_carry_over_the_botsort_tuning(self):
        # The stock OC-SORT defaults are 0.25 for both thresholds, which spawns
        # the short-lived duplicate tracks the shadow suppression absorbs.
        self.assertEqual(self.config["track_high_thresh"], 0.50)
        self.assertEqual(self.config["track_low_thresh"], 0.10)
        self.assertEqual(self.config["new_track_thresh"], 0.70)

    def test_direction_consistency_is_configured_for_crossings(self):
        # delta_t and inertia are what OC-SORT adds over BoT-SORT: they stop a
        # track being handed to someone travelling the other way at a crossing.
        self.assertEqual(self.config["delta_t"], 3)
        self.assertGreater(self.config["inertia"], 0.0)

    def test_low_confidence_association_pass_stays_on(self):
        # A partially occluded person produces exactly these weak detections.
        self.assertTrue(self.config["use_byte"])

    def test_tracker_runs_no_appearance_model(self):
        # TransReID owns identity with a gallery behind it; a second appearance
        # opinion inside the tracker would only duplicate that work.
        self.assertFalse(self.config.get("with_reid", False))

    def test_fixed_cameras_need_no_global_motion_compensation(self):
        self.assertIsNone(self.config.get("gmc_method"))

    def test_botsort_fallback_config_is_still_loadable(self):
        # constants.py documents this file as a one-flag revert path, so it has
        # to keep parsing even though nothing loads it by default.
        fallback = load_tracker_config(FALLBACK_TRACKER_CONFIG_PATH)
        self.assertEqual(fallback["tracker_type"], "botsort")


if __name__ == "__main__":
    unittest.main()
