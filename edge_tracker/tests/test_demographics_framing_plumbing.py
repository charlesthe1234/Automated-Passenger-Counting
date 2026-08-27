"""Regression tests for carrying MiVOLO's framing from the frame to the worker.

The demographics worker runs long after the frame a crop came from has gone, so
the tight detection box, the face box and the neighbouring people all have to
be measured at intake and travel with the crop.  These cover that journey.
"""

import unittest

import numpy as np

from constants import (
    DEFAULT_REID_CROP_BOTTOM_PADDING,
    DEFAULT_REID_CROP_SIDE_PADDING,
)
from reid_crop_quality import (
    detection_bounds_inside_reid_crop,
    occluder_bounds_inside_reid_crop,
)
from reid_crops import crop_person
from reid_memory import AppearanceIdentityMemory
from tests.test_reid_intake_lifecycle import (
    CountingBatchExtractor,
    FakeResult,
    sharp_frame,
)
from pose_engine import get_standing_points


def frame(height=480, width=640):
    yy, xx = np.indices((height, width))
    checker = ((xx // 4 + yy // 4) % 2 * 255).astype(np.uint8)
    return np.repeat(checker[:, :, None], 3, axis=2)


class DetectionBoundsTests(unittest.TestCase):
    def test_the_stored_bounds_recover_the_detection_box_from_the_crop(self):
        image = frame()
        box = (200.0, 100.0, 280.0, 340.0)
        bounds = detection_bounds_inside_reid_crop(image, box)
        crop = crop_person(image, box)
        crop_height, crop_width = crop.shape[:2]

        x1 = round(bounds[0] * crop_width)
        y1 = round(bounds[1] * crop_height)
        x2 = round(bounds[2] * crop_width)
        y2 = round(bounds[3] * crop_height)
        self.assertEqual(x2 - x1, int(box[2] - box[0]))
        self.assertEqual(y2 - y1, int(box[3] - box[1]))

    def test_the_bottom_padding_is_the_largest_part_cut_back_off(self):
        image = frame()
        box = (200.0, 100.0, 280.0, 340.0)
        bounds = detection_bounds_inside_reid_crop(image, box)
        top_trim = bounds[1]
        bottom_trim = 1.0 - bounds[3]
        self.assertGreater(bottom_trim, top_trim)
        self.assertGreater(DEFAULT_REID_CROP_BOTTOM_PADDING, DEFAULT_REID_CROP_SIDE_PADDING)

    def test_a_crop_clamped_by_the_frame_edge_still_maps_back(self):
        # Against the edge there is no padding to remove on that side, so the
        # detection box reaches the crop boundary.
        image = frame()
        bounds = detection_bounds_inside_reid_crop(image, (0.0, 0.0, 80.0, 240.0))
        self.assertEqual(bounds[0], 0.0)
        self.assertEqual(bounds[1], 0.0)
        self.assertLess(bounds[3], 1.0)

    def test_a_degenerate_box_has_no_bounds(self):
        self.assertIsNone(detection_bounds_inside_reid_crop(frame(), (50.0, 50.0, 50.0, 50.0)))


class OccluderBoundsTests(unittest.TestCase):
    def test_a_neighbour_overlapping_the_crop_is_reported(self):
        image = frame()
        boxes = [(200.0, 100.0, 280.0, 340.0), (260.0, 150.0, 320.0, 350.0)]
        occluders = occluder_bounds_inside_reid_crop(image, boxes[0], boxes, 0)
        self.assertEqual(len(occluders), 1)
        left, top, right, bottom = occluders[0]
        self.assertGreater(right, left)
        self.assertGreater(bottom, top)
        # The neighbour stands to the right, so it occupies the right edge.
        self.assertEqual(right, 1.0)
        self.assertGreater(left, 0.5)

    def test_a_neighbour_clear_of_the_crop_is_not_reported(self):
        image = frame()
        boxes = [(200.0, 100.0, 280.0, 340.0), (500.0, 100.0, 560.0, 340.0)]
        self.assertEqual(occluder_bounds_inside_reid_crop(image, boxes[0], boxes, 0), [])

    def test_the_subject_never_reports_itself(self):
        image = frame()
        box = (200.0, 100.0, 280.0, 340.0)
        self.assertEqual(occluder_bounds_inside_reid_crop(image, box, [box], 0), [])
        # A duplicate detection of the same box is the subject too.
        self.assertEqual(occluder_bounds_inside_reid_crop(image, box, [box, box], 0), [])

    def test_an_unresolved_shadow_of_the_subject_is_not_blanked(self):
        # A suppressed detection is a duplicate of the subject; blanking it
        # would blank the person the crop exists to show.
        image = frame()
        boxes = [(200.0, 100.0, 280.0, 340.0), (205.0, 105.0, 285.0, 345.0)]
        self.assertEqual(
            occluder_bounds_inside_reid_crop(
                image,
                boxes[0],
                boxes,
                0,
                suppressed_by_index=[False, True],
            ),
            [],
        )
        self.assertEqual(
            len(occluder_bounds_inside_reid_crop(image, boxes[0], boxes, 0)),
            1,
        )


class IntakeSamplePlumbingTests(unittest.TestCase):
    """What get_standing_points measures has to survive into the stored sample."""

    def setUp(self):
        self.memories = []

    def tearDown(self):
        for memory in self.memories:
            memory.close(drain=False, timeout=0.2)

    def make_memory(self):
        memory = AppearanceIdentityMemory(
            reid_extractor=CountingBatchExtractor(),
            intake_frames=5,
            intake_delay_seconds=0.0,
            blur_threshold=1.0,
            evidence_dir=None,
            enable_role_classification=False,
            enable_demographics=True,
        )
        self.memories.append(memory)
        return memory

    def observe(self, memory, frame_index, boxes, track_ids):
        return get_standing_points(
            FakeResult(track_ids, boxes=boxes),
            sharp_frame(),
            frame_index=frame_index,
            appearance_memory=memory,
            camera_id="cam_1",
            observation_time=float(frame_index),
        )

    def test_the_tight_body_bounds_reach_the_stored_sample(self):
        memory = self.make_memory()
        self.observe(memory, 1, [[10, 10, 60, 100]], [1])
        state = memory.pending_intake[("cam_1", 1)]
        stored = state["samples"][0]
        self.assertIsNotNone(stored["body_bounds"])
        self.assertLess(stored["body_bounds"][3], 1.0)
        self.assertEqual(stored["occluder_boxes"], ())
        # No pose estimator ran, so there is no face box -- and the crop is
        # still usable, read body-only.
        self.assertIsNone(stored["face_box"])

    def test_a_neighbouring_person_reaches_the_stored_sample(self):
        memory = self.make_memory()
        self.observe(memory, 1, [[10, 10, 60, 100], [55, 20, 85, 105]], [1, 2])
        stored = memory.pending_intake[("cam_1", 1)]["samples"][0]
        self.assertEqual(len(stored["occluder_boxes"]), 1)

    def test_the_framing_survives_into_the_demographics_candidates(self):
        memory = self.make_memory()
        self.observe(memory, 1, [[10, 10, 60, 100], [55, 20, 85, 105]], [1, 2])
        samples = memory.pending_intake[("cam_1", 1)]["samples"]
        candidate = AppearanceIdentityMemory._demographics_candidates(samples)[0]
        self.assertIsNotNone(candidate["body_bounds"])
        self.assertEqual(len(candidate["occluder_boxes"]), 1)


if __name__ == "__main__":
    unittest.main()
