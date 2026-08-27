"""Regression tests for the crops MiVOLO is given and the votes it produces.

The model itself is never loaded here: DemographicsEngine is built without
__init__ and handed a stub model, so these tests cover the framing and
aggregation rules rather than the network's own accuracy.
"""

import unittest

import numpy as np
import torch

from constants import (
    DEFAULT_DEMOGRAPHICS_BODY_ONLY_WEIGHT,
    DEFAULT_DEMOGRAPHICS_MIN_BODY_PIXELS,
    DEFAULT_DEMOGRAPHICS_MIN_FACE_PIXELS,
)
from demographics import (
    DemographicsEngine,
    DemographicsReading,
    _winsorized_weighted_mean,
)


def solid_crop(height, width, value=200):
    return np.full((height, width, 3), value, dtype=np.uint8)


class _StubOutputs:
    def __init__(self, ages, gender_indices, gender_probs):
        self.age_output = torch.tensor(ages, dtype=torch.float32).reshape(-1, 1)
        self.gender_class_idx = torch.tensor(gender_indices, dtype=torch.int64).reshape(-1, 1)
        self.gender_probs = torch.tensor(gender_probs, dtype=torch.float32).reshape(-1, 1)


class _StubModel:
    """Returns a scripted reading per crop and records what it was shown."""

    dtype = torch.float32

    def __init__(self, ages, gender_indices, gender_probs):
        self._ages = ages
        self._gender_indices = gender_indices
        self._gender_probs = gender_probs
        self.calls = 0

    def __call__(self, faces_input=None, body_input=None):
        self.calls += 1
        self.faces_input = faces_input
        self.body_input = body_input
        return _StubOutputs(self._ages, self._gender_indices, self._gender_probs)


class _StubProcessor:
    """Stands in for MiVOLO's processor, keeping the crop list it was passed."""

    def __init__(self):
        self.batches = []

    def __call__(self, images):
        self.batches.append(list(images))
        tensor = torch.zeros((len(images), 3, 8, 8), dtype=torch.float32)
        return {"pixel_values": tensor}


def build_engine(ages, gender_indices, gender_probs):
    engine = DemographicsEngine.__new__(DemographicsEngine)
    engine.device = "cpu"
    engine.model = _StubModel(ages, gender_indices, gender_probs)
    engine.processor = _StubProcessor()
    engine.id2label = {0: "female", 1: "male"}
    return engine


class CropFramingTests(unittest.TestCase):
    """What actually reaches the two branches of the model."""

    def test_the_body_crop_is_cut_back_to_the_detection_box(self):
        # The saved crop carries 2% side and 10% bottom padding; MiVOLO was
        # trained on the tight box, so the padding has to come back off.
        crop = solid_crop(300, 100)
        face, body = DemographicsEngine._frame_one_crop(
            {"crop": crop, "body_bounds": (0.02, 0.02, 0.98, 0.91), "face_box": None}
        )
        self.assertIsNone(face)
        self.assertEqual(body.shape[:2], (267, 96))

    def test_without_stored_bounds_the_whole_crop_stands_in_for_the_body(self):
        crop = solid_crop(300, 100)
        _face, body = DemographicsEngine._frame_one_crop({"crop": crop})
        self.assertEqual(body.shape[:2], (300, 100))

    def test_a_bare_array_is_accepted_and_read_body_only(self):
        face, body = DemographicsEngine._frame_one_crop(solid_crop(300, 100))
        self.assertIsNone(face)
        self.assertEqual(body.shape[:2], (300, 100))

    def test_the_face_is_cut_out_of_the_body_branch(self):
        # The two inputs are trained to be complementary, so the model must not
        # be shown the same face pixels in both.
        crop = solid_crop(300, 100)
        face, body = DemographicsEngine._frame_one_crop(
            {"crop": crop, "face_box": (0.30, 0.02, 0.70, 0.16), "body_bounds": None}
        )
        self.assertIsNotNone(face)
        self.assertEqual(face.shape[:2], (42, 40))
        self.assertTrue((body[6:48, 30:70] == 0).all())
        # Everything outside the face is left alone.
        self.assertTrue((body[100:, :] == 200).all())

    def test_the_face_is_cut_from_the_right_place_in_a_tightened_body_crop(self):
        # The face box is a fraction of the saved crop, but the body branch is
        # a sub-rectangle of it, so the box has to be rebased or the hole lands
        # in the wrong place.
        crop = solid_crop(300, 100)
        _face, body = DemographicsEngine._frame_one_crop(
            {
                "crop": crop,
                "face_box": (0.30, 0.10, 0.70, 0.24),
                "body_bounds": (0.0, 0.05, 1.0, 1.0),
            }
        )
        # 0.10 of the saved crop is y=30; the body crop starts at y=15.
        self.assertTrue((body[15:57, 30:70] == 0).all())
        self.assertTrue((body[0:14, :] == 200).all())

    def test_a_face_below_the_size_floor_is_dropped_rather_than_upscaled(self):
        crop = solid_crop(300, 100)
        span = (DEFAULT_DEMOGRAPHICS_MIN_FACE_PIXELS - 2) / 100.0
        face, body = DemographicsEngine._frame_one_crop(
            {"crop": crop, "face_box": (0.4, 0.02, 0.4 + span, 0.02 + span / 3.0)}
        )
        self.assertIsNone(face)
        self.assertIsNotNone(body)
        # A dropped face must not leave a hole punched in the body branch.
        self.assertTrue((body == 200).all())

    def test_a_face_exactly_on_the_size_floor_is_kept(self):
        crop = solid_crop(300, 100)
        width = DEFAULT_DEMOGRAPHICS_MIN_FACE_PIXELS / 100.0
        height = DEFAULT_DEMOGRAPHICS_MIN_FACE_PIXELS / 300.0
        face, _body = DemographicsEngine._frame_one_crop(
            {"crop": crop, "face_box": (0.3, 0.02, 0.3 + width, 0.02 + height)}
        )
        self.assertIsNotNone(face)
        self.assertEqual(min(face.shape[:2]), DEFAULT_DEMOGRAPHICS_MIN_FACE_PIXELS)

    def test_a_body_below_the_size_floor_is_read_face_only(self):
        # Upstream MiVOLO refuses person crops this small outright. A face can
        # still be legible when the body is not.
        side = DEFAULT_DEMOGRAPHICS_MIN_BODY_PIXELS - 10
        crop = solid_crop(side, side)
        face, body = DemographicsEngine._frame_one_crop(
            {"crop": crop, "face_box": (0.05, 0.05, 0.95, 0.95)}
        )
        self.assertIsNotNone(face)
        self.assertIsNone(body)

    def test_a_crop_too_small_for_either_branch_is_unusable(self):
        face, body = DemographicsEngine._frame_one_crop({"crop": solid_crop(20, 20)})
        self.assertIsNone(face)
        self.assertIsNone(body)

    def test_an_empty_or_missing_crop_is_unusable(self):
        self.assertEqual(DemographicsEngine._frame_one_crop({"crop": None}), (None, None))
        self.assertEqual(
            DemographicsEngine._frame_one_crop(np.zeros((0, 0, 3), dtype=np.uint8)),
            (None, None),
        )


class OccluderBlankingTests(unittest.TestCase):
    def test_a_neighbour_inside_the_crop_is_blanked(self):
        crop = solid_crop(300, 100)
        _face, body = DemographicsEngine._frame_one_crop(
            {"crop": crop, "occluder_boxes": ((0.0, 0.6, 0.2, 0.9),)}
        )
        self.assertTrue((body[180:270, 0:20] == 0).all())
        self.assertTrue((body[0:179, :] == 200).all())

    def test_a_neighbour_covering_most_of_the_crop_is_left_alone(self):
        # Past this size the thing being erased is the subject, not an
        # intruder, and a blank body branch is worse than a shared one.
        crop = solid_crop(300, 100)
        _face, body = DemographicsEngine._frame_one_crop(
            {"crop": crop, "occluder_boxes": ((0.0, 0.0, 1.0, 0.8),)}
        )
        self.assertTrue((body == 200).all())

    def test_neighbours_are_rebased_onto_a_tightened_body_crop(self):
        crop = solid_crop(300, 100)
        _face, body = DemographicsEngine._frame_one_crop(
            {
                "crop": crop,
                "body_bounds": (0.0, 0.10, 1.0, 1.0),
                "occluder_boxes": ((0.0, 0.20, 0.2, 0.40),),
            }
        )
        # y 0.20-0.40 of the saved crop is 60-120; the body starts at 30.
        self.assertTrue((body[30:90, 0:20] == 0).all())
        self.assertTrue((body[0:29, 0:20] == 200).all())

    def test_no_neighbours_means_the_crop_is_not_copied(self):
        crop = solid_crop(300, 100)
        _face, body = DemographicsEngine._frame_one_crop({"crop": crop})
        self.assertTrue(np.shares_memory(body, crop))


class AgeAggregationTests(unittest.TestCase):
    def test_an_outlier_is_pulled_to_the_median_not_averaged_in(self):
        # A plain mean of these is 41.6 -- nine years off four readings that
        # agree within three.  Clipping the outlier to six years past the
        # median of 32 lets it nudge the answer instead of dragging it.
        values = [30.0, 31.0, 32.0, 33.0, 82.0]
        weights = [1.0] * 5
        self.assertAlmostEqual(_winsorized_weighted_mean(values, weights, 6.0), 32.8, places=5)
        self.assertAlmostEqual(float(np.mean(values)), 41.6, places=5)

    def test_weights_shift_the_result_towards_the_trusted_readings(self):
        heavy = _winsorized_weighted_mean([30.0, 40.0], [1.0, 0.35], 6.0)
        even = _winsorized_weighted_mean([30.0, 40.0], [1.0, 1.0], 6.0)
        self.assertLess(heavy, even)

    def test_no_values_or_no_weight_gives_no_answer(self):
        self.assertIsNone(_winsorized_weighted_mean([], [], 6.0))
        self.assertIsNone(_winsorized_weighted_mean([30.0], [0.0], 6.0))


class BatchedInferenceTests(unittest.TestCase):
    def test_every_crop_goes_through_the_model_in_one_pass(self):
        engine = build_engine([30.0] * 5, [1] * 5, [0.9] * 5)
        crops = [{"crop": solid_crop(300, 100), "face_box": (0.3, 0.02, 0.7, 0.16)} for _ in range(5)]
        reading = engine.analyze_batch(crops)
        self.assertEqual(engine.model.calls, 1)
        self.assertEqual(engine.model.faces_input.shape[0], 5)
        self.assertEqual(engine.model.body_input.shape[0], 5)
        self.assertEqual(reading.samples_used, 5)
        self.assertEqual(reading.samples_with_face, 5)

    def test_a_missing_face_is_passed_as_none_for_the_zeroed_tensor(self):
        # MiVOLO's processor renders None as the zeroed face tensor the model
        # was trained to read as "no face available".
        engine = build_engine([30.0], [1], [0.9])
        engine.analyze_batch([{"crop": solid_crop(300, 100)}])
        face_batch = engine.processor.batches[0]
        self.assertEqual(face_batch, [None])

    def test_a_body_only_reading_is_weighted_below_one_with_a_face(self):
        engine = build_engine([50.0, 20.0], [1, 0], [0.9, 0.9])
        reading = engine.analyze_batch(
            [
                {"crop": solid_crop(300, 100)},
                {"crop": solid_crop(300, 100), "face_box": (0.3, 0.02, 0.7, 0.16)},
            ]
        )
        self.assertEqual(reading.samples_with_face, 1)
        # Both readings are clipped to within six years of the 35 median, then
        # the faceless one counts for less.
        expected = (41.0 * DEFAULT_DEMOGRAPHICS_BODY_ONLY_WEIGHT + 29.0) / (
            DEFAULT_DEMOGRAPHICS_BODY_ONLY_WEIGHT + 1.0
        )
        self.assertEqual(reading.age, int(round(expected)))
        # The face-bearing crop voted female, and outweighs the other.
        self.assertEqual(reading.gender, "female")

    def test_one_clear_face_outvotes_two_hesitant_faceless_readings(self):
        # The exact case the raw count got wrong: two distant crops with no
        # readable face guess male at barely over a coin toss, and one good
        # look at the person says female.  Counting votes returned male.
        engine = build_engine([30.0] * 3, [1, 1, 0], [0.51, 0.52, 0.99])
        reading = engine.analyze_batch(
            [
                {"crop": solid_crop(300, 100)},
                {"crop": solid_crop(300, 100)},
                {"crop": solid_crop(300, 100), "face_box": (0.3, 0.02, 0.7, 0.16)},
            ]
        )
        self.assertEqual(reading.gender, "female")

    def test_confidence_reflects_certainty_not_only_agreement(self):
        unanimous_but_unsure = build_engine([30.0] * 3, [1] * 3, [0.51] * 3).analyze_batch(
            [{"crop": solid_crop(300, 100), "face_box": (0.3, 0.02, 0.7, 0.16)}] * 3
        )
        unanimous_and_sure = build_engine([30.0] * 3, [1] * 3, [0.99] * 3).analyze_batch(
            [{"crop": solid_crop(300, 100), "face_box": (0.3, 0.02, 0.7, 0.16)}] * 3
        )
        self.assertAlmostEqual(unanimous_but_unsure.confidence, 0.51, places=5)
        self.assertAlmostEqual(unanimous_and_sure.confidence, 0.99, places=5)

    def test_a_tied_gender_vote_resolves_the_same_way_every_run(self):
        engine = build_engine([30.0, 30.0], [0, 1], [0.8, 0.8])
        crops = [{"crop": solid_crop(300, 100), "face_box": (0.3, 0.02, 0.7, 0.16)}] * 2
        first = engine.analyze_batch(crops).gender
        for _ in range(5):
            self.assertEqual(engine.analyze_batch(crops).gender, first)

    def test_an_age_of_one_is_reported_rather_than_discarded(self):
        # The old zero-filter dropped every reading at or below one year,
        # which silently threw away infants.
        engine = build_engine([1.0], [0], [0.9])
        reading = engine.analyze_batch([{"crop": solid_crop(300, 100)}])
        self.assertEqual(reading.age, 1)

    def test_a_non_positive_or_non_finite_age_is_discarded(self):
        engine = build_engine([0.0, float("nan"), 30.0], [0, 0, 0], [0.9, 0.9, 0.9])
        reading = engine.analyze_batch([{"crop": solid_crop(300, 100)}] * 3)
        self.assertEqual(reading.samples_used, 1)
        self.assertEqual(reading.age, 30)

    def test_nothing_usable_returns_the_unknown_reading(self):
        engine = build_engine([], [], [])
        self.assertEqual(
            engine.analyze_batch([]),
            DemographicsReading(0, "Unknown", 0.0, 0, 0),
        )
        self.assertEqual(engine.model.calls, 0)

    def test_an_unusable_crop_is_skipped_without_shifting_the_others(self):
        # A dropped crop must not misalign the model's output with the crops
        # that were actually sent.
        engine = build_engine([30.0, 60.0], [0, 1], [0.9, 0.9])
        reading = engine.analyze_batch(
            [
                {"crop": solid_crop(20, 20)},
                {"crop": solid_crop(300, 100)},
                {"crop": solid_crop(300, 100), "face_box": (0.3, 0.02, 0.7, 0.16)},
            ]
        )
        self.assertEqual(reading.samples_used, 2)
        self.assertEqual(reading.samples_with_face, 1)
        self.assertEqual(engine.model.faces_input.shape[0], 2)

    def test_a_gender_index_outside_the_dictionary_still_yields_an_age(self):
        engine = build_engine([30.0], [7], [0.9])
        reading = engine.analyze_batch([{"crop": solid_crop(300, 100)}])
        self.assertEqual(reading.age, 30)
        self.assertEqual(reading.gender, "Unknown")

    def test_string_keyed_gender_labels_are_read(self):
        # A config round-tripped through JSON comes back with string keys.
        engine = build_engine([30.0], [1], [0.9])
        engine.id2label = {"0": "female", "1": "male"}
        self.assertEqual(engine.analyze_batch([{"crop": solid_crop(300, 100)}]).gender, "male")


if __name__ == "__main__":
    unittest.main()
