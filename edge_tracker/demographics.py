"""Age and gender for one person, from MiVOLO V2.

The model reads two complementary inputs: a face crop, and a body crop with
the face cut out of it.  Its own pipeline builds those from a tight detection
box with other people blanked, so this module reproduces that framing from the
padded ReID crops the intake already collects, rather than handing the model a
crop shaped for a different purpose.

Face location comes from the pose landmarks taken during intake (see
face_region), not from a face detector of our own.  Either input may be
missing: MiVOLO's processor turns None into the zeroed tensor the model was
trained to read for that case, which is how upstream handles a face with no
body and a body with no face.  A reading taken without a face is kept, but
weighted down rather than trusted equally.
"""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

from constants import (
    DEFAULT_DEMOGRAPHICS_BODY_ONLY_WEIGHT,
    DEFAULT_DEMOGRAPHICS_MAX_AGE_SPREAD_YEARS,
    DEFAULT_DEMOGRAPHICS_MIN_BODY_PIXELS,
    DEFAULT_DEMOGRAPHICS_MIN_FACE_PIXELS,
)
from face_region import normalized_box_to_pixels

MIVOLO_MODEL_ID = "iitolstykh/mivolo_v2"
MIVOLO_REVISION = "53393526c220e34cdd7b722b36d22b6f9e5f4241"

# A neighbour box covering more of the crop than this is not an intruder to be
# blanked -- at that size the thing being erased is the subject.
MAX_BLANKED_OCCLUDER_AREA_RATIO = 0.5


@dataclass(frozen=True)
class DemographicsReading:
    """One person's estimate, with enough context to judge how much to trust it.

    ``age`` is 0 and ``gender`` is "Unknown" when nothing usable was found,
    which is the pair callers already treat as "no answer".
    """

    age: int
    gender: str
    confidence: float
    samples_used: int
    samples_with_face: int


def _as_framed_crop(item):
    """Accept either a bare crop or an intake candidate carrying its framing."""
    if isinstance(item, dict):
        return (
            item.get("crop"),
            item.get("face_box"),
            item.get("body_bounds"),
            tuple(item.get("occluder_boxes") or ()),
        )
    return item, None, None, ()


def _rebase_box(box, crop_width, crop_height, region_pixels):
    """Move a crop-normalized box into the pixel frame of a sub-region of it."""
    if box is None or region_pixels is None:
        return None
    pixels = normalized_box_to_pixels(box, crop_width, crop_height)
    if pixels is None:
        return None
    region_x1, region_y1, region_x2, region_y2 = region_pixels
    region_width = region_x2 - region_x1
    region_height = region_y2 - region_y1
    if region_width <= 0 or region_height <= 0:
        return None
    x1 = max(0, min(region_width, pixels[0] - region_x1))
    y1 = max(0, min(region_height, pixels[1] - region_y1))
    x2 = max(0, min(region_width, pixels[2] - region_x1))
    y2 = max(0, min(region_height, pixels[3] - region_y1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _blank_regions(crop, regions):
    """Zero the given pixel regions, copying only if there is something to zero."""
    blanked = None
    for region in regions:
        if region is None:
            continue
        x1, y1, x2, y2 = region
        if blanked is None:
            blanked = crop.copy()
        blanked[y1:y2, x1:x2] = 0
    return crop if blanked is None else blanked


def _winsorized_weighted_mean(values, weights, max_spread):
    """Weighted mean after pulling outliers back towards the median.

    This is MiVOLO's own aggregate_votes_winsorized rule -- clip each reading
    to within ``max_spread`` years of the median -- with weights added so a
    body-only reading counts for less than one taken from a clear face.  The
    plain mean it replaces let a single bad crop drag the answer by years.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or float(weights.sum()) <= 0.0:
        return None
    median = float(np.median(values))
    clipped = np.clip(values, median - float(max_spread), median + float(max_spread))
    return float((clipped * weights).sum() / weights.sum())


class DemographicsEngine:
    def __init__(self, device=None):
        print("Initializing Official MiVOLO V2 (Transformers Engine)...")
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = str(requested_device)
        model_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        self.config = AutoConfig.from_pretrained(
            MIVOLO_MODEL_ID,
            revision=MIVOLO_REVISION,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageClassification.from_pretrained(
            MIVOLO_MODEL_ID,
            revision=MIVOLO_REVISION,
            config=self.config,
            trust_remote_code=True,
            torch_dtype=model_dtype,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoImageProcessor.from_pretrained(
            MIVOLO_MODEL_ID,
            revision=MIVOLO_REVISION,
            trust_remote_code=True,
        )

        # Pull the official gender text dictionary from the model
        self.id2label = self.config.gender_id2label
        print("MiVOLO V2 Online & Ready!")

    def _gender_label(self, index):
        """Read the model's own gender dictionary, however its keys survived JSON."""
        for key in (int(index), str(int(index))):
            try:
                label = self.id2label[key]
            except (KeyError, IndexError, TypeError):
                continue
            if label is not None:
                return str(label)
        return None

    @staticmethod
    def _frame_one_crop(item):
        """Build the (face, body) pair MiVOLO expects from one stored crop.

        Either half may come back None, which the processor renders as the
        zeroed tensor standing for "not available".  Both being None means the
        crop is unusable and the caller should drop it.
        """
        crop, face_box, body_bounds, occluder_boxes = _as_framed_crop(item)
        if crop is None or getattr(crop, "size", 0) == 0:
            return None, None
        crop_height, crop_width = crop.shape[:2]

        # The saved crop is padded -- 10% below the detection box especially,
        # so the appearance model still sees shoes -- and MiVOLO was trained on
        # the detector's tight box.  Without stored bounds the whole crop has
        # to stand in for it.
        body_pixels = normalized_box_to_pixels(body_bounds, crop_width, crop_height)
        if body_pixels is None:
            body_pixels = (0, 0, crop_width, crop_height)

        face_crop = None
        face_pixels = normalized_box_to_pixels(face_box, crop_width, crop_height)
        if face_pixels is not None:
            face_x1, face_y1, face_x2, face_y2 = face_pixels
            # Below this the resize to 384 invents detail, and MiVOLO reads a
            # blurred smear as an averaged, middle-aged face.  Passing no face
            # at all is the better of the two answers.
            if min(face_x2 - face_x1, face_y2 - face_y1) >= DEFAULT_DEMOGRAPHICS_MIN_FACE_PIXELS:
                face_crop = crop[face_y1:face_y2, face_x1:face_x2]

        body_x1, body_y1, body_x2, body_y2 = body_pixels
        body_crop = crop[body_y1:body_y2, body_x1:body_x2]
        body_height, body_width = body_crop.shape[:2]
        if min(body_height, body_width) < DEFAULT_DEMOGRAPHICS_MIN_BODY_PIXELS:
            # Upstream refuses person crops this small outright, and reads the
            # face alone.  A face can still be legible when the body is not.
            return face_crop, None

        blank_regions = []
        if face_crop is not None:
            # The branches are trained to be complementary: the body carries
            # build and dress, with the face cut out so the model is not shown
            # the same pixels twice.
            blank_regions.append(
                _rebase_box(face_box, crop_width, crop_height, body_pixels)
            )

        body_area = float(body_height * body_width)
        for occluder in occluder_boxes:
            region = _rebase_box(occluder, crop_width, crop_height, body_pixels)
            if region is None:
                continue
            occluder_area = float((region[2] - region[0]) * (region[3] - region[1]))
            if body_area > 0.0 and occluder_area / body_area > MAX_BLANKED_OCCLUDER_AREA_RATIO:
                continue
            blank_regions.append(region)

        return face_crop, _blank_regions(body_crop, blank_regions)

    def analyze_batch(self, crop_list):
        """Estimate one person's age and gender from their best crops.

        Accepts the intake candidates -- crops with the face box, tight body
        bounds and neighbour boxes measured for them -- or bare crops, which
        are read body-only because nothing says where the face is.
        """
        face_inputs = []
        body_inputs = []
        has_face = []
        for item in crop_list or ():
            face_crop, body_crop = self._frame_one_crop(item)
            if face_crop is None and body_crop is None:
                continue
            face_inputs.append(face_crop)
            body_inputs.append(body_crop)
            has_face.append(face_crop is not None)

        if not face_inputs:
            return DemographicsReading(0, "Unknown", 0.0, 0, 0)

        # One forward pass for the whole set.  The processor takes a list and
        # concatenates it, so the launch and transfer cost is paid once rather
        # than per crop -- measured at 96ms to 67ms for five crops on a 2080
        # Ti.  The model is large enough that compute still dominates, so this
        # is a real saving rather than the near-free batch a smaller net gets.
        faces_tensor = self.processor(images=face_inputs)["pixel_values"].to(
            dtype=self.model.dtype, device=self.device
        )
        body_tensor = self.processor(images=body_inputs)["pixel_values"].to(
            dtype=self.model.dtype, device=self.device
        )

        with torch.no_grad():
            outputs = self.model(faces_input=faces_tensor, body_input=body_tensor)

        ages = outputs.age_output.detach().float().cpu().numpy().reshape(-1)
        gender_indices = outputs.gender_class_idx.detach().cpu().numpy().reshape(-1)
        gender_probs = outputs.gender_probs.detach().float().cpu().numpy().reshape(-1)

        age_values = []
        age_weights = []
        gender_scores = {}
        gender_weight_total = 0.0
        samples_with_face = 0

        for index in range(min(len(body_inputs), len(ages), len(gender_indices))):
            age = float(ages[index])
            if not np.isfinite(age) or age <= 0.0:
                continue
            face_present = bool(has_face[index])
            # MiVOLO's published age error for body-only input is several years
            # worse than face-plus-body, so a faceless reading is evidence but
            # must not outweigh a clear look at the person.
            weight = 1.0 if face_present else DEFAULT_DEMOGRAPHICS_BODY_ONLY_WEIGHT
            age_values.append(age)
            age_weights.append(weight)
            if face_present:
                samples_with_face += 1

            gender = self._gender_label(gender_indices[index])
            if gender is None:
                continue
            probability = float(gender_probs[index]) if index < len(gender_probs) else 1.0
            if not np.isfinite(probability):
                probability = 1.0
            # Weighted by the model's own confidence, so one hesitant guess can
            # no longer outvote two certain readings the way a raw count did.
            gender_scores[gender] = gender_scores.get(gender, 0.0) + probability * weight
            gender_weight_total += weight

        if not age_values:
            return DemographicsReading(0, "Unknown", 0.0, 0, samples_with_face)

        mean_age = _winsorized_weighted_mean(
            age_values,
            age_weights,
            DEFAULT_DEMOGRAPHICS_MAX_AGE_SPREAD_YEARS,
        )
        final_age = 0 if mean_age is None else int(round(mean_age))

        final_gender = "Unknown"
        confidence = 0.0
        if gender_scores:
            # Sorted by name first so a dead heat resolves the same way every
            # run, instead of following set iteration order as the old
            # max(set(...), key=count) did.
            final_gender, winning_score = max(
                sorted(gender_scores.items()),
                key=lambda item: item[1],
            )
            # Divided by the total weight rather than by the winning side's
            # own score, so the figure answers "how sure are we this person is
            # female" and not merely "did the votes agree".  Five unanimous but
            # barely-committed readings should not report certainty.
            if gender_weight_total > 0.0:
                confidence = winning_score / gender_weight_total

        return DemographicsReading(
            age=final_age,
            gender=final_gender,
            confidence=float(confidence),
            samples_used=len(age_values),
            samples_with_face=samples_with_face,
        )
