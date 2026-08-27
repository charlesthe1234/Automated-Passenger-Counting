"""Locating the face inside a ReID crop from the pose landmarks already taken.

MiVOLO wants a face crop and a body crop.  Nothing in this pipeline runs a face
detector, and adding one would mean another model, another download and another
share of the GPU.  MediaPipe is already run on the intake crop to decide body
completeness and orientation, and its first eleven landmarks describe the head,
so the face box is derived from those instead: no extra inference, and it works
in profile, where the frontal Haar cascade it replaces did not.

Boxes are returned in ReID-crop-normalized coordinates -- fractions of the
saved crop, not pixels -- because the consumer sees the crop long after the
frame it came from is gone, and a fraction survives any resize.
"""

import numpy as np

from constants import (
    DEFAULT_FACE_BOX_HEAD_WIDTH_SCALE,
    MEDIAPIPE_LEFT_EAR,
    MEDIAPIPE_LEFT_EYE,
    MEDIAPIPE_MOUTH_LEFT,
    MEDIAPIPE_MOUTH_RIGHT,
    MEDIAPIPE_NOSE,
    MEDIAPIPE_RIGHT_EAR,
    MEDIAPIPE_RIGHT_EYE,
    MIN_MEDIAPIPE_VISIBILITY,
)

# Head width implied by the distance between two landmarks, as a multiplier.
# These are anatomical proportions rather than tunables: ear-to-ear spans very
# nearly the whole head, the pupils sit about a third of that apart, and the
# mouth corners about a fifth.  They are listed widest-first so the estimate is
# always taken from the longest baseline available, where a pixel of landmark
# noise costs the least.
_HEAD_WIDTH_FROM_EAR_SPAN = 1.15
_HEAD_WIDTH_FROM_EYE_SPAN = 2.85
_HEAD_WIDTH_FROM_EYE_TO_EAR = 2.05
_HEAD_WIDTH_FROM_MOUTH_SPAN = 4.60


def _landmark_point(landmarks, index, min_visibility):
    """Return one landmark as (x, y) in pose-crop fractions, or None."""
    if landmarks is None or index >= len(landmarks):
        return None
    landmark = landmarks[index]
    if float(getattr(landmark, "visibility", 0.0)) < min_visibility:
        return None
    try:
        x = float(landmark.x)
        y = float(landmark.y)
    except (AttributeError, TypeError, ValueError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def _estimate_head_width(points, aspect_ratio):
    """Head width in pose-crop x-fractions, from the widest baseline available.

    ``aspect_ratio`` converts a y-fraction into the x-fraction that spans the
    same number of pixels, so distances measured on a non-square crop are not
    stretched along one axis before being compared.
    """

    def separation(first, second):
        if first is None or second is None:
            return None
        dx = first[0] - second[0]
        dy = (first[1] - second[1]) * aspect_ratio
        distance = float(np.hypot(dx, dy))
        return distance if distance > 0.0 else None

    left_ear = points.get("left_ear")
    right_ear = points.get("right_ear")
    left_eye = points.get("left_eye")
    right_eye = points.get("right_eye")

    ear_span = separation(left_ear, right_ear)
    if ear_span is not None:
        return ear_span * _HEAD_WIDTH_FROM_EAR_SPAN

    eye_span = separation(left_eye, right_eye)
    if eye_span is not None:
        return eye_span * _HEAD_WIDTH_FROM_EYE_SPAN

    # Turned away from the camera: one eye and the ear on the same side are
    # often the only pair left, and they still span most of a head profile.
    for eye, ear in (("left_eye", "left_ear"), ("right_eye", "right_ear")):
        eye_to_ear = separation(points.get(eye), points.get(ear))
        if eye_to_ear is not None:
            return eye_to_ear * _HEAD_WIDTH_FROM_EYE_TO_EAR

    mouth_span = separation(points.get("mouth_left"), points.get("mouth_right"))
    if mouth_span is not None:
        return mouth_span * _HEAD_WIDTH_FROM_MOUTH_SPAN

    return None


def estimate_face_box(
    landmarks,
    normalized_bounds=(0.0, 0.0, 1.0, 1.0),
    crop_width=None,
    crop_height=None,
    min_visibility=MIN_MEDIAPIPE_VISIBILITY,
    head_width_scale=DEFAULT_FACE_BOX_HEAD_WIDTH_SCALE,
):
    """Return (x1, y1, x2, y2) as fractions of the saved ReID crop, or None.

    ``normalized_bounds`` says where the saved ReID crop sits inside the larger
    image MediaPipe was given, matching assess_reid_body_completeness.  When
    MediaPipe ran on the ReID crop itself the default identity bounds apply.

    ``crop_width`` and ``crop_height`` are the pixel dimensions of the image
    the landmarks are normalized against.  They only set the aspect ratio used
    to keep the box square in pixels; without them it is assumed square, which
    for a person-shaped crop stretches the box vertically and is why they
    should be passed when known.

    Returns None whenever the head is not visible enough to size a box
    honestly.  That is a supported answer: MiVOLO accepts a missing face and
    was trained for it, so a guess here would be strictly worse than silence.
    """
    names = {
        "nose": MEDIAPIPE_NOSE,
        "left_eye": MEDIAPIPE_LEFT_EYE,
        "right_eye": MEDIAPIPE_RIGHT_EYE,
        "left_ear": MEDIAPIPE_LEFT_EAR,
        "right_ear": MEDIAPIPE_RIGHT_EAR,
        "mouth_left": MEDIAPIPE_MOUTH_LEFT,
        "mouth_right": MEDIAPIPE_MOUTH_RIGHT,
    }
    points = {}
    for name, index in names.items():
        point = _landmark_point(landmarks, index, min_visibility)
        if point is not None:
            points[name] = point
    if not points:
        return None

    left, top, right, bottom = map(float, normalized_bounds)
    bounds_width = right - left
    bounds_height = bottom - top
    if bounds_width <= 0.0 or bounds_height <= 0.0:
        return None

    if crop_width and crop_height:
        aspect_ratio = float(crop_height) / float(crop_width)
    else:
        aspect_ratio = 1.0

    head_width = _estimate_head_width(points, aspect_ratio)
    if head_width is None:
        return None

    # The nose is the closest landmark to the centre of a face from any angle.
    # Falling back to the centroid of whatever is visible skews towards the
    # side of the head still facing the camera, which is the correct bias: it
    # follows the face rather than the skull.
    centre = points.get("nose")
    if centre is None:
        visible = np.asarray(list(points.values()), dtype=float)
        centre = (float(visible[:, 0].mean()), float(visible[:, 1].mean()))

    half_width = 0.5 * head_width * float(head_width_scale)
    if half_width <= 0.0:
        return None
    # Square in pixels, which on a tall crop is a smaller span of y-fractions.
    half_height = half_width / aspect_ratio if aspect_ratio > 0.0 else half_width

    face_left = centre[0] - half_width
    face_right = centre[0] + half_width
    face_top = centre[1] - half_height
    face_bottom = centre[1] + half_height

    # Re-express against the saved ReID crop, then clamp.  A face partly
    # outside the saved crop is kept as the part that survives: the visible
    # half of a face at the crop edge still carries real signal.
    def to_crop_x(value):
        return (value - left) / bounds_width

    def to_crop_y(value):
        return (value - top) / bounds_height

    x1 = max(0.0, min(1.0, to_crop_x(face_left)))
    x2 = max(0.0, min(1.0, to_crop_x(face_right)))
    y1 = max(0.0, min(1.0, to_crop_y(face_top)))
    y2 = max(0.0, min(1.0, to_crop_y(face_bottom)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (float(x1), float(y1), float(x2), float(y2))


def normalized_box_to_pixels(box, width, height):
    """Turn a normalized box into integer pixel bounds inside a crop."""
    if box is None or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    px1 = max(0, min(int(width) - 1, int(round(x1 * width))))
    py1 = max(0, min(int(height) - 1, int(round(y1 * height))))
    px2 = max(0, min(int(width), int(round(x2 * width))))
    py2 = max(0, min(int(height), int(round(y2 * height))))
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2
