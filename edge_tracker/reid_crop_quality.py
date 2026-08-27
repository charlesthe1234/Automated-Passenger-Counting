"""Deciding whether a detection crop is fit to become ReID evidence.

Answers two separate questions: is enough of the body in frame, and is
someone else's body inside the crop we are about to store."""

import numpy as np

from constants import (
    DEFAULT_REID_CROP_BEHIND_FOOT_MARGIN,
    DEFAULT_REID_CROP_BOTTOM_PADDING,
    DEFAULT_REID_CROP_MAX_BEHIND_INTRUDER_AREA_RATIO,
    DEFAULT_REID_CROP_MAX_INTRUDER_AREA_RATIO,
    DEFAULT_REID_CROP_SIDE_PADDING,
    DEFAULT_REID_CROP_TOP_PADDING,
    DEFAULT_REID_FRAME_EDGE_MARGIN_PIXELS,
    MEDIAPIPE_LEFT_ANKLE,
    MEDIAPIPE_LEFT_EYE,
    MEDIAPIPE_LEFT_HIP,
    MEDIAPIPE_LEFT_KNEE,
    MEDIAPIPE_LEFT_SHOULDER,
    MEDIAPIPE_NOSE,
    MEDIAPIPE_RIGHT_ANKLE,
    MEDIAPIPE_RIGHT_EYE,
    MEDIAPIPE_RIGHT_HIP,
    MEDIAPIPE_RIGHT_KNEE,
    MEDIAPIPE_RIGHT_SHOULDER,
    MIN_MEDIAPIPE_VISIBILITY,
)
from reid_crops import clamp_box_to_frame


def assess_reid_body_completeness(
    landmarks,
    min_visibility=MIN_MEDIAPIPE_VISIBILITY,
    normalized_bounds=(0.0, 0.0, 1.0, 1.0),
    orientation=None,
    touches_vertical_frame_boundary=False,
    debug_details=None,
):
    """Return whether every major body region is visible inside the ReID crop.

    Only one landmark per paired region is required so side views and a
    single occluded limb can still qualify. ``normalized_bounds`` describes
    the saved ReID crop inside the larger image passed to MediaPipe.
    """
    body_regions = {
        "shoulders": (MEDIAPIPE_LEFT_SHOULDER, MEDIAPIPE_RIGHT_SHOULDER),
        "hips": (MEDIAPIPE_LEFT_HIP, MEDIAPIPE_RIGHT_HIP),
        "knees": (MEDIAPIPE_LEFT_KNEE, MEDIAPIPE_RIGHT_KNEE),
    }
    left, top, right, bottom = map(float, normalized_bounds)
    landmark_names = {
        MEDIAPIPE_NOSE: "nose",
        MEDIAPIPE_LEFT_EYE: "left_eye",
        MEDIAPIPE_RIGHT_EYE: "right_eye",
        MEDIAPIPE_LEFT_SHOULDER: "left_shoulder",
        MEDIAPIPE_RIGHT_SHOULDER: "right_shoulder",
        MEDIAPIPE_LEFT_HIP: "left_hip",
        MEDIAPIPE_RIGHT_HIP: "right_hip",
        MEDIAPIPE_LEFT_KNEE: "left_knee",
        MEDIAPIPE_RIGHT_KNEE: "right_knee",
        MEDIAPIPE_LEFT_ANKLE: "left_ankle",
        MEDIAPIPE_RIGHT_ANKLE: "right_ankle",
    }
    landmark_debug = {}

    def landmark_is_usable(index):
        if index in landmark_debug:
            return bool(landmark_debug[index]["usable"])
        details = {
            "index": int(index),
            "visibility": None,
            "presence": None,
            "x": None,
            "y": None,
            "finite": False,
            "within_saved_crop": False,
            "usable": False,
        }
        if landmarks is None or index >= len(landmarks):
            landmark_debug[index] = details
            return False
        landmark = landmarks[index]
        visibility = float(getattr(landmark, "visibility", 0.0))
        presence = getattr(landmark, "presence", None)
        details["visibility"] = visibility
        if presence is not None:
            presence = float(presence)
            details["presence"] = presence
        try:
            x = float(landmark.x)
            y = float(landmark.y)
        except (AttributeError, TypeError, ValueError):
            landmark_debug[index] = details
            return False
        details["x"] = x
        details["y"] = y
        details["finite"] = bool(np.isfinite(x) and np.isfinite(y))
        details["within_saved_crop"] = bool(
            details["finite"] and left <= x <= right and top <= y <= bottom
        )
        details["usable"] = bool(
            np.isfinite(visibility)
            and visibility >= float(min_visibility)
            and (presence is None or (np.isfinite(presence) and presence >= float(min_visibility)))
            and details["within_saved_crop"]
        )
        landmark_debug[index] = details
        return details["usable"]

    nose_usable = landmark_is_usable(MEDIAPIPE_NOSE)
    eye_usable = any(
        landmark_is_usable(index)
        for index in (MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE)
    )
    if orientation in {"front", "left_side", "right_side"}:
        head_usable = nose_usable and eye_usable
    else:
        head_usable = nose_usable or eye_usable

    missing_regions = []
    if not head_usable:
        missing_regions.append("head")
    missing_regions.extend(
        region_name
        for region_name, indices in body_regions.items()
        if not any(landmark_is_usable(index) for index in indices)
    )
    if not any(
        landmark_is_usable(index)
        for index in (MEDIAPIPE_LEFT_ANKLE, MEDIAPIPE_RIGHT_ANKLE)
    ):
        missing_regions.append("ankles")
    if touches_vertical_frame_boundary:
        missing_regions.append("frame_boundary")
    missing_regions = tuple(missing_regions)
    body_complete = not missing_regions
    if debug_details is not None:
        debug_details.clear()
        debug_details.update(
            {
                "body_complete": body_complete,
                "missing_regions": missing_regions,
                "orientation": orientation,
                "minimum_visibility": float(min_visibility),
                "saved_crop_normalized_bounds": (left, top, right, bottom),
                "touches_vertical_frame_boundary": bool(touches_vertical_frame_boundary),
                "landmarks": {
                    landmark_names.get(index, str(index)): details
                    for index, details in landmark_debug.items()
                },
            }
        )
    return body_complete, missing_regions


def detection_touches_vertical_frame_boundary(
    frame,
    box,
    margin_pixels=DEFAULT_REID_FRAME_EDGE_MARGIN_PIXELS,
):
    if frame is None or box is None or len(box) < 4:
        return False
    frame_height = int(frame.shape[0])
    if frame_height <= 0:
        return False
    try:
        y1 = float(box[1])
        y2 = float(box[3])
    except (TypeError, ValueError):
        return False
    margin = max(0.0, float(margin_pixels))
    return bool(y1 <= margin or y2 >= (frame_height - 1.0 - margin))


def _reid_crop_box(frame, box):
    """Return the exact padded and frame-clamped bounds used by crop_person."""
    x1, y1, x2, y2 = map(float, box)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    return clamp_box_to_frame(
        (
            x1 - box_width * DEFAULT_REID_CROP_SIDE_PADDING,
            y1 - box_height * DEFAULT_REID_CROP_TOP_PADDING,
            x2 + box_width * DEFAULT_REID_CROP_SIDE_PADDING,
            y2 + box_height * DEFAULT_REID_CROP_BOTTOM_PADDING,
        ),
        frame,
    )


def _reid_crop_intruder(
    frame,
    box,
    boxes,
    current_index,
    track_ids=None,
    suppressed_by_index=None,
    max_intruder_area_ratio=DEFAULT_REID_CROP_MAX_INTRUDER_AREA_RATIO,
    max_behind_intruder_area_ratio=DEFAULT_REID_CROP_MAX_BEHIND_INTRUDER_AREA_RATIO,
    behind_foot_margin=DEFAULT_REID_CROP_BEHIND_FOOT_MARGIN,
):
    """Return the largest real-person intrusion into a padded ReID crop.

    Two people can cover the same share of the crop and do very different
    damage to it.  One standing in front replaces the body the crop exists to
    show; one standing behind is himself hidden by the subject, so most of the
    box being measured is the subject's own pixels.  The second is charged
    against a looser budget rather than waved through, because a person behind
    and offset to one side still shows a limb beside the subject.

    Depth is read from the box bottom: with everyone standing on one floor, the
    person whose feet sit higher in the image is the one further from the
    camera.
    """
    reid_crop_box = _reid_crop_box(frame, box)
    if reid_crop_box is None:
        return None

    crop_x1, crop_y1, crop_x2, crop_y2 = map(float, reid_crop_box)
    crop_area = max(0.0, crop_x2 - crop_x1) * max(0.0, crop_y2 - crop_y1)
    if crop_area <= 0.0:
        return None

    current_box = np.asarray(box, dtype=float)
    subject_bottom = float(current_box[3])
    subject_height = max(0.0, float(current_box[3]) - float(current_box[1]))
    # A subject clipped by someone in front reports a bottom above their real
    # feet.  That can only make a genuine behind-intruder look nearer and draw
    # the stricter budget, so the error costs a crop rather than admitting a
    # contaminated one.
    behind_cutoff = subject_bottom - float(behind_foot_margin) * subject_height
    largest_intruder = None
    for intruder_index, intruder_box in enumerate(boxes):
        if intruder_index == current_index:
            continue
        if np.array_equal(np.asarray(intruder_box, dtype=float), current_box):
            continue
        if (
            suppressed_by_index is not None
            and intruder_index < len(suppressed_by_index)
            and suppressed_by_index[intruder_index]
        ):
            continue

        other_x1, other_y1, other_x2, other_y2 = map(float, intruder_box)
        intersection_width = max(
            0.0,
            min(crop_x2, other_x2) - max(crop_x1, other_x1),
        )
        intersection_height = max(
            0.0,
            min(crop_y2, other_y2) - max(crop_y1, other_y1),
        )
        intruder_area_ratio = (
            intersection_width * intersection_height / crop_area
        )
        intruder_is_behind = other_y2 < behind_cutoff
        allowed_area_ratio = float(
            max_behind_intruder_area_ratio
            if intruder_is_behind
            else max_intruder_area_ratio
        )
        if intruder_area_ratio <= allowed_area_ratio:
            continue

        intruder_track_id = (
            int(track_ids[intruder_index])
            if track_ids is not None and intruder_index < len(track_ids)
            else None
        )
        if (
            largest_intruder is None
            or intruder_area_ratio > largest_intruder[1]
        ):
            largest_intruder = (
                intruder_track_id,
                float(intruder_area_ratio),
                intruder_index,
                "behind" if intruder_is_behind else "in_front",
                allowed_area_ratio,
            )
    return largest_intruder


def _reid_bounds_inside_pose_crop(frame, box, pose_crop_box):
    """Express the actual saved ReID crop as normalized pose-crop bounds."""
    reid_crop_box = _reid_crop_box(frame, box)
    if reid_crop_box is None:
        return 0.0, 0.0, 1.0, 1.0

    pose_x1, pose_y1, pose_x2, pose_y2 = map(float, pose_crop_box)
    pose_width = max(1.0, pose_x2 - pose_x1)
    pose_height = max(1.0, pose_y2 - pose_y1)
    reid_x1, reid_y1, reid_x2, reid_y2 = map(float, reid_crop_box)
    return (
        max(0.0, min(1.0, (reid_x1 - pose_x1) / pose_width)),
        max(0.0, min(1.0, (reid_y1 - pose_y1) / pose_height)),
        max(0.0, min(1.0, (reid_x2 - pose_x1) / pose_width)),
        max(0.0, min(1.0, (reid_y2 - pose_y1) / pose_height)),
    )


def detection_bounds_inside_reid_crop(frame, box):
    """Express the tight detection box as normalized saved-ReID-crop bounds.

    The saved crop is deliberately padded -- 10% below the box especially, so a
    foot-occluded person still shows their shoes to the appearance model.  That
    padding is wrong for MiVOLO, which was trained on the detector's own tight
    box, so the demographics stage needs a way back to it from the crop alone.
    """
    reid_crop_box = _reid_crop_box(frame, box)
    if reid_crop_box is None:
        return None

    crop_x1, crop_y1, crop_x2, crop_y2 = map(float, reid_crop_box)
    crop_width = max(1.0, crop_x2 - crop_x1)
    crop_height = max(1.0, crop_y2 - crop_y1)
    box_x1, box_y1, box_x2, box_y2 = map(float, box)
    bounds = (
        max(0.0, min(1.0, (box_x1 - crop_x1) / crop_width)),
        max(0.0, min(1.0, (box_y1 - crop_y1) / crop_height)),
        max(0.0, min(1.0, (box_x2 - crop_x1) / crop_width)),
        max(0.0, min(1.0, (box_y2 - crop_y1) / crop_height)),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    return bounds


def occluder_bounds_inside_reid_crop(
    frame,
    box,
    boxes,
    current_index,
    suppressed_by_index=None,
):
    """Normalized bounds of every other person overlapping the saved crop.

    The intruder gate above rejects a crop only once a neighbour covers enough
    of it to spoil an appearance match; a smaller neighbour is admitted and
    still shows up as a second person's clothing in the body branch.  MiVOLO's
    own pipeline blanks those pixels rather than tolerating them, and it can do
    so here because the boxes are already known at intake time.

    Suppressed shadow detections are skipped for the same reason the intruder
    gate skips them: they are unresolved duplicates of the subject, so blanking
    them would blank the subject.
    """
    reid_crop_box = _reid_crop_box(frame, box)
    if reid_crop_box is None:
        return []

    crop_x1, crop_y1, crop_x2, crop_y2 = map(float, reid_crop_box)
    crop_width = max(1.0, crop_x2 - crop_x1)
    crop_height = max(1.0, crop_y2 - crop_y1)
    current_box = np.asarray(box, dtype=float)

    occluders = []
    for other_index, other_box in enumerate(boxes):
        if other_index == current_index:
            continue
        if np.array_equal(np.asarray(other_box, dtype=float), current_box):
            continue
        if (
            suppressed_by_index is not None
            and other_index < len(suppressed_by_index)
            and suppressed_by_index[other_index]
        ):
            continue

        other_x1, other_y1, other_x2, other_y2 = map(float, other_box)
        left = max(0.0, min(1.0, (other_x1 - crop_x1) / crop_width))
        top = max(0.0, min(1.0, (other_y1 - crop_y1) / crop_height))
        right = max(0.0, min(1.0, (other_x2 - crop_x1) / crop_width))
        bottom = max(0.0, min(1.0, (other_y2 - crop_y1) / crop_height))
        if right <= left or bottom <= top:
            continue
        occluders.append((float(left), float(top), float(right), float(bottom)))
    return occluders
