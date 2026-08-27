"""Estimating where a person actually meets the floor.

The map position depends on this point, so each strategy -- MediaPipe feet,
YOLO ankles, box bottom -- reports how much it should be trusted rather
than silently returning a guess."""

import numpy as np

from constants import (
    DEFAULT_ANATOMICAL_RATIO_EMA_ALPHA,
    DEFAULT_MAX_FOOT_JUMP_PIXELS_PER_FRAME,
    DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    DEFAULT_REID_FRAME_EDGE_MARGIN_PIXELS,
    LEFT_ANKLE_KEYPOINT_INDEX,
    MIN_ANKLE_CONFIDENCE,
    RIGHT_ANKLE_KEYPOINT_INDEX,
)
from core_math import (
    calculate_anatomical_anchor_pixels,
    calculate_anatomical_ratio,
    estimate_virtual_foot_from_ratio,
    get_anatomical_anchor_from_memory,
    get_anatomical_ratio_from_memory,
    recall_recent_foot_point,
    reject_impossible_foot_jump,
    remember_foot_point,
    store_anatomical_anchor,
    store_anatomical_ratio,
)
from identity_debug import identity_event

from human_orientation import (
    estimate_head_pitch,
    get_human_orientation,
)
from mediapipe_landmarks import (
    draw_mediapipe_skeleton,
    extract_mediapipe_body_points,
    extract_metrology_landmarks,
)
from face_region import estimate_face_box
from reid_crop_quality import (
    _reid_bounds_inside_pose_crop,
    assess_reid_body_completeness,
    detection_touches_vertical_frame_boundary,
)


def classify_box_bottom_evidence(
    frame,
    box,
    other_boxes=(),
    margin_pixels=DEFAULT_REID_FRAME_EDGE_MARGIN_PIXELS,
):
    """Say whether a detection's box bottom is a trustworthy ground point.

    Cross-camera association physics deliberately uses the raw box bottom rather
    than a smoothed pose foot, so that a tracker-ID teleport cannot hide behind
    filtering.  But the box bottom is only the person's feet when the feet are
    actually in the box.  Two situations break that, and both happen constantly
    in a crowd:

    * the box is clipped by the frame edge, so its bottom is the edge, not a foot
    * another person stands in front, so the box bottom is their body

    Returns ("hard", None) when the bottom can be trusted, else ("soft", reason).
    A soft ground point is still perfectly useful for scoring; it just must not
    be allowed to veto a confident appearance match.
    """
    if frame is None or box is None or len(box) < 4:
        return "soft", "no_detection_box"
    try:
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    except (TypeError, ValueError):
        return "soft", "no_detection_box"

    frame_height = int(frame.shape[0])
    frame_width = int(frame.shape[1])
    margin = max(0.0, float(margin_pixels))
    if frame_height > 0 and y2 >= (frame_height - 1.0 - margin):
        return "soft", "box_clipped_by_frame_bottom"
    if frame_width > 0 and (x1 <= margin or x2 >= (frame_width - 1.0 - margin)):
        # A horizontally clipped person may have their feet outside the frame
        # even though the box bottom sits well above the lower edge.
        return "soft", "box_clipped_by_frame_side"

    height = y2 - y1
    if height <= 0:
        return "soft", "degenerate_box"
    # Only the lower part of the body decides whether the feet are visible.
    foot_band_top = y2 - 0.25 * height
    # `or ()` would raise on a NumPy array, which is exactly what the tracker
    # passes here, so the empty case is handled explicitly.
    if other_boxes is None:
        other_boxes = ()
    for other in other_boxes:
        if other is None or len(other) < 4:
            continue
        try:
            ox1, oy1, ox2, oy2 = (float(other[0]), float(other[1]), float(other[2]), float(other[3]))
        except (TypeError, ValueError):
            continue
        if (ox1, oy1, ox2, oy2) == (x1, y1, x2, y2):
            continue
        overlap_width = min(x2, ox2) - max(x1, ox1)
        overlap_height = min(y2, oy2) - max(foot_band_top, oy1)
        if overlap_width <= 0 or overlap_height <= 0:
            continue
        band_area = max(1.0, (x2 - x1) * (y2 - foot_band_top))
        if (overlap_width * overlap_height) / band_area >= 0.20:
            return "soft", "feet_occluded_by_other_detection"
    return "hard", None


def run_three_d_level_shadow(
    estimator,
    pose_debug,
    method,
    camera_id,
    track_id,
    identity_id,
    identity_state,
    frame_index,
    map_projector,
    production_point=None,
):
    """Learn metric landmark heights and log a shadow position estimate.

    SHADOW MODE: this deliberately returns its result for logging only. The
    production ground point is not replaced, so enabling the checkbox cannot
    change tracking behaviour until the recorded errors justify it.

    Never raises into the tracking loop: an experimental geometry bug must not
    be able to stop a live evacuation run.
    """
    if estimator is None or not estimator.ready:
        return None
    landmarks = (pose_debug or {}).get("metrology_landmarks") or {}
    if not landmarks:
        return None
    try:
        strict_foot = (pose_debug or {}).get("strict_foot_point")
        # Only a strict, high-visibility foot observation is trusted to teach a
        # height. A held, smoothed or extrapolated point would bake its own
        # error into a long-lived per-identity memory.
        foot_is_direct = method == "mediapipe" and strict_foot is not None
        observation = {
            "camera_id": camera_id,
            "identity_id": identity_id,
            "foot_pixel": strict_foot,
            "landmarks": landmarks,
            "foot_is_direct": foot_is_direct,
            "identity_stable": identity_id is not None and identity_state == "confirmed",
        }
        learned = estimator.learn_landmark_heights(observation)
        best, candidates = estimator.estimate_ground_position(observation, identity_state)

        def to_map(pixel):
            if pixel is None or map_projector is None:
                return None
            projected = map_projector((float(pixel[0]), float(pixel[1])))
            return None if projected is None else (float(projected[0]), float(projected[1]))

        direct_map_point = to_map(strict_foot) if foot_is_direct else None
        # The position the tracker is actually using this frame. Without it the
        # log cannot score the existing anatomical-ratio fallback against either
        # the direct-foot truth or the metrology estimate, which is the entire
        # comparison the shadow run exists to make.
        production_map_point = to_map(production_point)

        # A frame with both a direct foot point and a metrology estimate is the
        # only place the fallback can be scored against ground truth.
        error_cm = None
        if direct_map_point is not None and best.point_cm is not None:
            error_cm = float(
                np.hypot(
                    best.point_cm[0] - direct_map_point[0],
                    best.point_cm[1] - direct_map_point[1],
                )
            )

        # Deliberately not throttled. The whole point of shadow mode is to score
        # every fallback frame against the direct-foot truth, and the required
        # metrics -- p90, p95, frame-to-frame jitter -- cannot be recovered from
        # a subsampled trace. The event only exists when the opt-in identity log
        # is enabled, so this costs an ordinary run nothing.
        identity_event(
            "three_d_level_shadow",
            console=False,
            camera_id=camera_id,
            track_key=(camera_id, track_id),
            identity_id=identity_id,
            identity_state=identity_state,
            frame_index=frame_index,
            production_method=method,
            foot_is_direct=foot_is_direct,
            direct_map_point=direct_map_point,
            production_map_point=production_map_point,
            learned_heights_cm={name: round(value, 1) for name, value in learned.items()},
            known_heights_cm={
                name: round(value, 1) for name, value in estimator.known_heights(identity_id).items()
            },
            best_estimate=best.as_log_dict(),
            candidates=[candidate.as_log_dict() for candidate in candidates],
            error_vs_direct_foot_cm=error_cm,
        )
        return best
    except Exception as error:  # experimental code must never break tracking
        identity_event(
            "three_d_level_shadow_failed",
            throttle_key=(camera_id, "error"),
            throttle_seconds=5.0,
            camera_id=camera_id,
            reason=str(error),
        )
        return None


def estimate_mediapipe_foot_point(
    frame,
    box,
    pose_estimator,
    anatomical_ratio_memory=None,
    anatomical_anchor_memory=None,
    last_foot_memory=None,
    track_id=None,
    identity_id=None,
    frame_index=None,
    pose_dropout_ttl_frames=DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    ratio_ema_alpha=DEFAULT_ANATOMICAL_RATIO_EMA_ALPHA,
    max_foot_jump_pixels_per_frame=DEFAULT_MAX_FOOT_JUMP_PIXELS_PER_FRAME,
    annotated_frame=None,
    pose_debug=None,
    collect_metrology_landmarks=False,
):
    if pose_debug is not None:
        body_details = {}
        body_complete, missing_regions = assess_reid_body_completeness(
            None,
            debug_details=body_details,
        )
        pose_debug["reid_body_complete"] = body_complete
        pose_debug["reid_missing_regions"] = missing_regions
        pose_debug["reid_body_details"] = body_details
        pose_debug["reid_face_box"] = None
    if pose_estimator is None:
        return None, "no_pose"

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    box_width = x2 - x1
    box_height = y2 - y1
    padding_x = box_width * 0.12
    padding_y = box_height * 0.08

    crop_x1 = max(0, int(x1 - padding_x))
    crop_y1 = max(0, int(y1 - padding_y))
    crop_x2 = min(frame_width, int(x2 + padding_x))
    crop_y2 = min(frame_height, int(y2 + padding_y))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        last_foot = recall_recent_foot_point(last_foot_memory, track_id, frame_index, pose_dropout_ttl_frames, identity_id=identity_id)
        if last_foot is not None:
            return last_foot, "last_seen"
        return None, "invalid_crop"

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    result = pose_estimator.detect(crop)
    if not result.pose_landmarks:
        last_foot = recall_recent_foot_point(last_foot_memory, track_id, frame_index, pose_dropout_ttl_frames, identity_id=identity_id)
        if last_foot is not None:
            return last_foot, "last_seen"
        return None, "no_visible_ankle"

    landmarks = result.pose_landmarks[0]
    if pose_debug is not None:
        pose_debug["orientation"] = get_human_orientation(landmarks)
    crop_height, crop_width = crop.shape[:2]
    if pose_debug is not None:
        reid_bounds = _reid_bounds_inside_pose_crop(
            frame,
            box,
            (crop_x1, crop_y1, crop_x2, crop_y2),
        )
        body_details = {}
        body_complete, missing_regions = assess_reid_body_completeness(
            landmarks,
            normalized_bounds=reid_bounds,
            orientation=pose_debug.get("orientation"),
            touches_vertical_frame_boundary=detection_touches_vertical_frame_boundary(frame, box),
            debug_details=body_details,
        )
        pose_debug["reid_body_complete"] = body_complete
        pose_debug["reid_missing_regions"] = missing_regions
        pose_debug["reid_body_details"] = body_details
        # Recorded against the saved ReID crop rather than this larger pose
        # crop, so the demographics stage can find the face in the only image
        # it will still have.
        pose_debug["reid_face_box"] = estimate_face_box(
            landmarks,
            normalized_bounds=reid_bounds,
            crop_width=crop_width,
            crop_height=crop_height,
        )
    if annotated_frame is not None:
        draw_mediapipe_skeleton(annotated_frame, landmarks, crop_width, crop_height, crop_x1, crop_y1)

    nose, shoulder, foot_point, strict_foot_point, left_shoulder, right_shoulder = extract_mediapipe_body_points(
        landmarks,
        crop_width,
        crop_height,
        crop_x1,
        crop_y1,
    )
    if collect_metrology_landmarks and pose_debug is not None:
        # EXPERIMENTAL: consumed only by 3D level detection, and skipped entirely
        # when that feature is off so a normal run does no extra landmark work.
        pose_debug["metrology_landmarks"] = extract_metrology_landmarks(
            landmarks, crop_width, crop_height, crop_x1, crop_y1
        )
        pose_debug["strict_foot_point"] = (
            None if strict_foot_point is None
            else (float(strict_foot_point[0]), float(strict_foot_point[1]))
        )
    head_pitch = estimate_head_pitch(landmarks, crop_width, crop_height, crop_x1, crop_y1)
    if pose_debug is not None:
        pose_debug["head_pitch"] = head_pitch
    head_allows_ratio_update = head_pitch == "looking_straight"

    stored_ratio = get_anatomical_ratio_from_memory(anatomical_ratio_memory, track_id, identity_id=identity_id)
    stored_anchor = get_anatomical_anchor_from_memory(anatomical_anchor_memory, track_id, identity_id=identity_id)

    # The loose foot point survives at MIN_MEDIAPIPE_VISIBILITY while the strict
    # one demands MIN_INITIAL_FOOT_VISIBILITY.  When only the loose one exists,
    # MediaPipe has placed a foot it is not confident about -- half an ankle
    # behind someone's leg, or a sole guessed in shadow.  That point is present,
    # so every downstream check treats it as a measurement, and it is the
    # single largest source of a map dot that will not settle.  A learned body
    # ratio built from frames where the feet *were* clearly visible beats it.
    low_confidence_foot = foot_point is not None and strict_foot_point is None
    if pose_debug is not None:
        pose_debug["strict_foot_visible"] = strict_foot_point is not None

    if stored_ratio is not None and (head_pitch == "looking_down" or low_confidence_foot):
        virtual_foot = estimate_virtual_foot_from_ratio(
            nose,
            shoulder,
            stored_ratio,
            # A lowered head shortens the live nose-to-shoulder span, so the
            # learned anchor stands in.  A merely uncertain foot does not
            # distort the head at all, so the live span is still the truth.
            anchor_pixels_override=stored_anchor if head_pitch == "looking_down" else None,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
        )
        if virtual_foot is not None:
            held_foot = reject_impossible_foot_jump(
                last_foot_memory,
                track_id,
                virtual_foot,
                frame_index,
                max_foot_jump_pixels_per_frame,
                identity_id=identity_id,
            )
            if held_foot is not None:
                return held_foot, "physics_hold"
            remember_foot_point(last_foot_memory, track_id, virtual_foot, frame_index, identity_id=identity_id)
            return virtual_foot, "anatomical_ratio"

    if foot_point is not None:
        held_foot = reject_impossible_foot_jump(
            last_foot_memory,
            track_id,
            foot_point,
            frame_index,
            max_foot_jump_pixels_per_frame,
            identity_id=identity_id,
        )
        if held_foot is not None:
            return held_foot, "physics_hold"
        if strict_foot_point is not None and head_allows_ratio_update:
            ratio = calculate_anatomical_ratio(nose, shoulder, strict_foot_point)
            anchor_pixels = calculate_anatomical_anchor_pixels(nose, shoulder)
            if ratio is not None:
                store_anatomical_ratio(
                    anatomical_ratio_memory,
                    track_id,
                    ratio,
                    identity_id=identity_id,
                    ema_alpha=ratio_ema_alpha,
                )
            if anchor_pixels is not None:
                store_anatomical_anchor(
                    anatomical_anchor_memory,
                    track_id,
                    anchor_pixels,
                    identity_id=identity_id,
                    ema_alpha=ratio_ema_alpha,
                )
        remember_foot_point(last_foot_memory, track_id, foot_point, frame_index, identity_id=identity_id)
        return foot_point, "mediapipe"

    if stored_ratio is not None:
        virtual_foot = estimate_virtual_foot_from_ratio(
            nose,
            shoulder,
            stored_ratio,
            anchor_pixels_override=stored_anchor if head_pitch == "looking_down" else None,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder,
        )
        if virtual_foot is not None:
            held_foot = reject_impossible_foot_jump(
                last_foot_memory,
                track_id,
                virtual_foot,
                frame_index,
                max_foot_jump_pixels_per_frame,
                identity_id=identity_id,
            )
            if held_foot is not None:
                return held_foot, "physics_hold"
            remember_foot_point(last_foot_memory, track_id, virtual_foot, frame_index, identity_id=identity_id)
            return virtual_foot, "anatomical_ratio"

    last_foot = recall_recent_foot_point(last_foot_memory, track_id, frame_index, pose_dropout_ttl_frames, identity_id=identity_id)
    if last_foot is not None:
        return last_foot, "last_seen"

    return None, "no_visible_ankle"


def estimate_yolo_pose_ankle_point(index, keypoint_xy, keypoint_conf):
    if keypoint_xy is None or index >= len(keypoint_xy):
        return None

    left_ankle = keypoint_xy[index][LEFT_ANKLE_KEYPOINT_INDEX]
    right_ankle = keypoint_xy[index][RIGHT_ANKLE_KEYPOINT_INDEX]
    ankles = []

    if keypoint_conf is None or keypoint_conf[index][LEFT_ANKLE_KEYPOINT_INDEX] >= MIN_ANKLE_CONFIDENCE:
        if not np.allclose(left_ankle, 0):
            ankles.append(left_ankle)

    if keypoint_conf is None or keypoint_conf[index][RIGHT_ANKLE_KEYPOINT_INDEX] >= MIN_ANKLE_CONFIDENCE:
        if not np.allclose(right_ankle, 0):
            ankles.append(right_ankle)

    if ankles:
        return np.mean(np.array(ankles), axis=0)

    return None


def estimate_box_bottom_point(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2, y2], dtype=float)
