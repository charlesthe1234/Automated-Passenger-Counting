"""Which way a person is facing, and how far their head is pitched."""

import numpy as np

from constants import (
    HEAD_DOWN_ANCHOR_FRACTION,
    MEDIAPIPE_LEFT_EYE,
    MEDIAPIPE_LEFT_SHOULDER,
    MEDIAPIPE_NOSE,
    MEDIAPIPE_RIGHT_EYE,
    MEDIAPIPE_RIGHT_SHOULDER,
    MIN_ANATOMICAL_ANCHOR_PIXELS,
    MIN_HEAD_DOWN_PIXELS,
    MIN_MEDIAPIPE_VISIBILITY,
)

from mediapipe_landmarks import (
    mean_available_points,
    visible_mediapipe_point,
)


def get_human_orientation(landmarks):
    """Return one of the four semantic gallery views for clear body poses.

    The calibrated screen-space shoulder ratio first determines whether the
    subject can genuinely be side-on. MediaPipe Z depth is only trusted once
    that gate passes, which avoids classifying a wide front/back pose as a
    side view.
    """
    if not landmarks or len(landmarks) <= 24:
        return None

    left_shoulder = landmarks[MEDIAPIPE_LEFT_SHOULDER]
    right_shoulder = landmarks[MEDIAPIPE_RIGHT_SHOULDER]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    if (
        left_shoulder.visibility < MIN_MEDIAPIPE_VISIBILITY
        or right_shoulder.visibility < MIN_MEDIAPIPE_VISIBILITY
        or left_hip.visibility < MIN_MEDIAPIPE_VISIBILITY
        or right_hip.visibility < MIN_MEDIAPIPE_VISIBILITY
    ):
        return None

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    average_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    average_hip_y = (left_hip.y + right_hip.y) / 2.0
    torso_height = max(0.01, abs(average_hip_y - average_shoulder_y))
    shoulder_ratio = shoulder_width / torso_height
    depth_difference = left_shoulder.z - right_shoulder.z

    if shoulder_ratio < 0.50:
        if depth_difference < -0.75:
            return "left_side"
        if depth_difference > 0.75:
            return "right_side"
        if shoulder_ratio < 0.15:
            if abs(depth_difference) < 0.05:
                return None
            return "left_side" if depth_difference < 0 else "right_side"

    if shoulder_ratio < 1.80:
        return None
    return "front" if left_shoulder.x > right_shoulder.x else "back"


def estimate_head_pitch(landmarks, crop_width, crop_height, offset_x, offset_y):
    nose = visible_mediapipe_point(landmarks, MEDIAPIPE_NOSE, crop_width, crop_height, offset_x, offset_y)
    left_eye = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_EYE, crop_width, crop_height, offset_x, offset_y)
    right_eye = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_EYE, crop_width, crop_height, offset_x, offset_y)
    left_shoulder = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_SHOULDER, crop_width, crop_height, offset_x, offset_y)
    right_shoulder = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_SHOULDER, crop_width, crop_height, offset_x, offset_y)

    eye_center = mean_available_points([left_eye, right_eye])
    shoulder = mean_available_points([left_shoulder, right_shoulder])
    if nose is None or eye_center is None or shoulder is None:
        return "unknown"

    anchor_pixels = float(np.linalg.norm(shoulder - nose))
    if anchor_pixels < MIN_ANATOMICAL_ANCHOR_PIXELS:
        return "unknown"

    nose_below_eyes = float(nose[1] - eye_center[1])
    down_threshold = max(MIN_HEAD_DOWN_PIXELS, anchor_pixels * HEAD_DOWN_ANCHOR_FRACTION)
    if nose_below_eyes > down_threshold:
        return "looking_down"
    return "looking_straight"
