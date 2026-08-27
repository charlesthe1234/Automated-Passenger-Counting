"""Reading MediaPipe pose landmarks: visibility filtering, image-space conversion,
and the body-point sets the rest of the pipeline works from."""

import cv2
import numpy as np

try:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
except ImportError:
    BaseOptions = None
    vision = None

from constants import (
    MEDIAPIPE_LEFT_ANKLE,
    MEDIAPIPE_LEFT_FOOT_INDEX,
    MEDIAPIPE_LEFT_HEEL,
    MEDIAPIPE_LEFT_HIP,
    MEDIAPIPE_LEFT_KNEE,
    MEDIAPIPE_LEFT_SHOULDER,
    MEDIAPIPE_NOSE,
    MEDIAPIPE_RIGHT_ANKLE,
    MEDIAPIPE_RIGHT_FOOT_INDEX,
    MEDIAPIPE_RIGHT_HEEL,
    MEDIAPIPE_RIGHT_HIP,
    MEDIAPIPE_RIGHT_KNEE,
    MEDIAPIPE_RIGHT_SHOULDER,
    MIN_INITIAL_FOOT_VISIBILITY,
    MIN_MEDIAPIPE_VISIBILITY,
)


def visible_mediapipe_point(landmarks, index, crop_width, crop_height, offset_x, offset_y):
    landmark = landmarks[index]
    if landmark.visibility < MIN_MEDIAPIPE_VISIBILITY:
        return None
    return np.array([
        offset_x + landmark.x * crop_width,
        offset_y + landmark.y * crop_height,
    ], dtype=float)


def mediapipe_landmark_visibility(landmarks, index):
    if landmarks is None or index >= len(landmarks):
        return 0.0
    return float(getattr(landmarks[index], "visibility", 0.0))


def mediapipe_point_with_min_visibility(landmarks, index, crop_width, crop_height, offset_x, offset_y, min_visibility):
    if mediapipe_landmark_visibility(landmarks, index) < min_visibility:
        return None
    landmark = landmarks[index]
    return np.array([
        offset_x + landmark.x * crop_width,
        offset_y + landmark.y * crop_height,
    ], dtype=float)


def mean_available_points(points):
    visible_points = [point for point in points if point is not None]
    if not visible_points:
        return None
    return np.mean(np.array(visible_points), axis=0)


def mediapipe_point_to_image(landmark, crop_width, crop_height, offset_x, offset_y):
    return (
        int(offset_x + landmark.x * crop_width),
        int(offset_y + landmark.y * crop_height),
    )


def draw_mediapipe_skeleton(annotated_frame, landmarks, crop_width, crop_height, offset_x, offset_y):
    if vision is None:
        return

    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        if start.visibility < MIN_MEDIAPIPE_VISIBILITY or end.visibility < MIN_MEDIAPIPE_VISIBILITY:
            continue

        start_point = mediapipe_point_to_image(start, crop_width, crop_height, offset_x, offset_y)
        end_point = mediapipe_point_to_image(end, crop_width, crop_height, offset_x, offset_y)
        cv2.line(annotated_frame, start_point, end_point, (0, 220, 0), 2)

    for landmark in landmarks:
        if landmark.visibility < MIN_MEDIAPIPE_VISIBILITY:
            continue

        point = mediapipe_point_to_image(landmark, crop_width, crop_height, offset_x, offset_y)
        cv2.circle(annotated_frame, point, 3, (0, 255, 255), -1)


def extract_mediapipe_body_points(landmarks, crop_width, crop_height, offset_x, offset_y):
    nose = visible_mediapipe_point(landmarks, MEDIAPIPE_NOSE, crop_width, crop_height, offset_x, offset_y)
    left_shoulder = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_SHOULDER, crop_width, crop_height, offset_x, offset_y)
    right_shoulder = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_SHOULDER, crop_width, crop_height, offset_x, offset_y)
    shoulder = mean_available_points([left_shoulder, right_shoulder])

    left_ankle = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_ANKLE, crop_width, crop_height, offset_x, offset_y)
    right_ankle = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_ANKLE, crop_width, crop_height, offset_x, offset_y)
    left_heel = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_HEEL, crop_width, crop_height, offset_x, offset_y)
    left_toe = visible_mediapipe_point(landmarks, MEDIAPIPE_LEFT_FOOT_INDEX, crop_width, crop_height, offset_x, offset_y)
    right_heel = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_HEEL, crop_width, crop_height, offset_x, offset_y)
    right_toe = visible_mediapipe_point(landmarks, MEDIAPIPE_RIGHT_FOOT_INDEX, crop_width, crop_height, offset_x, offset_y)

    foot_points = []
    strict_foot_points = []
    for ankle, heel, toe in ((left_ankle, left_heel, left_toe), (right_ankle, right_heel, right_toe)):
        if ankle is None:
            continue
        sole_point = mean_available_points([heel, toe])
        if sole_point is not None:
            foot_points.append(sole_point)
        else:
            foot_points.append(ankle)

    for ankle_index, heel_index, toe_index in (
        (MEDIAPIPE_LEFT_ANKLE, MEDIAPIPE_LEFT_HEEL, MEDIAPIPE_LEFT_FOOT_INDEX),
        (MEDIAPIPE_RIGHT_ANKLE, MEDIAPIPE_RIGHT_HEEL, MEDIAPIPE_RIGHT_FOOT_INDEX),
    ):
        ankle = mediapipe_point_with_min_visibility(
            landmarks, ankle_index, crop_width, crop_height, offset_x, offset_y, MIN_INITIAL_FOOT_VISIBILITY
        )
        heel = mediapipe_point_with_min_visibility(
            landmarks, heel_index, crop_width, crop_height, offset_x, offset_y, MIN_INITIAL_FOOT_VISIBILITY
        )
        toe = mediapipe_point_with_min_visibility(
            landmarks, toe_index, crop_width, crop_height, offset_x, offset_y, MIN_INITIAL_FOOT_VISIBILITY
        )
        if ankle is None:
            continue
        sole_point = mean_available_points([heel, toe])
        if sole_point is not None:
            strict_foot_points.append(sole_point)
        else:
            strict_foot_points.append(ankle)

    return (
        nose,
        shoulder,
        mean_available_points(foot_points),
        mean_available_points(strict_foot_points),
        left_shoulder,
        right_shoulder,
    )


def extract_metrology_landmarks(landmarks, crop_width, crop_height, offset_x, offset_y):
    """Pixel positions of the landmarks two-plane metrology can measure.

    EXPERIMENTAL: only read when 3D level detection is enabled. Deliberately
    separate from extract_mediapipe_body_points so the production foot pipeline
    keeps its exact behaviour, signature and test coverage.

    Hips and knees are collected here because they are far closer to the ground
    than the head: a height error or a lean costs proportionally less ground
    error the lower the landmark sits.
    """

    def centre(left_index, right_index):
        left = visible_mediapipe_point(landmarks, left_index, crop_width, crop_height, offset_x, offset_y)
        right = visible_mediapipe_point(landmarks, right_index, crop_width, crop_height, offset_x, offset_y)
        point = mean_available_points([left, right])
        return None if point is None else (float(point[0]), float(point[1]))

    nose = visible_mediapipe_point(landmarks, MEDIAPIPE_NOSE, crop_width, crop_height, offset_x, offset_y)
    found = {
        "knee_centre": centre(MEDIAPIPE_LEFT_KNEE, MEDIAPIPE_RIGHT_KNEE),
        "hip_centre": centre(MEDIAPIPE_LEFT_HIP, MEDIAPIPE_RIGHT_HIP),
        "shoulder_centre": centre(MEDIAPIPE_LEFT_SHOULDER, MEDIAPIPE_RIGHT_SHOULDER),
        "nose": None if nose is None else (float(nose[0]), float(nose[1])),
    }
    return {name: point for name, point in found.items() if point is not None}
