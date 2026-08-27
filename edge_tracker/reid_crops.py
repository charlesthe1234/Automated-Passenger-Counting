"""Turning a detection box into the crop the ReID models see.

The padding, clamping, and sharpness rules here decide what every later
appearance decision is made from, so they are kept together and away from
the identity bookkeeping that consumes them."""

import cv2
import numpy as np

from constants import (
    DEFAULT_REID_CROP_BOTTOM_PADDING,
    DEFAULT_REID_CROP_SIDE_PADDING,
    DEFAULT_REID_CROP_TOP_PADDING,
)


def clamp_box_to_frame(box, frame):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(frame_width - 1, int(x1)))
    y1 = max(0, min(frame_height - 1, int(y1)))
    x2 = max(0, min(frame_width, int(x2)))
    y2 = max(0, min(frame_height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_person(
    frame,
    box,
    side_padding=DEFAULT_REID_CROP_SIDE_PADDING,
    top_padding=DEFAULT_REID_CROP_TOP_PADDING,
    bottom_padding=DEFAULT_REID_CROP_BOTTOM_PADDING,
):
    x1, y1, x2, y2 = map(float, box)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    padded_box = (
        x1 - width * float(side_padding),
        y1 - height * float(top_padding),
        x2 + width * float(side_padding),
        y2 + height * float(bottom_padding),
    )
    clamped = clamp_box_to_frame(padded_box, frame)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    return frame[y1:y2, x1:x2]


def image_sharpness(crop):
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_color_reid_feature(crop):
    if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
        return None

    resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    upper = hsv[8:70, :]
    lower = hsv[58:124, :]
    hist_upper = cv2.calcHist([upper], [0, 1], None, [24, 16], [0, 180, 0, 256]).flatten()
    hist_lower = cv2.calcHist([lower], [0, 1], None, [24, 16], [0, 180, 0, 256]).flatten()
    feature = np.concatenate([hist_upper * 0.65, hist_lower * 0.35]).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-6:
        return None
    return feature / norm
