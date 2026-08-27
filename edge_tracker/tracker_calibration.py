"""Homography persistence and the interactive four-corner calibration flow."""

import cv2
import json
import numpy as np

from camera_stream import resize_to_fit
from constants import DEFAULT_TACTICAL_MAP_SIZE_CM
from core_math import extrapolate_fourth_corner


def save_homography(path, matrix, image_points, map_size_cm):
    payload = {
        "matrix": matrix.tolist(),
        "image_points": image_points.tolist(),
        "map_size_cm": map_size_cm,
        "point_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_homography(path, requested_map_size_cm=None):
    payload = json.loads(path.read_text(encoding="utf-8"))#encoding must be same as saved
    saved_map_size_cm = int(payload.get("map_size_cm", DEFAULT_TACTICAL_MAP_SIZE_CM))
    requested_map_size_cm = (
        saved_map_size_cm
        if requested_map_size_cm is None
        else int(requested_map_size_cm)
    )
    if requested_map_size_cm == saved_map_size_cm:
        return np.array(payload["matrix"], dtype=np.float32), saved_map_size_cm

    image_points = np.asarray(payload.get("image_points"), dtype=np.float32)
    if image_points.shape != (4, 2):
        raise RuntimeError(
            f"Calibration {path} was saved for {saved_map_size_cm} cm and cannot be "
            f"rescaled to {requested_map_size_cm} cm because its four image points are missing. "
            "Enable Setup calibration in the launcher."
        )
    map_points = np.array(
        [
            [0, 0],
            [requested_map_size_cm, 0],
            [requested_map_size_cm, requested_map_size_cm],
            [0, requested_map_size_cm],
        ],
        dtype=np.float32,
    )
    matrix, _ = cv2.findHomography(image_points, map_points)
    if matrix is None:
        raise RuntimeError(f"Unable to rescale calibration {path}.")
    save_homography(path, matrix, image_points, requested_map_size_cm)
    print(
        f"Updated {path} tactical-map scale from {saved_map_size_cm} cm "
        f"to {requested_map_size_cm} cm."
    )
    return matrix.astype(np.float32), requested_map_size_cm

def collect_calibration_points(frame, map_size_cm, matrix_path, missing_corner=None):
    clicked_points = []
    display_frame, scale = resize_to_fit(frame)
    corner_order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    clockwise_click_specs = {
        "top_left": [
            ("edge_from_next", "top edge near TL"),
            ("top_right", "TR"),
            ("bottom_right", "BR"),
            ("bottom_left", "BL"),
            ("edge_from_prev", "left edge near TL"),
        ],
        "top_right": [
            ("top_left", "TL"),
            ("edge_from_prev", "top edge near TR"),
            ("edge_from_next", "right edge near TR"),
            ("bottom_right", "BR"),
            ("bottom_left", "BL"),
        ],
        "bottom_right": [
            ("top_left", "TL"),
            ("top_right", "TR"),
            ("edge_from_prev", "right edge near BR"),
            ("edge_from_next", "bottom edge near BR"),
            ("bottom_left", "BL"),
        ],
        "bottom_left": [
            ("top_left", "TL"),
            ("top_right", "TR"),
            ("bottom_right", "BR"),
            ("edge_from_prev", "bottom edge near BL"),
            ("edge_from_next", "left edge near BL"),
        ],
    }
    if missing_corner is not None and missing_corner not in corner_order:
        raise ValueError(f"Missing corner must be one of: {', '.join(corner_order)}")

    click_specs = clockwise_click_specs.get(missing_corner)
    required_clicks = 5 if missing_corner else 4
    window_name = "Calibration: smart missing corner" if missing_corner else "Calibration: click TL, TR, BR, BL then press Enter"

    def build_missing_corner_points():
        known_corners = {}
        edge_points = {}
        for point, (point_key, _label) in zip(clicked_points, click_specs):
            if point_key.startswith("edge_"):
                edge_points[point_key] = point
            else:
                known_corners[point_key] = point
        missing_point = extrapolate_fourth_corner(
            known_corners,
            edge_points,
            missing_corner=missing_corner,
            allow_swapped_edges=False,
        )
        return known_corners, missing_point

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicked_points) >= required_clicks:
            return

        clicked_points.append([x / scale, y / scale])

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        preview = display_frame.copy()
        for index, (point_x, point_y) in enumerate(clicked_points):
            display_x = int(point_x * scale)
            display_y = int(point_y * scale)
            cv2.circle(preview, (display_x, display_y), 6, (0, 0, 255), -1)
            cv2.putText(
                preview,
                str(index + 1),
                (display_x + 8, display_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                3,
            )

        preview_points = clicked_points
        if missing_corner and len(clicked_points) == required_clicks:
            try:
                known_corners, missing_point = build_missing_corner_points()
                preview_points = [known_corners.get(corner, missing_point) for corner in corner_order]
            except ValueError:
                preview_points = clicked_points

        if len(preview_points) == 4:
            display_points = [[point_x * scale, point_y * scale] for point_x, point_y in preview_points]
            pts = np.array(display_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        if missing_corner:
            next_label = click_specs[len(clicked_points)][1] if len(clicked_points) < required_clicks else "Enter to save"
            full_order = " -> ".join(label for _key, label in click_specs)
            instruction = f"Clockwise order: {full_order}. Next: {next_label}. Enter=save, R=reset, Q=quit"
        else:
            instruction = "Click 4 floor corners in order: TL, TR, BR, BL. Enter=save, R=reset, Q=quit"

        cv2.putText(
            preview,
            instruction,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(clicked_points) == required_clicks:
            break
        if key == ord("r"):
            clicked_points.clear()
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            raise SystemExit("Calibration cancelled.")

    cv2.destroyWindow(window_name)

    if missing_corner:
        known_corners, missing_point = build_missing_corner_points()
        ordered_points = [known_corners.get(corner, missing_point) for corner in corner_order]
        image_points = np.array(ordered_points, dtype=np.float32)
    else:
        image_points = np.array(clicked_points, dtype=np.float32)
    map_points = np.array(
        [[0, 0], [map_size_cm, 0], [map_size_cm, map_size_cm], [0, map_size_cm]],
        dtype=np.float32,
    )

    matrix, _ = cv2.findHomography(image_points, map_points)
    if matrix is None:
        raise RuntimeError("Unable to calculate homography from selected points.")

    save_homography(matrix_path, matrix, image_points, map_size_cm)
    return matrix.astype(np.float32)

def ensure_homographies(contexts, setup_force):
    for context in contexts:
        if context.cap is None:
            continue

        success, first_frame = context.cap.read()
        if not success or first_frame is None:
            raise RuntimeError(f"Unable to read first frame for camera {context.camera_id}")

        if context.matrix_path.exists() and not setup_force:
            context.homography, context.map_size_cm = load_homography(
                context.matrix_path,
                requested_map_size_cm=context.map_size_cm,
            )
            print(f"Loaded homography for {context.camera_id} from {context.matrix_path}")
        else:
            print(f"Calibrating homography for {context.camera_id}...")
            context.homography = collect_calibration_points(
                first_frame,
                context.map_size_cm,
                context.matrix_path,
                missing_corner=context.missing_corner,
            )
            print(f"Saved homography for {context.camera_id} to {context.matrix_path}")
