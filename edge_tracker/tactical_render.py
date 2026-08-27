"""Drawing the tactical map: grids, point styles, canvases, and display windows."""

import base64
import cv2
import numpy as np

from constants import (
    DEFAULT_TACTICAL_MAP_GRID_COLUMNS,
    DEFAULT_TACTICAL_MAP_GRID_ROWS,
    DISPLAY_SIZE,
    POSITION_QUALITY_HARD,
    POSITION_QUALITY_NONE,
    POSITION_QUALITY_SOFT,
    POSITION_QUALITY_STALE,
    TACTICAL_MAP_SIZE,
)

from fused_person import (
    _display_authority_camera,
    normalize_tactical_entry,
)


MAP_COLOR_DARK_GREEN = (0, 160, 0)

MAP_COLOR_LIGHT_GREEN = (110, 215, 130)

MAP_COLOR_STALE = (150, 175, 150)

MAP_COLOR_YELLOW = (0, 215, 255)

MAP_COLOR_OUT_OF_ZONE = (0, 0, 255)

def per_camera_point_style(entry):
    """Colour one camera's own dot by how much that camera's point is worth.

    A provisional local track is yellow wherever it is drawn, so the same
    marker means the same thing on every window.  Otherwise the grade decides,
    and a stale point is drawn hollow so it can never be mistaken at a glance
    for a live measurement.
    """
    if entry.get("provisional"):
        return MAP_COLOR_YELLOW, -1
    quality = str(entry.get("position_quality") or POSITION_QUALITY_NONE).lower()
    if quality == POSITION_QUALITY_HARD:
        return MAP_COLOR_DARK_GREEN, -1
    if quality == POSITION_QUALITY_SOFT:
        return MAP_COLOR_LIGHT_GREEN, -1
    if quality == POSITION_QUALITY_STALE:
        return MAP_COLOR_STALE, 2
    return MAP_COLOR_STALE, 2

def fused_person_style(person):
    """Colour one fused dot by what the system has concluded about the person.

    Yellow is reserved for people carrying no master ID at all, so the colour
    can never contradict the label beside it: anything drawn as ``ID n`` is
    green, anything drawn as ``P17`` or ``Analyzing`` is yellow.  A confirmed
    master seen by a single camera stays green -- dropping it back to yellow
    would say the identity had been lost, when only a viewpoint was.
    """
    if person.get("identity_id") is None:
        return MAP_COLOR_YELLOW
    if len(set(person.get("sources", ()))) > 1:
        return MAP_COLOR_DARK_GREEN
    return MAP_COLOR_LIGHT_GREEN

def draw_tactical_grid(canvas, grid_columns, grid_rows):
    """Draw a complete grid whose boundaries always stay on the canvas.

    Each coordinate is derived independently from the inclusive pixel extent,
    so non-divisible canvas sizes cannot accumulate rounding error.  Two-pixel
    internal lines survive normal OpenCV window downscaling much more reliably
    than the previous one-pixel lines.
    """

    height, width = canvas.shape[:2]
    if width <= 0 or height <= 0:
        return
    columns = max(1, int(grid_columns))
    rows = max(1, int(grid_rows))
    grid_color = (195, 195, 195)
    border_color = (40, 40, 40)

    for column in range(1, columns):
        pixel_x = int(round(column * (width - 1) / columns))
        cv2.line(
            canvas,
            (pixel_x, 0),
            (pixel_x, height - 1),
            grid_color,
            2,
            cv2.LINE_8,
        )
    for row in range(1, rows):
        pixel_y = int(round(row * (height - 1) / rows))
        cv2.line(
            canvas,
            (0, pixel_y),
            (width - 1, pixel_y),
            grid_color,
            2,
            cv2.LINE_8,
        )

    # Draw this last so internal lines cannot lighten any part of the edge.
    cv2.rectangle(
        canvas,
        (0, 0),
        (width - 1, height - 1),
        border_color,
        3,
        cv2.LINE_8,
    )

def draw_top_left_text(
    image,
    text,
    *,
    left_margin,
    top_margin,
    font_face,
    font_scale,
    color,
    thickness,
):
    """Draw text from a top margin while respecting OpenCV's baseline origin."""

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font_face,
        font_scale,
        thickness,
    )
    image_height, image_width = image.shape[:2]
    origin_x = min(
        max(0, int(left_margin)),
        max(0, image_width - text_width - 1),
    )
    origin_y = int(top_margin) + text_height
    origin_y = min(origin_y, max(text_height, image_height - baseline - 1))
    origin_y = max(text_height, origin_y)
    cv2.putText(
        image,
        text,
        (origin_x, origin_y),
        font_face,
        font_scale,
        color,
        thickness,
    )
    return (origin_x, origin_y), (text_width, text_height), baseline

def create_tactical_map(
    points_cm,
    map_size_cm,
    title="Tactical map",
    color=None,
    grid_columns=DEFAULT_TACTICAL_MAP_GRID_COLUMNS,
    grid_rows=DEFAULT_TACTICAL_MAP_GRID_ROWS,
    show_evidence=False,
):
    """Draw one camera's own view of the floor.

    This is a debug window, not the product: it shows where this camera thinks
    each person is standing and how much that belief is worth, which is exactly
    the information needed to explain why the combined map preferred the other
    camera.  It deliberately never borrows a position from elsewhere.
    """
    canvas = np.full((TACTICAL_MAP_SIZE, TACTICAL_MAP_SIZE, 3), 245, dtype=np.uint8)
    scale = TACTICAL_MAP_SIZE / map_size_cm

    draw_tactical_grid(canvas, grid_columns, grid_rows)

    for index, raw_entry in enumerate(points_cm, start=1):
        entry = normalize_tactical_entry(raw_entry, index)
        if entry is None:
            # Detected, but this camera could not place the feet. Drawing it
            # would mean inventing a position this camera never had.
            continue
        map_x, map_y = entry["point"]
        pixel_x = int(round(map_x * scale))
        pixel_y = int(round(map_y * scale))
        in_zone = 0 <= map_x <= map_size_cm and 0 <= map_y <= map_size_cm
        point_color, thickness = per_camera_point_style(entry)
        if color is not None and entry.get("position_quality") is None:
            point_color = color
        if not in_zone:
            # Off-map is red on every window, but a stale point stays hollow so
            # it still reads as "not measured this frame".
            point_color = MAP_COLOR_OUT_OF_ZONE
        pixel_x = max(0, min(TACTICAL_MAP_SIZE - 1, pixel_x))
        pixel_y = max(0, min(TACTICAL_MAP_SIZE - 1, pixel_y))
        cv2.circle(canvas, (pixel_x, pixel_y), 9, point_color, thickness)
        cv2.putText(
            canvas,
            entry["label"],
            (pixel_x + 10, pixel_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            point_color,
            3,
        )
        if show_evidence and entry.get("position_quality"):
            detail = str(entry["position_quality"]).upper()
            if entry.get("position_quality_reason"):
                detail = f"{detail} | {entry['position_quality_reason']}"
            cv2.putText(
                canvas,
                detail,
                (pixel_x + 10, pixel_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                point_color,
                1,
            )

    draw_top_left_text(
        canvas,
        title,
        left_margin=14,
        top_margin=8,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.7,
        color=(40, 40, 40),
        thickness=2,
    )
    return canvas

def encode_image_to_base64(image, quality=80):
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")

def create_combined_tactical_map(
    fused_people,
    map_size_cm,
    grid_columns=DEFAULT_TACTICAL_MAP_GRID_COLUMNS,
    grid_rows=DEFAULT_TACTICAL_MAP_GRID_ROWS,
    show_evidence=False,
):
    canvas = np.full((TACTICAL_MAP_SIZE, TACTICAL_MAP_SIZE, 3), 245, dtype=np.uint8)
    scale = TACTICAL_MAP_SIZE / map_size_cm

    draw_tactical_grid(canvas, grid_columns, grid_rows)

    for person_index, person in enumerate(fused_people, start=1):
        center = person.get("center")
        if center is None:
            # Detected and still counted, but no camera could place their feet
            # this frame. Drawing them would mean inventing a position.
            continue
        map_x, map_y = center
        pixel_x = int(round(map_x * scale))
        pixel_y = int(round(map_y * scale))
        in_zone = 0 <= map_x <= map_size_cm and 0 <= map_y <= map_size_cm
        point_color = fused_person_style(person)
        if not in_zone:
            point_color = MAP_COLOR_OUT_OF_ZONE

        pixel_x = max(0, min(TACTICAL_MAP_SIZE - 1, pixel_x))
        pixel_y = max(0, min(TACTICAL_MAP_SIZE - 1, pixel_y))
        cv2.circle(canvas, (pixel_x, pixel_y), 11, point_color, -1)
        if person.get("identity_id") is not None:
            person_label = f"ID {person['identity_id']}"
        elif person.get("temporary_group_id") is not None:
            person_label = "Analyzing"
        else:
            person_label = f"P{person_index}"
        cv2.putText(
            canvas,
            person_label,
            (pixel_x + 10, pixel_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            point_color,
            3,
        )
        cv2.putText(
            canvas,
            "+".join(person["sources"]),
            (pixel_x + 10, pixel_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.84,
            point_color,
            2,
        )
        if show_evidence:
            # Why this dot sits where it does: which camera won the position,
            # and what the camera that lost was reporting.
            authority = _display_authority_camera(person)
            detail = f"pos:{authority}" if authority else "pos:-"
            for dropped in person.get("suppressed_duplicates", ()):
                detail = (
                    f"{detail}  x{dropped.get('camera_id')}"
                    f"/{str(dropped.get('position_quality') or '').upper()}"
                )
                if dropped.get("identity_id") is not None:
                    detail = f"{detail}(ID {dropped['identity_id']})"
            cv2.putText(
                canvas,
                detail,
                (pixel_x + 10, pixel_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                point_color,
                1,
            )

    draw_top_left_text(
        canvas,
        "Combined fused map",
        left_margin=14,
        top_margin=8,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1.4,
        color=(40, 40, 40),
        thickness=3,
    )
    return canvas

def create_runtime_display_windows(
    contexts,
    camera_window_size=DISPLAY_SIZE,
    tactical_window_size=(TACTICAL_MAP_SIZE, TACTICAL_MAP_SIZE),
):
    """Create every live-view window in manually resizable mode.

    Calling ``imshow`` without first creating a window uses OpenCV's fixed
    ``WINDOW_AUTOSIZE`` mode. Create and size the windows once at session
    startup, then leave them under operator control for the rest of the run.
    Deliberately do not restore geometry from a prior process: some OpenCV
    backends report the image client area but resize the decorated window,
    which can make a saved size shrink on every save/restore cycle.
    """

    window_sizes = []
    for context in contexts:
        window_sizes.extend(
            (
                (f"Camera {context.camera_id}", camera_window_size),
                (f"Map {context.camera_id}", tactical_window_size),
            )
        )
    window_sizes.append(("Combined tactical map", tactical_window_size))
    for window_name, (width, height) in window_sizes:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, int(width), int(height))
        except cv2.error:
            # Some window managers do not support programmatic sizing.
            # The live views should still open and remain manually resizable.
            continue
    return [window_name for window_name, _size in window_sizes]
