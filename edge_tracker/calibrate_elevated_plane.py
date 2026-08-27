"""Calibrate the elevated plane used by experimental 3D Level Detection.

EXPERIMENTAL: only needed when the launcher's "Enable 3D Level Detection"
checkbox is used.  Standard 2D tracking never reads the file this produces.

What you are calibrating
------------------------
Four points that sit **directly above the same four floor corners** used by the
existing ground calibration, all at one accurately measured height.  Marking
poles, a taped line on four pillars, or four tripods at equal height all work.

Click them in the SAME ORDER as the ground calibration:

    top_left, top_right, bottom_right, bottom_left

Two things that matter, and one that does not:

* The camera must NOT move between the ground and elevated calibrations.  The
  script prints a consistency residual and refuses to save a bad pair.
* All four marks must be at the same height and the plane must be parallel to
  the floor.
* The measured height only needs to be *consistent*, not precise.  A wrong value
  biases the reported metric heights but cancels out of ground position, because
  learning and applying share the same geometry.

Usage
-----
    python calibrate_elevated_plane.py --camera 1 --height-cm 170
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from camera_stream import LiveCamera, resize_to_fit
from constants import DEFAULT_ELEVATED_MATRIX_1, DEFAULT_ELEVATED_MATRIX_2
from launch_config import default_launch_values

ROOT = Path(__file__).resolve().parent
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def collect_points(camera, height_cm):
    """Click four elevated marks on a LIVE view, in the floor corner order.

    Live rather than a single still, because the practical way to do this is to
    carry one marked pole to each floor corner in turn.  A frozen frame would
    demand four poles standing simultaneously.

    SPACE freezes the picture so the mark can be clicked precisely while the
    pole is held steady; SPACE again resumes.
    """
    clicked: list[tuple[float, float]] = []
    frozen = None
    scale = 1.0
    window = f"Elevated calibration at {height_cm:.0f} cm"

    def on_mouse(event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((x / scale, y / scale))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            if frozen is None:
                success, frame = camera.read()
                if not success or frame is None:
                    continue
                display, scale = resize_to_fit(frame)
            else:
                display, scale = frozen

            canvas = display.copy()
            for index, point in enumerate(clicked):
                position = (int(point[0] * scale), int(point[1] * scale))
                cv2.drawMarker(canvas, position, (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
                cv2.putText(
                    canvas, CORNER_ORDER[index], (position[0] + 10, position[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            if len(clicked) < 4:
                banner = f"Hold the {height_cm:.0f} cm mark over the {CORNER_ORDER[len(clicked)]} floor corner, then click it"
                helper = "SPACE freeze/resume   BACKSPACE undo   ESC cancel"
            else:
                banner = "All four marks captured"
                helper = "ENTER save   BACKSPACE undo   ESC cancel"
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (0, 0, 0), -1)
            cv2.putText(canvas, banner, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(canvas, helper, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if frozen is not None:
                cv2.putText(
                    canvas, "FROZEN", (canvas.shape[1] - 110, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2,
                )

            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                return None
            if key == 32:
                frozen = None if frozen is not None else (display.copy(), scale)
            if key in (8, 127) and clicked:
                clicked.pop()
            if key in (13, 10) and len(clicked) == 4:
                return np.array(clicked, dtype=np.float32)
    finally:
        cv2.destroyWindow(window)


def main(argv=None):
    values = default_launch_values()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", choices=["1", "2"], default="1")
    parser.add_argument("--height-cm", type=float, required=True,
                        help="Measured height of the four marks above the floor, in centimetres.")
    parser.add_argument("--output", default=None, help="Where to write the elevated calibration.")
    parser.add_argument("--source", default=None, help="Override the camera source from backend/.env.")
    args = parser.parse_args(argv)

    if args.height_cm <= 0:
        parser.error("--height-cm must be positive.")

    ground_path = ROOT / str(values["matrix_1" if args.camera == "1" else "matrix_2"])
    if not ground_path.is_file():
        print(f"Ground calibration {ground_path} does not exist. Calibrate the floor first.")
        return 2
    default_output = DEFAULT_ELEVATED_MATRIX_1 if args.camera == "1" else DEFAULT_ELEVATED_MATRIX_2
    output_path = ROOT / (args.output or default_output)

    source = args.source or values["source_1" if args.camera == "1" else "source_2"]
    if not source:
        print(f"No source configured for camera {args.camera} in backend/.env.")
        return 2
    source_value = int(source) if str(source).isdigit() else source

    ground_payload = json.loads(ground_path.read_text(encoding="utf-8"))
    map_size_cm = float(ground_payload.get("map_size_cm"))

    camera = LiveCamera(source_value, camera_id=f"cam_{args.camera}")
    if not camera.is_opened():
        print(f"Unable to open camera {args.camera}.")
        return 2
    print(f"\nFloor corners already calibrated in {ground_path.name}:")
    for corner, point in zip(CORNER_ORDER, ground_payload.get("image_points", [])):
        print(f"  {corner:<13} image ({point[0]:7.0f}, {point[1]:7.0f})")
    print(
        f"\nHold one mark at {args.height_cm:.0f} cm directly above each of those floor\n"
        "corners in turn, and click it. A broom or pole with tape at the measured\n"
        "height works; it does not need to be plumb-perfect, but it must be vertical\n"
        "and the height must be the same every time.\n"
    )
    try:
        image_points = collect_points(camera, args.height_cm)
    finally:
        camera.release()

    if image_points is None:
        print("Cancelled; nothing was written.")
        return 1

    map_points = np.array(
        [[0, 0], [map_size_cm, 0], [map_size_cm, map_size_cm], [0, map_size_cm]],
        dtype=np.float32,
    )
    matrix, _ = cv2.findHomography(image_points, map_points)
    if matrix is None:
        print("Could not compute a homography from those four points.")
        return 2

    payload = {
        "matrix": np.asarray(matrix, dtype=float).tolist(),
        "image_points": np.asarray(image_points, dtype=float).tolist(),
        "map_size_cm": map_size_cm,
        "plane_height_cm": float(args.height_cm),
        "point_order": list(CORNER_ORDER),
    }

    # Validate before saving: a pair that cannot be scale-matched is worse than
    # no calibration at all, because it would silently distort every estimate.
    from three_d_level import CalibrationError, build_plane_calibration

    temporary = output_path.with_suffix(".candidate.json")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        calibration = build_plane_calibration(f"cam_{args.camera}", ground_path, temporary)
    except CalibrationError as error:
        temporary.unlink(missing_ok=True)
        print(f"\nRejected, nothing was written:\n  {error}")
        return 2
    temporary.replace(output_path)

    print(f"\nSaved {output_path}")
    print(f"  elevated plane height : {calibration.elevated_height_cm:.1f} cm")
    print(f"  consistency residual  : {calibration.residual:.5f}  (0 = perfect, reject above 0.02)")
    print("\nTick 'Enable 3D Level Detection' in the launcher to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
