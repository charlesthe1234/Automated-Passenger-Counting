"""Shared tracker launch configuration for tester and operator entry points.

This module intentionally has no Tkinter, Torch, or backend dependencies.  It
can therefore be imported by the technical launcher, the persistent CV worker,
and lightweight tests without loading computer-vision libraries.
"""

from __future__ import annotations

import getpass
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

from constants import (
    DEFAULT_CROSS_CAMERA_MAX_SKEW_SECONDS,
    DEFAULT_ELEVATED_MATRIX_1,
    DEFAULT_ELEVATED_MATRIX_2,
    DEFAULT_FUSION_DISTANCE_CM,
    DEFAULT_MEDIAPIPE_MODEL_PATH,
    DEFAULT_REID_DISTANCE_THRESHOLD,
    DEFAULT_TACTICAL_MAP_GRID_COLUMNS,
    DEFAULT_TACTICAL_MAP_GRID_ROWS,
    DEFAULT_TACTICAL_MAP_SIZE_CM,
    DEFAULT_TRACKER_CONFIG_PATH,
    DEFAULT_YOLO_NMS_IOU,
)


EDGE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = EDGE_DIRECTORY.parent
BACKEND_ENV_PATH = REPOSITORY_ROOT / "backend" / ".env"


def default_debug_run_id() -> str:
    """A unique run ID for a manually launched debugging session.

    Reusing one generic ID makes every terminal launch pile into a single
    ever-growing external run, so each launch gets its own identifier unless
    CV_RUN_ID is set explicitly.
    """
    operator = re.sub(r"[^A-Za-z0-9]+", "", getpass.getuser() or "operator")[:16] or "operator"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"debug_{operator}_{stamp}"


def read_env_file(path: Path = BACKEND_ENV_PATH) -> dict[str, str]:
    """Read the small KEY=VALUE subset used by this repository.

    Values are never printed here.  Full shell expansion is deliberately not
    supported; backend/.env is configuration, not executable shell code.
    """

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _camera_sources(env: Mapping[str, str]) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    raw_sources = env.get("CAMERA_URLS", "")
    for index, entry in enumerate(raw_sources.split(","), start=1):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            camera_id, source = entry.split("=", 1)
        else:
            camera_id, source = f"cam_{index}", entry
        camera_id = camera_id.strip()
        source = source.strip()
        if camera_id and source:
            sources.append((camera_id, source))
    if not sources and env.get("CAMERA_URL", "").strip():
        sources.append(
            (
                env.get("PRIMARY_CAMERA_ID", "cam_1").strip() or "cam_1",
                env["CAMERA_URL"].strip(),
            )
        )
    return sources


def is_truthy(value) -> bool:
    """Shared coercion so the launcher and the argument builder always agree."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def default_launch_values(env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Return the approved production preset and tester starting values."""

    env_values = dict(read_env_file() if env is None else env)
    if env is None:
        # Process-level overrides are useful for controlled deployments and
        # smoke tests; backend/.env remains the normal source of truth.
        for key, value in os.environ.items():
            if key in {"CAMERA_URL", "CAMERA_URLS", "PRIMARY_CAMERA_ID"} or key.startswith(
                ("CV_", "MQTT_")
            ):
                env_values[key] = value
    sources = _camera_sources(env_values)
    camera_1 = sources[0] if sources else ("cam_1", "")
    camera_2 = sources[1] if len(sources) > 1 else ("cam_2", "")
    legacy_yolo_device = env_values.get("CV_YOLO_DEVICE")
    yolo_device_1 = (
        env_values.get("CV_YOLO_DEVICE_1", legacy_yolo_device or "0") or "0"
    )
    yolo_device_2 = (
        env_values.get("CV_YOLO_DEVICE_2", legacy_yolo_device or "1") or "1"
    )
    return {
        "camera_mode": env_values.get(
            "CV_CAMERA_MODE", "both" if camera_2[1] else "camera_1"
        ),
        "setup": False,
        "use_mediapipe": env_values.get("CV_USE_MEDIAPIPE", "true"),
        "use_reid": env_values.get("CV_USE_REID", "true"),
        "enable_mivolo": env_values.get("CV_ENABLE_MIVOLO", "true"),
        "use_mqtt": env_values.get("CV_USE_MQTT", "true"),
        "disable_map_motion_filter": False,
        "debug_identity_events": False,
        # EXPERIMENTAL two-plane metrology. Defaults to off so an ordinary run
        # never reads elevated-plane calibration or learns metric heights. The
        # launcher checkbox drives it per run; the environment file is the
        # existing persistence mechanism for a remembered default.
        "enable_3d_level_detection": env_values.get("CV_ENABLE_3D_LEVEL_DETECTION", "false"),
        "elevated_matrix_1": env_values.get("CV_ELEVATED_HOMOGRAPHY_1", DEFAULT_ELEVATED_MATRIX_1),
        "elevated_matrix_2": env_values.get("CV_ELEVATED_HOMOGRAPHY_2", DEFAULT_ELEVATED_MATRIX_2),
        # Off by default: the confidence rule is the current behaviour, and this
        # only exists so the previous rule can be reproduced for comparison.
        "disable_position_confidence_gating": env_values.get(
            "CV_DISABLE_POSITION_CONFIDENCE_GATING", "false"
        ),
        # yolo_device remains as a compatibility alias for older callers.
        "yolo_device": yolo_device_1,
        "yolo_device_1": yolo_device_1,
        "yolo_device_2": yolo_device_2,
        "reid_device": env_values.get("CV_REID_DEVICE", "cuda:1") or "cuda:1",
        "mediapipe_delegate": env_values.get("CV_MEDIAPIPE_DELEGATE", "auto") or "auto",
        "source_1": camera_1[1],
        "source_2": camera_2[1],
        "camera_id_1": camera_1[0],
        "camera_id_2": camera_2[0],
        "matrix_1": env_values.get("CV_HOMOGRAPHY_1", "homography_matrix.json"),
        "matrix_2": env_values.get("CV_HOMOGRAPHY_2", "homography_matrix_2.json"),
        "missing_corner_1": env_values.get("CV_MISSING_CORNER_1", ""),
        "missing_corner_2": env_values.get("CV_MISSING_CORNER_2", ""),
        "model": env_values.get("CV_YOLO_MODEL", "yolo26m.pt") or "yolo26m.pt",
        "yolo_confidence": env_values.get("CV_YOLO_CONFIDENCE", "0.75") or "0.75",
        "yolo_nms_iou": env_values.get("CV_YOLO_NMS_IOU", str(DEFAULT_YOLO_NMS_IOU)),
        "tracker_config": env_values.get("CV_TRACKER_CONFIG", DEFAULT_TRACKER_CONFIG_PATH),
        "run_id": env_values.get("CV_RUN_ID", "").strip() or default_debug_run_id(),
        "map_size_cm": env_values.get("CV_MAP_SIZE_CM", str(DEFAULT_TACTICAL_MAP_SIZE_CM)),
        "map_grid_columns": env_values.get(
            "CV_MAP_GRID_COLUMNS", str(DEFAULT_TACTICAL_MAP_GRID_COLUMNS)
        ),
        "map_grid_rows": env_values.get(
            "CV_MAP_GRID_ROWS", str(DEFAULT_TACTICAL_MAP_GRID_ROWS)
        ),
        "fusion_distance_cm": env_values.get(
            "CV_FUSION_DISTANCE_CM", str(DEFAULT_FUSION_DISTANCE_CM)
        ),
        "cross_camera_max_skew_seconds": env_values.get(
            "CV_CROSS_CAMERA_MAX_SKEW_SECONDS",
            str(DEFAULT_CROSS_CAMERA_MAX_SKEW_SECONDS),
        ),
        "reid_distance_threshold": env_values.get(
            "CV_REID_DISTANCE_THRESHOLD", str(DEFAULT_REID_DISTANCE_THRESHOLD)
        ),
        "reid_checkpoint": env_values.get("CV_REID_CHECKPOINT", "transreid_msmt17.pth"),
        "reid_role_checkpoint": env_values.get(
            "CV_REID_ROLE_CHECKPOINT", "evacuation_mobilenet_v1.pth"
        ),
        "mediapipe_model": env_values.get(
            "CV_MEDIAPIPE_MODEL", DEFAULT_MEDIAPIPE_MODEL_PATH
        ),
        "mqtt_broker": env_values.get("MQTT_HOST", "localhost") or "localhost",
        "mqtt_port": env_values.get("MQTT_PORT", "1883") or "1883",
        "mqtt_username": env_values.get("MQTT_USERNAME", ""),
        "mqtt_password": env_values.get("MQTT_PASSWORD", ""),
        "reid_api_url": env_values.get("CV_REID_API_URL", "http://localhost:8000"),
        "backend_url": env_values.get("CV_BACKEND_URL", ""),
    }


def dashboard_launch_values(
    run_id: str,
    identity_debug_log: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Production values enforced for sessions started from the dashboard."""

    values = default_launch_values(env)
    values["run_id"] = run_id
    # Dashboard sessions must follow live movement rather than retaining an
    # identity's old map coordinate when the impossible-speed guard latches.
    # The technical launcher keeps its independent A/B comparison checkbox.
    values["disable_map_motion_filter"] = True
    if identity_debug_log is not None:
        values["debug_identity_events"] = True
        values["identity_debug_log"] = identity_debug_log
    return values


def build_tracker_arguments(values: Mapping[str, object]) -> list[str]:
    """Build the existing main_tracker CLI arguments without a shell string."""

    def text(name: str, fallback: str = "") -> str:
        value = values.get(name, fallback)
        return str(value).strip() if value is not None else fallback

    def enabled(name: str, fallback: bool = False) -> bool:
        return is_truthy(values.get(name, fallback))

    mode = text("camera_mode", "both").lower()
    selected_1 = mode in {"both", "camera_1", "camera 1 only"}
    selected_2 = mode in {"both", "camera_2", "camera 2 only"}
    if selected_1 and not text("source_1"):
        raise ValueError("Camera 1 source is not configured in backend/.env.")
    if selected_2 and not text("source_2"):
        raise ValueError("Camera 2 source is not configured in backend/.env.")

    yolo_device_1 = text("yolo_device_1", text("yolo_device", "0"))
    yolo_device_2 = text("yolo_device_2", yolo_device_1)
    primary_yolo_device = yolo_device_1 if selected_1 else yolo_device_2
    args = ["--device", primary_yolo_device]
    if selected_1 and selected_2:
        args.extend(["--device-2", yolo_device_2])
    if enabled("setup"):
        args.append("--setup")
    if enabled("debug_identity_events"):
        args.append("--debug-identity-events")
        if text("identity_debug_log"):
            args.extend(["--identity-debug-log", text("identity_debug_log")])
    # When the checkbox is unticked the flag is omitted entirely, so the tracker
    # never imports the experimental module or opens an elevated calibration.
    if enabled("disable_position_confidence_gating"):
        args.append("--disable-position-confidence-gating")
    if enabled("enable_3d_level_detection"):
        args.append("--enable-3d-level-detection")
        args.extend(["--elevated-matrix", text("elevated_matrix_1", DEFAULT_ELEVATED_MATRIX_1)])
        args.extend(["--elevated-matrix-2", text("elevated_matrix_2", DEFAULT_ELEVATED_MATRIX_2)])
    args.extend(["--model", text("model", "yolo26m.pt")])
    args.extend(["--conf", text("yolo_confidence", "0.75")])
    args.extend(["--iou", text("yolo_nms_iou", str(DEFAULT_YOLO_NMS_IOU))])
    args.extend(["--tracker-config", text("tracker_config", DEFAULT_TRACKER_CONFIG_PATH)])
    args.extend(["--map-size-cm", text("map_size_cm", str(DEFAULT_TACTICAL_MAP_SIZE_CM))])
    args.extend(
        [
            "--map-grid-columns",
            text("map_grid_columns", str(DEFAULT_TACTICAL_MAP_GRID_COLUMNS)),
            "--map-grid-rows",
            text("map_grid_rows", str(DEFAULT_TACTICAL_MAP_GRID_ROWS)),
        ]
    )
    if enabled("disable_map_motion_filter"):
        args.append("--disable-map-motion-filter")
    if enabled("use_mediapipe", True):
        args.append("--use-mediapipe-feet")
    args.extend(["--mediapipe-model", text("mediapipe_model", DEFAULT_MEDIAPIPE_MODEL_PATH)])
    args.extend(["--mediapipe-delegate", text("mediapipe_delegate", "auto").lower()])

    if enabled("use_reid", True):
        args.extend(
            [
                "--use-appearance-reid",
                "--reid-checkpoint",
                text("reid_checkpoint", "transreid_msmt17.pth"),
                "--fastreid-root",
                text("fastreid_root", "fast-reid"),
                "--reid-device",
                text("reid_device", "cuda:1"),
                "--reid-role-checkpoint",
                text("reid_role_checkpoint", "evacuation_mobilenet_v1.pth"),
            ]
        )
        if text("reid_api_url"):
            args.extend(["--reid-api-url", text("reid_api_url")])
        if text("reid_distance_threshold"):
            args.extend(["--reid-distance-threshold", text("reid_distance_threshold")])
        if not enabled("enable_mivolo", True):
            args.append("--no-demographics")

    if selected_1:
        args.extend(["--source", text("source_1")])
        args.extend(["--matrix", text("matrix_1", "homography_matrix.json")])
        args.extend(["--camera-id", text("camera_id_1", "cam_1")])
        if text("missing_corner_1") not in {"", "None"}:
            args.extend(["--missing-corner", text("missing_corner_1")])
    else:
        args.extend(["--source", text("source_2")])
        args.extend(["--matrix", text("matrix_2", "homography_matrix_2.json")])
        args.extend(["--camera-id", text("camera_id_2", "cam_2")])
        if text("missing_corner_2") not in {"", "None"}:
            args.extend(["--missing-corner", text("missing_corner_2")])

    if selected_1 and selected_2:
        args.extend(["--source-2", text("source_2")])
        args.extend(["--matrix-2", text("matrix_2", "homography_matrix_2.json")])
        args.extend(["--camera-id-2", text("camera_id_2", "cam_2")])
        if text("missing_corner_2") not in {"", "None"}:
            args.extend(["--missing-corner-2", text("missing_corner_2")])
        args.extend(["--fusion-distance-cm", text("fusion_distance_cm", "50")])
        args.extend(
            [
                "--cross-camera-max-skew-seconds",
                text("cross_camera_max_skew_seconds", "0.35"),
            ]
        )

    args.extend(["--run-id", text("run_id", "field_test_001")])
    if enabled("use_mqtt", True):
        args.extend(["--mqtt-broker", text("mqtt_broker", "localhost")])
        args.extend(["--mqtt-port", text("mqtt_port", "1883")])
        args.extend(["--mqtt-topic", "cag/tactical"])
        args.extend(["--mqtt-metrics-topic", "cag/metrics"])
        args.extend(["--mqtt-publish-interval", "0.2"])
        if text("mqtt_username"):
            args.extend(["--mqtt-username", text("mqtt_username")])
        if text("mqtt_password"):
            args.extend(["--mqtt-password", text("mqtt_password")])
    if text("backend_url"):
        args.extend(["--backend-url", text("backend_url")])
    return args


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<redacted>"
    if not parsed.scheme or "@" not in parsed.netloc:
        return value
    host = parsed.netloc.rsplit("@", 1)[1]
    return urlunsplit((parsed.scheme, f"<credentials>@{host}", parsed.path, parsed.query, parsed.fragment))


def redact_tracker_arguments(arguments: list[str]) -> list[str]:
    """Return a display/log-safe copy of a tracker argument vector."""

    redacted = list(arguments)
    secret_options = {"--mqtt-password"}
    url_options = {"--source", "--source-2"}
    for index, value in enumerate(redacted[:-1]):
        if value in secret_options:
            redacted[index + 1] = "<redacted>"
        elif value in url_options:
            redacted[index + 1] = _redact_url(redacted[index + 1])
    return redacted
