"""Command-line surface and torch runtime configuration for the tracker."""

import argparse

import torch

from constants import (
    DEFAULT_CROSS_CAMERA_MAX_SKEW_SECONDS,
    DEFAULT_DISPLAY_DUPLICATE_DISTANCE_CM,
    DEFAULT_ELEVATED_MATRIX_1,
    DEFAULT_ELEVATED_MATRIX_2,
    DEFAULT_FUSION_DISTANCE_CM,
    DEFAULT_MEDIAPIPE_MODEL_PATH,
    DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    DEFAULT_PROVISIONAL_HOLD_GRACE_FRAMES,
    DEFAULT_PROVISIONAL_HOLD_MAX_FRAMES,
    DEFAULT_PROVISIONAL_LOCATION_CONFIRM_FRAMES,
    DEFAULT_PROVISIONAL_PAIR_FRAMES,
    DEFAULT_REID_BLUR_THRESHOLD,
    DEFAULT_REID_DISTANCE_THRESHOLD,
    DEFAULT_REID_INTAKE_DELAY_SECONDS,
    DEFAULT_REID_INTAKE_RETRY_FRAMES,
    DEFAULT_REID_INTAKE_TIMEOUT_SECONDS,
    DEFAULT_REID_MEMORY_TTL_FRAMES,
    DEFAULT_REID_ROLE_CHECKPOINT,
    DEFAULT_REID_ROLE_CONFIDENCE,
    DEFAULT_REID_SEMANTIC_CONFIDENCE,
    DEFAULT_REID_SEMANTIC_COOLDOWN_FRAMES,
    DEFAULT_REID_SEMANTIC_RETRY_FRAMES,
    DEFAULT_REID_SIMILARITY_THRESHOLD,
    DEFAULT_RTSP_URL,
    DEFAULT_TACTICAL_MAP_GRID_COLUMNS,
    DEFAULT_TACTICAL_MAP_GRID_ROWS,
    DEFAULT_TACTICAL_MAP_SIZE_CM,
    DEFAULT_THREE_D_MAX_LEAN_DEGREES,
    DEFAULT_TRACKER_CONFIG_PATH,
    DEFAULT_YOLO_NMS_IOU,
)
from identity_debug import configure_identity_debug
from pathlib import Path


_torch_runtime_configured = False

def configure_torch_runtime(report=None):
    """Apply the tracker's global torch settings and report the active device.

    This used to run at import time, which mutated global torch state and wrote
    to stdout as a side effect of importing this module. Both entry points
    (main() for the tester launcher, cv_worker for the dashboard) reach GPU work
    through preload_models, so configuring here keeps the behaviour while
    leaving the module safe to import. Repeat calls are no-ops.
    """
    global _torch_runtime_configured
    if _torch_runtime_configured:
        return
    _torch_runtime_configured = True

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        # Fixed-size camera frames benefit from cuDNN's convolution autotuner.
        # It is safe to enable because the tracker does not train or change
        # input tensor shapes during a run.
        torch.backends.cudnn.benchmark = True

    emit_line = report if report is not None else print
    emit_line(f"Is CUDA available?: {cuda_available}")
    if cuda_available:
        emit_line(f"GPU Name: {torch.cuda.get_device_name(0)}")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Detect people feet and project them to a tactical map.")
    parser.add_argument("--source", default=DEFAULT_RTSP_URL, help="Camera/video source. Use 0 for webcam.")
    parser.add_argument("--source-2", default=None, help="Optional second camera/video source.")
    parser.add_argument("--model", default="yolo26m.pt", help="YOLO model path.")
    parser.add_argument("--use-mediapipe-feet", action="store_true", help="Use MediaPipe heel/toe landmarks inside each YOLO person box.")
    parser.add_argument("--mediapipe-model", default=DEFAULT_MEDIAPIPE_MODEL_PATH, help="MediaPipe pose landmarker .task model path.")
    parser.add_argument(
        "--mediapipe-delegate",
        choices=["auto", "cpu", "gpu", "gpu:0", "gpu:1"],
        default="auto",
        help="MediaPipe execution device. GPU indexes use NVIDIA PRIME/EGL routing on Ubuntu, with CPU fallback.",
    )
    parser.add_argument("--matrix", default="homography_matrix.json", help="Saved homography file for camera 1.")
    parser.add_argument("--matrix-2", default=None, help="Optional saved homography file for camera 2.")
    parser.add_argument("--setup", action="store_true", help="Force the 4-click homography setup for available cameras.")
    # EXPERIMENTAL: two-plane metrology. Off unless the launcher checkbox is
    # ticked, and every code path below is gated on this single value so an
    # unticked box means the geometry is never loaded, learned or evaluated.
    parser.add_argument(
        "--enable-3d-level-detection",
        action="store_true",
        help="Experimental: estimate hidden-foot ground positions from learned landmark heights.",
    )
    parser.add_argument(
        "--elevated-matrix",
        default=DEFAULT_ELEVATED_MATRIX_1,
        help="Elevated-plane calibration for camera 1 (only read with --enable-3d-level-detection).",
    )
    parser.add_argument(
        "--elevated-matrix-2",
        default=DEFAULT_ELEVATED_MATRIX_2,
        help="Elevated-plane calibration for camera 2 (only read with --enable-3d-level-detection).",
    )
    parser.add_argument(
        "--disable-position-confidence-gating",
        action="store_true",
        help=(
            "Revert to the previous rule where any cross-camera position disagreement can "
            "reject an appearance match, including positions inferred from a clipped or "
            "foot-occluded detection box. For A/B comparison only."
        ),
    )
    parser.add_argument(
        "--three-d-max-lean-degrees",
        type=float,
        default=DEFAULT_THREE_D_MAX_LEAN_DEGREES,
        help="Reject landmark-height learning beyond this torso lean, in degrees.",
    )
    parser.add_argument(
        "--disable-map-motion-filter",
        action="store_true",
        help="Disable tactical-map position smoothing and the impossible-speed hold.",
    )
    parser.add_argument("--missing-corner", choices=["top_left", "top_right", "bottom_right", "bottom_left"], default=None, help="Camera 1 hidden calibration corner. Click the other 3 corners plus 2 points on the hidden corner edges.")
    parser.add_argument("--missing-corner-2", choices=["top_left", "top_right", "bottom_right", "bottom_left"], default=None, help="Camera 2 hidden calibration corner.")
    parser.add_argument(
        "--map-size-cm",
        type=int,
        default=DEFAULT_TACTICAL_MAP_SIZE_CM,
        help="Square tent/tactical-map side length in centimeters.",
    )
    parser.add_argument(
        "--map-grid-columns",
        type=int,
        default=DEFAULT_TACTICAL_MAP_GRID_COLUMNS,
        help="Number of visual grid columns on the tactical map.",
    )
    parser.add_argument(
        "--map-grid-rows",
        type=int,
        default=DEFAULT_TACTICAL_MAP_GRID_ROWS,
        help="Number of visual grid rows on the tactical map.",
    )
    parser.add_argument("--conf", type=float, default=0.60, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=DEFAULT_YOLO_NMS_IOU, help="YOLO NMS IoU threshold. Lower values suppress overlapping duplicate boxes more aggressively.")
    parser.add_argument("--tracker-config", default=DEFAULT_TRACKER_CONFIG_PATH, help="Project tracker YAML path.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size. Lower values improve FPS at some accuracy cost.")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 YOLO inference on CUDA (disable with --no-half if needed).",
    )
    parser.add_argument("--device", type=str, default="0", help="Device to run YOLO on (e.g., 0, 1, cpu)")
    parser.add_argument(
        "--device-2",
        type=str,
        default=None,
        help="Device for camera 2 YOLO; defaults to --device when omitted.",
    )
    parser.add_argument("--fusion-distance-cm", type=float, default=DEFAULT_FUSION_DISTANCE_CM, help="Maximum distance for two camera detections to count as the same person.")
    parser.add_argument(
        "--display-duplicate-distance-cm",
        type=float,
        default=DEFAULT_DISPLAY_DUPLICATE_DISTANCE_CM,
        help=(
            "Ground distance below which two cross-camera dots are drawn as one person, "
            "because two people cannot stand that close. Display only; never merges identities. "
            "Set to 0 to disable this proximity rule; two dots sharing one master ID are "
            "still drawn once regardless, since that needs no geometry."
        ),
    )
    parser.add_argument("--cross-camera-max-skew-seconds", type=float, default=DEFAULT_CROSS_CAMERA_MAX_SKEW_SECONDS, help="Maximum capture-time difference for cross-camera association.")
    parser.add_argument("--provisional-pair-frames", type=int, default=DEFAULT_PROVISIONAL_PAIR_FRAMES, help="Consecutive close cross-camera observations required before reserving one shared ID.")
    parser.add_argument("--provisional-hold-grace-frames", type=int, default=DEFAULT_PROVISIONAL_HOLD_GRACE_FRAMES, help="Coordinator updates to defer a new master after a promising cross-camera pair temporarily fails.")
    parser.add_argument("--provisional-hold-max-frames", type=int, default=DEFAULT_PROVISIONAL_HOLD_MAX_FRAMES, help="Absolute coordinator-update cap for a cross-camera new-master hold.")
    parser.add_argument("--provisional-location-confirm-frames", type=int, default=DEFAULT_PROVISIONAL_LOCATION_CONFIRM_FRAMES, help="Stable paired frames required for location-only promotion when no comparable ReID angle appears.")
    parser.add_argument("--pose-dropout-ttl-frames", type=int, default=DEFAULT_POSE_DROPOUT_TTL_FRAMES, help="Frames to keep a tracked person using last known foot point when pose landmarks disappear.")
    parser.add_argument("--use-appearance-reid", action="store_true", help="Use crop appearance memory to keep stable IDs when the local tracker changes IDs.")
    parser.add_argument("--reid-checkpoint", default="transreid_msmt17.pth", help="Path to a TransReID checkpoint for appearance feature extraction.")
    parser.add_argument("--fastreid-root", default="fast-reid", help="Path to the extracted fast-reid folder used by the TransReID checkpoint.")
    parser.add_argument("--reid-db", default="evacuee_database_v7.pkl", help="Persistent ReID gallery database file.")
    parser.add_argument("--reid-api-url", default=None, help="FastAPI base URL used to persist ReID identities in SQLite instead of pickle.")
    parser.add_argument("--no-persistent-reid-db", action="store_true", help="Keep ReID identities in memory only for this run.")
    parser.add_argument("--reid-intake-frames", type=int, default=5, help="Rapid crops averaged into the temporary matching query; the best crop and its own vector become baseline.")
    parser.add_argument("--reid-gallery-update-interval-frames", type=int, default=DEFAULT_REID_SEMANTIC_COOLDOWN_FRAMES, help="Frames to wait after successfully queuing a missing semantic gallery view.")
    parser.add_argument("--reid-evidence-dir", default="angle_evidence_v7", help="Folder for raw ReID baseline and semantic-view crop snapshots.")
    parser.add_argument(
        "--reid-evidence-camera-id",
        action="append",
        default=None,
        help=(
            "Camera allowed to save ReID evidence PNGs. Repeat this option to select multiple "
            "cameras; when omitted, every active camera saves baseline and angle evidence."
        ),
    )
    parser.add_argument("--no-reid-evidence", action="store_true", help="Disable saving labeled ReID crop evidence snapshots.")
    parser.add_argument("--reid-similarity-threshold", type=float, default=DEFAULT_REID_SIMILARITY_THRESHOLD, help="Appearance similarity needed to reuse an old stable ID.")
    parser.add_argument("--reid-distance-threshold", type=float, default=DEFAULT_REID_DISTANCE_THRESHOLD, help="Strict cosine-distance threshold for ReID; a match must be below this value.")
    parser.add_argument("--reid-memory-ttl-frames", type=int, default=DEFAULT_REID_MEMORY_TTL_FRAMES, help="Frames to retain stale local tracker bindings; persistent master galleries do not expire.")
    parser.add_argument("--reid-intake-delay-seconds", type=float, default=DEFAULT_REID_INTAKE_DELAY_SECONDS, help="Delay after first sighting before collecting the five-crop intake burst.")
    parser.add_argument("--reid-intake-timeout-seconds", type=float, default=DEFAULT_REID_INTAKE_TIMEOUT_SECONDS, help="After this delay, accept the best available non-sharp intake frames rather than waiting forever.")
    parser.add_argument("--reid-blur-threshold", type=float, default=DEFAULT_REID_BLUR_THRESHOLD, help="Minimum variance-of-Laplacian score for a clear ReID crop.")
    parser.add_argument("--reid-semantic-confidence", type=float, default=DEFAULT_REID_SEMANTIC_CONFIDENCE, help="YOLO confidence required before filling a semantic gallery slot.")
    parser.add_argument("--reid-semantic-retry-frames", type=int, default=DEFAULT_REID_SEMANTIC_RETRY_FRAMES, help="Frames to wait after a semantic crop fails a quality/orientation gate.")
    parser.add_argument("--reid-intake-retry-frames", type=int, default=DEFAULT_REID_INTAKE_RETRY_FRAMES, help="Initial frame backoff after a failed five-crop TransReID batch.")
    parser.add_argument("--reid-role-checkpoint", default=DEFAULT_REID_ROLE_CHECKPOINT, help="MobileNetV2 CAG/evacuee/SCDF role checkpoint.")
    parser.add_argument("--reid-role-confidence", type=float, default=DEFAULT_REID_ROLE_CONFIDENCE, help="Minimum confidence required before assigning a CAG/SCDF role.")
    parser.add_argument("--no-reid-role-classification", action="store_true", help="Disable the MobileNet role gate and treat new identities as evacuees.")
    parser.add_argument("--no-demographics", action="store_true", help="Disable background MiVOLO age/gender analysis for new evacuees.")
    parser.add_argument("--reid-device", type=str, default="cuda:0", help="Device to run ReID on (e.g., cuda:0, cuda:1, cpu)")
    # TEMP_IDENTITY_DEBUG: opt-in troubleshooting controls; remove after the ID-split investigation.
    parser.add_argument("--debug-identity-events", action="store_true", help="Temporarily log ReID and fusion decision events for ID-split troubleshooting.")
    parser.add_argument(
        "--debug-fusion-detail",
        action="store_true",
        help=(
            "Record every cross-camera candidate pair each cycle, not only the ones that "
            "tripped a diagnostic flag. Requires --debug-identity-events. Very verbose."
        ),
    )
    parser.add_argument("--identity-debug-log", default="identity_debug_events.jsonl", help="Temporary JSONL identity-event log path.")
    parser.add_argument("--run-id", default="default", help="Run identifier for backend tracking.")
    parser.add_argument("--camera-id", default="cam_1", help="Camera identifier for backend tracking.")
    parser.add_argument("--camera-id-2", default="cam_2", help="Optional second camera identifier.")
    parser.add_argument("--mqtt-broker", default=None, help="MQTT broker hostname or IP address.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-topic", default="cag/tactical", help="MQTT topic to publish tactical data.")
    parser.add_argument("--mqtt-metrics-topic", default="cag/metrics", help="MQTT topic to publish metric/count data.")
    parser.add_argument("--mqtt-publish-interval", type=float, default=0.2, help="Seconds between MQTT publishes.")
    parser.add_argument("--mqtt-client-id", default="tactical-publisher", help="MQTT client identifier.")
    parser.add_argument("--mqtt-username", default=None, help="MQTT username if broker requires authentication.")
    parser.add_argument("--mqtt-password", default=None, help="MQTT password if broker requires authentication.")
    parser.add_argument("--mqtt-send-map-image", action="store_true", help="Send tactical map image in MQTT payload as base64.")
    parser.add_argument("--mqtt-image-quality", type=int, default=80, help="JPEG quality for MQTT map image (1-100).")
    parser.add_argument("--backend-url", default=None, help="HTTP backend base URL for POST updates.")
    parser.add_argument("--backend-path", default="/api/metrics", help="Backend API path for POST updates.")
    parser.add_argument("--http-timeout", type=int, default=5, help="Timeout for backend HTTP POST requests.")
    return parser.parse_args(argv)

def validate_tracker_args(args):
    if not args.source:
        raise ValueError("--source is required. Configure CAMERA_URLS in backend/.env.")
    # TEMP_IDENTITY_DEBUG: disabled unless explicitly selected in the launcher/CLI.
    identity_debug_log = Path(args.identity_debug_log).expanduser()
    if not identity_debug_log.is_absolute():
        identity_debug_log = Path(__file__).resolve().parent / identity_debug_log
    configure_identity_debug(
        args.debug_identity_events,
        identity_debug_log,
        context={"run_id": args.run_id},
        detail=args.debug_fusion_detail,
    )
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("--iou must be greater than 0 and at most 1.")
    if args.map_size_cm <= 0:
        raise ValueError("--map-size-cm must be greater than 0.")
    if not 1 <= args.map_grid_columns <= 50:
        raise ValueError("--map-grid-columns must be between 1 and 50.")
    if not 1 <= args.map_grid_rows <= 50:
        raise ValueError("--map-grid-rows must be between 1 and 50.")
    if args.reid_api_url and args.no_reid_evidence:
        raise ValueError("--reid-api-url requires saved ReID evidence images; remove --no-reid-evidence.")
    tracker_config_path = Path(args.tracker_config).expanduser()
    if not tracker_config_path.is_absolute():
        tracker_config_path = Path(__file__).resolve().parent / tracker_config_path
    if not tracker_config_path.is_file():
        raise FileNotFoundError(f"Tracker configuration not found: {tracker_config_path}")
    args.tracker_config = str(tracker_config_path)
    return args
