"""Multi-camera tracking runtime.

Owns the per-frame pipeline: model preloading, camera worker threads, the
per-camera detect/pose/ReID pass, and the loop that fuses, renders, and
publishes each cycle. The stages it drives live in sibling modules --
tracker_cli, tracker_calibration, fused_person, tactical_render,
camera_fusion, fusion_diagnostics, and dashboard_payload -- which this
module composes and which do not import it back.
"""

import torch
import concurrent.futures
try:
    from ultralytics import YOLO #human detection model 
except ImportError:
    YOLO = None
import cv2
import json
import traceback
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from constants import (
    DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    DEFAULT_REID_EMA_ALPHA,
    DEFAULT_TACTICAL_MAP_SIZE_CM,
    DEFAULT_TRACKER_CONFIG_PATH,
    DEFAULT_YOLO_NMS_IOU,
    FPS_EMA_ALPHA,
    POSITION_QUALITY_NONE,
)
from core_math import (
    camera_point_to_map,
    update_map_motion,
)
from camera_stream import (
    CameraContext,
    LiveCamera,
)
from mediapipe_runtime import create_mediapipe_pose_estimator
from pose_engine import get_standing_points
from reid_memory import AppearanceIdentityMemory
from reid_models import (
    EvacuationRoleClassifier,
    TransReIDFeatureExtractor,
)
from reid_backend_store import ReidBackendStore
from identity_debug import identity_event
from cross_camera_provisional import CrossCameraProvisionalCoordinator
from session_lock import CvRuntimeLock

from camera_fusion import (
    fuse_camera_points,
    suppress_display_duplicates,
)
from dashboard_payload import (
    build_payloads,
    create_mqtt_client,
    post_json,
)
from fusion_diagnostics import (
    build_frame_performance_snapshot,
    log_fusion_cycle_summary,
)
from tactical_render import (
    create_combined_tactical_map,
    create_runtime_display_windows,
    create_tactical_map,
    draw_top_left_text,
)
from tracker_calibration import ensure_homographies
from tracker_cli import (
    configure_torch_runtime,
    parse_args,
    validate_tracker_args,
)


def build_camera_contexts(args):
    contexts = []
    camera_sources = [args.source]
    camera_ids = [args.camera_id]
    matrix_paths = [Path(args.matrix)]
    camera_devices = [args.device]

    if args.source_2:
        camera_sources.append(args.source_2)
        camera_ids.append(args.camera_id_2)
        matrix_paths.append(Path(args.matrix_2 if args.matrix_2 else f"{Path(args.matrix).stem}_2{Path(args.matrix).suffix}"))
        camera_devices.append(getattr(args, "device_2", None) or args.device)

    for camera_id, source, matrix_path, device in zip(
        camera_ids,
        camera_sources,
        matrix_paths,
        camera_devices,
    ):
        source_value = int(source) if str(source).isdigit() else source
        context = CameraContext(camera_id, source_value, matrix_path, args.map_size_cm)
        context.device = str(device)
        contexts.append(context)

    if contexts:
        contexts[0].missing_corner = args.missing_corner
    if len(contexts) > 1:
        contexts[1].missing_corner = args.missing_corner_2

    return contexts


def build_three_d_level_estimator(contexts, args):
    """Construct the experimental two-plane estimator, or return None.

    EXPERIMENTAL: returns None immediately unless the launcher checkbox set
    --enable-3d-level-detection, so no elevated calibration file is opened and
    no metric height memory is created for an ordinary run.

    A calibration problem must never take down standard 2D tracking, so a
    failure is reported and the feature is disabled for this run instead of
    raising.
    """
    if not getattr(args, "enable_3d_level_detection", False):
        print("3D Level Detection: Disabled", flush=True)
        return None

    # Imported lazily so a disabled run never even loads the module.
    from three_d_level import CalibrationError, ThreeDLevelPositionEstimator

    elevated_paths = [args.elevated_matrix, args.elevated_matrix_2]
    pairs = {}
    for index, context in enumerate(contexts):
        elevated = elevated_paths[index] if index < len(elevated_paths) else None
        if not elevated:
            print(
                f"3D Level Detection: Unavailable - calibration error "
                f"(no elevated-plane file configured for {context.camera_id})",
                flush=True,
            )
            return None
        pairs[context.camera_id] = (context.matrix_path, Path(elevated))

    estimator = ThreeDLevelPositionEstimator(max_lean_degrees=args.three_d_max_lean_degrees)
    try:
        estimator.initialize(pairs)
    except CalibrationError as error:
        print(f"3D Level Detection: Unavailable - calibration error\n  {error}", flush=True)
        return None
    except Exception as error:  # never let experimental geometry stop tracking
        print(f"3D Level Detection: Unavailable - unexpected calibration failure\n  {error}", flush=True)
        return None

    for camera_id, health in estimator.validate_calibration().items():
        print(
            f"3D Level Detection: {camera_id} elevated plane at "
            f"{health['elevated_height_cm']:.1f} cm, calibration residual "
            f"{health['residual']:.5f}",
            flush=True,
        )
    print("3D Level Detection: Enabled (shadow mode - logging only)", flush=True)
    if not getattr(args, "debug_identity_events", False):
        # Shadow mode's only output is the identity event log, so running
        # without it produces no measurements at all.
        print(
            "3D Level Detection: WARNING - shadow results are written to the identity event "
            "log, which is switched off. Tick 'Temporary identity event log' in the launcher "
            "as well, or this run will record nothing to evaluate.",
            flush=True,
        )
    return estimator


def process_camera_frame(
    context,
    conf,
    device_id,
    pose_dropout_ttl_frames=DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    imgsz=640,
    half=True,
    nms_iou=DEFAULT_YOLO_NMS_IOU,
    tracker_config=DEFAULT_TRACKER_CONFIG_PATH,
    use_map_motion_filter=True,
):
    processing_timestamp = time.monotonic()
    if hasattr(context.cap, "read_with_metadata"):
        success, frame, captured_at, capture_sequence = context.cap.read_with_metadata()
    else:
        success, frame = context.cap.read()
        captured_at, capture_sequence = time.monotonic(), None

    if not success or frame is None:
        context.raw_frame = None
        context.tactical_points = []
        context.tactical_observations = []
        context.annotated_frame = None
        context.raw_detection_count = 0
        context.tracked_person_count = 0
        context.tactical_person_count = 0
        context.suppressed_track_count = 0
        return False

    if capture_sequence is not None and capture_sequence == context.last_capture_sequence:
        return True
    context.last_capture_sequence = capture_sequence
    if hasattr(context.cap, "prepare_frame"):
        frame = context.cap.prepare_frame(frame)
    context.raw_frame = frame
    context.tactical_points = []
    context.tactical_observations = []
    frame_timestamp = time.monotonic() if captured_at is None else float(captured_at)

    context.frame_index += 1

    # --- FPS tracking (EMA-smoothed so the readout doesn't jitter frame to frame) ---
    if context._last_frame_time is not None:
        frame_delta = processing_timestamp - context._last_frame_time
        if frame_delta > 1e-6:
            instantaneous_fps = 1.0 / frame_delta
            if context.fps <= 0.0:
                context.fps = instantaneous_fps
            else:
                context.fps = (1.0 - FPS_EMA_ALPHA) * context.fps + FPS_EMA_ALPHA * instantaneous_fps
    context._last_frame_time = processing_timestamp

    use_half = bool(half and str(device_id).lower() != "cpu" and torch.cuda.is_available())
    results = context.model.track(
        frame,
        classes=[0],
        conf=conf,
        iou=nms_iou,
        imgsz=imgsz,
        quantize=16 if use_half else None,
        persist=True,
        tracker=tracker_config,
        verbose=False,
        device=device_id,
    )
    result = results[0]
    annotated_frame = frame.copy()
    standing_points = get_standing_points(
        result,
        frame,
        context.pose_estimator,
        anatomical_ratio_memory=context.anatomical_ratio_memory,
        anatomical_anchor_memory=context.anatomical_anchor_memory,
        last_foot_memory=context.last_foot_memory,
        frame_index=context.frame_index,
        pose_dropout_ttl_frames=pose_dropout_ttl_frames,
        annotated_frame=annotated_frame,
        appearance_memory=context.appearance_memory,
        camera_id=context.camera_id,
        observation_time=frame_timestamp,
        use_mediapipe_feet=context.use_mediapipe_feet,
        map_projector=lambda image_point: camera_point_to_map(image_point, context.homography),
        map_size_cm=getattr(context, "map_size_cm", DEFAULT_TACTICAL_MAP_SIZE_CM),
        three_d_estimator=getattr(context, "three_d_estimator", None),
    )
    try:
        context.raw_detection_count = len(getattr(result, "boxes", ()))
    except TypeError:
        context.raw_detection_count = 0
    context.suppressed_track_count = sum(
        1 for standing_point in standing_points if standing_point.get("suppressed")
    )
    context.tracked_person_count = len(standing_points) - context.suppressed_track_count

    # Preserve Ultralytics' YOLO-pose skeleton for accepted detections, but
    # never call plot() on the unfiltered result because that would paint a
    # rejected shadow before the application could hide it.
    if not context.use_mediapipe_feet:
        accepted_indices = [
            index
            for index, standing_point in enumerate(standing_points)
            if not standing_point.get("suppressed")
        ]
        if accepted_indices:
            annotated_frame = result[accepted_indices].plot(img=frame.copy())

    draw_top_left_text(
        annotated_frame,
        f"FPS: {context.fps:.1f}",
        left_margin=12,
        top_margin=10,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1.6,
        color=(0, 255, 0),
        thickness=3,
    )

    for index, standing_point in enumerate(standing_points):
        if standing_point.get("suppressed"):
            continue
        point = standing_point["point"]
        speed_mps = None
        motion_status = None
        role = standing_point.get("role")
        if role == "scdf":
            label_color = (0, 165, 255)
        elif role == "cag":
            label_color = (0, 255, 255)
        elif role == "evacuee":
            label_color = (255, 150, 0)
        else:
            label_color = (255, 255, 255) if standing_point.get("identity_id") is None else (0, 0, 255)
        if index < len(result.boxes):
            box_values = result.boxes.xyxy[index].detach().cpu().numpy().astype(int)
            box_x1, box_y1, box_x2, box_y2 = box_values.tolist()
            cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), label_color, 2)
        map_point = None
        position_quality = standing_point.get("position_quality", POSITION_QUALITY_NONE)
        if point is not None:
            feet_x, feet_y = point
            raw_map_x, raw_map_y = camera_point_to_map((feet_x, feet_y), context.homography)
            motion_key = (
                ("identity", int(standing_point["identity_id"]))
                if standing_point.get("identity_id") is not None
                else ("temporary_group", standing_point["temporary_group_id"])
                if standing_point.get("temporary_group_id") is not None
                else ("track", context.camera_id, standing_point.get("track_id"))
            )
            if use_map_motion_filter:
                (map_x, map_y), speed_mps, motion_status = update_map_motion(
                    context.map_motion_memory,
                    motion_key,
                    (raw_map_x, raw_map_y),
                    frame_timestamp,
                    # Identity-keyed motion state is shared by design so a
                    # renumbered local track keeps its smoothing.  Naming the
                    # owning track lets the filter notice when two live tracks
                    # claim one identity in the same frame.
                    owner=(context.camera_id, standing_point.get("track_id")),
                    # A soft point is folded in gently.  At full alpha one
                    # uncertain frame contaminates the smoothed position for
                    # many frames afterwards, so the camera keeps reporting a
                    # bad point long after it can see the feet again.
                    quality=position_quality,
                )
            else:
                map_x, map_y = raw_map_x, raw_map_y
                motion_status = "unfiltered"
            map_point = (float(map_x), float(map_y))
            # The per-camera map is a debug view of this camera's own belief,
            # so it carries the grade and the reason alongside the coordinate.
            context.tactical_points.append({
                "point": (float(map_x), float(map_y)),
                "position_quality": position_quality,
                "position_quality_reason": standing_point.get("position_quality_reason"),
                "label": (
                    f"ID {standing_point['identity_id']}"
                    if standing_point.get("identity_id") is not None
                    else f"P{standing_point.get('track_id')}"
                ),
                "provisional": standing_point.get("identity_id") is None,
            })

            cv2.circle(annotated_frame, (feet_x, feet_y), radius=8, color=(0, 0, 255), thickness=-1)
            label_anchor = (feet_x + 10, feet_y - 10)
            label = f"({map_x:.0f}cm, {map_y:.0f}cm)"
        else:
            label_anchor = (20, 40 + index * 22)
            label = "(no visible foot)"

        # Emitted even without a ground point.  A person whose feet are hidden
        # is still on the floor and still the same person; dropping them here
        # deleted them from the cross-camera matcher as well as the map, which
        # reset the pairing streak and split their shared ID.  Downstream
        # decides what a point-less observation may do -- it holds an existing
        # pairing on identity, but cannot form one, and cannot be drawn.
        context.tactical_observations.append({
            "camera_id": context.camera_id,
            "local_track_id": standing_point.get("track_id"),
            "identity_id": standing_point.get("identity_id"),
            "temporary_group_id": standing_point.get("temporary_group_id"),
            "reid_confirmed": (
                standing_point.get("identity_id") is not None
                and bool(standing_point.get("reid_confirmed"))
            ),
            "identity_state": standing_point.get("identity_state"),
            "point": map_point,
            "position_quality": position_quality,
            "position_quality_reason": standing_point.get("position_quality_reason"),
            "captured_at": frame_timestamp,
            "frame_index": context.frame_index,
            "role": standing_point.get("role"),
            "inside_tactical_map": bool(standing_point.get("inside_tactical_map")),
        })

        if standing_point.get("temporary_group_id") is not None:
            label = f"ANALYZING {label}"
        elif standing_point.get("identity_id") is not None:
            label = f"ID {standing_point['identity_id']} {label}"
            if standing_point.get("reidentified"):
                label = f"{label} reid={standing_point['reid_similarity']:.2f}"
            if role in ("cag", "scdf"):
                label = f"{label} {role.upper()}"
            elif role == "evacuee":
                gender = standing_point.get("gender", "Unknown")
                age = standing_point.get("age", "Unknown")
                gallery_total = standing_point.get("gallery_total", 5)
                label = f"{label} {gender}/{age} ({standing_point.get('gallery_filled', 0)}/{gallery_total})"
            if standing_point.get("identity_state") == "provisional":
                label = f"{label} PROVISIONAL"
            elif standing_point.get("identity_state") == "challenged":
                label = f"{label} CHECK"
        elif standing_point.get("reid_intake_required", 0) > 1 and standing_point.get("reid_intake_count", 0) > 0:
            label = f"ANALYZING ({standing_point['reid_intake_count']}/{standing_point['reid_intake_required']}) {label}"
        elif standing_point["track_id"] is not None:
            label = f"T{standing_point['track_id']} {label}"
        ratio = standing_point.get("ratio")
        if speed_mps is not None:
            label = f"{label} v={speed_mps:.2f}m/s"
            if motion_status == "speed_hold":
                label = f"{label} HOLD"
        head_pitch = standing_point.get("head_pitch")
        if head_pitch == "looking_straight":
            label = f"{label} head=up"
        elif head_pitch == "looking_down":
            label = f"{label} head=down"
        elif head_pitch == "unknown":
            label = f"{label} head=?"
        if ratio is not None:
            label = f"{label} r={ratio:.3f}"
        if standing_point["method"] == "anatomical_ratio":
            label = f"{label} calc"
        elif standing_point["method"] == "last_seen":
            label = f"{label} last"
        elif standing_point["method"] == "physics_hold":
            label = f"{label} physics"
        elif standing_point["method"] == "no_visible_ankle":
            label = f"{label} no-ankle"
        cv2.putText(
            annotated_frame,
            label,
            label_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            label_color,
            3,
        )

    context.tactical_person_count = len(context.tactical_observations)
    context.annotated_frame = annotated_frame
    return True


class PreloadedCvModels:
    """Models loaded together for one pipeline process."""

    def __init__(
        self,
        yolo_models,
        pose_estimators,
        reid_extractor=None,
        role_classifier=None,
        demographics_engine=None,
    ):
        self.yolo_models = list(yolo_models)
        self.pose_estimators = list(pose_estimators)
        self.reid_extractor = reid_extractor
        self.role_classifier = role_classifier
        self.demographics_engine = demographics_engine

    def prepare_for_session(self):
        # Programmatic callers may reuse a bundle, although the dashboard now
        # launches a fresh process and invokes this only once per run.
        for model in self.yolo_models:
            if getattr(model, "predictor", None) is not None:
                model.predictor = None

    def close(self):
        for estimator in self.pose_estimators:
            if estimator is not None:
                estimator.close()


def preload_models(args, loading_stage=None, preload_optional_models=True):
    """Load every enabled model without opening an RTSP camera."""

    configure_torch_runtime()
    args = validate_tracker_args(args)
    contexts = build_camera_contexts(args)

    def stage(name):
        if loading_stage is not None:
            loading_stage(name)

    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO is not installed in .venv-cv-linux.")

    yolo_models = []
    pose_estimators = []
    reid_extractor = None
    role_classifier = None
    demographics_engine = None
    try:
        for index, context in enumerate(contexts, start=1):
            stage(
                f"Loading YOLO camera {index}/{len(contexts)} on {context.device}"
            )
            model = YOLO(args.model)
            if str(context.device).lower() != "cpu":
                preload_device = (
                    f"cuda:{context.device}"
                    if str(context.device).isdigit()
                    else context.device
                )
                model.to(preload_device)
            yolo_models.append(model)

        for index, _context in enumerate(contexts, start=1):
            stage(
                f"Loading MediaPipe camera {index}/{len(contexts)} "
                f"on {args.mediapipe_delegate}"
            )
            pose_estimators.append(
                create_mediapipe_pose_estimator(
                    args.use_mediapipe_feet or args.use_appearance_reid,
                    Path(args.mediapipe_model),
                    delegate=args.mediapipe_delegate,
                )
            )

        if args.use_appearance_reid:
            stage("Loading TransReID")
            reid_extractor = TransReIDFeatureExtractor(
                Path(args.reid_checkpoint),
                device=args.reid_device,
                fastreid_root=args.fastreid_root,
            )
            if not reid_extractor.is_available():
                raise RuntimeError(
                    "The configured TransReID checkpoint could not be loaded; "
                    "the production worker will not silently report ready with histogram fallback."
                )

            if preload_optional_models and not args.no_reid_role_classification:
                stage("Loading role classifier")
                role_classifier = EvacuationRoleClassifier(args.reid_role_checkpoint)
                if role_classifier.model is None:
                    raise RuntimeError("The configured role-classification checkpoint could not be loaded.")

            if preload_optional_models and not args.no_demographics:
                stage("Loading MiVOLO demographics")
                from demographics import DemographicsEngine

                demographics_engine = DemographicsEngine(device=args.reid_device)

        stage("Models ready")
        return PreloadedCvModels(
            yolo_models,
            pose_estimators,
            reid_extractor=reid_extractor,
            role_classifier=role_classifier,
            demographics_engine=demographics_engine,
        )
    except Exception:
        for estimator in pose_estimators:
            if estimator is not None:
                estimator.close()
        raise


def run_pipeline(args, models, stop_event=None, started_callback=None):
    """Run one camera session using an already-loaded model bundle."""

    # Normally already applied by preload_models; repeated calls are no-ops.
    configure_torch_runtime()
    args = validate_tracker_args(args)
    stop_event = stop_event or threading.Event()
    contexts = build_camera_contexts(args)
    if len(models.yolo_models) != len(contexts) or len(models.pose_estimators) != len(contexts):
        raise RuntimeError("The preloaded model count does not match the configured camera count.")
    models.prepare_for_session()

    shared_appearance_memory = None
    provisional_coordinator = None
    mqtt_client = None
    try:
        if args.use_appearance_reid:
            reid_backend_store = None
            if args.reid_api_url and not args.no_persistent_reid_db:
                reid_backend_store = ReidBackendStore(
                    args.reid_api_url,
                    run_id=args.run_id,
                    timeout=args.http_timeout,
                )
                print(f"ReID persistence: FastAPI/SQLite at {args.reid_api_url}")
            shared_appearance_memory = AppearanceIdentityMemory(
                similarity_threshold=args.reid_similarity_threshold,
                distance_threshold=args.reid_distance_threshold,
                ttl_frames=args.reid_memory_ttl_frames,
                ema_alpha=DEFAULT_REID_EMA_ALPHA,
                reid_extractor=models.reid_extractor,
                db_path=None if args.no_persistent_reid_db or reid_backend_store is not None else args.reid_db,
                persistence_store=reid_backend_store,
                intake_frames=args.reid_intake_frames,
                gallery_update_interval_frames=args.reid_gallery_update_interval_frames,
                evidence_dir=None if args.no_reid_evidence else args.reid_evidence_dir,
                evidence_camera_ids=(
                    set(args.reid_evidence_camera_id) if args.reid_evidence_camera_id else None
                ),
                intake_delay_seconds=args.reid_intake_delay_seconds,
                intake_timeout_seconds=args.reid_intake_timeout_seconds,
                blur_threshold=args.reid_blur_threshold,
                semantic_confidence_threshold=args.reid_semantic_confidence,
                semantic_retry_frames=args.reid_semantic_retry_frames,
                intake_retry_frames=args.reid_intake_retry_frames,
                role_checkpoint=args.reid_role_checkpoint,
                role_confidence_threshold=args.reid_role_confidence,
                enable_role_classification=not args.no_reid_role_classification,
                enable_demographics=not args.no_demographics,
                demographics_device=args.reid_device,
                role_classifier=models.role_classifier,
                demographics_engine=models.demographics_engine,
                cross_camera_fusion_distance_cm=args.fusion_distance_cm,
                cross_camera_max_skew_seconds=args.cross_camera_max_skew_seconds,
                position_confidence_gating=not args.disable_position_confidence_gating,
                provisional_location_confirm_frames=args.provisional_location_confirm_frames,
            )
            if len(contexts) >= 2:
                provisional_coordinator = CrossCameraProvisionalCoordinator(
                    shared_appearance_memory,
                    max_distance_cm=args.fusion_distance_cm,
                    max_skew_seconds=args.cross_camera_max_skew_seconds,
                    required_pair_frames=args.provisional_pair_frames,
                    location_confirm_frames=args.provisional_location_confirm_frames,
                    hold_grace_frames=args.provisional_hold_grace_frames,
                    hold_max_frames=args.provisional_hold_max_frames,
                )

        for index, context in enumerate(contexts):
            context.model = models.yolo_models[index]
            context.use_mediapipe_feet = bool(args.use_mediapipe_feet)
            context.pose_estimator = models.pose_estimators[index]
            context.appearance_memory = shared_appearance_memory
            context.cap = LiveCamera(context.source, camera_id=context.camera_id)
            if not context.cap.is_opened():
                raise RuntimeError(
                    f"Unable to open video source for camera {context.camera_id}."
                )

        ensure_homographies(contexts, args.setup)

        # Built after the ground calibration exists, because --setup may have
        # only just created it and the elevated plane is validated against it.
        three_d_estimator = build_three_d_level_estimator(contexts, args)
        for context in contexts:
            context.three_d_estimator = three_d_estimator

        if args.mqtt_broker:
            print(f"Attempting to connect to MQTT broker at {args.mqtt_broker}:{args.mqtt_port}...")
            mqtt_client = create_mqtt_client(
                args.mqtt_broker,
                args.mqtt_port,
                client_id=args.mqtt_client_id,
                username=args.mqtt_username,
                password=args.mqtt_password,
            )

        if started_callback is not None:
            started_callback()

        backend_post_url = None
        if args.backend_url:
            backend_post_url = urllib.parse.urljoin(
                args.backend_url.rstrip("/") + "/", args.backend_path.lstrip("/")
            )
        last_mqtt_publish = 0.0
        last_performance_frame_signature = None
        # How long each camera pairing has been holding, so one distorted frame
        # cannot split a settled pair into two dots and two headcounts.
        fusion_pair_memory = {}
        # Monotonic per-cycle label so every event from one fusion pass can be
        # joined without matching on timestamps.
        fusion_cycle_id = 0
        create_runtime_display_windows(contexts)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(contexts)), thread_name_prefix="camera"
        ) as camera_executor:
            while not stop_event.is_set():
                futures = [
                    camera_executor.submit(
                        process_camera_frame,
                        context,
                        args.conf,
                        context.device,
                        args.pose_dropout_ttl_frames,
                        args.imgsz,
                        args.half,
                        args.iou,
                        args.tracker_config,
                        not args.disable_map_motion_filter,
                    )
                    for context in contexts
                ]
                statuses = [future.result() for future in futures]
                if stop_event.is_set():
                    break
                if not any(statuses):
                    camera_states = [
                        {
                            "camera_id": context.camera_id,
                            "frame_available": bool(status),
                            "reader_running": bool(getattr(context.cap, "running", False)),
                            "last_sequence": getattr(context.cap, "sequence", None),
                        }
                        for context, status in zip(contexts, statuses)
                    ]
                    print(
                        f"[CAMERA_DEBUG] No camera frames available. Exiting. States: {camera_states}",
                        flush=True,
                    )
                    identity_event("tracking_exit_no_active_cameras", camera_states=camera_states)
                    break

                camera_observations = {
                    context.camera_id: context.tactical_observations for context in contexts
                }
                if provisional_coordinator is not None:
                    provisional_coordinator.update(
                        {
                            camera_id: [
                                observation
                                for observation in observations
                                if observation.get("inside_tactical_map")
                            ]
                            for camera_id, observations in camera_observations.items()
                        }
                    )
                map_size_cm = max(context.map_size_cm for context in contexts)
                fusion_cycle_id += 1
                fused_people = fuse_camera_points(
                    camera_observations,
                    args.fusion_distance_cm,
                    max_skew_seconds=args.cross_camera_max_skew_seconds,
                    require_reid=args.use_appearance_reid,
                    pair_memory=fusion_pair_memory,
                    fusion_cycle_id=fusion_cycle_id,
                )
                fused_people_before_suppression = fused_people
                # Presentation only, and deliberately last: everything that
                # feeds the identity layer -- the provisional coordinator above
                # and the appearance memory inside the camera workers -- has
                # already run and consumed its inputs. Nothing downstream of
                # here can reach back into them.
                suppression_stats = {}
                fused_people = suppress_display_duplicates(
                    fused_people,
                    duplicate_distance_cm=args.display_duplicate_distance_cm,
                    fusion_cycle_id=fusion_cycle_id,
                    stats=suppression_stats,
                )
                log_fusion_cycle_summary(
                    fusion_cycle_id,
                    camera_observations,
                    fused_people_before_suppression,
                    fused_people,
                    unresolved_count=suppression_stats.get("unresolved", 0),
                )
                performance_snapshot = build_frame_performance_snapshot(
                    contexts, fused_people
                )
                performance_frame_signature = tuple(
                    (camera["camera_id"], camera["frame_index"])
                    for camera in performance_snapshot["cameras"]
                )
                if performance_frame_signature != last_performance_frame_signature:
                    identity_event(
                        "frame_performance",
                        console=False,
                        **performance_snapshot,
                    )
                    last_performance_frame_signature = performance_frame_signature
                combined_map = create_combined_tactical_map(
                    fused_people,
                    map_size_cm,
                    grid_columns=args.map_grid_columns,
                    grid_rows=args.map_grid_rows,
                    show_evidence=args.debug_identity_events,
                )

                for context in contexts:
                    if context.annotated_frame is not None:
                        cv2.imshow(f"Camera {context.camera_id}", context.annotated_frame)
                        tactical_map = create_tactical_map(
                            context.tactical_points,
                            map_size_cm,
                            title=f"{context.camera_id} tactical map",
                            grid_columns=args.map_grid_columns,
                            grid_rows=args.map_grid_rows,
                            show_evidence=args.debug_identity_events,
                        )
                        cv2.imshow(f"Map {context.camera_id}", tactical_map)
                cv2.imshow("Combined tactical map", combined_map)

                if mqtt_client is not None:
                    now = time.monotonic()
                    if now - last_mqtt_publish >= args.mqtt_publish_interval:
                        last_mqtt_publish = now
                        tactical_payload, metric_payload = build_payloads(
                            contexts, args, fused_people, combined_map
                        )
                        try:
                            mqtt_client.publish(args.mqtt_topic, json.dumps(tactical_payload), qos=1)
                            mqtt_client.publish(
                                args.mqtt_metrics_topic, json.dumps(metric_payload), qos=1
                            )
                        except Exception as exc:
                            print(f"MQTT publish failed: {exc}")

                if backend_post_url:
                    tactical_payload, _ = build_payloads(
                        contexts, args, fused_people, combined_map
                    )
                    try:
                        post_json(backend_post_url, tactical_payload, timeout=args.http_timeout)
                    except urllib.error.URLError as exc:
                        print(f"Backend POST failed: {exc}")
                    except Exception as exc:
                        print(f"Unexpected backend POST error: {exc}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[CAMERA_DEBUG] Keyboard 'q' received. Exiting.", flush=True)
                    identity_event("tracking_exit_keyboard", key="q")
                    break
                if key == ord("r"):
                    print("Recalibrating all cameras...")
                    ensure_homographies(contexts, setup_force=True)
    finally:
        for context in contexts:
            if context.cap is not None:
                context.cap.release()
        if mqtt_client is not None:
            try:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
            except Exception as exc:
                print(f"MQTT cleanup failed: {exc}")
        if shared_appearance_memory is not None:
            shared_appearance_memory.close(drain=True)
        cv2.destroyAllWindows()
        identity_event("tracking_shutdown_complete")


def main(argv=None):
    args = parse_args(argv)
    with CvRuntimeLock("technical tester launcher"):
        # Preserve the legacy CLI/tester launch behaviour: the role and
        # demographics models remain lazy there. The dashboard worker calls
        # preload_models with its default and loads every enabled model.
        models = preload_models(args, preload_optional_models=False)
        try:
            run_pipeline(args, models)
        finally:
            models.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # TEMP_CAMERA_DEBUG: capture failures that previously disappeared
        # when the launcher's child console closed immediately.
        formatted_traceback = traceback.format_exc()
        identity_event(
            "tracking_unhandled_exception",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=formatted_traceback,
        )
        print(
            f"[SYSTEM_ERROR] Unhandled {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
