"""Per-detection pose analysis.

Owns get_standing_points, the pass that turns one camera frame and its
detections into standing points with a trust grade. The stages it drives --
landmark extraction, crop quality, orientation, ground-point estimation, and
the MediaPipe runtime -- live in sibling modules that do not import this one.
"""

import numpy as np

from constants import (
    DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    POSITION_QUALITY_NONE,
)
from core_math import (
    classify_position_quality,
    get_anatomical_ratio_from_memory,
)
from identity_debug import identity_event
from reid_crops import crop_person

from ground_point import (
    classify_box_bottom_evidence,
    estimate_box_bottom_point,
    estimate_mediapipe_foot_point,
    estimate_yolo_pose_ankle_point,
    run_three_d_level_shadow,
)
from face_region import estimate_face_box
from human_orientation import get_human_orientation
from reid_crop_quality import (
    _reid_crop_box,
    _reid_crop_intruder,
    assess_reid_body_completeness,
    detection_bounds_inside_reid_crop,
    detection_touches_vertical_frame_boundary,
    occluder_bounds_inside_reid_crop,
)


def get_standing_points(
    result,
    frame,
    pose_estimator=None,
    anatomical_ratio_memory=None,
    anatomical_anchor_memory=None,
    last_foot_memory=None,
    frame_index=None,
    pose_dropout_ttl_frames=DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    annotated_frame=None,
    appearance_memory=None,
    camera_id=None,
    observation_time=None,
    use_mediapipe_feet=True,
    map_projector=None,
    map_size_cm=None,
    three_d_estimator=None,
):
    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4), dtype=float)
    track_ids = result.boxes.id.cpu().numpy().astype(int) if getattr(result.boxes, "id", None) is not None else None
    keypoints = getattr(result, "keypoints", None)
    keypoint_xy = None
    keypoint_conf = None
    detection_confidences = None

    if keypoints is not None and keypoints.xy is not None:
        keypoint_xy = keypoints.xy.cpu().numpy()
        if keypoints.conf is not None:
            keypoint_conf = keypoints.conf.cpu().numpy()
    if getattr(result.boxes, "conf", None) is not None:
        detection_confidences = result.boxes.conf.cpu().numpy()

    # ReID crops are created only after the raw box-bottom projection enters
    # the tactical-map range. TransReID itself is deliberately deferred to
    # AppearanceIdentityMemory until an unmapped local track has completed
    # its rapid intake burst. Mapped tracks never run appearance inference.
    # Observe every box together before processing individuals so the memory
    # can identify a newly spawned, near-identical ByteTrack shadow relative
    # to the already established same-camera track.
    person_crops = [None] * len(boxes)
    active_identity_ids = set()
    if appearance_memory is not None:
        active_identity_ids = appearance_memory.observe_tracks(
            () if track_ids is None else track_ids,
            boxes,
            frame_index=frame_index,
            camera_id=camera_id,
            observed_at=observation_time,
        )

    suppressed_by_index = [False] * len(boxes)
    if appearance_memory is not None and track_ids is not None:
        for index in range(min(len(boxes), len(track_ids))):
            suppressed_by_index[index] = appearance_memory.is_track_suppressed(
                int(track_ids[index]),
                camera_id=camera_id,
            )

    standing_points = []
    for index, box in enumerate(boxes):
        track_id = int(track_ids[index]) if track_ids is not None and index < len(track_ids) else None
        identity_id = None
        temporary_group_id = None
        reid_similarity = 0.0
        reidentified = False
        assignment_metadata = {}
        reid_intake_count = 0
        reid_intake_required = 0
        pose_debug = {}
        detection_confidence = (
            float(detection_confidences[index])
            if detection_confidences is not None and index < len(detection_confidences)
            else None
        )
        if appearance_memory is not None and track_id is not None:
            identity_id = appearance_memory.lookup(track_id, camera_id=camera_id)
            temporary_group_id = appearance_memory.temporary_group(
                track_id,
                camera_id=camera_id,
            )

        map_point = None
        # Association physics must use the current raw detection, not a held
        # or smoothed foot point that could conceal a tracker-ID teleport.
        association_image_point = estimate_box_bottom_point(box)
        if association_image_point is not None and map_projector is not None:
            try:
                projected = map_projector((float(association_image_point[0]), float(association_image_point[1])))
                if projected is not None:
                    map_point = (float(projected[0]), float(projected[1]))
            except Exception:
                map_point = None
        # A clipped or foot-occluded box puts this point somewhere other than the
        # person's feet. It stays usable as evidence, but must not be allowed to
        # overrule appearance and split one person into two master identities.
        map_point_evidence, map_point_evidence_reason = classify_box_bottom_evidence(
            frame, box, other_boxes=boxes
        )

        inside_tactical_map = bool(
            map_point is not None
            and map_size_cm is not None
            and 0.0 <= map_point[0] <= float(map_size_cm)
            and 0.0 <= map_point[1] <= float(map_size_cm)
        )
        reid_crop_allowed = map_size_cm is None or inside_tactical_map
        track_suppressed = bool(
            index < len(suppressed_by_index) and suppressed_by_index[index]
        )
        if reid_crop_allowed:
            # Suppressed tracks retain their bounded shadow-verification crop.
            # For a canonical track, suppressed duplicate detections are
            # excluded as intruders by _reid_crop_intruder.
            intruder = None if track_suppressed else _reid_crop_intruder(
                frame,
                box,
                boxes,
                index,
                track_ids=track_ids,
                suppressed_by_index=suppressed_by_index,
            )
            if intruder is None:
                person_crops[index] = crop_person(frame, box)
            else:
                (
                    intruder_track_id,
                    intruder_area_ratio,
                    intruder_index,
                    intruder_depth,
                    intruder_allowed_area_ratio,
                ) = intruder
                # TEMP_IDENTITY_DEBUG: the boxes say which side of the subject
                # the intruder stood on -- compare bottom edges for depth
                # order, heights for relative distance -- and the pair of
                # ratios says how far over its own budget this one landed.
                # Together they are what retunes the behind budget without
                # another recorded session.
                identity_event(
                    "reid_crop_rejected_overlap",
                    track_key=(camera_id, track_id) if camera_id is not None else track_id,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    intruder_track_id=intruder_track_id,
                    intruder_area_ratio=intruder_area_ratio,
                    intruder_depth=intruder_depth,
                    intruder_allowed_area_ratio=intruder_allowed_area_ratio,
                    detection_box=box,
                    intruder_box=boxes[intruder_index],
                    reid_crop_box=_reid_crop_box(frame, box),
                    throttle_key=(
                        camera_id,
                        track_id if track_id is not None else index,
                        intruder_track_id
                        if intruder_track_id is not None
                        else intruder_index,
                    ),
                    throttle_seconds=1.0,
                    console=False,
                )
        else:
            # TEMP_IDENTITY_DEBUG: no crop is produced at all here, so intake
            # never sees this track and logs nothing of its own.
            identity_event(
                "reid_crop_rejected_off_map",
                track_key=(camera_id, track_id) if camera_id is not None else track_id,
                camera_id=camera_id,
                frame_index=frame_index,
                reason="outside_tactical_map",
                map_point=map_point,
                map_size_cm=map_size_cm,
                map_point_evidence=map_point_evidence,
                map_point_evidence_reason=map_point_evidence_reason,
                throttle_key=(camera_id, track_id if track_id is not None else index),
                throttle_seconds=1.0,
                console=False,
            )
        # MiVOLO reads a tight detection box with other people cut out of it,
        # while the saved crop is padded and may legitimately contain a
        # neighbour the intruder gate was willing to admit.  Both are recorded
        # against the crop now, because the frame they were measured in is gone
        # by the time the demographics worker runs.
        intake_body_bounds = None
        intake_occluder_boxes = ()
        if person_crops[index] is not None:
            intake_body_bounds = detection_bounds_inside_reid_crop(frame, box)
            intake_occluder_boxes = tuple(
                occluder_bounds_inside_reid_crop(
                    frame,
                    box,
                    boxes,
                    index,
                    suppressed_by_index=suppressed_by_index,
                )
            )
        if track_suppressed:
            # Never spend pose work or publish a tactical-map point for an
            # unresolved duplicate. ``assign`` is still called so the memory
            # can, after a short geometry-only probation, collect one bounded
            # five-crop appearance check. One-frame ghosts return before they
            # collect anything, and a verified duplicate never repeats it.
            if frame_index is not None and reid_crop_allowed:
                appearance_memory.assign(
                    track_id,
                    person_crops[index],
                    frame_index,
                    excluded_identity_ids=active_identity_ids,
                    camera_id=camera_id,
                    detection_confidence=detection_confidence,
                    intake_detection_box=box,
                    intake_body_bounds=intake_body_bounds,
                    intake_occluder_boxes=intake_occluder_boxes,
                    observed_at=observation_time,
                    map_point=map_point,
                )
            reid_intake_count = appearance_memory.pending_count(track_id, camera_id=camera_id)
            standing_points.append({
                "point": None,
                "track_id": track_id,
                "identity_id": None,
                "reid_similarity": 0.0,
                "reidentified": False,
                "reid_confirmed": False,
                "query_feature_space_id": None,
                "matched_slot": None,
                "reid_intake_count": reid_intake_count,
                "reid_intake_required": appearance_memory.required_intake_count(),
                "method": "shadow_suppressed",
                "ratio": None,
                "head_pitch": None,
                "orientation": None,
                "suppressed": True,
                "inside_tactical_map": inside_tactical_map,
            })
            continue

        if pose_estimator is not None and use_mediapipe_feet:
            point, method = estimate_mediapipe_foot_point(
                frame,
                box,
                pose_estimator,
                anatomical_ratio_memory=anatomical_ratio_memory,
                anatomical_anchor_memory=anatomical_anchor_memory,
                last_foot_memory=last_foot_memory,
                track_id=track_id,
                identity_id=identity_id,
                frame_index=frame_index,
                pose_dropout_ttl_frames=pose_dropout_ttl_frames,
                annotated_frame=annotated_frame,
                pose_debug=pose_debug,
                collect_metrology_landmarks=three_d_estimator is not None,
            )
        else:
            point = estimate_yolo_pose_ankle_point(index, keypoint_xy, keypoint_conf)
            method = "yolo_pose"
            if point is None:
                point = None
                method = "no_visible_ankle"

            # If MediaPipe is not being used for feet, run it only while a
            # track still needs its initial ReID intake or a semantic gallery
            # slot is due. The current ReID crop is reused directly.
            needs_reid_body_check = bool(
                reid_crop_allowed
                and
                pose_estimator is not None
                and appearance_memory is not None
                and track_id is not None
                and frame_index is not None
                and identity_id is None
            )
            semantic_probe_due = bool(
                reid_crop_allowed
                and
                person_crops[index] is not None
                and
                pose_estimator is not None
                and appearance_memory is not None
                and track_id is not None
                and frame_index is not None
                and appearance_memory.semantic_probe_due(
                    track_id,
                    person_crops[index],
                    frame_index,
                    detection_confidence,
                    camera_id=camera_id,
                )
            )
            if needs_reid_body_check or semantic_probe_due:
                reid_crop = person_crops[index]
                body_details = {}
                body_complete, missing_regions = assess_reid_body_completeness(
                    None,
                    debug_details=body_details,
                )
                pose_debug["reid_body_complete"] = body_complete
                pose_debug["reid_missing_regions"] = missing_regions
                pose_debug["reid_body_details"] = body_details
                pose_debug["reid_face_box"] = None
                if reid_crop is not None and reid_crop.size > 0:
                    semantic_result = pose_estimator.detect(reid_crop)
                    if semantic_result.pose_landmarks:
                        landmarks = semantic_result.pose_landmarks[0]
                        pose_debug["orientation"] = get_human_orientation(landmarks)
                        body_details = {}
                        body_complete, missing_regions = assess_reid_body_completeness(
                            landmarks,
                            orientation=pose_debug.get("orientation"),
                            touches_vertical_frame_boundary=detection_touches_vertical_frame_boundary(frame, box),
                            debug_details=body_details,
                        )
                        pose_debug["reid_body_complete"] = body_complete
                        pose_debug["reid_missing_regions"] = missing_regions
                        pose_debug["reid_body_details"] = body_details
                        # MediaPipe ran on the saved ReID crop itself here, so
                        # its landmarks are already in that crop's frame.
                        crop_height, crop_width = reid_crop.shape[:2]
                        pose_debug["reid_face_box"] = estimate_face_box(
                            landmarks,
                            crop_width=crop_width,
                            crop_height=crop_height,
                        )

        if (
            appearance_memory is not None
            and track_id is not None
            and frame_index is not None
            and reid_crop_allowed
        ):
            identity_id, reid_similarity, reidentified = appearance_memory.assign(
                track_id,
                person_crops[index],
                frame_index,
                excluded_identity_ids=active_identity_ids,
                camera_id=camera_id,
                detection_confidence=detection_confidence,
                orientation=pose_debug.get("orientation"),
                observed_at=observation_time,
                map_point=map_point,
                map_point_evidence=map_point_evidence,
                map_point_evidence_reason=map_point_evidence_reason,
                intake_body_complete=pose_debug.get("reid_body_complete"),
                intake_missing_regions=pose_debug.get("reid_missing_regions"),
                intake_body_details=pose_debug.get("reid_body_details"),
                intake_detection_box=box,
                intake_face_box=pose_debug.get("reid_face_box"),
                intake_body_bounds=intake_body_bounds,
                intake_occluder_boxes=intake_occluder_boxes,
            )
            if identity_id is not None:
                # The reservation is scoped to this camera. A matching track
                # in another camera may legitimately share the master ID.
                active_identity_ids.add(identity_id)
            assignment_metadata = appearance_memory.assignment_metadata(
                track_id,
                camera_id=camera_id,
            )
            temporary_group_id = appearance_memory.temporary_group(
                track_id,
                camera_id=camera_id,
            )
            reid_intake_count = appearance_memory.pending_count(track_id, camera_id=camera_id)
            reid_intake_required = appearance_memory.required_intake_count()

        ratio = None
        if anatomical_ratio_memory is not None:
            ratio = get_anatomical_ratio_from_memory(anatomical_ratio_memory, track_id, identity_id=identity_id)

        identity_metadata = appearance_memory.identity_metadata(identity_id) if appearance_memory is not None and identity_id is not None else {}

        # EXPERIMENTAL: shadow-mode only. Runs after identity resolution so the
        # learned height is attributed to the right person, and never feeds its
        # result back into `point`.
        if three_d_estimator is not None:
            run_three_d_level_shadow(
                three_d_estimator,
                pose_debug,
                method,
                camera_id,
                track_id,
                identity_id,
                (
                    "analyzing"
                    if temporary_group_id is not None
                    else assignment_metadata.get("identity_state")
                ),
                frame_index,
                map_projector,
                production_point=point,
            )

        # How far this ground point may be trusted, for the two consumers that
        # need opposite things from it: the tactical map weights by it, and the
        # cross-camera matcher refuses to let a soft point break a pairing.
        position_quality = classify_position_quality(
            method,
            box_evidence=map_point_evidence,
            strict_foot_visible=pose_debug.get("strict_foot_visible", True),
        )

        if point is None:
            standing_points.append({
                "point": None,
                "position_quality": POSITION_QUALITY_NONE,
                "position_quality_reason": method,
                "track_id": track_id,
                "identity_id": identity_id,
                "temporary_group_id": temporary_group_id,
                "identity_state": (
                    "analyzing"
                    if temporary_group_id is not None
                    else assignment_metadata.get("identity_state")
                ),
                "reid_similarity": reid_similarity,
                "reidentified": reidentified,
                "reid_confirmed": bool(assignment_metadata.get("appearance_confirmed", False)),
                "query_feature_space_id": assignment_metadata.get("query_feature_space_id"),
                "matched_slot": assignment_metadata.get("matched_slot"),
                "reid_intake_count": reid_intake_count,
                "reid_intake_required": reid_intake_required,
                "method": method,
                "ratio": ratio,
                "head_pitch": pose_debug.get("head_pitch"),
                "orientation": pose_debug.get("orientation"),
                "suppressed": False,
                "inside_tactical_map": inside_tactical_map,
                **identity_metadata,
            })
        else:
            standing_points.append({
                "point": (int(point[0]), int(point[1])),
                "position_quality": position_quality,
                "position_quality_reason": map_point_evidence_reason or method,
                "track_id": track_id,
                "identity_id": identity_id,
                "temporary_group_id": temporary_group_id,
                "identity_state": (
                    "analyzing"
                    if temporary_group_id is not None
                    else assignment_metadata.get("identity_state")
                ),
                "reid_similarity": reid_similarity,
                "reidentified": reidentified,
                "reid_confirmed": bool(assignment_metadata.get("appearance_confirmed", False)),
                "query_feature_space_id": assignment_metadata.get("query_feature_space_id"),
                "matched_slot": assignment_metadata.get("matched_slot"),
                "reid_intake_count": reid_intake_count,
                "reid_intake_required": reid_intake_required,
                "method": method,
                "ratio": ratio,
                "head_pitch": pose_debug.get("head_pitch"),
                "orientation": pose_debug.get("orientation"),
                "suppressed": False,
                "inside_tactical_map": inside_tactical_map,
                **identity_metadata,
            })

    return standing_points
