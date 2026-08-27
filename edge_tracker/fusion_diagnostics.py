"""Human-readable descriptions of why the fusion pass made each decision."""

import numpy as np

from constants import (
    DEFAULT_DISPLAY_DUPLICATE_DISTANCE_CM,
    POSITION_QUALITY_HARD,
)
from identity_debug import (
    identity_debug_enabled,
    identity_event,
)

from dashboard_payload import dashboard_eligible_people
from fused_person import (
    _display_authority_camera,
    display_position_quality,
)


DIAGNOSTIC_REJECTION_REASONS = frozenset({
    "different_master",
    "identity_missing",
    "reid_not_confirmed",
    "no_geometry_and_no_shared_identity",
    "time_skew",
})

def observation_diagnostic_key(observation):
    """Stable per-observation name: ``cam_1#17``."""
    return f"{observation.get('camera_id')}#{observation.get('local_track_id')}"

def candidate_diagnostic_key(left_observation, right_observation):
    """Stable name for one cross-camera candidate, independent of identity.

    Deliberately built from cameras and local track numbers rather than master
    IDs: the master is the thing under investigation when these logs are read,
    so keying on it would make a candidate change name at exactly the moment
    the reader most needs to follow it.
    """
    return (
        f"{observation_diagnostic_key(left_observation)}"
        f"~{observation_diagnostic_key(right_observation)}"
    )

def _identity_status(left_observation, right_observation):
    """Name the identity relationship, separately from the physical one."""
    left_identity = left_observation.get("identity_id")
    right_identity = right_observation.get("identity_id")
    if left_identity is not None and left_identity == right_identity:
        return "same_master"
    if left_identity is not None and right_identity is not None:
        return "different_master"
    left_group = left_observation.get("temporary_group_id")
    right_group = right_observation.get("temporary_group_id")
    if left_group is not None and left_group == right_group:
        return "same_temporary_group"
    if left_identity is None and right_identity is None:
        return "identity_missing_both"
    return "identity_missing_one"

def _physical_pairing_status(distance, max_distance_cm, tolerance_cm, previously_fused):
    """Name the physical relationship, separately from the identity one.

    The two questions -- "is this one body?" and "what is this body called?" --
    are answered by different evidence and fail independently, so the log names
    them independently.  Reading a rejection as "spatially_compatible +
    different_master" is immediately actionable; reading it as
    "fusion_candidate_rejected" is not.
    """
    if distance is None:
        return "no_shared_geometry"
    if distance <= float(max_distance_cm):
        return "established_pair" if previously_fused else "spatially_compatible"
    if distance <= tolerance_cm:
        return "within_tolerance_band"
    return "spatially_incompatible"

def _candidate_flags(record, duplicate_distance_cm=DEFAULT_DISPLAY_DUPLICATE_DISTANCE_CM):
    """Derive the searchable suspicious-condition names for one candidate.

    Kept as flags on the candidate record rather than separate events: every
    one of these is a property of a candidate that is already being written, so
    emitting them again as their own events would double the volume to say the
    same thing twice.  The flag strings still appear verbatim in the JSON, so
    grepping for a condition works exactly as if it were an event name.
    """
    flags = []
    distance = record.get("distance_cm")
    identity_status = record.get("identity_status")
    if distance is None:
        if identity_status in ("identity_missing_one", "identity_missing_both"):
            flags.append("no_geometry_identity_missing")
        return flags
    close = distance <= float(record.get("distance_limit_cm") or 0.0)
    left_hard = record.get("left_quality") == POSITION_QUALITY_HARD
    right_hard = record.get("right_quality") == POSITION_QUALITY_HARD
    if close and identity_status == "different_master":
        flags.append("close_different_master")
        if left_hard and right_hard:
            flags.append("close_hard_hard_identity_conflict")
    if close and identity_status in ("identity_missing_one", "identity_missing_both"):
        flags.append("close_identity_missing")
    if close and record.get("rejection_reason") == "reid_not_confirmed":
        flags.append("close_reid_not_confirmed")
    if distance <= float(duplicate_distance_cm) and identity_status == "different_master":
        flags.append("possible_duplicate_master")
    if record.get("split_deferred"):
        flags.append("established_pair_split_deferred")
    return flags

def _person_diagnostic_summary(person):
    """Compact description of one display person, for the cycle summary."""
    return {
        "displayed_id": (
            f"ID {person['identity_id']}"
            if person.get("identity_id") is not None
            else "Analyzing"
            if person.get("temporary_group_id") is not None
            else "P"
        ),
        "master_id": person.get("identity_id"),
        "temporary_group_id": person.get("temporary_group_id"),
        "identity_state": person.get("identity_state"),
        "center": person.get("center"),
        "sources": list(person.get("sources", ())),
        "position_quality": display_position_quality(person),
        "authority_camera": _display_authority_camera(person),
        "tracks": [observation_diagnostic_key(o) for o in person.get("observations", ())],
    }

def log_fusion_cycle_summary(
    fusion_cycle_id,
    camera_observations,
    people_before,
    people_after,
    unresolved_count=0,
):
    """One record answering 'what happened this cycle' without a manual join.

    Deliberately a single event rather than a snapshot plus a display result:
    the two would repeat most of their content, and the question being asked --
    what did each camera see, what came out, and what did the display stage
    change -- is one question.
    """
    if not identity_debug_enabled():
        return
    cameras = {}
    for camera_id, observations in (camera_observations or {}).items():
        cameras[str(camera_id)] = [
            {
                "track": observation_diagnostic_key(observation),
                "master_id": observation.get("identity_id"),
                "temporary_group_id": observation.get("temporary_group_id"),
                "identity_state": observation.get("identity_state"),
                "point": observation.get("point"),
                "position_quality": observation.get("position_quality"),
                "position_quality_reason": observation.get("position_quality_reason"),
                "reid_confirmed": bool(observation.get("reid_confirmed")),
                "inside_tactical_map": bool(observation.get("inside_tactical_map")),
            }
            for observation in observations or ()
        ]
    suppressed = len(people_before) - len(people_after)
    # TEMP_IDENTITY_DEBUG
    identity_event(
        "fusion_cycle_summary",
        console=False,
        fusion_cycle_id=fusion_cycle_id,
        camera_observation_counts={key: len(value) for key, value in cameras.items()},
        cameras=cameras,
        fused_count_before_suppression=len(people_before),
        fused_count_after_suppression=len(people_after),
        suppressed_duplicate_count=suppressed,
        unresolved_duplicate_count=unresolved_count,
        people=[_person_diagnostic_summary(person) for person in people_after],
        displayed_ids=[
            person.get("identity_id")
            for person in people_after
            if person.get("identity_id") is not None
        ],
    )

def build_frame_performance_snapshot(contexts, fused_people):
    """Return one extraction-friendly performance record for the current loop.

    ``context.fps`` is the same EMA-smoothed number painted onto each camera
    window.  ``system_fps`` uses the slower active camera because the combined
    pipeline waits for every camera future before fusion and publication.
    """

    camera_stats = []
    active_fps = []
    for context in contexts:
        try:
            fps = float(getattr(context, "fps", 0.0) or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        if not np.isfinite(fps) or fps < 0.0:
            fps = 0.0
        if fps > 0.0:
            active_fps.append(fps)

        camera_stats.append(
            {
                "camera_id": str(context.camera_id),
                "frame_index": int(getattr(context, "frame_index", 0) or 0),
                "fps": round(fps, 3),
                "raw_detection_count": int(
                    getattr(context, "raw_detection_count", 0) or 0
                ),
                "tracked_person_count": int(
                    getattr(context, "tracked_person_count", 0) or 0
                ),
                "tactical_person_count": int(
                    getattr(context, "tactical_person_count", 0) or 0
                ),
                "suppressed_track_count": int(
                    getattr(context, "suppressed_track_count", 0) or 0
                ),
            }
        )

    dashboard_people = dashboard_eligible_people(fused_people)
    return {
        "frame_index": max(
            (camera["frame_index"] for camera in camera_stats), default=0
        ),
        "system_fps": round(min(active_fps), 3) if active_fps else 0.0,
        "people_count": len(fused_people),
        "fused_people_count": len(fused_people),
        "confirmed_people_count": len(dashboard_people),
        "confirmed_evacuee_count": sum(
            1 for person in dashboard_people if person.get("role") == "evacuee"
        ),
        "camera_fps": {
            camera["camera_id"]: camera["fps"] for camera in camera_stats
        },
        "camera_person_counts": {
            camera["camera_id"]: camera["tracked_person_count"]
            for camera in camera_stats
        },
        "cameras": camera_stats,
    }
