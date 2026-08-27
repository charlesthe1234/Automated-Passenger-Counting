"""Pairing observations across cameras into fused people, and suppressing duplicates."""

import numpy as np

from constants import (
    DEFAULT_CROSS_CAMERA_MAX_SKEW_SECONDS,
    DEFAULT_DISPLAY_DUPLICATE_DISTANCE_CM,
    DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    DEFAULT_POSITION_PAIR_FRAMES,
    DEFAULT_POSITION_SPLIT_FRAMES,
    POSITION_QUALITY_NONE,
    SOFT_POSITION_GATE_MULTIPLIER,
)
from core_math import (
    distance_cm,
    is_soft_position,
    position_quality_rank,
)
from identity_debug import (
    identity_debug_detail_enabled,
    identity_debug_enabled,
    identity_event,
)

from fused_person import (
    _weighted_center,
    display_position_quality,
)
from fusion_diagnostics import (
    DIAGNOSTIC_REJECTION_REASONS,
    _candidate_flags,
    _identity_status,
    _physical_pairing_status,
    candidate_diagnostic_key,
    observation_diagnostic_key,
)


def _best_one_to_one_pairs(left, right, candidate_costs):
    """Maximize valid pair count, then minimize total spatial cost.

    Camera support is currently limited to two streams. The dynamic program
    avoids the order-dependent greedy failure where two shoulder-to-shoulder
    people are paired incorrectly. For unusually large crowds, it falls back
    to a deterministic global edge sort to avoid exponential state growth.
    """
    if not left or not right or not candidate_costs:
        return []
    if len(right) > len(left):
        swapped = {(right_index, left_index): cost for (left_index, right_index), cost in candidate_costs.items()}
        return [(right_index, left_index) for left_index, right_index in _best_one_to_one_pairs(right, left, swapped)]
    if len(right) > 18:
        used_left, used_right, pairs = set(), set(), []
        for (left_index, right_index), _cost in sorted(candidate_costs.items(), key=lambda item: item[1]):
            if left_index not in used_left and right_index not in used_right:
                used_left.add(left_index)
                used_right.add(right_index)
                pairs.append((left_index, right_index))
        return pairs

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(left_index, used_right_mask):
        if left_index >= len(left):
            return 0, 0.0, ()
        best = solve(left_index + 1, used_right_mask)
        for right_index in range(len(right)):
            bit = 1 << right_index
            cost = candidate_costs.get((left_index, right_index))
            if cost is None or used_right_mask & bit:
                continue
            matched_count, total_cost, pairs = solve(left_index + 1, used_right_mask | bit)
            candidate = (matched_count + 1, total_cost + cost, ((left_index, right_index),) + pairs)
            if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
        return best

    return list(solve(0, 0)[2])

def _fusion_pair_key(left_observation, right_observation):
    """Identify a camera pairing across frames.

    Keyed on the shared identity whenever both sides agree on one, because the
    identity is the thing being protected from splitting and it survives either
    camera's local tracker renumbering the person mid-walk.  Track numbers are
    the fallback, for observations no identity has claimed yet.
    """
    left_identity = left_observation.get("identity_id")
    if left_identity is not None and left_identity == right_observation.get("identity_id"):
        return ("master", left_identity)
    left_group = left_observation.get("temporary_group_id")
    if left_group is not None and left_group == right_observation.get("temporary_group_id"):
        return ("temporary", left_group)
    return (
        ("track", left_observation.get("camera_id"), left_observation.get("local_track_id")),
        ("track", right_observation.get("camera_id"), right_observation.get("local_track_id")),
    )

def _age_fusion_pair_memory(pair_memory, seen_keys, fused_keys, max_idle_frames):
    """Expire pairing history so the dict cannot grow for the life of the run.

    A pair that both cameras still see but which has stopped fusing loses its
    standing one frame at a time, so a pairing that genuinely broke stops
    drawing on credit it earned before the two people walked apart.
    """
    for key in list(pair_memory):
        entry = pair_memory[key]
        if key in fused_keys:
            entry["age"] = 0
            continue
        if key in seen_keys:
            entry["age"] = 0
            remaining = int(entry.get("fused_frames", 0)) - 1
            if remaining > 0:
                entry["fused_frames"] = remaining
            else:
                entry.pop("fused_frames", None)
            continue
        entry["age"] = int(entry.get("age", 0)) + 1
        if entry["age"] > max_idle_frames:
            pair_memory.pop(key, None)

def fuse_camera_points(
    camera_observations,
    max_distance_cm,
    max_skew_seconds=DEFAULT_CROSS_CAMERA_MAX_SKEW_SECONDS,
    require_reid=False,
    pair_memory=None,
    fusion_cycle_id=None,
):
    """Fuse two camera observation sets with appearance as a hard veto.

    Homography and capture time only nominate a pair. When ReID is enabled,
    both observations must already resolve to the same shared master ID;
    unknown or different identities remain separate rather than being merged
    merely because two people are physically close.

    Two rules keep one person from rendering as two dots.  A soft ground point
    -- inferred from body proportions, or measured in a clipped box -- may hold
    a pairing together but may never break one, because it may simply not be
    where the person is standing.  And a pair that has been fusing is only
    broken after the cameras disagree for several consecutive frames, since a
    single frame of homography distortion at a grazing angle is indis-
    tinguishable from two people genuinely walking apart.  ``pair_memory`` is
    the caller-owned dict holding that history; without one the function stays
    stateless and splits on the first disagreement, as it did before.
    """
    camera_ids = list(camera_observations)
    normalized = {}
    for camera_id, observations in camera_observations.items():
        normalized[camera_id] = []
        for observation in observations:
            if isinstance(observation, dict):
                candidate = dict(observation)
            else:
                candidate = {
                    "camera_id": camera_id,
                    "local_track_id": None,
                    "identity_id": None,
                    "reid_confirmed": False,
                    "point": tuple(observation),
                    "captured_at": 0.0,
                }
            try:
                captured_at = float(candidate.get("captured_at", 0.0))
            except (TypeError, ValueError):
                captured_at = float("nan")
            if not np.isfinite(captured_at):
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "fusion_observation_dropped",
                    throttle_key=(camera_id, candidate.get("local_track_id"), "invalid_time"),
                    throttle_seconds=1.0,
                    camera_id=camera_id,
                    local_track_id=candidate.get("local_track_id"),
                    master_id=candidate.get("identity_id"),
                    captured_at=candidate.get("captured_at"),
                    reason="invalid_time",
                )
                continue

            raw_point = candidate.get("point")
            point = None
            if raw_point is not None:
                try:
                    parsed = np.asarray(raw_point, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    parsed = None
                if parsed is not None and parsed.size == 2 and np.all(np.isfinite(parsed)):
                    point = (float(parsed[0]), float(parsed[1]))
                else:
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "fusion_observation_dropped",
                        throttle_key=(camera_id, candidate.get("local_track_id"), "invalid_point"),
                        throttle_seconds=1.0,
                        camera_id=camera_id,
                        local_track_id=candidate.get("local_track_id"),
                        master_id=candidate.get("identity_id"),
                        point=raw_point,
                        reason="invalid_point",
                    )
            # Kept even with no ground point.  Someone whose feet are hidden is
            # still one person and still the same person, and deleting them
            # here used to take them out of the cross-camera matcher too --
            # resetting the pairing streak and splitting their shared ID at the
            # exact moment the cameras most needed to agree about them.
            candidate["point"] = point
            if point is None:
                candidate["position_quality"] = POSITION_QUALITY_NONE
            candidate["captured_at"] = captured_at
            normalized[camera_id].append(candidate)

    def singleton(observation):
        point = observation.get("point")
        center = None if point is None else (float(point[0]), float(point[1]))
        return {
            "center": center,
            "points": [] if center is None else [center],
            "sources": [observation["camera_id"]],
            "observations": [observation],
            "identity_id": observation.get("identity_id"),
            "temporary_group_id": observation.get("temporary_group_id"),
            "identity_state": observation.get("identity_state"),
            "role": observation.get("role"),
        }

    def association_key(observation):
        identity_id = observation.get("identity_id")
        if identity_id is not None:
            return "master", identity_id
        temporary_group_id = observation.get("temporary_group_id")
        if temporary_group_id is not None:
            return "temporary", temporary_group_id
        return None

    if len(camera_ids) < 2:
        return [singleton(observation) for observations in normalized.values() for observation in observations]

    left = normalized[camera_ids[0]]
    right = normalized[camera_ids[1]]
    pair_history = pair_memory if pair_memory is not None else {}
    tolerance_cm = float(max_distance_cm) * SOFT_POSITION_GATE_MULTIPLIER
    seen_pair_keys = set()
    candidate_costs = {}
    # Diagnostics only. Assembled solely when logging is on, keyed by the same
    # (left_index, right_index) the algorithm uses, so a record can be updated
    # at whichever exit the candidate takes without restructuring the loop.
    diagnostics_on = identity_debug_enabled()
    candidate_records = {} if diagnostics_on else None

    def note(left_index, right_index, left_observation, right_observation, **fields):
        if candidate_records is None:
            return
        record = candidate_records.setdefault(
            (left_index, right_index),
            {
                "candidate_id": candidate_diagnostic_key(left_observation, right_observation),
                "physical_pair_key": [
                    observation_diagnostic_key(left_observation),
                    observation_diagnostic_key(right_observation),
                ],
                "left_camera": left_observation.get("camera_id"),
                "left_track": left_observation.get("local_track_id"),
                "left_master": left_observation.get("identity_id"),
                "left_temporary_group": left_observation.get("temporary_group_id"),
                "left_point": left_observation.get("point"),
                "left_quality": left_observation.get("position_quality"),
                "left_quality_reason": left_observation.get("position_quality_reason"),
                "left_reid_confirmed": bool(left_observation.get("reid_confirmed")),
                "left_identity_state": left_observation.get("identity_state"),
                "left_inside_tactical_map": bool(left_observation.get("inside_tactical_map")),
                "right_camera": right_observation.get("camera_id"),
                "right_track": right_observation.get("local_track_id"),
                "right_master": right_observation.get("identity_id"),
                "right_temporary_group": right_observation.get("temporary_group_id"),
                "right_point": right_observation.get("point"),
                "right_quality": right_observation.get("position_quality"),
                "right_quality_reason": right_observation.get("position_quality_reason"),
                "right_reid_confirmed": bool(right_observation.get("reid_confirmed")),
                "right_identity_state": right_observation.get("identity_state"),
                "right_inside_tactical_map": bool(right_observation.get("inside_tactical_map")),
                "identity_status": _identity_status(left_observation, right_observation),
                "distance_limit_cm": float(max_distance_cm),
                "tolerance_limit_cm": tolerance_cm,
                "passed_distance_gate": False,
                "passed_identity_gate": None,
                "eligible_for_assignment": False,
                "selected": False,
                "rejection_reason": None,
            },
        )
        record.update(fields)
    for left_index, left_observation in enumerate(left):
        for right_index, right_observation in enumerate(right):
            time_skew = abs(float(left_observation.get("captured_at", 0.0)) - float(right_observation.get("captured_at", 0.0)))
            pair_debug_key = (
                left_observation.get("camera_id"),
                left_observation.get("local_track_id"),
                right_observation.get("camera_id"),
                right_observation.get("local_track_id"),
            )
            same_known_master = (
                left_observation.get("identity_id") is not None
                and left_observation.get("identity_id") == right_observation.get("identity_id")
            )
            same_temporary_group = (
                left_observation.get("temporary_group_id") is not None
                and left_observation.get("temporary_group_id")
                == right_observation.get("temporary_group_id")
            )
            pair_key = _fusion_pair_key(left_observation, right_observation)
            seen_pair_keys.add(pair_key)
            history = pair_history.get(pair_key)
            # Either this function has watched the pair hold, or the
            # cross-camera coordinator has -- it tracks the same thing from its
            # own matching history, and it is the only witness for a caller
            # that keeps no pair memory of its own.
            previously_fused = bool(
                (history and history.get("fused_frames", 0) >= DEFAULT_POSITION_PAIR_FRAMES)
                or (
                    (same_known_master or same_temporary_group)
                    and left_observation.get("location_pair_recent")
                    and right_observation.get("location_pair_recent")
                )
            )
            soft_pair = is_soft_position(
                left_observation.get("position_quality")
            ) or is_soft_position(right_observation.get("position_quality"))

            left_point = left_observation.get("point")
            right_point = right_observation.get("point")
            note(
                left_index,
                right_index,
                left_observation,
                right_observation,
                fusion_cycle_id=fusion_cycle_id,
                time_skew_seconds=time_skew,
                time_skew_limit_seconds=float(max_skew_seconds),
                previously_fused=previously_fused,
                soft_pair=soft_pair,
                pair_history=dict(history) if history else None,
            )
            if left_point is None or right_point is None:
                # No shared geometry to judge by this frame.  Identity may hold
                # these two together -- that is the whole reason a foot-less
                # observation is still emitted -- but it may never invent a
                # pairing between two people nothing has ever linked.
                note(
                    left_index, right_index, left_observation, right_observation,
                    distance_cm=None,
                    physical_pairing_status=_physical_pairing_status(
                        None, max_distance_cm, tolerance_cm, previously_fused
                    ),
                )
                if not (same_known_master or same_temporary_group):
                    note(
                        left_index, right_index, left_observation, right_observation,
                        rejection_reason="no_geometry_and_no_shared_identity",
                    )
                    continue
                if time_skew > float(max_skew_seconds):
                    note(
                        left_index, right_index, left_observation, right_observation,
                        rejection_reason="time_skew",
                    )
                    continue
                # Ranked below every real geometric match so a camera that can
                # actually see this person is always preferred as the partner.
                candidate_costs[(left_index, right_index)] = tolerance_cm + float(max_distance_cm)
                note(
                    left_index, right_index, left_observation, right_observation,
                    passed_identity_gate=True,
                    eligible_for_assignment=True,
                    association_cost=candidate_costs[(left_index, right_index)],
                    geometric_cost=None,
                    time_tiebreak_cost=None,
                )
                continue

            spatial_distance = distance_cm(left_point, right_point)
            note(
                left_index, right_index, left_observation, right_observation,
                distance_cm=spatial_distance,
                physical_pairing_status=_physical_pairing_status(
                    spatial_distance, max_distance_cm, tolerance_cm, previously_fused
                ),
                passed_distance_gate=spatial_distance <= float(max_distance_cm),
            )
            if spatial_distance > float(max_distance_cm):
                tolerated_reason = None
                if spatial_distance > tolerance_cm:
                    # Too far apart to be one person under any reading of the
                    # evidence, however poor either point is.
                    tolerated_reason = None
                elif not previously_fused:
                    # Never fused, so there is no pairing to protect. A soft
                    # point earns patience, not the benefit of the doubt.
                    tolerated_reason = None
                elif soft_pair:
                    # Soft may sustain, never break. No violation is recorded:
                    # a point that might not be at the person's feet is not
                    # evidence that two people are standing apart.
                    tolerated_reason = "soft_position_may_not_break_pair"
                else:
                    violations = int((history or {}).get("violation_frames", 0)) + 1
                    pair_history.setdefault(pair_key, {})["violation_frames"] = violations
                    if violations < DEFAULT_POSITION_SPLIT_FRAMES:
                        # Both cameras claim a measured foot, so this may be a
                        # real separation -- but one frame of grazing-angle
                        # distortion looks exactly the same. Wait for it to
                        # persist before putting a second dot on the map.
                        tolerated_reason = "awaiting_consistent_separation"
                note(
                    left_index, right_index, left_observation, right_observation,
                    split_deferred=bool(tolerated_reason),
                    split_deferred_reason=tolerated_reason,
                    consecutive_violations=int(
                        pair_history.get(pair_key, {}).get("violation_frames", 0)
                    ),
                    violations_required=DEFAULT_POSITION_SPLIT_FRAMES,
                    passed_distance_gate=bool(tolerated_reason),
                    rejection_reason=None if tolerated_reason else "distance",
                )
                if tolerated_reason is not None:
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "fusion_pair_split_deferred",
                        throttle_key=(*pair_debug_key, tolerated_reason),
                        throttle_seconds=1.0,
                        reason=tolerated_reason,
                        left_camera=left_observation.get("camera_id"),
                        left_track=left_observation.get("local_track_id"),
                        left_master=left_observation.get("identity_id"),
                        left_quality=left_observation.get("position_quality"),
                        left_quality_reason=left_observation.get("position_quality_reason"),
                        right_camera=right_observation.get("camera_id"),
                        right_track=right_observation.get("local_track_id"),
                        right_master=right_observation.get("identity_id"),
                        right_quality=right_observation.get("position_quality"),
                        right_quality_reason=right_observation.get("position_quality_reason"),
                        distance_cm=spatial_distance,
                        distance_limit_cm=float(max_distance_cm),
                        tolerance_limit_cm=tolerance_cm,
                        consecutive_violations=int(
                            pair_history.get(pair_key, {}).get("violation_frames", 0)
                        ),
                        violations_required=DEFAULT_POSITION_SPLIT_FRAMES,
                    )
                else:
                    if (
                        (len(left) == 1 and len(right) == 1)
                        or same_known_master
                        or spatial_distance <= 2.0 * float(max_distance_cm)
                    ):
                        identity_event(
                            "fusion_candidate_rejected",
                            throttle_key=(*pair_debug_key, "distance"),
                            throttle_seconds=1.0,
                            reason="distance",
                            left_camera=left_observation.get("camera_id"),
                            left_track=left_observation.get("local_track_id"),
                            left_master=left_observation.get("identity_id"),
                            left_frame_index=left_observation.get("frame_index"),
                            left_point=left_observation.get("point"),
                            right_camera=right_observation.get("camera_id"),
                            right_track=right_observation.get("local_track_id"),
                            right_master=right_observation.get("identity_id"),
                            right_frame_index=right_observation.get("frame_index"),
                            right_point=right_observation.get("point"),
                            distance_cm=spatial_distance,
                            distance_limit_cm=float(max_distance_cm),
                            time_skew_seconds=time_skew,
                        )
                    continue
            elif history is not None:
                # Back in agreement. The streak must be consecutive, or a pair
                # that wobbles once every few minutes would eventually collect
                # enough violations to be split on evidence that was never
                # simultaneous.
                history.pop("violation_frames", None)
            # A pairing that has been holding tolerates a wider capture skew,
            # for the same reason it tolerates a wider distance: the cost of
            # dropping it for one ragged frame is a duplicate person.
            skew_limit_seconds = float(max_skew_seconds) * (2.0 if previously_fused else 1.0)
            if time_skew > skew_limit_seconds:
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "fusion_candidate_rejected",
                    throttle_key=(*pair_debug_key, "time_skew"),
                    throttle_seconds=1.0,
                    reason="time_skew",
                    left_camera=left_observation.get("camera_id"),
                    left_track=left_observation.get("local_track_id"),
                    left_master=left_observation.get("identity_id"),
                    left_frame_index=left_observation.get("frame_index"),
                    right_camera=right_observation.get("camera_id"),
                    right_track=right_observation.get("local_track_id"),
                    right_master=right_observation.get("identity_id"),
                    right_frame_index=right_observation.get("frame_index"),
                    distance_cm=spatial_distance,
                    time_skew_seconds=time_skew,
                    time_skew_limit_seconds=skew_limit_seconds,
                )
                note(
                    left_index, right_index, left_observation, right_observation,
                    rejection_reason="time_skew",
                    time_skew_limit_seconds=skew_limit_seconds,
                )
                continue
            if require_reid:
                left_identity = left_observation.get("identity_id")
                right_identity = right_observation.get("identity_id")
                left_association = association_key(left_observation)
                right_association = association_key(right_observation)
                left_state = left_observation.get("identity_state")
                right_state = right_observation.get("identity_state")
                if (
                    left_association is not None
                    and left_association == right_association
                    and left_association[0] == "temporary"
                ):
                    rejection_reason = None
                elif left_identity is None or right_identity is None:
                    rejection_reason = "identity_missing"
                elif left_identity != right_identity:
                    rejection_reason = "different_master"
                elif (
                    left_state in ("provisional", "challenged")
                    or right_state in ("provisional", "challenged")
                ):
                    # Location has already established a one-to-one shared
                    # provisional ID. Keep counting it once while comparable
                    # angle evidence is still being collected.
                    rejection_reason = None
                elif not left_observation.get("reid_confirmed") or not right_observation.get("reid_confirmed"):
                    rejection_reason = "reid_not_confirmed"
                else:
                    rejection_reason = None
                if rejection_reason is not None:
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "fusion_candidate_rejected",
                        throttle_key=(*pair_debug_key, rejection_reason),
                        throttle_seconds=1.0,
                        reason=rejection_reason,
                        left_camera=left_observation.get("camera_id"),
                        left_track=left_observation.get("local_track_id"),
                        left_master=left_identity,
                        left_frame_index=left_observation.get("frame_index"),
                        left_reid_confirmed=bool(left_observation.get("reid_confirmed")),
                        right_camera=right_observation.get("camera_id"),
                        right_track=right_observation.get("local_track_id"),
                        right_master=right_identity,
                        right_frame_index=right_observation.get("frame_index"),
                        right_reid_confirmed=bool(right_observation.get("reid_confirmed")),
                        distance_cm=spatial_distance,
                        time_skew_seconds=time_skew,
                    )
                    note(
                        left_index, right_index, left_observation, right_observation,
                        passed_identity_gate=False,
                        rejection_reason=rejection_reason,
                    )
                    continue
                note(
                    left_index, right_index, left_observation, right_observation,
                    passed_identity_gate=True,
                )
            time_tiebreak = time_skew / max(float(max_skew_seconds), 1e-9)
            candidate_costs[(left_index, right_index)] = spatial_distance + time_tiebreak * 1e-3
            note(
                left_index, right_index, left_observation, right_observation,
                eligible_for_assignment=True,
                geometric_cost=spatial_distance,
                time_tiebreak_cost=time_tiebreak * 1e-3,
                association_cost=candidate_costs[(left_index, right_index)],
            )

    pairs = _best_one_to_one_pairs(left, right, candidate_costs)
    paired_left = {left_index for left_index, _ in pairs}
    paired_right = {right_index for _, right_index in pairs}
    fused_pair_keys = set()
    fused_people = []
    for left_index, right_index in pairs:
        observations = [left[left_index], right[right_index]]
        points = [
            tuple(observation["point"])
            for observation in observations
            if observation.get("point") is not None
        ]
        identities = {observation.get("identity_id") for observation in observations}
        identities.discard(None)
        temporary_groups = {
            observation.get("temporary_group_id") for observation in observations
        }
        temporary_groups.discard(None)
        roles = {
            str(observation.get("role")).strip().lower()
            for observation in observations
            if observation.get("role") is not None
        }
        identity_states = {observation.get("identity_state") for observation in observations}
        if len(temporary_groups) == 1:
            fused_identity_state = "analyzing"
        elif "challenged" in identity_states:
            fused_identity_state = "challenged"
        elif "provisional" in identity_states:
            fused_identity_state = "provisional"
        else:
            fused_identity_state = identity_states.pop() if len(identity_states) == 1 else None
        # TEMP_IDENTITY_DEBUG
        identity_event(
            "fusion_pair_accepted",
            throttle_key=(
                observations[0].get("camera_id"),
                observations[0].get("local_track_id"),
                observations[1].get("camera_id"),
                observations[1].get("local_track_id"),
            ),
            throttle_seconds=1.0,
            left_camera=observations[0].get("camera_id"),
            left_track=observations[0].get("local_track_id"),
            left_master=observations[0].get("identity_id"),
            left_frame_index=observations[0].get("frame_index"),
            right_camera=observations[1].get("camera_id"),
            right_track=observations[1].get("local_track_id"),
            right_master=observations[1].get("identity_id"),
            right_frame_index=observations[1].get("frame_index"),
            distance_cm=distance_cm(points[0], points[1]) if len(points) == 2 else None,
            time_skew_seconds=abs(
                float(observations[0].get("captured_at", 0.0))
                - float(observations[1].get("captured_at", 0.0))
            ),
        )
        pair_key = _fusion_pair_key(observations[0], observations[1])
        fused_pair_keys.add(pair_key)
        entry = pair_history.setdefault(pair_key, {})
        # Capped: the streak only ever answers "has this pairing settled?", so
        # letting it climb for the length of a shift would buy a stale pairing
        # an unbounded amount of patience on the way back down.
        entry["fused_frames"] = min(
            int(entry.get("fused_frames", 0)) + 1, DEFAULT_POSITION_PAIR_FRAMES * 2
        )
        entry["age"] = 0
        fused_people.append({
            "center": _weighted_center(observations),
            "points": points,
            "sources": [observation["camera_id"] for observation in observations],
            "observations": observations,
            "identity_id": identities.pop() if len(identities) == 1 else None,
            "temporary_group_id": (
                temporary_groups.pop() if len(temporary_groups) == 1 else None
            ),
            "identity_state": fused_identity_state,
            "role": roles.pop() if len(roles) == 1 else None,
        })
    if candidate_records is not None:
        selected = set(pairs)
        detail = identity_debug_detail_enabled()
        for index_pair, record in sorted(candidate_records.items()):
            record["selected"] = index_pair in selected
            record["flags"] = _candidate_flags(record)
            # Normal runs record only candidates worth reading about;
            # --debug-fusion-detail records the whole matrix.  A plain distance
            # rejection is deliberately not "worth reading about": in a scene
            # with n people most of the n^2 candidate pairs are two different
            # people metres apart, and logging every one of them would bury the
            # handful of pairs that actually disagreed.
            interesting = (
                detail
                or record["flags"]
                or record["rejection_reason"] in DIAGNOSTIC_REJECTION_REASONS
            )
            if not interesting:
                continue
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "cross_camera_association_decision",
                console=False,
                **record,
            )

    _age_fusion_pair_memory(
        pair_history,
        seen_pair_keys,
        fused_pair_keys,
        DEFAULT_POSE_DROPOUT_TTL_FRAMES,
    )
    fused_people.extend(singleton(observation) for index, observation in enumerate(left) if index not in paired_left)
    fused_people.extend(singleton(observation) for index, observation in enumerate(right) if index not in paired_right)
    for camera_id in camera_ids[2:]:
        fused_people.extend(singleton(observation) for observation in normalized[camera_id])
    return fused_people

def _duplicate_authority(left, right):
    """Decide which of two co-located display people owns the dot.

    Returns ``(keeper, suppressed, reason)``, or ``(None, None, reason)`` when
    the pair must be left alone.  Quality decides first, because the whole
    point is that one camera can see the feet and the other cannot.  Everything
    below that is about refusing to invent certainty.
    """
    left_rank = position_quality_rank(display_position_quality(left))
    right_rank = position_quality_rank(display_position_quality(right))
    if left_rank != right_rank:
        keeper, suppressed = (left, right) if left_rank > right_rank else (right, left)
        return keeper, suppressed, "better_position_evidence"

    left_identity = left.get("identity_id")
    right_identity = right.get("identity_id")
    if (left_identity is None) != (right_identity is None):
        # One side has a master and the other has none.  There are not two
        # competing claims here, only one claim and one anonymous detection, so
        # keeping the named one loses nothing.
        keeper, suppressed = (left, right) if left_identity is not None else (right, left)
        return keeper, suppressed, "only_one_side_has_a_master"

    if left_identity is not None and left_identity != right_identity:
        # Equally good evidence, two different names.  Choosing between them
        # would be inventing a fact the location layer does not have, and the
        # loser's identity would silently vanish from the map.  Two dots is the
        # honest answer until appearance resolves it.
        return None, None, "equal_evidence_identity_conflict"

    def order(person):
        observation = (person.get("observations") or [{}])[0]
        return (
            -len(person.get("sources", ())),
            str(observation.get("camera_id")),
            str(observation.get("local_track_id")),
        )

    # Same master, or neither has one. Ordered explicitly so the survivor never
    # depends on where these two happened to land in the list.
    keeper, suppressed = sorted((left, right), key=order)
    return keeper, suppressed, "deterministic_tiebreak"

def suppress_display_duplicates(
    people,
    duplicate_distance_cm=DEFAULT_DISPLAY_DUPLICATE_DISTANCE_CM,
    fusion_cycle_id=None,
    stats=None,
):
    """Stop one physical person being drawn twice by two disagreeing cameras.

    Normal fusion has already run and has already applied every identity rule.
    What reaches here are the people it refused to combine -- and in practice a
    large share of those refusals are two cameras looking at the same person
    from opposite sides while the identity layer temporarily disagrees about
    the name, or has not issued one yet.  Geometry can settle that without
    appearance: two standing adults cannot put their foot centroids 35 cm
    apart, so two such points are one person drawn twice.

    This is a presentation decision and nothing more.  It returns a new list,
    mutates no observation, and touches neither the appearance memory nor the
    cross-camera coordinator -- both of which have already consumed their
    inputs by the time this runs.  A suppressed observation keeps its identity
    and stays reachable through the survivor's ``suppressed_duplicates`` for
    debugging.

    Two people are a duplicate on either of two independent grounds:

    * ``physical_proximity`` -- their feet are closer together than two people
      can stand, so whatever the names say, this is one body.
    * ``same_master`` -- they already carry the same master ID, which makes them
      one person by definition and needs no geometry at all.  This is the case
      where one camera holds a stale or coasted point for someone the other
      camera can still see: the two dots drift metres apart, every distance gate
      correctly declines them, and the map draws one person twice.  Grouping
      them here is what finally lets the existing quality ranking choose, so a
      live measurement wins over a stale one.

    Only people whose source cameras are disjoint are ever compared, which is
    what keeps two genuinely distinct people seen by one camera apart: they
    share a source, so they are never candidates.  That also means two tracks in
    a single camera wearing one master ID are deliberately left alone -- that is
    an identity fault, and hiding it here would only make it harder to find.
    """
    limit = float(duplicate_distance_cm)
    if stats is not None:
        stats.setdefault("suppressed", 0)
        stats.setdefault("unresolved", 0)
    if len(people) < 2:
        return list(people)

    survivors = list(people)
    suppressed_ids = set()
    # Physically-inseparable pairs are resolved before same-master ones, and
    # closest first within each group, so a certain duplicate is settled before
    # a more distant coincidence can claim either side of it.
    candidates = []
    for left_index in range(len(survivors)):
        for right_index in range(left_index + 1, len(survivors)):
            left, right = survivors[left_index], survivors[right_index]
            if not set(left.get("sources", ())).isdisjoint(set(right.get("sources", ()))):
                # Same camera contributed to both. Two nearby detections in one
                # view are two people (or a tracker duplicate), never this.
                continue
            same_master = (
                left.get("identity_id") is not None
                and left.get("identity_id") == right.get("identity_id")
            )
            left_center, right_center = left.get("center"), right.get("center")
            if left_center is None or right_center is None:
                # No geometry to compare. One shared master still makes these
                # one person, and the ranking below will prefer whichever side
                # actually has a position.
                if not same_master:
                    continue
                distance = None
            else:
                distance = distance_cm(left_center, right_center)
            if same_master:
                basis = "same_master"
            elif distance is not None and limit > 0.0 and distance <= limit:
                basis = "physical_proximity"
            else:
                continue
            candidates.append(
                (
                    0 if basis == "physical_proximity" else 1,
                    float("inf") if distance is None else distance,
                    left_index,
                    right_index,
                    basis,
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    for _order, _sort_distance, left_index, right_index, basis in candidates:
        if left_index in suppressed_ids or right_index in suppressed_ids:
            continue
        left, right = survivors[left_index], survivors[right_index]
        left_center, right_center = left.get("center"), right.get("center")
        distance = (
            None
            if left_center is None or right_center is None
            else distance_cm(left_center, right_center)
        )
        keeper, dropped, reason = _duplicate_authority(left, right)
        left_observation = (left.get("observations") or [{}])[0]
        right_observation = (right.get("observations") or [{}])[0]
        candidate_id = candidate_diagnostic_key(left_observation, right_observation)
        if keeper is None:
            if stats is not None:
                stats["unresolved"] += 1
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "cross_camera_display_duplicate_unresolved",
                throttle_key=(left.get("identity_id"), right.get("identity_id"), reason),
                throttle_seconds=1.0,
                reason=reason,
                duplicate_basis=basis,
                fusion_cycle_id=fusion_cycle_id,
                candidate_id=candidate_id,
                left_master=left.get("identity_id"),
                left_sources=list(left.get("sources", ())),
                left_quality=display_position_quality(left),
                right_master=right.get("identity_id"),
                right_sources=list(right.get("sources", ())),
                right_quality=display_position_quality(right),
                distance_cm=distance,
                duplicate_limit_cm=limit,
            )
            continue
        dropped_index = right_index if dropped is right else left_index
        suppressed_ids.add(dropped_index)
        if stats is not None:
            stats["suppressed"] += 1
        keeper_observation = (keeper.get("observations") or [{}])[0]
        dropped_observation = (dropped.get("observations") or [{}])[0]
        # TEMP_IDENTITY_DEBUG
        identity_event(
            "cross_camera_display_duplicate_suppressed",
            throttle_key=(
                keeper.get("identity_id"),
                dropped.get("identity_id"),
                reason,
            ),
            throttle_seconds=1.0,
            reason=reason,
            duplicate_basis=basis,
            # A same-master pair further apart than the fusion limit means the
            # identity layer is holding one ID across two places. The display is
            # tidied here, but the underlying fault stays greppable.
            same_master_split_cm=(
                distance
                if basis == "same_master" and distance is not None and distance > limit
                else None
            ),
            fusion_cycle_id=fusion_cycle_id,
            candidate_id=candidate_id,
            kept_camera=keeper_observation.get("camera_id"),
            kept_track=keeper_observation.get("local_track_id"),
            kept_master=keeper.get("identity_id"),
            kept_quality=display_position_quality(keeper),
            kept_quality_reason=keeper_observation.get("position_quality_reason"),
            kept_point=keeper.get("center"),
            kept_sources=list(keeper.get("sources", ())),
            suppressed_camera=dropped_observation.get("camera_id"),
            suppressed_track=dropped_observation.get("local_track_id"),
            suppressed_master=dropped.get("identity_id"),
            suppressed_quality=display_position_quality(dropped),
            suppressed_quality_reason=dropped_observation.get("position_quality_reason"),
            suppressed_point=dropped.get("center"),
            suppressed_sources=list(dropped.get("sources", ())),
            identity_disagreement=bool(
                keeper.get("identity_id") is not None
                and dropped.get("identity_id") is not None
                and keeper.get("identity_id") != dropped.get("identity_id")
            ),
            distance_cm=distance,
            duplicate_limit_cm=limit,
        )
        # Copy rather than mutate: the caller's list and every observation in it
        # belong to the fusion stage, and this stage must stay side-effect free.
        merged = dict(keeper)
        merged["suppressed_duplicates"] = list(keeper.get("suppressed_duplicates", ())) + [
            {
                "camera_id": dropped_observation.get("camera_id"),
                "local_track_id": dropped_observation.get("local_track_id"),
                "identity_id": dropped.get("identity_id"),
                "identity_state": dropped.get("identity_state"),
                "position_quality": display_position_quality(dropped),
                "position_quality_reason": dropped_observation.get("position_quality_reason"),
                "center": dropped.get("center"),
                "distance_cm": distance,
                "reason": reason,
                "basis": basis,
            }
        ]
        keeper_index = left_index if keeper is left else right_index
        survivors[keeper_index] = merged

    return [person for index, person in enumerate(survivors) if index not in suppressed_ids]
