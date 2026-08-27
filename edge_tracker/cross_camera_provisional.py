"""Location-first provisional identity coordination across two cameras.

The coordinator intentionally owns only short-lived geometric pairing state.
Identity allocation, track bindings, angle evidence, and promotion remain the
responsibility of ``AppearanceIdentityMemory``.
"""

import collections
from functools import lru_cache
import math

from identity_debug import identity_debug_enabled, identity_event, state_changed


class CrossCameraProvisionalCoordinator:
    """Create and refresh shared provisional IDs from stable spatial pairs."""

    def __init__(
        self,
        memory,
        max_distance_cm,
        max_skew_seconds,
        required_pair_frames=3,
        location_confirm_frames=12,
        hold_grace_frames=5,
        hold_max_frames=12,
        ambiguity_margin_cm=10.0,
        max_movement_disagreement_cm=None,
    ):
        self.memory = memory
        self.max_distance_cm = max(0.0, float(max_distance_cm))
        self.max_skew_seconds = max(0.0, float(max_skew_seconds))
        self.required_pair_frames = max(1, int(required_pair_frames))
        self.location_confirm_frames = max(
            self.required_pair_frames,
            int(location_confirm_frames),
        )
        self.hold_grace_frames = max(0, int(hold_grace_frames))
        self.hold_max_frames = max(
            self.required_pair_frames,
            int(hold_max_frames),
        )
        self.ambiguity_margin_cm = max(0.0, float(ambiguity_margin_cm))
        self.max_movement_disagreement_cm = (
            max(20.0, self.max_distance_cm * 0.5)
            if max_movement_disagreement_cm is None
            else max(0.0, float(max_movement_disagreement_cm))
        )
        self._update_index = 0
        self._pair_streaks = {}
        self._pair_holds = {}
        self._identity_last_match_update = {}
        # Diagnostics only: how many evaluations landed on each reason since the
        # last summary, so collapsing repeats into transitions does not lose the
        # sense of how often each outcome actually occurred.
        self._reason_counts = collections.Counter()
        self._reason_counts_flushed_at = 0

    def _apply_memory_hold(self, pair_key):
        apply_hold = getattr(self.memory, "hold_new_master_creation", None)
        if callable(apply_hold):
            return apply_hold(pair_key[0], pair_key[1], pair_key)
        return ()

    def _release_pair_hold(self, pair_key, reason):
        state = self._pair_holds.pop(pair_key, None)
        if state is None:
            return
        release_hold = getattr(self.memory, "release_new_master_hold", None)
        resumed_keys = ()
        if callable(release_hold):
            resumed_keys = release_hold(pair_key[0], pair_key[1], pair_key, reason)
        identity_event(
            "provisional_pair_hold_released",
            console=False,
            coordinator_update_index=self._update_index,
            pair_track_keys=pair_key,
            hold_started_update=state["started_update"],
            hold_duration_updates=self._update_index - state["started_update"],
            grace_started_update=state.get("grace_started_update"),
            resumed_track_keys=resumed_keys,
            reason=reason,
        )

    def _mark_pair_valid_for_hold(self, pair_key):
        state = self._pair_holds.get(pair_key)
        if state is None:
            held_keys = self._apply_memory_hold(pair_key)
            state = {
                "started_update": self._update_index,
                "last_valid_update": self._update_index,
                "grace_started_update": None,
                "held_track_keys": tuple(held_keys),
            }
            self._pair_holds[pair_key] = state
            identity_event(
                "provisional_pair_hold_started",
                console=False,
                coordinator_update_index=self._update_index,
                pair_track_keys=pair_key,
                held_track_keys=held_keys,
                grace_frames=self.hold_grace_frames,
                maximum_hold_frames=self.hold_max_frames,
            )
            return

        if state.get("grace_started_update") is not None:
            identity_event(
                "provisional_pair_hold_recovered",
                console=False,
                coordinator_update_index=self._update_index,
                pair_track_keys=pair_key,
                grace_started_update=state["grace_started_update"],
                recovery_updates=self._update_index - state["grace_started_update"],
            )
        state["last_valid_update"] = self._update_index
        state["grace_started_update"] = None

    def _mark_pair_failed_for_hold(self, pair_key, reason):
        state = self._pair_holds.get(pair_key)
        if state is None:
            return
        if reason in {
            "different_master",
            "ambiguous_pair",
            "not_selected_one_to_one",
            "movement_disagreement",
        }:
            self._release_pair_hold(pair_key, reason)
            return
        if state.get("grace_started_update") is None:
            state["grace_started_update"] = self._update_index
            identity_event(
                "provisional_pair_hold_grace_started",
                console=False,
                coordinator_update_index=self._update_index,
                pair_track_keys=pair_key,
                grace_frames=self.hold_grace_frames,
                reason=reason,
            )

    def _expire_pair_holds(self):
        for pair_key, state in list(self._pair_holds.items()):
            if self._update_index - state["started_update"] >= self.hold_max_frames:
                self._release_pair_hold(pair_key, "maximum_hold_expired")
                continue
            grace_started = state.get("grace_started_update")
            if (
                grace_started is not None
                and self._update_index - grace_started > self.hold_grace_frames
            ):
                self._release_pair_hold(pair_key, "grace_expired")

    def _mark_active_holds_unobserved(self, evaluated_pair_keys):
        for pair_key in list(self._pair_holds):
            if pair_key not in evaluated_pair_keys:
                self._mark_pair_failed_for_hold(pair_key, "pair_not_observed")

    @staticmethod
    def _track_key(observation):
        camera_id = observation.get("camera_id")
        track_id = observation.get("local_track_id")
        if camera_id is None or track_id is None:
            return None
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            return None
        return str(camera_id), track_id

    @classmethod
    def _normalize_observations(cls, camera_id, observations):
        normalized = []
        for observation in observations or ():
            if not isinstance(observation, dict):
                continue
            observation.setdefault("camera_id", camera_id)
            key = cls._track_key(observation)
            try:
                point = observation.get("point")
                point_x = float(point[0])
                point_y = float(point[1])
                captured_at = float(observation.get("captured_at", 0.0))
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            if key is None or not all(
                math.isfinite(value) for value in (point_x, point_y, captured_at)
            ):
                continue
            normalized.append(
                {
                    "observation": observation,
                    "track_key": key,
                    "point": (point_x, point_y),
                    "captured_at": captured_at,
                }
            )
        return normalized

    def _patch_identity(self, item):
        observation = item["observation"]
        identity_id = self.memory.lookup_track_key(item["track_key"])
        track_state_lookup = getattr(self.memory, "track_identity_state", None)
        if identity_id is not None and callable(track_state_lookup):
            state = track_state_lookup(item["track_key"])
        else:
            state = self.memory.identity_state(identity_id) if identity_id is not None else None
        location_managed = False
        location_managed_lookup = getattr(self.memory, "identity_is_location_managed", None)
        if identity_id is not None and callable(location_managed_lookup):
            location_managed = bool(location_managed_lookup(identity_id))
        is_temporary = identity_id is not None and identity_id < 0
        observation["identity_id"] = None if is_temporary else identity_id
        observation["temporary_group_id"] = (
            f"tmp_{abs(int(identity_id))}" if is_temporary else None
        )
        observation["identity_state"] = "analyzing" if is_temporary else state
        observation["location_managed"] = location_managed
        observation["location_pair_recent"] = bool(
            identity_id is not None
            and self._update_index
            - int(self._identity_last_match_update.get(identity_id, -10_000))
            <= 2
        )
        if is_temporary or state in ("provisional", "challenged"):
            observation["reid_confirmed"] = False
        elif state == "confirmed" and location_managed:
            observation["reid_confirmed"] = True
        else:
            # Preserve the normal ReID memory's feature-source decision.  A
            # color-histogram fallback must not be upgraded to TransReID-
            # confirmed merely because this coordinator observed the track.
            observation["reid_confirmed"] = bool(observation.get("reid_confirmed"))
        return identity_id, state

    @staticmethod
    def _best_pairs(left, right, candidate_costs):
        """Maximize pair count, then exactly minimize total spatial distance."""
        if not left or not right or not candidate_costs:
            return []
        if len(right) > len(left):
            swapped_costs = {
                (right_index, left_index): cost
                for (left_index, right_index), cost in candidate_costs.items()
            }
            return [
                (right_index, left_index)
                for left_index, right_index in CrossCameraProvisionalCoordinator._best_pairs(
                    right,
                    left,
                    swapped_costs,
                )
            ]
        if len(right) > 18:
            used_left = set()
            used_right = set()
            pairs = []
            for (left_index, right_index), _cost in sorted(
                candidate_costs.items(),
                key=lambda item: item[1],
            ):
                if left_index in used_left or right_index in used_right:
                    continue
                used_left.add(left_index)
                used_right.add(right_index)
                pairs.append((left_index, right_index))
            return pairs

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
                count, total_cost, pairs = solve(left_index + 1, used_right_mask | bit)
                candidate = (
                    count + 1,
                    total_cost + cost,
                    ((left_index, right_index),) + pairs,
                )
                if candidate[0] > best[0] or (
                    candidate[0] == best[0] and candidate[1] < best[1]
                ):
                    best = candidate
            return best

        return list(solve(0, 0)[2])

    def _candidate_pairs(self, left, right):
        candidate_costs = {}
        candidate_details = {}
        evaluated_pair_keys = set()
        for left_index, left_item in enumerate(left):
            left_identity, _ = self._patch_identity(left_item)
            for right_index, right_item in enumerate(right):
                right_identity, _ = self._patch_identity(right_item)
                pair_key = (left_item["track_key"], right_item["track_key"])
                evaluated_pair_keys.add(pair_key)
                time_skew = abs(left_item["captured_at"] - right_item["captured_at"])
                spatial_distance = math.dist(left_item["point"], right_item["point"])
                recoverable_split = False
                if (
                    left_identity is not None
                    and right_identity is not None
                    and left_identity != right_identity
                ):
                    can_recover = getattr(
                        self.memory,
                        "can_recover_provisional_pair",
                        None,
                    )
                    recoverable_split = bool(
                        callable(can_recover)
                        and can_recover(
                            left_item["track_key"],
                            right_item["track_key"],
                        )
                    )
                    if not recoverable_split:
                        self._log_pair_evaluation(
                            left_item,
                            right_item,
                            accepted=False,
                            reason="different_master",
                            distance_cm=spatial_distance,
                            time_skew_seconds=time_skew,
                        )
                        continue
                if time_skew > self.max_skew_seconds:
                    self._log_pair_evaluation(
                        left_item,
                        right_item,
                        accepted=False,
                        reason="time_skew",
                        distance_cm=spatial_distance,
                        time_skew_seconds=time_skew,
                    )
                    continue
                if spatial_distance > self.max_distance_cm:
                    self._log_pair_evaluation(
                        left_item,
                        right_item,
                        accepted=False,
                        reason="distance",
                        distance_cm=spatial_distance,
                        time_skew_seconds=time_skew,
                    )
                    continue
                candidate_costs[(left_index, right_index)] = spatial_distance
                candidate_details[(left_index, right_index)] = (
                    left_item,
                    right_item,
                    spatial_distance,
                    time_skew,
                )
        pairs = self._best_pairs(left, right, candidate_costs)
        selected_pairs = set(pairs)
        unambiguous_pairs = []
        for pair_indices, details in candidate_details.items():
            left_index, right_index = pair_indices
            left_item, right_item, spatial_distance, time_skew = details
            if pair_indices not in selected_pairs:
                self._log_pair_evaluation(
                    left_item,
                    right_item,
                    accepted=False,
                    reason="not_selected_one_to_one",
                    distance_cm=spatial_distance,
                    time_skew_seconds=time_skew,
                )
                continue
            selected_cost = candidate_costs[(left_index, right_index)]
            alternatives = [
                cost
                for (candidate_left, candidate_right), cost in candidate_costs.items()
                if (
                    (candidate_left == left_index and candidate_right != right_index)
                    or (candidate_right == right_index and candidate_left != left_index)
                )
            ]
            if alternatives and min(alternatives) - selected_cost < self.ambiguity_margin_cm:
                self._log_pair_evaluation(
                    left_item,
                    right_item,
                    accepted=False,
                    reason="ambiguous_pair",
                    distance_cm=spatial_distance,
                    time_skew_seconds=time_skew,
                    nearest_alternative_distance_cm=min(alternatives),
                    ambiguity_margin_cm=self.ambiguity_margin_cm,
                )
                continue
            unambiguous_pairs.append((left_index, right_index))
        return unambiguous_pairs, evaluated_pair_keys

    REASON_COUNT_FLUSH_UPDATES = 30

    def _flush_reason_counts(self):
        """Emit the aggregate that transition-only logging would otherwise lose.

        Recording one event per state change answers "what changed and when",
        but not "how often did this keep happening".  A periodic tally restores
        the second without restoring the volume.
        """
        if not identity_debug_enabled() or not self._reason_counts:
            return
        if self._update_index - self._reason_counts_flushed_at < self.REASON_COUNT_FLUSH_UPDATES:
            return
        # TEMP_IDENTITY_DEBUG
        identity_event(
            "provisional_pair_reason_counts",
            console=False,
            coordinator_update_index=self._update_index,
            updates_covered=self._update_index - self._reason_counts_flushed_at,
            reason_counts=dict(self._reason_counts),
            evaluations=sum(self._reason_counts.values()),
        )
        self._reason_counts.clear()
        self._reason_counts_flushed_at = self._update_index

    def _streak_before_update(self, pair_key):
        previous = self._pair_streaks.get(pair_key)
        if previous is None or previous[1] != self._update_index - 1:
            return 0
        return int(previous[0])

    def _log_pair_evaluation(
        self,
        left_item,
        right_item,
        *,
        accepted,
        reason,
        distance_cm,
        time_skew_seconds,
        streak_before=None,
        streak_after=None,
        **extra,
    ):
        pair_key = (left_item["track_key"], right_item["track_key"])
        # Behaviour, not logging: this must run for every evaluation regardless
        # of whether the evaluation is worth writing down.
        if not accepted:
            self._mark_pair_failed_for_hold(pair_key, reason)
        if streak_before is None:
            streak_before = self._streak_before_update(pair_key)
        if streak_after is None:
            streak_after = int(streak_before) + 1 if accepted else 0
        if not identity_debug_enabled():
            return
        self._reason_counts[str(reason)] += 1
        # One record per *transition*, not per evaluation.  A pair sitting in
        # the same decision for two hundred frames says nothing new each frame,
        # but the moment it moves different_master -> shared_identity_refreshed
        # -> distance is exactly what a reader is looking for.  A time-based
        # throttle would hide a state that changed and changed back inside the
        # window; keying on the decision itself cannot.
        decision_state = (
            bool(accepted),
            str(reason),
            # Crossing the streak requirement is a meaningful move even when
            # the reason is unchanged.
            int(streak_after) >= int(self.required_pair_frames),
        )
        if not state_changed("provisional_pair", pair_key, decision_state):
            return
        left_observation = left_item["observation"]
        right_observation = right_item["observation"]
        identity_event(
            "provisional_pair_evaluated",
            console=False,
            coordinator_update_index=self._update_index,
            left_track_key=left_item["track_key"],
            left_frame_index=left_observation.get("frame_index"),
            left_identity_id=left_observation.get("identity_id"),
            left_point=left_item["point"],
            right_track_key=right_item["track_key"],
            right_frame_index=right_observation.get("frame_index"),
            right_identity_id=right_observation.get("identity_id"),
            right_point=right_item["point"],
            distance_cm=distance_cm,
            distance_limit_cm=self.max_distance_cm,
            time_skew_seconds=time_skew_seconds,
            time_skew_limit_seconds=self.max_skew_seconds,
            accepted=bool(accepted),
            reason=str(reason),
            streak_before=int(streak_before),
            streak_after=int(streak_after),
            streak_required=self.required_pair_frames,
            **extra,
        )

    def _advance_streak(self, pair_key, left_point, right_point):
        previous = self._pair_streaks.get(pair_key)
        consecutive = previous is not None and previous[1] == self._update_index - 1
        streak_before = int(previous[0]) if consecutive else 0
        movement_disagreement = None
        reset_reason = None if consecutive else ("new_pair" if previous is None else "missed_update")
        if consecutive and len(previous) >= 4:
            left_motion = (
                left_point[0] - previous[2][0],
                left_point[1] - previous[2][1],
            )
            right_motion = (
                right_point[0] - previous[3][0],
                right_point[1] - previous[3][1],
            )
            movement_disagreement = math.dist(left_motion, right_motion)
            if movement_disagreement > self.max_movement_disagreement_cm:
                consecutive = False
                reset_reason = "movement_disagreement"
        streak = previous[0] + 1 if consecutive else 1
        self._pair_streaks[pair_key] = (
            streak,
            self._update_index,
            left_point,
            right_point,
        )
        return {
            "streak": int(streak),
            "streak_before": streak_before,
            "movement_disagreement_cm": movement_disagreement,
            "movement_disagreement_limit_cm": self.max_movement_disagreement_cm,
            "reset_reason": reset_reason,
        }

    def _prune_streaks(self):
        # A missed update breaks consecutiveness immediately, but retain the
        # small record briefly so a camera dropout never affects memory-owned
        # track bindings or provisional identities.
        stale_after = max(self.required_pair_frames, self.location_confirm_frames)
        cutoff = self._update_index - stale_after
        for pair_key, state in list(self._pair_streaks.items()):
            last_seen = state[1]
            if last_seen <= cutoff:
                self._pair_streaks.pop(pair_key, None)

    def update(self, camera_observations):
        """Update geometric evidence and patch observations with shared IDs.

        ``camera_observations`` is mutated in place and returned for convenient
        use by the caller. Only the first two camera streams are paired.
        """
        self._update_index += 1
        if not isinstance(camera_observations, dict):
            self._mark_active_holds_unobserved(set())
            self._expire_pair_holds()
            self._prune_streaks()
            return camera_observations

        camera_ids = list(camera_observations)[:2]
        normalized_by_camera = {
            camera_id: self._normalize_observations(
                camera_id,
                camera_observations.get(camera_id),
            )
            for camera_id in camera_ids
        }

        # Always refresh existing identity/state fields, including updates
        # made asynchronously by the ReID worker between camera frames.
        for items in normalized_by_camera.values():
            for item in items:
                self._patch_identity(item)

        if len(camera_ids) < 2:
            self._mark_active_holds_unobserved(set())
            self._expire_pair_holds()
            self._prune_streaks()
            return camera_observations

        left = normalized_by_camera[camera_ids[0]]
        right = normalized_by_camera[camera_ids[1]]
        pairs, evaluated_pair_keys = self._candidate_pairs(left, right)
        accepted_pair_keys = set()
        for left_index, right_index in pairs:
            left_item = left[left_index]
            right_item = right[right_index]
            pair_key = (left_item["track_key"], right_item["track_key"])
            accepted_pair_keys.add(pair_key)
            streak_result = self._advance_streak(
                pair_key,
                left_item["point"],
                right_item["point"],
            )
            pair_streak = streak_result["streak"]
            movement_contradiction = (
                streak_result["reset_reason"] == "movement_disagreement"
            )

            left_identity, _ = self._patch_identity(left_item)
            right_identity, _ = self._patch_identity(right_item)
            identity_id = (
                left_identity
                if left_identity is not None and left_identity == right_identity
                else None
            )
            if movement_contradiction:
                self._mark_pair_failed_for_hold(pair_key, "movement_disagreement")
            elif identity_id is None:
                self._mark_pair_valid_for_hold(pair_key)
            elif pair_key in self._pair_holds:
                self._release_pair_hold(pair_key, "shared_identity_established")

            provisional_requested = bool(
                identity_id is None
                and not movement_contradiction
                and pair_streak >= self.required_pair_frames
            )
            if provisional_requested:
                identity_id = self.memory.create_provisional_pair(
                    left_item["track_key"],
                    right_item["track_key"],
                )
                # Treat memory as authoritative in case it reused an existing
                # provisional ID or declined an unsafe merge.
                left_identity, _ = self._patch_identity(left_item)
                right_identity, _ = self._patch_identity(right_item)
                if left_identity is None or left_identity != right_identity:
                    identity_id = None
                else:
                    identity_id = left_identity
                self._release_pair_hold(
                    pair_key,
                    (
                        "provisional_group_established"
                        if identity_id is not None
                        else "provisional_group_declined"
                    ),
                )

            distance_value = math.dist(left_item["point"], right_item["point"])
            time_skew_value = abs(
                left_item["captured_at"] - right_item["captured_at"]
            )
            self._log_pair_evaluation(
                left_item,
                right_item,
                accepted=True,
                reason=(
                    "provisional_group_created_or_attached"
                    if provisional_requested and identity_id is not None
                    else "provisional_group_declined"
                    if provisional_requested
                    else "shared_identity_refreshed"
                    if identity_id is not None
                    else "streak_advanced"
                ),
                distance_cm=distance_value,
                time_skew_seconds=time_skew_value,
                streak_before=streak_result["streak_before"],
                streak_after=pair_streak,
                reset_reason=streak_result["reset_reason"],
                movement_disagreement_cm=streak_result[
                    "movement_disagreement_cm"
                ],
                movement_disagreement_limit_cm=streak_result[
                    "movement_disagreement_limit_cm"
                ],
                provisional_requested=provisional_requested,
                resulting_identity_id=identity_id,
                resulting_temporary_group_id=(
                    f"tmp_{abs(int(identity_id))}"
                    if identity_id is not None and identity_id < 0
                    else None
                ),
            )

            if identity_id is not None:
                observations = [
                    left_item["observation"],
                    right_item["observation"],
                ]
                self.memory.note_location_match(identity_id, pair_streak, observations)
                self._identity_last_match_update[identity_id] = self._update_index
                for item in (left_item, right_item):
                    _identity_id, state = self._patch_identity(item)
                    item["observation"]["location_pair_streak"] = pair_streak
                    item["observation"]["location_confirmed"] = bool(
                        pair_streak >= self.location_confirm_frames
                        or state == "confirmed"
                    )

        for pair_key, state in list(self._pair_streaks.items()):
            if (
                state[1] == self._update_index - 1
                and pair_key not in evaluated_pair_keys
                and pair_key not in accepted_pair_keys
            ):
                if not identity_debug_enabled():
                    continue
                self._reason_counts["pair_not_observed"] += 1
                if not state_changed(
                    "provisional_pair", pair_key, (False, "pair_not_observed", False)
                ):
                    continue
                identity_event(
                    "provisional_pair_evaluated",
                    console=False,
                    coordinator_update_index=self._update_index,
                    left_track_key=pair_key[0],
                    right_track_key=pair_key[1],
                    accepted=False,
                    reason="pair_not_observed",
                    streak_before=int(state[0]),
                    streak_after=0,
                    streak_required=self.required_pair_frames,
                    distance_limit_cm=self.max_distance_cm,
                    time_skew_limit_seconds=self.max_skew_seconds,
                )

        self._flush_reason_counts()
        self._mark_active_holds_unobserved(evaluated_pair_keys)
        self._expire_pair_holds()

        self._prune_streaks()
        return camera_observations
