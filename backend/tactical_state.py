from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from threading import Lock

from models import TacticalPosition, TacticalStateCreate, TacticalStateRead


STALE_AFTER_SECONDS = 5.0


class TacticalStateStore:
    def __init__(self, stale_after_seconds: float = STALE_AFTER_SECONDS):
        self.stale_after_seconds = stale_after_seconds
        self._lock = Lock()
        self._states: dict[tuple[str, str], TacticalStateRead] = {}

    def update(self, payload: TacticalStateCreate) -> TacticalStateRead:
        now = datetime.now(timezone.utc)
        positions = self._classify_positions(
            payload.positions_cm,
            map_size_cm=payload.map_size_cm,
            outside_context_cm=payload.outside_context_cm,
        )
        evacuee_positions = [
            position
            for position in positions
            if position.role == "evacuee" and position.identity_state == "confirmed"
        ]
        inside_count = sum(1 for position in evacuee_positions if position.area == "inside")
        outside_visible_count = sum(1 for position in evacuee_positions if position.area == "outside_visible")
        analyzing_count = sum(
            1
            for position in positions
            if position.identity_state != "confirmed" or position.role is None
        )
        state = TacticalStateRead(
            timestamp=payload.timestamp or now,
            received_at=now,
            camera_id=payload.camera_id,
            run_id=payload.run_id,
            camera_source=payload.camera_source,
            people_count=inside_count,
            inside_count=inside_count,
            outside_visible_count=outside_visible_count,
            total_visible_count=len(positions),
            analyzing_count=analyzing_count,
            positions_cm=positions,
            map_size_cm=payload.map_size_cm,
            outside_context_cm=payload.outside_context_cm,
            has_data=True,
            stale=False,
            age_seconds=0,
        )
        with self._lock:
            self._states[(payload.camera_id, payload.run_id)] = state
        return state

    def clear_run(self, run_id: str) -> int:
        """Drop cached tactical state for one run, leaving other runs alone."""
        with self._lock:
            matching = [key for key in self._states if key[1] == run_id]
            for key in matching:
                del self._states[key]
        return len(matching)

    def _classify_positions(
        self,
        positions: list[TacticalPosition],
        *,
        map_size_cm: int,
        outside_context_cm: int,
    ) -> list[TacticalPosition]:
        visible_positions: list[TacticalPosition] = []
        min_visible = -outside_context_cm
        max_visible = map_size_cm + outside_context_cm

        for position in positions:
            if not isfinite(position.x) or not isfinite(position.y):
                continue

            is_inside = 0 <= position.x <= map_size_cm and 0 <= position.y <= map_size_cm
            is_visible_outside = (
                min_visible <= position.x <= max_visible
                and min_visible <= position.y <= max_visible
            )

            if is_inside:
                visible_positions.append(position.model_copy(update={"area": "inside"}))
            elif is_visible_outside:
                visible_positions.append(position.model_copy(update={"area": "outside_visible"}))

        return visible_positions

    def latest(self, camera_id: str | None = None, run_id: str | None = None) -> TacticalStateRead:
        with self._lock:
            candidates = [
                state
                for state in self._states.values()
                if (not camera_id or state.camera_id == camera_id) and (not run_id or state.run_id == run_id)
            ]

        if not candidates:
            return TacticalStateRead(camera_id=camera_id, run_id=run_id)

        latest_state = max(candidates, key=lambda state: state.received_at or datetime.min.replace(tzinfo=timezone.utc))
        return self._with_stale_status(latest_state)

    def _with_stale_status(self, state: TacticalStateRead) -> TacticalStateRead:
        if state.received_at is None:
            return state.model_copy(update={"stale": True, "age_seconds": None})

        age_seconds = max(0.0, (datetime.now(timezone.utc) - state.received_at).total_seconds())
        return state.model_copy(
            update={
                "age_seconds": round(age_seconds, 1),
                "stale": age_seconds > self.stale_after_seconds,
            }
        )


tactical_store = TacticalStateStore()
