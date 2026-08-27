"""Accessors over a fused person and its per-camera observations.

These read the shared person/observation dict shape. Both the fusion pass
and the diagnostics that describe it need them, so they sit below both and
depend on neither."""

import numpy as np

from constants import POSITION_QUALITY_NONE
from core_math import (
    position_quality_rank,
    position_quality_weight,
)


def normalize_tactical_entry(entry, fallback_index):
    """Accept either a bare ``(x, y)`` or a full per-camera observation entry.

    The tuple form is what this drew before quality existed, and several call
    sites and tests still pass it, so it keeps working and simply renders as an
    ungraded point.
    """
    if isinstance(entry, dict):
        point = entry.get("point")
        if point is None:
            return None
        return {
            "point": (float(point[0]), float(point[1])),
            "position_quality": entry.get("position_quality"),
            "position_quality_reason": entry.get("position_quality_reason"),
            "label": entry.get("label") or str(fallback_index),
            "provisional": bool(entry.get("provisional")),
        }
    return {
        "point": (float(entry[0]), float(entry[1])),
        "position_quality": None,
        "position_quality_reason": None,
        "label": str(fallback_index),
        "provisional": False,
    }

def _authoritative_observations(observations):
    """Keep only the observations of the highest grade present.

    Camera authority is deliberately absolute rather than a steep weighting.  A
    camera that cannot see the feet is not a weak witness to where the person
    is standing -- it is reporting a different place entirely, usually the body
    of whoever is standing in front.  Letting it pull the dot even slightly
    toward that place moves the dot somewhere nobody is.  When one camera has a
    measured foot and the other has a guess, the measurement simply wins.

    Returns the positioned observations that share the best grade, so two hard
    cameras still average together exactly as before.
    """
    positioned = [
        observation for observation in observations if observation.get("point") is not None
    ]
    if not positioned:
        return []
    best_rank = max(
        position_quality_rank(observation.get("position_quality"))
        for observation in positioned
    )
    return [
        observation
        for observation in positioned
        if position_quality_rank(observation.get("position_quality")) == best_rank
    ]

def _weighted_center(observations):
    """Where the map draws this person, given who could actually see their feet.

    The authoritative cameras are chosen first, then averaged among themselves.
    Observations carrying no grade at all weigh nothing, which leaves callers
    that never set one -- and pairings where neither camera can see feet --
    with the plain mean this used to return unconditionally.
    """
    authoritative = _authoritative_observations(observations)
    if not authoritative:
        return None
    weighted_sum = np.zeros(2, dtype=float)
    total_weight = 0.0
    points = []
    for observation in authoritative:
        point = observation["point"]
        points.append((float(point[0]), float(point[1])))
        weight = position_quality_weight(observation.get("position_quality"))
        if weight <= 0.0:
            continue
        weighted_sum += weight * np.asarray(point, dtype=float)
        total_weight += weight
    if total_weight > 0.0:
        center = weighted_sum / total_weight
        return (float(center[0]), float(center[1]))
    center = np.mean(np.asarray(points, dtype=float), axis=0)
    return (float(center[0]), float(center[1]))

def display_position_quality(person):
    """Best grade any camera achieved for this display person."""
    qualities = [
        observation.get("position_quality")
        for observation in person.get("observations", ())
        if observation.get("point") is not None
    ]
    if not qualities:
        return POSITION_QUALITY_NONE
    return max(qualities, key=position_quality_rank)

def _display_authority_camera(person):
    """Which camera the drawn position actually came from."""
    authoritative = _authoritative_observations(person.get("observations", ()))
    cameras = sorted({str(o.get("camera_id")) for o in authoritative if o.get("camera_id")})
    return "+".join(cameras) if cameras else None
