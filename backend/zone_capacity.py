"""Zone occupancy parsing and capacity grading.

These three functions decide what a zone's occupancy *is* and whether it counts
as safe, warning, or critical. Both the live dashboard (crud.get_zone_status)
and the exported shift report (reports.shift_report) answer that question, and
they have to answer it identically -- a threshold changed in one copy and not
the other would let the report contradict the screen an operator acted on.

They lived as byte-identical private copies in both modules until they were
extracted here.
"""

from __future__ import annotations

import json
from typing import Any

# Percent-of-capacity boundaries. Shared so the dashboard and the report cannot
# drift apart on what "critical" means.
CRITICAL_PERCENT = 85
WARNING_PERCENT = 60


def parse_zone_counts(value: str | None) -> dict[str, Any]:
    """Decode the stored zone_counts JSON blob, tolerating absent or bad data."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def coerce_zone_count(value: Any) -> int | None:
    """Reduce one zone's stored value to a headcount.

    A zone may be recorded as a plain number or as a dict of sub-areas, which is
    summed. Booleans are rejected before the numeric branch because bool is a
    subclass of int and True would otherwise count as one person.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, dict):
        nested_counts = [coerce_zone_count(item) for item in value.values()]
        valid_counts = [count for count in nested_counts if count is not None]
        return sum(valid_counts) if valid_counts else None
    return None


def capacity_status(count: int, capacity: int | None) -> tuple[float | None, str]:
    """Grade an occupancy against a zone capacity.

    Returns (percent_used, status). An unset or non-positive capacity yields
    (None, "unknown") rather than a divide-by-zero or a false "safe".
    """
    if not capacity or capacity <= 0:
        return None, "unknown"

    percent_used = round((count / capacity) * 100, 1)
    if percent_used >= CRITICAL_PERCENT:
        return percent_used, "critical"
    if percent_used >= WARNING_PERCENT:
        return percent_used, "warning"
    return percent_used, "safe"
