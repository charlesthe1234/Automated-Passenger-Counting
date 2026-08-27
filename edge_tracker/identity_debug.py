"""Opt-in event logger for diagnosing temporary cross-camera ID splits.

TEMP_IDENTITY_DEBUG: This module and every call tagged with the same marker are
temporary troubleshooting code. The logger is disabled by default and can be
removed after the identity-split investigation is complete.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time


_enabled = False
_detail_enabled = False
_log_path = None
_lock = threading.Lock()
_last_emitted = {}
_last_state = {}
_write_warning_emitted = False
_base_fields = {}


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _json_safe(to_list())
        except (TypeError, ValueError):
            pass
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_safe(scalar())
        except (TypeError, ValueError):
            pass
    return str(value)


def identity_debug_enabled():
    """True when any event would actually be recorded.

    Call sites guard expensive payload assembly with this.  ``identity_event``
    already returns immediately when logging is off, but Python evaluates every
    keyword argument *before* the call, so a caller building a per-candidate
    diagnostic pays for it on every frame of a production run unless it asks
    first.
    """
    return _enabled


def identity_debug_detail_enabled():
    """True when the caller should emit exhaustive per-candidate diagnostics.

    Separate from :func:`identity_debug_enabled` so the ordinary debug log stays
    readable: normal runs record only candidates that tripped a diagnostic flag,
    while this opts in to the complete candidate matrix every cycle.
    """
    return _enabled and _detail_enabled


def state_changed(scope, key, state):
    """True when ``state`` differs from the last one seen for ``key``.

    Lets a caller collapse a stream of identical decisions into one event per
    transition without a time-based throttle, which would hide a state that
    changed and changed back inside the throttle window.  Returns False (and
    records nothing) while logging is disabled, so callers must already be
    inside an :func:`identity_debug_enabled` guard.
    """
    if not _enabled:
        return False
    cache_key = (str(scope), str(key))
    with _lock:
        previous = _last_state.get(cache_key)
        if previous == state:
            return False
        _last_state[cache_key] = state
    return True


def configure_identity_debug(enabled=False, log_path=None, context=None, detail=False):
    """Enable the temporary logger and start a fresh JSONL trace."""

    global _enabled, _detail_enabled, _log_path, _last_emitted, _last_state
    global _write_warning_emitted, _base_fields
    with _lock:
        _enabled = bool(enabled)
        _detail_enabled = bool(detail)
        _log_path = Path(log_path) if _enabled and log_path else None
        _last_emitted = {}
        _last_state = {}
        _write_warning_emitted = False
        _base_fields = {
            str(key): _json_safe(value)
            for key, value in dict(context or {}).items()
        }
        if _log_path is not None:
            try:
                _log_path.parent.mkdir(parents=True, exist_ok=True)
                _log_path.write_text("", encoding="utf-8")
            except OSError as exc:
                print(f"[IDENTITY_DEBUG] Unable to create {_log_path}: {exc}", flush=True)
                _log_path = None

    if _enabled:
        identity_event("debug_logging_started", log_path=_log_path)


def identity_event(
    event,
    *,
    throttle_key=None,
    throttle_seconds=0.0,
    console=True,
    **fields,
):
    """Print and persist one identity decision event when debugging is enabled."""

    global _write_warning_emitted
    if not _enabled:
        return

    now_monotonic = time.monotonic()
    cache_key = None if throttle_key is None else (str(event), str(throttle_key))
    with _lock:
        if cache_key is not None:
            last_emitted = _last_emitted.get(cache_key)
            if last_emitted is not None and now_monotonic - last_emitted < float(throttle_seconds):
                return
            _last_emitted[cache_key] = now_monotonic

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            **_base_fields,
            **{str(key): _json_safe(value) for key, value in fields.items()},
        }
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if console:
            print(f"[IDENTITY_DEBUG] {line}", flush=True)

        if _log_path is not None:
            try:
                with _log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                if not _write_warning_emitted:
                    print(f"[IDENTITY_DEBUG] Unable to write {_log_path}: {exc}", flush=True)
                    _write_warning_emitted = True
