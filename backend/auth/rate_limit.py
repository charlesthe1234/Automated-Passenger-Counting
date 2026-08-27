"""In-memory login rate limiting for a single backend process.

Counters live in process memory only, so they reset when FastAPI restarts. That
is an accepted limitation for this deployment and is documented for operators.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

MAX_FAILURES = 5
FAILURE_WINDOW_SECONDS = 5 * 60
BLOCK_SECONDS = 15 * 60


@dataclass
class _Attempts:
    failures: list[float] = field(default_factory=list)
    blocked_until: float = 0.0


class LoginRateLimiter:
    def __init__(
        self,
        *,
        max_failures: int = MAX_FAILURES,
        window_seconds: int = FAILURE_WINDOW_SECONDS,
        block_seconds: int = BLOCK_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self._max_failures = max_failures
        self._window_seconds = window_seconds
        self._block_seconds = block_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, _Attempts] = {}

    def _prune(self, entry: _Attempts, now: float) -> None:
        cutoff = now - self._window_seconds
        entry.failures = [moment for moment in entry.failures if moment > cutoff]

    def retry_after_seconds(self, key: str) -> int:
        """Return remaining block seconds, or 0 when the key may attempt login."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return 0
            if entry.blocked_until > now:
                return int(entry.blocked_until - now) + 1
            return 0

    def register_failure(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            entry = self._entries.setdefault(key, _Attempts())
            self._prune(entry, now)
            entry.failures.append(now)
            if len(entry.failures) >= self._max_failures:
                entry.blocked_until = now + self._block_seconds
                entry.failures.clear()

    def register_success(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()


login_rate_limiter = LoginRateLimiter()
