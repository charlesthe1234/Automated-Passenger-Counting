"""Cross-entry-point ownership lock for the Ubuntu CV runtime."""

from __future__ import annotations

import os
import weakref
from pathlib import Path


LOCK_PATH = Path("/tmp/cag-passenger-monitoring-cv.lock")
_ACTIVE_LOCKS = weakref.WeakSet()


def _close_inherited_locks_in_child() -> None:
    """Drop lock descriptors inherited by a newly forked child process.

    A child must never keep an entry-point lock alive after its owning parent
    exits. Merely marking the descriptor non-inheritable only protects exec;
    it does not protect against libraries that create workers with fork.
    """
    for runtime_lock in list(_ACTIVE_LOCKS):
        handle = runtime_lock._handle
        if handle is None:
            continue
        try:
            handle.close()
        except OSError:
            pass
        runtime_lock._handle = None
    _ACTIVE_LOCKS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_inherited_locks_in_child)


class CvRuntimeBusyError(RuntimeError):
    pass


class CvRuntimeLock:
    def __init__(self, owner: str, path: Path = LOCK_PATH):
        self.owner = str(owner)
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> "CvRuntimeLock":
        if self._handle is not None:
            return self
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - Ubuntu always provides fcntl.
            raise RuntimeError("The CV runtime lock requires a POSIX operating system.") from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            current_owner = handle.read().strip() or "another CV process"
            handle.close()
            raise CvRuntimeBusyError(
                f"Computer vision is already owned by {current_owner}. "
                "Stop that entry point before starting this one."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{self.owner} (pid {os.getpid()})")
        handle.flush()
        self._handle = handle
        _ACTIVE_LOCKS.add(self)
        return self

    def release(self) -> None:
        if self._handle is None:
            _ACTIVE_LOCKS.discard(self)
            return
        try:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            _ACTIVE_LOCKS.discard(self)

    def __enter__(self) -> "CvRuntimeLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
