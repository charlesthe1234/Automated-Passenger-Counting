"""Thread-safe lifecycle manager for the persistent Ubuntu CV worker."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
from datetime import datetime, timezone
from typing import Any

from config import Settings, settings


logger = logging.getLogger(__name__)
VALID_STATES = {"offline", "loading", "ready", "starting", "running", "stopping", "failed"}
RTSP_CREDENTIAL_PATTERN = re.compile(r"(rtsp://)[^\s/@]+(?::[^\s/@]*)?@", re.IGNORECASE)


class CvTransitionError(RuntimeError):
    pass


def _worker_environment_from(app_settings: Settings) -> dict[str, str]:
    """Environment for the managed CV worker process.

    pydantic-settings reads backend/.env into the Settings object without
    exporting it to os.environ, so the service token is passed explicitly here
    rather than relying on inheritance. It travels in the environment, never in
    the command line, so it cannot appear in process listings.
    """
    environment = dict(os.environ)
    service_token = app_settings.cv_service_token.get_secret_value()
    if service_token:
        environment["CV_SERVICE_TOKEN"] = service_token
    return environment


class CvManager:
    def _worker_environment(self) -> dict[str, str]:
        return _worker_environment_from(self.settings)

    def __init__(self, app_settings: Settings):
        self.settings = app_settings
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._reader_threads: list[threading.Thread] = []
        self._shutdown_requested = False
        self._status: dict[str, Any] = {
            "state": "offline",
            "run_id": None,
            "started_at": None,
            "stopped_at": None,
            "pid": None,
            "loading_stage": None,
            "error": None,
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_log_text(value: str) -> str:
        return RTSP_CREDENTIAL_PATTERN.sub(r"\1<credentials>@", str(value))

    def _log_event(self, event: str, **fields: Any) -> None:
        safe_fields = {
            key: self._safe_log_text(value) if isinstance(value, str) else value
            for key, value in fields.items()
        }
        payload = {"timestamp": self._utc_now(), "component": "cv_manager", "event": event, **safe_fields}
        logger.info("CV %s: %s", event, safe_fields)
        log_path = self.settings.cv_worker_log_path
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")
        except OSError:
            logger.exception("Unable to append CV structured log at %s", log_path)

    def _set_status_locked(self, state: str, **updates: Any) -> None:
        if state not in VALID_STATES:
            raise ValueError(f"Invalid CV state {state!r}")
        previous = self._status["state"]
        self._status.update(updates)
        self._status["state"] = state
        if previous != state:
            self._log_event("state_transition", previous_state=previous, state=state, error=updates.get("error"))

    def start_worker(self) -> None:
        with self._lock:
            if not self.settings.cv_enabled:
                self._set_status_locked("offline", error=None, loading_stage="Disabled")
                return
            if self._process is not None and self._process.poll() is None:
                return
            python_path = self.settings.cv_worker_python_path
            script_path = self.settings.cv_worker_script_path
            if not python_path.is_file() or not os.access(python_path, os.X_OK):
                self._set_status_locked("failed", error=f"CV Python environment not found: {python_path}")
                return
            if not script_path.is_file():
                self._set_status_locked("failed", error=f"CV worker script not found: {script_path}")
                return

            self._shutdown_requested = False
            self._set_status_locked(
                "loading",
                error=None,
                loading_stage="Starting worker process",
                started_at=None,
                stopped_at=None,
            )
            try:
                process = subprocess.Popen(
                    [str(python_path), "-u", str(script_path)],
                    cwd=str(script_path.parent),
                    env=self._worker_environment(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                self._set_status_locked("failed", error=f"Unable to start CV worker: {exc}")
                return
            self._process = process
            self._status["pid"] = process.pid
            self._log_event("worker_started", pid=process.pid)

            stdout_thread = threading.Thread(
                target=self._read_protocol,
                args=(process,),
                name="cv-worker-protocol",
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._read_worker_logs,
                args=(process,),
                name="cv-worker-logs",
                daemon=True,
            )
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process,),
                name="cv-worker-monitor",
                daemon=True,
            )
            self._reader_threads = [stdout_thread, stderr_thread, monitor_thread]
            for thread in self._reader_threads:
                thread.start()

    def _read_protocol(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._log_event("protocol_error", message="Worker emitted non-JSON protocol output")
                continue
            if message.get("type") != "status" or message.get("state") not in VALID_STATES:
                self._log_event("protocol_error", message="Worker emitted an invalid status object")
                continue
            with self._lock:
                if process is not self._process:
                    return
                updates = {
                    key: message[key]
                    for key in (
                        "run_id",
                        "started_at",
                        "stopped_at",
                        "loading_stage",
                        "error",
                    )
                    if key in message
                }
                if isinstance(updates.get("error"), str):
                    updates["error"] = self._safe_log_text(updates["error"])
                self._set_status_locked(message["state"], **updates)

    def _read_worker_logs(self, process: subprocess.Popen[str]) -> None:
        if process.stderr is None:
            return
        for line in process.stderr:
            message = self._safe_log_text(line.rstrip())
            if message:
                self._log_event("worker_log", message=message)

    def _monitor_process(self, process: subprocess.Popen[str]) -> None:
        return_code = process.wait()
        with self._lock:
            if process is not self._process:
                return
            previous_state = self._status["state"]
            self._process = None
            self._status["pid"] = None
            if self._shutdown_requested:
                self._set_status_locked("offline", stopped_at=self._utc_now())
            elif previous_state != "failed":
                self._set_status_locked(
                    "failed",
                    stopped_at=self._utc_now(),
                    error=f"CV worker exited unexpectedly with code {return_code}.",
                )
            self._log_event("worker_exited", return_code=return_code)
        self._close_process_streams(process)

    def _close_process_streams(self, process: subprocess.Popen[str]) -> None:
        current = threading.current_thread()
        for thread in list(self._reader_threads):
            if thread is not current and thread.is_alive():
                thread.join(timeout=1.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def _write_command_locked(self, command: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise CvTransitionError("The CV worker is not running.")
        encoded = json.dumps(command, separators=(",", ":")) + "\n"
        try:
            with self._write_lock:
                process.stdin.write(encoded)
                process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._set_status_locked("failed", error=f"Lost communication with CV worker: {exc}")
            raise CvTransitionError(self._status["error"]) from exc

    def start_session(self, run_id: str | None = None) -> dict[str, Any]:
        requested_run_id = str(run_id).strip() if run_id else None
        with self._lock:
            state = self._status["state"]
            active_run_id = self._status.get("run_id")
            if state in {"starting", "running"}:
                if requested_run_id is None or requested_run_id == active_run_id:
                    return self.status()
                raise CvTransitionError(
                    f"CV session {active_run_id!r} is already {state}; a second session cannot start."
                )
            if state != "ready":
                raise CvTransitionError(f"CV session cannot start while worker is {state}.")
            next_run_id = requested_run_id or active_run_id
            self._write_command_locked({"command": "start", "run_id": next_run_id})
            self._set_status_locked(
                "starting",
                run_id=next_run_id,
                started_at=self._utc_now(),
                stopped_at=None,
                error=None,
            )
            return self.status()

    def stop_session(self) -> dict[str, Any]:
        with self._lock:
            state = self._status["state"]
            if state in {"ready", "offline", "failed", "loading"}:
                return self.status()
            if state == "stopping":
                return self.status()
            self._write_command_locked({"command": "stop"})
            self._set_status_locked("stopping")
            return self.status()

    def mqtt_broker_reachable(self) -> bool:
        try:
            with socket.create_connection(
                (self.settings.mqtt_host, self.settings.mqtt_port), timeout=0.2
            ):
                return True
        except OSError:
            return False

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self._status)
        snapshot.update(
            {
                "ready": snapshot["state"] == "ready",
                "running": snapshot["state"] == "running",
                "mqtt_broker_reachable": self.mqtt_broker_reachable(),
            }
        )
        return snapshot

    def shutdown(self, timeout: float = 15.0) -> None:
        with self._lock:
            process = self._process
            if process is None:
                self._set_status_locked("offline", pid=None, stopped_at=self._utc_now())
                return
            self._shutdown_requested = True
            if process.poll() is None:
                try:
                    self._write_command_locked({"command": "shutdown"})
                except CvTransitionError:
                    pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._log_event("worker_forced_stop", pid=process.pid)
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        with self._lock:
            if self._process is process:
                self._process = None
            self._set_status_locked("offline", pid=None, stopped_at=self._utc_now())
        self._close_process_streams(process)


cv_manager = CvManager(settings)
