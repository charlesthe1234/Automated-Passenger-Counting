#!/usr/bin/env python3
"""Dashboard CV supervisor with one fresh pipeline process per run."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_STREAM = sys.stdout
# Keep the supervisor/manager protocol machine-readable. Production pipeline
# diagnostics and child stderr are forwarded to the backend's structured log.
sys.stdout = sys.stderr
print(f"CV worker Python: {sys.executable}", flush=True)

VALID_STATES = {"offline", "loading", "ready", "starting", "running", "stopping", "failed"}
LOG_EVIDENCE_DIRECTORY = Path(__file__).resolve().parent.parent / "LogEvidance"
_write_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(**payload) -> None:
    message = {"type": "status", "timestamp": utc_now(), **payload}
    with _write_lock:
        PROTOCOL_STREAM.write(json.dumps(message, separators=(",", ":")) + "\n")
        PROTOCOL_STREAM.flush()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--session",
        metavar="RUN_ID",
        help="Internal mode: load models and run exactly one dashboard session.",
    )
    parser.add_argument(
        "--session-script",
        type=Path,
        default=Path(__file__).resolve(),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--identity-debug-log",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-evidence-directory",
        type=Path,
        default=LOG_EVIDENCE_DIRECTORY,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _session_log_paths(log_directory: Path, run_id: str) -> tuple[Path, Path]:
    """Return the same paired debug-log paths used by the technical launcher."""

    safe_run_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in (str(run_id).strip() or "dashboard_run")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{safe_run_id}_{timestamp}"
    return (
        log_directory / f"{stem}.jsonl",
        log_directory / f"{stem}.console.log",
    )


def _send_command(process: subprocess.Popen[str], command: dict) -> bool:
    if process.poll() is not None or process.stdin is None:
        return False
    try:
        process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        process.stdin.flush()
    except (BrokenPipeError, OSError):
        return False
    return True


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _close_process_streams(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def run_session(run_id: str, identity_debug_log: Path | None = None) -> int:
    """Load a new model bundle, run one session, then exit completely."""

    stop_event = threading.Event()

    def set_state(value: str, **extra) -> None:
        emit(state=value, run_id=run_id, **extra)

    def read_commands() -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                set_state("failed", error="Session worker received invalid JSON command.")
                stop_event.set()
                continue
            name = command.get("command")
            if name == "stop":
                stop_event.set()
            elif name == "shutdown":
                stop_event.set()
                return
            else:
                set_state("failed", error=f"Unknown session command: {name!r}")
                stop_event.set()

        # The supervisor disappeared. Exit immediately so cameras, CUDA memory,
        # and the runtime lock cannot be orphaned behind the backend process.
        os._exit(0)

    reader = threading.Thread(target=read_commands, name="cv-session-command-reader", daemon=True)
    reader.start()

    models = None
    try:
        from launch_config import build_tracker_arguments, dashboard_launch_values
        from main_tracker import parse_args, preload_models, run_pipeline
        from session_lock import CvRuntimeLock

        with CvRuntimeLock("dashboard CV session"):
            set_state("loading", loading_stage="Reading production configuration", error=None)
            values = dashboard_launch_values(run_id, identity_debug_log)
            arguments = build_tracker_arguments(values)
            args = parse_args(arguments)

            def loading_stage(stage: str) -> None:
                if not stop_event.is_set():
                    set_state("loading", loading_stage=stage, error=None)

            models = preload_models(args, loading_stage=loading_stage)
            if stop_event.is_set():
                set_state("ready", loading_stage="Complete", stopped_at=utc_now(), error=None)
                return 0

            set_state("starting", loading_stage="Complete", started_at=utc_now(), error=None)

            def mark_running() -> None:
                if stop_event.is_set():
                    set_state("stopping")
                else:
                    set_state("running", error=None)

            run_pipeline(
                args,
                models,
                stop_event=stop_event,
                started_callback=mark_running,
            )
            set_state("ready", loading_stage="Complete", stopped_at=utc_now(), error=None)
            return 0
    except Exception as exc:
        set_state("failed", stopped_at=utc_now(), error=str(exc))
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if models is not None:
            models.close()


def run_supervisor(
    session_script: Path,
    log_evidence_directory: Path = LOG_EVIDENCE_DIRECTORY,
) -> int:
    """Relay manager commands to a newly spawned one-run process each time."""

    command_queue: queue.Queue[dict] = queue.Queue()
    shutdown_event = threading.Event()
    session_stop_requested = threading.Event()
    state_lock = threading.Lock()
    child_lock = threading.Lock()
    state = {"value": "offline", "run_id": None}
    child = {"process": None}

    def set_state(value: str, **extra) -> None:
        with state_lock:
            state["value"] = value
            if "run_id" in extra:
                state["run_id"] = extra["run_id"]
            current_run_id = state["run_id"]
        payload = {key: value for key, value in extra.items() if key != "run_id"}
        emit(state=value, run_id=current_run_id, **payload)

    def read_commands() -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                set_state("failed", error="Worker received invalid JSON command.")
                continue
            name = command.get("command")
            if name == "start":
                session_stop_requested.clear()
                command_queue.put(command)
            elif name in {"stop", "shutdown"}:
                session_stop_requested.set()
                if name == "shutdown":
                    shutdown_event.set()
                with child_lock:
                    process = child["process"]
                    if process is not None:
                        _send_command(process, {"command": name})
                if name == "stop":
                    set_state("stopping")
                else:
                    command_queue.put(command)
                    return
            else:
                set_state("failed", error=f"Unknown worker command: {name!r}")

        # EOF means FastAPI disappeared without its shutdown handshake.
        with child_lock:
            process = child["process"]
        _terminate_process(process)
        os._exit(0)

    reader = threading.Thread(target=read_commands, name="cv-worker-command-reader", daemon=True)
    reader.start()

    session_script = session_script.expanduser().resolve()
    if not session_script.is_file():
        set_state("failed", error=f"CV session script not found: {session_script}")
        return 1

    set_state("ready", loading_stage="Starts fresh for every run", error=None)
    try:
        while not shutdown_event.is_set():
            command = command_queue.get()
            if command.get("command") == "shutdown":
                break
            if command.get("command") != "start":
                continue

            with state_lock:
                current_state = state["value"]
            if current_state == "stopping" and session_stop_requested.is_set():
                set_state("ready", stopped_at=utc_now(), error=None)
                continue
            if current_state != "ready":
                set_state(
                    current_state,
                    error=f"Cannot start while worker is {current_state}.",
                )
                continue

            run_id = str(command.get("run_id") or "").strip()
            if not run_id:
                set_state("failed", error="A dashboard run ID is required.")
                continue

            identity_debug_log, runtime_log = _session_log_paths(
                log_evidence_directory,
                run_id,
            )
            try:
                log_evidence_directory.mkdir(parents=True, exist_ok=True)
                runtime_stream = runtime_log.open("w", encoding="utf-8")
            except OSError as exc:
                set_state(
                    "failed",
                    stopped_at=utc_now(),
                    error=f"Unable to create dashboard debug logs: {exc}",
                )
                continue
            print(
                f"Dashboard session debug logs: {identity_debug_log} and {runtime_log}",
                file=sys.stderr,
                flush=True,
            )

            set_state(
                "starting",
                run_id=run_id,
                started_at=utc_now(),
                stopped_at=None,
                error=None,
            )
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(session_script),
                        f"--session={run_id}",
                        f"--identity-debug-log={identity_debug_log}",
                    ],
                    cwd=str(Path(__file__).resolve().parent),
                    env=dict(os.environ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                runtime_stream.close()
                set_state("failed", stopped_at=utc_now(), error=f"Unable to start CV session: {exc}")
                continue

            with child_lock:
                child["process"] = process
                if session_stop_requested.is_set():
                    _send_command(process, {"command": "stop"})

            def relay_stderr(
                session_process: subprocess.Popen[str],
                session_runtime_stream,
            ) -> None:
                runtime_log_available = True
                try:
                    if session_process.stderr is None:
                        return
                    for line in session_process.stderr:
                        if runtime_log_available:
                            try:
                                session_runtime_stream.write(line)
                                session_runtime_stream.flush()
                            except OSError as exc:
                                runtime_log_available = False
                                print(
                                    f"Unable to continue dashboard console log {runtime_log}: {exc}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                        sys.stderr.write(line)
                        sys.stderr.flush()
                finally:
                    try:
                        session_runtime_stream.close()
                    except OSError:
                        pass

            stderr_thread = threading.Thread(
                target=relay_stderr,
                args=(process, runtime_stream),
                name=f"cv-session-{process.pid}-logs",
                daemon=True,
            )
            stderr_thread.start()

            if process.stdout is not None:
                for line in process.stdout:
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        print("Session worker emitted non-JSON protocol output", file=sys.stderr, flush=True)
                        continue
                    if message.get("type") != "status" or message.get("state") not in VALID_STATES:
                        print("Session worker emitted an invalid status object", file=sys.stderr, flush=True)
                        continue
                    updates = {
                        key: value
                        for key, value in message.items()
                        if key not in {"type", "state", "timestamp"}
                    }
                    set_state(message["state"], **updates)

            return_code = process.wait()
            stderr_thread.join(timeout=1.0)
            _close_process_streams(process)
            with child_lock:
                if child["process"] is process:
                    child["process"] = None

            if shutdown_event.is_set():
                break
            with state_lock:
                completed_state = state["value"]
            if return_code != 0 and completed_state != "failed":
                set_state(
                    "failed",
                    stopped_at=utc_now(),
                    error=f"CV session exited unexpectedly with code {return_code}.",
                )
            elif return_code == 0 and completed_state not in {"ready", "failed"}:
                set_state("ready", stopped_at=utc_now(), loading_stage="Complete", error=None)
    finally:
        with child_lock:
            process = child["process"]
        _terminate_process(process)
        if process is not None:
            _close_process_streams(process)
        set_state("offline", stopped_at=utc_now())
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.session:
        return run_session(args.session, identity_debug_log=args.identity_debug_log)
    return run_supervisor(args.session_script, args.log_evidence_directory)


if __name__ == "__main__":
    raise SystemExit(main())
