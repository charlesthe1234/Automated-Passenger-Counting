#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def emit(state, **fields):
    print(
        json.dumps(
            {
                "type": "status",
                "state": state,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **fields,
            }
        ),
        flush=True,
    )


mode = os.environ.get("FAKE_CV_MODE", "ready")
command_log = os.environ.get("FAKE_CV_COMMAND_LOG")
if mode == "startup_failure":
    raise SystemExit(3)

emit("loading", loading_stage="Fake model loading", error=None)
if mode == "model_failure":
    emit("failed", error="Fake model failed to load")
    raise SystemExit(2)
emit("ready", loading_stage="Complete", error=None)

for line in sys.stdin:
    command = json.loads(line)
    if command_log:
        with Path(command_log).open("a", encoding="utf-8") as handle:
            handle.write(command["command"] + "\n")
    if command["command"] == "start":
        run_id = command.get("run_id") or "field_test_001"
        emit("starting", run_id=run_id)
        emit("running", run_id=run_id)
    elif command["command"] == "stop":
        emit("stopping")
        emit("ready", stopped_at=datetime.now(timezone.utc).isoformat())
    elif command["command"] == "shutdown":
        emit("offline", stopped_at=datetime.now(timezone.utc).isoformat())
        break
