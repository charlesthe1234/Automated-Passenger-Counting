#!/usr/bin/env python3
"""One-run protocol process used by the CV supervisor lifecycle test."""

import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True)
parser.add_argument("--identity-debug-log", type=Path)
args = parser.parse_args()

pid_log = os.environ.get("FAKE_CV_SESSION_PID_LOG")
if pid_log:
    with Path(pid_log).open("a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()} {args.session}\n")

if args.identity_debug_log:
    args.identity_debug_log.write_text(
        json.dumps({"event": "fake_debug_logging_started", "run_id": args.session}) + "\n",
        encoding="utf-8",
    )
print(f"fake session diagnostic for {args.session}", file=sys.stderr, flush=True)

emit("loading", run_id=args.session, loading_stage="Fake fresh model loading", error=None)
emit("starting", run_id=args.session)
emit("running", run_id=args.session, error=None)

for line in sys.stdin:
    command = json.loads(line)
    if command.get("command") in {"stop", "shutdown"}:
        emit(
            "ready",
            run_id=args.session,
            loading_stage="Complete",
            stopped_at=datetime.now(timezone.utc).isoformat(),
            error=None,
        )
        break
