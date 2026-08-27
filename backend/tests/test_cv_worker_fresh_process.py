import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from tests import BACKEND_DIR, TESTS_DIR

ROOT = BACKEND_DIR.parent
CV_WORKER = ROOT / "edge_tracker" / "cv_worker.py"
FAKE_SESSION = TESTS_DIR / "fake_cv_session.py"


class CvWorkerFreshProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.pid_log = Path(self.temporary.name) / "session-pids.log"
        self.log_directory = Path(self.temporary.name) / "LogEvidance"
        environment = dict(os.environ)
        environment["FAKE_CV_SESSION_PID_LOG"] = str(self.pid_log)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(CV_WORKER),
                "--session-script",
                str(FAKE_SESSION),
                "--log-evidence-directory",
                str(self.log_directory),
            ],
            cwd=str(ROOT / "edge_tracker"),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.statuses = queue.Queue()

        def read_statuses():
            if self.process.stdout is None:
                return
            for line in self.process.stdout:
                self.statuses.put(json.loads(line))

        self.reader = threading.Thread(target=read_statuses, daemon=True)
        self.reader.start()

    def tearDown(self):
        if self.process.poll() is None:
            self._send("shutdown")
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()
        self.temporary.cleanup()

    def _send(self, command, run_id=None):
        payload = {"command": command}
        if run_id is not None:
            payload["run_id"] = run_id
        self.assertIsNotNone(self.process.stdin)
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _wait_for(self, state, run_id=None):
        while True:
            message = self.statuses.get(timeout=3.0)
            if message.get("state") != state:
                continue
            if run_id is not None and message.get("run_id") != run_id:
                continue
            return message

    def test_each_start_uses_a_different_session_process(self):
        self._wait_for("ready")

        self._send("start", "run_one")
        self._wait_for("running", "run_one")
        self._send("stop")
        self._wait_for("ready", "run_one")

        self._send("start", "run_two")
        self._wait_for("running", "run_two")
        self._send("stop")
        self._wait_for("ready", "run_two")

        sessions = self.pid_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual([line.split(maxsplit=1)[1] for line in sessions], ["run_one", "run_two"])
        self.assertNotEqual(sessions[0].split()[0], sessions[1].split()[0])

        identity_logs = sorted(self.log_directory.glob("*.jsonl"))
        console_logs = sorted(self.log_directory.glob("*.console.log"))
        self.assertEqual(len(identity_logs), 2)
        self.assertEqual(len(console_logs), 2)
        self.assertEqual(
            {path.name.removesuffix(".jsonl") for path in identity_logs},
            {path.name.removesuffix(".console.log") for path in console_logs},
        )
        self.assertEqual(
            {json.loads(path.read_text(encoding="utf-8"))["run_id"] for path in identity_logs},
            {"run_one", "run_two"},
        )
        self.assertTrue(
            all("fake session diagnostic" in path.read_text(encoding="utf-8") for path in console_logs)
        )


if __name__ == "__main__":
    unittest.main()
