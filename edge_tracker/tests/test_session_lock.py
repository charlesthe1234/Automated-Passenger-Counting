import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


from tests import EDGE_TRACKER_DIR

sys.path.insert(0, str(EDGE_TRACKER_DIR))

from session_lock import CvRuntimeBusyError, CvRuntimeLock


class SessionLockTests(unittest.TestCase):
    def test_only_one_entry_point_can_own_cv_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.lock"
            first = CvRuntimeLock("dashboard", path=path).acquire()
            try:
                with self.assertRaisesRegex(CvRuntimeBusyError, "dashboard"):
                    CvRuntimeLock("tester", path=path).acquire()
            finally:
                first.release()

            second = CvRuntimeLock("tester", path=path).acquire()
            second.release()

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork support")
    def test_forked_child_does_not_keep_parent_lock_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.lock"
            owner_script = """
import os
import sys
import time
from session_lock import CvRuntimeLock

runtime_lock = CvRuntimeLock("short-lived owner", path=sys.argv[1]).acquire()
child_pid = os.fork()
if child_pid == 0:
    os.close(1)
    os.close(2)
    time.sleep(10)
    os._exit(0)
print(child_pid, flush=True)
os._exit(0)
"""
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(EDGE_TRACKER_DIR)
            owner = subprocess.Popen(
                [sys.executable, "-c", owner_script, str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
            stdout, stderr = owner.communicate(timeout=3)
            self.assertEqual(owner.returncode, 0, stderr)
            child_pid = int(stdout.strip())

            try:
                replacement = CvRuntimeLock("replacement", path=path).acquire()
                replacement.release()
            finally:
                try:
                    os.kill(child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
