import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from identity_debug import configure_identity_debug, identity_event


class IdentityDebugTests(unittest.TestCase):
    def tearDown(self):
        configure_identity_debug(False)

    def test_persist_only_event_does_not_flood_console(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.jsonl"
            configure_identity_debug(True, path, context={"run_id": "test"})
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                identity_event("per_frame_detail", console=False, frame_index=594)

            self.assertEqual(output.getvalue(), "")
            payloads = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(payloads[-1]["event"], "per_frame_detail")
            self.assertEqual(payloads[-1]["frame_index"], 594)
            self.assertNotIn("console", payloads[-1])


if __name__ == "__main__":
    unittest.main()
