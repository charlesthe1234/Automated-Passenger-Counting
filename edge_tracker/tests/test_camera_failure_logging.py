import threading
import unittest
from unittest.mock import patch

from camera_stream import LiveCamera


class _FailedCapture:
    def read(self):
        return False, None

    def isOpened(self):
        return True


class _ExplodingCapture:
    def read(self):
        raise RuntimeError("test capture failure")


class CameraFailureLoggingTests(unittest.TestCase):
    @staticmethod
    def _camera(capture):
        camera = LiveCamera.__new__(LiveCamera)
        camera.camera_id = "cam_test"
        camera.cap = capture
        camera.lock = threading.Lock()
        camera.running = True
        camera.ret = True
        camera.frame = object()
        camera.sequence = 12
        return camera

    def test_false_capture_result_logs_and_stops_reader(self):
        camera = self._camera(_FailedCapture())
        with patch("camera_stream.identity_event") as event:
            camera.update()

        self.assertFalse(camera.running)
        self.assertFalse(camera.ret)
        event.assert_called_once_with(
            "camera_stream_read_failed",
            camera_id="cam_test",
            reason="capture_returned_false",
            last_sequence=12,
            capture_opened=True,
        )

    def test_capture_exception_logs_exception_details(self):
        camera = self._camera(_ExplodingCapture())
        with patch("camera_stream.identity_event") as event:
            camera.update()

        self.assertFalse(camera.running)
        self.assertFalse(camera.ret)
        self.assertIsNone(camera.frame)
        event.assert_called_once_with(
            "camera_stream_read_failed",
            camera_id="cam_test",
            reason="capture_exception",
            exception_type="RuntimeError",
            exception_message="test capture failure",
            last_sequence=12,
        )


if __name__ == "__main__":
    unittest.main()
