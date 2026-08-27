from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from config import settings


class CameraStreamer:
    def __init__(self, camera_id: str, camera_url: str) -> None:
        self.camera_id = camera_id
        self.camera_url = camera_url
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._resolution: dict[str, int] = {"width": 1280, "height": 720}
        self._latest_jpeg = self._make_placeholder("Camera stream starting")
        self._camera_connected = False
        self._last_error: str | None = "Camera stream has not connected yet."
        self._last_frame_at: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-streamer-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._release_capture()

    def latest_frame(self) -> bytes:
        with self._lock:
            return bytes(self._latest_jpeg)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "camera_connected": self._camera_connected,
                "resolution": dict(self._resolution),
                "last_frame_at": self._last_frame_at,
                "last_error": self._last_error,
                "stream_path": f"/api/cameras/{self.camera_id}/stream",
            }

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                self._connect()
                if self._capture is None or not self._capture.isOpened():
                    self._sleep_with_stop(settings.camera_reconnect_seconds)
                    continue

            success, frame = self._capture.read()
            if not success or frame is None:
                self._set_placeholder("Camera read failed. Retrying connection.", connected=False)
                self._release_capture()
                self._sleep_with_stop(settings.camera_reconnect_seconds)
                continue

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(settings.camera_jpeg_quality)],
            )
            if not ok:
                self._set_placeholder("JPEG encoding failed. Waiting for next frame.", connected=False)
                continue

            height, width = frame.shape[:2]
            with self._lock:
                self._latest_jpeg = buffer.tobytes()
                self._camera_connected = True
                self._last_error = None
                self._last_frame_at = datetime.now(timezone.utc).isoformat()
                self._resolution = {"width": int(width), "height": int(height)}

    def _connect(self) -> None:
        self._release_capture()
        self._set_placeholder("Connecting to camera stream...", connected=False)
        capture = cv2.VideoCapture(self.camera_url)
        if capture.isOpened():
            self._capture = capture
            return

        capture.release()
        self._capture = None
        self._set_placeholder(
            f"{self.camera_id} unavailable. Retrying in {settings.camera_reconnect_seconds} seconds.",
            connected=False,
        )

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _set_placeholder(self, message: str, connected: bool) -> None:
        with self._lock:
            self._latest_jpeg = self._make_placeholder(message)
            self._camera_connected = connected
            self._last_error = None if connected else message
            self._last_frame_at = datetime.now(timezone.utc).isoformat()

    def _make_placeholder(self, message: str) -> bytes:
        width = int(self._resolution.get("width", 1280))
        height = int(self._resolution.get("height", 720))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (15, 23, 42)
        cv2.putText(frame, "CAG Passenger Monitoring", (48, 86), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (248, 250, 252), 3)
        cv2.putText(frame, self.camera_id, (48, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (103, 232, 249), 2)
        cv2.putText(frame, message, (48, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (203, 213, 225), 2)
        cv2.putText(
            frame,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            (48, height - 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (148, 163, 184),
            2,
        )
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return b""
        return buffer.tobytes()

    def _sleep_with_stop(self, seconds: int) -> None:
        self._stop_event.wait(max(1, seconds))


class CameraManager:
    def __init__(self, sources: dict[str, str], primary_camera_id: str) -> None:
        self._streamers = {
            camera_id: CameraStreamer(camera_id=camera_id, camera_url=camera_url)
            for camera_id, camera_url in sources.items()
        }
        self.primary_camera_id = (
            primary_camera_id if primary_camera_id in self._streamers else next(iter(self._streamers), primary_camera_id)
        )

    def start_all(self) -> None:
        for streamer in self._streamers.values():
            streamer.start()

    def stop_all(self) -> None:
        for streamer in self._streamers.values():
            streamer.stop()

    def get(self, camera_id: str) -> CameraStreamer | None:
        return self._streamers.get(camera_id)

    def primary(self) -> CameraStreamer:
        streamer = self.get(self.primary_camera_id)
        if streamer is None:
            raise RuntimeError("No camera streams are configured.")
        return streamer

    def all_status(self) -> list[dict[str, Any]]:
        return [streamer.status() for streamer in self._streamers.values()]


camera_manager = CameraManager(settings.camera_source_map, settings.primary_camera_id)


def mjpeg_frame_generator(streamer: CameraStreamer) -> Iterator[bytes]:
    while True:
        frame = streamer.latest_frame()
        yield b"--frame\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
        yield frame
        yield b"\r\n"
        time.sleep(0.05)
