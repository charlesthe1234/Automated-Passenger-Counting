import unittest
from types import SimpleNamespace

from fusion_diagnostics import build_frame_performance_snapshot


class FramePerformanceLoggingTests(unittest.TestCase):
    def test_snapshot_uses_displayed_fps_and_includes_people_counts(self):
        contexts = [
            SimpleNamespace(
                camera_id="cam_1",
                frame_index=41,
                fps=9.8764,
                raw_detection_count=3,
                tracked_person_count=2,
                tactical_person_count=2,
                suppressed_track_count=1,
            ),
            SimpleNamespace(
                camera_id="cam_2",
                frame_index=42,
                fps=8.1236,
                raw_detection_count=2,
                tracked_person_count=2,
                tactical_person_count=1,
                suppressed_track_count=0,
            ),
        ]
        fused_people = [
            {"identity_id": 1, "identity_state": "confirmed", "role": "evacuee"},
            {"identity_id": 2, "identity_state": "confirmed", "role": "cag"},
        ]

        snapshot = build_frame_performance_snapshot(contexts, fused_people)

        self.assertEqual(snapshot["frame_index"], 42)
        self.assertEqual(snapshot["system_fps"], 8.124)
        self.assertEqual(snapshot["people_count"], 2)
        self.assertEqual(snapshot["fused_people_count"], 2)
        self.assertEqual(snapshot["confirmed_people_count"], 2)
        self.assertEqual(snapshot["confirmed_evacuee_count"], 1)
        self.assertEqual(snapshot["camera_fps"], {"cam_1": 9.876, "cam_2": 8.124})
        self.assertEqual(snapshot["camera_person_counts"], {"cam_1": 2, "cam_2": 2})
        self.assertEqual(snapshot["cameras"][0]["suppressed_track_count"], 1)

    def test_snapshot_handles_uninitialised_fps(self):
        context = SimpleNamespace(camera_id="cam_1", frame_index=1, fps=0.0)

        snapshot = build_frame_performance_snapshot([context], [])

        self.assertEqual(snapshot["system_fps"], 0.0)
        self.assertEqual(snapshot["people_count"], 0)
        self.assertEqual(snapshot["camera_person_counts"], {"cam_1": 0})


if __name__ == "__main__":
    unittest.main()
