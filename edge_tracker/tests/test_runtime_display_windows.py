import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import cv2
import numpy as np

from tactical_render import (
    create_runtime_display_windows,
    create_tactical_map,
    draw_tactical_grid,
    draw_top_left_text,
)


class RuntimeDisplayWindowTests(unittest.TestCase):
    @patch("tactical_render.cv2.getWindowImageRect")
    @patch("tactical_render.cv2.resizeWindow")
    @patch("tactical_render.cv2.namedWindow")
    def test_each_launch_uses_the_same_fixed_resizable_window_sizes(
        self, named_window, resize_window, image_rect
    ):
        contexts = [
            SimpleNamespace(camera_id="cam_1"),
            SimpleNamespace(camera_id="cam_2"),
        ]

        first_names = create_runtime_display_windows(contexts)
        second_names = create_runtime_display_windows(contexts)

        expected_names = [
            "Camera cam_1",
            "Map cam_1",
            "Camera cam_2",
            "Map cam_2",
            "Combined tactical map",
        ]
        self.assertEqual(first_names, expected_names)
        self.assertEqual(second_names, expected_names)
        self.assertEqual(named_window.call_count, len(expected_names) * 2)
        for named_call in named_window.call_args_list:
            self.assertEqual(named_call.args[1], cv2.WINDOW_NORMAL)

        one_launch_sizes = [
            call("Camera cam_1", 1280, 720),
            call("Map cam_1", 600, 600),
            call("Camera cam_2", 1280, 720),
            call("Map cam_2", 600, 600),
            call("Combined tactical map", 600, 600),
        ]
        self.assertEqual(
            resize_window.call_args_list,
            one_launch_sizes + one_launch_sizes,
        )
        image_rect.assert_not_called()

    def test_grid_coordinates_and_borders_stay_inside_non_divisible_canvas(self):
        height, width = 113, 157
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)

        draw_tactical_grid(canvas, grid_columns=5, grid_rows=4)

        expected_x = [round(index * (width - 1) / 5) for index in range(1, 5)]
        expected_y = [round(index * (height - 1) / 4) for index in range(1, 4)]
        for pixel_x in expected_x:
            self.assertTrue(np.array_equal(canvas[10, pixel_x], (195, 195, 195)))
        for pixel_y in expected_y:
            self.assertTrue(np.array_equal(canvas[pixel_y, 10], (195, 195, 195)))

        self.assertTrue(np.all(canvas[0] == 40))
        self.assertTrue(np.all(canvas[-1] == 40))
        self.assertTrue(np.all(canvas[:, 0] == 40))
        self.assertTrue(np.all(canvas[:, -1] == 40))

    def test_grid_survives_repeated_small_and_large_resizing(self):
        canvas = np.full((600, 600, 3), 245, dtype=np.uint8)
        draw_tactical_grid(canvas, grid_columns=5, grid_rows=5)

        for target_size in (120, 173, 300, 463, 900, 173):
            interpolation = (
                cv2.INTER_AREA if target_size < canvas.shape[0] else cv2.INTER_NEAREST
            )
            resized = cv2.resize(
                canvas,
                (target_size, target_size),
                interpolation=interpolation,
            )
            self.assertLess(int(resized[0].mean()), 200)
            self.assertLess(int(resized[-1].mean()), 200)
            self.assertLess(int(resized[:, 0].mean()), 200)
            self.assertLess(int(resized[:, -1].mean()), 200)

            for index in range(1, 5):
                coordinate = round(index * (target_size - 1) / 5)
                low = max(0, coordinate - 2)
                high = min(target_size, coordinate + 3)
                self.assertLess(float(resized[:, low:high].mean()), 239.0)
                self.assertLess(float(resized[low:high, :].mean()), 239.0)

    def test_display_changes_do_not_move_tactical_coordinates(self):
        tactical_map = create_tactical_map(
            [(240.0, 240.0)],
            map_size_cm=480.0,
            color=(0, 160, 0),
        )

        self.assertEqual(tactical_map.shape, (600, 600, 3))
        # A 240 cm coordinate on a 480 cm map remains centred at pixel 300.
        # The current per-camera style is a hollow circle, so check its four
        # cardinal points rather than its intentionally empty centre.
        for pixel_y, pixel_x in ((291, 300), (309, 300), (300, 291), (300, 309)):
            self.assertTrue(
                np.array_equal(tactical_map[pixel_y, pixel_x], (0, 160, 0))
            )

    def test_fps_text_metrics_keep_varied_values_inside_frame(self):
        for text in ("FPS: 8.4", "FPS: 30.0", "FPS: 120.5"):
            frame = np.zeros((100, 400, 3), dtype=np.uint8)

            origin, (text_width, text_height), baseline = draw_top_left_text(
                frame,
                text,
                left_margin=12,
                top_margin=10,
                font_face=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=1.6,
                color=(0, 255, 0),
                thickness=3,
            )

            origin_x, origin_y = origin
            self.assertEqual(origin_x, 12)
            self.assertGreaterEqual(origin_y - text_height, 10)
            self.assertLess(origin_y + baseline, frame.shape[0])
            self.assertLess(origin_x + text_width, frame.shape[1])
            drawn_rows = np.flatnonzero(np.any(frame != 0, axis=(1, 2)))
            self.assertGreaterEqual(int(drawn_rows.min()), 10)


if __name__ == "__main__":
    unittest.main()
