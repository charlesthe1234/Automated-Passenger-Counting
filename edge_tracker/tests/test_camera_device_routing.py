"""Tests for assigning independent YOLO devices to two camera pipelines."""

import unittest
from unittest.mock import patch

from main_tracker import (
    build_camera_contexts,
    preload_models,
)
from tracker_cli import parse_args


class _FakeYolo:
    instances = []

    def __init__(self, model_path):
        self.model_path = model_path
        self.loaded_device = None
        self.predictor = None
        self.__class__.instances.append(self)

    def to(self, device):
        self.loaded_device = device
        return self


class CameraDeviceRoutingTests(unittest.TestCase):
    def setUp(self):
        _FakeYolo.instances = []

    def test_contexts_receive_independent_camera_devices(self):
        args = parse_args(
            [
                "--source",
                "0",
                "--source-2",
                "1",
                "--device",
                "0",
                "--device-2",
                "1",
            ]
        )
        contexts = build_camera_contexts(args)
        self.assertEqual([context.device for context in contexts], ["0", "1"])

    def test_camera_two_falls_back_to_camera_one_device(self):
        args = parse_args(
            ["--source", "0", "--source-2", "1", "--device", "cpu"]
        )
        contexts = build_camera_contexts(args)
        self.assertEqual([context.device for context in contexts], ["cpu", "cpu"])

    def test_preloader_places_each_yolo_model_on_its_camera_device(self):
        args = parse_args(
            [
                "--source",
                "0",
                "--source-2",
                "1",
                "--device",
                "0",
                "--device-2",
                "1",
            ]
        )
        with (
            patch("main_tracker.YOLO", _FakeYolo),
            patch("main_tracker.create_mediapipe_pose_estimator", return_value=None),
        ):
            models = preload_models(args)

        self.assertEqual(len(models.yolo_models), 2)
        self.assertEqual(
            [model.loaded_device for model in models.yolo_models],
            ["cuda:0", "cuda:1"],
        )


if __name__ == "__main__":
    unittest.main()
