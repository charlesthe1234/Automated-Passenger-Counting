import os
import subprocess
import sys
import unittest
from unittest.mock import patch


from tests import EDGE_TRACKER_DIR

sys.path.insert(0, str(EDGE_TRACKER_DIR))

from mediapipe_runtime import _configure_mediapipe_gpu_device


GPU_0_UUID = "GPU-00000000-0000-0000-0000-000000000000"
GPU_1_UUID = "GPU-11111111-1111-1111-1111-111111111111"


class MediaPipeDeviceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.saved_environment = {
            name: os.environ.get(name)
            for name in (
                "__NV_PRIME_RENDER_OFFLOAD",
                "__NV_PRIME_RENDER_OFFLOAD_PROVIDER",
                "__GLX_VENDOR_LIBRARY_NAME",
            )
        }

    def tearDown(self):
        for name, value in self.saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    @staticmethod
    def command_result(arguments, **_kwargs):
        command = arguments[0]
        if command == "nvidia-smi":
            output = f"0, {GPU_0_UUID}\n1, {GPU_1_UUID}\n"
        elif command == "nvidia-settings":
            # The X display GPU is physical/CUDA GPU 1; GPU 0 is the first
            # PRIME render-offload provider. This matches the test machine.
            output = f"names: GPU-0, {GPU_1_UUID}\nnames: GPU-1, {GPU_0_UUID}\n"
        elif command == "xrandr":
            output = "Provider 0 name:NVIDIA-0\nProvider 1 name:NVIDIA-G0\n"
        else:
            raise AssertionError(f"Unexpected command: {arguments}")
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    @patch("mediapipe_runtime.subprocess.run", side_effect=command_result)
    def test_gpu_zero_uses_prime_offload_provider(self, _run):
        provider = _configure_mediapipe_gpu_device("0")
        self.assertEqual(provider, "NVIDIA-G0")
        self.assertEqual(os.environ["__NV_PRIME_RENDER_OFFLOAD"], "1")
        self.assertEqual(
            os.environ["__NV_PRIME_RENDER_OFFLOAD_PROVIDER"], "NVIDIA-G0"
        )
        self.assertEqual(os.environ["__GLX_VENDOR_LIBRARY_NAME"], "nvidia")

    @patch("mediapipe_runtime.subprocess.run", side_effect=command_result)
    def test_display_gpu_clears_prime_offload(self, _run):
        os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        os.environ["__NV_PRIME_RENDER_OFFLOAD_PROVIDER"] = "NVIDIA-G0"
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
        provider = _configure_mediapipe_gpu_device("1")
        self.assertEqual(provider, "display")
        self.assertNotIn("__NV_PRIME_RENDER_OFFLOAD", os.environ)
        self.assertNotIn("__NV_PRIME_RENDER_OFFLOAD_PROVIDER", os.environ)
        self.assertNotIn("__GLX_VENDOR_LIBRARY_NAME", os.environ)

    @patch("mediapipe_runtime.subprocess.run", side_effect=command_result)
    def test_unknown_gpu_is_rejected_instead_of_silently_misrouted(self, _run):
        with self.assertRaisesRegex(RuntimeError, "GPU 2 was not found"):
            _configure_mediapipe_gpu_device("2")


if __name__ == "__main__":
    unittest.main()
