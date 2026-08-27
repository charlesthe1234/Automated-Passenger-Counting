"""Constructing the MediaPipe pose estimator and pinning it to the right GPU."""

import cv2
import os
import re
import subprocess
import sys

try:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
except ImportError:
    BaseOptions = None
    vision = None
try:
    import mediapipe as mp
except ImportError:
    mp = None


class MediaPipePoseEstimator:
    def __init__(self, model_path, delegate="cpu"):
        delegate_name = str(delegate).strip().lower()
        if delegate_name.startswith("gpu:"):
            _configure_mediapipe_gpu_device(delegate_name.split(":", 1)[1])
        delegate_value = (
            BaseOptions.Delegate.GPU
            if delegate_name.startswith("gpu")
            else BaseOptions.Delegate.CPU
        )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=delegate_value,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.delegate = delegate_name

    def detect(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.landmarker.detect(mp_image)

    def close(self):
        self.landmarker.close()


def _command_output(arguments):
    completed = subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout


def _nvidia_gpu_uuids_by_cuda_index():
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    result = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2 and parts[0].isdigit():
            result[parts[0]] = parts[1]
    return result


def _nvidia_x_gpu_uuids():
    output = _command_output(["nvidia-settings", "-q", "gpus", "-t"])
    return re.findall(
        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
        output,
    )


def _configure_mediapipe_gpu_device(device_index):
    """Route MediaPipe's process-wide NVIDIA EGL context to a CUDA-indexed GPU."""

    device_index = str(device_index).strip()
    cuda_uuids = _nvidia_gpu_uuids_by_cuda_index()
    requested_uuid = cuda_uuids.get(device_index)
    if requested_uuid is None:
        raise RuntimeError(f"NVIDIA GPU {device_index} was not found")

    x_gpu_uuids = _nvidia_x_gpu_uuids()
    if requested_uuid not in x_gpu_uuids:
        raise RuntimeError(
            f"NVIDIA GPU {device_index} is not exposed by the current X server"
        )

    x_gpu_position = x_gpu_uuids.index(requested_uuid)
    if x_gpu_position == 0:
        os.environ.pop("__NV_PRIME_RENDER_OFFLOAD", None)
        os.environ.pop("__NV_PRIME_RENDER_OFFLOAD_PROVIDER", None)
        os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)
        return "display"

    provider = f"NVIDIA-G{x_gpu_position - 1}"
    providers = _command_output(["xrandr", "--listproviders"])
    if provider not in providers:
        raise RuntimeError(
            f"NVIDIA PRIME provider {provider} for GPU {device_index} was not found"
        )
    os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    os.environ["__NV_PRIME_RENDER_OFFLOAD_PROVIDER"] = provider
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    return provider


def create_mediapipe_pose_estimator(enabled, model_path, delegate="auto"):
    if not enabled:
        return None
    if mp is None or BaseOptions is None or vision is None:
        print("MediaPipe is not installed. Falling back to YOLO/keypoint/box foot estimation.")
        return None
    if not model_path.exists():
        print(f"MediaPipe model file not found: {model_path}. Falling back to YOLO/keypoint/box foot estimation.")
        return None

    requested_delegate = str(delegate).strip().lower()
    if requested_delegate == "auto":
        requested_delegate = "gpu" if sys.platform.startswith("linux") else "cpu"

    try:
        estimator = MediaPipePoseEstimator(model_path, delegate=requested_delegate)
        device_label = requested_delegate.upper().replace(":", " ")
        print(f"MediaPipe Pose Landmarker is running on: {device_label}")
        return estimator
    except Exception as exc:
        if not requested_delegate.startswith("gpu"):
            raise
        print(f"MediaPipe {requested_delegate.upper()} unavailable ({exc}). Falling back to CPU.")
        estimator = MediaPipePoseEstimator(model_path, delegate="cpu")
        print("MediaPipe Pose Landmarker is running on: CPU")
        return estimator
