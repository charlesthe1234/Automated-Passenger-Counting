"""The two checkpoint-backed models the identity memory composes.

TransReIDFeatureExtractor produces appearance features; EvacuationRoleClassifier
labels a crop as evacuee or staff. Both are collaborators handed to
AppearanceIdentityMemory rather than behaviour it inherits, so they load,
fail, and are tested independently of it."""

import cv2
import hashlib
import json
import numpy as np
import sys
import threading

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from constants import DEFAULT_REID_ROLE_CHECKPOINT
from pathlib import Path


def _sha256_file(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap_torch_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


class TransReIDFeatureExtractor:
    def __init__(self, checkpoint_path, device="cuda", fastreid_root="fast-reid"):
        self.model = None
        self.backend = None
        self.checkpoint_path = Path(checkpoint_path)
        self._checkpoint_sha256 = None
        self._config_sha256 = None
        self.fastreid_root = Path(fastreid_root) if fastreid_root else None
        # Cameras now run their frame pipelines concurrently in worker
        # threads (see main_tracker.py). This single ReID model instance is
        # shared across all of them, so we serialize the actual forward
        # passes to avoid any cross-thread CUDA/state issues. Everything
        # else (crop resizing, numpy post-processing) still happens outside
        # the lock and can overlap freely.
        self._lock = threading.Lock()
        if torch is None:
            print("Warning: torch is not available, TransReID feature extractor disabled.")
            return

        self.device = torch.device(device)

        print(f"[Hardware Check] TransReID is running on: {self.device.type.upper()}")
        if self._load_transreid_jpm_model():
            return
        print(
            "Warning: Exact TransReID JPM/SIE loading failed; appearance ReID is disabled "
            "instead of using an incompatible partial-weight fallback."
        )

    def is_available(self):
        return self.model is not None

    def feature_space_id(self, dimension):
        """Fingerprint the exact model and preprocessing feature space."""
        if self._checkpoint_sha256 is None:
            self._checkpoint_sha256 = _sha256_file(self.checkpoint_path)
        if self.backend == "transreid_jpm":
            config_path = Path(__file__).resolve().parent / "transreid_jpm.py"
            if self._config_sha256 is None:
                self._config_sha256 = _sha256_file(config_path)
            model_spec = "msmt17-vit-base-transreid-jpm-sie15-stride12"
            preprocess = "resize128x256-inter_linear-bgr2rgb-chw-f32-div255-mean0.5-std0.5-sie0"
        else:
            model_spec = "vit_base_patch16_224-img240x224"
            preprocess = "resize224x240-inter_area-bgr2rgb-chw-f32-div255-default_cfg_norm"
        metadata = {
            "kind": "transreid",
            "backend": self.backend,
            "checkpoint_sha256": self._checkpoint_sha256,
            "config_sha256": self._config_sha256,
            "model_spec": model_spec,
            "preprocess": preprocess,
            "preprocess_revision": 1,
            "dimension": int(dimension),
        }
        canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "fs1:" + hashlib.sha256(canonical).hexdigest()

    def _load_transreid_jpm_model(self):
        if self.fastreid_root is None or not self.fastreid_root.exists():
            print(f"TransReID ViT dependency folder not found: {self.fastreid_root}.")
            return False

        root_text = str(self.fastreid_root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        try:
            from fastreid.modeling.backbones.vision_transformer import VisionTransformer
            from transreid_jpm import (
                TRANSREID_FEATURE_DIM,
                build_transreid_jpm_from_checkpoint,
            )

            if not self.checkpoint_path.exists():
                print(f"TransReID checkpoint not found: {self.checkpoint_path}")
                return False

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            model, spec = build_transreid_jpm_from_checkpoint(checkpoint, VisionTransformer)
            model.eval()
            model.to(self.device)
            print(
                "TransReID JPM/SIE backend loaded. Missing: 0, Unexpected: 0; "
                f"classes: {spec['num_classes']}, SIE cameras: {spec['camera_count']}, "
                f"feature dimension: {TRANSREID_FEATURE_DIM}"
            )

            self.model = model
            self.backend = "transreid_jpm"
            return True
        except Exception as exc:
            print(f"Unable to load exact TransReID JPM/SIE checkpoint: {exc}")
            self.model = None
            self.backend = None
            return False

    def extract(self, crop):
        if self.model is None or crop is None:
            return None

        try:
            if self.backend == "transreid_jpm":
                resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                tensor = tensor.sub(0.5).div(0.5)
                tensor = tensor.unsqueeze(0).to(self.device)
                with self._lock, torch.no_grad():
                    features = self.model(tensor)
                feature = features.detach().cpu().numpy().ravel().astype(np.float32)
                norm = float(np.linalg.norm(feature))
                if norm <= 1e-6:
                    return None
                return feature / norm

            resized = cv2.resize(crop, (224, 240), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            tensor = tensor.unsqueeze(0).to(self.device)
            if hasattr(self.model, 'default_cfg') and self.model.default_cfg is not None:
                mean = torch.tensor(self.model.default_cfg.get('mean', (0.5, 0.5, 0.5)), device=self.device).view(3, 1, 1)
                std = torch.tensor(self.model.default_cfg.get('std', (0.5, 0.5, 0.5)), device=self.device).view(3, 1, 1)
                tensor = (tensor - mean) / std
            with self._lock, torch.no_grad():
                features = self.model(tensor)
            feature = features.detach().cpu().numpy().ravel().astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if norm <= 1e-6:
                return None
            return feature / norm
        except Exception:
            return None

    def extract_many(self, crops):
        crops = [crop for crop in crops if crop is not None and crop.size > 0]
        if self.model is None or not crops:
            return []

        if self.backend != "transreid_jpm":
            return [feature for feature in (self.extract(crop) for crop in crops) if feature is not None]

        try:
            tensors = []
            for crop in crops:
                resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                tensors.append(tensor.sub(0.5).div(0.5))

            batched_tensor = torch.stack(tensors).to(self.device)
            with self._lock, torch.no_grad():
                features = self.model(batched_tensor)

            normalized_features = []
            for feature in features.detach().cpu().numpy().astype(np.float32):
                norm = float(np.linalg.norm(feature))
                if norm > 1e-6:
                    normalized_features.append(feature / norm)
            return normalized_features
        except Exception:
            return [feature for feature in (self.extract(crop) for crop in crops) if feature is not None]

    def extract_many_aligned(self, crops):
        """Like extract_many, but the returned list is always the same
        length as `crops`, with None standing in for any crop that failed to
        produce a feature. extract_many() silently drops failures, which
        breaks positional alignment with track_ids -- unsafe for per-frame
        batching where a feature must be matched back to a specific person.
        """
        if self.model is None or not crops:
            return [None] * len(crops)

        if self.backend != "transreid_jpm":
            return [self.extract(crop) if crop is not None and crop.size > 0 else None for crop in crops]

        valid_indices = [i for i, crop in enumerate(crops) if crop is not None and crop.size > 0]
        if not valid_indices:
            return [None] * len(crops)

        try:
            tensors = []
            for i in valid_indices:
                resized = cv2.resize(crops[i], (128, 256), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                tensors.append(tensor.sub(0.5).div(0.5))

            batched_tensor = torch.stack(tensors).to(self.device)
            with self._lock, torch.no_grad():
                features = self.model(batched_tensor)

            results = [None] * len(crops)
            for local_index, original_index in enumerate(valid_indices):
                feature = features[local_index].detach().cpu().numpy().astype(np.float32)
                norm = float(np.linalg.norm(feature))
                if norm > 1e-6:
                    results[original_index] = feature / norm
            return results
        except Exception:
            return [self.extract(crop) if crop is not None and crop.size > 0 else None for crop in crops]


class EvacuationRoleClassifier:
    """CPU-only MobileNetV2 gate used by the v7 intake path."""

    CLASS_NAMES = ("cag", "evacuee", "scdf")

    def __init__(self, checkpoint_path=DEFAULT_REID_ROLE_CHECKPOINT):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None
        self.transform = None
        if torch is None or nn is None or self.checkpoint_path is None or not self.checkpoint_path.exists():
            return

        try:
            from torchvision import transforms
            from torchvision.models import mobilenet_v2

            model = mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.CLASS_NAMES))
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = unwrap_torch_checkpoint(checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            print(f"Role classifier loaded from {self.checkpoint_path} on CPU")
        except Exception as exc:
            print(f"Unable to load evacuation role classifier: {exc}")
            self.model = None
            self.transform = None

    def predict(self, crop):
        if self.model is None or self.transform is None or crop is None or crop.size == 0:
            return "evacuee", 0.0

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(self.model(tensor)[0], dim=0)
            confidence, class_index = torch.max(probabilities, dim=0)
            return self.CLASS_NAMES[int(class_index.item())], float(confidence.item())
        except Exception as exc:
            print(f"Role classification failed: {exc}")
            return "evacuee", 0.0
