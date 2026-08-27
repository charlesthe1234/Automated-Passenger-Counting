from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


SERVICE_TOKEN_HEADER = "X-CV-Service-Token"


def _resolve_service_token():
    """Find the deployment's CV service token without ever using a CLI argument.

    The managed worker receives it in its environment from the backend. A
    manually launched debugging run falls back to the same Git-ignored
    environment file the launcher already reads, so no separate secret file is
    required on a single-machine deployment. A future split deployment only
    needs the environment variable set on the CV workstation.
    """
    token = os.environ.get("CV_SERVICE_TOKEN", "").strip()
    if token:
        return token
    try:
        from launch_config import read_env_file

        return read_env_file().get("CV_SERVICE_TOKEN", "").strip()
    except Exception:
        return ""


class ReidBackendStore:
    """Persist ReID masters through FastAPI, whose SQLite DB is the source of truth."""

    def __init__(self, base_url, run_id="default", timeout=10):
        self.base_url = str(base_url).rstrip("/")
        self.run_id = str(run_id or "default")
        self.timeout = max(1, int(timeout))
        self._uploaded_slot_digests = {}
        self._service_token = _resolve_service_token()
        if not self._service_token:
            print(
                "ReID persistence: CV_SERVICE_TOKEN is not set. Identity and gallery "
                "uploads will be rejected by the backend with HTTP 401.",
                flush=True,
            )

    def _url(self, path, query=None):
        url = urllib.parse.urljoin(self.base_url + "/", path.lstrip("/"))
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return url

    def _request_json(self, method, path, payload=None, query=None, body=None, headers=None):
        if body is None and payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers = {**(headers or {}), "Content-Type": "application/json"}
        headers = dict(headers or {})
        if self._service_token:
            headers[SERVICE_TOKEN_HEADER] = self._service_token
        request = urllib.request.Request(
            self._url(path, query=query),
            data=body,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response_body = response.read()
        return json.loads(response_body.decode("utf-8")) if response_body else None

    @staticmethod
    def _multipart_body(fields, file_field, file_path):
        boundary = f"----cag-reid-{uuid4().hex}"
        line_break = b"\r\n"
        chunks = []
        for name, value in fields.items():
            if value is None:
                continue
            chunks.extend(
                [
                    f"--{boundary}".encode("ascii"),
                    f'Content-Disposition: form-data; name="{name}"'.encode("utf-8"),
                    b"",
                    str(value).encode("utf-8"),
                ]
            )

        path = Path(file_path)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}".encode("ascii"),
                (
                    f'Content-Disposition: form-data; name="{file_field}"; '
                    f'filename="{path.name}"'
                ).encode("utf-8"),
                f"Content-Type: {content_type}".encode("ascii"),
                b"",
                path.read_bytes(),
                f"--{boundary}--".encode("ascii"),
                b"",
            ]
        )
        return line_break.join(chunks), f"multipart/form-data; boundary={boundary}"

    def load_payload(self):
        payload = self._request_json(
            "GET",
            "/api/evacuees/reid-gallery",
            query={"run_id": self.run_id},
        )
        identities = payload.get("identities", {}) if isinstance(payload, dict) else {}
        for raw_identity_id, record in identities.items():
            gallery = record.get("gallery", {})
            for slot_name, slot in gallery.items():
                if not slot:
                    continue
                feature_b64 = slot.pop("feature_b64", None)
                if not feature_b64:
                    gallery[slot_name] = None
                    continue
                feature = np.frombuffer(base64.b64decode(feature_b64), dtype=np.float32).copy()
                slot["feature"] = feature
                slot["image_path"] = None
                captured_at = slot.get("captured_at")
                if isinstance(captured_at, str):
                    slot["captured_at_iso"] = captured_at
                slot["captured_at"] = time.monotonic()
                self._uploaded_slot_digests[(int(raw_identity_id), slot_name)] = slot.get("digest")
        return payload

    @staticmethod
    def _numeric_age(value):
        try:
            age = float(value)
        except (TypeError, ValueError):
            return None
        return age if 0 < age <= 120 else None

    @staticmethod
    def _latest_camera(record):
        candidates = [slot for slot in record.get("gallery", {}).values() if slot and slot.get("camera_id")]
        if not candidates:
            return None
        latest = max(candidates, key=lambda slot: float(slot.get("captured_at", 0.0)))
        return latest.get("camera_id")

    def save_identity(self, identity_id, record):
        identity_id = int(identity_id)
        metadata = {
            "role": str(record.get("role") or "evacuee"),
            "role_confidence": float(record.get("role_confidence") or 0.0),
            "age": self._numeric_age(record.get("age")),
            "gender": str(record.get("gender") or "unknown"),
            "last_seen_at": datetime.now(timezone.utc).isoformat(),
            "last_camera_id": self._latest_camera(record),
            "current_status": "inside",
        }
        self._request_json(
            "PUT",
            f"/api/evacuees/by-master/{urllib.parse.quote(self.run_id, safe='')}/{identity_id}",
            payload=metadata,
        )

        for slot_name, slot in record.get("gallery", {}).items():
            if not slot:
                continue
            digest = slot.get("digest")
            if self._uploaded_slot_digests.get((identity_id, slot_name)) == digest:
                continue
            image_path = slot.get("image_path")
            if not image_path or not Path(image_path).is_file():
                continue

            feature = np.asarray(slot.get("feature"), dtype=np.float32).reshape(-1)
            fields = {
                "feature_b64": base64.b64encode(feature.tobytes()).decode("ascii"),
                "feature_dimension": int(feature.size),
                "feature_space_id": slot.get("feature_space_id"),
                "feature_source": slot.get("feature_source"),
                "digest": digest,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "captured_frame": slot.get("captured_frame"),
                "camera_id": slot.get("camera_id"),
                "sharpness": slot.get("sharpness"),
                "detection_confidence": slot.get("detection_confidence"),
            }
            body, content_type = self._multipart_body(fields, "image", image_path)
            self._request_json(
                "PUT",
                (
                    f"/api/evacuees/by-master/{urllib.parse.quote(self.run_id, safe='')}/"
                    f"{identity_id}/views/{slot_name}"
                ),
                body=body,
                headers={"Content-Type": content_type},
            )
            self._uploaded_slot_digests[(identity_id, slot_name)] = digest
