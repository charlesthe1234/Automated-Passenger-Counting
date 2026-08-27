from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

import crud
from config import Settings, settings
from models import MetricLogCreate, SystemAlertCreate, TacticalStateCreate
from tactical_state import tactical_store

# A publisher stuck on a wrong or deleted run id must not flood the log.
REJECTION_LOG_INTERVAL_SECONDS = 30.0

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover - keeps the app bootable until requirements are installed.
    mqtt = None


logger = logging.getLogger(__name__)


class MqttBridge:
    def __init__(self, app_settings: Settings):
        self.settings = app_settings
        self.client = None
        self._lock = threading.Lock()
        self._last_metric_log_at = 0.0
        self._latest_zone_counts_by_run: dict[str, dict[str, int]] = {}
        self._latest_camera_online_count_by_run: dict[str, int] = {}
        self._last_rejection_log_at: dict[str, float] = {}
        self._stopping = False

    def start(self) -> None:
        if not self.settings.mqtt_enabled:
            logger.info("MQTT bridge disabled.")
            return
        if mqtt is None:
            logger.error("MQTT bridge enabled but paho-mqtt is not installed.")
            return

        self.client = self._create_client()
        self._stopping = False
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        if self.settings.mqtt_username:
            self.client.username_pw_set(
                self.settings.mqtt_username,
                self.settings.mqtt_password or None,
            )

        logger.info("Starting MQTT bridge to %s:%s.", self.settings.mqtt_host, self.settings.mqtt_port)
        self.client.connect_async(self.settings.mqtt_host, self.settings.mqtt_port, keepalive=60)
        self.client.loop_start()

    def stop(self) -> None:
        if self.client is None:
            return
        logger.info("Stopping MQTT bridge.")
        self._stopping = True
        self.client.loop_stop()
        self.client.disconnect()
        self.client = None

    def _create_client(self):
        try:
            return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="cag-fastapi-bridge")
        except (AttributeError, TypeError):
            return mqtt.Client(client_id="cag-fastapi-bridge")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        logger.info("MQTT bridge connected with result %s.", reason_code)
        for topic in self._topics():
            client.subscribe(topic, qos=1)
            logger.info("MQTT bridge subscribed to %s.", topic)

    def _on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags_or_reason_code,
        reason_code=None,
        properties=None,
    ) -> None:
        """Handle both Paho MQTT callback API versions without stopping its network loop."""
        effective_reason_code = (
            reason_code if reason_code is not None else disconnect_flags_or_reason_code
        )
        if self._stopping:
            logger.info("MQTT bridge disconnected during shutdown with result %s.", effective_reason_code)
        else:
            logger.warning(
                "MQTT bridge disconnected with result %s. Paho will attempt reconnect.",
                effective_reason_code,
            )

    def _on_message(self, client, userdata, message) -> None:
        topic = message.topic
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid MQTT JSON on %s: %s", topic, exc)
            return

        try:
            if topic == self.settings.mqtt_topic_metrics:
                self.handle_metrics_payload(payload)
            elif topic == self.settings.mqtt_topic_tactical:
                self.handle_tactical_payload(payload)
            elif topic == self.settings.mqtt_topic_alerts:
                self.handle_alert_payload(payload)
            else:
                logger.debug("Ignoring MQTT message on unhandled topic %s.", topic)
        except Exception:
            logger.exception("MQTT bridge failed while handling topic %s.", topic)

    def handle_metrics_payload(self, payload: dict[str, Any]) -> None:
        metric = self._metric_from_payload(payload)
        if metric is None:
            return

        now = time.monotonic()
        with self._lock:
            if now - self._last_metric_log_at < self.settings.mqtt_metric_log_interval_seconds:
                return
            self._last_metric_log_at = now

        self._write_run_scoped(metric.run_id, lambda db: crud.create_metric_log(db, metric))

    def handle_tactical_payload(self, payload: dict[str, Any]) -> None:
        try:
            tactical = TacticalStateCreate(**payload)
        except ValidationError as exc:
            logger.warning("Ignoring invalid MQTT tactical payload: %s", exc)
            return

        # Tactical state is memory-only, but it must still respect run policy so
        # a deleted or mismatched run cannot repopulate the live map.
        if self._run_scoped_write_allowed(tactical.run_id):
            tactical_store.update(tactical)

    def handle_alert_payload(self, payload: dict[str, Any]) -> None:
        try:
            alert = SystemAlertCreate(**payload)
        except ValidationError as exc:
            logger.warning("Ignoring invalid MQTT alert payload: %s", exc)
            return

        self._write_run_scoped(alert.run_id, lambda db: crud.create_system_alert(db, alert))

    def _log_rejection(self, run_id: str, reason: str) -> None:
        """Rate-limited so a misconfigured publisher cannot flood the log."""
        now = time.monotonic()
        with self._lock:
            last = self._last_rejection_log_at.get(run_id, 0.0)
            if now - last < REJECTION_LOG_INTERVAL_SECONDS:
                return
            self._last_rejection_log_at[run_id] = now
        logger.warning("Dropped CV payload: %s", reason)

    def _run_scoped_write_allowed(self, run_id: str) -> bool:
        """Validate run policy and record ingestion without writing a row."""
        return self._write_run_scoped(run_id, lambda db: None) is not False

    def _write_run_scoped(self, run_id: str, writer):
        """Validate run policy and perform the write in one serialized transaction."""
        from runs.service import IngestionRejected, resolve_ingestion_run
        from runs.write_guard import RunWriteConflict, immediate_write

        try:
            with immediate_write() as db:
                resolve_ingestion_run(db, run_id)
                return writer(db)
        except IngestionRejected as reason:
            self._log_rejection(run_id, str(reason))
            return False
        except RunWriteConflict:
            self._log_rejection(run_id, "the database stayed locked")
            return False

    def clear_run_cache(self, run_id: str) -> None:
        """Forget per-run zone/camera caches after a run is deleted."""
        with self._lock:
            self._latest_zone_counts_by_run.pop(run_id, None)
            self._latest_camera_online_count_by_run.pop(run_id, None)
            self._last_rejection_log_at.pop(run_id, None)

    def _metric_from_payload(self, payload: dict[str, Any]) -> MetricLogCreate | None:
        run_id = str(payload.get("run_id") or "default")
        incoming_zone_counts = self._extract_zone_counts(payload)

        with self._lock:
            if incoming_zone_counts:
                run_zone_counts = self._latest_zone_counts_by_run.setdefault(run_id, {})
                run_zone_counts.update(incoming_zone_counts)
            else:
                run_zone_counts = self._latest_zone_counts_by_run.setdefault(run_id, {})

            if payload.get("camera_online_count") is not None:
                try:
                    self._latest_camera_online_count_by_run[run_id] = max(0, int(payload["camera_online_count"]))
                except (TypeError, ValueError):
                    pass

            zone_counts = dict(run_zone_counts)
            camera_online_count = self._latest_camera_online_count_by_run.get(run_id)

        try:
            passenger_count = max(0, int(payload.get("passenger_count", 0)))
        except (TypeError, ValueError):
            passenger_count = 0

        if payload.get("passenger_count") is None and zone_counts:
            passenger_count = sum(zone_counts.values())

        try:
            return MetricLogCreate(
                passenger_count=passenger_count,
                run_id=run_id,
                zone_counts=zone_counts or payload.get("zone_counts"),
                camera_online_count=camera_online_count,
                timestamp=payload.get("timestamp") or datetime.now(timezone.utc),
            )
        except ValidationError as exc:
            logger.warning("Ignoring invalid MQTT metric payload: %s", exc)
            return None

    def _extract_zone_counts(self, payload: dict[str, Any]) -> dict[str, int]:
        zone_counts = payload.get("zone_counts")
        if not isinstance(zone_counts, dict):
            camera_id = payload.get("camera_id")
            if camera_id:
                zone_counts = {str(camera_id): payload.get("passenger_count", 0)}
            else:
                return {}

        normalized: dict[str, int] = {}
        for zone_id, value in zone_counts.items():
            try:
                count = max(0, int(value))
            except (TypeError, ValueError):
                continue
            normalized[str(zone_id)] = count
        return normalized

    def _topics(self) -> list[str]:
        return [
            self.settings.mqtt_topic_metrics,
            self.settings.mqtt_topic_tactical,
            self.settings.mqtt_topic_alerts,
        ]


mqtt_bridge = MqttBridge(settings)
