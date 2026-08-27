from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    camera_url: str = Field(
        "",
        validation_alias="CAMERA_URL",
    )
    camera_urls: str = Field("", validation_alias="CAMERA_URLS")
    primary_camera_id: str = Field("cam_1", validation_alias="PRIMARY_CAMERA_ID")
    camera_reconnect_seconds: int = Field(5, validation_alias="CAMERA_RECONNECT_SECONDS")
    camera_jpeg_quality: int = Field(80, validation_alias="CAMERA_JPEG_QUALITY")
    sqlite_db_path: str = Field("./passenger_monitoring.db", validation_alias="SQLITE_DB_PATH")
    observation_upload_dir: str = Field("./uploads/observations", validation_alias="OBSERVATION_UPLOAD_DIR")
    evacuee_upload_dir: str = Field("./uploads/evacuees", validation_alias="EVACUEE_UPLOAD_DIR")
    frontend_dist_dir: str = Field("../frontend/dist", validation_alias="FRONTEND_DIST_DIR")
    zone_capacities_json: str = Field('{"cam_1":150,"cam_2":150}', validation_alias="ZONE_CAPACITIES_JSON")
    mqtt_enabled: bool = Field(False, validation_alias="MQTT_ENABLED")
    mqtt_host: str = Field("localhost", validation_alias="MQTT_HOST")
    mqtt_port: int = Field(1883, validation_alias="MQTT_PORT")
    mqtt_username: str = Field("", validation_alias="MQTT_USERNAME")
    mqtt_password: str = Field("", validation_alias="MQTT_PASSWORD")
    mqtt_topic_metrics: str = Field("cag/metrics", validation_alias="MQTT_TOPIC_METRICS")
    mqtt_topic_tactical: str = Field("cag/tactical", validation_alias="MQTT_TOPIC_TACTICAL")
    mqtt_topic_alerts: str = Field("cag/alerts", validation_alias="MQTT_TOPIC_ALERTS")
    mqtt_metric_log_interval_seconds: float = Field(1.0, validation_alias="MQTT_METRIC_LOG_INTERVAL_SECONDS")
    cv_enabled: bool = Field(True, validation_alias="CV_ENABLED")
    cv_worker_python: str = Field(
        "../.venv-cv-linux/bin/python", validation_alias="CV_WORKER_PYTHON"
    )
    cv_worker_script: str = Field(
        "../edge_tracker/cv_worker.py", validation_alias="CV_WORKER_SCRIPT"
    )
    cv_worker_log: str = Field(
        "../LogEvidance/cv_service.jsonl", validation_alias="CV_WORKER_LOG"
    )
    cv_control_allow_lan: bool = Field(False, validation_alias="CV_CONTROL_ALLOW_LAN")
    cv_control_token: SecretStr = Field(
        default=SecretStr(""), validation_alias="CV_CONTROL_TOKEN"
    )
    # Transitional only. The normal React workflow authorizes CV Start/Stop with
    # an admin session plus CSRF; this re-enables the pre-auth token-only path
    # for approved scripts that have not migrated yet. Run Manager removes it.
    cv_control_legacy_token_enabled: bool = Field(
        False, validation_alias="CV_CONTROL_LEGACY_TOKEN_ENABLED"
    )
    cors_origins: str = Field(
        "http://localhost:5173,http://localhost:3000",
        validation_alias="CORS_ORIGINS",
    )

    # --- Authentication -------------------------------------------------
    app_env: str = Field("production", validation_alias="APP_ENV")
    allow_demo_account_seeding: bool = Field(
        False, validation_alias="ALLOW_DEMO_ACCOUNT_SEEDING"
    )
    # Password login over plain HTTP from a non-loopback client is refused
    # unless this is explicitly enabled for a controlled demo network.
    auth_allow_insecure_http: bool = Field(
        False, validation_alias="AUTH_ALLOW_INSECURE_HTTP"
    )
    auth_session_idle_minutes: int = Field(
        720, ge=1, le=10080, validation_alias="AUTH_SESSION_IDLE_MINUTES"
    )
    auth_session_absolute_hours: int = Field(
        24, ge=1, le=720, validation_alias="AUTH_SESSION_ABSOLUTE_HOURS"
    )
    auth_trust_proxy_headers: bool = Field(
        False, validation_alias="AUTH_TRUST_PROXY_HEADERS"
    )

    # --- CV service authentication --------------------------------------
    # One deployment-wide secret. FastAPI validates it; the CV worker sends it
    # as X-CV-Service-Token. Provisioned through Git-ignored environment files.
    cv_service_token: SecretStr = Field(
        default=SecretStr(""), validation_alias="CV_SERVICE_TOKEN"
    )

    # --- Run/session management -----------------------------------------
    # Compatibility default for the initial Run Manager release: the team still
    # launches the CV pipeline from a terminal for debugging, and that data must
    # keep arriving. Such runs are recorded as external/unmanaged, never as
    # completed managed runs. Tighten to false only once every launch path goes
    # through Run Manager.
    allow_unmanaged_run_ingestion: bool = Field(
        True, validation_alias="ALLOW_UNMANAGED_RUN_INGESTION"
    )
    external_run_active_window_seconds: int = Field(
        10, ge=1, le=3600, validation_alias="EXTERNAL_RUN_ACTIVE_WINDOW_SECONDS"
    )
    run_start_timeout_seconds: int = Field(
        120, ge=5, le=3600, validation_alias="RUN_START_TIMEOUT_SECONDS"
    )
    run_stop_timeout_seconds: int = Field(
        30, ge=5, le=3600, validation_alias="RUN_STOP_TIMEOUT_SECONDS"
    )
    run_reconcile_interval_seconds: float = Field(
        2.0, ge=0.5, le=60.0, validation_alias="RUN_RECONCILE_INTERVAL_SECONDS"
    )
    sqlite_busy_timeout_ms: int = Field(
        5000, ge=100, le=60000, validation_alias="SQLITE_BUSY_TIMEOUT_MS"
    )
    sqlite_write_lock_retry_count: int = Field(
        3, ge=0, le=10, validation_alias="SQLITE_WRITE_LOCK_RETRY_COUNT"
    )

    @property
    def is_demo_env(self) -> bool:
        return self.app_env.strip().lower() == "demo"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def zone_capacity_map(self) -> dict[str, int]:
        try:
            raw_value = json.loads(self.zone_capacities_json)
        except json.JSONDecodeError:
            return {}

        if not isinstance(raw_value, dict):
            return {}

        capacities: dict[str, int] = {}
        for zone_id, capacity in raw_value.items():
            try:
                normalized_capacity = int(capacity)
            except (TypeError, ValueError):
                continue
            if str(zone_id).strip() and normalized_capacity > 0:
                capacities[str(zone_id).strip()] = normalized_capacity
        return capacities

    @property
    def camera_source_map(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        entries = [entry.strip() for entry in self.camera_urls.split(",") if entry.strip()]

        for index, entry in enumerate(entries, start=1):
            if "=" in entry:
                camera_id, camera_url = entry.split("=", 1)
                camera_id = camera_id.strip()
            else:
                camera_id = f"cam_{index}"
                camera_url = entry

            camera_url = camera_url.strip()
            if camera_id and camera_url:
                sources[camera_id] = camera_url

        if not sources and self.camera_url:
            sources[self.primary_camera_id] = self.camera_url

        return sources

    @property
    def observation_upload_path(self) -> Path:
        configured = Path(self.observation_upload_dir)
        if configured.is_absolute():
            return configured
        return Path(__file__).resolve().parent / configured

    @property
    def evacuee_upload_path(self) -> Path:
        configured = Path(self.evacuee_upload_dir)
        if configured.is_absolute():
            return configured
        return Path(__file__).resolve().parent / configured

    @property
    def frontend_dist_path(self) -> Path:
        configured = Path(self.frontend_dist_dir)
        if configured.is_absolute():
            return configured
        return Path(__file__).resolve().parent / configured

    def _backend_relative_path(self, configured_value: str) -> Path:
        configured = Path(configured_value).expanduser()
        if configured.is_absolute():
            return Path(os.path.abspath(configured))

        # Keep the final path lexical instead of resolving symlinks. Python
        # virtual-environment executables are normally symlinks to the base
        # interpreter; resolving one before launching it bypasses the venv and
        # loads packages from the base installation instead.
        return Path(os.path.abspath(Path(__file__).resolve().parent / configured))

    @property
    def cv_worker_python_path(self) -> Path:
        return self._backend_relative_path(self.cv_worker_python)

    @property
    def cv_worker_script_path(self) -> Path:
        return self._backend_relative_path(self.cv_worker_script)

    @property
    def cv_worker_log_path(self) -> Path:
        return self._backend_relative_path(self.cv_worker_log)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
