import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from constants import (
    DEFAULT_TRACKER_CONFIG_PATH,
    DEFAULT_YOLO_NMS_IOU,
)
from launch_config import (
    build_tracker_arguments,
    default_launch_values,
    is_truthy,
    redact_tracker_arguments,
)
from session_lock import CvRuntimeBusyError, CvRuntimeLock


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "main_tracker.py"
PROJECT_ROOT = ROOT.parent
BACKEND_DIRECTORY = PROJECT_ROOT / "backend"
DATABASE_PATH = BACKEND_DIRECTORY / "passenger_monitoring.db"
REID_EVIDENCE_DIRECTORY = ROOT / "angle_evidence_v7"
LOG_EVIDENCE_DIRECTORY = ROOT.parent / "LogEvidance"
SCREEN_RECORDING_DIRECTORY = Path.home() / "Desktop" / "VIdeoEvidance"
CORNER_OPTIONS = ["None", "top_left", "top_right", "bottom_right", "bottom_left"]

# Reset clears passenger data only. Operator accounts, their sessions, the
# security audit trail, and the schema version live in the same SQLite file and
# must survive, so the file itself is no longer deleted.
#
# `runs` is included because leaving run rows behind after wiping their data
# would show a history of empty runs. Deletion tombstones are deliberately NOT
# written: Reset is a full wipe, not a targeted delete, and tombstoning would
# permanently blacklist those run IDs from being reused.
PASSENGER_DATA_TABLES = (
    "evacuee_gallery_views",  # child rows first
    "evacuee_identities",
    "passenger_observations",
    "metric_logs",
    "system_alerts",
    "run_events",
    "pending_file_deletions",
    "runs",
)

# Uploaded passenger evidence lives outside the database and was previously
# left behind by Reset, which is why it accumulated across runs.
EVIDENCE_UPLOAD_DIRECTORY = BACKEND_DIRECTORY / "uploads" / "evacuees"
OBSERVATION_UPLOAD_DIRECTORY = BACKEND_DIRECTORY / "uploads" / "observations"

def clear_evidence_files():
    """Remove uploaded evidence so it does not outlive its database rows."""
    removed = 0
    for root in (EVIDENCE_UPLOAD_DIRECTORY, OBSERVATION_UPLOAD_DIRECTORY):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                    removed += 1
                elif path.is_dir():
                    path.rmdir()
            except OSError:
                # A file that refuses to go is reported by count, not by
                # aborting the whole reset.
                continue
    return removed



class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ECDS Camera Launcher")
        self.geometry("760x620")
        self.minsize(500, 400) # Reduced minimum size so it fits on smaller laptop screens

        initial = default_launch_values()
        initial_camera_mode = {
            "both": "Both cameras",
            "camera_1": "Camera 1 only",
            "camera_2": "Camera 2 only",
        }.get(str(initial["camera_mode"]), "Both cameras")
        self.camera_mode = tk.StringVar(value=initial_camera_mode)
        self.setup = tk.BooleanVar(value=False)
        self.use_mediapipe = tk.BooleanVar(value=True)
        self.use_reid = tk.BooleanVar(value=True)
        self.enable_mivolo = tk.BooleanVar(value=True)
        self.use_mqtt = tk.BooleanVar(value=True)
        self.record_screen = tk.BooleanVar(value=False)
        self.disable_map_motion_filter = tk.BooleanVar(value=False)
        # TEMP_IDENTITY_DEBUG: remove with the temporary troubleshooting logger.
        self.debug_identity_events = tk.BooleanVar(value=False)
        # EXPERIMENTAL: off by default while the two-plane metrology is being
        # evaluated in shadow mode. The remembered value comes from the same
        # environment file the rest of the launcher already reads, rather than a
        # second, incompatible settings store.
        self.enable_3d_level_detection = tk.BooleanVar(
            value=is_truthy(default_launch_values().get("enable_3d_level_detection"))
        )
        # Unticked keeps the current rule, where a ground point inferred from a
        # clipped or foot-occluded box may not reject a confident appearance
        # match. Ticking it restores the older behaviour for comparison.
        self.disable_position_confidence_gating = tk.BooleanVar(
            value=is_truthy(default_launch_values().get("disable_position_confidence_gating"))
        )

        device_labels = {"0": "GPU (0)", "1": "GPU (1)"}
        self.yolo_device_1 = tk.StringVar(
            value=device_labels.get(str(initial["yolo_device_1"]), "CPU")
        )
        self.yolo_device_2 = tk.StringVar(
            value=device_labels.get(str(initial["yolo_device_2"]), "CPU")
        )
        self.reid_device = tk.StringVar(
            value={"cuda:0": "GPU (0)", "cuda:1": "GPU (1)"}.get(
                str(initial["reid_device"]), "CPU"
            )
        )
        mediapipe_labels = {
            "auto": "Auto",
            "gpu": "GPU (display/default)",
            "gpu:0": "GPU (0)",
            "gpu:1": "GPU (1)",
            "cpu": "CPU",
        }
        self.mediapipe_delegate = tk.StringVar(
            value=mediapipe_labels.get(
                str(initial["mediapipe_delegate"]).strip().lower(), "Auto"
            )
        )

        self.source_1 = tk.StringVar(value=str(initial["source_1"]))
        self.source_2 = tk.StringVar(value=str(initial["source_2"]))
        self.camera_id_1 = tk.StringVar(value=str(initial["camera_id_1"]))
        self.camera_id_2 = tk.StringVar(value=str(initial["camera_id_2"]))
        self.matrix_1 = tk.StringVar(value=str(initial["matrix_1"]))
        self.matrix_2 = tk.StringVar(value=str(initial["matrix_2"]))
        self.corner_1 = tk.StringVar(value="None")
        self.corner_2 = tk.StringVar(value="None")

        self.model = tk.StringVar(value=str(initial["model"]))
        self.yolo_confidence = tk.StringVar(value=str(initial["yolo_confidence"]))
        self.yolo_nms_iou = tk.StringVar(value=str(initial["yolo_nms_iou"]))
        self.tracker_config = tk.StringVar(value=str(initial["tracker_config"]))
        self.run_id = tk.StringVar(value=str(initial["run_id"]))
        self.tent_size_cm = tk.StringVar(value=str(initial["map_size_cm"]))
        self.map_grid_columns = tk.StringVar(value=str(initial["map_grid_columns"]))
        self.map_grid_rows = tk.StringVar(value=str(initial["map_grid_rows"]))

        # --- Fusion settings: every knob that affects how cam_1 and cam_2
        # detections get merged into one cross-camera identity. ---
        self.fusion_distance = tk.StringVar(value=str(initial["fusion_distance_cm"]))
        self.fusion_max_skew_seconds = tk.StringVar(
            value=str(initial["cross_camera_max_skew_seconds"])
        )
        # Note: whether ReID is a hard veto on fusion is controlled by the
        # existing "Appearance ReID" checkbox (self.use_reid) below -- it's
        # the same --use-appearance-reid flag main_tracker.py passes straight
        # through as fuse_camera_points(require_reid=...). No separate toggle.
        self.reid_distance_threshold = tk.StringVar(
            value=str(initial["reid_distance_threshold"])
        )

        self.mqtt_broker = tk.StringVar(value=str(initial["mqtt_broker"]))
        self.mqtt_port = tk.StringVar(value=str(initial["mqtt_port"]))
        self.reid_api_url = tk.StringVar(value=str(initial["reid_api_url"]))
        self.tracker_process: subprocess.Popen | None = None

        self.command_preview = tk.StringVar(value="")
        self._build_ui()
        self._refresh_preview()

    def _build_ui(self):
        # --- SCROLLING SETUP ---
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        container = ttk.Frame(canvas, padding=16)
        canvas_window = canvas.create_window((0, 0), window=container, anchor="nw")
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Configure canvas resizing
        container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        # Mousewheel binding for Windows
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.bind_all("<MouseWheel>", _on_mousewheel)
        self.bind_all("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"))
        self.bind_all("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"))
        # --- END SCROLLING SETUP ---

        title = ttk.Label(container, text="Camera Run Setup", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w")

        mode_frame = ttk.LabelFrame(container, text="Run Mode", padding=12)
        mode_frame.pack(fill="x", pady=(12, 8))
        ttk.Label(mode_frame, text="Camera selection").grid(row=0, column=0, sticky="w")
        mode = ttk.Combobox(
            mode_frame,
            textvariable=self.camera_mode,
            values=["Camera 1 only", "Camera 2 only", "Both cameras"],
            state="readonly",
            width=22,
        )
        mode.grid(row=0, column=1, sticky="w", padx=(12, 0))
        mode.bind("<<ComboboxSelected>>", self._on_change)

        options_frame = ttk.Frame(mode_frame)
        options_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(12, 0))
        for label, variable in [
            ("Setup calibration", self.setup),
            ("MediaPipe feet", self.use_mediapipe),
            ("Appearance ReID", self.use_reid),
            ("MQTT publish", self.use_mqtt),
        ]:
            checkbox = ttk.Checkbutton(options_frame, text=label, variable=variable, command=self._refresh_preview)
            checkbox.pack(side="left", padx=(0, 18))

        ttk.Label(mode_frame, text="Camera 1 YOLO Device").grid(row=2, column=0, sticky="w", pady=(12, 0))
        yolo_box_1 = ttk.Combobox(mode_frame, textvariable=self.yolo_device_1, values=["GPU (0)", "GPU (1)", "CPU"], state="readonly", width=22)
        yolo_box_1.grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(12, 0))
        yolo_box_1.bind("<<ComboboxSelected>>", self._on_change)

        ttk.Label(mode_frame, text="Camera 2 YOLO Device").grid(row=3, column=0, sticky="w", pady=(4, 0))
        yolo_box_2 = ttk.Combobox(mode_frame, textvariable=self.yolo_device_2, values=["GPU (0)", "GPU (1)", "CPU"], state="readonly", width=22)
        yolo_box_2.grid(row=3, column=1, sticky="w", padx=(12, 0), pady=(4, 0))
        yolo_box_2.bind("<<ComboboxSelected>>", self._on_change)

        # 🚨 REID DEVICE DROPDOWN 🚨
        ttk.Label(mode_frame, text="ReID Device").grid(row=4, column=0, sticky="w", pady=(4, 0))
        reid_box = ttk.Combobox(mode_frame, textvariable=self.reid_device, values=["GPU (0)", "GPU (1)", "CPU"], state="readonly", width=22)
        reid_box.grid(row=4, column=1, sticky="w", padx=(12, 0), pady=(4, 0))
        reid_box.bind("<<ComboboxSelected>>", self._on_change)

        ttk.Label(mode_frame, text="MediaPipe Device (both cameras)").grid(row=5, column=0, sticky="w", pady=(4, 0))
        mediapipe_box = ttk.Combobox(
            mode_frame,
            textvariable=self.mediapipe_delegate,
            values=["Auto", "GPU (0)", "GPU (1)", "GPU (display/default)", "CPU"],
            state="readonly",
            width=22,
        )
        mediapipe_box.grid(row=5, column=1, sticky="w", padx=(12, 0), pady=(4, 0))
        mediapipe_box.bind("<<ComboboxSelected>>", self._on_change)
        ttk.Checkbutton(
            mode_frame,
            text="MiVOLO age/gender analysis",
            variable=self.enable_mivolo,
            command=self._refresh_preview,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        # TEMP_IDENTITY_DEBUG
        ttk.Checkbutton(
            mode_frame,
            text="Temporary identity event log",
            variable=self.debug_identity_events,
            command=self._refresh_preview,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            mode_frame,
            text="Record desktop screen (no audio)",
            variable=self.record_screen,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            mode_frame,
            text="Disable tactical-map speed hold",
            variable=self.disable_map_motion_filter,
            command=self._refresh_preview,
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))
        # EXPERIMENTAL: two-plane metrology. Unticking this must remove every
        # trace of the feature from the run, not merely hide its output.
        ttk.Checkbutton(
            mode_frame,
            text="Enable 3D Level Detection (experimental)",
            variable=self.enable_3d_level_detection,
            command=self._refresh_preview,
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.three_d_status = ttk.Label(mode_frame, text="3D Level Detection: Disabled")
        self.three_d_status.grid(row=11, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Checkbutton(
            mode_frame,
            text="Disable position confidence gating (legacy location rule)",
            variable=self.disable_position_confidence_gating,
            command=self._refresh_preview,
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(8, 0))

        camera_frame = ttk.LabelFrame(container, text="Cameras", padding=12)
        camera_frame.pack(fill="x", pady=8)
        camera_frame.columnconfigure(1, weight=1)

        self._add_camera_rows(camera_frame, row_offset=0, label="Camera 1", source=self.source_1, camera_id=self.camera_id_1, matrix=self.matrix_1, corner=self.corner_1)
        ttk.Separator(camera_frame).grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        self._add_camera_rows(camera_frame, row_offset=5, label="Camera 2", source=self.source_2, camera_id=self.camera_id_2, matrix=self.matrix_2, corner=self.corner_2)

        advanced = ttk.LabelFrame(container, text="Common Settings", padding=12)
        advanced.pack(fill="x", pady=8)
        advanced.columnconfigure(1, weight=1)
        self._entry(advanced, "YOLO model", self.model, 0)
        self._entry(advanced, "YOLO confidence", self.yolo_confidence, 1, width=12)
        self._entry(advanced, "YOLO NMS IoU", self.yolo_nms_iou, 2, width=12)
        self._entry(advanced, "Tracker config", self.tracker_config, 3)
        self._entry(advanced, "Run ID", self.run_id, 4)
        self._entry(advanced, "MQTT broker", self.mqtt_broker, 5)
        self._entry(advanced, "MQTT port", self.mqtt_port, 6, width=12)
        self._entry(advanced, "ReID backend URL", self.reid_api_url, 7)
        self._entry(advanced, "Tent side length (cm)", self.tent_size_cm, 8, width=12)
        self._entry(advanced, "Tactical-map grid columns", self.map_grid_columns, 9, width=12)
        self._entry(advanced, "Tactical-map grid rows", self.map_grid_rows, 10, width=12)
        ttk.Label(
            advanced,
            text="Grid columns and rows only change the visual layout; they do not change the tent size or calibration.",
            wraplength=560,
            justify="left",
            foreground="#555555",
        ).grid(row=11, column=0, columnspan=3, sticky="w", pady=(2, 0))

        fusion_frame = ttk.LabelFrame(container, text="Cross-Camera Fusion", padding=12)
        fusion_frame.pack(fill="x", pady=8)
        fusion_frame.columnconfigure(1, weight=1)
        ttk.Label(
            fusion_frame,
            text="Controls how a cam_1 detection and a cam_2 detection get merged into one tracked person.",
            wraplength=560,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self._entry(fusion_frame, "Max distance (cm)", self.fusion_distance, 1, width=12)
        ttk.Label(
            fusion_frame,
            text="How far apart (on the tactical map) two camera detections can be and still be considered the same person.",
            wraplength=560,
            justify="left",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self._entry(fusion_frame, "Max time skew (s)", self.fusion_max_skew_seconds, 3, width=12)
        ttk.Label(
            fusion_frame,
            text="How far apart in capture time two camera detections can be and still be paired together.",
            wraplength=560,
            justify="left",
            foreground="#555555",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self._entry(fusion_frame, "ReID match strictness (0-1)", self.reid_distance_threshold, 5, width=12)
        ttk.Label(
            fusion_frame,
            text=(
                "How much appearance (ReID) weighs into fusion. Lower = stricter appearance match required "
                "before two detections merge (fewer false merges, more missed merges). Higher = looser matching "
                "(more merges, more risk of merging different people). Only applies when \"Appearance ReID\" is "
                "checked above -- otherwise fusion uses distance/time only."
            ),
            wraplength=560,
            justify="left",
            foreground="#555555",
        ).grid(row=6, column=0, columnspan=3, sticky="w")

        preview_frame = ttk.LabelFrame(container, text="Command Preview", padding=12)
        preview_frame.pack(fill="both", expand=True, pady=8)
        preview = tk.Text(preview_frame, height=4, wrap="word")
        preview.pack(fill="both", expand=True)
        preview.configure(state="disabled")
        self.preview_widget = preview

        button_frame = ttk.Frame(container)
        button_frame.pack(fill="x", pady=(8, 0))
        ttk.Button(button_frame, text="Refresh", command=self._refresh_preview).pack(side="left")
        ttk.Button(
            button_frame,
            text="Reset database and ReID evidence…",
            command=self._reset_run_data,
        ).pack(side="left", padx=(12, 0))
        ttk.Button(button_frame, text="Start", command=self._start).pack(side="right")

        for variable in [
            self.yolo_device_1,
            self.yolo_device_2,
            self.reid_device,
            self.mediapipe_delegate,
            self.source_1,
            self.source_2,
            self.camera_id_1,
            self.camera_id_2,
            self.matrix_1,
            self.matrix_2,
            self.corner_1,
            self.corner_2,
            self.model,
            self.yolo_confidence,
            self.yolo_nms_iou,
            self.tracker_config,
            self.run_id,
            self.tent_size_cm,
            self.map_grid_columns,
            self.map_grid_rows,
            self.fusion_distance,
            self.fusion_max_skew_seconds,
            self.reid_distance_threshold,
            self.mqtt_broker,
            self.mqtt_port,
            self.reid_api_url,
        ]:
            variable.trace_add("write", lambda *_args: self._refresh_preview())

    def _add_camera_rows(self, parent, row_offset, label, source, camera_id, matrix, corner):
        ttk.Label(parent, text=label, font=("Segoe UI", 10, "bold")).grid(row=row_offset, column=0, sticky="w", pady=(0, 4))
        self._entry(parent, "Source", source, row_offset + 1)
        self._entry(parent, "Camera ID", camera_id, row_offset + 2, width=18)
        self._entry(parent, "Matrix", matrix, row_offset + 3, width=28)
        ttk.Label(parent, text="Location").grid(row=row_offset + 2, column=2, sticky="e", padx=(16, 8))
        corner_box = ttk.Combobox(parent, textvariable=corner, values=CORNER_OPTIONS, state="readonly", width=18)
        corner_box.grid(row=row_offset + 3, column=2, sticky="e")
        corner_box.bind("<<ComboboxSelected>>", self._on_change)

    def _entry(self, parent, label, variable, row, width=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=3)
        return entry

    def _on_change(self, _event=None):
        self._refresh_preview()

    def _selected_cameras(self):
        mode = self.camera_mode.get()
        if mode == "Camera 1 only":
            return ["1"]
        if mode == "Camera 2 only":
            return ["2"]
        return ["1", "2"]

    def _build_command(self, identity_debug_log=None):
        selected = self._selected_cameras()
        values = default_launch_values()
        values.update(
            {
                "camera_mode": (
                    "camera_1" if selected == ["1"] else "camera_2" if selected == ["2"] else "both"
                ),
                "setup": self.setup.get(),
                "use_mediapipe": self.use_mediapipe.get(),
                "use_reid": self.use_reid.get(),
                "enable_mivolo": self.enable_mivolo.get(),
                "use_mqtt": self.use_mqtt.get(),
                "disable_map_motion_filter": self.disable_map_motion_filter.get(),
                "debug_identity_events": self.debug_identity_events.get(),
                "identity_debug_log": identity_debug_log,
                "enable_3d_level_detection": self.enable_3d_level_detection.get(),
                "disable_position_confidence_gating": self.disable_position_confidence_gating.get(),
                "yolo_device_1": {"GPU (0)": "0", "GPU (1)": "1"}.get(
                    self.yolo_device_1.get(), "cpu"
                ),
                "yolo_device_2": {"GPU (0)": "0", "GPU (1)": "1"}.get(
                    self.yolo_device_2.get(), "cpu"
                ),
                "reid_device": {"GPU (0)": "cuda:0", "GPU (1)": "cuda:1"}.get(
                    self.reid_device.get(), "cpu"
                ),
                "mediapipe_delegate": {
                    "GPU (0)": "gpu:0",
                    "GPU (1)": "gpu:1",
                    "GPU (display/default)": "gpu",
                    "CPU": "cpu",
                }.get(self.mediapipe_delegate.get(), "auto"),
                "source_1": self.source_1.get().strip(),
                "source_2": self.source_2.get().strip(),
                "camera_id_1": self.camera_id_1.get().strip(),
                "camera_id_2": self.camera_id_2.get().strip(),
                "matrix_1": self.matrix_1.get().strip(),
                "matrix_2": self.matrix_2.get().strip(),
                "missing_corner_1": self.corner_1.get(),
                "missing_corner_2": self.corner_2.get(),
                "model": self.model.get().strip() or "yolo26m.pt",
                "yolo_confidence": self.yolo_confidence.get().strip() or "0.75",
                "yolo_nms_iou": self.yolo_nms_iou.get().strip() or str(DEFAULT_YOLO_NMS_IOU),
                "tracker_config": self.tracker_config.get().strip() or DEFAULT_TRACKER_CONFIG_PATH,
                "run_id": self.run_id.get().strip() or "field_test_001",
                "map_size_cm": self.tent_size_cm.get().strip(),
                "map_grid_columns": self.map_grid_columns.get().strip(),
                "map_grid_rows": self.map_grid_rows.get().strip(),
                "fusion_distance_cm": self.fusion_distance.get().strip(),
                "cross_camera_max_skew_seconds": self.fusion_max_skew_seconds.get().strip(),
                "reid_distance_threshold": self.reid_distance_threshold.get().strip(),
                "mqtt_broker": self.mqtt_broker.get().strip(),
                "mqtt_port": self.mqtt_port.get().strip(),
                "reid_api_url": self.reid_api_url.get().strip(),
            }
        )
        return [sys.executable, str(SCRIPT_PATH), *build_tracker_arguments(values)]

    def _three_d_calibration_problem(self):
        """Return a message when 3D level detection cannot start, else None.

        Checked before launch so an invalid experimental calibration surfaces in
        the launcher instead of silently degrading a live run. Standard 2D
        tracking is never blocked by this.
        """
        if not self.enable_3d_level_detection.get():
            return None
        from three_d_level import CalibrationError, build_plane_calibration

        values = default_launch_values()
        selected = self._selected_cameras()
        pairs = []
        if "1" in selected:
            pairs.append((self.camera_id_1.get() or "Camera 1", self.matrix_1.get(),
                          values.get("elevated_matrix_1")))
        if "2" in selected:
            pairs.append((self.camera_id_2.get() or "Camera 2", self.matrix_2.get(),
                          values.get("elevated_matrix_2")))
        for camera_id, ground, elevated in pairs:
            try:
                build_plane_calibration(camera_id, ROOT / str(ground), ROOT / str(elevated))
            except CalibrationError as error:
                return (
                    f"3D Level Detection could not start because the elevated-plane "
                    f"calibration for {camera_id} is missing or invalid. The system will "
                    f"continue with standard 2D ground-plane tracking.\n\n"
                    f"File: {elevated}\n{error}"
                )
            except Exception as error:
                return (
                    f"3D Level Detection could not start for {camera_id}. The system will "
                    f"continue with standard 2D ground-plane tracking.\n\n{error}"
                )
        return None

    def _refresh_three_d_status(self):
        # _refresh_preview runs once while the window is still being built, so
        # the label may not exist yet.
        label = getattr(self, "three_d_status", None)
        if label is None:
            return
        if not self.enable_3d_level_detection.get():
            label.configure(text="3D Level Detection: Disabled")
            return
        problem = self._three_d_calibration_problem()
        label.configure(
            text=(
                "3D Level Detection: Unavailable - calibration error"
                if problem
                else "3D Level Detection: Enabled"
            )
        )

    def _refresh_preview(self):
        self._refresh_three_d_status()
        command = self._build_command()
        safe_command = redact_tracker_arguments(command)
        pretty = subprocess.list2cmdline(safe_command) if os.name == "nt" else shlex.join(safe_command)
        self.preview_widget.configure(state="normal")
        self.preview_widget.delete("1.0", "end")
        self.preview_widget.insert("1.0", pretty)
        self.preview_widget.configure(state="disabled")

    def _validate(self):
        selected = self._selected_cameras()
        checks = []
        if "1" in selected:
            checks.extend([
                ("Camera 1 source", self.source_1.get()),
                ("Camera 1 ID", self.camera_id_1.get()),
                ("Camera 1 matrix", self.matrix_1.get()),
            ])
        if "2" in selected:
            checks.extend([
                ("Camera 2 source", self.source_2.get()),
                ("Camera 2 ID", self.camera_id_2.get()),
                ("Camera 2 matrix", self.matrix_2.get()),
            ])
        for label, value in checks:
            if not value.strip():
                messagebox.showerror("Missing value", f"{label} cannot be empty.")
                return False
        try:
            nms_iou = float(self.yolo_nms_iou.get())
        except ValueError:
            messagebox.showerror("Invalid value", "YOLO NMS IoU must be a number greater than 0 and at most 1.")
            return False
        if not 0.0 < nms_iou <= 1.0:
            messagebox.showerror("Invalid value", "YOLO NMS IoU must be greater than 0 and at most 1.")
            return False

        try:
            tent_size_cm = int(self.tent_size_cm.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Tent side length (cm) must be a whole number.")
            return False
        if tent_size_cm <= 0:
            messagebox.showerror("Invalid value", "Tent side length (cm) must be greater than 0.")
            return False

        for label, variable in [
            ("Tactical-map grid columns", self.map_grid_columns),
            ("Tactical-map grid rows", self.map_grid_rows),
        ]:
            try:
                grid_count = int(variable.get())
            except ValueError:
                messagebox.showerror("Invalid value", f"{label} must be a whole number.")
                return False
            if not 1 <= grid_count <= 50:
                messagebox.showerror("Invalid value", f"{label} must be between 1 and 50.")
                return False

        if "1" in selected and "2" in selected:
            try:
                fusion_distance_cm = float(self.fusion_distance.get())
            except ValueError:
                messagebox.showerror("Invalid value", "Fusion max distance (cm) must be a number.")
                return False
            if fusion_distance_cm <= 0:
                messagebox.showerror("Invalid value", "Fusion max distance (cm) must be greater than 0.")
                return False

            try:
                fusion_skew_seconds = float(self.fusion_max_skew_seconds.get())
            except ValueError:
                messagebox.showerror("Invalid value", "Fusion max time skew (s) must be a number.")
                return False
            if fusion_skew_seconds <= 0:
                messagebox.showerror("Invalid value", "Fusion max time skew (s) must be greater than 0.")
                return False

        if self.use_reid.get():
            try:
                reid_distance_threshold = float(self.reid_distance_threshold.get())
            except ValueError:
                messagebox.showerror("Invalid value", "ReID match strictness must be a number.")
                return False
            if not 0.0 < reid_distance_threshold <= 1.0:
                messagebox.showerror(
                    "Invalid value",
                    "ReID match strictness should be greater than 0 and at most 1 (it's a cosine distance).",
                )
                return False

        return True

    @staticmethod
    def _record_screen(process, output_path, fps=10.0):
        writer = None
        try:
            import cv2
            import numpy as np
            from PIL import ImageGrab

            first_image = ImageGrab.grab(all_screens=True)
            width = first_image.width - (first_image.width % 2)
            height = first_image.height - (first_image.height % 2)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("Unable to open the MP4 screen-recording writer.")

            frame_interval = 1.0 / float(fps)
            next_capture = time.monotonic()
            while process.poll() is None:
                now = time.monotonic()
                if now < next_capture:
                    time.sleep(min(next_capture - now, 0.05))
                    continue

                image = ImageGrab.grab(all_screens=True)
                rgb_frame = np.asarray(image)
                if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 4:
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2BGR)
                else:
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                if bgr_frame.shape[1] != width or bgr_frame.shape[0] != height:
                    bgr_frame = cv2.resize(bgr_frame, (width, height))
                writer.write(bgr_frame)

                next_capture += frame_interval
                if next_capture < now:
                    next_capture = now + frame_interval
        except Exception as exc:
            error_path = output_path.with_suffix(".error.txt")
            try:
                error_path.write_text(str(exc), encoding="utf-8")
            except OSError:
                pass
        finally:
            if writer is not None:
                writer.release()

    @staticmethod
    def _matching_process_ids(predicate):
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,args="],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Unable to inspect running processes: {exc}") from exc

        process_ids = []
        for line in result.stdout.splitlines():
            pid_text, _, command = line.strip().partition(" ")
            if pid_text.isdigit() and predicate(command):
                process_ids.append(int(pid_text))
        return process_ids

    @staticmethod
    def _stop_process(pid, signal_number=signal.SIGTERM):
        try:
            process_group_id = os.getpgid(pid)
            if process_group_id == pid:
                os.killpg(process_group_id, signal_number)
            else:
                os.kill(pid, signal_number)
        except ProcessLookupError:
            return

    @staticmethod
    def _process_has_stopped(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True

        if os.name != "nt":
            try:
                process_state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
            except (FileNotFoundError, IndexError):
                return True
            return process_state == "Z"
        return False

    @classmethod
    def _wait_for_processes_to_stop(cls, process_ids, timeout_seconds=8.0):
        deadline = time.monotonic() + timeout_seconds
        remaining = set(process_ids)
        while remaining and time.monotonic() < deadline:
            for pid in tuple(remaining):
                if cls._process_has_stopped(pid):
                    remaining.remove(pid)
            if remaining:
                time.sleep(0.1)
        return remaining

    def _reset_run_data(self):
        confirmation = simpledialog.askstring(
            "Reset database and ReID evidence",
            "This stops the tracker, backend, and Mosquitto; then permanently deletes:\n"
            f"• {DATABASE_PATH.name}, {DATABASE_PATH.name}-wal, and {DATABASE_PATH.name}-shm\n"
            f"• {REID_EVIDENCE_DIRECTORY.name}/\n\n"
            "Type RESET to continue.",
            parent=self,
        )
        if confirmation != "RESET":
            return

        try:
            tracker_pids = self._matching_process_ids(
                lambda command: str(SCRIPT_PATH) in command
            )
            backend_pids = self._matching_process_ids(
                lambda command: str(BACKEND_DIRECTORY) in command
                and "uvicorn" in command
                and "main:app" in command
            )
            project_process_ids = sorted(set(tracker_pids + backend_pids))
            for pid in project_process_ids:
                self._stop_process(pid)
            remaining_process_ids = self._wait_for_processes_to_stop(project_process_ids)
            if remaining_process_ids:
                for pid in remaining_process_ids:
                    self._stop_process(pid, signal.SIGKILL)
                remaining_process_ids = self._wait_for_processes_to_stop(
                    remaining_process_ids,
                    timeout_seconds=2.0,
                )
            if remaining_process_ids:
                joined_pids = ", ".join(str(pid) for pid in sorted(remaining_process_ids))
                raise RuntimeError(f"These project processes could not be stopped: {joined_pids}")
            self.tracker_process = None

            service_status = subprocess.run(
                ["systemctl", "is-active", "--quiet", "mosquitto"],
                check=False,
            )
            if service_status.returncode == 0:
                stop_service = subprocess.run(
                    ["pkexec", "systemctl", "stop", "mosquitto"],
                    check=False,
                )
                if stop_service.returncode != 0:
                    raise RuntimeError("Mosquitto was not stopped; no data was deleted.")

            removed_paths = []
            database_summary = self._clear_passenger_data()
            if REID_EVIDENCE_DIRECTORY.exists():
                shutil.rmtree(REID_EVIDENCE_DIRECTORY)
                removed_paths.append(str(REID_EVIDENCE_DIRECTORY))

            summary = "\n".join(f"• {path}" for path in removed_paths) or "• (no evidence folder)"
            messagebox.showinfo(
                "Reset complete",
                "The tracker, backend, and Mosquitto are stopped.\n\n"
                + database_summary
                + "\n\nRemoved:\n"
                + summary,
            )
        except (OSError, RuntimeError) as exc:
            messagebox.showerror("Reset not completed", str(exc))

    def _clear_passenger_data(self):
        """Delete passenger rows while keeping operator accounts intact.

        The database is no longer removed, because it also holds users,
        sessions, the audit trail, and the migration revision. Deleting the file
        would sign everyone out permanently and force a fresh bootstrap.
        """
        if not DATABASE_PATH.exists():
            return "No database existed, so there was no passenger data to clear."

        connection = sqlite3.connect(DATABASE_PATH)
        try:
            existing = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "users" not in existing:
                # Pre-authentication database: nothing to protect, so keep the
                # original behaviour and remove the file outright.
                connection.close()
                removed = []
                for database_file in (
                    DATABASE_PATH,
                    DATABASE_PATH.with_name(f"{DATABASE_PATH.name}-wal"),
                    DATABASE_PATH.with_name(f"{DATABASE_PATH.name}-shm"),
                ):
                    if database_file.exists():
                        database_file.unlink()
                        removed.append(database_file.name)
                return "Removed the pre-authentication database: " + ", ".join(removed)

            deleted_counts = {}
            for table in PASSENGER_DATA_TABLES:
                if table in existing:
                    cursor = connection.execute(f"DELETE FROM {table}")
                    deleted_counts[table] = cursor.rowcount
            connection.commit()
            account_count = connection.execute("SELECT count(*) FROM users").fetchone()[0]
        finally:
            connection.close()

        removed_images = clear_evidence_files()
        cleared = ", ".join(
            f"{table.replace('_', ' ')}: {count}"
            for table, count in deleted_counts.items()
            if count
        ) or "nothing to clear"
        return (
            f"Passenger data cleared ({cleared}).\n"
            f"{removed_images} evidence image(s) removed.\n"
            f"{account_count} operator account(s) and the sign-in history were kept."
        )


    def _start(self):
        if not self._validate():
            return

        # An unusable experimental calibration must not silently ride along with
        # the checkbox still showing as enabled, and must not stop 2D tracking.
        three_d_problem = self._three_d_calibration_problem()
        if three_d_problem is not None:
            messagebox.showerror("3D Level Detection unavailable", three_d_problem)
            self.enable_3d_level_detection.set(False)
            self._refresh_preview()

        try:
            preflight_lock = CvRuntimeLock("technical tester launcher").acquire()
        except CvRuntimeBusyError as exc:
            messagebox.showerror("Computer vision already running", str(exc))
            return
        else:
            preflight_lock.release()

        safe_run_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in (self.run_id.get().strip() or "field_test")
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        identity_debug_log = None
        if self.debug_identity_events.get():
            try:
                LOG_EVIDENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
                identity_debug_log = LOG_EVIDENCE_DIRECTORY / f"{safe_run_id}_{timestamp}.jsonl"
            except OSError as exc:
                messagebox.showerror("Log folder unavailable", str(exc))
                return

        command = self._build_command(identity_debug_log=identity_debug_log)
        runtime_log = None
        try:
            if os.name == "nt":
                process = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                LOG_EVIDENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
                runtime_log = LOG_EVIDENCE_DIRECTORY / f"{safe_run_id}_{timestamp}.console.log"
                runtime_stream = runtime_log.open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=str(ROOT),
                        stdout=runtime_stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    runtime_stream.close()
        except OSError as exc:
            messagebox.showerror("Launch failed", str(exc))
            return

        self.tracker_process = process

        recording_path = None
        if self.record_screen.get():
            try:
                SCREEN_RECORDING_DIRECTORY.mkdir(parents=True, exist_ok=True)
                recording_path = SCREEN_RECORDING_DIRECTORY / f"{safe_run_id}_{timestamp}.mp4"
                threading.Thread(
                    target=self._record_screen,
                    args=(process, recording_path),
                    daemon=False,
                    name="desktop-screen-recorder",
                ).start()
            except OSError as exc:
                messagebox.showwarning("Recording not started", str(exc))

        message = (
            "Camera detection has started in a new console window."
            if os.name == "nt"
            else "Camera detection has started in the background."
        )
        if runtime_log is not None:
            message += f"\n\nRuntime output:\n{runtime_log}"
        if identity_debug_log is not None:
            message += f"\n\nLog evidence:\n{identity_debug_log}"
        if recording_path is not None:
            message += f"\n\nScreen recording:\n{recording_path}"
        messagebox.showinfo("Started", message)


if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()
