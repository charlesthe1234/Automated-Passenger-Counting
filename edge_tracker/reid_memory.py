"""Cross-camera identity memory.

Owns AppearanceIdentityMemory: the gallery of confirmed and provisional
identities, the arbitration that decides which track holds which identity,
and the worker threads behind intake, semantic probes, conflict resolution,
and evidence persistence.

The models it scores with live in reid_models, and the cropping those models
consume lives in reid_crops. Neither imports this module back.
"""

import hashlib
import copy
import os
import pickle
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
from constants import (
    DEFAULT_DEMOGRAPHICS_CROP_COUNT,
    DEFAULT_DEMOGRAPHICS_MAX_REFRESHES,
    DEFAULT_DEMOGRAPHICS_REFRESH_QUALITY_RATIO,
    DEFAULT_GALLERY_ADMISSION_DISTANCE,
    DEFAULT_IDENTITY_AUDIT_CONTEST_PATIENCE_SECONDS,
    DEFAULT_IDENTITY_AUDIT_INTERVAL_SECONDS,
    DEFAULT_IDENTITY_AUDIT_MARGIN,
    DEFAULT_IDENTITY_AUDIT_ROUNDS,
    DEFAULT_NEW_MATCH_POSITION_SLACK_RATIO,
    DEFAULT_NEW_MATCH_POSITION_SPLIT_FRAMES,
    DEFAULT_PHYSICAL_CONFLICT_BLUR_TIMEOUT_SECONDS,
    DEFAULT_PHYSICAL_CONFLICT_RECOVERY_GRACE_FRAMES,
    DEFAULT_PHYSICAL_CONFLICT_RECOVERY_MAX_FRAMES,
    DEFAULT_PHYSICAL_CONFLICT_REID_FRAMES,
    DEFAULT_PHYSICAL_CONFLICT_REID_MARGIN,
    DEFAULT_PHYSICAL_CONFLICT_STALL_SECONDS,
    DEFAULT_POSITION_SPLIT_FRAMES,
    DEFAULT_PROVISIONAL_BASELINE_VETO_DISTANCE,
    DEFAULT_PROVISIONAL_CHALLENGE_DISTANCE,
    DEFAULT_PROVISIONAL_LOCATION_CONFIRM_FRAMES,
    DEFAULT_PROVISIONAL_MERGE_DISTANCE,
    DEFAULT_PROVISIONAL_SPLIT_RECOVERY_SECONDS,
    DEFAULT_REID_BLUR_THRESHOLD,
    DEFAULT_REID_DISTANCE_THRESHOLD,
    DEFAULT_REID_EMA_ALPHA,
    DEFAULT_REID_INTAKE_DELAY_SECONDS,
    DEFAULT_REID_INTAKE_RETRY_FRAMES,
    DEFAULT_REID_INTAKE_TIMEOUT_SECONDS,
    DEFAULT_REID_MAX_RETRY_FRAMES,
    DEFAULT_REID_MEMORY_TTL_FRAMES,
    DEFAULT_REID_QUEUE_SIZE,
    DEFAULT_REID_ROLE_CHECKPOINT,
    DEFAULT_REID_ROLE_CONFIDENCE,
    DEFAULT_REID_SEMANTIC_CONFIDENCE,
    DEFAULT_REID_SEMANTIC_COOLDOWN_FRAMES,
    DEFAULT_REID_SEMANTIC_RETRY_FRAMES,
    DEFAULT_REID_SHADOW_CENTER_DISTANCE_RATIO,
    DEFAULT_REID_SHADOW_CONTAINMENT_THRESHOLD,
    DEFAULT_REID_SHADOW_IOU_THRESHOLD,
    DEFAULT_REID_SHADOW_PROBATION_FRAMES,
    DEFAULT_REID_SHADOW_SEPARATION_FRAMES,
    DEFAULT_REID_SIMILARITY_THRESHOLD,
    DEFAULT_TRACK_ABANDON_FRAMES,
    REID_GALLERY_SLOTS,
    REID_SEMANTIC_SLOTS,
)
from identity_debug import identity_event

from face_region import normalized_box_to_pixels
from reid_crops import (
    compute_color_reid_feature,
    image_sharpness,
)
from reid_models import EvacuationRoleClassifier


class AppearanceIdentityMemory:
    """Shared, asynchronous five-slot TransReID identity coordinator.

    Local tracker IDs are namespaced by camera. A new local track contributes
    five quality-controlled crops, which are processed as one batch by a
    background worker. Mapped tracks are dictionary lookups except for a
    bounded single-crop inference used to fill a genuinely missing semantic
    orientation slot.
    """

    SCHEMA_VERSION = 3

    def __init__(
        self,
        similarity_threshold=DEFAULT_REID_SIMILARITY_THRESHOLD,
        ttl_frames=DEFAULT_REID_MEMORY_TTL_FRAMES,
        track_abandon_frames=DEFAULT_TRACK_ABANDON_FRAMES,
        ema_alpha=DEFAULT_REID_EMA_ALPHA,
        reid_extractor=None,
        verbose=False,
        distance_threshold=None,
        morph_threshold=0.08,
        max_gallery_size=5,
        db_path=None,
        persistence_store=None,
        intake_frames=5,
        gallery_update_interval_frames=DEFAULT_REID_SEMANTIC_COOLDOWN_FRAMES,
        evidence_dir=None,
        evidence_camera_ids=None,
        intake_delay_seconds=DEFAULT_REID_INTAKE_DELAY_SECONDS,
        intake_timeout_seconds=DEFAULT_REID_INTAKE_TIMEOUT_SECONDS,
        blur_threshold=DEFAULT_REID_BLUR_THRESHOLD,
        semantic_confidence_threshold=DEFAULT_REID_SEMANTIC_CONFIDENCE,
        semantic_cooldown_frames=None,
        semantic_retry_frames=DEFAULT_REID_SEMANTIC_RETRY_FRAMES,
        intake_retry_frames=DEFAULT_REID_INTAKE_RETRY_FRAMES,
        max_retry_frames=DEFAULT_REID_MAX_RETRY_FRAMES,
        queue_size=DEFAULT_REID_QUEUE_SIZE,
        role_checkpoint=DEFAULT_REID_ROLE_CHECKPOINT,
        role_confidence_threshold=DEFAULT_REID_ROLE_CONFIDENCE,
        enable_role_classification=True,
        enable_demographics=True,
        demographics_device=None,
        role_classifier=None,
        demographics_engine=None,
        cross_camera_fusion_distance_cm=None,
        # When True, only two genuinely measured ground points may reject an
        # appearance match. Exposed as a switch so the previous behaviour can be
        # reproduced for comparison without editing code.
        position_confidence_gating=True,
        cross_camera_max_skew_seconds=0.35,
        shadow_iou_threshold=DEFAULT_REID_SHADOW_IOU_THRESHOLD,
        shadow_containment_threshold=DEFAULT_REID_SHADOW_CONTAINMENT_THRESHOLD,
        shadow_center_distance_ratio=DEFAULT_REID_SHADOW_CENTER_DISTANCE_RATIO,
        shadow_probation_frames=DEFAULT_REID_SHADOW_PROBATION_FRAMES,
        shadow_separation_frames=DEFAULT_REID_SHADOW_SEPARATION_FRAMES,
        provisional_challenge_distance=DEFAULT_PROVISIONAL_CHALLENGE_DISTANCE,
        provisional_baseline_veto_distance=DEFAULT_PROVISIONAL_BASELINE_VETO_DISTANCE,
        provisional_merge_distance=DEFAULT_PROVISIONAL_MERGE_DISTANCE,
        gallery_admission_distance=DEFAULT_GALLERY_ADMISSION_DISTANCE,
        identity_audit_interval_seconds=DEFAULT_IDENTITY_AUDIT_INTERVAL_SECONDS,
        identity_audit_margin=DEFAULT_IDENTITY_AUDIT_MARGIN,
        identity_audit_rounds=DEFAULT_IDENTITY_AUDIT_ROUNDS,
        identity_audit_contest_patience_seconds=DEFAULT_IDENTITY_AUDIT_CONTEST_PATIENCE_SECONDS,
        provisional_location_confirm_frames=DEFAULT_PROVISIONAL_LOCATION_CONFIRM_FRAMES,
        provisional_split_recovery_seconds=DEFAULT_PROVISIONAL_SPLIT_RECOVERY_SECONDS,
        physical_conflict_recovery_grace_frames=DEFAULT_PHYSICAL_CONFLICT_RECOVERY_GRACE_FRAMES,
        physical_conflict_recovery_max_frames=DEFAULT_PHYSICAL_CONFLICT_RECOVERY_MAX_FRAMES,
        physical_conflict_reid_frames=DEFAULT_PHYSICAL_CONFLICT_REID_FRAMES,
        physical_conflict_reid_margin=DEFAULT_PHYSICAL_CONFLICT_REID_MARGIN,
        physical_conflict_blur_timeout_seconds=(
            DEFAULT_PHYSICAL_CONFLICT_BLUR_TIMEOUT_SECONDS
        ),
        physical_conflict_stall_seconds=DEFAULT_PHYSICAL_CONFLICT_STALL_SECONDS,
        start_worker=True,
    ):
        del morph_threshold, max_gallery_size  # incompatible with fixed named slots
        self.similarity_threshold = float(similarity_threshold)
        self.distance_threshold = (
            max(0.0, 1.0 - self.similarity_threshold)
            if distance_threshold is None
            else float(distance_threshold)
        )
        if distance_threshold is None and similarity_threshold == DEFAULT_REID_SIMILARITY_THRESHOLD:
            self.distance_threshold = DEFAULT_REID_DISTANCE_THRESHOLD
        self.ttl_frames = max(1, int(ttl_frames))  # local bindings only; masters never expire
        # 0 disables the sweep entirely, for tests that drive visibility by hand.
        self.track_abandon_frames = max(0, int(track_abandon_frames))
        self.ema_alpha = float(ema_alpha)
        self.reid_extractor = reid_extractor
        self.verbose = bool(verbose)
        self.db_path = Path(db_path) if db_path else None
        self.persistence_store = persistence_store
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.evidence_camera_ids = (
            None
            if evidence_camera_ids is None
            else {str(camera_id) for camera_id in evidence_camera_ids}
        )
        self.intake_frames = max(1, int(intake_frames))
        self.intake_delay_seconds = max(0.0, float(intake_delay_seconds))
        self.intake_timeout_seconds = max(self.intake_delay_seconds, float(intake_timeout_seconds))
        self.blur_threshold = max(0.0, float(blur_threshold))
        self.semantic_confidence_threshold = float(semantic_confidence_threshold)
        self.semantic_cooldown_frames = max(
            1,
            int(
                gallery_update_interval_frames
                if semantic_cooldown_frames is None
                else semantic_cooldown_frames
            ),
        )
        self.semantic_retry_frames = max(1, int(semantic_retry_frames))
        self.intake_retry_frames = max(1, int(intake_retry_frames))
        self.max_retry_frames = max(self.intake_retry_frames, int(max_retry_frames))
        self.role_checkpoint = Path(role_checkpoint) if role_checkpoint else None
        self.role_confidence_threshold = min(1.0, max(0.0, float(role_confidence_threshold)))
        self.enable_role_classification = bool(enable_role_classification)
        self.enable_demographics = bool(enable_demographics)
        self.demographics_device = demographics_device
        self.cross_camera_fusion_distance_cm = (
            None
            if cross_camera_fusion_distance_cm is None
            else max(0.0, float(cross_camera_fusion_distance_cm))
        )
        self.cross_camera_max_skew_seconds = max(0.0, float(cross_camera_max_skew_seconds))
        self.position_confidence_gating = bool(position_confidence_gating)
        self.shadow_iou_threshold = min(1.0, max(0.0, float(shadow_iou_threshold)))
        self.shadow_containment_threshold = min(
            1.0,
            max(0.0, float(shadow_containment_threshold)),
        )
        self.shadow_center_distance_ratio = max(0.0, float(shadow_center_distance_ratio))
        self.shadow_probation_frames = max(0, int(shadow_probation_frames))
        self.shadow_separation_frames = max(1, int(shadow_separation_frames))
        self.provisional_challenge_distance = max(
            self.distance_threshold,
            float(provisional_challenge_distance),
        )
        # Cross-angle baselines of one person are far apart by construction, so
        # this veto boundary must stay at or above the same-angle challenge
        # distance.  It withholds a location-only promotion; it never confirms.
        self.provisional_baseline_veto_distance = max(
            self.provisional_challenge_distance,
            float(provisional_baseline_veto_distance),
        )
        # Deliberately independent of the cross-angle veto.  Tying the two
        # together assumed both answered the same question; they do not.  The
        # veto compares one camera's baseline with the other's, which is the
        # widest gap a single person produces.  This compares a new crop with
        # the whole gallery it is joining, and the measured spread there is
        # much narrower -- holding it to the veto's floor is what admitted a
        # second man's photographs at 0.44.
        self.gallery_admission_distance = max(0.0, float(gallery_admission_distance))
        self.provisional_merge_distance = max(0.0, float(provisional_merge_distance))
        self.identity_audit_interval_seconds = max(
            0.0,
            float(identity_audit_interval_seconds),
        )
        self.identity_audit_margin = max(0.0, float(identity_audit_margin))
        self.identity_audit_rounds = max(1, int(identity_audit_rounds))
        self.identity_audit_contest_patience_seconds = max(
            0.0,
            float(identity_audit_contest_patience_seconds),
        )
        # Per track: when the next audit is due and which rival master has been
        # winning, so a single lucky frame can never move a binding.
        self.identity_audit_state = {}
        # Existing-ID matches near the configured threshold are much more
        # vulnerable when evacuees wear similar clothing.  Distances above
        # this strong boundary require the same master to win a second,
        # independently collected intake batch before the binding is final.
        self.strong_match_distance = min(self.distance_threshold, 0.20)
        self.provisional_location_confirm_frames = max(
            1,
            int(provisional_location_confirm_frames),
        )
        self.provisional_split_recovery_seconds = max(
            0.0,
            float(provisional_split_recovery_seconds),
        )
        self.physical_conflict_reid_frames = max(
            1,
            int(physical_conflict_reid_frames),
        )
        self.physical_conflict_recovery_grace_frames = max(
            0,
            int(physical_conflict_recovery_grace_frames),
        )
        self.physical_conflict_recovery_max_frames = max(
            self.physical_conflict_recovery_grace_frames,
            int(physical_conflict_recovery_max_frames),
        )
        self.physical_conflict_reid_margin = max(
            0.0,
            float(physical_conflict_reid_margin),
        )
        self.physical_conflict_blur_timeout_seconds = max(
            0.0,
            float(physical_conflict_blur_timeout_seconds),
        )
        # Relaxation must get its chance before a hold is written off as
        # starved, or the stall rule would fire on holds the blur timeout was
        # about to rescue.
        self.physical_conflict_stall_seconds = max(
            self.physical_conflict_blur_timeout_seconds,
            float(physical_conflict_stall_seconds),
        )

        self.identities = {}
        self.next_identity_id = 1
        # Location-only cross-camera groups use negative internal tokens so
        # they cannot consume or appear as permanent master numbers.  A
        # positive ID is allocated only after their global ReID search has
        # finished without matching an established identity.
        self.next_temporary_group_id = -1
        self.track_to_identity = {}
        self.track_last_seen = {}
        self.track_results = {}
        self.track_binding_metadata = {}
        self.track_generations = {}
        self.pending_intake = {}
        # Tracks in a promising cross-camera location pair may finish their
        # GPU intake, but an unmatched result must not allocate a permanent
        # master until the short coordinator hold is resolved.
        self.new_master_holds = {}
        self.visible_track_keys_by_camera = {}
        self.track_boxes = {}
        self.shadow_tracks = {}
        self.pending_semantic_slots = set()
        self.next_semantic_attempt_frame = {}
        self.semantic_probe_quality = {}
        self.recent_master_observations = {}
        # track_key -> ("hard"|"soft", reason). How much the physical-distance
        # gate is allowed to conclude from this track's current ground point.
        self.track_position_evidence = {}
        self.physical_violation_counts = {}
        self.physical_conflicts = {}
        self.physical_conflict_rejections = {}
        self.physical_conflict_recovery_holds = {}
        self._next_physical_conflict_token = 1
        self._evidence_capture_paths = {}
        # Evidence for tracks attached by location to an existing master is
        # quarantined here. It is committed to the permanent gallery/folder
        # only after appearance or the stable-location fallback confirms it.
        self.pending_member_evidence = {}

        self._lock = threading.RLock()
        self._persistence_lock = threading.Lock()
        self._persistence_condition = threading.Condition()
        self._pending_persistence = {}
        self._persistence_active = False
        self._persistence_stopping = False
        self._task_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._demographics_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        # The in-process thread only transfers crops to a lightweight helper
        # process. PNG hashing, encoding, and disk I/O happen outside this
        # process so they cannot hold the GIL or the identity lock.
        self._evidence_queue = queue.Queue()
        self._stop_token = object()
        self._worker = None
        self._demographics_worker = None
        self._evidence_worker = None
        self._evidence_process = None
        self._persistence_worker = None
        # Programmatic sessions may inject worker-preloaded models. CLI users
        # retain the original lazy-loading behaviour when these are omitted.
        self._role_classifier = role_classifier
        self._demographics_engine = demographics_engine
        self._closed = False

        self.load_database()
        if self.persistence_store is not None:
            self._persistence_worker = threading.Thread(
                target=self._persistence_worker_loop,
                name="reid-persistence",
                daemon=True,
            )
            self._persistence_worker.start()
        if start_worker:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="reid-analyst",
                daemon=True,
            )
            self._worker.start()
        if self.evidence_dir is not None:
            worker_script = Path(__file__).resolve().with_name("reid_evidence_writer.py")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._evidence_process = subprocess.Popen(
                [sys.executable, "-u", str(worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                creationflags=creationflags,
                bufsize=0,
            )
            self._evidence_worker = threading.Thread(
                target=self._evidence_worker_loop,
                name="reid-evidence-sender",
                daemon=True,
            )
            self._evidence_worker.start()

    @staticmethod
    def _empty_gallery():
        return {slot_name: None for slot_name in REID_GALLERY_SLOTS}

    @staticmethod
    def _track_key(track_id, camera_id=None):
        local_id = int(track_id)
        return local_id if camera_id is None else (str(camera_id), local_id)

    @staticmethod
    def _public_identity_id(identity_id):
        return identity_id if identity_id is not None and identity_id > 0 else None

    @staticmethod
    def _temporary_group_token(identity_id):
        if identity_id is None or identity_id >= 0:
            return None
        return f"tmp_{abs(int(identity_id))}"

    @staticmethod
    def _camera_from_key(track_key):
        return track_key[0] if isinstance(track_key, tuple) else None

    @staticmethod
    def _normalized_box(box):
        if box is None:
            return None
        try:
            values = np.asarray(box, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 4 or not np.all(np.isfinite(values[:4])):
            return None
        x1, y1, x2, y2 = map(float, values[:4])
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _shadow_overlap_score(self, candidate_box, canonical_box):
        """Return duplicate-likeness when two boxes cover the same person.

        Geometry is deliberately only a nomination gate. A surviving
        replacement must still pass the normal five-crop appearance check
        before it can inherit the canonical track's master ID.
        """

        candidate = self._normalized_box(candidate_box)
        canonical = self._normalized_box(canonical_box)
        if candidate is None or canonical is None:
            return None

        ax1, ay1, ax2, ay2 = candidate
        bx1, by1, bx2, by2 = canonical
        intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height
        if intersection <= 0.0:
            return None

        candidate_area = (ax2 - ax1) * (ay2 - ay1)
        canonical_area = (bx2 - bx1) * (by2 - by1)
        union = candidate_area + canonical_area - intersection
        iou = intersection / union if union > 0.0 else 0.0
        containment = intersection / min(candidate_area, canonical_area)

        candidate_center = np.asarray(((ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5))
        canonical_center = np.asarray(((bx1 + bx2) * 0.5, (by1 + by2) * 0.5))
        canonical_diagonal = float(np.hypot(bx2 - bx1, by2 - by1))
        center_ratio = (
            float(np.linalg.norm(candidate_center - canonical_center)) / canonical_diagonal
            if canonical_diagonal > 1e-6
            else float("inf")
        )
        if center_ratio > self.shadow_center_distance_ratio:
            return None
        if iou < self.shadow_iou_threshold and containment < self.shadow_containment_threshold:
            return None
        return max(iou, containment), iou, containment, center_ratio

    @staticmethod
    def _normalize_feature(feature):
        if feature is None:
            return None
        array = np.asarray(feature, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(array.astype(np.float64)))
        if norm <= 1e-6:
            return None
        return (array.astype(np.float64) / norm).astype(np.float32)

    @staticmethod
    def _crop_digest(crop):
        if crop is None or crop.size == 0:
            return None
        digest = hashlib.sha256()
        digest.update(str(crop.shape).encode("ascii"))
        digest.update(str(crop.dtype).encode("ascii"))
        digest.update(crop.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _quality_score(sample):
        return float(sample.get("sharpness", 0.0)) * max(1.0, float(sample.get("area", 0.0)) ** 0.5)

    @staticmethod
    def _demographics_face_pixels(sample):
        """Area in crop pixels of the face MiVOLO would be given, or 0."""
        crop = sample.get("crop")
        face_box = sample.get("face_box")
        if crop is None or getattr(crop, "size", 0) == 0 or face_box is None:
            return 0.0
        height, width = crop.shape[:2]
        pixels = normalized_box_to_pixels(face_box, width, height)
        if pixels is None:
            return 0.0
        x1, y1, x2, y2 = pixels
        return float((x2 - x1) * (y2 - y1))

    @classmethod
    def _demographics_quality(cls, sample):
        """Rank a crop by how much of a face it actually offers MiVOLO.

        Returned as (has_face, score) so a crop showing a face always outranks
        one that does not, however sharp or large the faceless crop is.  Within
        a tier the rule matches the baseline hero's -- sharpness times the root
        of the area -- but measured on the face rather than the whole body,
        because it is the face that decides an age.
        """
        crop = sample.get("crop")
        if crop is None or getattr(crop, "size", 0) == 0:
            return (0, 0.0)
        sharpness = float(sample.get("sharpness") or 0.0)
        face_pixels = cls._demographics_face_pixels(sample)
        if face_pixels > 0.0:
            return (1, sharpness * max(1.0, face_pixels ** 0.5))
        height, width = crop.shape[:2]
        return (0, sharpness * max(1.0, float(height * width) ** 0.5))

    @classmethod
    def _demographics_candidates(cls, samples, limit=DEFAULT_DEMOGRAPHICS_CROP_COUNT):
        """The best crops for an age estimate, best first, with their framing.

        The intake burst is five consecutive frames of one camera, so its
        crops are near-duplicates of each other and were never ranked for this
        purpose -- only the single baseline hero was.  Ranking them here is
        what stops the four the gallery rejected from carrying equal weight in
        someone's recorded age.
        """
        ranked = sorted(
            (sample for sample in samples if sample.get("crop") is not None),
            key=cls._demographics_quality,
            reverse=True,
        )
        candidates = []
        for sample in ranked[: max(1, int(limit))]:
            candidates.append(
                {
                    "crop": sample["crop"].copy(),
                    "face_box": sample.get("face_box"),
                    "body_bounds": sample.get("body_bounds"),
                    "occluder_boxes": tuple(sample.get("occluder_boxes") or ()),
                    "sharpness": float(sample.get("sharpness") or 0.0),
                    "camera_id": sample.get("camera_id"),
                    "frame_index": sample.get("frame_index"),
                }
            )
        return candidates

    @classmethod
    def _merge_demographics_pool(cls, existing, samples, limit=DEFAULT_DEMOGRAPHICS_CROP_COUNT):
        """Keep the best crops seen for one person, best first.

        The pool is retained rather than consumed, so a later re-estimate votes
        over the good crops from before as well as the new one.  A single
        excellent look at a face should improve the answer, not replace a
        five-crop consensus with a one-crop guess.
        """
        pool = list(existing or ())
        pool.extend(cls._demographics_candidates(samples, limit=limit))
        pool.sort(key=cls._demographics_quality, reverse=True)
        return pool[: max(1, int(limit))]

    @classmethod
    def _demographics_pool_quality(cls, candidates):
        """How good a look at the face a set of crops represents, as one number.

        Only face-bearing crops count.  A set with no face at all scores zero,
        which is what makes the first crop that does show a face able to
        trigger a re-estimate however sharp the faceless ones were.
        """
        best = 0.0
        for candidate in candidates or ():
            has_face, score = cls._demographics_quality(candidate)
            if has_face:
                best = max(best, float(score))
        return best

    @classmethod
    def _sample_debug_summary(cls, sample):
        crop = sample.get("crop")
        return {
            "frame_index": sample.get("frame_index"),
            "camera_id": sample.get("camera_id"),
            "observed_at": sample.get("observed_at"),
            "crop_shape": None if crop is None else tuple(int(value) for value in crop.shape),
            "area": sample.get("area"),
            "sharpness": sample.get("sharpness"),
            "detection_confidence": sample.get("detection_confidence"),
            "detection_box": sample.get("detection_box"),
            "orientation": sample.get("orientation"),
            "map_point": sample.get("map_point"),
            "body_complete": sample.get("body_complete"),
            "body_details": sample.get("body_details"),
            "baseline_quality_score": cls._quality_score(sample),
        }

    def _new_record(self, role="evacuee", role_confidence=0.0, identity_state="confirmed"):
        demographics_status = "Pending" if self.enable_demographics else "Disabled"
        return {
            "identity_state": str(identity_state),
            "location_managed": identity_state in ("provisional", "challenged"),
            "confirmation_reason": None,
            "role": role,
            "role_confidence": float(role_confidence),
            "role_classified": identity_state == "confirmed",
            "age": demographics_status if role == "evacuee" else "N/A",
            "gender": demographics_status if role == "evacuee" else "N/A",
            "gallery": self._empty_gallery(),
            "camera_views": {},
            "camera_baselines": {},
            "member_track_keys": set(),
            "pending_member_keys": set(),
            "challenged_member_keys": set(),
            # Members whose appearance check already argued against this master.
            # They may still be confirmed by a later positive ReID result, but
            # never by the stable-location fallback alone.
            "appearance_rejected_member_keys": set(),
            "pending_member_location_streaks": {},
            "global_reid_checked_track_keys": set(),
            "location_match_frames": 0,
            "reid_comparisons": {},
            "hits": 0,
            "last_seen_monotonic": time.monotonic(),
        }

    def load_database(self):
        if self.persistence_store is not None:
            try:
                payload = self.persistence_store.load_payload()
            except Exception as exc:
                print(f"Unable to load ReID identities from backend: {exc}. Starting with an empty gallery.")
                return
            evidence_enabled = False
            source_label = "FastAPI/SQLite backend"
        else:
            if self.db_path is None or not self.db_path.exists():
                return
            try:
                with self.db_path.open("rb") as handle:
                    payload = pickle.load(handle)
            except Exception as exc:
                raise RuntimeError(f"Unable to load ReID database {self.db_path}: {exc}") from exc
            evidence_enabled = bool(payload.get("evidence_enabled", False)) if isinstance(payload, dict) else False
            source_label = str(self.db_path)

        if not isinstance(payload, dict) or payload.get("schema_version") != self.SCHEMA_VERSION:
            raise RuntimeError(
                f"ReID database {self.db_path} is not the fresh five-slot schema. "
                "Delete or move the old database before starting this version."
            )
        loaded_identities = payload.get("identities")
        if not isinstance(loaded_identities, dict):
            raise RuntimeError(f"ReID database {self.db_path} has no identities dictionary.")

        identities = {}
        known_identity_ids = set()
        for raw_identity_id, raw_record in loaded_identities.items():
            try:
                identity_id = int(raw_identity_id)
            except (TypeError, ValueError):
                if self.persistence_store is not None:
                    print(f"Skipping backend ReID identity with invalid ID: {raw_identity_id!r}")
                    continue
                raise RuntimeError(f"Invalid identity ID {raw_identity_id!r} in {self.db_path}.")
            known_identity_ids.add(identity_id)
            if not isinstance(raw_record, dict):
                continue
            gallery = raw_record.get("gallery")
            if not isinstance(gallery, dict) or set(gallery) != set(REID_GALLERY_SLOTS):
                raise RuntimeError(
                    f"ReID database {self.db_path} contains an invalid five-slot gallery."
                )
            if gallery.get("baseline") is None:
                if self.persistence_store is not None:
                    print(
                        f"Skipping incomplete backend ReID identity {identity_id}: "
                        "no baseline gallery slot."
                    )
                    continue
                raise RuntimeError(f"Identity {raw_identity_id} has no baseline gallery slot.")
            normalized_gallery = self._empty_gallery()
            baseline_feature_space = None
            for slot_name in REID_GALLERY_SLOTS:
                slot = gallery.get(slot_name)
                if slot is None:
                    continue
                if not isinstance(slot, dict):
                    raise RuntimeError(f"Invalid {slot_name} slot for identity {raw_identity_id}.")
                feature = self._normalize_feature(slot.get("feature"))
                if feature is None:
                    raise RuntimeError(f"Invalid feature in {slot_name} slot for identity {raw_identity_id}.")
                feature_space_id = slot.get("feature_space_id")
                if not isinstance(feature_space_id, str) or not feature_space_id:
                    raise RuntimeError(
                        f"Missing feature-space ID in {slot_name} slot for identity {raw_identity_id}."
                    )
                if int(slot.get("feature_dimension", -1)) != int(feature.size):
                    raise RuntimeError(
                        f"Feature dimension mismatch in {slot_name} slot for identity {raw_identity_id}."
                    )
                if baseline_feature_space is None:
                    baseline_feature_space = feature_space_id
                elif feature_space_id != baseline_feature_space:
                    raise RuntimeError(f"Mixed feature spaces for identity {raw_identity_id}.")
                evidence_expected = bool(slot.get("evidence_expected", evidence_enabled))
                if evidence_enabled and evidence_expected:
                    image_path = slot.get("image_path")
                    saved_crop = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path else None
                    if saved_crop is None or self._crop_digest(saved_crop) != slot.get("digest"):
                        raise RuntimeError(
                            f"Missing or corrupt evidence for {slot_name} slot of identity {raw_identity_id}."
                        )
                normalized_slot = dict(slot)
                normalized_slot["feature"] = feature
                normalized_slot["evidence_expected"] = evidence_expected
                normalized_gallery[slot_name] = normalized_slot

            record = dict(raw_record)
            record["gallery"] = normalized_gallery
            record["identity_state"] = "confirmed"
            record["role_classified"] = True
            record.setdefault("confirmation_reason", "loaded_gallery")
            record.setdefault("camera_views", {})
            record.setdefault("camera_baselines", {})
            record["member_track_keys"] = set()
            record["pending_member_keys"] = set()
            record["challenged_member_keys"] = set()
            record["appearance_rejected_member_keys"] = set()
            record["pending_member_location_streaks"] = {}
            record["global_reid_checked_track_keys"] = set()
            record.setdefault("location_managed", False)
            record.setdefault("location_match_frames", 0)
            record.setdefault("reid_comparisons", {})
            identities[identity_id] = record

        with self._lock:
            self.identities = identities
            self.next_identity_id = max(
                set(self.identities).union(known_identity_ids),
                default=0,
            ) + 1
        print(f"Loaded {len(identities)} five-slot ReID identities from {source_label}")

    def _persistence_worker_loop(self):
        while True:
            with self._persistence_condition:
                while not self._pending_persistence and not self._persistence_stopping:
                    self._persistence_condition.wait()
                if self._persistence_stopping and not self._pending_persistence:
                    return
                identity_id = next(iter(self._pending_persistence))
                snapshot = self._pending_persistence.pop(identity_id)
                self._persistence_active = True

            try:
                self.persistence_store.save_identity(identity_id, snapshot)
            except Exception as exc:
                print(f"Unable to save ReID identity {identity_id} to backend: {exc}")
            finally:
                with self._persistence_condition:
                    self._persistence_active = False
                    self._persistence_condition.notify_all()

    def _persistence_is_idle(self):
        if self.persistence_store is None:
            return True
        with self._persistence_condition:
            return not self._pending_persistence and not self._persistence_active

    def _wait_for_persistence_idle(self, timeout):
        if self.persistence_store is None:
            return True
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._persistence_condition:
            while self._pending_persistence or self._persistence_active:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._persistence_condition.wait(timeout=remaining)
            return True

    # Crops kept on a record for the demographics worker are working state,
    # not identity data: they are megabytes each, and a record is deep-copied
    # for every backend save and pickled in full for every local one.  Slots
    # store only features and file paths for exactly this reason.
    TRANSIENT_CROP_KEYS = ("pending_demographics_crops", "demographics_crop_pool")

    @classmethod
    def _without_transient_crops(cls, record):
        if record is None:
            return None
        if not any(key in record for key in cls.TRANSIENT_CROP_KEYS):
            return record
        return {
            key: value
            for key, value in record.items()
            if key not in cls.TRANSIENT_CROP_KEYS
        }

    def save_database(self, identity_id=None):
        if self.persistence_store is not None:
            if identity_id is None:
                return
            identity_id = int(identity_id)
            with self._lock:
                record = self.identities.get(identity_id)
                snapshot = (
                    copy.deepcopy(self._without_transient_crops(record))
                    if record is not None
                    else None
                )
            if snapshot is None:
                return
            if snapshot.get("gallery", {}).get("baseline") is None:
                return
            with self._persistence_condition:
                if self._persistence_stopping:
                    return
                # Keep only the newest unsent snapshot for each identity. This
                # prevents bursts of angle/evidence updates from creating an
                # HTTP backlog while preserving the final state.
                self._pending_persistence[identity_id] = snapshot
                self._persistence_condition.notify()
            return
        if self.db_path is None:
            return
        with self._persistence_lock:
            with self._lock:
                payload = {
                    "schema_version": self.SCHEMA_VERSION,
                    "evidence_enabled": self.evidence_dir is not None,
                    # Provisional records intentionally stay in memory.  They
                    # do not yet have a valid baseline and must not enter the
                    # strict persisted five-slot schema.
                    "identities": {
                        saved_id: self._without_transient_crops(record)
                        for saved_id, record in self.identities.items()
                        if record.get("gallery", {}).get("baseline") is not None
                    },
                }
                serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.db_path.with_name(f"{self.db_path.name}.tmp")
            try:
                with temporary_path.open("wb") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self.db_path)
            except Exception as exc:
                print(f"Unable to save ReID database {self.db_path}: {exc}")
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _evidence_output_path(self, identity_id, slot_name, frame_index, camera_id):
        identity_id = int(identity_id)
        folder_name = (
            f"Temporary_{abs(identity_id):04d}"
            if identity_id < 0
            else f"Master_{identity_id:04d}"
        )
        master_dir = self.evidence_dir / folder_name
        camera_label = "camera" if camera_id is None else str(camera_id).replace("/", "_").replace("\\", "_")
        return master_dir / f"Slot_{slot_name}_{camera_label}_frame_{int(frame_index)}.png"

    def _make_slot(
        self,
        identity_id,
        slot_name,
        feature,
        sample,
        feature_source,
        feature_space_id,
        track_key=None,
    ):
        crop = sample["crop"]
        normalized_feature = self._normalize_feature(feature)
        if normalized_feature is None:
            raise RuntimeError(f"Refusing to store an invalid {slot_name} feature for ID {identity_id}.")
        camera_id = sample.get("camera_id")
        evidence_expected = bool(
            self.evidence_dir is not None
            and (
                self.evidence_camera_ids is None
                or str(camera_id) in self.evidence_camera_ids
            )
        )
        digest = None
        evidence_task = None
        image_path = None
        if evidence_expected:
            capture_key = (int(identity_id), str(camera_id), int(sample["frame_index"]))
            image_path = self._evidence_capture_paths.get(capture_key)
            if image_path is None:
                output_path = self._evidence_output_path(
                    identity_id,
                    slot_name,
                    sample["frame_index"],
                    camera_id,
                )
                image_path = str(output_path)
                self._evidence_capture_paths[capture_key] = image_path
                evidence_task = {
                    "identity_id": int(identity_id),
                    "slot_name": str(slot_name),
                    # The intake/semantic task already owns this crop. The
                    # evidence path does not make another full-resolution copy.
                    "crop": crop,
                    "output_path": image_path,
                }
        slot = {
            "feature": normalized_feature,
            "feature_source": feature_source,
            "feature_space_id": feature_space_id,
            "feature_dimension": int(normalized_feature.size),
            "image_path": image_path,
            "digest": digest,
            "evidence_expected": evidence_expected,
            "captured_frame": int(sample["frame_index"]),
            "captured_at": float(sample.get("observed_at", time.monotonic())),
            "camera_id": camera_id,
            "sharpness": float(sample.get("sharpness", 0.0)),
            "detection_confidence": sample.get("detection_confidence"),
            # Who supplied this crop.  When the audit later rules that track
            # was on the wrong person, this is the only way to find what it
            # left behind and take it back out.
            "contributed_by_track_key": None if track_key is None else tuple(track_key),
        }
        return slot, evidence_task

    def _queue_evidence_save(self, evidence_task):
        if evidence_task is not None:
            self._evidence_queue.put_nowait(evidence_task)

    def _send_evidence_task(self, evidence_task):
        process = self._evidence_process
        if (
            process is None
            or process.poll() is not None
            or process.stdin is None
            or process.stdout is None
        ):
            return {"ok": False, "error": "evidence writer process is not running"}
        pickle.dump(evidence_task, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        process.stdin.flush()
        result = pickle.load(process.stdout)
        if not isinstance(result, dict):
            return {"ok": False, "error": "invalid response from evidence writer process"}
        return result

    @staticmethod
    def _slots_with_path(record, image_path):
        for slot in record.get("gallery", {}).values():
            if isinstance(slot, dict) and slot.get("image_path") == image_path:
                yield slot
        for slot in record.get("camera_baselines", {}).values():
            if isinstance(slot, dict) and slot.get("image_path") == image_path:
                yield slot
        for camera_gallery in record.get("camera_views", {}).values():
            for slot in camera_gallery.values():
                if isinstance(slot, dict) and slot.get("image_path") == image_path:
                    yield slot

    def _complete_evidence_save_locked(self, evidence_task, digest):
        record = self.identities.get(evidence_task["identity_id"])
        if record is None:
            return
        for slot in self._slots_with_path(record, evidence_task["output_path"]):
            slot["digest"] = digest

    def _rollback_failed_evidence_locked(self, evidence_task):
        identity_id = evidence_task["identity_id"]
        failed_path = evidence_task["output_path"]
        record = self.identities.get(identity_id)
        if record is None:
            return

        gallery = record.get("gallery", {})
        baseline = gallery.get("baseline")
        if isinstance(baseline, dict) and baseline.get("image_path") == failed_path:
            self.identities.pop(identity_id, None)
            for track_key, bound_identity_id in list(self.track_to_identity.items()):
                if bound_identity_id == identity_id:
                    self.track_to_identity.pop(track_key, None)
                    self.track_binding_metadata.pop(track_key, None)
                    self.track_results.pop(track_key, None)
            return

        for slot_name, slot in list(gallery.items()):
            if isinstance(slot, dict) and slot.get("image_path") == failed_path:
                gallery[slot_name] = None
        for camera_id, slot in list(record.get("camera_baselines", {}).items()):
            if isinstance(slot, dict) and slot.get("image_path") == failed_path:
                record["camera_baselines"].pop(camera_id, None)
        for camera_gallery in record.get("camera_views", {}).values():
            for slot_name, slot in list(camera_gallery.items()):
                if isinstance(slot, dict) and slot.get("image_path") == failed_path:
                    camera_gallery[slot_name] = None

    def _evidence_worker_loop(self):
        while True:
            evidence_task = self._evidence_queue.get()
            try:
                if evidence_task is self._stop_token:
                    process = self._evidence_process
                    if process is not None and process.poll() is None and process.stdin is not None:
                        try:
                            pickle.dump(None, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
                            process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass
                    return
                try:
                    result = self._send_evidence_task(evidence_task)
                except (BrokenPipeError, EOFError, OSError, pickle.PickleError) as exc:
                    result = {"ok": False, "error": str(exc)}
                if result.get("ok"):
                    with self._lock:
                        self._complete_evidence_save_locked(
                            evidence_task,
                            result.get("digest"),
                        )
                    self.save_database(evidence_task["identity_id"])
                else:
                    print(
                        f"Unable to save {evidence_task['slot_name']} ReID evidence for "
                        f"ID {evidence_task['identity_id']}: {result.get('error', 'unknown error')}"
                    )
                    with self._lock:
                        self._rollback_failed_evidence_locked(evidence_task)
                    identity_event(
                        "reid_evidence_save_failed",
                        master_id=evidence_task.get("identity_id"),
                        slot_name=evidence_task.get("slot_name"),
                        output_path=evidence_task.get("output_path"),
                    )
            finally:
                self._evidence_queue.task_done()

    def _feature_space_id(self, feature_source, feature):
        dimension = int(np.asarray(feature).size)
        if feature_source == "color_histogram":
            contract = f"color-histogram-v1-resize64x128-hsv-bins24x16-regions-weighted-dim{dimension}"
            return "fs1:" + hashlib.sha256(contract.encode("utf-8")).hexdigest()
        provider = getattr(self.reid_extractor, "feature_space_id", None)
        if callable(provider):
            return str(provider(dimension))
        explicit = getattr(self.reid_extractor, "feature_space_id", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        extractor_type = type(self.reid_extractor)
        contract = (
            f"transreid-adapter-v1:{extractor_type.__module__}."
            f"{extractor_type.__qualname__}:dim{dimension}"
        )
        return "fs1:" + hashlib.sha256(contract.encode("utf-8")).hexdigest()

    def _extract_aligned_features(self, crops):
        if not crops:
            return [], "none", None

        features = None
        source = "transreid"
        if self.reid_extractor is not None and hasattr(self.reid_extractor, "extract_many_aligned"):
            features = self.reid_extractor.extract_many_aligned(crops)
        elif self.reid_extractor is not None and hasattr(self.reid_extractor, "extract_many"):
            extracted = self.reid_extractor.extract_many(crops)
            if len(extracted) == len(crops):
                features = extracted

        extractor_available = self.reid_extractor is not None
        availability_check = getattr(self.reid_extractor, "is_available", None)
        if callable(availability_check):
            extractor_available = bool(availability_check())
        if features is None or not any(self._normalize_feature(feature) is not None for feature in features):
            if extractor_available:
                raise RuntimeError("TransReID extraction failed; refusing to create a fallback identity.")
            source = "color_histogram"
            features = [compute_color_reid_feature(crop) for crop in crops]
        normalized = [self._normalize_feature(feature) for feature in features]
        valid = [feature for feature in normalized if feature is not None]
        dimensions = {int(feature.size) for feature in valid}
        if len(dimensions) > 1:
            raise RuntimeError("Extractor returned inconsistent feature dimensions.")
        feature_space_id = self._feature_space_id(source, valid[0]) if valid else None
        return normalized, source, feature_space_id

    def _clear_physical_violation_counts_locked(self, identity_id):
        for violation_key in list(self.physical_violation_counts):
            if violation_key and violation_key[0] == identity_id:
                self.physical_violation_counts.pop(violation_key, None)

    def _related_physical_conflict_tokens_locked(self, track_key):
        """Return active conflict tokens involving another track in this camera."""

        camera_id = self._camera_from_key(track_key)
        if camera_id is None:
            return set()
        return {
            state["token"]
            for state in self.physical_conflicts.values()
            if track_key not in state.get("candidates", {})
            and any(
                self._camera_from_key(candidate_key) == camera_id
                for candidate_key in state.get("candidates", {})
            )
        }

    def _start_physical_conflict_recovery_hold_locked(
        self,
        track_key,
        source_identity_id,
        source_conflict_token,
        frame_index,
    ):
        """Delay a new master while a connected same-camera swap is repaired."""

        if track_key is None:
            return None
        start_frame = 0 if frame_index is None else int(frame_index)
        state = {
            "source_identity_id": int(source_identity_id),
            "source_conflict_token": source_conflict_token,
            "started_frame": start_frame,
            "grace_until_frame": (
                start_frame + self.physical_conflict_recovery_grace_frames
            ),
            "expires_frame": (
                start_frame + self.physical_conflict_recovery_max_frames
            ),
            "related_conflict_tokens": self._related_physical_conflict_tokens_locked(
                track_key
            ),
        }
        self.physical_conflict_recovery_holds[track_key] = state
        identity_event(
            "physical_conflict_recovery_hold_started",
            track_key=track_key,
            camera_id=self._camera_from_key(track_key),
            rejected_master_id=source_identity_id,
            source_conflict_token=source_conflict_token,
            frame_index=frame_index,
            grace_until_frame=state["grace_until_frame"],
            expires_frame=state["expires_frame"],
            related_conflict_tokens=sorted(state["related_conflict_tokens"]),
            reason="wait_for_connected_same_camera_conflict",
        )
        return state

    def _release_physical_conflict_recovery_hold_locked(
        self,
        track_key,
        reason,
        rearm_deferred_intake=True,
    ):
        state = self.physical_conflict_recovery_holds.pop(track_key, None)
        if state is None:
            return False
        intake_state = self.pending_intake.get(track_key)
        resumed = False
        if (
            rearm_deferred_intake
            and intake_state is not None
            and intake_state.pop("deferred_by_physical_conflict_hold", False)
        ):
            intake_state["submitted"] = False
            intake_state["next_retry_frame"] = int(
                intake_state.get("last_frame") or state["started_frame"]
            ) + 1
            resumed = True
        identity_event(
            "physical_conflict_recovery_hold_released",
            track_key=track_key,
            camera_id=self._camera_from_key(track_key),
            rejected_master_id=state["source_identity_id"],
            source_conflict_token=state["source_conflict_token"],
            related_conflict_tokens=sorted(state["related_conflict_tokens"]),
            resumed_intake=resumed,
            reason=reason,
        )
        return True

    def _refresh_physical_conflict_recovery_hold_locked(self, track_key, frame_index):
        state = self.physical_conflict_recovery_holds.get(track_key)
        if state is None:
            return False
        if self.track_to_identity.get(track_key) is not None:
            self._release_physical_conflict_recovery_hold_locked(
                track_key,
                "track_rebound",
                rearm_deferred_intake=False,
            )
            return False

        current_frame = int(frame_index)
        if current_frame > int(state["expires_frame"]):
            self._release_physical_conflict_recovery_hold_locked(
                track_key,
                "maximum_hold_expired",
            )
            return False

        related_tokens = self._related_physical_conflict_tokens_locked(track_key)
        if related_tokens:
            new_tokens = related_tokens - state["related_conflict_tokens"]
            state["related_conflict_tokens"].update(related_tokens)
            if new_tokens:
                identity_event(
                    "physical_conflict_recovery_hold_linked",
                    track_key=track_key,
                    camera_id=self._camera_from_key(track_key),
                    related_conflict_tokens=sorted(new_tokens),
                    frame_index=frame_index,
                )
            return True
        if current_frame <= int(state["grace_until_frame"]):
            return True

        self._release_physical_conflict_recovery_hold_locked(
            track_key,
            "no_connected_conflict_found",
        )
        return False

    def _release_recovery_holds_for_conflict_locked(self, conflict_token, reason):
        for track_key, state in list(self.physical_conflict_recovery_holds.items()):
            if conflict_token not in state.get("related_conflict_tokens", set()):
                continue
            self._release_physical_conflict_recovery_hold_locked(
                track_key,
                reason,
            )

    def _cancel_physical_conflict_locked(self, identity_id, reason):
        state = self.physical_conflicts.pop(identity_id, None)
        if state is None:
            return False
        challenger_key = state.get("challenger_key")
        if challenger_key is not None:
            intake_state = self.pending_intake.get(challenger_key)
            if (
                intake_state is not None
                and int(intake_state.get("generation", -1))
                == int(state.get("challenger_generation", -2))
            ):
                intake_state.pop("deferred_by_contested_identity_claim", None)
                intake_state["submitted"] = False
                intake_state["next_retry_frame"] = max(
                    int(intake_state.get("next_retry_frame", 0)),
                    int(state.get("started_frame") or 0) + 1,
                )
        self._clear_physical_violation_counts_locked(identity_id)
        self._release_recovery_holds_for_conflict_locked(
            state["token"],
            f"related_conflict_{reason}",
        )
        identity_event(
            "physical_conflict_hold_cancelled",
            master_id=identity_id,
            candidate_track_keys=sorted(state["candidates"], key=repr),
            conflict_token=state["token"],
            challenger_track_key=challenger_key,
            reason=reason,
        )
        return True

    def _start_physical_conflict_locked(
        self,
        identity_id,
        track_key,
        other_observation,
        frame_index,
        distance_cm,
    ):
        """Hold two incompatible claimants until clean ReID evidence chooses one."""

        if identity_id is None or identity_id <= 0 or track_key is None:
            return False
        record = self.identities.get(identity_id)
        baseline = (record or {}).get("gallery", {}).get("baseline")
        if baseline is None:
            return False
        other_key = other_observation.get("track_key")
        if (
            other_key is None
            or other_key == track_key
            or self._camera_from_key(other_key) == self._camera_from_key(track_key)
            or self.track_to_identity.get(track_key) != identity_id
            or self.track_to_identity.get(other_key) != identity_id
        ):
            return False

        existing = self.physical_conflicts.get(identity_id)
        candidate_keys = {track_key, other_key}
        if existing is not None:
            return set(existing["candidates"]) == candidate_keys

        token = self._next_physical_conflict_token
        self._next_physical_conflict_token += 1
        self.physical_conflicts[identity_id] = {
            "token": token,
            "candidates": {key: [] for key in candidate_keys},
            "last_frames": {},
            "submitted": False,
            "attempts": 0,
            "started_frame": None if frame_index is None else int(frame_index),
            # Cameras count frames independently, so only wall clock can tell
            # an audit on one camera how long a contest on another has run.
            "started_monotonic": time.monotonic(),
        }
        candidate_cameras = {
            self._camera_from_key(candidate_key) for candidate_key in candidate_keys
        }
        for held_key, recovery_state in self.physical_conflict_recovery_holds.items():
            if (
                held_key not in candidate_keys
                and self._camera_from_key(held_key) in candidate_cameras
            ):
                recovery_state["related_conflict_tokens"].add(token)
                identity_event(
                    "physical_conflict_recovery_hold_linked",
                    track_key=held_key,
                    camera_id=self._camera_from_key(held_key),
                    related_conflict_tokens=[token],
                    frame_index=frame_index,
                )
        identity_event(
            "physical_conflict_hold_started",
            master_id=identity_id,
            candidate_track_keys=sorted(candidate_keys, key=repr),
            conflict_token=token,
            frame_index=frame_index,
            distance_cm=distance_cm,
            distance_limit_cm=self.cross_camera_fusion_distance_cm,
            required_clean_crops=self.physical_conflict_reid_frames,
            required_distance_margin=self.physical_conflict_reid_margin,
            reason="cross_camera_location_conflict",
        )
        return True

    def _holds_a_confirmed_master_locked(self, track_key):
        """Whether this track already owns a real identity, not a placeholder.

        A challenger that holds a master has nothing to win and must not
        unseat anyone.  A member of a temporary group holds only the group's
        negative placeholder, which is not an identity -- the group has not
        been given a number yet, and may be about to be given one that already
        belongs to somebody.  Treating that placeholder as ownership refused
        every contest raised from the pairing path, which is the path most
        identities are created through: two boxes of a man who already had an
        ID recognised it at 0.147 and 0.135, could not challenge for it, and
        were issued a second one.
        """

        identity_id = self.track_to_identity.get(track_key)
        return identity_id is not None and identity_id > 0

    def _report_owner_blocked_matches_locked(
        self,
        owner_blocked_matches,
        track_key,
        camera_id,
        phase,
        frame_index,
    ):
        """Record how well a newcomer matched the masters it was not shown.

        These scores were computed and then discarded unless they happened to
        start a contest, which left the interesting question -- how close was
        the right answer that nobody looked at -- unanswerable after the fact.
        Duplicate IDs are diagnosed from exactly this number.
        """

        if not owner_blocked_matches:
            return
        identity_event(
            "owner_blocked_masters_scored",
            console=False,
            track_key=track_key,
            camera_id=camera_id,
            phase=str(phase),
            frame_index=frame_index,
            contest_distance=self.strong_match_distance,
            distance_threshold=self.distance_threshold,
            scored=[
                {
                    "master_id": match["identity_id"],
                    "distance": match["distance"],
                    "matched_slot": match.get("matched_slot"),
                    "would_contest": float(match["distance"]) < self.strong_match_distance,
                }
                for match in sorted(
                    owner_blocked_matches,
                    key=lambda item: float(item["distance"]),
                )
            ],
        )

    def _contest_owner_blocked_master_locked(
        self,
        owner_blocked_matches,
        challenger_key,
        camera_id,
        task,
        feature_source,
        feature_space_id,
    ):
        """Challenge the live holder of a master this track matches strongly.

        A master is excluded from the search while another box in the same
        camera holds it, so nothing measured whether the holder is really that
        person.  When the excluded master turns out to be a strong match for
        the newcomer, the holder has to defend it -- otherwise the newcomer is
        handed a duplicate ID for someone who already exists, and the two are
        left indistinguishable to everything downstream.
        """

        already_lost = self.physical_conflict_rejections.get(challenger_key, ())
        for match in sorted(
            owner_blocked_matches or (),
            key=lambda item: float(item["distance"]),
        ):
            identity_id = match["identity_id"]
            # Two people the model cannot tell apart would contest, lose, and
            # contest again forever, leaving the newcomer permanently without
            # an ID.  One defeat settles it: fall through and allocate.
            if identity_id in already_lost:
                continue
            owners = sorted(
                self._visible_same_camera_identity_owners_locked(
                    identity_id,
                    camera_id,
                    excluded_key=challenger_key,
                ),
                key=repr,
            )
            # Exactly one holder, or there is no single incumbent to put on
            # trial and the arbiter has nothing to compare against.
            if len(owners) != 1:
                continue
            contested = dict(match)
            contested["incumbent_track_key"] = owners[0]
            if self._start_contested_identity_claim_locked(
                contested,
                task,
                feature_source,
                feature_space_id,
            ):
                identity_event(
                    "owner_blocked_master_contested",
                    master_id=identity_id,
                    challenger_track_key=challenger_key,
                    incumbent_track_key=owners[0],
                    camera_id=camera_id,
                    challenger_distance=match["distance"],
                    challenger_matched_slot=match.get("matched_slot"),
                    strong_match_distance=self.strong_match_distance,
                    frame_index=task.get("frame_index"),
                    reason="excluded_master_matched_the_newcomer",
                )
                return True
        return False

    def _location_hold_is_stalled_locked(self, state):
        """Whether a location hold has lost the ability to answer itself.

        A hold decides by comparing clean crops from *both* of its candidates.
        When one side cannot supply any -- a camera too soft for the blur gate,
        a body the detector keeps clipping, a track that has stopped yielding
        crops -- no verdict is reachable, and the hold sits on the master until
        one of its candidates disappears.  That costs nothing while nobody else
        wants the master, and everything the moment its real owner is standing
        in front of another camera being refused.

        An appearance contest is never stalled in this sense: it is already the
        better question, and restarting it would throw away the evidence it has
        collected.  Only geometry-born holds stand aside.
        """

        if state.get("challenger_key") is not None:
            return False
        age = time.monotonic() - float(state.get("started_monotonic") or 0.0)
        if age < self.physical_conflict_stall_seconds:
            return False
        return any(
            len(samples) < self.physical_conflict_reid_frames
            for samples in state.get("candidates", {}).values()
        )

    def _start_contested_identity_claim_locked(
        self,
        rejected_match,
        task,
        feature_source,
        feature_space_id,
    ):
        """Hold a strong newcomer while its conflicting owner is rechecked.

        Normal intake must not hand a master to two physically incompatible
        people.  During a tracker swap, however, the incompatible observation
        can itself belong to the wrong person.  In that narrow case a strong
        appearance match earns an owner-versus-challenger arbitration instead
        of immediately allocating a duplicate master.
        """

        identity_id = rejected_match.get("identity_id")
        challenger_key = task.get("track_key")
        rejection = rejected_match.get("physical_rejection") or {}
        incumbent_key = rejection.get("other_track_key") or rejected_match.get(
            "incumbent_track_key"
        )
        challenger_distance = rejected_match.get("distance")
        if (
            identity_id not in self.identities
            or challenger_key is None
            or incumbent_key is None
            or incumbent_key == challenger_key
            or challenger_distance is None
            or float(challenger_distance) >= self.strong_match_distance
            or feature_source != "transreid"
            or self._holds_a_confirmed_master_locked(challenger_key)
            or self.track_to_identity.get(incumbent_key) != identity_id
        ):
            return False

        # Two boxes in one camera are usually one person detected twice, and a
        # duplicate must never contest the original -- the shadow machinery
        # owns that case.  But when the boxes barely overlap they are two
        # people, and refusing the contest is what let a man be denied his own
        # identity because someone else was wearing it in the same view.
        if self._camera_from_key(challenger_key) == self._camera_from_key(incumbent_key):
            challenger_box = self.track_boxes.get(challenger_key)
            incumbent_box = self.track_boxes.get(incumbent_key)
            if (
                challenger_box is None
                or incumbent_box is None
                or self._shadow_overlap_score(challenger_box, incumbent_box) is not None
            ):
                return False

        incumbent_camera = self._camera_from_key(incumbent_key)
        visible_incumbents = self.visible_track_keys_by_camera.get(
            incumbent_camera
        )
        if visible_incumbents is not None and incumbent_key not in visible_incumbents:
            return False
        # One dispute per master at a time, so a contest cannot be restarted
        # out from under itself.  A location hold that has stopped being able
        # to answer its own question is the exception: it will never conclude,
        # and while it stands the master cannot be contested at all.  That is
        # how a man matching his own ID at 0.047 was issued a duplicate --
        # queued behind a hold whose other candidate had been refused for blur
        # for four seconds and would be refused for four more.
        active_conflict = self.physical_conflicts.get(identity_id)
        if active_conflict is not None and not self._location_hold_is_stalled_locked(
            active_conflict
        ):
            return False

        samples = task.get("samples") or ()
        if len(samples) < self.physical_conflict_reid_frames:
            return False
        challenger_samples = sorted(
            samples,
            key=self._quality_score,
            reverse=True,
        )[: self.physical_conflict_reid_frames]
        challenger_samples = [
            {**sample, "crop": sample["crop"].copy()}
            for sample in challenger_samples
        ]
        challenger_samples.sort(key=lambda sample: int(sample.get("frame_index", 0)))

        # Every check has passed, so the stalled hold is really being replaced
        # rather than discarded on the way to another refusal.  Cancelling it
        # properly matters: it clears the violation streaks and releases the
        # recovery holds that were linked to its token.
        if active_conflict is not None:
            self._cancel_physical_conflict_locked(
                identity_id,
                "superseded_by_contested_identity_claim",
            )

        token = self._next_physical_conflict_token
        self._next_physical_conflict_token += 1
        self.physical_conflicts[identity_id] = {
            "token": token,
            "candidates": {
                challenger_key: challenger_samples,
                incumbent_key: [],
            },
            "last_frames": {
                challenger_key: max(
                    int(sample.get("frame_index", 0))
                    for sample in challenger_samples
                )
            },
            "submitted": False,
            "attempts": 0,
            "started_frame": int(task.get("frame_index", 0)),
            "started_monotonic": time.monotonic(),
            "challenger_key": challenger_key,
            "challenger_generation": int(task.get("generation", -1)),
            "challenger_seed_samples": challenger_samples,
        }
        intake_state = self.pending_intake.get(challenger_key)
        if intake_state is not None:
            intake_state["deferred_by_contested_identity_claim"] = True
        identity_event(
            "contested_identity_claim_started",
            master_id=identity_id,
            challenger_track_key=challenger_key,
            incumbent_track_key=incumbent_key,
            conflict_token=token,
            frame_index=task.get("frame_index"),
            challenger_distance=challenger_distance,
            challenger_matched_slot=rejected_match.get("matched_slot"),
            strong_match_distance=self.strong_match_distance,
            physical_distance_cm=rejection.get("distance_cm"),
            physical_distance_limit_cm=rejection.get("distance_limit_cm"),
            required_clean_incumbent_crops=self.physical_conflict_reid_frames,
            required_distance_margin=self.physical_conflict_reid_margin,
            feature_source=feature_source,
            feature_space_id=feature_space_id,
            reason="strong_reid_match_blocked_by_conflicting_owner_location",
        )
        return True

    def _physical_match_allowed_locked(
        self,
        identity_id,
        camera_id,
        map_point,
        observed_at,
        track_key=None,
        frame_index=None,
        defer_bound_conflict=False,
        established_binding=False,
        rejection_context=None,
    ):
        """Decide whether this camera's position is compatible with the master.

        ``established_binding`` says whether the track already holds this
        identity.  It decides how much patience the disagreement earns: a
        binding that has been working is protected from one distorted frame,
        while a track claiming a master it has never held is refused on the
        spot.  A grace period there would let a genuinely different person
        occupy an occupied identity for several frames on the strength of no
        history at all.
        """
        if camera_id is None or self.cross_camera_fusion_distance_cm is None:
            return True
        if observed_at is None or not np.isfinite(float(observed_at)):
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "physical_match_rejected",
                throttle_key=(identity_id, camera_id, "invalid_observation_time"),
                throttle_seconds=1.0,
                master_id=identity_id,
                camera_id=camera_id,
                reason="invalid_observation_time",
                observed_at=observed_at,
            )
            return False
        normalized_point = None
        if map_point is not None:
            point_array = np.asarray(map_point, dtype=float).reshape(-1)
            if point_array.size != 2 or not np.all(np.isfinite(point_array)):
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "physical_match_rejected",
                    throttle_key=(identity_id, camera_id, "invalid_map_point"),
                    throttle_seconds=1.0,
                    master_id=identity_id,
                    camera_id=camera_id,
                    reason="invalid_map_point",
                    map_point=map_point,
                )
                return False
            normalized_point = point_array
        observations = self.recent_master_observations.get(identity_id, {})
        active_conflict = self.physical_conflicts.get(identity_id)
        compared_other_camera = False
        for other_camera, observation in observations.items():
            if other_camera == str(camera_id):
                continue
            time_skew = abs(float(observed_at) - float(observation["observed_at"]))
            if time_skew > self.cross_camera_max_skew_seconds:
                continue
            other_point = observation.get("map_point")
            if normalized_point is None or other_point is None:
                continue
            compared_other_camera = True
            physical_distance = float(np.linalg.norm(normalized_point - np.asarray(other_point, dtype=float)))
            if physical_distance > self.cross_camera_fusion_distance_cm:
                other_key = observation.get("track_key")
                # A veto requires two real measurements. If either side's ground
                # point came from a clipped or foot-occluded box, the two points
                # may disagree simply because one of them is not where the person
                # is standing -- which is not evidence that they are two people.
                # Measured appearance keeps priority over inferred geometry.
                incoming_evidence = self._position_evidence(track_key)
                other_evidence = str(observation.get("evidence") or "hard")
                if self.position_confidence_gating and (
                    incoming_evidence != "hard" or other_evidence != "hard"
                ):
                    identity_event(
                        "physical_match_soft_evidence_ignored",
                        throttle_key=(identity_id, camera_id, other_camera, "soft"),
                        throttle_seconds=1.0,
                        master_id=identity_id,
                        camera_id=camera_id,
                        other_camera_id=other_camera,
                        track_key=track_key,
                        other_track_key=other_key,
                        distance_cm=physical_distance,
                        distance_limit_cm=self.cross_camera_fusion_distance_cm,
                        time_skew_seconds=time_skew,
                        incoming_evidence=incoming_evidence,
                        incoming_evidence_reason=self.track_position_evidence.get(
                            track_key, ("hard", None)
                        )[1],
                        other_evidence=other_evidence,
                        other_evidence_reason=observation.get("evidence_reason"),
                        reason="soft_position_may_not_veto_appearance",
                    )
                    # Not a violation: leave the streak alone and keep looking.
                    continue
                if (
                    active_conflict is not None
                    and track_key in active_conflict.get("candidates", {})
                    and other_key in active_conflict.get("candidates", {})
                ):
                    identity_event(
                        "physical_conflict_hold_active",
                        throttle_key=(identity_id, track_key, other_key),
                        throttle_seconds=1.0,
                        master_id=identity_id,
                        track_key=track_key,
                        other_track_key=other_key,
                        conflict_token=active_conflict["token"],
                        distance_cm=physical_distance,
                        distance_limit_cm=self.cross_camera_fusion_distance_cm,
                    )
                    return True
                violation_key = (identity_id, str(camera_id), str(other_camera))
                violation_count = int(self.physical_violation_counts.get(violation_key, 0)) + 1
                self.physical_violation_counts[violation_key] = violation_count
                # Required of every established binding, not only the
                # location-managed ones.  The soft-evidence gate above already
                # turns away points a camera knows are unreliable, but it
                # cannot see the ones it does not know about: at a grazing
                # angle a perfectly visible foot still projects to a ground
                # position that swings metres between frames, and that point is
                # graded hard because nothing in the pipeline can tell it is
                # wrong.  Breaking a working binding on one such frame -- as a
                # threshold of 1 did -- put a duplicate person on the map for
                # exactly as long as the wobble lasted.  Consecutive by
                # construction: any in-range frame pops the counter below.
                # Only a marginal overshoot earns a second look.  A point
                # several times past the limit is another person, and treating
                # it as error-bar noise would hand one master to two tracks
                # standing metres apart.
                marginal_overshoot = physical_distance <= (
                    self.cross_camera_fusion_distance_cm
                    * DEFAULT_NEW_MATCH_POSITION_SLACK_RATIO
                )
                violations_required = (
                    DEFAULT_POSITION_SPLIT_FRAMES
                    if established_binding
                    else (
                        DEFAULT_NEW_MATCH_POSITION_SPLIT_FRAMES
                        if marginal_overshoot
                        else 1
                    )
                )
                violation_threshold_reached = bool(
                    violation_count >= violations_required
                )
                conflict_held = bool(
                    violation_threshold_reached
                    and defer_bound_conflict
                    and self._start_physical_conflict_locked(
                        identity_id,
                        track_key,
                        observation,
                        frame_index,
                        physical_distance,
                    )
                )
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    (
                        "physical_match_warning"
                        if not violation_threshold_reached or conflict_held
                        else "physical_match_rejected"
                    ),
                    throttle_key=(identity_id, camera_id, other_camera, "distance"),
                    throttle_seconds=1.0,
                    master_id=identity_id,
                    camera_id=camera_id,
                    other_camera_id=other_camera,
                    reason="distance",
                    map_point=normalized_point,
                    other_map_point=other_point,
                    distance_cm=physical_distance,
                    distance_limit_cm=self.cross_camera_fusion_distance_cm,
                    time_skew_seconds=time_skew,
                    time_skew_limit_seconds=self.cross_camera_max_skew_seconds,
                    consecutive_violations=violation_count,
                    violations_required=violations_required,
                    held_for_appearance_arbitration=conflict_held,
                )
                if violation_threshold_reached:
                    if conflict_held:
                        return True
                    if rejection_context is not None:
                        rejection_context.update(
                            {
                                "reason": "distance",
                                "identity_id": identity_id,
                                "track_key": track_key,
                                "camera_id": camera_id,
                                "map_point": tuple(map(float, normalized_point)),
                                "other_track_key": other_key,
                                "other_camera_id": other_camera,
                                "other_map_point": tuple(map(float, other_point)),
                                "observed_at": float(observed_at),
                                "other_observed_at": float(
                                    observation["observed_at"]
                                ),
                                "distance_cm": physical_distance,
                                "distance_limit_cm": (
                                    self.cross_camera_fusion_distance_cm
                                ),
                            }
                        )
                    return False
                continue
            self.physical_violation_counts.pop(
                (identity_id, str(camera_id), str(other_camera)),
                None,
            )
        # Recovered geometry only answers the question a location conflict
        # asked.  A contested claim is raised on appearance -- a challenger
        # matching this master far better than the owner wearing it -- and
        # agreeing positions say nothing about that.  Worse, the tracker swap
        # such a claim exists to undo tends to move *every* owner track onto
        # one body, which makes the master perfectly self-consistent and fires
        # this cancel on every frame: one huddle produced 50 consecutive claims
        # at 0.04 distance, each killed ~29ms after it opened, none surviving
        # the three frames needed to collect the incumbent crops that would
        # have judged them.  A challenger-led contest ends by verdict, by a
        # candidate rebinding, or by a candidate leaving the camera -- never
        # because the master stopped being in two places at once.
        if (
            active_conflict is not None
            and active_conflict.get("challenger_key") is None
            and track_key in active_conflict.get("candidates", {})
            and compared_other_camera
        ):
            self._cancel_physical_conflict_locked(identity_id, "locations_recovered")
        return True

    def _nearby_master_context_locked(self, map_point, observed_at, camera_id, limit=3):
        """Which existing masters were standing near this new one, and how near.

        Diagnostics only -- nothing here influences whether the master is
        created.  A duplicate master is nearly always born on top of the person
        it duplicates, so recording the neighbours at birth turns "why does M8
        exist" from an archaeology exercise into one field.
        """
        empty = {
            "nearest_masters": [],
            "nearest_master_distance_cm": None,
            "physically_nearby_master_ids": [],
        }
        if map_point is None or self.cross_camera_fusion_distance_cm is None:
            return empty
        try:
            point = np.asarray(map_point, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return empty
        if point.size != 2 or not np.all(np.isfinite(point)):
            return empty

        neighbours = []
        for other_id, cameras in self.recent_master_observations.items():
            for other_camera, observation in cameras.items():
                other_point = observation.get("map_point")
                if other_point is None:
                    continue
                try:
                    other = np.asarray(other_point, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if other.size != 2 or not np.all(np.isfinite(other)):
                    continue
                skew = None
                if observed_at is not None and observation.get("observed_at") is not None:
                    skew = abs(float(observed_at) - float(observation["observed_at"]))
                neighbours.append(
                    {
                        "master_id": self._public_identity_id(other_id),
                        "camera_id": other_camera,
                        "distance_cm": float(np.linalg.norm(point - other)),
                        "time_skew_seconds": skew,
                        "evidence": observation.get("evidence"),
                        "other_camera": other_camera != str(camera_id),
                    }
                )
        if not neighbours:
            return empty
        neighbours.sort(key=lambda item: item["distance_cm"])
        return {
            "nearest_masters": neighbours[:limit],
            "nearest_master_distance_cm": neighbours[0]["distance_cm"],
            "physically_nearby_master_ids": sorted(
                {
                    item["master_id"]
                    for item in neighbours
                    if item["distance_cm"] <= float(self.cross_camera_fusion_distance_cm)
                    and item["master_id"] is not None
                }
            ),
        }

    def _remember_position_evidence(self, track_key, evidence, reason=None):
        """Record how trustworthy this track's ground point is for this frame."""
        strength = str(evidence or "hard").lower()
        if strength not in {"hard", "soft"}:
            strength = "soft"
        self.track_position_evidence[track_key] = (strength, reason)

    def _position_evidence(self, track_key):
        """Default to hard so callers that never supply evidence keep today's rules."""
        return self.track_position_evidence.get(track_key, ("hard", None))[0]

    def _record_master_observation_locked(self, identity_id, track_key, map_point, observed_at):
        camera_id = self._camera_from_key(track_key)
        if camera_id is None:
            return
        normalized_point = None
        if map_point is not None:
            point_array = np.asarray(map_point, dtype=float).reshape(-1)
            if point_array.size == 2 and np.all(np.isfinite(point_array)):
                normalized_point = (float(point_array[0]), float(point_array[1]))
        evidence, reason = self.track_position_evidence.get(track_key, ("hard", None))
        self.recent_master_observations.setdefault(identity_id, {})[str(camera_id)] = {
            "track_key": track_key,
            "map_point": normalized_point,
            "observed_at": float(observed_at),
            # Stored so the other camera knows, later, whether this point was a
            # real foot measurement or an inference from a clipped box.
            "evidence": evidence,
            "evidence_reason": reason,
        }

    def _collect_physical_conflict_sample_locked(
        self,
        identity_id,
        track_key,
        crop,
        frame_index,
        detection_confidence,
        observed_at,
        map_point,
        body_complete,
    ):
        state = self.physical_conflicts.get(identity_id)
        if state is None or track_key not in state["candidates"] or state["submitted"]:
            return
        if crop is None or crop.size == 0 or body_complete is False:
            return
        if state["last_frames"].get(track_key) == int(frame_index):
            return

        sharpness = image_sharpness(crop)
        # A camera too soft to ever clear the gallery's bar would otherwise
        # deadlock every contest it takes part in, holding the master hostage
        # while its real owner is refused.  Arbitration stores nothing, and its
        # winner must still clear the distance margin, so a blurred feature
        # costs an inconclusive round rather than a wrong revocation.
        blur_relaxed = (
            time.monotonic() - float(state.get("started_monotonic") or 0.0)
            >= self.physical_conflict_blur_timeout_seconds
        )
        if sharpness <= self.blur_threshold and not blur_relaxed:
            identity_event(
                "physical_conflict_crop_rejected",
                console=False,
                throttle_key=(identity_id, track_key, "blur"),
                throttle_seconds=1.0,
                master_id=identity_id,
                track_key=track_key,
                conflict_token=state["token"],
                frame_index=frame_index,
                reason="blur",
                sharpness=sharpness,
                blur_threshold=self.blur_threshold,
                blur_timeout_seconds=self.physical_conflict_blur_timeout_seconds,
            )
            return

        state["last_frames"][track_key] = int(frame_index)
        samples = state["candidates"][track_key]
        samples.append(
            {
                "crop": crop.copy(),
                "frame_index": int(frame_index),
                "camera_id": self._camera_from_key(track_key),
                "observed_at": float(observed_at),
                "sharpness": sharpness,
                "area": int(crop.shape[0] * crop.shape[1]),
                "detection_confidence": (
                    None
                    if detection_confidence is None
                    else float(detection_confidence)
                ),
                "map_point": None if map_point is None else tuple(map(float, map_point)),
                "body_complete": body_complete,
            }
        )
        if len(samples) > self.physical_conflict_reid_frames:
            del samples[:-self.physical_conflict_reid_frames]
        identity_event(
            "physical_conflict_crop_accepted",
            console=False,
            master_id=identity_id,
            track_key=track_key,
            conflict_token=state["token"],
            frame_index=frame_index,
            accepted_sample_count=len(samples),
            required_sample_count=self.physical_conflict_reid_frames,
            sharpness=sharpness,
            # A verdict reached only because the gate stood aside is worth
            # being able to find in a log when a contest is decided wrongly.
            blur_relaxed=blur_relaxed and sharpness <= self.blur_threshold,
        )

        if not all(
            len(candidate_samples) >= self.physical_conflict_reid_frames
            for candidate_samples in state["candidates"].values()
        ):
            return

        task = {
            "type": "physical_conflict",
            "identity_id": identity_id,
            "conflict_token": state["token"],
            "candidate_samples": {
                candidate_key: [
                    {**sample, "crop": sample["crop"].copy()}
                    for sample in candidate_samples[: self.physical_conflict_reid_frames]
                ]
                for candidate_key, candidate_samples in state["candidates"].items()
            },
        }
        state["submitted"] = True
        if not self._queue_task_locked(task):
            current = self.physical_conflicts.get(identity_id)
            if current is not None and current.get("token") == state["token"]:
                current["submitted"] = False

    def _process_physical_conflict_task(self, task):
        candidate_samples = task["candidate_samples"]
        candidate_keys = sorted(candidate_samples, key=repr)
        flat_crops = []
        sample_counts = []
        for key in candidate_keys:
            crops = [sample["crop"] for sample in candidate_samples[key]]
            flat_crops.extend(crops)
            sample_counts.append(len(crops))

        features, feature_source, feature_space_id = self._extract_aligned_features(
            flat_crops
        )
        candidate_queries = {}
        offset = 0
        for key, count in zip(candidate_keys, sample_counts):
            usable = [
                self._normalize_feature(feature)
                for feature in features[offset : offset + count]
            ]
            offset += count
            usable = [feature for feature in usable if feature is not None]
            candidate_queries[key] = (
                None
                if not usable
                else self._normalize_feature(
                    np.mean(np.asarray(usable, dtype=np.float32), axis=0)
                )
            )

        identity_id = task["identity_id"]
        with self._lock:
            state = self.physical_conflicts.get(identity_id)
            if state is None or state.get("token") != task.get("conflict_token"):
                return
            challenger_key = state.get("challenger_key")
            challenger_identity_id = (
                None
                if challenger_key is None
                else self.track_to_identity.get(challenger_key)
            )
            # A challenger holding a real master has nothing to win and must
            # not unseat anyone.  A member of a temporary group holds only that
            # group's negative placeholder, which is not an identity -- and the
            # pairing path raises its contests from exactly such members, so
            # demanding "holds nothing" here made every one of them unwinnable.
            # One session opened five at 0.07-0.10 against the challenger's own
            # ID and cancelled all five as "candidate_binding_changed" before
            # they could be judged, while the group promoted itself into the
            # duplicate the contest existed to prevent.  The start of a contest
            # is already gated on _holds_a_confirmed_master_locked, which draws
            # the line at a positive id; this is the same line.
            challenger_is_current = bool(
                challenger_key is not None
                and (challenger_identity_id is None or challenger_identity_id < 0)
                and self._intake_task_is_current_locked(
                    {
                        "track_key": challenger_key,
                        "generation": state.get("challenger_generation"),
                    }
                )
            )
            bound_candidates_are_current = all(
                key == challenger_key
                or self.track_to_identity.get(key) == identity_id
                for key in candidate_keys
            )
            if (
                (challenger_key is not None and not challenger_is_current)
                or not bound_candidates_are_current
            ):
                self._cancel_physical_conflict_locked(
                    identity_id,
                    "candidate_binding_changed",
                )
                return

            distances = {}
            matched_slots = {}
            for key in candidate_keys:
                query = candidate_queries.get(key)
                if query is None:
                    distances[key] = None
                    matched_slots[key] = None
                    continue
                matched_identity, matched_slot, distance = self._target_identity_match_locked(
                    identity_id,
                    query,
                    feature_space_id,
                    debug_context={
                        "phase": "physical_conflict_arbitration",
                        "track_key": key,
                        "target_master_id": identity_id,
                        "conflict_token": state["token"],
                    },
                    return_rejected=True,
                )
                distances[key] = (
                    distance if matched_identity == identity_id else None
                )
                matched_slots[key] = (
                    matched_slot if matched_identity == identity_id else None
                )

            ranked = sorted(
                candidate_keys,
                key=lambda key: (
                    float("inf") if distances[key] is None else distances[key],
                    repr(key),
                ),
            )
            winner_key, loser_key = ranked[0], ranked[1]
            winner_distance = distances[winner_key]
            loser_distance = distances[loser_key]
            distance_margin = (
                None
                if winner_distance is None or loser_distance is None
                else float(loser_distance) - float(winner_distance)
            )
            if challenger_key is None:
                decisive = bool(
                    winner_distance is not None
                    and winner_distance < self.distance_threshold
                    and (
                        loser_distance is None
                        or distance_margin >= self.physical_conflict_reid_margin
                    )
                )
            else:
                incumbent_key = next(
                    key for key in candidate_keys if key != challenger_key
                )
                challenger_distance = distances[challenger_key]
                incumbent_distance = distances[incumbent_key]
                challenger_wins = bool(
                    challenger_distance is not None
                    and challenger_distance < self.distance_threshold
                    and (
                        incumbent_distance is None
                        or float(incumbent_distance) - float(challenger_distance)
                        >= self.physical_conflict_reid_margin
                    )
                )
                incumbent_upheld = bool(
                    incumbent_distance is not None
                    and incumbent_distance < self.distance_threshold
                    and not challenger_wins
                )
                decisive = challenger_wins or incumbent_upheld
                if decisive:
                    winner_key = (
                        challenger_key if challenger_wins else incumbent_key
                    )
                    loser_key = (
                        incumbent_key if challenger_wins else challenger_key
                    )
                    winner_distance = distances[winner_key]
                    loser_distance = distances[loser_key]
                    distance_margin = (
                        None
                        if winner_distance is None or loser_distance is None
                        else float(loser_distance) - float(winner_distance)
                    )
            if not decisive:
                state["attempts"] = int(state.get("attempts", 0)) + 1
                state["submitted"] = False
                if challenger_key is None:
                    state["candidates"] = {key: [] for key in candidate_keys}
                    state["last_frames"] = {}
                else:
                    challenger_samples = state.get(
                        "challenger_seed_samples",
                        [],
                    )
                    state["candidates"] = {
                        key: (
                            [
                                {**sample, "crop": sample["crop"].copy()}
                                for sample in challenger_samples
                            ]
                            if key == challenger_key
                            else []
                        )
                        for key in candidate_keys
                    }
                    state["last_frames"] = {
                        challenger_key: max(
                            (
                                int(sample.get("frame_index", 0))
                                for sample in challenger_samples
                            ),
                            default=0,
                        )
                    }
                identity_event(
                    "physical_conflict_reid_inconclusive",
                    master_id=identity_id,
                    conflict_token=state["token"],
                    challenger_track_key=challenger_key,
                    candidate_distances={repr(key): distances[key] for key in candidate_keys},
                    distance_threshold=self.distance_threshold,
                    required_distance_margin=self.physical_conflict_reid_margin,
                    observed_distance_margin=distance_margin,
                    attempts=state["attempts"],
                    feature_source=feature_source,
                    feature_space_id=feature_space_id,
                )
                return

            self.physical_conflicts.pop(identity_id, None)
            loser_is_challenger = loser_key == challenger_key
            winner_is_challenger = winner_key == challenger_key
            if not loser_is_challenger:
                loser_camera = str(self._camera_from_key(loser_key))
                observation = self.recent_master_observations.get(identity_id, {}).get(
                    loser_camera
                )
                if observation is not None and observation.get("track_key") == loser_key:
                    self.recent_master_observations[identity_id].pop(loser_camera, None)
                self._clear_local_binding_locked(loser_key)
            # Losing bars this track from the master for the rest of its life,
            # on both the match path and the contest path.  That is the right
            # answer for a challenger that attacked and lost -- two people the
            # model cannot separate would otherwise contest forever and leave
            # the newcomer with no ID at all -- but it is far too strong for a
            # track that never contested anything and merely came second in a
            # geometry arbitration.  Twice now that has barred a man from his
            # own ID at 0.181 and 0.229, both inside the 0.30 match threshold,
            # and both times the next crop measured better than the one he lost
            # with.  A loser that still matches has not been shown to be
            # somebody else, so it keeps the right to be matched again.
            loser_is_no_longer_a_match = (
                loser_distance is None
                or float(loser_distance) >= self.distance_threshold
            )
            if loser_is_challenger or loser_is_no_longer_a_match:
                self.physical_conflict_rejections.setdefault(loser_key, set()).add(
                    identity_id
                )
            loser_frame_index = max(
                (
                    int(sample.get("frame_index", 0))
                    for sample in candidate_samples.get(loser_key, ())
                ),
                default=0,
            )
            if loser_is_challenger:
                intake_state = self.pending_intake.get(loser_key)
                if (
                    intake_state is not None
                    and int(intake_state.get("generation", -1))
                    == int(state.get("challenger_generation", -2))
                ):
                    intake_state.pop(
                        "deferred_by_contested_identity_claim",
                        None,
                    )
                    intake_state["submitted"] = False
                    intake_state["next_retry_frame"] = max(
                        int(intake_state.get("next_retry_frame", 0)),
                        loser_frame_index + 1,
                    )
            else:
                self._start_physical_conflict_recovery_hold_locked(
                    loser_key,
                    identity_id,
                    task["conflict_token"],
                    loser_frame_index,
                )
            self._release_recovery_holds_for_conflict_locked(
                task["conflict_token"],
                "related_conflict_resolved",
            )
            self._clear_physical_violation_counts_locked(identity_id)

            if winner_is_challenger:
                winner_camera = self._camera_from_key(winner_key)
                self._clear_nonvisible_same_camera_owners_locked(
                    identity_id,
                    winner_camera,
                    preserved_key=winner_key,
                )
                # A winner that came from the pairing path is still listed as a
                # member of its temporary group.  Overwriting the mapping alone
                # would leave that group claiming a track which now belongs to
                # a real master, so the placeholder is released the same way
                # every other departure from a group releases it.
                if challenger_identity_id is not None and challenger_identity_id < 0:
                    self._clear_local_binding_locked(winner_key)
                self.track_to_identity[winner_key] = identity_id
                record = self.identities.get(identity_id)
                if record is not None:
                    record.setdefault("member_track_keys", set()).add(winner_key)
                    record["hits"] = int(record.get("hits", 0)) + 1
                    record["last_seen_monotonic"] = time.monotonic()
                latest_winner_sample = max(
                    candidate_samples.get(winner_key, ()),
                    key=lambda sample: int(sample.get("frame_index", 0)),
                )
                self._record_master_observation_locked(
                    identity_id,
                    winner_key,
                    latest_winner_sample.get("map_point"),
                    latest_winner_sample.get("observed_at", time.monotonic()),
                )
                self.pending_intake.pop(winner_key, None)
                self.shadow_tracks.pop(winner_key, None)
                self.physical_conflict_rejections.pop(winner_key, None)

            winner_metadata = self.track_binding_metadata.setdefault(winner_key, {})
            baseline_space = (
                self.identities[identity_id]["gallery"]["baseline"].get(
                    "feature_space_id"
                )
            )
            winner_metadata.update(
                {
                    "query_feature_space_id": feature_space_id,
                    "matched_feature_space_id": baseline_space,
                    "matched_slot": matched_slots[winner_key],
                    "distance": winner_distance,
                    "appearance_confirmed": bool(
                        feature_source == "transreid"
                        and feature_space_id == baseline_space
                    ),
                    "feature_source": feature_source,
                    "conflict_resolution": "appearance",
                }
            )
            self.track_results[winner_key] = {
                "similarity": 1.0 - float(winner_distance),
                "reidentified": True,
                "matched_slot": matched_slots[winner_key],
            }
            identity_event(
                "physical_conflict_reid_resolved",
                master_id=identity_id,
                conflict_token=task["conflict_token"],
                challenger_track_key=challenger_key,
                winner_track_key=winner_key,
                loser_track_key=loser_key,
                winner_distance=winner_distance,
                loser_distance=loser_distance,
                distance_margin=distance_margin,
                required_distance_margin=self.physical_conflict_reid_margin,
                matched_slot=matched_slots[winner_key],
                feature_source=feature_source,
                feature_space_id=feature_space_id,
                reason=(
                    "stronger_master_gallery_match"
                    if challenger_key is None
                    else "stronger_contested_identity_claim"
                    if winner_is_challenger
                    else "incumbent_identity_claim_upheld"
                ),
            )

    def _matching_identity_locked(
        self,
        query_feature,
        query_feature_space_id=None,
        excluded_identity_ids=None,
        camera_id=None,
        map_point=None,
        observed_at=None,
        debug_context=None,
        return_rejected=False,
        track_key=None,
        physically_rejected_matches=None,
        owner_blocked_matches=None,
    ):
        excluded = set(excluded_identity_ids or ())
        best_identity = None
        best_slot = None
        best_distance = float("inf")

        for identity_id, record in self.identities.items():
            # An excluded master is one another live box already holds, not one
            # appearance has ruled out.  Skipping it outright meant a person
            # whose own identity was being worn by someone else was compared
            # only against the people he is not, and allocated a duplicate.
            # Scoring it cannot bind it -- it only makes the match visible so
            # the caller can contest the holder instead of minting a new ID.
            is_owner_blocked = identity_id in excluded
            if is_owner_blocked and owner_blocked_matches is None:
                continue
            physical_rejection = {}
            physical_allowed = bool(
                observed_at is None
                or self._physical_match_allowed_locked(
                    identity_id,
                    camera_id,
                    map_point,
                    observed_at,
                    track_key=track_key,
                    rejection_context=physical_rejection,
                )
            )
            identity_best_slot = None
            identity_best_distance = float("inf")
            gallery = record.get("gallery", {})
            for slot_name in REID_GALLERY_SLOTS:
                slot = gallery.get(slot_name)
                if not slot:
                    continue
                if (
                    query_feature_space_id is not None
                    and slot.get("feature_space_id") != query_feature_space_id
                ):
                    continue
                saved_feature = self._normalize_feature(slot.get("feature"))
                if saved_feature is None or saved_feature.shape != query_feature.shape:
                    continue
                # Re-normalize in float64 for the comparison itself. This
                # keeps an exact 0.35 boundary from slipping below the strict
                # threshold solely because float32 normalization rounded up.
                query64 = np.asarray(query_feature, dtype=np.float64)
                saved64 = np.asarray(saved_feature, dtype=np.float64)
                query64 /= np.linalg.norm(query64)
                saved64 /= np.linalg.norm(saved64)
                distance = 1.0 - float(np.dot(query64, saved64))
                if distance < identity_best_distance:
                    identity_best_distance = distance
                    identity_best_slot = slot_name
                if not is_owner_blocked and physical_allowed and distance < best_distance:
                    best_distance = distance
                    best_identity = identity_id
                    best_slot = slot_name
            if is_owner_blocked:
                # Every blocked master is recorded, not just the ones close
                # enough to contest.  The score of the master that was skipped
                # is how a duplicate ID gets explained afterwards, and keeping
                # only the ones that acted left exactly the near-misses -- the
                # interesting cases -- invisible.  The contest applies its own
                # bar to this list.
                if identity_best_slot is not None:
                    owner_blocked_matches.append(
                        {
                            "identity_id": identity_id,
                            "matched_slot": identity_best_slot,
                            "distance": identity_best_distance,
                        }
                    )
                continue
            if (
                not physical_allowed
                and physically_rejected_matches is not None
                and identity_best_slot is not None
                and identity_best_distance < self.distance_threshold
                and physical_rejection.get("reason") == "distance"
            ):
                physically_rejected_matches.append(
                    {
                        "identity_id": identity_id,
                        "matched_slot": identity_best_slot,
                        "distance": identity_best_distance,
                        "physical_rejection": physical_rejection,
                    }
                )

        accepted = best_identity is not None and best_distance < self.distance_threshold
        if debug_context is not None:
            # TEMP_IDENTITY_DEBUG: log the best rejected candidate before the
            # normal return value intentionally discards its distance.
            identity_event(
                "reid_match_decision",
                **dict(debug_context),
                camera_id=camera_id,
                map_point=map_point,
                observed_at=observed_at,
                excluded_master_ids=sorted(excluded),
                query_feature_space_id=query_feature_space_id,
                best_master_id=best_identity,
                best_slot=best_slot,
                best_distance=None if best_identity is None else best_distance,
                distance_threshold=self.distance_threshold,
                accepted=accepted,
                rejection_reason=(
                    None
                    if accepted
                    else "no_compatible_gallery"
                    if best_identity is None
                    else "distance_threshold"
                ),
            )
        if not accepted:
            if return_rejected:
                return (
                    best_identity,
                    best_slot,
                    None if best_identity is None else best_distance,
                )
            return None, None, None
        return best_identity, best_slot, best_distance

    def find_matching_identity(
        self,
        feature,
        frame_index=None,
        excluded_identity_ids=None,
        feature_space_id=None,
    ):
        del frame_index  # persistent master galleries do not expire by camera frame count
        normalized = self._normalize_feature(feature)
        if normalized is None:
            return None, 0.0, None
        with self._lock:
            identity_id, _slot_name, distance = self._matching_identity_locked(
                normalized,
                query_feature_space_id=feature_space_id,
                excluded_identity_ids=excluded_identity_ids,
            )
        if identity_id is None:
            return None, 0.0, None
        return identity_id, 1.0 - distance, distance

    def _same_camera_active_ids_locked(self, camera_id, excluded_track_key=None):
        if camera_id is None:
            visible_keys = self.visible_track_keys_by_camera.get(None, set())
        else:
            visible_keys = self.visible_track_keys_by_camera.get(str(camera_id), set())
        return {
            self.track_to_identity[key]
            for key in visible_keys
            if key != excluded_track_key and key in self.track_to_identity
        }

    def _target_identity_match_locked(
        self,
        identity_id,
        query_feature,
        feature_space_id,
        debug_context=None,
        return_rejected=False,
    ):
        """Compare with exactly one master, independent of normal exclusions."""

        if identity_id not in self.identities:
            return None, None, None
        excluded = set(self.identities)
        excluded.discard(identity_id)
        return self._matching_identity_locked(
            query_feature,
            query_feature_space_id=feature_space_id,
            excluded_identity_ids=excluded,
            debug_context=debug_context,
            return_rejected=return_rejected,
        )

    def _visible_same_camera_identity_owners_locked(self, identity_id, camera_id, excluded_key=None):
        camera_key = None if camera_id is None else str(camera_id)
        return {
            key
            for key in self.visible_track_keys_by_camera.get(camera_key, set())
            if key != excluded_key and self.track_to_identity.get(key) == identity_id
        }

    def _clear_nonvisible_same_camera_owners_locked(self, identity_id, camera_id, preserved_key=None):
        """Retire stale local aliases without ever stealing from a live box."""

        camera_key = None if camera_id is None else str(camera_id)
        visible_keys = self.visible_track_keys_by_camera.get(camera_key, set())
        stale_owner_keys = [
            owner_key
            for owner_key, owner_identity_id in list(self.track_to_identity.items())
            if (
                owner_key != preserved_key
                and owner_identity_id == identity_id
                and self._camera_from_key(owner_key) == camera_key
                and owner_key not in visible_keys
            )
        ]
        for owner_key in stale_owner_keys:
            self._clear_local_binding_locked(owner_key)
        return stale_owner_keys

    def _reject_second_visible_owner_locked(
        self,
        identity_id,
        track_keys,
        existing_ids=None,
        event_name="binding_declined",
    ):
        """Return True when binding these keys would give one master two live owners.

        The ReID intake and shadow-handoff paths already refuse to take a
        master that a visible same-camera track owns.  The location-driven
        paths were added later to recover people whose opposite-facing camera
        angles defeat appearance matching, and they bypassed that rule.  A
        second live owner is not merely cosmetic: both tracks then read and
        write the same identity-keyed foot and motion memory, so two people
        render as a single map point and every cross-camera distance check for
        that master starts failing.

        Callers must invoke this before mutating any state, so a refusal is a
        clean no-op.
        """

        if identity_id not in self.identities:
            return False
        keys = list(track_keys)
        batch = set(keys)
        if existing_ids is None:
            existing_ids = [self.track_to_identity.get(key) for key in keys]
        for key, existing_id in zip(keys, existing_ids):
            if existing_id == identity_id:
                # Already this master's owner; re-affirming is not a new claim.
                continue
            conflicting = {
                owner_key
                for owner_key in self._visible_same_camera_identity_owners_locked(
                    identity_id,
                    self._camera_from_key(key),
                    excluded_key=key,
                )
                if owner_key not in batch
            }
            if not conflicting:
                continue
            identity_event(
                event_name,
                master_id=self._public_identity_id(identity_id),
                temporary_group_id=self._temporary_group_token(identity_id),
                track_key=key,
                camera_id=self._camera_from_key(key),
                conflicting_track_keys=sorted(conflicting, key=repr),
                reason="visible_same_camera_owner",
            )
            return True
        return False

    def _promote_verified_shadow_locked(self, key, shadow):
        """Atomically move a verified master onto its surviving local track."""

        identity_id = shadow.get("identity_id")
        if not shadow.get("verified") or identity_id not in self.identities:
            return None
        camera_id = self._camera_from_key(key)
        if self._visible_same_camera_identity_owners_locked(
            identity_id,
            camera_id,
            excluded_key=key,
        ):
            return None

        self._clear_nonvisible_same_camera_owners_locked(
            identity_id,
            camera_id,
            preserved_key=key,
        )
        verification = dict(shadow.get("verification", {}))
        self.track_to_identity[key] = identity_id
        self.track_binding_metadata[key] = {
            "query_feature_space_id": verification.get("query_feature_space_id"),
            "matched_feature_space_id": verification.get("matched_feature_space_id"),
            "matched_slot": verification.get("matched_slot"),
            "distance": verification.get("distance"),
            "appearance_confirmed": bool(verification.get("appearance_confirmed", False)),
            "feature_source": verification.get("feature_source"),
            "handoff_from_track_key": shadow.get("canonical_key"),
        }
        self.track_results[key] = {
            "similarity": float(verification.get("similarity", 0.0)),
            "reidentified": True,
            "matched_slot": verification.get("matched_slot"),
        }
        # TEMP_IDENTITY_DEBUG
        identity_event(
            "shadow_handoff_committed",
            track_key=key,
            previous_track_key=shadow.get("canonical_key"),
            master_id=identity_id,
            matched_slot=verification.get("matched_slot"),
            distance=verification.get("distance"),
            appearance_confirmed=bool(verification.get("appearance_confirmed", False)),
        )
        self.pending_intake.pop(key, None)
        self.shadow_tracks.pop(key, None)
        return identity_id

    def _next_track_generation_locked(self, key):
        generation = int(self.track_generations.get(key, 0)) + 1
        self.track_generations[key] = generation
        return generation

    def _clear_local_binding_locked(self, key, clear_last_seen=False):
        if key in self.physical_conflict_recovery_holds:
            self._release_physical_conflict_recovery_hold_locked(
                key,
                "track_binding_cleared",
                rearm_deferred_intake=False,
            )
        for conflict_identity_id, state in list(self.physical_conflicts.items()):
            if key in state.get("candidates", {}):
                self._cancel_physical_conflict_locked(
                    conflict_identity_id,
                    "candidate_binding_cleared",
                )
        self._discard_pending_member_evidence_locked(key, "track_binding_cleared")
        # A rival tally belongs to one binding.  Carrying it across a rebind
        # would let wins earned against a previous master count against a new
        # one, so the audit restarts from scratch.
        self.identity_audit_state.pop(key, None)
        identity_id = self.track_to_identity.pop(key, None)
        camera_id = self._camera_from_key(key)
        camera_observations = self.recent_master_observations.get(identity_id, {})
        observation = camera_observations.get(str(camera_id))
        if observation is not None and observation.get("track_key") == key:
            camera_observations.pop(str(camera_id), None)
        record = self.identities.get(identity_id)
        if record is not None:
            record.setdefault("member_track_keys", set()).discard(key)
            record.setdefault("pending_member_keys", set()).discard(key)
            record.setdefault("challenged_member_keys", set()).discard(key)
            record.setdefault("appearance_rejected_member_keys", set()).discard(key)
            record.setdefault("pending_member_location_streaks", {}).pop(key, None)
        self.track_results.pop(key, None)
        self.track_binding_metadata.pop(key, None)
        self.pending_intake.pop(key, None)
        self.new_master_holds.pop(key, None)
        self.shadow_tracks.pop(key, None)
        self._next_track_generation_locked(key)
        if clear_last_seen:
            self.track_last_seen.pop(key, None)

    def _release_shadow_locked(self, key, reason="unspecified"):
        """Release a candidate into normal intake and invalidate stale work."""

        shadow = self.shadow_tracks.get(key)
        if shadow is not None:
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "shadow_released",
                track_key=key,
                canonical_track_key=shadow.get("canonical_key"),
                target_master_id=shadow.get("identity_id"),
                reason=reason,
                overlap_frames=shadow.get("overlap_frames"),
                separation_frames=shadow.get("separation_frames"),
            )
        self.shadow_tracks.pop(key, None)
        if key in self.pending_intake:
            self.pending_intake.pop(key, None)
            self.track_results.pop(key, None)
            self.track_binding_metadata.pop(key, None)
            self._next_track_generation_locked(key)

    def _intake_task_is_current_locked(self, task, require_visible=True):
        key = task.get("track_key")
        state = self.pending_intake.get(key)
        if (
            state is None
            or not state.get("submitted")
            or int(state.get("generation", -1)) != int(task.get("generation", -2))
        ):
            return False
        if require_visible:
            camera_key = self._camera_from_key(key)
            if camera_key in self.visible_track_keys_by_camera:
                return key in self.visible_track_keys_by_camera[camera_key]
        return True

    def _abandon_missing_track_bindings_locked(self, camera_key, visible_keys, frame_index):
        """Strike off tracks the tracker has already given up on.

        The TTL check above only ever looks at tracks that came back.  A track
        that never returns is in no visible set, so nothing examined it and its
        binding survived for the life of the process -- keeping it on its
        group's member list, where the promotion roll call waited on photos it
        could never deliver and everyone in that group stayed "analysing".
        """

        if self.track_abandon_frames <= 0 or frame_index is None:
            return ()
        abandoned = []
        for key, identity_id in list(self.track_to_identity.items()):
            if key in visible_keys or self._camera_from_key(key) != camera_key:
                continue
            last_seen = self.track_last_seen.get(key)
            if last_seen is None:
                continue
            absent_frames = int(frame_index) - int(last_seen[0])
            if absent_frames <= self.track_abandon_frames:
                continue
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "track_binding_abandoned",
                track_key=key,
                master_id=self._public_identity_id(identity_id),
                temporary_group_id=self._temporary_group_token(identity_id),
                camera_id=camera_key,
                frame_index=frame_index,
                last_seen_frame_index=last_seen[0],
                absent_frames=absent_frames,
                abandon_after_frames=self.track_abandon_frames,
                reason="track_never_returned",
            )
            self._clear_local_binding_locked(key)
            self.track_last_seen.pop(key, None)
            self.pending_intake.pop(key, None)
            abandoned.append(key)
        return tuple(abandoned)

    def observe_tracks(
        self,
        track_ids,
        boxes,
        frame_index=None,
        camera_id=None,
        observed_at=None,
    ):
        """Register one camera frame and nominate newly spawned shadow tracks.

        A candidate is only nominated when it is a newly visible, unbound
        local track whose box strongly overlaps an older, currently visible
        track in the same camera. The older track may already own a master ID
        or still be completing its initial intake. Brief candidates are
        suppressed without ReID; persistent candidates receive one bounded
        five-crop appearance check. If the canonical track disappears, a
        matching candidate inherits its master through an atomic handoff.
        """

        raw_track_ids = list(track_ids if track_ids is not None else ())
        ordered_keys = [self._track_key(track_id, camera_id) for track_id in raw_track_ids]
        keys = set(ordered_keys)
        raw_boxes = list(boxes) if boxes is not None else []
        current_boxes = {}
        for index, key in enumerate(ordered_keys):
            if index >= len(raw_boxes):
                break
            normalized = self._normalized_box(raw_boxes[index])
            if normalized is not None:
                current_boxes[key] = normalized

        camera_key = None if camera_id is None else str(camera_id)
        with self._lock:
            previous_keys = set(self.visible_track_keys_by_camera.get(camera_key, set()))

            for missing_key in previous_keys - keys:
                self.physical_conflict_rejections.pop(missing_key, None)
                for conflict_identity_id, conflict_state in list(
                    self.physical_conflicts.items()
                ):
                    if missing_key in conflict_state.get("candidates", {}):
                        self._cancel_physical_conflict_locked(
                            conflict_identity_id,
                            "candidate_track_disappeared",
                        )
                if missing_key in self.physical_conflict_recovery_holds:
                    self._release_physical_conflict_recovery_hold_locked(
                        missing_key,
                        "track_disappeared",
                        rearm_deferred_intake=False,
                    )
                if missing_key in self.pending_intake:
                    self.pending_intake.pop(missing_key, None)
                    self.track_results.pop(missing_key, None)
                    self.track_binding_metadata.pop(missing_key, None)
                    self._next_track_generation_locked(missing_key)
                # A vanished provisional track must not be resurrected by a
                # background task that was already extracting its features.
                self.shadow_tracks.pop(missing_key, None)

            # A brief detector wobble must not turn a duplicate into a new
            # identity. Require consecutive independent-motion observations
            # before releasing a nominated shadow.
            for key, shadow in list(self.shadow_tracks.items()):
                if self._camera_from_key(key) != camera_key or key not in keys:
                    continue
                canonical_key = shadow.get("canonical_key")
                identity_id = shadow.get("identity_id")
                mapped_identity_id = self.track_to_identity.get(canonical_key)
                if mapped_identity_id in self.identities:
                    if identity_id is not None and identity_id != mapped_identity_id:
                        self._release_shadow_locked(key, reason="canonical_master_changed")
                        continue
                    identity_id = mapped_identity_id
                    shadow["identity_id"] = identity_id
                    shadow["provisional"] = False
                elif identity_id is None:
                    # A provisional canonical is useful only while its own
                    # intake is still alive. Its disappearance or cancellation
                    # releases the newer candidate immediately.
                    if canonical_key not in keys or canonical_key not in self.pending_intake:
                        self._release_shadow_locked(
                            key,
                            reason="provisional_canonical_disappeared_or_intake_cancelled",
                        )
                        continue
                elif not shadow.get("verified"):
                    # An unverified target whose local owner was revoked is no
                    # longer a safe handoff candidate.
                    self._release_shadow_locked(
                        key,
                        reason="target_binding_revoked_before_verification",
                    )
                    continue

                if canonical_key in keys and key in current_boxes and canonical_key in current_boxes:
                    overlap_score = self._shadow_overlap_score(
                        current_boxes[key],
                        current_boxes[canonical_key],
                    )
                    if overlap_score is None:
                        shadow["separation_frames"] = int(shadow.get("separation_frames", 0)) + 1
                        if shadow["separation_frames"] >= self.shadow_separation_frames:
                            self._release_shadow_locked(key, reason="separated_from_canonical")
                            continue
                    else:
                        shadow["separation_frames"] = 0
                        shadow["overlap_frames"] = int(shadow.get("overlap_frames", 0)) + 1
                        shadow["overlap_score"] = overlap_score
                shadow["last_frame"] = None if frame_index is None else int(frame_index)
                shadow["last_seen"] = (
                    time.monotonic() if observed_at is None else float(observed_at)
                )

            canonical_keys = []
            for canonical_key in keys:
                if canonical_key not in current_boxes:
                    continue
                mapped_identity_id = self.track_to_identity.get(canonical_key)
                if mapped_identity_id in self.identities:
                    canonical_keys.append((canonical_key, mapped_identity_id, False))
                elif canonical_key in previous_keys and canonical_key in self.pending_intake:
                    # The older track is still building its first five-crop
                    # master. Hold the newly spawned overlap behind it instead
                    # of allowing two identical intakes to race into IDs 1/2.
                    canonical_keys.append((canonical_key, None, True))
            for key in keys - previous_keys:
                if (
                    key in self.track_to_identity
                    or key in self.pending_intake
                    or key in self.shadow_tracks
                    or key not in current_boxes
                ):
                    continue
                best_canonical = None
                best_identity_id = None
                best_provisional = False
                best_score = None
                for canonical_key, canonical_identity_id, provisional in canonical_keys:
                    if canonical_key == key:
                        continue
                    canonical_box = current_boxes.get(canonical_key, self.track_boxes.get(canonical_key))
                    score = self._shadow_overlap_score(current_boxes[key], canonical_box)
                    if score is not None and (best_score is None or score > best_score):
                        best_canonical = canonical_key
                        best_identity_id = canonical_identity_id
                        best_provisional = provisional
                        best_score = score
                if best_canonical is None:
                    continue
                self.shadow_tracks[key] = {
                    "canonical_key": best_canonical,
                    "identity_id": best_identity_id,
                    "provisional": best_provisional,
                    "verified": False,
                    "first_frame": None if frame_index is None else int(frame_index),
                    "last_frame": None if frame_index is None else int(frame_index),
                    "first_seen": time.monotonic() if observed_at is None else float(observed_at),
                    "last_seen": time.monotonic() if observed_at is None else float(observed_at),
                    "overlap_score": best_score,
                    "overlap_frames": 1,
                    "separation_frames": 0,
                }
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "shadow_nominated",
                    track_key=key,
                    canonical_track_key=best_canonical,
                    target_master_id=best_identity_id,
                    provisional=best_provisional,
                    overlap_score=best_score,
                    frame_index=frame_index,
                    camera_id=camera_id,
                )

            self.track_boxes.update(current_boxes)
            self.visible_track_keys_by_camera[camera_key] = keys
            if frame_index is not None:
                for key in keys:
                    previous_seen = self.track_last_seen.get(key)
                    if previous_seen is None:
                        continue
                    frame_gap = int(frame_index) - int(previous_seen[0])
                    if frame_gap < 0 or frame_gap > self.ttl_frames:
                        previous_identity_id = self.track_to_identity.get(key)
                        # TEMP_IDENTITY_DEBUG
                        identity_event(
                            "track_binding_reset",
                            track_key=key,
                            master_id=previous_identity_id,
                            camera_id=camera_id,
                            frame_index=frame_index,
                            previous_frame_index=previous_seen[0],
                            frame_gap=frame_gap,
                            ttl_frames=self.ttl_frames,
                            reason="frame_rewind" if frame_gap < 0 else "frame_gap_exceeded_ttl",
                        )
                        self._clear_local_binding_locked(key)
                self._abandon_missing_track_bindings_locked(
                    camera_key,
                    keys,
                    frame_index,
                )
            return {
                self.track_to_identity[key]
                for key in keys
                if key in self.track_to_identity
            }

    def mapped_identity_ids(self, track_ids, camera_id=None, frame_index=None):
        """Backward-compatible visibility update without box-aware shadows."""

        return self.observe_tracks(
            track_ids,
            None,
            frame_index=frame_index,
            camera_id=camera_id,
        )

    def is_track_suppressed(self, track_id, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            shadow = self.shadow_tracks.get(key)
            if shadow is None or key in self.track_to_identity:
                return False
            camera_key = self._camera_from_key(key)
            return shadow.get("canonical_key") in self.visible_track_keys_by_camera.get(
                camera_key,
                set(),
            )

    def lookup(self, track_id, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            identity_id = self.track_to_identity.get(key)
            return self._public_identity_id(identity_id)

    def temporary_group(self, track_id, camera_id=None):
        """Return the public token for an unnumbered location group."""

        key = self._track_key(track_id, camera_id)
        with self._lock:
            identity_id = self.track_to_identity.get(key)
            if identity_id is None or identity_id >= 0:
                return None
            return self._temporary_group_token(identity_id)

    def hold_new_master_creation(self, left_track_key, right_track_key, hold_token):
        """Prevent unmatched intake results from allocating a new master."""

        track_keys = (left_track_key, right_track_key)
        with self._lock:
            held_keys = []
            for key in track_keys:
                if self.track_to_identity.get(key) is not None:
                    continue
                tokens = self.new_master_holds.setdefault(key, set())
                if hold_token not in tokens:
                    tokens.add(hold_token)
                held_keys.append(key)
            if held_keys:
                identity_event(
                    "new_master_hold_applied",
                    console=False,
                    hold_token=hold_token,
                    held_track_keys=held_keys,
                    pair_track_keys=track_keys,
                )
            return tuple(held_keys)

    def release_new_master_hold(self, left_track_key, right_track_key, hold_token, reason):
        """Release one pair hold and re-arm any completed deferred intake."""

        track_keys = (left_track_key, right_track_key)
        resumed_keys = []
        with self._lock:
            for key in track_keys:
                tokens = self.new_master_holds.get(key)
                if tokens is not None:
                    tokens.discard(hold_token)
                    if not tokens:
                        self.new_master_holds.pop(key, None)
                if key in self.new_master_holds:
                    continue
                state = self.pending_intake.get(key)
                if state is None or not state.pop("deferred_by_new_master_hold", False):
                    continue
                state["submitted"] = False
                state["next_retry_frame"] = int(state.get("last_frame") or 0) + 1
                resumed_keys.append(key)
            identity_event(
                "new_master_hold_released",
                console=False,
                hold_token=hold_token,
                pair_track_keys=track_keys,
                resumed_track_keys=resumed_keys,
                reason=reason,
            )
        return tuple(resumed_keys)

    def lookup_track_key(self, track_key):
        """Return the shared ID for an already-canonical camera/track key."""

        with self._lock:
            return self.track_to_identity.get(track_key)

    def identity_state(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            return record.get("identity_state") if record is not None else None

    @staticmethod
    def _track_identity_state_locked(record, track_key):
        if track_key in record.get("challenged_member_keys", ()):
            return "challenged"
        if track_key in record.get("pending_member_keys", ()):
            return "provisional"
        return record.get("identity_state", "confirmed")

    def track_identity_state(self, track_key):
        with self._lock:
            identity_id = self.track_to_identity.get(track_key)
            record = self.identities.get(identity_id)
            if record is None:
                return None
            return self._track_identity_state_locked(record, track_key)

    def _start_provisional_split_recovery_locked(
        self,
        identity_id,
        revoked_track_key,
        frame_index,
        observed_at,
    ):
        """Keep a temporary singleton recoverable, but never indefinitely."""

        record = self.identities.get(identity_id)
        if (
            identity_id is None
            or identity_id >= 0
            or record is None
            or record.get("identity_state") not in ("provisional", "challenged")
        ):
            return False
        remaining_keys = {
            key
            for key in record.get("member_track_keys", ())
            if self.track_to_identity.get(key) == identity_id
        }
        remaining_cameras = {self._camera_from_key(key) for key in remaining_keys}
        if not remaining_keys:
            self.identities.pop(identity_id, None)
            self.recent_master_observations.pop(identity_id, None)
            self._clear_physical_violation_counts_locked(identity_id)
            return False
        if len(remaining_cameras) >= 2:
            return False

        current_wall_time = time.monotonic()
        recovery = record.get("split_recovery")
        if not isinstance(recovery, dict):
            recovery = {
                "started_monotonic": current_wall_time,
                "expires_monotonic": (
                    current_wall_time + self.provisional_split_recovery_seconds
                ),
                "revoked_track_keys": set(),
            }
            record["split_recovery"] = recovery
        recovery.setdefault("revoked_track_keys", set()).add(revoked_track_key)
        recovery["last_split_frame"] = (
            None if frame_index is None else int(frame_index)
        )
        recovery["last_split_observed_at"] = observed_at
        identity_event(
            "provisional_split_recovery_started",
            temporary_group_id=self._temporary_group_token(identity_id),
            revoked_track_key=revoked_track_key,
            remaining_track_keys=sorted(remaining_keys, key=repr),
            frame_index=frame_index,
            observed_at=observed_at,
            recovery_seconds=self.provisional_split_recovery_seconds,
            reason="temporary_group_became_singleton",
        )
        return True

    def _provisional_split_recovery_expired_locked(self, identity_id):
        record = self.identities.get(identity_id)
        recovery = None if record is None else record.get("split_recovery")
        if not isinstance(recovery, dict):
            return False
        return time.monotonic() >= float(recovery.get("expires_monotonic", 0.0))

    def _dissolve_provisional_split_locked(self, identity_id, reason):
        record = self.identities.get(identity_id)
        if record is None or identity_id >= 0:
            return ()
        member_keys = tuple(
            sorted(
                (
                    key
                    for key in record.get("member_track_keys", ())
                    if self.track_to_identity.get(key) == identity_id
                ),
                key=repr,
            )
        )
        temporary_group_id = self._temporary_group_token(identity_id)
        for member_key in member_keys:
            self._clear_local_binding_locked(member_key)
        self.identities.pop(identity_id, None)
        self.recent_master_observations.pop(identity_id, None)
        self._clear_physical_violation_counts_locked(identity_id)
        identity_event(
            "provisional_split_recovery_expired",
            temporary_group_id=temporary_group_id,
            released_track_keys=member_keys,
            reason=reason,
        )
        return member_keys

    def _recoverable_split_pair_locked(self, left_track_key, right_track_key):
        track_keys = (left_track_key, right_track_key)
        mapped = [self.track_to_identity.get(key) for key in track_keys]
        if any(identity_id is None for identity_id in mapped):
            return None
        confirmed_indexes = [
            index
            for index, identity_id in enumerate(mapped)
            if identity_id > 0
            and self.identities.get(identity_id, {}).get("identity_state") == "confirmed"
        ]
        temporary_indexes = [
            index
            for index, identity_id in enumerate(mapped)
            if identity_id < 0
            and self.identities.get(identity_id, {}).get("identity_state")
            in ("provisional", "challenged")
        ]
        if len(confirmed_indexes) != 1 or len(temporary_indexes) != 1:
            return None

        confirmed_index = confirmed_indexes[0]
        temporary_index = temporary_indexes[0]
        confirmed_key = track_keys[confirmed_index]
        temporary_key = track_keys[temporary_index]
        confirmed_identity_id = mapped[confirmed_index]
        temporary_identity_id = mapped[temporary_index]
        temporary_record = self.identities.get(temporary_identity_id)
        recovery = temporary_record.get("split_recovery")
        if not isinstance(recovery, dict) or self._provisional_split_recovery_expired_locked(
            temporary_identity_id
        ):
            return None
        live_temporary_members = {
            key
            for key in temporary_record.get("member_track_keys", ())
            if self.track_to_identity.get(key) == temporary_identity_id
        }
        if live_temporary_members != {temporary_key}:
            return None
        if confirmed_key not in recovery.get("revoked_track_keys", ()):
            return None
        return (
            confirmed_identity_id,
            confirmed_key,
            temporary_identity_id,
            temporary_key,
        )

    def can_recover_provisional_pair(self, left_track_key, right_track_key):
        """Whether geometry may reunite this exact split temporary pair."""

        with self._lock:
            return (
                self._recoverable_split_pair_locked(left_track_key, right_track_key)
                is not None
            )

    def _recover_provisional_split_locked(self, left_track_key, right_track_key):
        recovery_pair = self._recoverable_split_pair_locked(
            left_track_key,
            right_track_key,
        )
        if recovery_pair is None:
            return None
        (
            confirmed_identity_id,
            confirmed_key,
            temporary_identity_id,
            temporary_key,
        ) = recovery_pair
        confirmed_record = self.identities[confirmed_identity_id]
        temporary_group_id = self._temporary_group_token(temporary_identity_id)

        # Drop every result and queued generation collected for the old
        # temporary group. The next assign call must collect a fresh batch and
        # compare it specifically with the newly created master.
        self._clear_local_binding_locked(temporary_key)
        self.identities.pop(temporary_identity_id, None)
        self.recent_master_observations.pop(temporary_identity_id, None)
        self._clear_physical_violation_counts_locked(temporary_identity_id)

        self.track_to_identity[temporary_key] = confirmed_identity_id
        confirmed_record.setdefault("member_track_keys", set()).add(temporary_key)
        confirmed_record.setdefault("pending_member_keys", set()).add(temporary_key)
        confirmed_record.setdefault("challenged_member_keys", set()).discard(temporary_key)
        confirmed_record.setdefault("appearance_rejected_member_keys", set()).discard(
            temporary_key
        )
        confirmed_record.setdefault("pending_member_location_streaks", {})[
            temporary_key
        ] = 0
        confirmed_record["location_managed"] = True
        self.track_binding_metadata[temporary_key] = {
            "appearance_confirmed": False,
            "identity_state": "provisional",
            "confirmation_reason": None,
            "provisional_intake_complete": False,
            "temporary_group_id": None,
        }
        self.new_master_holds.pop(temporary_key, None)
        identity_event(
            "provisional_split_recovery_attached",
            temporary_group_id=temporary_group_id,
            master_id=confirmed_identity_id,
            confirmed_track_key=confirmed_key,
            recovering_track_key=temporary_key,
            reason="renewed_cross_camera_location",
        )
        return confirmed_identity_id

    def identity_is_location_managed(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            return bool(record and record.get("location_managed"))

    def create_provisional_pair(self, left_track_key, right_track_key):
        """Create an internal, unnumbered group for a stable camera pair."""

        track_keys = (left_track_key, right_track_key)
        if any(not isinstance(key, tuple) or len(key) != 2 for key in track_keys):
            return None
        with self._lock:
            mapped = [self.track_to_identity.get(key) for key in track_keys]
            mapped_ids = {identity_id for identity_id in mapped if identity_id is not None}
            if len(mapped_ids) == 1 and mapped[0] == mapped[1]:
                return mapped[0]

            recovered_identity_id = self._recover_provisional_split_locked(
                left_track_key,
                right_track_key,
            )
            if recovered_identity_id is not None:
                return recovered_identity_id

            provisional_ids = {
                identity_id
                for identity_id in mapped_ids
                if self.identities.get(identity_id, {}).get("identity_state")
                in ("provisional", "challenged")
            }
            confirmed_ids = mapped_ids - provisional_ids
            if len(confirmed_ids) > 1 or len(provisional_ids) > 1 or (
                confirmed_ids and provisional_ids
            ):
                return None

            attaching_to_confirmed = bool(confirmed_ids)
            created_new_record = False
            if confirmed_ids:
                identity_id = next(iter(confirmed_ids))
            elif provisional_ids:
                identity_id = next(iter(provisional_ids))
            else:
                identity_id = None

            if identity_id is not None and any(
                existing_id is None
                and identity_id in self.physical_conflict_rejections.get(key, ())
                for key, existing_id in zip(track_keys, mapped)
            ):
                identity_event(
                    "provisional_pair_declined",
                    master_id=identity_id,
                    member_track_keys=sorted(track_keys, key=repr),
                    reason="appearance_conflict_loser_cannot_reclaim_master",
                )
                return None

            # Single-owner invariant.  Geometry alone must never hand a master
            # to a second live track in a camera where another visible track
            # already owns it; the two would then share every identity-keyed
            # position memory and collapse onto one map point.  Checked before
            # any mutation so a refusal leaves no partial state behind.
            if identity_id is not None and self._reject_second_visible_owner_locked(
                identity_id,
                track_keys,
                mapped,
                "provisional_pair_declined",
            ):
                return None

            if identity_id is None:
                identity_id = self.next_temporary_group_id
                self.next_temporary_group_id -= 1
                self.identities[identity_id] = self._new_record(identity_state="provisional")
                created_new_record = True

            record = self.identities[identity_id]
            record["location_managed"] = True
            if attaching_to_confirmed:
                established_baseline = record.get("gallery", {}).get("baseline")
                if established_baseline is not None:
                    record.setdefault("camera_baselines", {})[
                        "__established_master__"
                    ] = dict(established_baseline)
                # Reuse any already-labelled views from the established
                # camera.  The later camera still completes its own intake
                # before the geometric link becomes fully confirmed.
                for slot_name, slot in record.get("gallery", {}).items():
                    if not slot or not slot.get("camera_id"):
                        continue
                    camera_id = str(slot["camera_id"])
                    if slot_name == "baseline":
                        record.setdefault("camera_baselines", {}).setdefault(
                            camera_id,
                            dict(slot),
                        )
                    elif slot_name in REID_SEMANTIC_SLOTS:
                        camera_gallery = record.setdefault("camera_views", {}).setdefault(
                            camera_id,
                            {name: None for name in REID_SEMANTIC_SLOTS},
                        )
                        if camera_gallery.get(slot_name) is None:
                            camera_gallery[slot_name] = dict(slot)
            for key, existing_id in zip(track_keys, mapped):
                if existing_id not in (None, identity_id):
                    return None
                self.track_to_identity[key] = identity_id
                record["member_track_keys"].add(key)
                newly_attached = existing_id is None
                if attaching_to_confirmed and newly_attached:
                    record.setdefault("pending_member_keys", set()).add(key)
                    record.setdefault("pending_member_location_streaks", {})[key] = 0
                    # This camera's stored crops used to be erased here so a
                    # newcomer could not be confirmed on an older track's
                    # evidence.  They are genuine views of this person and are
                    # now kept: emptying them reopened four slots on every
                    # re-attachment, and during a swap the newcomer filled them
                    # with somebody else.  The confirmation is scoped instead
                    # -- an identity that already owns a master gallery is
                    # re-checked against that gallery, never against the two
                    # cameras' stored views of each other.
                state = self.pending_intake.get(key)
                if state is not None:
                    state["provisional_identity_id"] = identity_id
                if created_new_record or newly_attached:
                    metadata = self.track_binding_metadata.setdefault(key, {})
                    metadata.update(
                        {
                            "appearance_confirmed": False,
                            "identity_state": (
                                "provisional"
                                if attaching_to_confirmed
                                else record["identity_state"]
                            ),
                            "confirmation_reason": None,
                            "provisional_intake_complete": False,
                            "temporary_group_id": (
                                f"tmp_{abs(int(identity_id))}"
                                if identity_id < 0
                                else None
                            ),
                        }
                    )

            identity_event(
                (
                    "provisional_member_attached"
                    if attaching_to_confirmed
                    else "provisional_identity_created"
                ),
                temporary_group_id=(
                    f"tmp_{abs(int(identity_id))}" if identity_id < 0 else None
                ),
                master_id=identity_id if identity_id > 0 else None,
                member_track_keys=track_keys,
                reason="repeated_cross_camera_location",
            )
            return identity_id

    def _track_key_is_visible_locked(self, key):
        camera_key = self._camera_from_key(key)
        if camera_key in self.visible_track_keys_by_camera:
            return key in self.visible_track_keys_by_camera[camera_key]
        # A camera that has not reported yet cannot vouch either way, so the
        # key is treated as present rather than silently dropped.
        return True

    def _provisional_global_reid_complete_locked(self, identity_id):
        """Return whether every present provisional member was searched globally.

        Members are counted only while their camera can still see them.  The
        sweep in ``_abandon_missing_track_bindings_locked`` should already have
        removed anything long gone; this is the backstop that keeps one missed
        cleanup from stranding a whole group indefinitely.
        """

        record = self.identities.get(identity_id)
        if record is None:
            return False
        member_keys = {
            key
            for key in record.get("member_track_keys", ())
            if self.track_to_identity.get(key) == identity_id
            and self._track_key_is_visible_locked(key)
        }
        checked_keys = set(record.get("global_reid_checked_track_keys", ()))
        return bool(member_keys) and member_keys.issubset(checked_keys)

    @staticmethod
    def _slot_is_better(candidate, existing):
        if candidate is None:
            return False
        if existing is None:
            return True
        return (
            float(candidate.get("sharpness", 0.0)),
            float(candidate.get("detection_confidence") or 0.0),
        ) > (
            float(existing.get("sharpness", 0.0)),
            float(existing.get("detection_confidence") or 0.0),
        )

    def _identity_reference_features_locked(self, record, feature_space_id):
        """Every stored view of this identity that is comparable to a new crop."""

        references = []
        for slot_name, slot in (record.get("gallery") or {}).items():
            if slot and slot.get("feature_space_id") == feature_space_id:
                references.append((f"gallery:{slot_name}", slot))
        # Once the master gallery is complete it is the whole answer, and the
        # per-camera views are the working notes that produced it.  Adding them
        # back only widens the reference set, and a wider set is easier to slip
        # past: the same stray crop sat 0.318 from the five gallery slots but
        # 0.260 from the thirteen, because the rule keeps whichever reference
        # is friendliest.  More photos lower the bar; they never raise it.
        if self._gallery_is_sealed_locked(record):
            return references
        for camera_id, slot in (record.get("camera_baselines") or {}).items():
            if slot and slot.get("feature_space_id") == feature_space_id:
                references.append((f"baseline:{camera_id}", slot))
        for camera_id, camera_gallery in (record.get("camera_views") or {}).items():
            for slot_name, slot in (camera_gallery or {}).items():
                if slot and slot.get("feature_space_id") == feature_space_id:
                    references.append((f"view:{camera_id}:{slot_name}", slot))
        return references

    def _gallery_admission_rejected_locked(
        self,
        identity_id,
        record,
        slot,
        slot_name,
        scope,
        track_key=None,
    ):
        """Refuse a crop that agrees with none of the identity's stored views.

        A swapped local track produces crops that pass every quality gate while
        showing the wrong person.  Only a comparison against what the identity
        already looks like can catch that, and it must be a comparison against
        the *best* matching stored view: a genuine new angle disagrees with
        some stored angles, but should still resemble at least one.
        """

        if record is None or not slot:
            return False
        # A track the audit has already caught matching another master is
        # under suspicion, and its crops are the ones that poison a gallery.
        # It contributes nothing until it has answered that charge -- this is
        # the only guard that also covers a gallery still being filled, where
        # sealing has nothing to protect yet.
        disputed = self.identity_audit_state.get(track_key, {}).get("rivals") or {}
        if disputed:
            identity_event(
                "gallery_admission_rejected",
                reason="track_identity_disputed",
                master_id=self._public_identity_id(identity_id),
                temporary_group_id=self._temporary_group_token(identity_id),
                track_key=track_key,
                camera_id=slot.get("camera_id"),
                orientation=slot_name,
                scope=str(scope),
                disputed_rival_master_ids=sorted(disputed),
                candidate_image_path=slot.get("image_path"),
            )
            return True
        candidate = self._normalize_feature(slot.get("feature"))
        if candidate is None:
            return False
        feature_space_id = slot.get("feature_space_id")
        references = [
            (label, reference)
            for label, reference in self._identity_reference_features_locked(
                record,
                feature_space_id,
            )
            if reference is not slot
        ]
        distances = []
        for label, reference in references:
            reference_feature = self._normalize_feature(reference.get("feature"))
            if reference_feature is None or reference_feature.shape != candidate.shape:
                continue
            distance = 1.0 - float(
                np.dot(
                    np.asarray(candidate, dtype=np.float64),
                    np.asarray(reference_feature, dtype=np.float64),
                )
            )
            distances.append((distance, label, reference))
        if not distances:
            # The first view of an identity has nothing to be judged against.
            return False
        ordered = sorted(distances, key=lambda item: (item[0], item[1]))
        # The closest stored view used to decide this alone, so one forgiving
        # angle could vouch for a stranger while every other view disagreed:
        # the crop that put Haoran into Mikail's gallery was 0.26 from one
        # photo and 0.32-0.41 from the other nine.  The median answers what
        # the gallery thinks as a whole rather than what its friendliest
        # member thinks, and cannot be swung by a single outlier.
        verdict_distance = statistics.median(item[0] for item in ordered)
        best_distance, best_label, best_reference = ordered[0]
        # Naming the stored photo that decided this is the whole point of the
        # record: a wrong admission is only diagnosable by opening the crop
        # that vouched for it alongside the crop it let in.
        comparisons = [
            {
                "stored_view": label,
                "distance": distance,
                "image_path": reference.get("image_path"),
                "camera_id": reference.get("camera_id"),
                "captured_frame": reference.get("captured_frame"),
            }
            for distance, label, reference in sorted(
                distances,
                key=lambda item: (item[0], item[1]),
            )
        ]
        shared = {
            "master_id": self._public_identity_id(identity_id),
            "temporary_group_id": self._temporary_group_token(identity_id),
            "track_key": track_key,
            "camera_id": slot.get("camera_id"),
            "orientation": slot_name,
            "scope": str(scope),
            "best_distance": best_distance,
            "verdict_distance": verdict_distance,
            "closest_stored_view": best_label,
            "closest_stored_view_image_path": best_reference.get("image_path"),
            "closest_stored_view_camera_id": best_reference.get("camera_id"),
            "closest_stored_view_captured_frame": best_reference.get("captured_frame"),
            "admission_distance": self.gallery_admission_distance,
            "compared_view_count": len(distances),
            "captured_frame": slot.get("captured_frame"),
            "sharpness": slot.get("sharpness"),
            "detection_confidence": slot.get("detection_confidence"),
            "feature_space_id": feature_space_id,
            "candidate_image_path": slot.get("image_path"),
            "image_path": slot.get("image_path"),
            "comparisons": comparisons,
        }
        if verdict_distance >= self.gallery_admission_distance:
            # TEMP_IDENTITY_DEBUG: console=True -- refusing a genuine new angle
            # would quietly starve an identity's gallery, so it must be visible.
            identity_event(
                "gallery_admission_rejected",
                reason="unlike_stored_views",
                **shared,
            )
            return True
        identity_event(
            "gallery_admission_accepted",
            console=False,
            reason="matches_stored_view",
            **shared,
        )
        return False

    def _stage_pending_member_evidence_locked(
        self,
        identity_id,
        track_key,
        camera_id,
        baseline_slot,
        baseline_task,
        semantic_slots,
    ):
        stage = self.pending_member_evidence.setdefault(
            track_key,
            {
                "identity_id": identity_id,
                "camera_id": str(camera_id),
                "baseline": None,
                "baseline_task": None,
                "views": {},
                "view_tasks": {},
            },
        )
        if stage.get("identity_id") != identity_id:
            stage.clear()
            stage.update(
                {
                    "identity_id": identity_id,
                    "camera_id": str(camera_id),
                    "baseline": None,
                    "baseline_task": None,
                    "views": {},
                    "view_tasks": {},
                }
            )
        if self._slot_is_better(baseline_slot, stage.get("baseline")):
            stage["baseline"] = baseline_slot
            stage["baseline_task"] = baseline_task
        for slot_name, (slot, evidence_task) in semantic_slots.items():
            if self._slot_is_better(slot, stage["views"].get(slot_name)):
                stage["views"][slot_name] = slot
                stage["view_tasks"][slot_name] = evidence_task

    def _slot_admits(self, candidate, existing, sealed):
        """Quality decides between crops; a sealed gallery refuses replacements."""

        if sealed and existing is not None:
            return False
        return self._slot_is_better(candidate, existing)

    def _gallery_is_sealed_locked(self, record):
        """Whether this identity already holds a baseline and all four sides.

        A complete gallery stops accepting replacements.  Slot quality alone
        decided overwrites before, and sharpness cannot tell one man from
        another: during a swap a crisp crop of the wrong person beat the
        slightly softer crop of the right one and took the slot -- including,
        once, the baseline every other comparison is measured against.  Each
        swap therefore transplanted a few photos permanently, and two of them
        in one huddle blended two identities into an average of both, after
        which no rival could ever win by a margin and the audit was left with
        nothing to separate them.

        The cost is accepted deliberately: an identity built from poor early
        crops keeps them for the session.  A gallery that cannot improve is
        recoverable; one that silently absorbs the wrong person is not.
        """

        if record is None:
            return False
        gallery = record.get("gallery") or {}
        return all(gallery.get(slot_name) is not None for slot_name in REID_GALLERY_SLOTS)

    def _retain_unconfirmed_evidence_locked(self, identity_id, track_key, camera_id, stage):
        """Write a withheld crop where it can be looked at, not where it counts.

        Discarding these outright would leave a wrong attachment invisible: the
        only way to see that a swapped box was feeding somebody else's identity
        is to open the crops it offered.  They go to a folder of their own so
        the master's folder stays a faithful picture of that master's gallery,
        which is what makes it usable for inspection at all.
        """

        if self.evidence_dir is None:
            return False
        track_label = f"{camera_id}_{track_key[1]}" if len(track_key) == 2 else str(track_key)
        review_dir = (
            self.evidence_dir
            / "Unconfirmed"
            / f"Master_{int(identity_id):04d}"
            / f"track_{track_label}"
        )
        retained = False
        tasks = [("baseline", stage.get("baseline_task"))]
        tasks.extend(stage.get("view_tasks", {}).items())
        for slot_name, task in tasks:
            if not task:
                continue
            original = Path(task.get("output_path") or f"{slot_name}.png")
            task["output_path"] = str(review_dir / original.name)
            self._queue_evidence_save(task)
            retained = True
        return retained

    def _commit_pending_member_evidence_locked(self, identity_id, track_key):
        stage = self.pending_member_evidence.pop(track_key, None)
        record = self.identities.get(identity_id)
        if stage is None or record is None or stage.get("identity_id") != identity_id:
            return False
        camera_id = str(stage.get("camera_id"))
        # Staged crops are never written into the identity.  They were gathered
        # while the track was only a positional guess, and delivering them on
        # confirmation is how a swapped box put its own photographs into
        # somebody else's gallery.  Confirmation does not need them either --
        # the global ReID check compares a fresh crop against the master
        # gallery and never reads staging.  Anything the member contributes
        # from here on arrives through the ordinary path, as a track that has
        # already proved who it is, and can still fill an empty master slot.
        withheld = self._retain_unconfirmed_evidence_locked(
            identity_id,
            track_key,
            camera_id,
            stage,
        )
        identity_event(
            "pending_member_evidence_withheld",
            master_id=identity_id,
            track_key=track_key,
            camera_id=camera_id,
            withheld_orientations=sorted(stage.get("views", {})),
            withheld_baseline=stage.get("baseline") is not None,
            retained_for_review=withheld,
            reason="crops_gathered_before_the_track_was_confirmed",
        )
        return True

    @staticmethod
    def _defer_provisional_evidence_locked(record, evidence_task):
        """Hold an unpromoted group's crop until the group is proven real."""

        if record is None or evidence_task is None:
            return
        record.setdefault("deferred_evidence_tasks", []).append(evidence_task)

    def _flush_deferred_evidence_locked(self, record, reason, final_identity_id=None):
        """Write a now-trusted group's held crops to the evidence folder.

        A group is numbered only at promotion, so its held tasks still address
        the ``Temporary_NNNN`` folder.  Re-address them to the master's folder
        and keep the slots' ``image_path`` in step, otherwise the saved digest
        would never find its slot.
        """

        if record is None:
            return 0
        tasks = record.pop("deferred_evidence_tasks", [])
        for evidence_task in tasks:
            if (
                final_identity_id is not None
                and int(evidence_task.get("identity_id", 0)) != int(final_identity_id)
            ):
                # Match slots on the exact stored string.  Round-tripping it
                # through Path would rewrite the separators and silently miss.
                previous_path = str(evidence_task.get("output_path"))
                folder = (
                    f"Temporary_{abs(int(final_identity_id)):04d}"
                    if int(final_identity_id) < 0
                    else f"Master_{int(final_identity_id):04d}"
                )
                parsed = Path(previous_path)
                new_path = str(parsed.parent.parent / folder / parsed.name)
                for slot in self._slots_with_path(record, previous_path):
                    slot["image_path"] = new_path
                for capture_key, path in list(self._evidence_capture_paths.items()):
                    if path == previous_path:
                        self._evidence_capture_paths[capture_key] = new_path
                evidence_task["identity_id"] = int(final_identity_id)
                evidence_task["output_path"] = new_path
            self._queue_evidence_save(evidence_task)
        if tasks:
            identity_event(
                "deferred_evidence_released",
                console=False,
                master_id=self._public_identity_id(final_identity_id),
                released_count=len(tasks),
                reason=reason,
            )
        return len(tasks)

    def _discard_pending_member_evidence_locked(self, track_key, reason):
        stage = self.pending_member_evidence.pop(track_key, None)
        if stage is not None:
            identity_event(
                "pending_member_evidence_discarded",
                master_id=stage.get("identity_id"),
                track_key=track_key,
                camera_id=stage.get("camera_id"),
                reason=reason,
            )

    def _borderline_match_needs_retry_locked(
        self,
        track_key,
        identity_id,
        matched_slot,
        distance,
        feature_space_id,
        frame_index,
        phase,
    ):
        metadata = self.track_binding_metadata.setdefault(track_key, {})
        if distance is None or distance <= self.strong_match_distance:
            metadata.pop("tentative_reid_match", None)
            return False
        if distance >= self.distance_threshold:
            metadata.pop("tentative_reid_match", None)
            return False

        previous = metadata.get("tentative_reid_match") or {}
        same_candidate = bool(
            previous.get("identity_id") == identity_id
            and previous.get("feature_space_id") == feature_space_id
        )
        confirmations = int(previous.get("confirmations", 0)) + 1 if same_candidate else 1
        if confirmations >= 2:
            metadata.pop("tentative_reid_match", None)
            identity_event(
                "borderline_reid_confirmed",
                track_key=track_key,
                master_id=identity_id,
                matched_slot=matched_slot,
                distance=distance,
                strong_distance_threshold=self.strong_match_distance,
                distance_threshold=self.distance_threshold,
                confirmations=confirmations,
                phase=phase,
            )
            return False

        metadata["tentative_reid_match"] = {
            "identity_id": identity_id,
            "feature_space_id": feature_space_id,
            "matched_slot": matched_slot,
            "distance": float(distance),
            "confirmations": confirmations,
        }
        state = self.pending_intake.get(track_key)
        if state is not None:
            retry_started_at = (
                state["samples"][-1].get("observed_at", time.monotonic())
                if state.get("samples")
                else time.monotonic()
            )
            state["submitted"] = False
            state["samples"] = []
            state["first_seen"] = float(retry_started_at)
            state["next_retry_frame"] = int(frame_index) + self.intake_retry_frames
            state["generation"] = self._next_track_generation_locked(track_key)
        identity_event(
            "borderline_reid_deferred",
            track_key=track_key,
            candidate_master_id=identity_id,
            matched_slot=matched_slot,
            distance=distance,
            strong_distance_threshold=self.strong_match_distance,
            distance_threshold=self.distance_threshold,
            confirmations=confirmations,
            confirmations_required=2,
            phase=phase,
        )
        return True

    def _merge_agreement_rejected_locked(
        self,
        provisional_id,
        target_id,
        query_feature,
        feature_space_id,
        matched_slot,
        best_distance,
    ):
        """Weigh a merge against the target's whole gallery, not its best slot.

        Folding a group into an existing master is the largest commitment the
        system makes -- it hands one person's photographs to another identity
        permanently -- and it turned on a single closest-slot comparison.  One
        man merged into another's on 0.28993 against a 0.30 bar, having first
        won three separate intake batches at 0.26-0.29, so nothing here was a
        fluke of one frame.  Measured across the target's whole gallery that
        pair sat at 0.378, while genuinely different people in the same session
        ran 0.53-0.79 and the one real duplicate ran 0.279.  The evidence was
        present; only the rule was looking at one number.
        """

        target = self.identities.get(target_id)
        provisional = self.identities.get(provisional_id)
        if target is None or provisional is None:
            return False
        target_refs = [
            slot
            for _label, slot in self._identity_reference_features_locked(
                target,
                feature_space_id,
            )
        ]
        incoming = [
            slot
            for _label, slot in self._identity_reference_features_locked(
                provisional,
                feature_space_id,
            )
        ]
        incoming_features = [self._normalize_feature(s.get("feature")) for s in incoming]
        # The crop that raised the question counts as evidence too, and may be
        # all a young group has.
        incoming_features.append(self._normalize_feature(query_feature))
        distances = []
        for left in incoming_features:
            if left is None:
                continue
            for slot in target_refs:
                right = self._normalize_feature(slot.get("feature"))
                if right is None or right.shape != left.shape:
                    continue
                distances.append(
                    1.0
                    - float(
                        np.dot(
                            np.asarray(left, dtype=np.float64),
                            np.asarray(right, dtype=np.float64),
                        )
                    )
                )
        if not distances:
            # Nothing comparable beyond the match that got us here; the merge
            # keeps whatever guarantee that comparison already provided.
            return False
        agreement = statistics.median(distances)
        rejected = agreement >= self.provisional_merge_distance
        identity_event(
            "provisional_merge_agreement",
            console=bool(rejected),
            target_master_id=target_id,
            temporary_group_id=self._temporary_group_token(provisional_id),
            matched_slot=matched_slot,
            best_slot_distance=best_distance,
            gallery_agreement_distance=agreement,
            merge_distance_limit=self.provisional_merge_distance,
            compared_pair_count=len(distances),
            accepted=not rejected,
            reason=(
                "gallery_disagrees_with_the_best_slot"
                if rejected
                else "gallery_agrees"
            ),
        )
        return rejected

    def _merge_provisional_into_confirmed_locked(
        self,
        provisional_identity_id,
        target_identity_id,
        matched_track_key,
        matched_slot,
        distance,
        feature_source,
        feature_space_id,
    ):
        """Move a location-paired provisional group onto an existing master."""

        provisional = self.identities.get(provisional_identity_id)
        target = self.identities.get(target_identity_id)
        if (
            provisional is None
            or target is None
            or provisional_identity_id == target_identity_id
            or provisional.get("identity_state") not in ("provisional", "challenged")
            or target.get("identity_state", "confirmed") != "confirmed"
        ):
            return None

        member_keys = {
            key
            for key in provisional.get("member_track_keys", ())
            if self.track_to_identity.get(key) == provisional_identity_id
        }
        if matched_track_key not in member_keys:
            return None

        # The per-key loop below only retires *non-visible* same-camera owners,
        # so without this a merge could hand the target master to a second live
        # track while the first one is still on screen.  Refuse before touching
        # the target record; the caller then leaves the provisional group alone
        # and it can merge later, once the camera has one owner again.
        if self._reject_second_visible_owner_locked(
            target_identity_id,
            sorted(member_keys, key=repr),
            [provisional_identity_id] * len(member_keys),
            "provisional_merge_declined",
        ):
            # Remember which master this group really belongs to.  Declining is
            # not enough on its own: the group would otherwise reach its stable
            # location streak and be promoted to a brand-new master, so the
            # refusal would manufacture the very duplicate it exists to prevent.
            provisional["merge_blocked_by_master"] = target_identity_id
            return None

        target["location_managed"] = True
        target["last_member_confirmation_reason"] = "global_reid"
        target["last_seen_monotonic"] = max(
            float(target.get("last_seen_monotonic", 0.0)),
            float(provisional.get("last_seen_monotonic", 0.0)),
        )
        target["hits"] = int(target.get("hits", 0)) + int(provisional.get("hits", 0))

        # A merge brings a whole group's crops at once, so an established
        # identity would be overwritten wholesale by whoever the group turns
        # out to contain.  Empty slots still fill; filled ones stand.
        sealed = self._gallery_is_sealed_locked(target)
        target_baselines = target.setdefault("camera_baselines", {})
        for camera_id, slot in provisional.get("camera_baselines", {}).items():
            existing = target_baselines.get(camera_id)
            if self._slot_admits(slot, existing, sealed):
                target_baselines[camera_id] = dict(slot)

        target_views = target.setdefault("camera_views", {})
        for camera_id, camera_gallery in provisional.get("camera_views", {}).items():
            destination = target_views.setdefault(
                camera_id,
                {slot_name: None for slot_name in REID_SEMANTIC_SLOTS},
            )
            for slot_name in REID_SEMANTIC_SLOTS:
                slot = camera_gallery.get(slot_name)
                if self._slot_admits(slot, destination.get(slot_name), sealed):
                    destination[slot_name] = dict(slot)

        baseline_space = (target.get("gallery", {}).get("baseline") or {}).get(
            "feature_space_id"
        )
        for slot_name in REID_SEMANTIC_SLOTS:
            candidates = [
                camera_gallery.get(slot_name)
                for camera_gallery in target_views.values()
                if camera_gallery.get(slot_name) is not None
                and camera_gallery.get(slot_name).get("feature_space_id") == baseline_space
            ]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda slot: (
                    float(slot.get("sharpness", 0.0)),
                    float(slot.get("detection_confidence") or 0.0),
                ),
            )
            if self._slot_admits(best, target.get("gallery", {}).get(slot_name), sealed):
                target["gallery"][slot_name] = dict(best)

        target_observations = self.recent_master_observations.setdefault(
            target_identity_id,
            {},
        )
        for camera_id, observation in self.recent_master_observations.pop(
            provisional_identity_id,
            {},
        ).items():
            existing = target_observations.get(camera_id)
            if existing is None or float(observation.get("observed_at", 0.0)) >= float(
                existing.get("observed_at", 0.0)
            ):
                target_observations[camera_id] = dict(observation)

        for key in sorted(member_keys, key=repr):
            camera_id = self._camera_from_key(key)
            self._clear_nonvisible_same_camera_owners_locked(
                target_identity_id,
                camera_id,
                preserved_key=key,
            )
            self.track_to_identity[key] = target_identity_id
            target.setdefault("member_track_keys", set()).add(key)
            metadata = self.track_binding_metadata.setdefault(key, {})
            is_appearance_match = key == matched_track_key
            metadata.update(
                {
                    "query_feature_space_id": (
                        feature_space_id
                        if is_appearance_match
                        else metadata.get("query_feature_space_id")
                    ),
                    "matched_feature_space_id": baseline_space,
                    "matched_slot": matched_slot if is_appearance_match else None,
                    "distance": distance if is_appearance_match else None,
                    "appearance_confirmed": bool(is_appearance_match),
                    "feature_source": (
                        feature_source
                        if is_appearance_match
                        else metadata.get("feature_source")
                    ),
                    "identity_state": "confirmed",
                    "confirmation_reason": "global_reid",
                    "provisional_intake_complete": True,
                    "temporary_group_id": None,
                }
            )
            self.track_results[key] = {
                "similarity": (
                    1.0 - float(distance)
                    if is_appearance_match and distance is not None
                    else 0.0
                ),
                "reidentified": True,
                "matched_slot": matched_slot if is_appearance_match else None,
            }
            self.pending_intake.pop(key, None)
            self.shadow_tracks.pop(key, None)
            self._next_track_generation_locked(key)

        for pending_key in list(self.pending_semantic_slots):
            if isinstance(pending_key, tuple) and pending_key and pending_key[0] == provisional_identity_id:
                self.pending_semantic_slots.discard(pending_key)
        for attempt_key in list(self.next_semantic_attempt_frame):
            if isinstance(attempt_key, tuple) and attempt_key and attempt_key[0] == provisional_identity_id:
                self.next_semantic_attempt_frame.pop(attempt_key, None)
        for violation_key in list(self.physical_violation_counts):
            if isinstance(violation_key, tuple) and violation_key and violation_key[0] == provisional_identity_id:
                self.physical_violation_counts.pop(violation_key, None)

        # Appearance has just vouched for this group, so its held crops are
        # trustworthy.  The slots were copied onto the target above, so hand
        # the tasks over and flush against the target -- flushing against the
        # provisional would rewrite paths on slots nobody reads again.
        target.setdefault("deferred_evidence_tasks", []).extend(
            provisional.pop("deferred_evidence_tasks", [])
        )
        self._flush_deferred_evidence_locked(
            target,
            "merged_into_existing_master",
            final_identity_id=target_identity_id,
        )
        self.identities.pop(provisional_identity_id, None)
        identity_event(
            "provisional_global_reid_merged",
            provisional_master_id=(
                provisional_identity_id if provisional_identity_id > 0 else None
            ),
            temporary_group_id=(
                f"tmp_{abs(int(provisional_identity_id))}"
                if provisional_identity_id < 0
                else None
            ),
            master_id=target_identity_id,
            matched_track_key=matched_track_key,
            member_track_keys=sorted(member_keys, key=repr),
            matched_slot=matched_slot,
            distance=distance,
            distance_threshold=self.distance_threshold,
            feature_source=feature_source,
            feature_space_id=feature_space_id,
        )
        return target_identity_id

    def _baseline_cross_camera_comparisons_locked(self, identity_id):
        """Compare each camera's baseline against every other camera's.

        The stable-location shortcut never asks whether the two cameras are
        looking at the same person, only whether each is already known.  This
        supplies the missing question.  Returns ``(best_distance, comparisons,
        skipped)`` where ``best_distance`` is ``None`` when no two baselines
        were comparable at all.
        """

        record = self.identities.get(identity_id)
        if record is None:
            return None, [], []
        baselines = {
            str(camera_id): slot
            for camera_id, slot in (record.get("camera_baselines") or {}).items()
            if slot
        }
        comparisons = []
        skipped = []
        camera_ids = sorted(baselines)
        for left_index, left_camera in enumerate(camera_ids):
            for right_camera in camera_ids[left_index + 1 :]:
                left_slot = baselines[left_camera]
                right_slot = baselines[right_camera]
                left_space = left_slot.get("feature_space_id")
                right_space = right_slot.get("feature_space_id")
                if left_space != right_space:
                    skipped.append(
                        {
                            "left_camera": left_camera,
                            "right_camera": right_camera,
                            "reason": "feature_space_mismatch",
                            "left_feature_space_id": left_space,
                            "right_feature_space_id": right_space,
                        }
                    )
                    continue
                left_feature = self._normalize_feature(left_slot.get("feature"))
                right_feature = self._normalize_feature(right_slot.get("feature"))
                if (
                    left_feature is None
                    or right_feature is None
                    or left_feature.shape != right_feature.shape
                ):
                    skipped.append(
                        {
                            "left_camera": left_camera,
                            "right_camera": right_camera,
                            "reason": "feature_unusable",
                        }
                    )
                    continue
                distance = 1.0 - float(
                    np.dot(
                        np.asarray(left_feature, dtype=np.float64),
                        np.asarray(right_feature, dtype=np.float64),
                    )
                )
                comparisons.append(
                    {
                        "left_camera": left_camera,
                        "right_camera": right_camera,
                        "distance": distance,
                        "feature_space_id": left_space,
                        "left_feature_source": left_slot.get("feature_source"),
                        "right_feature_source": right_slot.get("feature_source"),
                        "left_sharpness": left_slot.get("sharpness"),
                        "right_sharpness": right_slot.get("sharpness"),
                        "left_detection_confidence": left_slot.get("detection_confidence"),
                        "right_detection_confidence": right_slot.get("detection_confidence"),
                        "left_captured_frame": left_slot.get("captured_frame"),
                        "right_captured_frame": right_slot.get("captured_frame"),
                        "left_image_path": left_slot.get("image_path"),
                        "right_image_path": right_slot.get("image_path"),
                    }
                )
        best_distance = (
            min(item["distance"] for item in comparisons) if comparisons else None
        )
        return best_distance, comparisons, skipped

    def _location_promotion_appearance_vetoed_locked(self, identity_id, reason):
        """Withhold a location-only promotion when the baselines clearly differ.

        Every outcome is logged, including the passing and non-comparable ones,
        so a veto that fires on a single person is visible in the trace rather
        than silently costing that person their ID.
        """

        record = self.identities.get(identity_id)
        if record is None:
            return False
        best_distance, comparisons, skipped = (
            self._baseline_cross_camera_comparisons_locked(identity_id)
        )
        member_keys = sorted(record.get("member_track_keys", ()), key=repr)
        shared = {
            "master_id": self._public_identity_id(identity_id),
            "temporary_group_id": self._temporary_group_token(identity_id),
            "promotion_reason": str(reason),
            "member_track_keys": member_keys,
            "veto_distance": self.provisional_baseline_veto_distance,
            "best_distance": best_distance,
            "comparisons": comparisons,
            "skipped_comparisons": skipped,
        }
        if best_distance is None:
            # Nothing comparable.  The gate cannot judge, so it must not block;
            # otherwise a feature-space change would strand every pair.
            identity_event(
                "location_promotion_baseline_check_skipped",
                console=False,
                throttle_key=(identity_id, "baseline_check_skipped"),
                throttle_seconds=1.0,
                reason="no_comparable_baselines",
                **shared,
            )
            return False
        if best_distance >= self.provisional_baseline_veto_distance:
            # TEMP_IDENTITY_DEBUG: console=True on purpose -- a veto against one
            # real person is the failure mode this gate must not hide.
            identity_event(
                "location_promotion_baseline_vetoed",
                reason="baseline_appearance_disagreement",
                **shared,
            )
            return True
        identity_event(
            "location_promotion_baseline_passed",
            console=False,
            reason="baseline_appearance_agreement",
            **shared,
        )
        return False

    def _member_contest_in_flight_locked(self, identity_id):
        """A contest one of this group's own tracks has raised and not yet won.

        Promotion and arbitration run on separate clocks, and unsynchronised
        they race: the group numbers itself while the contest that would have
        handed its member back to an existing master is still being judged.
        Arbitration is the better answer -- it is decided on appearance against
        the master itself, rather than on how long two boxes have agreed about
        where they are -- so promotion waits for the verdict.

        Bounded by the same patience the audit uses, so a contest that never
        concludes delays a number rather than withholding one forever.
        """

        record = self.identities.get(identity_id)
        if record is None:
            return None
        member_keys = set(record.get("member_track_keys", ()))
        if not member_keys:
            return None
        now = time.monotonic()
        for master_id, state in self.physical_conflicts.items():
            challenger_key = state.get("challenger_key")
            if challenger_key is None or challenger_key not in member_keys:
                continue
            age = now - float(state.get("started_monotonic") or now)
            if (
                self.identity_audit_contest_patience_seconds > 0.0
                and age > self.identity_audit_contest_patience_seconds
            ):
                continue
            return {
                "master_id": master_id,
                "token": state.get("token"),
                "age_seconds": age,
                "challenger_track_key": challenger_key,
            }
        return None

    def _promote_provisional_locked(self, identity_id, reason):
        record = self.identities.get(identity_id)
        if record is None or record.get("identity_state") == "confirmed":
            return None

        # Numbering this group would settle by fiat the very question a live
        # contest is in the middle of answering, and would answer it the wrong
        # way: a new master for someone who already has one.
        contest = self._member_contest_in_flight_locked(identity_id)
        if contest is not None:
            identity_event(
                "provisional_promotion_deferred",
                throttle_key=(identity_id, "member_contesting_master"),
                throttle_seconds=1.0,
                master_id=self._public_identity_id(identity_id),
                temporary_group_id=self._temporary_group_token(identity_id),
                reason="member_contesting_master",
                contested_master_id=contest["master_id"],
                contest_token=contest["token"],
                contest_age_seconds=contest["age_seconds"],
                challenger_track_key=contest["challenger_track_key"],
                patience_seconds=self.identity_audit_contest_patience_seconds,
            )
            return None

        if not self._provisional_global_reid_complete_locked(identity_id):
            identity_event(
                "provisional_promotion_deferred",
                throttle_key=(identity_id, "global_reid_incomplete"),
                throttle_seconds=1.0,
                master_id=identity_id,
                reason="global_reid_incomplete",
                checked_track_keys=sorted(
                    record.get("global_reid_checked_track_keys", ()),
                    key=repr,
                ),
                member_track_keys=sorted(
                    record.get("member_track_keys", ()),
                    key=repr,
                ),
            )
            return None

        # A group that appearance says belongs to an existing master, but whose
        # merge is currently blocked by a live same-camera owner, must wait
        # rather than become a second master for the same person.  The check is
        # live rather than sticky, so it clears by itself as soon as the
        # conflicting local track disappears and the merge can be retried.
        blocked_target = record.get("merge_blocked_by_master")
        if blocked_target is not None and blocked_target in self.identities:
            member_keys = sorted(record.get("member_track_keys", ()), key=repr)
            still_blocking = self._reject_second_visible_owner_locked(
                blocked_target,
                member_keys,
                [identity_id] * len(member_keys),
                "provisional_promotion_blocked",
            )
            if still_blocking:
                identity_event(
                    "provisional_promotion_deferred",
                    throttle_key=(identity_id, "merge_target_owner_visible"),
                    throttle_seconds=1.0,
                    master_id=self._public_identity_id(identity_id),
                    temporary_group_id=self._temporary_group_token(identity_id),
                    reason="merge_target_owner_visible",
                    blocked_merge_target_master_id=blocked_target,
                    member_track_keys=member_keys,
                )
                return None
            record.pop("merge_blocked_by_master", None)

        # Only the location-only path needs this.  ``same_angle_reid`` already
        # compared the two cameras and agreed, and re-judging that agreement on
        # cross-angle baselines would overturn the stronger evidence.
        if reason == "stable_location" and self._location_promotion_appearance_vetoed_locked(
            identity_id,
            reason,
        ):
            # Deferred, not challenged: the verdict is recomputed on the next
            # location match, so a sharper baseline can still clear the pair.
            return None

        candidates = list(record.get("camera_baselines", {}).values())
        for camera_gallery in record.get("camera_views", {}).values():
            candidates.extend(slot for slot in camera_gallery.values() if slot is not None)
        candidates = [slot for slot in candidates if slot and slot.get("feature") is not None]
        if not candidates:
            return None

        baseline = max(
            candidates,
            key=lambda slot: (
                float(slot.get("sharpness", 0.0)),
                float(slot.get("detection_confidence") or 0.0),
            ),
        )
        baseline_space = baseline.get("feature_space_id")
        record["gallery"]["baseline"] = dict(baseline)
        for slot_name in REID_SEMANTIC_SLOTS:
            semantic_candidates = [
                camera_gallery.get(slot_name)
                for camera_gallery in record.get("camera_views", {}).values()
                if camera_gallery.get(slot_name) is not None
                and camera_gallery.get(slot_name).get("feature_space_id") == baseline_space
            ]
            if semantic_candidates:
                record["gallery"][slot_name] = dict(
                    max(
                        semantic_candidates,
                        key=lambda slot: (
                            float(slot.get("sharpness", 0.0)),
                            float(slot.get("detection_confidence") or 0.0),
                        ),
                    )
                )

        temporary_group_id = identity_id if identity_id < 0 else None
        master_id = identity_id
        if temporary_group_id is not None:
            master_id = self.next_identity_id
            self.next_identity_id += 1
            self.identities[master_id] = record
            self.identities.pop(temporary_group_id, None)
            observations = self.recent_master_observations.pop(temporary_group_id, None)
            if observations is not None:
                self.recent_master_observations[master_id] = observations
            for key, mapped_identity_id in list(self.track_to_identity.items()):
                if mapped_identity_id == temporary_group_id:
                    self.track_to_identity[key] = master_id
            for state in self.pending_intake.values():
                if state.get("provisional_identity_id") == temporary_group_id:
                    state["provisional_identity_id"] = master_id

        record["identity_state"] = "confirmed"
        record["confirmation_reason"] = str(reason)
        # The group is now a real person, so its held crops may be written.
        self._flush_deferred_evidence_locked(
            record, f"promoted_{reason}", final_identity_id=master_id
        )
        for key in record.get("member_track_keys", ()):
            if self.track_to_identity.get(key) != master_id:
                continue
            metadata = self.track_binding_metadata.setdefault(key, {})
            metadata["identity_state"] = "confirmed"
            metadata["confirmation_reason"] = str(reason)
            metadata["appearance_confirmed"] = reason == "same_angle_reid"
            metadata["matched_feature_space_id"] = baseline_space
            metadata["temporary_group_id"] = None

        identity_event(
            "provisional_identity_promoted",
            temporary_group_id=(
                f"tmp_{abs(int(temporary_group_id))}"
                if temporary_group_id is not None
                else None
            ),
            master_id=master_id,
            reason=reason,
            location_match_frames=record.get("location_match_frames", 0),
        )
        return master_id

    def _confirm_pending_members_locked(
        self,
        identity_id,
        reason,
        appearance_confirmed,
        member_keys=None,
    ):
        record = self.identities.get(identity_id)
        if record is None:
            return False
        available_keys = set(record.get("pending_member_keys", ())) | set(
            record.get("challenged_member_keys", ())
        )
        if reason == "stable_location":
            # Standing in the right place for long enough is a fallback for
            # members appearance could not judge -- never an override for ones
            # it already judged and rejected.  Without this, a track that ReID
            # scored 0.458 against this master is still committed, and its crop
            # becomes part of the permanent gallery.
            rejected_keys = available_keys & set(
                record.get("appearance_rejected_member_keys", ())
            )
            if rejected_keys:
                identity_event(
                    "stable_location_confirmation_withheld",
                    throttle_key=(identity_id, "appearance_rejected"),
                    throttle_seconds=1.0,
                    master_id=self._public_identity_id(identity_id),
                    temporary_group_id=self._temporary_group_token(identity_id),
                    withheld_track_keys=sorted(rejected_keys, key=repr),
                    reason="appearance_already_rejected",
                )
            available_keys -= rejected_keys
        available_keys = {
            key
            for key in available_keys
            if self._camera_from_key(key) not in self.visible_track_keys_by_camera
            or key
            in self.visible_track_keys_by_camera.get(self._camera_from_key(key), set())
        }
        pending_keys = (
            available_keys
            if member_keys is None
            else available_keys & set(member_keys)
        )
        if not pending_keys:
            return False
        record.setdefault("pending_member_keys", set()).difference_update(pending_keys)
        record.setdefault("challenged_member_keys", set()).difference_update(pending_keys)
        for key in pending_keys:
            record.setdefault("pending_member_location_streaks", {}).pop(key, None)
            self._commit_pending_member_evidence_locked(identity_id, key)
        record["last_member_confirmation_reason"] = str(reason)
        baseline_space = (record.get("gallery", {}).get("baseline") or {}).get(
            "feature_space_id"
        )
        # Empty master slots are filled from what the cameras have stored, but
        # a slot that already holds a crop keeps it.  This used to put the
        # established crop into a sharpness contest against the newcomer's and
        # take the sharper -- which is how a crisp photograph of Haoran
        # replaced a softer, correct photograph of Mikail.  Sharpness cannot
        # tell two people apart, so it must never be the deciding vote on who
        # an identity looks like.
        for slot_name in REID_SEMANTIC_SLOTS:
            if record.get("gallery", {}).get(slot_name) is not None:
                continue
            candidates = [
                camera_gallery.get(slot_name)
                for camera_gallery in record.get("camera_views", {}).values()
                if camera_gallery.get(slot_name) is not None
                and camera_gallery.get(slot_name).get("feature_space_id") == baseline_space
            ]
            if candidates:
                record["gallery"][slot_name] = dict(
                    max(
                        candidates,
                        key=lambda slot: (
                            float(slot.get("sharpness", 0.0)),
                            float(slot.get("detection_confidence") or 0.0),
                        ),
                    )
                )
        for key in pending_keys:
            if self.track_to_identity.get(key) != identity_id:
                continue
            metadata = self.track_binding_metadata.setdefault(key, {})
            metadata["identity_state"] = "confirmed"
            metadata["confirmation_reason"] = str(reason)
            metadata["appearance_confirmed"] = bool(appearance_confirmed)
            metadata["matched_feature_space_id"] = baseline_space
            metadata["provisional_intake_complete"] = True
        identity_event(
            "provisional_members_confirmed",
            master_id=identity_id,
            member_track_keys=sorted(pending_keys, key=repr),
            reason=reason,
            appearance_confirmed=bool(appearance_confirmed),
        )
        return True

    def _evaluate_provisional_evidence_locked(self, identity_id):
        record = self.identities.get(identity_id)
        if record is None:
            return False
        confirmed_with_pending_members = (
            record.get("identity_state") == "confirmed"
            and bool(
                set(record.get("pending_member_keys", ()))
                | set(record.get("challenged_member_keys", ()))
            )
        )
        if record.get("identity_state") == "confirmed" and not confirmed_with_pending_members:
            return False
        # Comparing the two cameras' stored views against each other only
        # answers a question about someone with no record: are these two fresh
        # tracks one person?  Once a master gallery exists there is a direct
        # answer available -- compare the newcomer to that gallery -- and the
        # cross-camera route becomes a hazard, because a newcomer who supplies
        # no sharper crop is silently represented by the view already stored
        # for its camera, and confirmed on evidence that is not its own.  In
        # this run that route confirmed 1 member; the direct check confirmed 57.
        # The identity holding a master ID at all is the test, not whether its
        # gallery is complete: an unfinished gallery is exactly where a
        # newcomer's own crop is missing and a stored one would stand in.
        if confirmed_with_pending_members:
            identity_event(
                "same_angle_route_skipped",
                console=False,
                throttle_key=(identity_id, "same_angle_skipped"),
                throttle_seconds=5.0,
                master_id=self._public_identity_id(identity_id),
                pending_member_track_keys=sorted(
                    set(record.get("pending_member_keys", ()))
                    | set(record.get("challenged_member_keys", ())),
                    key=repr,
                ),
                reason="master_gallery_exists_use_global_reid",
            )
            return False
        camera_views = {
            str(camera_id): dict(camera_gallery)
            for camera_id, camera_gallery in record.get("camera_views", {}).items()
        }
        if confirmed_with_pending_members:
            for track_key, stage in self.pending_member_evidence.items():
                if (
                    stage.get("identity_id") != identity_id
                    or track_key
                    not in (
                        set(record.get("pending_member_keys", ()))
                        | set(record.get("challenged_member_keys", ()))
                    )
                ):
                    continue
                camera_id = str(stage.get("camera_id"))
                staged_gallery = camera_views.setdefault(
                    camera_id,
                    {slot_name: None for slot_name in REID_SEMANTIC_SLOTS},
                )
                for slot_name, slot in stage.get("views", {}).items():
                    if self._slot_is_better(slot, staged_gallery.get(slot_name)):
                        staged_gallery[slot_name] = slot
        camera_ids = sorted(camera_views)
        distances = []
        for left_index, left_camera in enumerate(camera_ids):
            for right_camera in camera_ids[left_index + 1 :]:
                for slot_name in REID_SEMANTIC_SLOTS:
                    left_slot = camera_views[left_camera].get(slot_name)
                    right_slot = camera_views[right_camera].get(slot_name)
                    if not left_slot or not right_slot:
                        continue
                    if left_slot.get("feature_space_id") != right_slot.get("feature_space_id"):
                        continue
                    left_feature = self._normalize_feature(left_slot.get("feature"))
                    right_feature = self._normalize_feature(right_slot.get("feature"))
                    if left_feature is None or right_feature is None or left_feature.shape != right_feature.shape:
                        continue
                    distance = 1.0 - float(
                        np.dot(
                            np.asarray(left_feature, dtype=np.float64),
                            np.asarray(right_feature, dtype=np.float64),
                        )
                    )
                    comparison_key = f"{left_camera}:{right_camera}:{slot_name}"
                    record["reid_comparisons"][comparison_key] = distance
                    distances.append((distance, left_camera, right_camera, slot_name))

        if not distances:
            return False
        best_distance, left_camera, right_camera, slot_name = min(distances)
        if best_distance < self.distance_threshold:
            identity_event(
                "provisional_reid_confirmed",
                master_id=identity_id,
                left_camera=left_camera,
                right_camera=right_camera,
                orientation=slot_name,
                distance=best_distance,
                distance_threshold=self.distance_threshold,
            )
            if confirmed_with_pending_members:
                evidence_member_keys = {
                    key
                    for key in (
                        set(record.get("pending_member_keys", ()))
                        | set(record.get("challenged_member_keys", ()))
                    )
                    if str(self._camera_from_key(key)) in (left_camera, right_camera)
                }
                return self._confirm_pending_members_locked(
                    identity_id,
                    "same_angle_reid",
                    appearance_confirmed=True,
                    member_keys=evidence_member_keys,
                )
            return self._promote_provisional_locked(identity_id, "same_angle_reid")

        if best_distance >= self.provisional_challenge_distance:
            if confirmed_with_pending_members:
                challenged_keys = {
                    key
                    for key in record.get("pending_member_keys", ())
                    if str(self._camera_from_key(key)) in (left_camera, right_camera)
                }
                record.setdefault("pending_member_keys", set()).difference_update(challenged_keys)
                record.setdefault("challenged_member_keys", set()).update(challenged_keys)
                for key in challenged_keys:
                    self._discard_pending_member_evidence_locked(
                        key,
                        "same_angle_reid_challenged",
                    )
                affected_keys = challenged_keys
            else:
                record["identity_state"] = "challenged"
                affected_keys = set(record.get("member_track_keys", ()))
            for key in affected_keys:
                metadata = self.track_binding_metadata.setdefault(key, {})
                metadata["identity_state"] = "challenged"
            identity_event(
                "provisional_reid_challenged",
                master_id=identity_id,
                left_camera=left_camera,
                right_camera=right_camera,
                orientation=slot_name,
                distance=best_distance,
                challenge_distance=self.provisional_challenge_distance,
            )
        else:
            identity_event(
                "provisional_reid_inconclusive",
                throttle_key=(identity_id, left_camera, right_camera, slot_name),
                throttle_seconds=1.0,
                master_id=identity_id,
                left_camera=left_camera,
                right_camera=right_camera,
                orientation=slot_name,
                distance=best_distance,
                distance_threshold=self.distance_threshold,
                challenge_distance=self.provisional_challenge_distance,
            )
        return False

    def note_location_match(self, identity_id, pair_streak, observations):
        """Record continued geometric agreement and apply the safe fallback."""

        promoted_identity_id = None
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None:
                return None
            record["location_managed"] = True
            # The coordinator's streak is already consecutive.  Store the
            # current value rather than the historical maximum so a camera
            # gap cannot cause an immediate stale promotion later.
            record["location_match_frames"] = int(pair_streak)
            matched_cameras = []
            currently_matched_pending_keys = set()
            for observation in observations:
                camera_id = observation.get("camera_id")
                local_track_id = observation.get("local_track_id")
                if camera_id is None or local_track_id is None:
                    continue
                key = self._track_key(local_track_id, camera_id)
                if self.track_to_identity.get(key) != identity_id:
                    continue
                matched_cameras.append(str(camera_id))
                record["member_track_keys"].add(key)
                if key in record.get("pending_member_keys", ()):
                    currently_matched_pending_keys.add(key)
                    record.setdefault("pending_member_location_streaks", {})[key] = int(
                        pair_streak
                    )
                self._record_master_observation_locked(
                    identity_id,
                    key,
                    observation.get("point"),
                    observation.get("captured_at"),
                )
            for left_camera in matched_cameras:
                for right_camera in matched_cameras:
                    if left_camera != right_camera:
                        self.physical_violation_counts.pop(
                            (identity_id, left_camera, right_camera),
                            None,
                        )
            if (
                record.get("identity_state") == "provisional"
                and record["location_match_frames"] >= self.provisional_location_confirm_frames
                and len(record.get("camera_baselines", {})) >= 2
            ):
                promoted_identity_id = self._promote_provisional_locked(
                    identity_id,
                    "stable_location",
                )
            elif (
                record.get("identity_state") == "confirmed"
                and bool(currently_matched_pending_keys)
            ):
                ready_member_keys = {
                    key
                    for key in currently_matched_pending_keys
                    if int(record.get("pending_member_location_streaks", {}).get(key, 0))
                    >= self.provisional_location_confirm_frames
                    and (
                        self.pending_member_evidence.get(key, {}).get("baseline")
                        is not None
                        or record.get("camera_baselines", {}).get(
                            str(self._camera_from_key(key))
                        )
                        is not None
                    )
                }
                if ready_member_keys:
                    promoted_identity_id = (
                        identity_id
                        if self._confirm_pending_members_locked(
                        identity_id,
                        "stable_location",
                        appearance_confirmed=False,
                        member_keys=ready_member_keys,
                        )
                        else None
                    )
            state = record.get("identity_state")
        if promoted_identity_id is not None:
            self._start_pending_demographics(promoted_identity_id)
            self.save_database(promoted_identity_id)
        return state

    def assignment_metadata(self, track_id, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            return dict(self.track_binding_metadata.get(key, {}))

    def pending_count(self, track_id, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            state = self.pending_intake.get(key)
            return len(state.get("samples", ())) if state else 0

    def required_intake_count(self):
        return self.intake_frames

    def gallery_status(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None:
                return 0, len(REID_GALLERY_SLOTS)
            gallery = record.get("gallery", {})
            return sum(gallery.get(slot) is not None for slot in REID_GALLERY_SLOTS), len(REID_GALLERY_SLOTS)

    def identity_metadata(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None:
                return {}
            baseline = record.get("gallery", {}).get("baseline") or {}
            identity_state = record.get("identity_state", "confirmed")
            if identity_state in ("provisional", "challenged"):
                gallery_filled = sum(
                    slot is not None
                    for camera_gallery in record.get("camera_views", {}).values()
                    for slot in camera_gallery.values()
                )
                gallery_total = 2 * len(REID_SEMANTIC_SLOTS)
            else:
                gallery_filled = sum(
                    record.get("gallery", {}).get(slot) is not None
                    for slot in REID_GALLERY_SLOTS
                )
                gallery_total = len(REID_GALLERY_SLOTS)
            return {
                "identity_state": identity_state,
                "confirmation_reason": record.get("confirmation_reason"),
                "role": record.get("role", "evacuee"),
                "age": record.get("age", "Unknown"),
                "gender": record.get("gender", "Unknown"),
                "gallery_filled": gallery_filled,
                "gallery_total": gallery_total,
                "feature_source": baseline.get("feature_source"),
            }

    def semantic_probe_due(self, track_id, crop, frame_index, detection_confidence, camera_id=None):
        key = self._track_key(track_id, camera_id)
        with self._lock:
            identity_id = self.track_to_identity.get(key)
            if identity_id is None:
                return False
            record = self.identities.get(identity_id)
            if record is None:
                return False
            identity_state = record.get("identity_state", "confirmed")
            camera_key = self._camera_from_key(key)
            camera_specific = (
                identity_state in ("provisional", "challenged")
                or bool(record.get("location_managed"))
            )
            if camera_specific:
                camera_gallery = record.get("camera_views", {}).get(camera_key, {})
                gallery_complete = all(camera_gallery.get(slot) is not None for slot in REID_SEMANTIC_SLOTS)
            else:
                gallery_complete = all(
                    record.get("gallery", {}).get(slot) is not None
                    for slot in REID_SEMANTIC_SLOTS
                )
            if gallery_complete:
                return False
            semantic_clock_key = (identity_id, camera_key)
            if int(frame_index) < int(self.next_semantic_attempt_frame.get(semantic_clock_key, 0)):
                return False
        if detection_confidence is None or float(detection_confidence) <= self.semantic_confidence_threshold:
            with self._lock:
                self.next_semantic_attempt_frame[semantic_clock_key] = (
                    int(frame_index) + self.semantic_retry_frames
                )
            return False
        sharpness = image_sharpness(crop)
        with self._lock:
            if self.track_to_identity.get(key) != identity_id:
                return False
            if sharpness <= self.blur_threshold:
                self.next_semantic_attempt_frame[semantic_clock_key] = (
                    int(frame_index) + self.semantic_retry_frames
                )
                return False
            self.semantic_probe_quality[(key, int(frame_index))] = sharpness
        return True

    def _queue_task_locked(self, task):
        if self._worker is None:
            self._process_task(task)
            return True
        try:
            self._task_queue.put_nowait(task)
            return True
        except queue.Full:
            if self.verbose:
                print("ReID analyst queue is full; task will be retried.")
            return False

    def _schedule_semantic_locked(
        self,
        key,
        identity_id,
        crop,
        frame_index,
        orientation,
        detection_confidence,
        observed_at,
        body_complete=None,
    ):
        record = self.identities.get(identity_id)
        if record is None:
            return
        identity_state = record.get("identity_state", "confirmed")
        camera_id = self._camera_from_key(key)
        camera_specific = (
            identity_state in ("provisional", "challenged")
            or bool(record.get("location_managed"))
        )
        gallery = (
            record.setdefault("camera_views", {}).setdefault(
                camera_id,
                {slot: None for slot in REID_SEMANTIC_SLOTS},
            )
            if camera_specific
            else record.get("gallery", {})
        )
        if all(gallery.get(slot) is not None for slot in REID_SEMANTIC_SLOTS):
            return
        semantic_clock_key = (identity_id, camera_id)
        if int(frame_index) < int(self.next_semantic_attempt_frame.get(semantic_clock_key, 0)):
            return

        sharpness = self.semantic_probe_quality.pop((key, int(frame_index)), None)
        if sharpness is None:
            sharpness = image_sharpness(crop)
        if (
            detection_confidence is None
            or float(detection_confidence) <= self.semantic_confidence_threshold
            or sharpness <= self.blur_threshold
            or orientation not in REID_SEMANTIC_SLOTS
            or (camera_specific and body_complete is not True)
        ):
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_retry_frames
            return
        pending_key = (
            (identity_id, camera_id, orientation)
            if camera_specific
            else (identity_id, orientation)
        )
        if gallery.get(orientation) is not None or pending_key in self.pending_semantic_slots:
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_retry_frames
            return

        task = {
            "type": "provisional_semantic" if camera_specific else "semantic",
            "track_key": key,
            "identity_id": identity_id,
            "slot_name": orientation,
            "sample": {
                "crop": crop.copy(),
                "frame_index": int(frame_index),
                "camera_id": self._camera_from_key(key),
                "observed_at": float(observed_at),
                "sharpness": sharpness,
                "area": int(crop.shape[0] * crop.shape[1]),
                "detection_confidence": float(detection_confidence),
                "orientation": orientation,
            },
        }
        if self._queue_task_locked(task):
            self.pending_semantic_slots.add(pending_key)
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_cooldown_frames
        else:
            self.next_semantic_attempt_frame[semantic_clock_key] = int(frame_index) + self.semantic_retry_frames

    def assign(
        self,
        track_id,
        crop,
        frame_index,
        precomputed_feature=None,
        excluded_identity_ids=None,
        camera_id=None,
        detection_confidence=None,
        orientation=None,
        observed_at=None,
        map_point=None,
        map_point_evidence="hard",
        map_point_evidence_reason=None,
        intake_body_complete=None,
        intake_missing_regions=None,
        intake_body_details=None,
        intake_detection_box=None,
        intake_face_box=None,
        intake_body_bounds=None,
        intake_occluder_boxes=(),
    ):
        """Report which master identity a local track belongs to, this frame.

        Called once per detection per frame, so the common cases are kept
        cheap.  The work falls into three phases, one helper each:

        1. Housekeeping -- expire a binding the track has been away from too
           long, and resolve whether this track is a duplicate "shadow" of an
           older one that already owns the intake.
        2. ``_resolve_bound_identity_locked`` -- if the track already holds an
           identity, either service it and finish, or release it.
        3. ``_collect_intake_crop_locked`` -- otherwise offer this crop to the
           track's intake burst, which earns it an identity once enough good
           crops have arrived.

        Returns ``(master_id, similarity, reidentified)``.  ``master_id`` is
        always the public form: a provisional group is an internal negative id
        and must leave here as None.
        """
        del precomputed_feature
        key = self._track_key(track_id, camera_id)
        # Recorded per track so every later physical check in this frame -- and
        # the stored master observation other cameras compare against -- knows
        # whether this ground point was actually measured or merely implied.
        self._remember_position_evidence(key, map_point_evidence, map_point_evidence_reason)
        now = time.monotonic() if observed_at is None else float(observed_at)
        if crop is None or crop.size == 0:
            # Published like every other exit: a temporary group is an internal
            # negative id and must leave as ``None``.  Returning it raw let a
            # crop rejected by the overlap gate hand callers a "master" that
            # then read the group's default evacuee role and displaced its
            # ``temporary`` fusion association key.
            with self._lock:
                return self._public_identity_id(self.track_to_identity.get(key)), 0.0, False

        with self._lock:
            previous_seen = self.track_last_seen.get(key)
            if previous_seen is not None:
                frame_gap = int(frame_index) - int(previous_seen[0])
                if frame_gap < 0 or frame_gap > self.ttl_frames:
                    previous_identity_id = self.track_to_identity.get(key)
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "track_binding_reset",
                        track_key=key,
                        master_id=previous_identity_id,
                        camera_id=camera_id,
                        frame_index=frame_index,
                        previous_frame_index=previous_seen[0],
                        frame_gap=frame_gap,
                        ttl_frames=self.ttl_frames,
                        reason="frame_rewind" if frame_gap < 0 else "frame_gap_exceeded_ttl",
                    )
                    self._clear_local_binding_locked(key)
            self.track_last_seen[key] = (int(frame_index), now)
            self._refresh_physical_conflict_recovery_hold_locked(key, frame_index)

            handoff_identity_id = None
            handoff_from_key = None
            shadow = self.shadow_tracks.get(key)
            if shadow is not None:
                handoff_from_key = shadow.get("canonical_key")
                camera_key = self._camera_from_key(key)
                visible_keys = self.visible_track_keys_by_camera.get(camera_key, set())
                canonical_visible = handoff_from_key in visible_keys
                mapped_target = self.track_to_identity.get(handoff_from_key)
                handoff_identity_id = shadow.get("identity_id")

                if mapped_target in self.identities:
                    if handoff_identity_id is not None and handoff_identity_id != mapped_target:
                        self._release_shadow_locked(key, reason="canonical_master_changed_during_assign")
                        shadow = None
                        handoff_identity_id = None
                        handoff_from_key = None
                    else:
                        handoff_identity_id = mapped_target
                        shadow["identity_id"] = mapped_target
                        shadow["provisional"] = False
                elif handoff_identity_id is None:
                    if canonical_visible and handoff_from_key in self.pending_intake:
                        # The older overlapping track has not produced its
                        # master yet. It owns the only intake until that race
                        # resolves, regardless of how long both boxes persist.
                        return None, 0.0, False
                    self._release_shadow_locked(key, reason="provisional_canonical_no_longer_pending")
                    shadow = None
                    handoff_from_key = None
                elif not shadow.get("verified"):
                    self._release_shadow_locked(key, reason="unverified_target_binding_missing")
                    shadow = None
                    handoff_identity_id = None
                    handoff_from_key = None

                if shadow is not None and shadow.get("verified"):
                    if canonical_visible:
                        # One successful five-crop comparison is enough. Keep
                        # the duplicate hidden without repeating GPU work.
                        return None, 0.0, False
                    promoted_identity_id = self._promote_verified_shadow_locked(key, shadow)
                    if promoted_identity_id is None:
                        # Another live same-camera owner appeared between the
                        # frame observation and this assignment. Preserve the
                        # single-owner invariant and wait for the next frame.
                        return None, 0.0, False
                    handoff_identity_id = promoted_identity_id
                    shadow = None
                elif (
                    shadow is not None
                    and canonical_visible
                    and int(shadow.get("overlap_frames", 0)) <= self.shadow_probation_frames
                ):
                    # The first few overlap frames are treated as detector
                    # noise. A persistent candidate earns one appearance test
                    # only after this cheap probation has elapsed.
                    return None, 0.0, False

            identity_id = self.track_to_identity.get(key)
            if (
                identity_id is not None
                and identity_id < 0
                and self._provisional_split_recovery_expired_locked(identity_id)
            ):
                self._dissolve_provisional_split_locked(
                    identity_id,
                    "recovery_window_expired",
                )
                identity_id = self.track_to_identity.get(key)
            (
                bound_answer,
                identity_id,
                provisional_identity_id,
            ) = self._resolve_bound_identity_locked(
                key,
                crop,
                frame_index,
                camera_id,
                now,
                map_point,
                orientation,
                detection_confidence,
                identity_id,
                intake_body_complete,
                intake_face_box,
                intake_body_bounds,
                intake_occluder_boxes,
            )
            if bound_answer is not None:
                return bound_answer
            if identity_id is not None:
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "track_binding_reset",
                    track_key=key,
                    master_id=identity_id,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    reason="master_record_missing",
                )
                self._clear_local_binding_locked(key)

            return self._collect_intake_crop_locked(
                key,
                crop,
                frame_index,
                camera_id,
                now,
                map_point,
                orientation,
                detection_confidence,
                excluded_identity_ids,
                handoff_identity_id,
                handoff_from_key,
                provisional_identity_id,
                intake_body_complete,
                intake_missing_regions,
                intake_body_details,
                intake_detection_box,
                intake_face_box,
                intake_body_bounds,
                intake_occluder_boxes,
            )

    def _resolve_bound_identity_locked(
        self,
        key,
        crop,
        frame_index,
        camera_id,
        now,
        map_point,
        orientation,
        detection_confidence,
        identity_id,
        intake_body_complete,
        intake_face_box,
        intake_body_bounds,
        intake_occluder_boxes,
    ):
        """Service a track that already holds an identity, or let go of it.

        A track arrives here still bound to the master it had last frame.  That
        binding is only kept if the person is still physically where the master
        is: a track whose ground point has drifted impossibly far is released
        rather than allowed to drag the master across the map.  A binding that
        survives gets its gallery, semantic and audit work scheduled here, and
        the caller is done for this frame.

        Returns ``(answer, identity_id, provisional_identity_id)``.  ``answer``
        is the tuple ``assign`` should return, or None to carry on into the
        intake burst -- and because this block reassigns both identity values,
        they are handed back rather than mutated in the caller's scope.

        Runs with ``self._lock`` already held by ``assign``.
        """
        provisional_identity_id = None
        if identity_id is None or identity_id not in self.identities:
            # Nothing bound, or bound to a master that no longer exists.
            # Reporting and clearing that dangling binding is the caller's.
            return None, identity_id, provisional_identity_id

        record = self.identities[identity_id]
        track_identity_state = self._track_identity_state_locked(record, key)
        if track_identity_state in ("provisional", "challenged"):
            if not self._physical_match_allowed_locked(
                identity_id,
                self._camera_from_key(key),
                map_point,
                now,
                track_key=key,
                established_binding=True,
            ):
                revoked_identity_id = identity_id
                self._clear_local_binding_locked(key)
                identity_event(
                    "provisional_binding_revoked",
                    track_key=key,
                    master_id=revoked_identity_id,
                    camera_id=self._camera_from_key(key),
                    frame_index=frame_index,
                    map_point=map_point,
                    observed_at=now,
                    reason="repeated_physical_mismatch",
                )
                self._start_provisional_split_recovery_locked(
                    revoked_identity_id,
                    key,
                    frame_index,
                    now,
                )
                identity_id = None
                track_identity_state = None
            else:
                provisional_identity_id = identity_id
        if provisional_identity_id is not None:
            record["hits"] = int(record.get("hits", 0)) + 1
            record["last_seen_monotonic"] = now
            record.setdefault("member_track_keys", set()).add(key)
            self._record_master_observation_locked(identity_id, key, map_point, now)
            metadata = self.track_binding_metadata.setdefault(key, {})
            metadata["identity_state"] = track_identity_state
            metadata["confirmation_reason"] = record.get("confirmation_reason")
            metadata["appearance_confirmed"] = False
            if metadata.get("provisional_intake_complete"):
                self._schedule_semantic_locked(
                    key,
                    identity_id,
                    crop,
                    frame_index,
                    orientation,
                    detection_confidence,
                    now,
                    body_complete=intake_body_complete,
                )
                return (
                    self._public_identity_id(identity_id),
                    1.0,
                    False,
                ), identity_id, provisional_identity_id
            # Continue through the normal quality-controlled intake,
            # but its worker will add evidence to this reserved ID
            # instead of matching/creating an independent master.
            identity_id = None
        elif identity_id is not None and not self._physical_match_allowed_locked(
            identity_id,
            self._camera_from_key(key),
            map_point,
            now,
            track_key=key,
            frame_index=frame_index,
            defer_bound_conflict=True,
            established_binding=True,
        ):
            revoked_identity_id = identity_id
            camera_observations = self.recent_master_observations.get(identity_id, {})
            camera_observations.pop(str(self._camera_from_key(key)), None)
            self._clear_local_binding_locked(key)
            identity_id = None
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "track_binding_revoked",
                track_key=key,
                master_id=revoked_identity_id,
                camera_id=self._camera_from_key(key),
                map_point=map_point,
                observed_at=now,
                frame_index=frame_index,
                reason="physical_match_rejected",
            )
            if self.verbose:
                print(
                    f"ReID: revoked {key} -> Master {revoked_identity_id} "
                    "after an impossible map jump"
                )
        elif identity_id is not None:
            answer, identity_id = self._service_bound_master_locked(
                key,
                crop,
                frame_index,
                now,
                map_point,
                orientation,
                detection_confidence,
                identity_id,
                intake_body_complete,
                intake_face_box,
                intake_body_bounds,
                intake_occluder_boxes,
            )
            return answer, identity_id, provisional_identity_id

        return None, identity_id, provisional_identity_id

    def _service_bound_master_locked(
        self,
        key,
        crop,
        frame_index,
        now,
        map_point,
        orientation,
        detection_confidence,
        identity_id,
        intake_body_complete,
        intake_face_box,
        intake_body_bounds,
        intake_occluder_boxes,
    ):
        """Do this frame's work for a track still holding a confirmed master.

        The binding has already survived the physical check, so the person is
        where the master says they are.  What remains is to record the sighting
        and schedule the optional per-frame work: a semantic gallery slot, an
        identity audit, and a demographics re-estimate when this frame happens
        to carry a face box.  None of that may run while the master is contested
        -- until appearance names the rightful owner, either claimant would be
        writing into someone else's gallery.

        Returns ``(answer, identity_id)``; every path here is terminal for the
        frame.  ``identity_id`` comes back None when the binding was lost while
        the conflict sample was being collected.

        Runs with ``self._lock`` already held.
        """
        self._collect_physical_conflict_sample_locked(
            identity_id,
            key,
            crop,
            frame_index,
            detection_confidence,
            now,
            map_point,
            intake_body_complete,
        )
        # A synchronous test worker may resolve the conflict while
        # collecting this sample. Never restore the losing key.
        if self.track_to_identity.get(key) != identity_id:
            return None, None

        record = self.identities[identity_id]
        record["hits"] = int(record.get("hits", 0)) + 1
        record["last_seen_monotonic"] = now
        self._record_master_observation_locked(identity_id, key, map_point, now)
        result = self.track_results.pop(key, None)
        # Do not update a master gallery from either claimant
        # until appearance has chosen the rightful owner.
        if identity_id not in self.physical_conflicts:
            self._schedule_semantic_locked(
                key,
                identity_id,
                crop,
                frame_index,
                orientation,
                detection_confidence,
                now,
                body_complete=intake_body_complete,
            )
            self._schedule_identity_audit_locked(
                key,
                identity_id,
                crop,
                frame_index,
                detection_confidence,
                now,
                intake_body_complete,
            )
            # Only frames that already ran MediaPipe carry a
            # face box, so this costs a sharpness measurement
            # at the semantic probe rate, not once per frame.
            if intake_face_box is not None and self.enable_demographics:
                refresh_pool = self._consider_demographics_refresh_locked(
                    identity_id,
                    [
                        {
                            "crop": crop,
                            "sharpness": image_sharpness(crop),
                            "face_box": intake_face_box,
                            "body_bounds": intake_body_bounds,
                            "occluder_boxes": intake_occluder_boxes,
                            "camera_id": self._camera_from_key(key),
                            "frame_index": int(frame_index),
                        }
                    ],
                )
                if refresh_pool:
                    self._queue_demographics(
                        identity_id,
                        refresh_pool,
                        "closer_view",
                    )
        if result is not None:
            return (
                identity_id,
                result["similarity"],
                result["reidentified"],
            ), identity_id
        return (
            identity_id,
            1.0,
            False,
        ), identity_id

    def _collect_intake_crop_locked(
        self,
        key,
        crop,
        frame_index,
        camera_id,
        now,
        map_point,
        orientation,
        detection_confidence,
        excluded_identity_ids,
        handoff_identity_id,
        handoff_from_key,
        provisional_identity_id,
        intake_body_complete,
        intake_missing_regions,
        intake_body_details,
        intake_detection_box,
        intake_face_box,
        intake_body_bounds,
        intake_occluder_boxes,
    ):
        """Offer one crop to this track's intake burst, and submit when full.

        A new track earns an identity only after contributing a fixed number of
        crops that pass every quality gate below.  The gates are deliberately a
        flat sequence rather than nested conditions: each one rejects the crop
        for a single, separately logged reason, and every rejection leaves the
        caller with the same answer -- no master yet, no similarity, not
        reidentified -- because a rejected crop is simply a frame that did not
        count.

        Runs with ``self._lock`` already held by ``assign``.
        """
        state = self.pending_intake.get(key)
        if state is None:
            state = {
                "first_seen": now,
                "last_frame": None,
                "samples": [],
                "submitted": False,
                "generation": self._next_track_generation_locked(key),
                "next_retry_frame": int(frame_index),
                "failure_count": 0,
                "handoff_identity_id": handoff_identity_id,
                "handoff_from_key": handoff_from_key,
                "provisional_identity_id": provisional_identity_id,
            }
            self.pending_intake[key] = state
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "track_intake_started",
                track_key=key,
                camera_id=camera_id,
                frame_index=frame_index,
                observed_at=now,
                map_point=map_point,
                handoff_target_master_id=handoff_identity_id,
                handoff_from_track_key=handoff_from_key,
            )
        if provisional_identity_id is not None:
            state["provisional_identity_id"] = provisional_identity_id
        if state["submitted"]:
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "intake_crop_skipped",
                console=False,
                throttle_key=(key, "batch_submitted"),
                throttle_seconds=1.0,
                track_key=key,
                camera_id=camera_id,
                frame_index=frame_index,
                reason="batch_already_submitted",
                accepted_sample_count=len(state["samples"]),
                required_sample_count=self.intake_frames,
                generation=state.get("generation"),
                provisional_identity_id=state.get("provisional_identity_id"),
            )
            return self._public_identity_id(provisional_identity_id), 0.0, False
        if now - float(state["first_seen"]) < self.intake_delay_seconds:
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "intake_crop_skipped",
                console=False,
                throttle_key=(key, "intake_delay"),
                throttle_seconds=1.0,
                track_key=key,
                camera_id=camera_id,
                frame_index=frame_index,
                reason="intake_delay_not_elapsed",
                seconds_since_first_seen=now - float(state["first_seen"]),
                intake_delay_seconds=self.intake_delay_seconds,
                accepted_sample_count=len(state["samples"]),
                required_sample_count=self.intake_frames,
                generation=state.get("generation"),
            )
            return self._public_identity_id(provisional_identity_id), 0.0, False
        if state["last_frame"] == int(frame_index):
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "intake_crop_skipped",
                console=False,
                throttle_key=(key, "duplicate_frame"),
                throttle_seconds=1.0,
                track_key=key,
                camera_id=camera_id,
                frame_index=frame_index,
                reason="frame_already_sampled",
                accepted_sample_count=len(state["samples"]),
                required_sample_count=self.intake_frames,
                generation=state.get("generation"),
            )
            return self._public_identity_id(provisional_identity_id), 0.0, False

        # A timeout may relax the blur gate, but it must never turn a
        # partial-body image into the permanent ReID baseline.
        if intake_body_complete is False:
            state["last_frame"] = int(frame_index)
            missing_regions = tuple(intake_missing_regions or ())
            previous_rejection = state.get("last_body_rejection")
            should_log = (
                previous_rejection is None
                or previous_rejection[1] != missing_regions
                or int(frame_index) - int(previous_rejection[0]) >= 30
            )
            if should_log:
                # TEMP_IDENTITY_DEBUG
                identity_event(
                    "intake_crop_rejected",
                    track_key=key,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    observed_at=now,
                    reason="missing_body_parts",
                    missing_regions=missing_regions,
                )
                identity_event(
                    "intake_crop_rejected_detail",
                    console=False,
                    track_key=key,
                    camera_id=camera_id,
                    frame_index=frame_index,
                    generation=state.get("generation"),
                    reason="missing_body_parts",
                    missing_regions=missing_regions,
                    crop_shape=tuple(int(value) for value in crop.shape),
                    detection_box=intake_detection_box,
                    detection_confidence=detection_confidence,
                    orientation=orientation,
                    map_point=map_point,
                    body_details=intake_body_details,
                )
                state["last_body_rejection"] = (int(frame_index), missing_regions)
            return self._public_identity_id(provisional_identity_id), 0.0, False
        state.pop("last_body_rejection", None)

        sharpness = image_sharpness(crop)
        timed_out = now - float(state["first_seen"]) >= self.intake_timeout_seconds
        if sharpness <= self.blur_threshold and not timed_out:
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "intake_crop_rejected",
                console=False,
                throttle_key=(key, "blur"),
                throttle_seconds=1.0,
                track_key=key,
                camera_id=camera_id,
                frame_index=frame_index,
                observed_at=now,
                reason="too_blurry",
                sharpness=sharpness,
                blur_threshold=self.blur_threshold,
                seconds_since_first_seen=now - float(state["first_seen"]),
                intake_timeout_seconds=self.intake_timeout_seconds,
                accepted_sample_count=len(state["samples"]),
                required_sample_count=self.intake_frames,
                crop_shape=tuple(int(value) for value in crop.shape),
                detection_confidence=detection_confidence,
                orientation=orientation,
                generation=state.get("generation"),
            )
            return self._public_identity_id(provisional_identity_id), 0.0, False

        state["last_frame"] = int(frame_index)
        state["samples"].append(
            {
                "crop": crop.copy(),
                "frame_index": int(frame_index),
                "camera_id": camera_id,
                "observed_at": now,
                "sharpness": sharpness,
                "area": int(crop.shape[0] * crop.shape[1]),
                "detection_confidence": None if detection_confidence is None else float(detection_confidence),
                "detection_box": (
                    None
                    if intake_detection_box is None
                    else tuple(float(value) for value in intake_detection_box)
                ),
                "orientation": orientation if orientation in REID_SEMANTIC_SLOTS else None,
                "map_point": None if map_point is None else tuple(map(float, map_point)),
                "body_complete": intake_body_complete,
                "body_details": copy.deepcopy(intake_body_details),
                # Framing for MiVOLO, all as fractions of this crop so it
                # survives being stored and copied away from its frame.
                "face_box": (
                    None
                    if intake_face_box is None
                    else tuple(float(value) for value in intake_face_box)
                ),
                "body_bounds": (
                    None
                    if intake_body_bounds is None
                    else tuple(float(value) for value in intake_body_bounds)
                ),
                "occluder_boxes": tuple(
                    tuple(float(value) for value in occluder)
                    for occluder in (intake_occluder_boxes or ())
                ),
            }
        )
        if len(state["samples"]) > self.intake_frames:
            state["samples"] = state["samples"][-self.intake_frames :]
        if len(state["samples"]) < self.intake_frames:
            identity_event(
                "intake_crop_accepted",
                console=False,
                track_key=key,
                camera_id=camera_id,
                frame_index=frame_index,
                generation=state.get("generation"),
                accepted_sample_count=len(state["samples"]),
                required_sample_count=self.intake_frames,
                provisional_identity_id=state.get("provisional_identity_id"),
                sample=self._sample_debug_summary(state["samples"][-1]),
            )
            return self._public_identity_id(provisional_identity_id), 0.0, False
        identity_event(
            "intake_crop_accepted",
            console=False,
            track_key=key,
            camera_id=camera_id,
            frame_index=frame_index,
            generation=state.get("generation"),
            accepted_sample_count=len(state["samples"]),
            required_sample_count=self.intake_frames,
            provisional_identity_id=state.get("provisional_identity_id"),
            sample=self._sample_debug_summary(state["samples"][-1]),
        )
        if int(frame_index) < int(state.get("next_retry_frame", 0)):
            return self._public_identity_id(provisional_identity_id), 0.0, False

        camera_key = self._camera_from_key(key)
        visible_peer_keys = set(self.visible_track_keys_by_camera.get(camera_key, set()))
        visible_peer_keys.discard(key)

        task = {
            "type": "intake",
            "track_key": key,
            "camera_id": camera_id,
            "frame_index": int(frame_index),
            "samples": [
                {**sample, "crop": sample["crop"].copy()}
                for sample in state["samples"][: self.intake_frames]
            ],
            "excluded_identity_ids": (
                set(excluded_identity_ids or ())
                | set(self.physical_conflict_rejections.get(key, ()))
            ),
            "same_camera_peer_keys": visible_peer_keys,
            "generation": state["generation"],
            "handoff_identity_id": state.get("handoff_identity_id"),
            "handoff_from_key": state.get("handoff_from_key"),
            "provisional_identity_id": state.get("provisional_identity_id"),
        }
        if self._queue_task_locked(task):
            state["submitted"] = True
            identity_event(
                "intake_batch_submitted",
                console=False,
                track_key=key,
                camera_id=camera_id,
                frame_index=frame_index,
                generation=state.get("generation"),
                provisional_identity_id=state.get("provisional_identity_id"),
                sample_count=len(task["samples"]),
                samples=[
                    self._sample_debug_summary(sample)
                    for sample in task["samples"]
                ],
            )
        else:
            state["next_retry_frame"] = int(frame_index) + self.intake_retry_frames
        return self._public_identity_id(provisional_identity_id), 0.0, False

    def _store_provisional_intake_locked(
        self,
        identity_id,
        task,
        samples,
        features,
        valid_indices,
        hero_index,
        feature_source,
        feature_space_id,
        role,
        role_confidence,
    ):
        key = task["track_key"]
        camera_id = str(task.get("camera_id"))
        record = self.identities.get(identity_id)
        track_identity_state = (
            self._track_identity_state_locked(record, key)
            if record is not None
            else None
        )
        if (
            record is None
            or (
                track_identity_state not in ("provisional", "challenged")
                and not (
                    track_identity_state == "confirmed"
                    and record.get("location_managed")
                )
            )
            or self.track_to_identity.get(key) != identity_id
        ):
            return None, None

        self._apply_role_vote_locked(record, role, role_confidence)

        new_provisional_record = record.get("identity_state") in (
            "provisional",
            "challenged",
        )
        target_confirmation = not new_provisional_record
        best_semantic_samples = {}
        for index in valid_indices:
            slot_name = samples[index].get("orientation")
            if slot_name not in REID_SEMANTIC_SLOTS:
                continue
            previous = best_semantic_samples.get(slot_name)
            if previous is None or self._quality_score(samples[index]) > self._quality_score(samples[previous]):
                best_semantic_samples[slot_name] = index

        if target_confirmation:
            baseline_slot, baseline_task = self._make_slot(
                identity_id,
                f"baseline_{camera_id}",
                features[hero_index],
                samples[hero_index],
                feature_source,
                feature_space_id,
                track_key=key,
            )
            staged_semantic_slots = {}
            for slot_name, index in best_semantic_samples.items():
                staged_semantic_slots[slot_name] = self._make_slot(
                    identity_id,
                    f"{camera_id}_{slot_name}",
                    features[index],
                    samples[index],
                    feature_source,
                    feature_space_id,
                    track_key=key,
                )
            self._stage_pending_member_evidence_locked(
                identity_id,
                key,
                camera_id,
                baseline_slot,
                baseline_task,
                staged_semantic_slots,
            )
            identity_event(
                "baseline_selected",
                console=False,
                track_key=key,
                camera_id=camera_id,
                master_id=identity_id,
                temporary_group_id=(
                    f"tmp_{abs(int(identity_id))}" if identity_id < 0 else None
                ),
                slot_name=f"baseline_{camera_id}",
                baseline_state="pending_member_confirmation",
                feature_source=feature_source,
                feature_space_id=feature_space_id,
                selection_rule="maximum_sharpness_times_square_root_area",
                evidence_path=baseline_slot.get("image_path"),
                selected_sample=self._sample_debug_summary(samples[hero_index]),
            )
        else:
            camera_baselines = record.setdefault("camera_baselines", {})
            if camera_id not in camera_baselines:
                slot, evidence_task = self._make_slot(
                    identity_id,
                    f"baseline_{camera_id}",
                    features[hero_index],
                    samples[hero_index],
                    feature_source,
                    feature_space_id,
                    track_key=key,
                )
                camera_baselines[camera_id] = slot
                # An unpromoted group is only a geometric guess: two people who
                # merely walked close together for a few frames share it.  The
                # feature must stay in memory so the group can verify itself,
                # but the PNG is withheld until promotion, so an unverified
                # guess can never put one person's photo in another's folder.
                self._defer_provisional_evidence_locked(record, evidence_task)
                identity_event(
                    "baseline_selected",
                    console=False,
                    track_key=key,
                    camera_id=camera_id,
                    master_id=identity_id if identity_id > 0 else None,
                    temporary_group_id=(
                        f"tmp_{abs(int(identity_id))}" if identity_id < 0 else None
                    ),
                    slot_name=f"baseline_{camera_id}",
                    baseline_state="provisional_camera_baseline",
                    feature_source=feature_source,
                    feature_space_id=feature_space_id,
                    selection_rule="maximum_sharpness_times_square_root_area",
                    evidence_path=slot.get("image_path"),
                    selected_sample=self._sample_debug_summary(samples[hero_index]),
                )

            camera_gallery = record.setdefault("camera_views", {}).setdefault(
                camera_id,
                {slot_name: None for slot_name in REID_SEMANTIC_SLOTS},
            )
            for slot_name, index in best_semantic_samples.items():
                if camera_gallery.get(slot_name) is not None:
                    continue
                slot, evidence_task = self._make_slot(
                    identity_id,
                    f"{camera_id}_{slot_name}",
                    features[index],
                    samples[index],
                    feature_source,
                    feature_space_id,
                    track_key=key,
                )
                camera_gallery[slot_name] = slot
                self._defer_provisional_evidence_locked(record, evidence_task)

        record.setdefault("member_track_keys", set()).add(key)
        record["last_seen_monotonic"] = time.monotonic()
        if (
            role == "evacuee"
            and self.enable_demographics
            and record.get("age") == "Pending"
            and not record.get("pending_demographics_crops")
        ):
            candidates = self._demographics_candidates(samples)
            # The same list serves twice: as the trigger that queues MiVOLO on
            # promotion, and as the pool a later re-estimate votes from.
            record["pending_demographics_crops"] = candidates
            record["demographics_crop_pool"] = list(candidates)
        latest_spatial_sample = next(
            (sample for sample in reversed(samples) if sample.get("map_point") is not None),
            samples[-1],
        )
        self._record_master_observation_locked(
            identity_id,
            key,
            latest_spatial_sample.get("map_point"),
            latest_spatial_sample.get("observed_at", time.monotonic()),
        )

        query_feature = self._normalize_feature(
            np.mean(
                np.asarray(
                    [features[index] for index in valid_indices],
                    dtype=np.float32,
                ),
                axis=0,
            )
        )
        if query_feature is None:
            raise RuntimeError("The provisional intake fingerprint had zero norm.")

        dynamic_exclusions = self._same_camera_active_ids_locked(
            camera_id,
            excluded_track_key=key,
        )
        match_phase = (
            "provisional_global_match"
            if new_provisional_record
            else "location_target_confirmation"
        )
        if new_provisional_record:
            excluded = {
                candidate_id
                for candidate_id, candidate_record in self.identities.items()
                if candidate_id == identity_id
                or candidate_record.get("identity_state", "confirmed") != "confirmed"
            }
        else:
            # This is a new member provisionally attached to an already
            # confirmed master. Its mandatory global check is specifically
            # against that master; never merge the established master record
            # into a different identity because one pending member matched it.
            excluded = set(self.identities)
            excluded.discard(identity_id)
        excluded |= dynamic_exclusions - ({identity_id} if target_confirmation else set())
        submitted_peer_ids = {
            self.track_to_identity[peer_key]
            for peer_key in task.get("same_camera_peer_keys", ())
            if peer_key in self.track_to_identity
        }
        identity_event(
            "reid_match_context",
            phase=match_phase,
            provisional_master_id=self._public_identity_id(identity_id),
            temporary_group_id=self._temporary_group_token(identity_id),
            track_key=key,
            camera_id=camera_id,
            frame_index=task.get("frame_index"),
            generation=task.get("generation"),
            caller_excluded_master_ids=sorted(task.get("excluded_identity_ids", ())),
            dynamic_same_camera_master_ids=sorted(dynamic_exclusions),
            submitted_peer_master_ids=sorted(submitted_peer_ids),
            map_point=latest_spatial_sample.get("map_point"),
            observed_at=latest_spatial_sample.get("observed_at"),
            feature_source=feature_source,
            feature_space_id=feature_space_id,
        )
        debug_context = {
            "phase": match_phase,
            "provisional_master_id": self._public_identity_id(identity_id),
            "temporary_group_id": self._temporary_group_token(identity_id),
            "track_key": key,
            "frame_index": task.get("frame_index"),
            "generation": task.get("generation"),
        }
        if target_confirmation:
            # Location may have attached (or even stable-confirmed) this track
            # while its intake task was waiting in the worker queue. Compare
            # with that latest target directly. A stale submission-time
            # exclusion must never send the task back to the create-ID path.
            matched_identity_id, matched_slot, distance = (
                self._target_identity_match_locked(
                    identity_id,
                    query_feature,
                    feature_space_id,
                    debug_context=debug_context,
                    return_rejected=True,
                )
            )
        else:
            # A group's members are searched here before it may be promoted,
            # and this is the search that decides whether a location pair is a
            # new person or one the system already holds.  Masters another live
            # box owns were skipped outright, so a man whose own identity was
            # being worn by someone else was compared only against the people
            # he is not, found nothing, and had a second ID minted for him.
            # Most identities are created down this path, not by lone intake.
            owner_blocked_matches = []
            matched_identity_id, matched_slot, distance = self._matching_identity_locked(
                query_feature,
                query_feature_space_id=feature_space_id,
                excluded_identity_ids=excluded,
                camera_id=camera_id,
                map_point=latest_spatial_sample.get("map_point"),
                observed_at=latest_spatial_sample.get("observed_at"),
                debug_context=debug_context,
                track_key=key,
                owner_blocked_matches=owner_blocked_matches,
            )
            self._report_owner_blocked_matches_locked(
                owner_blocked_matches,
                key,
                camera_id,
                match_phase,
                task.get("frame_index"),
            )
            if matched_identity_id is None and self._contest_owner_blocked_master_locked(
                owner_blocked_matches,
                key,
                camera_id,
                task,
                feature_source,
                feature_space_id,
            ):
                return None, None
        target_accepted = bool(
            target_confirmation
            and matched_identity_id == identity_id
            and distance is not None
            and distance < self.distance_threshold
        )
        accepted_existing_match = bool(
            matched_identity_id is not None
            and distance is not None
            and distance < self.distance_threshold
            and (not target_confirmation or target_accepted)
        )
        if accepted_existing_match and self._borderline_match_needs_retry_locked(
            key,
            matched_identity_id,
            matched_slot,
            distance,
            feature_space_id,
            task.get("frame_index", 0),
            match_phase,
        ):
            return None, None
        record.setdefault("global_reid_checked_track_keys", set()).add(key)
        identity_event(
            "provisional_global_reid_checked",
            provisional_master_id=self._public_identity_id(identity_id),
            temporary_group_id=self._temporary_group_token(identity_id),
            track_key=key,
            camera_id=camera_id,
            matched_master_id=matched_identity_id,
            matched_slot=matched_slot,
            distance=distance,
            distance_threshold=self.distance_threshold,
            accepted=(
                target_accepted
                if target_confirmation
                else matched_identity_id is not None
            ),
            ignored_caller_excluded_master_ids=sorted(
                task.get("excluded_identity_ids", ())
            ),
            dynamic_same_camera_master_ids=sorted(dynamic_exclusions),
            feature_source=feature_source,
            feature_space_id=feature_space_id,
        )
        if (
            matched_identity_id is not None
            and new_provisional_record
            and self._merge_agreement_rejected_locked(
                identity_id,
                matched_identity_id,
                query_feature,
                feature_space_id,
                matched_slot,
                distance,
            )
        ):
            matched_identity_id = None
        if matched_identity_id is not None and new_provisional_record:
            merged_identity_id = self._merge_provisional_into_confirmed_locked(
                identity_id,
                matched_identity_id,
                key,
                matched_slot,
                distance,
                feature_source,
                feature_space_id,
            )
            if merged_identity_id is not None:
                return None, merged_identity_id

        if target_confirmation:
            if target_accepted:
                self._confirm_pending_members_locked(
                    identity_id,
                    "global_reid",
                    appearance_confirmed=True,
                    member_keys={key},
                )
                record.setdefault("pending_member_keys", set()).discard(key)
                record.setdefault("challenged_member_keys", set()).discard(key)
                record.setdefault("pending_member_location_streaks", {}).pop(key, None)
                record.setdefault("appearance_rejected_member_keys", set()).discard(key)
                target_state = "confirmed"
                target_reason = "global_reid"
                identity_event(
                    "location_assignment_reid_confirmed",
                    master_id=identity_id,
                    track_key=key,
                    camera_id=camera_id,
                    matched_slot=matched_slot,
                    distance=distance,
                    distance_threshold=self.distance_threshold,
                )
            elif distance is not None and distance >= self.provisional_challenge_distance:
                self._discard_pending_member_evidence_locked(
                    key,
                    "global_reid_challenged",
                )
                record.setdefault("pending_member_keys", set()).discard(key)
                record.setdefault("challenged_member_keys", set()).add(key)
                record.setdefault("appearance_rejected_member_keys", set()).add(key)
                target_state = "challenged"
                target_reason = "global_reid_challenged"
                identity_event(
                    "location_assignment_reid_challenged",
                    master_id=identity_id,
                    track_key=key,
                    camera_id=camera_id,
                    matched_slot=matched_slot,
                    distance=distance,
                    challenge_distance=self.provisional_challenge_distance,
                )
            else:
                if distance is not None and distance >= self.distance_threshold:
                    # Between the match and challenge thresholds appearance is
                    # not confident enough to break the binding -- the cameras
                    # face each other, so a cross-view distance here is often
                    # genuine.  But it is confident enough to bar the
                    # stable-location shortcut from committing this crop.
                    record.setdefault("appearance_rejected_member_keys", set()).add(key)
                target_state = self._track_identity_state_locked(record, key)
                target_reason = record.get("confirmation_reason")
                identity_event(
                    "location_assignment_reid_inconclusive",
                    master_id=identity_id,
                    track_key=key,
                    camera_id=camera_id,
                    matched_slot=matched_slot,
                    distance=distance,
                    distance_threshold=self.distance_threshold,
                    challenge_distance=self.provisional_challenge_distance,
                )
        else:
            target_state = track_identity_state
            target_reason = record.get("confirmation_reason")

        self.track_binding_metadata[key] = {
            "query_feature_space_id": feature_space_id,
            "matched_feature_space_id": (
                feature_space_id if target_accepted else None
            ),
            "matched_slot": matched_slot if target_confirmation else None,
            "distance": distance if target_confirmation else None,
            "appearance_confirmed": target_accepted,
            "feature_source": feature_source,
            "identity_state": target_state,
            "confirmation_reason": target_reason,
            "provisional_intake_complete": True,
        }
        self.pending_intake.pop(key, None)
        self.shadow_tracks.pop(key, None)
        identity_event(
            "provisional_track_analyzed",
            track_key=key,
            camera_id=camera_id,
            master_id=self._public_identity_id(identity_id),
            temporary_group_id=self._temporary_group_token(identity_id),
            frame_index=task.get("frame_index"),
            stored_orientations=sorted(best_semantic_samples),
            feature_source=feature_source,
            feature_space_id=feature_space_id,
        )
        if target_confirmation:
            # Accepted, inconclusive, and challenged target checks are all
            # terminal for this queued task. None may fall through to the
            # normal global match/create path and overwrite the location ID.
            return None, identity_id
        promoted = self._evaluate_provisional_evidence_locked(identity_id)
        return (
            promoted,
            promoted,
        )

    def _start_pending_demographics(self, identity_id):
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None or record.get("identity_state") != "confirmed":
                return
            candidates = record.pop("pending_demographics_crops", None)
            if not candidates or record.get("role") != "evacuee" or not self.enable_demographics:
                return
            record.setdefault("demographics_crop_pool", list(candidates))
            record["demographics_quality"] = self._demographics_pool_quality(candidates)
        self._queue_demographics(identity_id, candidates, "promotion")

    def _queue_demographics(self, identity_id, candidates, reason):
        """Hand one crop set to the MiVOLO worker, or give up loudly."""
        self._ensure_demographics_worker()
        try:
            self._demographics_queue.put_nowait(
                {
                    "identity_id": identity_id,
                    "candidates": candidates,
                    "reason": reason,
                }
            )
        except queue.Full:
            with self._lock:
                record = self.identities.get(identity_id)
                # A refresh that cannot be queued leaves the earlier answer
                # standing.  Only a first estimate has nothing to fall back on.
                if record is not None and record.get("age") == "Pending":
                    record["age"] = "Unknown"
                    record["gender"] = "Unknown"

    def _consider_demographics_refresh_locked(self, identity_id, samples):
        """Re-run MiVOLO when a much closer look at the face turns up.

        The first estimate is made from whatever the person looked like during
        the intake burst that created them, which for someone who entered at
        the far end of the room is five distant faces.  That answer used to be
        final for the whole session.

        Only crops the pipeline already analyses can be considered.  A
        confirmed identity with a full gallery stops producing pose landmarks,
        so there is no face box for its later frames, and running MediaPipe on
        every mapped track just to get one would cost more per frame than the
        occasional stale age does.  A second camera picking the person up, or a
        gallery slot still being filled, both do produce one.

        Returns the crop pool to re-analyse, or None to leave the answer alone.
        """
        if not self.enable_demographics:
            return None
        record = self.identities.get(identity_id)
        if record is None or record.get("role") != "evacuee":
            return None
        if record.get("identity_state") != "confirmed":
            return None
        if int(record.get("demographics_refreshes") or 0) >= DEFAULT_DEMOGRAPHICS_MAX_REFRESHES:
            return None
        # An estimate that has never been answered has no quality to beat, and
        # queueing a second one now would race the first for the same record.
        if record.get("age") == "Pending" or record.get("pending_demographics_crops"):
            return None

        pool = self._merge_demographics_pool(
            record.get("demographics_crop_pool"),
            samples,
        )
        new_quality = self._demographics_pool_quality(pool)
        if new_quality <= 0.0:
            return None
        record["demographics_crop_pool"] = pool
        previous_quality = float(record.get("demographics_quality") or 0.0)
        if new_quality <= previous_quality * DEFAULT_DEMOGRAPHICS_REFRESH_QUALITY_RATIO:
            return None

        record["demographics_quality"] = new_quality
        record["demographics_refreshes"] = int(record.get("demographics_refreshes") or 0) + 1
        if record["demographics_refreshes"] >= DEFAULT_DEMOGRAPHICS_MAX_REFRESHES:
            # No further re-estimate can use these, and five crops per person
            # is real memory on an edge device.  The queued copy keeps its own
            # reference until the worker is done with it.
            record.pop("demographics_crop_pool", None)
        identity_event(
            "demographics_refresh_scheduled",
            console=False,
            master_id=self._public_identity_id(identity_id),
            previous_quality=previous_quality,
            new_quality=new_quality,
            refresh_count=record["demographics_refreshes"],
            previous_age=record.get("age"),
            previous_gender=record.get("gender"),
        )
        return list(pool)

    def _apply_role_vote_locked(self, record, role, role_confidence):
        """Keep the most confident role verdict any camera has produced.

        Each camera classifies its own intake burst, so one person yields one
        vote per camera.  A crop that hides the evidence -- a side-on view of a
        marked vest, say -- reads as little more than a guess, and the burst
        that merely finished first used to be the only one that counted.  The
        confident camera now overrules the doubtful one instead.

        The rule is a running maximum, so a later doubtful vote can never
        displace a confident verdict and the label cannot oscillate as more
        views arrive.
        """

        if record is None:
            return False
        confidence = float(role_confidence)
        if record.get("role_classified") and confidence <= float(
            record.get("role_confidence") or 0.0
        ):
            return False
        changed = record.get("role") != role
        record["role"] = role
        record["role_confidence"] = confidence
        record["role_classified"] = True
        if role != "evacuee":
            # Demographics describe evacuees only.  A staff verdict that lands
            # after the intake crops were stashed has to drop them too, or the
            # promotion that follows would still queue MiVOLO for somebody the
            # system now knows is staff.
            record["age"] = "N/A"
            record["gender"] = "N/A"
            record.pop("pending_demographics_crops", None)
            record.pop("demographics_crop_pool", None)
            record.pop("demographics_quality", None)
        return changed

    def _get_role_classifier(self):
        if not self.enable_role_classification:
            return None
        if self._role_classifier is None:
            self._role_classifier = EvacuationRoleClassifier(self.role_checkpoint)
        return self._role_classifier

    def _process_intake_task(self, task):
        with self._lock:
            if not self._intake_task_is_current_locked(task):
                return
        samples = task["samples"]
        crops = [sample["crop"] for sample in samples]
        features, feature_source, feature_space_id = self._extract_aligned_features(crops)
        valid_indices = [index for index, feature in enumerate(features) if feature is not None]
        if not valid_indices:
            raise RuntimeError("No ReID feature could be extracted from the intake burst.")

        query_feature = self._normalize_feature(
            np.mean(np.asarray([features[index] for index in valid_indices], dtype=np.float32), axis=0)
        )
        if query_feature is None:
            raise RuntimeError("The intake fingerprint had zero norm.")
        hero_index = max(valid_indices, key=lambda index: self._quality_score(samples[index]))
        identity_event(
            "intake_baseline_candidate_selected",
            console=False,
            track_key=task.get("track_key"),
            camera_id=task.get("camera_id"),
            frame_index=task.get("frame_index"),
            generation=task.get("generation"),
            provisional_identity_id=task.get("provisional_identity_id"),
            feature_source=feature_source,
            feature_space_id=feature_space_id,
            selection_rule="maximum_sharpness_times_square_root_area",
            selected_sample_index=hero_index,
            selected_sample=self._sample_debug_summary(samples[hero_index]),
            samples=[self._sample_debug_summary(sample) for sample in samples],
        )
        role_classifier = self._get_role_classifier()
        # The classifier's own top class stands.  A confidence floor here used
        # to rewrite every unsure CAG/SCDF read to "evacuee", which cost a
        # marked staff member his role whenever the one camera allowed to vote
        # had caught him at an angle that hid the marking.  Doubtful crops now
        # lose to confident ones instead, in _apply_role_vote_locked.
        role, role_confidence = (
            role_classifier.predict(crops[hero_index])
            if role_classifier is not None
            else ("evacuee", 0.0)
        )

        provisional_handled = False
        promoted_identity_id = None
        persist_identity_id = None
        provisional_identity_id = task.get("provisional_identity_id")
        with self._lock:
            mapped_identity_id = self.track_to_identity.get(task["track_key"])
            if mapped_identity_id is not None:
                # The live binding is newer than the snapshot captured when
                # this worker task was queued (for example, a provisional ID
                # may already have merged into an established master).
                provisional_identity_id = mapped_identity_id
            provisional_record = self.identities.get(provisional_identity_id)
            provisional_track_state = (
                self._track_identity_state_locked(provisional_record, task["track_key"])
                if provisional_record is not None
                else None
            )
            if (
                provisional_record is not None
                and (
                    provisional_track_state in ("provisional", "challenged")
                    or (
                        provisional_track_state == "confirmed"
                        and provisional_record.get("location_managed")
                    )
                )
                and mapped_identity_id == provisional_identity_id
                and self._intake_task_is_current_locked(task)
            ):
                promoted_identity_id, persist_identity_id = self._store_provisional_intake_locked(
                    provisional_identity_id,
                    task,
                    samples,
                    features,
                    valid_indices,
                    hero_index,
                    feature_source,
                    feature_space_id,
                    role,
                    role_confidence,
                )
                provisional_handled = True
        if provisional_handled:
            if promoted_identity_id is not None:
                self._start_pending_demographics(promoted_identity_id)
            if persist_identity_id is not None:
                self.save_database(persist_identity_id)
            return

        demographics_task = None
        key = task["track_key"]
        camera_id = task.get("camera_id")
        handoff_identity_id = task.get("handoff_identity_id")
        handoff_from_key = task.get("handoff_from_key")
        handoff_committed = False
        latest_spatial_sample = next(
            (sample for sample in reversed(samples) if sample.get("map_point") is not None),
            samples[-1],
        )
        with self._lock:
            if not self._intake_task_is_current_locked(task):
                return
            if handoff_identity_id is not None:
                shadow = self.shadow_tracks.get(key)
                if (
                    shadow is None
                    or shadow.get("identity_id") != handoff_identity_id
                    or shadow.get("canonical_key") != handoff_from_key
                ):
                    return
            dynamic_exclusions = self._same_camera_active_ids_locked(camera_id, excluded_track_key=key)
            submitted_peer_ids = {
                self.track_to_identity[peer_key]
                for peer_key in task.get("same_camera_peer_keys", ())
                if peer_key in self.track_to_identity
            }
            excluded = (
                set(task.get("excluded_identity_ids", ()))
                | dynamic_exclusions
                | submitted_peer_ids
            )
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "reid_match_context",
                track_key=key,
                camera_id=camera_id,
                frame_index=task.get("frame_index"),
                generation=task.get("generation"),
                caller_excluded_master_ids=sorted(task.get("excluded_identity_ids", ())),
                dynamic_same_camera_master_ids=sorted(dynamic_exclusions),
                submitted_peer_master_ids=sorted(submitted_peer_ids),
                map_point=latest_spatial_sample.get("map_point"),
                observed_at=latest_spatial_sample.get("observed_at"),
                feature_source=feature_source,
                feature_space_id=feature_space_id,
            )
            identity_id = None
            matched_slot = None
            distance = None

            if handoff_identity_id is not None:
                identity_id, matched_slot, distance = self._target_identity_match_locked(
                    handoff_identity_id,
                    query_feature,
                    feature_space_id,
                    debug_context={
                        "phase": "shadow_target_match",
                        "track_key": key,
                        "target_master_id": handoff_identity_id,
                        "frame_index": task.get("frame_index"),
                        "generation": task.get("generation"),
                    },
                )
                if identity_id is not None:
                    visible_owners = self._visible_same_camera_identity_owners_locked(
                        handoff_identity_id,
                        camera_id,
                        excluded_key=key,
                    )
                    if visible_owners:
                        # Appearance confirms that both simultaneous boxes are
                        # the same person. Remember that result, keep exactly
                        # one visible owner, and do not run TransReID again.
                        shadow = self.shadow_tracks.get(key)
                        if shadow is not None:
                            shadow["canonical_key"] = min(visible_owners, key=repr)
                            shadow["identity_id"] = handoff_identity_id
                            shadow["provisional"] = False
                            shadow["verified"] = True
                            baseline = self.identities[handoff_identity_id]["gallery"]["baseline"]
                            shadow["verification"] = {
                                "query_feature_space_id": feature_space_id,
                                "matched_feature_space_id": baseline.get("feature_space_id"),
                                "matched_slot": matched_slot,
                                "distance": distance,
                                "similarity": 1.0 - float(distance),
                                "appearance_confirmed": bool(
                                    feature_source == "transreid"
                                    and feature_space_id == baseline.get("feature_space_id")
                                ),
                                "feature_source": feature_source,
                            }
                            # TEMP_IDENTITY_DEBUG
                            identity_event(
                                "shadow_verified",
                                track_key=key,
                                canonical_track_key=shadow.get("canonical_key"),
                                master_id=handoff_identity_id,
                                matched_slot=matched_slot,
                                distance=distance,
                                feature_source=feature_source,
                            )
                        self.pending_intake.pop(key, None)
                        self._next_track_generation_locked(key)
                        return

                    self._clear_nonvisible_same_camera_owners_locked(
                        handoff_identity_id,
                        camera_id,
                        preserved_key=key,
                    )
                    handoff_committed = True
                else:
                    # Appearance has vetoed the geometric handoff. Reuse this
                    # already-computed batch in the normal match/create path,
                    # while explicitly preventing a second attempt at the
                    # rejected canonical master.
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "shadow_appearance_veto",
                        track_key=key,
                        canonical_track_key=handoff_from_key,
                        target_master_id=handoff_identity_id,
                        reason="target_master_distance_rejected",
                    )
                    self.shadow_tracks.pop(key, None)
                    excluded.add(handoff_identity_id)

            physically_rejected_matches = []
            owner_blocked_matches = []
            while identity_id is None:
                candidate_identity_id, candidate_slot, candidate_distance = self._matching_identity_locked(
                    query_feature,
                    query_feature_space_id=feature_space_id,
                    excluded_identity_ids=excluded,
                    camera_id=camera_id,
                    map_point=latest_spatial_sample.get("map_point"),
                    observed_at=latest_spatial_sample.get("observed_at"),
                    debug_context={
                        "phase": "normal_intake_match",
                        "track_key": key,
                        "frame_index": task.get("frame_index"),
                        "generation": task.get("generation"),
                    },
                    track_key=key,
                    physically_rejected_matches=physically_rejected_matches,
                    owner_blocked_matches=owner_blocked_matches,
                )
                if candidate_identity_id is None:
                    matched_slot = None
                    distance = None
                    break
                visible_owners = self._visible_same_camera_identity_owners_locked(
                    candidate_identity_id,
                    camera_id,
                    excluded_key=key,
                )
                if visible_owners:
                    # Visibility is the final single-owner guard. Do not steal
                    # a master even if a stale submission snapshot omitted it;
                    # try the next eligible gallery instead.
                    # TEMP_IDENTITY_DEBUG
                    identity_event(
                        "reid_candidate_excluded",
                        track_key=key,
                        candidate_master_id=candidate_identity_id,
                        reason="visible_same_camera_owner",
                        visible_owner_track_keys=visible_owners,
                        camera_id=camera_id,
                    )
                    excluded.add(candidate_identity_id)
                    continue
                identity_id = candidate_identity_id
                matched_slot = candidate_slot
                distance = candidate_distance
                self._clear_nonvisible_same_camera_owners_locked(
                    identity_id,
                    camera_id,
                    preserved_key=key,
                )
            if (
                identity_id is not None
                and self._borderline_match_needs_retry_locked(
                    key,
                    identity_id,
                    matched_slot,
                    distance,
                    feature_space_id,
                    task.get("frame_index", 0),
                    "normal_intake_match",
                )
            ):
                return
            reidentified = identity_id is not None
            if identity_id is None:
                strongest_blocked_match = min(
                    physically_rejected_matches,
                    key=lambda match: float(match["distance"]),
                    default=None,
                )
                if (
                    strongest_blocked_match is not None
                    and self._start_contested_identity_claim_locked(
                        strongest_blocked_match,
                        task,
                        feature_source,
                        feature_space_id,
                    )
                ):
                    return
                self._report_owner_blocked_matches_locked(
                    owner_blocked_matches,
                    key,
                    camera_id,
                    "normal_intake_match",
                    task.get("frame_index"),
                )
                # Nobody eligible matched, but a master this camera had ruled
                # out may be the right answer worn by the wrong track.  Contest
                # its holder rather than allocate a second ID for one person.
                if self._contest_owner_blocked_master_locked(
                    owner_blocked_matches,
                    key,
                    camera_id,
                    task,
                    feature_source,
                    feature_space_id,
                ):
                    return
                recovery_held = self._refresh_physical_conflict_recovery_hold_locked(
                    key,
                    task.get("frame_index", 0),
                )
                if recovery_held:
                    recovery_state = self.physical_conflict_recovery_holds.get(key)
                    state = self.pending_intake.get(key)
                    if (
                        recovery_state is not None
                        and state is not None
                        and int(state.get("generation", -1))
                        == int(task.get("generation", -2))
                    ):
                        state["deferred_by_physical_conflict_hold"] = True
                        identity_event(
                            "physical_conflict_new_master_deferred",
                            track_key=key,
                            camera_id=camera_id,
                            frame_index=task.get("frame_index"),
                            generation=task.get("generation"),
                            rejected_master_id=recovery_state[
                                "source_identity_id"
                            ],
                            source_conflict_token=recovery_state[
                                "source_conflict_token"
                            ],
                            related_conflict_tokens=sorted(
                                recovery_state["related_conflict_tokens"]
                            ),
                            grace_until_frame=recovery_state[
                                "grace_until_frame"
                            ],
                            expires_frame=recovery_state["expires_frame"],
                            excluded_master_ids=sorted(excluded),
                            reason="wait_for_connected_same_camera_conflict",
                        )
                    return
                hold_tokens = tuple(sorted(map(repr, self.new_master_holds.get(key, ()))))
                if hold_tokens:
                    state = self.pending_intake.get(key)
                    if (
                        state is not None
                        and int(state.get("generation", -1))
                        == int(task.get("generation", -2))
                    ):
                        # The five-crop GPU analysis has completed. Preserve
                        # the intake and wait for the location coordinator to
                        # either create a provisional group or release the
                        # bounded hold. No permanent ID is allocated here.
                        state["deferred_by_new_master_hold"] = True
                        identity_event(
                            "new_master_creation_deferred",
                            track_key=key,
                            camera_id=camera_id,
                            frame_index=task.get("frame_index"),
                            generation=task.get("generation"),
                            hold_tokens=hold_tokens,
                            feature_source=feature_source,
                            feature_space_id=feature_space_id,
                            reason="promising_cross_camera_pair",
                        )
                    return
                identity_id = self.next_identity_id
                baseline_sample = samples[hero_index]
                baseline_slot, baseline_evidence_task = self._make_slot(
                    identity_id,
                    "baseline",
                    features[hero_index],
                    baseline_sample,
                    feature_source,
                    feature_space_id,
                    track_key=key,
                )
                self.next_identity_id += 1
                self.identities[identity_id] = self._new_record(role, role_confidence)
                self.identities[identity_id]["gallery"]["baseline"] = baseline_slot
                self._queue_evidence_save(baseline_evidence_task)
                identity_event(
                    "baseline_selected",
                    console=False,
                    track_key=key,
                    camera_id=camera_id,
                    master_id=identity_id,
                    slot_name="baseline",
                    feature_source=feature_source,
                    feature_space_id=feature_space_id,
                    selection_rule="maximum_sharpness_times_square_root_area",
                    evidence_path=baseline_slot.get("image_path"),
                    selected_sample=self._sample_debug_summary(baseline_sample),
                )
                # TEMP_IDENTITY_DEBUG
                strongest_rejected = min(
                    physically_rejected_matches,
                    key=lambda match: float(match["distance"]),
                    default=None,
                )
                identity_event(
                    "master_created",
                    track_key=key,
                    camera_id=camera_id,
                    master_id=identity_id,
                    frame_index=task.get("frame_index"),
                    generation=task.get("generation"),
                    map_point=latest_spatial_sample.get("map_point"),
                    observed_at=latest_spatial_sample.get("observed_at"),
                    feature_source=feature_source,
                    feature_space_id=feature_space_id,
                    excluded_master_ids=sorted(excluded),
                    reason="no_eligible_master_match",
                    # Everything below answers the question this event could not
                    # answer before: is this a genuinely new person, or a second
                    # ID for somebody already on the map?  A new master standing
                    # on top of an existing one, or rejected only just outside
                    # the ReID threshold, is the duplicate-master signature.
                    reid_distance_threshold=self.distance_threshold,
                    best_rejected_master_id=(
                        None if strongest_rejected is None else strongest_rejected.get("identity_id")
                    ),
                    best_rejected_distance=(
                        None if strongest_rejected is None else float(strongest_rejected["distance"])
                    ),
                    best_rejected_slot=(
                        None if strongest_rejected is None else strongest_rejected.get("matched_slot")
                    ),
                    owner_blocked_master_ids=sorted(
                        {
                            match.get("identity_id")
                            for match in owner_blocked_matches
                            if match.get("identity_id") is not None
                        }
                    ),
                    **self._nearby_master_context_locked(
                        latest_spatial_sample.get("map_point"),
                        latest_spatial_sample.get("observed_at"),
                        camera_id,
                    ),
                )
                if role == "evacuee" and self.enable_demographics:
                    candidates = self._demographics_candidates(samples)
                    self.identities[identity_id]["demographics_crop_pool"] = list(candidates)
                    self.identities[identity_id]["demographics_quality"] = (
                        self._demographics_pool_quality(candidates)
                    )
                    demographics_task = {
                        "identity_id": identity_id,
                        "candidates": candidates,
                        "reason": "identity_created",
                    }
            record = self.identities[identity_id]
            baseline_space = record["gallery"]["baseline"].get("feature_space_id")
            if baseline_space != feature_space_id:
                raise RuntimeError("Refusing to mix incompatible ReID feature spaces in one master gallery.")

            # Reuse already-computed intake features for distinct, reliable
            # semantic views instead of scheduling avoidable future GPU calls.
            best_semantic_samples = {}
            for index in valid_indices:
                if index == hero_index:
                    continue
                slot_name = samples[index].get("orientation")
                if slot_name not in REID_SEMANTIC_SLOTS:
                    continue
                previous = best_semantic_samples.get(slot_name)
                if previous is None or self._quality_score(samples[index]) > self._quality_score(samples[previous]):
                    best_semantic_samples[slot_name] = index
            for slot_name, index in best_semantic_samples.items():
                if record["gallery"].get(slot_name) is None:
                    slot, evidence_task = self._make_slot(
                        identity_id,
                        slot_name,
                        features[index],
                        samples[index],
                        feature_source,
                        feature_space_id,
                        track_key=key,
                    )
                    record["gallery"][slot_name] = slot
                    self._queue_evidence_save(evidence_task)

            record["hits"] = int(record.get("hits", 0)) + 1
            record["last_seen_monotonic"] = time.monotonic()
            self.track_to_identity[key] = identity_id
            self._release_physical_conflict_recovery_hold_locked(
                key,
                "track_reidentified",
                rearm_deferred_intake=False,
            )
            self.physical_conflict_rejections.pop(key, None)
            self.track_binding_metadata[key] = {
                "query_feature_space_id": feature_space_id,
                "matched_feature_space_id": baseline_space,
                "matched_slot": matched_slot,
                "distance": distance,
                "appearance_confirmed": bool(
                    feature_source == "transreid" and feature_space_id == baseline_space
                ),
                "feature_source": feature_source,
                "handoff_from_track_key": handoff_from_key if handoff_committed else None,
            }
            # TEMP_IDENTITY_DEBUG
            identity_event(
                "track_bound",
                track_key=key,
                camera_id=camera_id,
                master_id=identity_id,
                frame_index=task.get("frame_index"),
                generation=task.get("generation"),
                reidentified=reidentified,
                matched_slot=matched_slot,
                distance=distance,
                appearance_confirmed=bool(
                    feature_source == "transreid" and feature_space_id == baseline_space
                ),
                feature_source=feature_source,
                handoff_from_track_key=handoff_from_key if handoff_committed else None,
            )
            self._record_master_observation_locked(
                identity_id,
                key,
                latest_spatial_sample.get("map_point"),
                latest_spatial_sample.get("observed_at", time.monotonic()),
            )
            self.track_results[key] = {
                "similarity": 0.0 if distance is None else 1.0 - float(distance),
                "reidentified": reidentified,
                "matched_slot": matched_slot,
            }
            self.pending_intake.pop(key, None)
            self.shadow_tracks.pop(key, None)

        self.save_database(identity_id)

        if demographics_task is not None:
            self._ensure_demographics_worker()
            try:
                self._demographics_queue.put_nowait(demographics_task)
            except queue.Full:
                print(f"Demographics queue full; ID {identity_id} marked Unknown.")
                with self._lock:
                    record = self.identities.get(identity_id)
                    if record is not None:
                        record["age"] = "Unknown"
                        record["gender"] = "Unknown"
                self.save_database(identity_id)

        if self.verbose:
            if reidentified:
                print(
                    f"ReID: {key} -> Master {identity_id} via {matched_slot} "
                    f"(distance={distance:.3f})"
                )
            else:
                print(f"ReID: created Master {identity_id} from one {len(crops)}-crop batch")

    def _schedule_identity_audit_locked(
        self,
        key,
        identity_id,
        crop,
        frame_index,
        detection_confidence,
        now,
        body_complete,
    ):
        """Re-check a settled binding on an interval.

        Only clean, complete, well-separated crops are used.  A crop taken
        while two people overlap says nothing trustworthy about which of them
        the box is on, and acting on one would deepen a swap instead of
        repairing it -- those never reach here, because ``person_crops`` is
        only populated when no intruder shares the box.
        """

        if self.identity_audit_interval_seconds <= 0.0 or crop is None or crop.size == 0:
            return
        record = self.identities.get(identity_id)
        if record is None or record.get("identity_state") != "confirmed":
            return
        if body_complete is False or identity_id in self.physical_conflicts:
            return
        if (record.get("gallery") or {}).get("baseline") is None:
            return
        state = self.identity_audit_state.setdefault(
            key,
            {"next_due": now + self.identity_audit_interval_seconds, "rivals": {}},
        )
        if now < float(state.get("next_due", 0.0)):
            return
        if (
            detection_confidence is not None
            and float(detection_confidence) <= self.semantic_confidence_threshold
        ):
            return
        if image_sharpness(crop) <= self.blur_threshold:
            return
        state["next_due"] = now + self.identity_audit_interval_seconds
        self._task_queue.put_nowait(
            {
                "type": "identity_audit",
                "track_key": key,
                "identity_id": identity_id,
                "camera_id": self._camera_from_key(key),
                "frame_index": int(frame_index),
                # Slots are stamped with the frame observation time, so the
                # checkpoint has to use that same clock or the comparison
                # against captured_at is meaningless.
                "observed_at": now,
                "crop": crop.copy(),
            }
        )

    def _withdraw_track_contributions_locked(self, track_key, identity_id, since):
        """Take back the crops a track left behind on a master it has just lost.

        A repair moves the track and used to leave its photographs where they
        were.  If the audit now says that box was Haoran, every crop it gave
        Mikail was Haoran's, and leaving them is how two galleries slowly
        become one blurred average of two people -- after which no rival can
        ever win by a margin and the audit has nothing left to separate them.

        Only crops taken since the track last passed an audit are withdrawn.
        A passing audit is a checkpoint: before it the binding was vouched for,
        after it the track may already have drifted onto someone else.
        """

        record = self.identities.get(identity_id)
        if record is None:
            return ()
        withdrawn = []

        def contributed(slot):
            if not slot or slot.get("contributed_by_track_key") != tuple(track_key):
                return False
            if since is None:
                return True
            captured_at = slot.get("captured_at")
            return captured_at is None or float(captured_at) >= float(since)

        for slot_name in list((record.get("gallery") or {})):
            if contributed(record["gallery"].get(slot_name)):
                withdrawn.append(("gallery", slot_name, record["gallery"].pop(slot_name)))
                record["gallery"][slot_name] = None
        for camera_id in list((record.get("camera_baselines") or {})):
            if contributed(record["camera_baselines"].get(camera_id)):
                withdrawn.append(
                    ("baseline", camera_id, record["camera_baselines"].pop(camera_id))
                )
        for camera_id, camera_gallery in (record.get("camera_views") or {}).items():
            for slot_name in list(camera_gallery):
                if contributed(camera_gallery.get(slot_name)):
                    withdrawn.append(
                        (f"view:{camera_id}", slot_name, camera_gallery[slot_name])
                    )
                    camera_gallery[slot_name] = None

        if not withdrawn:
            return ()
        # The baseline anchors every later comparison, so an identity must
        # never be left without one while other views remain.
        if record.get("gallery", {}).get("baseline") is None:
            replacement = max(
                (
                    slot
                    for slot in (record.get("gallery") or {}).values()
                    if slot and slot.get("feature") is not None
                ),
                key=lambda slot: (
                    float(slot.get("sharpness", 0.0)),
                    float(slot.get("detection_confidence") or 0.0),
                ),
                default=None,
            )
            if replacement is not None:
                record["gallery"]["baseline"] = dict(replacement)
        identity_event(
            "track_contributions_withdrawn",
            master_id=self._public_identity_id(identity_id),
            track_key=track_key,
            withdrawn=[
                {
                    "scope": scope,
                    "slot": slot_name,
                    "image_path": (slot or {}).get("image_path"),
                    "captured_frame": (slot or {}).get("captured_frame"),
                }
                for scope, slot_name, slot in withdrawn
            ],
            withdrawn_since=since,
            baseline_replaced=record.get("gallery", {}).get("baseline") is not None,
            reason="the_track_was_ruled_to_belong_to_another_master",
        )
        return tuple(withdrawn)

    def _contest_holding_track_locked(self, track_key, bound_identity_id):
        """The live contest this track is answering to, if any.

        A contest outranks the audit: it weighs two claimants against each
        other with several crops apiece, where the audit judges one track on
        one.  Patience is bounded because a contest starved of clean crops --
        which is exactly what a huddle produces -- would otherwise mute the
        audit for as long as the huddle lasted.
        """

        del bound_identity_id  # a contest anywhere still owns this track
        now = time.monotonic()
        for identity_id, state in self.physical_conflicts.items():
            # Only a contest this track is actually standing in is disrupted by
            # moving it.  One between two other tracks is none of its business.
            if track_key not in state.get("candidates", {}):
                continue
            age = now - float(state.get("started_monotonic") or now)
            if (
                self.identity_audit_contest_patience_seconds > 0.0
                and age > self.identity_audit_contest_patience_seconds
            ):
                continue
            return {
                "master_id": self._public_identity_id(identity_id),
                "token": state.get("token"),
                "age_seconds": age,
            }
        return None

    def _process_identity_audit_task(self, task):
        features, feature_source, feature_space_id = self._extract_aligned_features(
            [task["crop"]]
        )
        query = self._normalize_feature(features[0] if features else None)
        if query is None:
            return

        key = task["track_key"]
        bound_identity_id = task["identity_id"]
        with self._lock:
            if self.track_to_identity.get(key) != bound_identity_id:
                return
            # Scheduling checked for a contest, but the GPU queue means minutes
            # of frames can pass before this runs, and a contest started in
            # between.  Reassigning one of its claimants now leaves the
            # arbitration waiting on a track that is no longer there.
            contest = self._contest_holding_track_locked(key, bound_identity_id)
            if contest is not None:
                identity_event(
                    "identity_audit_yielded_to_contest",
                    console=False,
                    throttle_key=(key, "audit_yielded"),
                    throttle_seconds=5.0,
                    track_key=key,
                    bound_master_id=bound_identity_id,
                    contest_master_id=contest["master_id"],
                    contest_token=contest["token"],
                    contest_age_seconds=contest["age_seconds"],
                    patience_seconds=self.identity_audit_contest_patience_seconds,
                    reason="contest_decides_this_track",
                )
                return
            state = self.identity_audit_state.setdefault(
                key,
                {"next_due": 0.0, "rivals": {}},
            )
            _bound_match, bound_slot, bound_distance = self._target_identity_match_locked(
                bound_identity_id,
                query,
                feature_space_id,
                debug_context={"phase": "identity_audit", "track_key": key},
                return_rejected=True,
            )
            rival_id, rival_slot, rival_distance = self._matching_identity_locked(
                query,
                query_feature_space_id=feature_space_id,
                excluded_identity_ids={bound_identity_id},
                camera_id=task.get("camera_id"),
                debug_context={"phase": "identity_audit", "track_key": key},
                track_key=key,
            )
            shared = {
                "track_key": key,
                "camera_id": task.get("camera_id"),
                "frame_index": task.get("frame_index"),
                "bound_master_id": bound_identity_id,
                "bound_distance": bound_distance,
                "bound_slot": bound_slot,
                "rival_master_id": rival_id,
                "rival_distance": rival_distance,
                "rival_slot": rival_slot,
                "audit_margin": self.identity_audit_margin,
                "required_rounds": self.identity_audit_rounds,
                "feature_source": feature_source,
            }
            contradicted = bool(
                rival_id is not None
                and rival_distance is not None
                and rival_distance < self.distance_threshold
                and (
                    bound_distance is None
                    or rival_distance <= bound_distance - self.identity_audit_margin
                )
            )
            if not contradicted:
                if state.get("rivals"):
                    state["rivals"] = {}
                # A passing audit is a checkpoint.  Crops stored before it were
                # taken while the binding was known good; anything after it is
                # unvouched for, and is what a later repair takes back.
                state["last_agreed_at"] = task.get("observed_at")
                identity_event(
                    "identity_audit_agreed",
                    console=False,
                    reason="binding_upheld",
                    **shared,
                )
                return

            # A rival only counts while it keeps winning.  Any other outcome
            # clears the tally, so two similar people cannot slowly accumulate
            # enough scattered wins to trade identities.
            rivals = state.setdefault("rivals", {})
            rounds_won = int(rivals.get(rival_id, 0)) + 1
            state["rivals"] = {rival_id: rounds_won}
            if rounds_won < self.identity_audit_rounds:
                identity_event(
                    "identity_audit_contradicted",
                    reason="awaiting_confirmation",
                    rounds_won=rounds_won,
                    **shared,
                )
                return
            if self._reject_second_visible_owner_locked(
                rival_id,
                [key],
                event_name="identity_audit_repair_declined",
            ):
                identity_event(
                    "identity_audit_repair_deferred",
                    reason="rival_master_already_has_a_visible_owner",
                    rounds_won=rounds_won,
                    **shared,
                )
                return
            # Take back what this track gave the master it is leaving, before
            # the binding is cleared and the link is lost.
            self._withdraw_track_contributions_locked(
                key,
                bound_identity_id,
                state.get("last_agreed_at"),
            )
            self._clear_local_binding_locked(key)
            self.track_to_identity[key] = rival_id
            rival_record = self.identities.get(rival_id)
            if rival_record is not None:
                rival_record.setdefault("member_track_keys", set()).add(key)
                rival_record["last_seen_monotonic"] = time.monotonic()
            metadata = self.track_binding_metadata.setdefault(key, {})
            metadata["identity_state"] = "confirmed"
            metadata["confirmation_reason"] = "identity_audit"
            metadata["appearance_confirmed"] = True
            metadata["matched_feature_space_id"] = feature_space_id
            state["rivals"] = {}
            identity_event(
                "identity_audit_repaired",
                reason="rival_master_won_consecutive_audits",
                rounds_won=rounds_won,
                **shared,
            )
            if self.verbose:
                print(
                    f"ReID: audit moved {key} from Master {bound_identity_id} "
                    f"to Master {rival_id}"
                )

    def _process_semantic_task(self, task):
        sample = task["sample"]
        features, feature_source, feature_space_id = self._extract_aligned_features([sample["crop"]])
        feature = features[0] if features else None
        if feature is None:
            raise RuntimeError("No feature could be extracted for the semantic slot.")

        identity_id = task["identity_id"]
        slot_name = task["slot_name"]
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None or record.get("gallery", {}).get(slot_name) is not None:
                return
            # Discard stale work if ByteTrack remapped this local key while
            # the GPU task was waiting in the queue.
            if self.track_to_identity.get(task["track_key"]) != identity_id:
                return
            baseline = record.get("gallery", {}).get("baseline") or {}
            if baseline.get("feature_space_id") != feature_space_id:
                raise RuntimeError("Semantic crop used an incompatible ReID feature space.")
            slot, evidence_task = self._make_slot(
                identity_id,
                slot_name,
                feature,
                sample,
                feature_source,
                feature_space_id,
                track_key=task["track_key"],
            )
            if self._gallery_admission_rejected_locked(
                identity_id,
                record,
                slot,
                slot_name,
                "master_gallery",
                track_key=task["track_key"],
            ):
                return
            record["gallery"][slot_name] = slot
            self._queue_evidence_save(evidence_task)
        self.save_database(identity_id)
        if self.verbose:
            print(f"ReID: filled {slot_name} for Master {identity_id}")

    def _process_provisional_semantic_task(self, task):
        sample = task["sample"]
        features, feature_source, feature_space_id = self._extract_aligned_features([sample["crop"]])
        feature = features[0] if features else None
        if feature is None:
            raise RuntimeError("No feature could be extracted for the provisional semantic slot.")

        identity_id = task["identity_id"]
        slot_name = task["slot_name"]
        camera_id = str(sample.get("camera_id"))
        promoted_identity_id = None
        stored_for_confirmed = False
        with self._lock:
            record = self.identities.get(identity_id)
            if record is None or record.get("identity_state") not in (
                "provisional",
                "challenged",
                "confirmed",
            ):
                return
            if self.track_to_identity.get(task["track_key"]) != identity_id:
                return
            track_camera = self._camera_from_key(task["track_key"])
            if (
                track_camera in self.visible_track_keys_by_camera
                and task["track_key"]
                not in self.visible_track_keys_by_camera.get(track_camera, set())
            ):
                return
            track_state = self._track_identity_state_locked(record, task["track_key"])
            pending_target_member = bool(
                record.get("identity_state") == "confirmed"
                and track_state in ("provisional", "challenged")
            )
            if pending_target_member:
                staged_views = self.pending_member_evidence.get(
                    task["track_key"], {}
                ).get("views", {})
                if staged_views.get(slot_name) is not None:
                    return
            else:
                camera_gallery = record.setdefault("camera_views", {}).setdefault(
                    camera_id,
                    {name: None for name in REID_SEMANTIC_SLOTS},
                )
                if camera_gallery.get(slot_name) is not None:
                    return
            slot, evidence_task = self._make_slot(
                identity_id,
                f"{camera_id}_{slot_name}",
                feature,
                sample,
                feature_source,
                feature_space_id,
                track_key=task["track_key"],
            )
            if self._gallery_admission_rejected_locked(
                identity_id,
                record,
                slot,
                slot_name,
                "pending_member" if pending_target_member else "camera_view",
                track_key=task["track_key"],
            ):
                return
            if pending_target_member:
                self._stage_pending_member_evidence_locked(
                    identity_id,
                    task["track_key"],
                    camera_id,
                    None,
                    None,
                    {slot_name: (slot, evidence_task)},
                )
            else:
                camera_gallery[slot_name] = slot
                self._queue_evidence_save(evidence_task)
            identity_event(
                "provisional_angle_stored",
                master_id=identity_id if identity_id > 0 else None,
                temporary_group_id=(
                    f"tmp_{abs(int(identity_id))}" if identity_id < 0 else None
                ),
                camera_id=camera_id,
                orientation=slot_name,
                frame_index=sample.get("frame_index"),
            )
            if record.get("identity_state") == "confirmed" and not pending_target_member:
                if record.get("gallery", {}).get(slot_name) is None:
                    baseline_space = (record.get("gallery", {}).get("baseline") or {}).get(
                        "feature_space_id"
                    )
                    if baseline_space == feature_space_id:
                        record["gallery"][slot_name] = dict(camera_gallery[slot_name])
                stored_for_confirmed = True
            else:
                promoted_identity_id = self._evaluate_provisional_evidence_locked(
                    identity_id
                )
        if promoted_identity_id is not None:
            self._start_pending_demographics(promoted_identity_id)
        persistence_identity_id = promoted_identity_id or (
            identity_id if stored_for_confirmed else None
        )
        if persistence_identity_id is not None:
            self.save_database(persistence_identity_id)

    def _process_task(self, task):
        if task["type"] == "intake":
            self._process_intake_task(task)
        elif task["type"] == "physical_conflict":
            self._process_physical_conflict_task(task)
        elif task["type"] == "semantic":
            self._process_semantic_task(task)
        elif task["type"] == "provisional_semantic":
            self._process_provisional_semantic_task(task)
        elif task["type"] == "identity_audit":
            self._process_identity_audit_task(task)
        else:
            raise ValueError(f"Unknown ReID analyst task: {task['type']}")

    def _worker_loop(self):
        while True:
            task = self._task_queue.get()
            try:
                if task is self._stop_token:
                    return
                self._process_task(task)
            except Exception as exc:
                print(f"ReID analyst task failed: {exc}")
                if isinstance(task, dict) and task.get("type") == "intake":
                    with self._lock:
                        state = self.pending_intake.get(task.get("track_key"))
                        if (
                            state is not None
                            and int(state.get("generation", -1)) == int(task.get("generation", -2))
                        ):
                            failure_count = int(state.get("failure_count", 0)) + 1
                            retry_frames = min(
                                self.max_retry_frames,
                                self.intake_retry_frames * (2 ** min(failure_count - 1, 8)),
                            )
                            state["submitted"] = False
                            state["samples"] = []
                            state["last_frame"] = None
                            task_samples = task.get("samples") or ()
                            state["first_seen"] = float(
                                task_samples[-1].get("observed_at", time.monotonic())
                                if task_samples
                                else time.monotonic()
                            )
                            state["failure_count"] = failure_count
                            state["next_retry_frame"] = (
                                int(task.get("frame_index", 0)) + int(retry_frames)
                            )
                            state["generation"] = self._next_track_generation_locked(
                                task.get("track_key")
                            )
                            # TEMP_IDENTITY_DEBUG
                            identity_event(
                                "intake_task_failed",
                                track_key=task.get("track_key"),
                                camera_id=task.get("camera_id"),
                                frame_index=task.get("frame_index"),
                                generation=task.get("generation"),
                                error=str(exc),
                                failure_count=failure_count,
                                retry_frames=retry_frames,
                                next_retry_frame=state.get("next_retry_frame"),
                            )
                elif isinstance(task, dict) and task.get("type") == "physical_conflict":
                    with self._lock:
                        state = self.physical_conflicts.get(task.get("identity_id"))
                        if (
                            state is not None
                            and state.get("token") == task.get("conflict_token")
                        ):
                            state["submitted"] = False
                            challenger_key = state.get("challenger_key")
                            challenger_samples = state.get(
                                "challenger_seed_samples",
                                [],
                            )
                            state["candidates"] = {
                                key: (
                                    [
                                        {**sample, "crop": sample["crop"].copy()}
                                        for sample in challenger_samples
                                    ]
                                    if key == challenger_key
                                    else []
                                )
                                for key in state["candidates"]
                            }
                            state["last_frames"] = (
                                {
                                    challenger_key: max(
                                        (
                                            int(sample.get("frame_index", 0))
                                            for sample in challenger_samples
                                        ),
                                        default=0,
                                    )
                                }
                                if challenger_key is not None
                                else {}
                            )
                            state["attempts"] = int(state.get("attempts", 0)) + 1
                            identity_event(
                                "physical_conflict_task_failed",
                                master_id=task.get("identity_id"),
                                conflict_token=task.get("conflict_token"),
                                error=str(exc),
                                attempts=state["attempts"],
                            )
                elif isinstance(task, dict) and task.get("type") in ("semantic", "provisional_semantic"):
                    with self._lock:
                        sample = task.get("sample", {})
                        semantic_clock_key = (
                            task.get("identity_id"),
                            self._camera_from_key(task.get("track_key")),
                        )
                        self.next_semantic_attempt_frame[semantic_clock_key] = (
                            int(sample.get("frame_index", 0)) + self.semantic_retry_frames
                        )
            finally:
                if isinstance(task, dict) and task.get("type") in ("semantic", "provisional_semantic"):
                    with self._lock:
                        camera_id = self._camera_from_key(task.get("track_key"))
                        pending_key = (
                            (task.get("identity_id"), camera_id, task.get("slot_name"))
                            if task.get("type") == "provisional_semantic"
                            else (task.get("identity_id"), task.get("slot_name"))
                        )
                        self.pending_semantic_slots.discard(pending_key)
                self._task_queue.task_done()

    def _ensure_demographics_worker(self):
        if self._demographics_worker is not None:
            return
        with self._lock:
            if self._demographics_worker is None:
                self._demographics_worker = threading.Thread(
                    target=self._demographics_worker_loop,
                    name="demographics-analyst",
                    daemon=True,
                )
                self._demographics_worker.start()

    def _demographics_worker_loop(self):
        while True:
            task = self._demographics_queue.get()
            try:
                if task is self._stop_token:
                    return
                if self._demographics_engine is None:
                    from demographics import DemographicsEngine

                    self._demographics_engine = DemographicsEngine(device=self.demographics_device)
                reading = self._demographics_engine.analyze_batch(task["candidates"])
                with self._lock:
                    record = self.identities.get(task["identity_id"])
                    if record is not None and record.get("role") == "evacuee":
                        record["age"] = reading.age
                        record["gender"] = reading.gender
                        # Kept for the event log and for operators reading the
                        # database directly; the backend schema still carries
                        # only the age and gender themselves.
                        record["demographics_confidence"] = reading.confidence
                identity_event(
                    "demographics_estimated",
                    console=False,
                    master_id=self._public_identity_id(task["identity_id"]),
                    reason=task.get("reason"),
                    age=reading.age,
                    gender=reading.gender,
                    confidence=reading.confidence,
                    crops_offered=len(task["candidates"]),
                    crops_used=reading.samples_used,
                    crops_with_face=reading.samples_with_face,
                )
                self.save_database(task["identity_id"])
            except Exception as exc:
                print(f"Demographics analysis failed: {exc}")
                if isinstance(task, dict):
                    with self._lock:
                        record = self.identities.get(task.get("identity_id"))
                        # A failed refresh must not erase a good earlier
                        # reading; only an unanswered first estimate is
                        # downgraded to Unknown.
                        if record is not None and record.get("age") in ("Pending", "Analyzing"):
                            record["age"] = "Unknown"
                            record["gender"] = "Unknown"
                    self.save_database(task.get("identity_id"))
            finally:
                self._demographics_queue.task_done()

    def wait_for_idle(self, timeout=5.0):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if (
                self._task_queue.unfinished_tasks == 0
                and self._demographics_queue.unfinished_tasks == 0
                and self._evidence_queue.unfinished_tasks == 0
                and self._persistence_is_idle()
            ):
                return True
            time.sleep(0.005)
        return (
            self._task_queue.unfinished_tasks == 0
            and self._demographics_queue.unfinished_tasks == 0
            and self._evidence_queue.unfinished_tasks == 0
            and self._persistence_is_idle()
        )

    def close(self, drain=True, timeout=10.0):
        if self._closed:
            return
        self._closed = True
        if drain:
            self.wait_for_idle(timeout=timeout)
        if self._worker is not None and self._worker.is_alive():
            self._task_queue.put(self._stop_token)
            self._worker.join(timeout=timeout)
        if self._demographics_worker is not None and self._demographics_worker.is_alive():
            self._demographics_queue.put(self._stop_token)
            self._demographics_worker.join(timeout=timeout)
        if self._evidence_worker is not None and self._evidence_worker.is_alive():
            self._evidence_queue.put(self._stop_token)
            self._evidence_worker.join(timeout=timeout)
        if self._evidence_process is not None:
            try:
                self._evidence_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._evidence_process.terminate()
                self._evidence_process.wait(timeout=2.0)
            finally:
                if self._evidence_process.stdin is not None:
                    self._evidence_process.stdin.close()
                if self._evidence_process.stdout is not None:
                    self._evidence_process.stdout.close()
        if self.persistence_store is not None:
            with self._lock:
                identity_ids = list(self.identities)
            for identity_id in identity_ids:
                self.save_database(identity_id)
            if drain:
                self._wait_for_persistence_idle(timeout=timeout)
            with self._persistence_condition:
                if not drain:
                    self._pending_persistence.clear()
                self._persistence_stopping = True
                self._persistence_condition.notify_all()
            if self._persistence_worker is not None and self._persistence_worker.is_alive():
                self._persistence_worker.join(timeout=timeout)
