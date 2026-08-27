"""Experimental two-plane metrology for ground positions when feet are hidden.

EXPERIMENTAL: every entry point here is reached only when the launcher's
"Enable 3D Level Detection" checkbox is ticked, which sets
``--enable-3d-level-detection``.  When the flag is absent this module is never
imported by the tracker, no elevated calibration is read, and no metric height
memory exists.

Why this exists
---------------
When MediaPipe cannot see a person's feet the current fallback extrapolates the
ground point downward from the nose and shoulders.  That baseline is roughly
11% of a person's height, so the extrapolation multiplies landmark jitter by
1/ratio -- up to 25x with the configured floor of 0.04.  Two-plane metrology
replaces the extrapolation with an intersection: given a landmark's *metric*
height above the floor, its pixel maps to a ground position through the
homography of the plane at that height.  Pixel sensitivity then stops depending
on which landmark is used.

What is and is not identifiable
-------------------------------
A landmark pixel alone gives two equations for three unknowns (X, Y, h), so it
can NOT recover both a person's ground position and their unknown height.  This
module is therefore only usable once the metric height of that landmark on that
specific person has already been learned from frames where the feet *were*
directly visible.  ``estimate_ground_position`` returns None rather than guess.

Geometry
--------
For a camera P = K[R|t], the world->image homography of the plane Z = h is

    H_h = K[r1, r2, r3*h + t] = H_0 + h * v * e3^T,    v = K*r3

``v`` is the vertical vanishing point.  The family is affine in h with a rank-1
update, so two calibrated planes determine every intermediate plane.

Two consequences worth knowing:

* A mis-measured elevated-plane height biases the *reported* metric heights but
  cancels out of ground position, because learning and applying share the same
  (equally wrong) ``v``.  The measurement only needs to be consistent.
* The scale-consistency residual detects disagreement *between the two
  calibrations*.  It cannot detect the camera moving after both were captured --
  both become equally wrong and the residual stays zero.  Live drift needs fixed
  reference markers, which is a separate mechanism.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from constants import DEFAULT_THREE_D_MAX_LEAN_DEGREES

# Landmark keys this module understands, ordered lowest-first.  Lower landmarks
# are preferred because real body lean displaces the head far more than the
# knees, and because a height error costs less ground error lower down.  Note
# this is NOT a jitter argument: two-plane metrology has near-flat pixel
# sensitivity across landmarks, unlike the anatomical-ratio extrapolation.
METROLOGY_LANDMARKS = ("knee_centre", "hip_centre", "shoulder_centre", "nose")

LANDMARK_SOURCES = {
    "knee_centre": "learned_knee_height",
    "hip_centre": "learned_hip_height",
    "shoulder_centre": "learned_shoulder_height",
    "nose": "learned_nose_height",
}

# Evidence strength each source is permitted to carry.  The physical-distance
# gate must consult this rather than treating every point as fact: a soft point
# may inform a score but must never veto a confident appearance match.
SOURCE_EVIDENCE = {
    "direct_feet": "hard",
    "other_camera_direct_feet": "hard",
    "learned_knee_height": "moderate",
    "learned_hip_height": "moderate",
    "learned_shoulder_height": "soft",
    "learned_nose_height": "soft",
    "trajectory_prediction": "soft",
    "anatomical_ratio": "soft",
    "box_bottom": "soft",
    "last_seen": "soft",
    "unavailable": "none",
}

# Plausible metric heights, in cm, for a standing adult.  A learned value
# outside these bounds means the pose, the identity association or the
# calibration is wrong, so the sample is discarded rather than averaged in.
LANDMARK_HEIGHT_BOUNDS = {
    "knee_centre": (30.0, 70.0),
    "hip_centre": (65.0, 125.0),
    "shoulder_centre": (110.0, 175.0),
    "nose": (125.0, 195.0),
}

DEFAULT_RESIDUAL_TOLERANCE = 0.02
DEFAULT_HEIGHT_WINDOW = 31
DEFAULT_MIN_HEIGHT_SAMPLES = 7
DEFAULT_HEIGHT_EMA_ALPHA = 0.05
# Beyond this angle between the observed torso and the expected projected
# vertical, the person is bending or crouching and their landmark heights stop
# being measurable. The value lives in constants.py so the tracker can expose it
# as a CLI default without importing this experimental module on a disabled run.
#
# The angle only responds to lean *across* the camera's viewing azimuth, which
# is precisely the lean that corrupts a measured height; lean along the azimuth
# is close to invisible in the angle and also close to harmless in the height.
# Ordinary walking lean is a few degrees and survives this gate deliberately --
# it is handled by the median window instead, because a walker leans both ways
# and the median of many frames converges on the truth while a single-frame
# rejection rule would simply discard most useful samples.
DEFAULT_MAX_LEAN_DEGREES = DEFAULT_THREE_D_MAX_LEAN_DEGREES
DEFAULT_PIXEL_SIGMA = 3.0


class CalibrationError(Exception):
    """Raised when an elevated-plane calibration is missing or unusable."""


@dataclass(frozen=True)
class PositionEstimate:
    """A ground position plus everything needed to decide how far to trust it.

    ``evidence`` is deliberately separate from ``confidence``.  Callers need a
    three-way policy decision (may veto / may score / must abstain), and burying
    that in a float threshold spreads the policy across every call site.
    """

    point_cm: tuple[float, float] | None
    source: str
    confidence: float
    uncertainty_cm: float
    landmark: str | None = None
    camera_id: str | None = None

    @property
    def evidence(self) -> str:
        return SOURCE_EVIDENCE.get(self.source, "soft")

    def as_log_dict(self) -> dict:
        return {
            "point_cm": None if self.point_cm is None else [round(float(v), 1) for v in self.point_cm],
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
            "uncertainty_cm": round(float(self.uncertainty_cm), 1),
            "landmark": self.landmark,
            "camera_id": self.camera_id,
            "evidence": self.evidence,
        }


UNAVAILABLE = PositionEstimate(None, "unavailable", 0.0, float("inf"))


@dataclass(frozen=True)
class PlaneCalibration:
    """Two-plane geometry for one camera, in world->image form."""

    camera_id: str
    ground_world_to_image: np.ndarray
    vertical_vanishing_point: np.ndarray
    elevated_height_cm: float
    residual: float
    map_size_cm: float
    ground_path: str = ""
    elevated_path: str = ""

    def homography_at_height(self, height_cm: float) -> np.ndarray:
        """World->image homography for the plane Z = height_cm."""
        return self.ground_world_to_image + float(height_cm) * np.outer(
            self.vertical_vanishing_point, (0.0, 0.0, 1.0)
        )


def _load_stored_matrix(path, expected_map_size_cm=None):
    """Read one saved calibration file in the project's image->world form."""
    file_path = Path(path)
    if not file_path.is_file():
        raise CalibrationError(f"Calibration file not found: {file_path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"Calibration file {file_path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict) or "matrix" not in payload:
        raise CalibrationError(f"Calibration file {file_path} has no 'matrix' entry.")
    matrix = np.asarray(payload.get("matrix"), dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise CalibrationError(f"Calibration file {file_path} does not hold a finite 3x3 matrix.")
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise CalibrationError(f"Calibration file {file_path} holds a singular matrix.")
    map_size_cm = payload.get("map_size_cm")
    if expected_map_size_cm is not None and map_size_cm is not None:
        if abs(float(map_size_cm) - float(expected_map_size_cm)) > 1e-6:
            raise CalibrationError(
                f"Calibration file {file_path} was saved for a {map_size_cm} cm map but the "
                f"ground calibration uses {expected_map_size_cm} cm. Both planes must share "
                "one world coordinate system."
            )
    return matrix, payload


def build_plane_calibration(
    camera_id,
    ground_path,
    elevated_path,
    residual_tolerance=DEFAULT_RESIDUAL_TOLERANCE,
):
    """Recover the vertical vanishing point from two saved plane calibrations.

    Both files hold image->world matrices at an arbitrary projective scale, so
    the raw matrices must never be differenced directly.  They are inverted to
    world->image, where columns one and two are K*r1 and K*r2 and are identical
    for every plane parallel to the ground.  Matching those columns fixes the
    relative scale; only then does the third-column difference give h*v.
    """
    ground_matrix, ground_payload = _load_stored_matrix(ground_path)
    map_size_cm = ground_payload.get("map_size_cm")
    elevated_matrix, elevated_payload = _load_stored_matrix(elevated_path, map_size_cm)

    height_cm = elevated_payload.get("plane_height_cm")
    if height_cm is None:
        raise CalibrationError(
            f"Elevated calibration {elevated_path} has no 'plane_height_cm'. The measured "
            "height of the elevated plane above the floor is required."
        )
    try:
        height_cm = float(height_cm)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"Elevated calibration {elevated_path} has a non-numeric height.") from exc
    if not math.isfinite(height_cm) or height_cm <= 0.0:
        raise CalibrationError(
            f"Elevated calibration {elevated_path} height must be a positive number of "
            f"centimetres, got {height_cm}."
        )

    ground_w2i = np.linalg.inv(ground_matrix)
    elevated_w2i = np.linalg.inv(elevated_matrix)

    denominator = float(np.sum(elevated_w2i[:, :2] * elevated_w2i[:, :2]))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise CalibrationError(f"Elevated calibration {elevated_path} is degenerate.")
    scale = float(np.sum(ground_w2i[:, :2] * elevated_w2i[:, :2])) / denominator
    if not math.isfinite(scale) or abs(scale) < 1e-12:
        raise CalibrationError(
            f"Elevated calibration {elevated_path} could not be scale-matched to the ground "
            "plane. The two files are probably not from the same camera."
        )
    elevated_w2i = scale * elevated_w2i

    reference = float(np.linalg.norm(ground_w2i[:, :2]))
    residual = float(np.linalg.norm(elevated_w2i[:, :2] - ground_w2i[:, :2]) / max(reference, 1e-12))
    if residual > float(residual_tolerance):
        raise CalibrationError(
            f"Ground and elevated calibrations for {camera_id} disagree (residual "
            f"{residual:.4f} > {float(residual_tolerance):.4f}). The camera most likely moved "
            "between the two calibrations, the planes are not parallel, or lens distortion is "
            "significant. Recalibrate both planes without moving the camera."
        )

    vertical = (elevated_w2i[:, 2] - ground_w2i[:, 2]) / height_cm
    if not np.all(np.isfinite(vertical)) or float(np.linalg.norm(vertical)) < 1e-12:
        raise CalibrationError(
            f"Could not recover a vertical vanishing point for {camera_id}; the two planes "
            "appear identical."
        )

    return PlaneCalibration(
        camera_id=str(camera_id),
        ground_world_to_image=ground_w2i,
        vertical_vanishing_point=vertical,
        elevated_height_cm=height_cm,
        residual=residual,
        map_size_cm=float(map_size_cm) if map_size_cm is not None else float("nan"),
        ground_path=str(ground_path),
        elevated_path=str(elevated_path),
    )


def ground_from_landmark(calibration, pixel, height_cm):
    """Ground position (cm) of a landmark pixel at a known metric height."""
    homography = calibration.homography_at_height(height_cm)
    try:
        world = np.linalg.solve(homography, np.array([float(pixel[0]), float(pixel[1]), 1.0]))
    except np.linalg.LinAlgError:
        return None
    if not math.isfinite(world[2]) or abs(world[2]) < 1e-9:
        return None
    point = world[:2] / world[2]
    if not np.all(np.isfinite(point)):
        return None
    return (float(point[0]), float(point[1]))


def height_from_pixels(calibration, foot_pixel, landmark_pixel):
    """Metric height of a landmark, given the same person's ground pixel.

    The landmark pixel lies on the image line joining the ground pixel to the
    vertical vanishing point, so h follows from a least-squares solve of
    ``landmark ~ ground + h * v``.

    The ground term must be the foot point re-projected through H_0, not the
    foot pixel normalised to w = 1.  ``h * v`` is expressed in H_0's homogeneous
    scale, so discarding that scale would silently corrupt every height.
    """
    homography = calibration.ground_world_to_image
    try:
        world = np.linalg.solve(
            homography, np.array([float(foot_pixel[0]), float(foot_pixel[1]), 1.0])
        )
    except np.linalg.LinAlgError:
        return None
    if not math.isfinite(world[2]) or abs(world[2]) < 1e-9:
        return None
    ground_xy = world[:2] / world[2]
    ground = homography @ np.array([ground_xy[0], ground_xy[1], 1.0])
    landmark = np.array([float(landmark_pixel[0]), float(landmark_pixel[1]), 1.0])
    numerator = np.cross(landmark, ground)
    denominator = np.cross(landmark, calibration.vertical_vanishing_point)
    scale = float(denominator @ denominator)
    if scale < 1e-12:
        return None
    height = -float(numerator @ denominator) / scale
    if not math.isfinite(height):
        return None
    return height


def expected_vertical_direction(calibration, pixel):
    """Unit image direction that world-vertical projects to at this pixel.

    Perspective makes people at the edges of the frame lean in different
    apparent directions, so the image Y axis is not world-vertical and the
    shoulder line is not a reliable substitute.  The direction toward the
    vertical vanishing point is the calibrated answer.
    """
    vertical = calibration.vertical_vanishing_point
    if abs(vertical[2]) < 1e-9:
        direction = vertical[:2].astype(float)
    else:
        direction = (vertical[:2] / vertical[2]) - np.array([float(pixel[0]), float(pixel[1])])
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None
    return direction / norm


def lean_degrees(calibration, foot_pixel, head_pixel):
    """Angle between the observed torso axis and the expected projected vertical."""
    expected = expected_vertical_direction(calibration, foot_pixel)
    if expected is None:
        return None
    observed = np.array(
        [float(head_pixel[0]) - float(foot_pixel[0]), float(head_pixel[1]) - float(foot_pixel[1])]
    )
    norm = float(np.linalg.norm(observed))
    if norm < 1e-6:
        return None
    cosine = float(np.clip(np.dot(observed / norm, expected), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def position_uncertainty_cm(calibration, pixel, height_cm, height_sigma_cm, pixel_sigma=DEFAULT_PIXEL_SIGMA):
    """Propagate pixel and height uncertainty into a ground-position radius.

    Finite differences on the analytic mapping, which is cheap and avoids
    hand-deriving a Jacobian that would silently rot if the geometry changes.
    """
    base = ground_from_landmark(calibration, pixel, height_cm)
    if base is None:
        return float("inf")
    base_array = np.array(base)
    terms = []
    for offset in ((pixel_sigma, 0.0), (0.0, pixel_sigma)):
        shifted = ground_from_landmark(
            calibration, (pixel[0] + offset[0], pixel[1] + offset[1]), height_cm
        )
        if shifted is None:
            return float("inf")
        terms.append(float(np.linalg.norm(np.array(shifted) - base_array)))
    raised = ground_from_landmark(calibration, pixel, height_cm + max(height_sigma_cm, 1e-6))
    if raised is None:
        return float("inf")
    terms.append(float(np.linalg.norm(np.array(raised) - base_array)))
    return float(math.sqrt(sum(term * term for term in terms)))


@dataclass
class _HeightMemory:
    """Robust per-identity, per-landmark metric height in centimetres.

    Deliberately distinct from the dimensionless anatomical-ratio memory: these
    values are metres-scale physical quantities and the two must never be mixed.
    A median window absorbs outliers from bad pose frames before a slow EMA
    smooths what survives.
    """

    window: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_HEIGHT_WINDOW))
    value: float | None = None
    samples: int = 0

    def observe(self, height_cm, ema_alpha=DEFAULT_HEIGHT_EMA_ALPHA, min_samples=DEFAULT_MIN_HEIGHT_SAMPLES):
        self.window.append(float(height_cm))
        self.samples += 1
        if len(self.window) < min_samples:
            return None
        median = float(np.median(np.asarray(self.window, dtype=float)))
        if self.value is None:
            self.value = median
        else:
            self.value = (1.0 - ema_alpha) * self.value + ema_alpha * median
        return self.value

    @property
    def sigma_cm(self):
        """Spread of the accepted window, floored so confidence never reads as exact."""
        if len(self.window) < 2:
            return 12.0
        spread = float(np.percentile(np.asarray(self.window, dtype=float), 75)) - float(
            np.percentile(np.asarray(self.window, dtype=float), 25)
        )
        return max(2.0, spread)


class ThreeDLevelPositionEstimator:
    """Feature-gated two-plane metrology, learning and estimation.

    The tracker constructs this only when 3D level detection is enabled, so an
    unticked checkbox means no elevated calibration is read and no metric height
    is ever learned or stored.
    """

    def __init__(
        self,
        max_lean_degrees=DEFAULT_MAX_LEAN_DEGREES,
        height_ema_alpha=DEFAULT_HEIGHT_EMA_ALPHA,
        min_height_samples=DEFAULT_MIN_HEIGHT_SAMPLES,
        pixel_sigma=DEFAULT_PIXEL_SIGMA,
    ):
        self.calibrations: dict[str, PlaneCalibration] = {}
        self.heights: dict[tuple, _HeightMemory] = {}
        self.max_lean_degrees = float(max_lean_degrees)
        self.height_ema_alpha = float(height_ema_alpha)
        self.min_height_samples = int(min_height_samples)
        self.pixel_sigma = float(pixel_sigma)
        self._initialised = False

    # ---------------------------------------------------------------- setup

    def initialize(self, camera_calibrations):
        """Build geometry for every camera.

        ``camera_calibrations`` maps camera_id -> (ground_path, elevated_path).
        Raises CalibrationError so the caller can disable the feature for the
        run and report which camera failed, rather than crashing tracking.
        """
        built = {}
        for camera_id, paths in dict(camera_calibrations).items():
            ground_path, elevated_path = paths
            built[str(camera_id)] = build_plane_calibration(camera_id, ground_path, elevated_path)
        self.calibrations = built
        self._initialised = True
        return self

    def validate_calibration(self):
        """Report per-camera health for logging and the launcher status line."""
        return {
            camera_id: {
                "residual": calibration.residual,
                "elevated_height_cm": calibration.elevated_height_cm,
                "ground_path": calibration.ground_path,
                "elevated_path": calibration.elevated_path,
            }
            for camera_id, calibration in self.calibrations.items()
        }

    @property
    def ready(self):
        return self._initialised and bool(self.calibrations)

    # ------------------------------------------------------------- learning

    def _memory_key(self, identity_id, landmark):
        return (int(identity_id), str(landmark))

    def learn_landmark_heights(self, observation):
        """Record metric landmark heights from a frame with directly seen feet.

        Only called when the foot point is a real observation, the identity is
        stable and the person is upright, because every one of those failing
        would bake a wrong height into a long-lived per-identity memory.
        """
        if not self.ready:
            return {}
        identity_id = observation.get("identity_id")
        camera_id = str(observation.get("camera_id"))
        foot_pixel = observation.get("foot_pixel")
        landmarks = observation.get("landmarks") or {}
        calibration = self.calibrations.get(camera_id)
        if calibration is None or identity_id is None or foot_pixel is None:
            return {}
        if not observation.get("foot_is_direct"):
            return {}
        if not observation.get("identity_stable"):
            return {}

        head_pixel = landmarks.get("nose") or landmarks.get("shoulder_centre")
        if head_pixel is not None:
            lean = lean_degrees(calibration, foot_pixel, head_pixel)
            if lean is None or lean > self.max_lean_degrees:
                return {}

        learned = {}
        for name in METROLOGY_LANDMARKS:
            pixel = landmarks.get(name)
            if pixel is None:
                continue
            height = height_from_pixels(calibration, foot_pixel, pixel)
            if height is None:
                continue
            low, high = LANDMARK_HEIGHT_BOUNDS[name]
            if not low <= height <= high:
                continue
            memory = self.heights.setdefault(self._memory_key(identity_id, name), _HeightMemory())
            settled = memory.observe(
                height, ema_alpha=self.height_ema_alpha, min_samples=self.min_height_samples
            )
            if settled is not None:
                learned[name] = settled
        return learned

    def known_heights(self, identity_id):
        if identity_id is None:
            return {}
        result = {}
        for name in METROLOGY_LANDMARKS:
            memory = self.heights.get(self._memory_key(identity_id, name))
            if memory is not None and memory.value is not None:
                result[name] = memory.value
        return result

    def forget_identity(self, identity_id):
        for name in METROLOGY_LANDMARKS:
            self.heights.pop(self._memory_key(identity_id, name), None)

    # ----------------------------------------------------------- estimation

    def candidate_positions(self, observation):
        """Ground positions from every landmark whose height is already known."""
        if not self.ready:
            return []
        camera_id = str(observation.get("camera_id"))
        calibration = self.calibrations.get(camera_id)
        identity_id = observation.get("identity_id")
        landmarks = observation.get("landmarks") or {}
        if calibration is None or identity_id is None:
            return []

        heights = self.known_heights(identity_id)
        if not heights:
            return []

        foot_pixel = observation.get("foot_pixel")
        head_pixel = landmarks.get("nose") or landmarks.get("shoulder_centre")
        lean = None
        if foot_pixel is not None and head_pixel is not None:
            lean = lean_degrees(calibration, foot_pixel, head_pixel)

        candidates = []
        for name in METROLOGY_LANDMARKS:
            pixel = landmarks.get(name)
            height = heights.get(name)
            if pixel is None or height is None:
                continue
            point = ground_from_landmark(calibration, pixel, height)
            if point is None:
                continue
            memory = self.heights.get(self._memory_key(identity_id, name))
            sigma = memory.sigma_cm if memory is not None else 12.0
            uncertainty = position_uncertainty_cm(
                calibration, pixel, height, sigma, pixel_sigma=self.pixel_sigma
            )
            confidence = self._confidence(name, uncertainty, lean, memory)
            candidates.append(
                PositionEstimate(
                    point_cm=point,
                    source=LANDMARK_SOURCES[name],
                    confidence=confidence,
                    uncertainty_cm=uncertainty,
                    landmark=name,
                    camera_id=camera_id,
                )
            )
        return candidates

    def _confidence(self, landmark, uncertainty_cm, lean, memory):
        if not math.isfinite(uncertainty_cm):
            return 0.0
        # Start from how tight the geometry is, then penalise thin evidence and
        # postures the model does not represent.
        score = 1.0 / (1.0 + uncertainty_cm / 15.0)
        if memory is not None and memory.samples < 4 * self.min_height_samples:
            score *= 0.75
        if lean is not None and lean > self.max_lean_degrees:
            # A leaning torso displaces high landmarks far more than low ones.
            index = METROLOGY_LANDMARKS.index(landmark)
            score *= max(0.15, 1.0 - 0.22 * (index + 1))
        return float(max(0.0, min(1.0, score)))

    def estimate_ground_position(self, observation, identity_state=None):
        """Best available metrology position, or an explicit unavailable result.

        Returning ``unavailable`` is the correct answer whenever the geometry is
        not trustworthy: the physical-distance gate treats a missing point as
        "no objection", so no position is strictly safer than a wrong one.
        """
        candidates = self.candidate_positions(observation)
        if not candidates:
            return UNAVAILABLE, []

        # An identity that is not settled may well be the wrong person, and
        # their stored height would then be meaningless. Keep the geometry but
        # refuse to let it carry hard weight.
        unstable = identity_state is not None and identity_state != "confirmed"

        best = candidates[0]  # already lowest-landmark-first
        spread = self._candidate_spread(candidates)
        if spread is not None and spread > 40.0:
            # Positions that drift apart with landmark height indicate lean, bad
            # pose, a wrong height association or calibration error. Trust the
            # lowest landmark and say so with a wider uncertainty.
            best = replace(
                best,
                confidence=best.confidence * 0.5,
                uncertainty_cm=max(best.uncertainty_cm, spread),
            )
        elif spread is not None and spread < 12.0 and len(candidates) > 1:
            best = replace(best, confidence=min(1.0, best.confidence * 1.15))

        if unstable:
            best = replace(best, confidence=best.confidence * 0.5)
        return best, candidates

    @staticmethod
    def _candidate_spread(candidates):
        points = [candidate.point_cm for candidate in candidates if candidate.point_cm is not None]
        if len(points) < 2:
            return None
        array = np.asarray(points, dtype=float)
        centre = array.mean(axis=0)
        return float(np.max(np.linalg.norm(array - centre, axis=1)))

    def close(self):
        self.calibrations = {}
        self.heights = {}
        self._initialised = False
