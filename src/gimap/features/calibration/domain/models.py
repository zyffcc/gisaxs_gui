"""Calibration domain models owned by the Calibration feature."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from src.gimap.shared.detector_io.models import DetectorImage  # noqa: F401  (re-exported)
from src.gimap.shared.geometry.conventions import index_center_from_canonical
from src.gimap.shared.version import SOFTWARE

CENTER_CONVENTION_KEY = "center_convention"
"""Where an exported calibration says how its beam centres are counted: ``metadata["center_convention"]``.

It is written inside ``metadata``, the one part of the file a reader of format version 1 does not check key by
key, so a GIMaP from before this field still reads a file written now.
"""
INDEX_CENTER_CONVENTION = {
    "name": "index",
    "description": (
        "center_x_px and center_y_px of every candidate count pixel centres from 0: the centre of the first "
        "pixel is (0.0, 0.0), x is the column and y the row, row 0 at the top. GIMaP's Geometry Calibration "
        "and Analyze windows and its instrument profiles put pixel corners on integers (the centre of the "
        "first pixel is (0.5, 0.5)): the same beam centre reads 0.5 px higher in x and in y there."
    ),
}
"""The convention of ``CalibrationCandidate.center_x_px`` / ``center_y_px``, in memory and in the file."""
UNKNOWN_CENTER_CONVENTION = "This calibration file states a beam-centre convention this version does not know."
MANUAL_REFINEMENT_WARNING = "Geometry was manually adjusted after automatic calibration."
TILT_KEYS = ("tilt_refined", "rot1_deg", "rot2_deg", "rot3_deg", "rot1_sigma_deg", "rot2_sigma_deg",
             "point_rms_px", "untilted_point_rms_px")
"""Candidate keys written only for a tilted detector (``CalibrationCandidate.to_dict``)."""


def _stated_center_convention(metadata: dict[str, Any]) -> str:
    """Take the convention out of the file's ``metadata``; a file from before the field holds indices."""
    stated = metadata.pop(CENTER_CONVENTION_KEY, None)
    if stated is None:
        return "index"
    name = stated.get("name") if isinstance(stated, dict) else stated
    if name not in ("index", "canonical"):
        raise ValueError(UNKNOWN_CENTER_CONVENTION)
    return name


@dataclass(frozen=True)
class CalibrationStandard:
    key: str
    display_name: str
    q_values_inv_angstrom: tuple[float, ...]
    relative_intensities: Optional[tuple[float, ...]] = None
    notes: str = ""


@dataclass
class MatchedRing:
    theoretical_index: int
    q_inv_angstrom: float
    observed_radius_px: float
    predicted_radius_px: float
    residual_px: float
    prominence: float = 0.0
    coverage: float = 0.0


@dataclass
class CalibrationCandidate:
    standard_key: str
    center_x_px: float
    center_y_px: float
    distance_mm: float
    detector_rotation_deg: float = 0.0
    matched_ring_count: int = 0
    rms_residual_px: float = float("inf")
    distance_consistency: float = 0.0
    azimuthal_coverage: float = 0.0
    score: float = 0.0
    confidence: str = "Low"
    warnings: list[str] = field(default_factory=list)
    matched_rings: list[MatchedRing] = field(default_factory=list)
    detected_peak_radii_px: list[float] = field(default_factory=list)
    valid_data_fraction: float = 0.0
    center_method: str = ""
    center_support: float = 0.0
    harmonic_quality: float = 0.0
    center_x_sigma_px: Optional[float] = None
    center_y_sigma_px: Optional[float] = None
    distance_sigma_mm: Optional[float] = None
    uncertainty_method: str = ""
    """How the 1σ of the centre and the distance were found (``optimizer.estimate_uncertainty``): ``ring jackknife``
    with three or more matched rings, else ``least squares``; empty when none could be."""
    tilt_refined: bool = False
    """Whether rot1 and rot2 were fitted with the centre and the distance (``tilt_fit.refine_ring_geometry``)."""
    rot1_deg: float = 0.0
    """pyFAI's rot1, rot2, rot3 of the detector (degrees). ``distance_mm`` stays the distance from the sample to the
    direct beam and ``center_*`` the direct beam, whatever the tilt (a ``.poni`` is converted when written)."""
    rot2_deg: float = 0.0
    rot3_deg: float = 0.0
    rot1_sigma_deg: Optional[float] = None
    rot2_sigma_deg: Optional[float] = None
    point_rms_px: Optional[float] = None
    """RMS residual on the ring points used by the final fit."""
    untilted_point_rms_px: Optional[float] = None
    """A zero-tilt refit on the final points, or the original circle-fit RMS in a legacy tilted record."""
    initial_point_rms_px: Optional[float] = None
    """Starting geometry on the same final points. Absent in older calibrations."""

    def clear_fit_statistics(self) -> None:
        """A manually adjusted centre or distance no longer describes the fitted geometry."""
        self.initial_point_rms_px = self.point_rms_px = self.untilted_point_rms_px = None
        self.center_x_sigma_px = self.center_y_sigma_px = self.distance_sigma_mm = None
        self.rot1_sigma_deg = self.rot2_sigma_deg = None
        self.uncertainty_method = ""

    def to_dict(self) -> dict[str, Any]:
        """Version 1 candidate fields. The enclosing record keeps newer diagnostics in metadata."""
        values = asdict(self)
        values.pop("initial_point_rms_px")
        if not (self.tilt_refined or self.rot1_deg or self.rot2_deg or self.rot3_deg):
            for key in TILT_KEYS:
                values.pop(key, None)
        return values

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationCandidate":
        values = dict(payload)
        values["matched_rings"] = [MatchedRing(**item) for item in values.get("matched_rings", [])]
        candidate = cls(**values)
        if MANUAL_REFINEMENT_WARNING in candidate.warnings:
            candidate.clear_fit_statistics()
        return candidate


def _point_fit_geometry(candidate: CalibrationCandidate) -> dict:
    """Identify the geometry and ring assignment, even if an older reader reorders the candidates."""
    return {
        "standard_key": candidate.standard_key,
        "center_x_px": candidate.center_x_px,
        "center_y_px": candidate.center_y_px,
        "distance_mm": candidate.distance_mm,
        "rotations_deg": [candidate.rot1_deg, candidate.rot2_deg, candidate.rot3_deg],
        "theoretical_indices": [ring.theoretical_index for ring in candidate.matched_rings],
    }


@dataclass
class CalibrationResult:
    source_image: str
    source_image_size: int
    source_image_mtime_ns: int
    image_sha256: str
    energy_kev: float
    wavelength_angstrom: float
    detector_name: Optional[str]
    pixel_size_x_m: float
    pixel_size_y_m: float
    selected_candidate: CalibrationCandidate
    candidates: list[CalibrationCandidate]
    calibration_timestamp: str
    software_version: str = SOFTWARE
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The exported record. The numbers and their keys are those of every earlier export; the file starts
        with what it is and with the convention of its beam centres, so nobody has to guess it."""
        body = vars(self).copy()
        body["selected_candidate"] = self.selected_candidate.to_dict()
        body["candidates"] = [c.to_dict() for c in self.candidates]
        metadata = deepcopy(body.pop("metadata") or {})
        metadata.pop(CENTER_CONVENTION_KEY, None)
        metadata.pop("ring_point_fits", None)
        point_fits = []
        for candidate in (self.selected_candidate, *self.candidates):
            if candidate.initial_point_rms_px is not None:
                fit = {"geometry": _point_fit_geometry(candidate), "initial_point_rms_px": candidate.initial_point_rms_px,
                       "point_rms_px": candidate.point_rms_px}
                if fit not in point_fits:
                    point_fits.append(fit)
        if point_fits:
            metadata["ring_point_fits"] = point_fits
        return {
            "format": "gimap-geometry-calibration",
            "format_version": 1,
            "metadata": {CENTER_CONVENTION_KEY: dict(INDEX_CENTER_CONVENTION), **metadata},
            **body,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationResult":
        """Read an exported record: one without ``center_convention`` (every file before the field) holds
        pixel-centre indices, as does one that names "index"; "canonical" centres are converted."""
        values = {key: value for key, value in payload.items() if key not in {"format", "format_version"}}
        values["metadata"] = dict(values.get("metadata") or {})
        convention = _stated_center_convention(values["metadata"])
        point_fits = values["metadata"].pop("ring_point_fits", [])
        values["selected_candidate"] = CalibrationCandidate.from_dict(values["selected_candidate"])
        values["candidates"] = [CalibrationCandidate.from_dict(item) for item in values.get("candidates", [])]
        for candidate in (values["selected_candidate"], *values["candidates"]):
            if convention == "canonical":
                candidate.center_x_px, candidate.center_y_px = index_center_from_canonical(
                    candidate.center_x_px, candidate.center_y_px
                )
            if MANUAL_REFINEMENT_WARNING not in candidate.warnings:
                geometry = _point_fit_geometry(candidate)
                for fit in point_fits:
                    if fit["geometry"] == geometry:
                        candidate.initial_point_rms_px = fit["initial_point_rms_px"]
                        candidate.point_rms_px = fit["point_rms_px"]
                        break
        return cls(**values)
