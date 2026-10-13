"""Calibration domain models owned by the Calibration feature."""

from __future__ import annotations

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationCandidate":
        values = dict(payload)
        values["matched_rings"] = [MatchedRing(**item) for item in values.get("matched_rings", [])]
        return cls(**values)


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
        body = asdict(self)
        metadata = {
            key: value for key, value in (body.pop("metadata") or {}).items() if key != CENTER_CONVENTION_KEY
        }
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
        values["selected_candidate"] = CalibrationCandidate.from_dict(values["selected_candidate"])
        values["candidates"] = [CalibrationCandidate.from_dict(item) for item in values.get("candidates", [])]
        if convention == "canonical":
            for candidate in (values["selected_candidate"], *values["candidates"]):
                candidate.center_x_px, candidate.center_y_px = index_center_from_canonical(
                    candidate.center_x_px, candidate.center_y_px
                )
        return cls(**values)
