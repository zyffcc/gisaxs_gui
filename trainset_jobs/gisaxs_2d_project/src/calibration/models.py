from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class DetectorImage:
    """A detector frame in the application's display orientation.

    ``mask`` is True for invalid pixels. Data are float32; NXS masked pixels
    remain NaN, matching the existing GIWAXS loader.
    """

    data: np.ndarray
    mask: Optional[np.ndarray]
    source_path: Path
    detector_name: Optional[str] = None
    pixel_size_x_m: Optional[float] = None
    pixel_size_y_m: Optional[float] = None
    energy_kev: Optional[float] = None
    wavelength_angstrom: Optional[float] = None
    distance_m: Optional[float] = None
    beam_center_x_px: Optional[float] = None
    beam_center_y_px: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    software_version: str = "GIMaP v0.0.2"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["format"] = "gimap-geometry-calibration"
        payload["format_version"] = 1
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationResult":
        values = {key: value for key, value in payload.items() if key not in {"format", "format_version"}}
        values["selected_candidate"] = CalibrationCandidate.from_dict(values["selected_candidate"])
        values["candidates"] = [CalibrationCandidate.from_dict(item) for item in values.get("candidates", [])]
        return cls(**values)
