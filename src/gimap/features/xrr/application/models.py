"""Framework-neutral XRR application requests and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..domain import SpecularGeometry


SourceKind = Literal["auto", "nxs", "cbf"]
AngleMode = Literal["linear", "nxs_dataset"]
Aggregation = Literal["sum", "mean"]


@dataclass(frozen=True)
class XrrSeriesSpec:
    source_path: Path
    source_kind: SourceKind = "auto"
    pattern: str = "*.cbf"
    angle_mode: AngleMode = "linear"
    theta_start_deg: float = 0.0
    theta_step_deg: float = 0.01
    angle_dataset_path: str = ""


@dataclass(frozen=True)
class XrrFrameRef:
    path: Path
    frame_index: int
    sequence_index: int
    theta_deg: float | None = None

    @property
    def label(self) -> str:
        suffix = f" · frame {self.frame_index + 1}" if self.frame_index else ""
        return f"{self.path.name}{suffix}"


@dataclass(frozen=True)
class XrrDetectorFrame:
    data: np.ndarray
    invalid_mask: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class XrrSeriesInspection:
    frame_count: int
    first_ref: XrrFrameRef
    first_frame: XrrDetectorFrame


@dataclass(frozen=True)
class XrrExtractionSettings:
    radius_px: int = 0
    aggregation: Aggregation = "sum"


@dataclass(frozen=True)
class XrrExtractionRequest:
    series: XrrSeriesSpec
    geometry: SpecularGeometry
    extraction: XrrExtractionSettings


@dataclass(frozen=True)
class XrrPoint:
    sequence_index: int
    source_name: str
    frame_index: int
    theta_deg: float
    qz_inv_angstrom: float
    intensity: float | None
    roi_center_x_px: float
    roi_center_y_px: float
    valid_pixels: int


@dataclass(frozen=True)
class XrrExtractionProgress:
    completed: int
    total: int
    point: XrrPoint
    preview: np.ndarray | None = None
    preview_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class XrrExtractionResult:
    points: tuple[XrrPoint, ...]

    @property
    def qz(self) -> np.ndarray:
        return np.asarray([point.qz_inv_angstrom for point in self.points], dtype=float)

    @property
    def intensity(self) -> np.ndarray:
        return np.asarray(
            [float("nan") if point.intensity is None else point.intensity for point in self.points],
            dtype=float,
        )


@dataclass(frozen=True)
class XrrCalibrationGeometry:
    """The geometry of the last applied Geometry Calibration, in the units of the XRR window
    (``None``: not stored). Pixel indices as numpy rows/columns, row 0 at the top."""

    distance_mm: float | None = None
    energy_kev: float | None = None
    pixel_size_x_um: float | None = None
    pixel_size_y_um: float | None = None
    beam_center_x_px: float | None = None
    beam_center_y_px: float | None = None
    source_image: str = ""
    timestamp: str = ""
    image_shape: tuple[int, int] | None = None
    """(rows, columns) of the calibrated frame; ``None`` for calibrations that did not record it."""
    detector: str = ""


@dataclass(frozen=True)
class ExportXrrCurveRequest:
    """The points to write; ``settings`` (the extraction that made them), ``geometry_sources``
    (where each geometry value came from: file, last calibration, default, typed …) and the
    ``calibration`` those values were taken from go into the JSON record next to the CSV."""

    path: Path
    result: XrrExtractionResult
    settings: XrrExtractionRequest | None = None
    geometry_sources: dict[str, str] = field(default_factory=dict)
    calibration: XrrCalibrationGeometry | None = None


@dataclass(frozen=True)
class ExportedXrrCurve:
    csv_path: Path
    record_path: Path | None = None


__all__ = [
    "ExportXrrCurveRequest",
    "ExportedXrrCurve",
    "XrrCalibrationGeometry",
    "XrrDetectorFrame",
    "XrrExtractionProgress",
    "XrrExtractionRequest",
    "XrrExtractionResult",
    "XrrExtractionSettings",
    "XrrFrameRef",
    "XrrPoint",
    "XrrSeriesInspection",
    "XrrSeriesSpec",
]
