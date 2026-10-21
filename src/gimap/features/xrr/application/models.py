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


# What a series consists of, as ``(template, values)`` for the inspection text (English templates;
# a series that is one NeXus / HDF5 source uses the reader's own summary instead).
ONE_FRAME_EACH = "{files} files · 1 frame each"
SETS_ONE_FRAME_EACH = "{sets} sets of {modules} module files · 1 frame each"
FRAMES_IN_ALL = "{files} files · {frames} frames in all"
MORE_REMARKS = "{count} more files have a remark like these"
# A series stored as one frame per file: files of the folder that are numbered like it but are no frame
# of it. Said because the angles of a linear scan are counted by position in the series.
FILES_LEFT_OUT = (
    "{count} files numbered like the series are left out (another frame size, or not one frame): {names}. "
    "With angles from start and step, the frames after a file that is left out get the angle of their position"
)


@dataclass(frozen=True)
class XrrFrameRef:
    path: Path
    frame_index: int
    sequence_index: int
    theta_deg: float | None = None
    name: str = ""
    """What the source is called when that is not its file name (a set of module files: its scan)."""
    frames_in_file: int = 1
    """Frames the source of this frame holds: with more than one the label names the frame, also frame 1."""

    @property
    def label(self) -> str:
        numbered = self.frame_index or self.frames_in_file > 1
        suffix = f" · frame {self.frame_index + 1}" if numbered else ""
        return f"{self.name or self.path.name}{suffix}"


@dataclass(frozen=True)
class XrrSeriesLayout:
    """What the frames of a series are stored in: found when the series is inspected."""

    files: int
    """Sources the frames come from (a set of module files, or a master with its data files, is one)."""
    frames: int
    summary: tuple[str, dict]
    """``(template, values)``: ``("{files} files · 1 frame each", {"files": 12})``, or the reader's own
    summary of one source (``"{modules} module files · {frames} frames · stitched"``)."""
    warnings: tuple[tuple[str, dict], ...] = ()
    """``(template, values)`` sentences of the reader: a module file missing, unequal frame counts …"""
    members: tuple[str, ...] = ()
    """Every file the frames are read from, in the order of the frames (all module files of a set, a master
    and its data files): a point of the curve names only the first file of its source."""

    def summary_text(self) -> str:
        """The summary in English, for records (the window translates the template)."""
        return self.summary[0].format(**self.summary[1])

    def warning_texts(self) -> tuple[str, ...]:
        return tuple(template.format(**values) for template, values in self.warnings)


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
    layout: XrrSeriesLayout | None = None
    """What the series consists of (``None``: the repository does not say)."""


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
    layout: XrrSeriesLayout | None = None
    """What the series being extracted is stored in (so a run that is stopped knows it too)."""


@dataclass(frozen=True)
class XrrExtractionResult:
    points: tuple[XrrPoint, ...]
    layout: XrrSeriesLayout | None = None
    """What the series the points were extracted from was stored in, as the extraction found it: its
    files and what the reader said about them (frames left out …). ``None``: the repository does not say."""

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
    "XrrSeriesLayout",
    "XrrSeriesSpec",
]
