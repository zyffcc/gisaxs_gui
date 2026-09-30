"""Requests and results of the Fitting file use cases (curves, fit results, in-situ series)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np

from .errors import FileOperationError
from ..domain import CurveData


T = TypeVar("T")


@dataclass(frozen=True)
class OperationResult(Generic[T]):
    value: T | None = None
    error: FileOperationError | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.error is None):
            raise ValueError("OperationResult must contain exactly one of value or error")

    @property
    def succeeded(self) -> bool:
        return self.error is None


InSituSourceKind = Literal["curve"]
_INSITU_FRAME_MARKER = "::gimap-frame="
CURVE_SUFFIXES = (".dat", ".txt")
"""1D curve files an in-situ series can fit (Analyze writes ``*_fit_input.dat``)."""
DEFAULT_CURVE_PATTERN = "*_fit_input.dat"


@dataclass(frozen=True)
class DiscoverInSituFramesRequest:
    """A folder of curve files (``pattern``, optionally below child folders)."""

    root: Path
    pattern: str = DEFAULT_CURVE_PATTERN
    recursive: bool = False


@dataclass(frozen=True)
class InSituSourceFrame:
    """One curve of an in-situ series, with a JSON-safe token."""

    path: Path
    frame_index: int = 0
    source_kind: InSituSourceKind = "curve"

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")

    @property
    def token(self) -> str:
        path = str(self.path)
        if self.frame_index == 0:
            return path
        return f"{path}{_INSITU_FRAME_MARKER}{self.frame_index}"

    @property
    def display_name(self) -> str:
        return self.path.name

    @classmethod
    def from_token(cls, token: str) -> "InSituSourceFrame":
        path_text, marker, frame_text = str(token).rpartition(_INSITU_FRAME_MARKER)
        if not marker:
            return cls(path=Path(token))
        return cls(path=Path(path_text), frame_index=int(frame_text))


@dataclass(frozen=True)
class LoadCurveRequest:
    path: Path
    q_source_unit: str = "angstrom"


@dataclass(frozen=True)
class ExportFitResultRequest:
    path: Path
    q: np.ndarray
    intensity: np.ndarray
    header_lines: tuple[str, ...] = ()
    x_column_name: str = "q (nm^-1)"
    y_column_name: str = "Intensity (a.u.)"


@dataclass(frozen=True)
class ExportedFitResult:
    path: Path
    row_count: int
    delimiter: str


@dataclass(frozen=True)
class FigureSeries:
    """One plotted layer of a curve figure: markers (``scatter``) or a ``line``."""

    label: str
    x: np.ndarray
    y: np.ndarray
    color: str
    style: Literal["scatter", "line"] = "scatter"


@dataclass(frozen=True)
class ExportCurveFigureRequest:
    """A publication figure of the curve plot; the format follows the path suffix."""

    path: Path
    series: tuple[FigureSeries, ...]
    x_label: str
    y_label: str
    x_scale: Literal["linear", "log", "symlog"] = "linear"
    log_y: bool = False


CurveOperationResult = OperationResult[CurveData]
ExportOperationResult = OperationResult[ExportedFitResult]
FigureOperationResult = OperationResult[Path]
