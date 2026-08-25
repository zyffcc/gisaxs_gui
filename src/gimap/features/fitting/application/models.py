"""Fitting file use cases 的 requests/results。"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class LoadScatteringFileRequest:
    path: Path
    frame_index: int = 0
    stack_count: int = 1


InSituSourceKind = Literal["cbf", "nxs"]
_INSITU_FRAME_MARKER = "::gimap-frame="


@dataclass(frozen=True)
class DiscoverInSituFramesRequest:
    """Describe one detector source tree without loading detector images."""

    root: Path
    source_kind: InSituSourceKind
    pattern: str = ""
    recursive: bool = True
    expected_nxs_modules: int = 1

    def __post_init__(self) -> None:
        if self.expected_nxs_modules < 1:
            raise ValueError("expected_nxs_modules must be at least one")


@dataclass(frozen=True)
class InSituSourceFrame:
    """Lightweight locator with a JSON-safe token for one logical frame."""

    path: Path
    frame_index: int = 0
    source_kind: InSituSourceKind = "cbf"
    module_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")

    @property
    def token(self) -> str:
        path = str(self.path)
        if self.source_kind == "cbf" and self.frame_index == 0:
            return path
        return f"{path}{_INSITU_FRAME_MARKER}{self.frame_index}"

    @property
    def display_name(self) -> str:
        if self.source_kind == "nxs":
            return f"{self.path.name} · frame {self.frame_index + 1}"
        return self.path.name

    @classmethod
    def from_token(cls, token: str) -> "InSituSourceFrame":
        path_text, marker, frame_text = str(token).rpartition(_INSITU_FRAME_MARKER)
        if not marker:
            path = Path(token)
            kind: InSituSourceKind = "nxs" if path.suffix.lower() == ".nxs" else "cbf"
            return cls(path=path, source_kind=kind)
        path = Path(path_text)
        kind = "nxs" if path.suffix.lower() == ".nxs" else "cbf"
        return cls(path=path, frame_index=int(frame_text), source_kind=kind)


@dataclass(frozen=True)
class ScatteringSequenceInfo:
    """Navigation metadata for one logical detector-file sequence."""

    source_path: Path
    logical_path: Path
    series_paths: tuple[Path, ...]
    frame_count: int

    @property
    def uses_internal_frames(self) -> bool:
        return len(self.series_paths) > 1


@dataclass(frozen=True)
class ScatteringFileData:
    image: np.ndarray
    source_path: Path
    source_files: tuple[Path, ...]
    frame_index: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float32)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("Scattering image must be a non-empty 2D array")
        object.__setattr__(self, "image", image)


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


CurveOperationResult = OperationResult[CurveData]
ScatteringOperationResult = OperationResult[ScatteringFileData]
ExportOperationResult = OperationResult[ExportedFitResult]
