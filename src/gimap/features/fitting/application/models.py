"""Fitting file use cases 的 requests/results。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

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
