"""Prediction application requests/results。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from ..domain import ModelRuntimeInfo, PredictionModule


@dataclass(frozen=True)
class IndexedPredictionFile:
    path: Path
    index: int


@dataclass(frozen=True)
class LoadedPredictionImage:
    image: np.ndarray
    source_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float32)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("Prediction image must be a non-empty 2D array")
        if not self.source_paths:
            raise ValueError("Prediction image requires at least one source path")
        object.__setattr__(self, "image", image)


@dataclass(frozen=True)
class PreprocessedPredictionInput:
    values: np.ndarray
    steps: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.asarray(self.values, dtype=np.float32))


@dataclass(frozen=True)
class PredictImageRequest:
    image: np.ndarray
    module: PredictionModule
    model_path: Path
    allow_unsafe_lambda: bool = False
    precision_policy: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class PredictPreparedInputRequest:
    values: np.ndarray
    module: PredictionModule
    model_path: Path
    preprocess_steps: tuple[Mapping[str, Any], ...] = ()
    allow_unsafe_lambda: bool = False
    precision_policy: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class ImagePredictionResult:
    outputs: Mapping[str, Any]
    model_input: np.ndarray
    preprocess_steps: tuple[Mapping[str, Any], ...] = ()
    runtime: ModelRuntimeInfo | None = None


@dataclass(frozen=True)
class PredictFileBatchRequest:
    paths: tuple[Path, ...]
    module: PredictionModule
    model_path: Path
    allow_unsafe_lambda: bool = False
    precision_policy: str | None = None
    timeout_seconds: float | None = None


PredictionItemStatus = Literal["succeeded", "failed", "cancelled"]


@dataclass(frozen=True)
class FilePredictionResult:
    paths: tuple[Path, ...]
    status: PredictionItemStatus
    prediction: ImagePredictionResult | None = None
    error_message: str = ""


@dataclass(frozen=True)
class PredictMultipleFilesRequest:
    batches: tuple[tuple[Path, ...], ...]
    module: PredictionModule
    model_path: Path
    continue_on_error: bool = True
    allow_unsafe_lambda: bool = False
    precision_policy: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class PredictionProgress:
    completed: int
    total: int
    current_paths: tuple[Path, ...] = ()
    status: PredictionItemStatus | Literal["running"] = "running"
    message: str = ""

    @property
    def fraction(self) -> float:
        return 1.0 if self.total == 0 else min(1.0, self.completed / self.total)


@dataclass(frozen=True)
class MultiplePredictionResult:
    items: tuple[FilePredictionResult, ...] = field(default_factory=tuple)
    cancelled: bool = False

    @property
    def failed_count(self) -> int:
        return sum(item.status == "failed" for item in self.items)


@dataclass(frozen=True)
class PredictionExportItem:
    """Framework-neutral snapshot of one completed legacy prediction row."""

    filename: str
    filepath: str
    stack_count: int
    timestamp: str | None
    processing_time: float
    confidence: float | None
    prediction_data: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PredictionArrayExportRequest:
    path: Path
    values: np.ndarray
    fmt: str = "%.6g"
    header: str = ""
    comments: str = "# "
