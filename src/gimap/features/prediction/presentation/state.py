"""Prediction typed display state。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..application import (
    FilePredictionResult,
    ImagePredictionResult,
    LoadedPredictionImage,
)
from ..application import ModelRuntimeInfo, PredictionModule


Status = Literal["idle", "loading", "running", "ready", "cancelled", "error"]


@dataclass(frozen=True)
class PredictionState:
    module_status: Status = "idle"
    modules: tuple[PredictionModule, ...] = ()
    current_module: PredictionModule | None = None
    model_status: Status = "idle"
    model_path: Path | None = None
    model_runtime: ModelRuntimeInfo | None = None
    image_status: Status = "idle"
    current_image: LoadedPredictionImage | None = None
    prediction_status: Status = "idle"
    prediction: ImagePredictionResult | None = None
    batch_status: Status = "idle"
    batch_progress: float = 0.0
    batch_results: tuple[FilePredictionResult, ...] = ()
    error_message: str | None = None
    status_message: str = "Ready"
