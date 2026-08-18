"""Classification typed presentation state。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..application import DatasetSummary, ExperimentResult, FeatureMatrix, SavedModelPackage


Status = Literal["idle", "loading", "running", "ready", "cancelled", "error"]


@dataclass(frozen=True)
class ClassificationState:
    dataset_status: Status = "idle"
    samples: tuple = ()
    summary: DatasetSummary = field(default_factory=DatasetSummary)
    training_status: Status = "idle"
    experiment: ExperimentResult | None = None
    feature_matrix: FeatureMatrix | None = None
    embedding_status: Status = "idle"
    embedding: object | None = None
    prediction_status: Status = "idle"
    predictions: tuple = ()
    active_package: SavedModelPackage | None = None
    progress: float = 0.0
    status_message: str = "Ready"
    error_message: str | None = None
