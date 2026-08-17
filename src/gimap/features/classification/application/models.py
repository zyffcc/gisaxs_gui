"""Classification application request/result models。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..domain import (
    AlgorithmConfig,
    DatasetSource,
    DatasetSummary,
    ExperimentResult,
    FeatureMatrix,
    PredictionResult,
    PreprocessingConfig,
    ProjectionConfig,
    SavedModelPackage,
    ValidationConfig,
)


@dataclass(frozen=True)
class ImportedDataset:
    samples: tuple
    summary: DatasetSummary


@dataclass(frozen=True)
class ImportDatasetRequest:
    sources: tuple[DatasetSource, ...]


@dataclass(frozen=True)
class BuildFeatureMatrixRequest:
    samples: tuple
    preprocessing: PreprocessingConfig
    require_labels: bool = True


@dataclass(frozen=True)
class ClassificationTrainingRequest:
    feature_matrix: FeatureMatrix
    preprocessing: PreprocessingConfig
    algorithms: tuple[AlgorithmConfig, ...]
    validation: ValidationConfig
    projection: ProjectionConfig
    ranking_metric: str
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class EmbeddingRequest:
    values: np.ndarray
    method: str
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    values: np.ndarray
    method: str


@dataclass(frozen=True)
class ClassificationPredictionRequest:
    feature_matrix: FeatureMatrix
    package: SavedModelPackage


@dataclass(frozen=True)
class ClassificationPredictionOutput:
    items: tuple[PredictionResult, ...]


@dataclass(frozen=True)
class SaveClassificationModelRequest:
    path: Path
    package: SavedModelPackage


@dataclass(frozen=True)
class BuildClassificationModelPackageRequest:
    pipeline: object
    algorithm_id: str
    display_name: str
    class_names: tuple[str, ...]
    data_type: str
    input_shape: tuple[int, int] | None
    preprocessing: PreprocessingConfig
    projection: ProjectionConfig
    algorithm_parameters: dict
    validation: ValidationConfig
    evaluation_metrics: dict[str, float]
    software_version: str = "gisaxs_gui"
    training_date: str = ""

    def resolved_training_date(self) -> str:
        return self.training_date or datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class ClassificationTrainingOutput:
    experiment: ExperimentResult
    feature_matrix: FeatureMatrix
