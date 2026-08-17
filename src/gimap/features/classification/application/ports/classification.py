"""Classification application ports。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ..models import (
    ClassificationPredictionRequest,
    ClassificationTrainingRequest,
    EmbeddingRequest,
    EmbeddingResult,
    ImportedDataset,
)
from ...domain import (
    DatasetSource,
    ExperimentResult,
    FeatureMatrix,
    PredictionResult,
    PreprocessingConfig,
    SavedModelPackage,
)


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


class ClassificationDatasetPort(Protocol):
    def import_sources(
        self,
        sources: tuple[DatasetSource, ...],
        *,
        on_progress: ProgressCallback | None = None,
        is_cancelled: CancelCallback | None = None,
    ) -> ImportedDataset: ...

    def build_feature_matrix(
        self,
        samples: tuple,
        preprocessing: PreprocessingConfig,
        *,
        require_labels: bool,
    ) -> FeatureMatrix: ...


class ClassifierTrainerPort(Protocol):
    def train(
        self,
        request: ClassificationTrainingRequest,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ExperimentResult: ...

    def cancel(self) -> bool: ...


class EmbeddingPort(Protocol):
    def embed(self, request: EmbeddingRequest) -> EmbeddingResult: ...

    def cancel(self) -> bool: ...


class ClassifierPredictorPort(Protocol):
    def predict(
        self, request: ClassificationPredictionRequest
    ) -> tuple[PredictionResult, ...]: ...


class ClassificationModelRepository(Protocol):
    def save(self, path: Path, package: SavedModelPackage) -> None: ...

    def load(self, path: Path) -> SavedModelPackage: ...


class RuntimeVersionPort(Protocol):
    def version(self, distribution: str) -> str: ...
