"""Classification application ports。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ..models import (
    ClassificationPredictionRequest,
    ClassificationTrainingRequest,
    EmbeddingRequest,
    EmbeddingResult,
    ImportedDataset,
)
from ...domain import (
    AlgorithmConfig,
    ClassificationSample,
    DatasetSource,
    DatasetSummary,
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

    def validate_dataset(
        self, samples: tuple[ClassificationSample, ...]
    ) -> DatasetSummary: ...

    def summarize_by_label(
        self, samples: tuple[ClassificationSample, ...]
    ) -> dict[str, dict[str, object]]: ...


class ClassifierCatalogPort(Protocol):
    def default_algorithm_configs(self) -> list[AlgorithmConfig]: ...


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


class ClassificationArtifactRepository(Protocol):
    def save_session(self, path: Path, values: dict[str, Any]) -> None: ...

    def load_session(self, path: Path) -> dict[str, Any]: ...

    def export_csv(
        self,
        path: Path,
        columns: tuple[str, ...],
        rows: tuple[tuple[object, ...], ...],
    ) -> None: ...


class RuntimeVersionPort(Protocol):
    def version(self, distribution: str) -> str: ...
