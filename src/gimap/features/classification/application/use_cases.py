"""Classification framework-neutral use cases。"""

from __future__ import annotations

from pathlib import Path

from .models import (
    BuildFeatureMatrixRequest,
    BuildClassificationModelPackageRequest,
    ClassificationPredictionOutput,
    ClassificationCsvRequest,
    ClassificationSessionRequest,
    ClassificationPredictionRequest,
    ClassificationTrainingOutput,
    ClassificationTrainingRequest,
    EmbeddingRequest,
    ImportDatasetRequest,
    SaveClassificationModelRequest,
)
from .ports import (
    ClassificationDatasetPort,
    ClassificationArtifactRepository,
    ClassificationModelRepository,
    ClassifierCatalogPort,
    ClassifierPredictorPort,
    ClassifierTrainerPort,
    EmbeddingPort,
    RuntimeVersionPort,
)
from ..domain import SavedModelPackage


class ImportClassificationDataset:
    def __init__(self, datasets: ClassificationDatasetPort):
        self._datasets = datasets

    def execute(self, request: ImportDatasetRequest, *, on_progress=None, is_cancelled=None):
        return self._datasets.import_sources(
            request.sources,
            on_progress=on_progress,
            is_cancelled=is_cancelled,
        )


class BuildClassificationFeatures:
    def __init__(self, datasets: ClassificationDatasetPort):
        self._datasets = datasets

    def execute(self, request: BuildFeatureMatrixRequest):
        return self._datasets.build_feature_matrix(
            request.samples,
            request.preprocessing,
            require_labels=request.require_labels,
        )


class ValidateClassificationDataset:
    def __init__(self, datasets: ClassificationDatasetPort):
        self._datasets = datasets

    def execute(self, samples):
        return self._datasets.validate_dataset(tuple(samples))


class SummarizeClassificationDataset:
    def __init__(self, datasets: ClassificationDatasetPort):
        self._datasets = datasets

    def execute(self, samples):
        return self._datasets.summarize_by_label(tuple(samples))


class EstimateClassificationFeatureMemory:
    def execute(self, matrix) -> str:
        bytes_used = int(matrix.X.shape[0] * matrix.X.shape[1] * 8)
        if bytes_used < 1024:
            return f"{bytes_used} B"
        if bytes_used < 1024**2:
            return f"{bytes_used / 1024:.1f} KB"
        return f"{bytes_used / 1024**2:.1f} MB"


class ListClassificationAlgorithms:
    def __init__(self, catalog: ClassifierCatalogPort):
        self._catalog = catalog

    def execute(self):
        return self._catalog.default_algorithm_configs()


class TrainClassifiers:
    def __init__(self, trainer: ClassifierTrainerPort):
        self._trainer = trainer

    def execute(self, request: ClassificationTrainingRequest, *, on_progress=None):
        experiment = self._trainer.train(request, on_progress=on_progress)
        experiment.preprocessing_config = request.preprocessing
        experiment.input_shape = request.feature_matrix.input_shape
        experiment.warnings.extend(request.feature_matrix.warnings)
        return ClassificationTrainingOutput(experiment, request.feature_matrix)

    def cancel(self) -> bool:
        return self._trainer.cancel()


class ComputeClassificationEmbedding:
    def __init__(self, embedding: EmbeddingPort):
        self._embedding = embedding

    def execute(self, request: EmbeddingRequest):
        return self._embedding.embed(request)

    def cancel(self) -> bool:
        return self._embedding.cancel()


class PredictClassification:
    def __init__(self, predictor: ClassifierPredictorPort):
        self._predictor = predictor

    def execute(self, request: ClassificationPredictionRequest):
        return ClassificationPredictionOutput(self._predictor.predict(request))


class SaveClassificationModel:
    def __init__(self, repository: ClassificationModelRepository):
        self._repository = repository

    def execute(self, request: SaveClassificationModelRequest) -> Path:
        path = Path(request.path)
        self._repository.save(path, request.package)
        return path


class LoadClassificationModel:
    def __init__(self, repository: ClassificationModelRepository):
        self._repository = repository

    def execute(self, path: Path):
        return self._repository.load(Path(path))


class SaveClassificationSession:
    def __init__(self, repository: ClassificationArtifactRepository):
        self._repository = repository

    def execute(self, request: ClassificationSessionRequest) -> Path:
        path = Path(request.path)
        self._repository.save_session(path, request.values)
        return path


class LoadClassificationSession:
    def __init__(self, repository: ClassificationArtifactRepository):
        self._repository = repository

    def execute(self, path: Path) -> dict:
        return self._repository.load_session(Path(path))


class ExportClassificationCsv:
    def __init__(self, repository: ClassificationArtifactRepository):
        self._repository = repository

    def execute(self, request: ClassificationCsvRequest) -> Path:
        path = Path(request.path)
        self._repository.export_csv(path, request.columns, request.rows)
        return path


class BuildClassificationModelPackage:
    def __init__(self, versions: RuntimeVersionPort):
        self._versions = versions

    def execute(self, request: BuildClassificationModelPackageRequest):
        return SavedModelPackage(
            pipeline=request.pipeline,
            algorithm_id=request.algorithm_id,
            display_name=request.display_name,
            class_names=list(request.class_names),
            data_type=request.data_type,
            input_shape=request.input_shape,
            preprocessing_config=request.preprocessing,
            projection_config=request.projection,
            algorithm_parameters=dict(request.algorithm_parameters),
            sklearn_version=self._versions.version("scikit-learn"),
            numpy_version=self._versions.version("numpy"),
            software_version=request.software_version,
            training_date=request.resolved_training_date(),
            validation_config=request.validation,
            evaluation_metrics=dict(request.evaluation_metrics),
        )
