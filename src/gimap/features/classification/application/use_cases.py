"""Classification framework-neutral use cases。"""

from __future__ import annotations

from pathlib import Path

from .models import (
    BuildFeatureMatrixRequest,
    BuildClassificationModelPackageRequest,
    ClassificationPredictionOutput,
    ClassificationPredictionRequest,
    ClassificationTrainingOutput,
    ClassificationTrainingRequest,
    EmbeddingRequest,
    ImportDatasetRequest,
    SaveClassificationModelRequest,
)
from .ports import (
    ClassificationDatasetPort,
    ClassificationModelRepository,
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
