from pathlib import Path

import numpy as np

from src.gimap.features.classification.application import (
    BuildClassificationFeatures,
    BuildClassificationModelPackage,
    BuildClassificationModelPackageRequest,
    BuildFeatureMatrixRequest,
    ClassificationPredictionRequest,
    ClassificationTrainingRequest,
    ComputeClassificationEmbedding,
    EmbeddingRequest,
    EmbeddingResult,
    ImportClassificationDataset,
    ImportDatasetRequest,
    LoadClassificationModel,
    PredictClassification,
    SaveClassificationModel,
    SaveClassificationModelRequest,
    TrainClassifiers,
)
from src.gimap.features.classification.application.models import ImportedDataset
from src.gimap.features.classification.domain import (
    AlgorithmConfig,
    ClassificationSample,
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


def _sample(label="A"):
    return ClassificationSample("id", "/tmp/one.npy", "one.npy", label, "1D")


def _matrix():
    return FeatureMatrix(
        X=np.array([[1.0, 2.0], [2.0, 3.0]]),
        y=np.array(["A", "B"]),
        samples=[_sample("A"), _sample("B")],
        data_type="1D",
        input_shape=(2, 2),
    )


class _Datasets:
    def __init__(self):
        self.imported = None
        self.built = None

    def import_sources(self, sources, *, on_progress=None, is_cancelled=None):
        self.imported = sources
        return ImportedDataset((_sample(),), DatasetSummary(classes=1, total_samples=1))

    def build_feature_matrix(self, samples, preprocessing, *, require_labels):
        self.built = (samples, preprocessing, require_labels)
        return _matrix()


class _Trainer:
    def __init__(self):
        self.request = None
        self.cancelled = False

    def train(self, request, *, on_progress=None):
        self.request = request
        if on_progress:
            on_progress(1, 1, "fake")
        return ExperimentResult([], request.ranking_metric, ["A", "B"], ["a", "b"], request.feature_matrix.y)

    def cancel(self):
        self.cancelled = True
        return True


class _Embedding:
    def embed(self, request):
        return EmbeddingResult(request.values[:, :2], request.method)

    def cancel(self):
        return True


class _Predictor:
    def predict(self, request):
        return (
            PredictionResult("/tmp/one.npy", "one.npy", "A", 0.9, None, "ok"),
        )


class _Models:
    def __init__(self):
        self.saved = None

    def save(self, path, package):
        self.saved = (path, package)

    def load(self, path):
        assert path == self.saved[0]
        return self.saved[1]


class _Versions:
    def version(self, distribution):
        return {"scikit-learn": "fake-sklearn", "numpy": "fake-numpy"}[distribution]


def _package():
    return SavedModelPackage(
        pipeline=object(),
        algorithm_id="fake",
        display_name="Fake",
        class_names=["A", "B"],
        data_type="1D",
        input_shape=(2, 2),
        preprocessing_config=PreprocessingConfig(),
        projection_config=ProjectionConfig(),
        algorithm_parameters={},
        sklearn_version="not-loaded",
        numpy_version=np.__version__,
        software_version="test",
        training_date="now",
        validation_config=ValidationConfig(),
        evaluation_metrics={},
    )


def test_data_use_cases_separate_loading_and_feature_construction():
    datasets = _Datasets()
    source = DatasetSource("A", paths=["/tmp"])

    imported = ImportClassificationDataset(datasets).execute(
        ImportDatasetRequest((source,))
    )
    matrix = BuildClassificationFeatures(datasets).execute(
        BuildFeatureMatrixRequest(imported.samples, PreprocessingConfig())
    )

    assert imported.summary.total_samples == 1
    assert matrix.X.shape == (2, 2)
    assert datasets.built[2] is True


def test_training_use_case_uses_fake_classifier_without_ml_runtime():
    trainer = _Trainer()
    preprocessing = PreprocessingConfig(normalize="zscore")
    request = ClassificationTrainingRequest(
        feature_matrix=_matrix(),
        preprocessing=preprocessing,
        algorithms=(AlgorithmConfig("fake", "Fake", True),),
        validation=ValidationConfig(),
        projection=ProjectionConfig(),
        ranking_metric="macro_f1",
    )

    output = TrainClassifiers(trainer).execute(request)

    assert output.feature_matrix is request.feature_matrix
    assert output.experiment.preprocessing_config is preprocessing
    assert trainer.request is request


def test_embedding_prediction_and_model_repository_ports_need_no_sklearn(tmp_path):
    matrix = _matrix()
    embedding = ComputeClassificationEmbedding(_Embedding()).execute(
        EmbeddingRequest(matrix.X, "PCA 2D")
    )
    prediction = PredictClassification(_Predictor()).execute(
        ClassificationPredictionRequest(matrix, _package())
    )
    repository = _Models()
    package = _package()
    path = tmp_path / "model.joblib"

    SaveClassificationModel(repository).execute(
        SaveClassificationModelRequest(path, package)
    )
    loaded = LoadClassificationModel(repository).execute(path)

    assert embedding.values.shape == (2, 2)
    assert prediction.items[0].predicted_label == "A"
    assert loaded is package


def test_model_package_metadata_uses_version_port_without_importing_sklearn():
    package = BuildClassificationModelPackage(_Versions()).execute(
        BuildClassificationModelPackageRequest(
            pipeline="fake-pipeline",
            algorithm_id="fake",
            display_name="Fake",
            class_names=("A", "B"),
            data_type="1D",
            input_shape=(2, 2),
            preprocessing=PreprocessingConfig(),
            projection=ProjectionConfig(),
            algorithm_parameters={"alpha": 1},
            validation=ValidationConfig(),
            evaluation_metrics={"macro_f1": 1.0},
            training_date="2026-01-02T03:04:05",
        )
    )

    assert package.sklearn_version == "fake-sklearn"
    assert package.numpy_version == "fake-numpy"
    assert package.training_date == "2026-01-02T03:04:05"
