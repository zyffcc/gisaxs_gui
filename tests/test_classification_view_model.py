from pathlib import Path

import numpy as np

from src.gimap.app import AppContext
from src.gimap.features.classification.application import (
    ClassificationPredictionOutput,
    ClassificationTrainingOutput,
    EmbeddingResult,
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
    ValidationConfig,
)
from src.gimap.features.classification.presentation import ClassificationViewModel
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)


class _Call:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def execute(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.result

    def cancel(self):
        return True


def _objects():
    samples = (
        ClassificationSample("one", "/tmp/one.npy", "one.npy", "A", "1D"),
        ClassificationSample("two", "/tmp/two.npy", "two.npy", "B", "1D"),
    )
    matrix = FeatureMatrix(
        np.array([[1.0], [2.0]]),
        np.array(["A", "B"]),
        list(samples),
        input_shape=(2, 1),
    )
    experiment = ExperimentResult(
        [], "macro_f1", ["A", "B"], ["one", "two"], matrix.y
    )
    return samples, matrix, experiment


def _view_model(**overrides):
    samples, matrix, experiment = _objects()
    defaults = {
        "import_dataset": _Call(
            ImportedDataset(samples, DatasetSummary(classes=2, total_samples=2))
        ),
        "build_features": _Call(matrix),
        "validate_dataset": _Call(DatasetSummary(classes=2, total_samples=2)),
        "summarize_dataset": _Call({"A": {"files": 1}}),
        "estimate_feature_memory": _Call("16 B"),
        "list_algorithms": _Call((AlgorithmConfig("fake", "Fake", True),)),
        "train_classifiers": _Call(ClassificationTrainingOutput(experiment, matrix)),
        "compute_embedding": _Call(
            EmbeddingResult(np.array([[0.0, 1.0], [1.0, 0.0]]), "PCA 2D")
        ),
        "predict_classification": _Call(
            ClassificationPredictionOutput(
                (PredictionResult("/tmp/one.npy", "one.npy", "A", 0.8, None, "ok"),)
            )
        ),
        "save_model": _Call(Path("model.joblib")),
        "load_model": _Call(object()),
        "build_model_package": _Call(object()),
        "save_session": _Call(Path("session.json")),
        "load_session": _Call({"ranking_metric": "macro_f1"}),
        "export_csv": _Call(Path("results.csv")),
    }
    defaults.update(overrides)
    context = AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )
    return ClassificationViewModel(context=context, **defaults), samples


def test_view_model_import_and_settings_without_qapplication():
    view_model, _samples = _view_model()
    source = DatasetSource("A", paths=["/tmp"])

    view_model.save_settings({"ranking_metric": "macro_f1"})
    imported = view_model.import_sources((source,))

    assert view_model.load_settings()["ranking_metric"] == "macro_f1"
    assert imported.summary.classes == 2
    assert view_model.state.dataset_status == "ready"


def test_view_model_exposes_dataset_analysis_through_application_commands():
    view_model, samples = _view_model()
    matrix = view_model.build_features(samples, PreprocessingConfig())

    assert view_model.validate_dataset(samples).classes == 2
    assert view_model.summarize_dataset(samples)["A"]["files"] == 1
    assert view_model.estimate_feature_memory(matrix) == "16 B"
    assert view_model.default_algorithms()[0].algorithm_id == "fake"


def test_view_model_training_and_embedding_state_use_fake_ml_ports():
    view_model, samples = _view_model()
    preprocessing = PreprocessingConfig()

    trained = view_model.train(
        samples,
        preprocessing,
        (AlgorithmConfig("fake", "Fake", True),),
        ValidationConfig(),
        ProjectionConfig(),
        "macro_f1",
    )
    embedded, matrix = view_model.compute_embedding(
        samples, preprocessing, "PCA 2D"
    )

    assert trained.experiment.ranking_metric == "macro_f1"
    assert view_model.state.training_status == "ready"
    assert embedded.values.shape == (2, 2)
    assert matrix.X.shape == (2, 1)


def test_view_model_maps_training_error_to_display_state():
    view_model, samples = _view_model(
        train_classifiers=_Call(error=RuntimeError("worker crashed"))
    )

    result = view_model.train(
        samples,
        PreprocessingConfig(),
        (AlgorithmConfig("fake", "Fake", True),),
        ValidationConfig(),
        ProjectionConfig(),
        "macro_f1",
    )

    assert result is None
    assert view_model.state.training_status == "error"
    assert view_model.state.error_message == "worker crashed"
