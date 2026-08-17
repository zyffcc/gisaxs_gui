from pathlib import Path

import numpy as np

from src.gimap.app.jobs import JobProgress, JobResult
from src.gimap.features.classification.application import (
    ClassificationTrainingRequest,
    EmbeddingRequest,
)
from src.gimap.features.classification.domain import (
    AlgorithmConfig,
    ClassificationSample,
    ExperimentResult,
    FeatureMatrix,
    PreprocessingConfig,
    ProjectionConfig,
    ValidationConfig,
)
from src.gimap.features.classification.infrastructure import (
    JobRunnerClassifierTrainer,
    JobRunnerEmbeddingAdapter,
)
from src.gimap.features.classification.infrastructure.adapters.job_serialization import (
    decode_array,
    encode_array,
    serialize_experiment,
)


class _Runner:
    def __init__(self, value):
        self.value = value
        self.request = None
        self.cancelled = None

    def run(self, request, on_progress=None):
        self.request = request
        if on_progress:
            on_progress(JobProgress(request.job_id, 1, 1, "fake complete"))
        return JobResult(request.job_id, "succeeded", self.value)

    def cancel(self, job_id):
        self.cancelled = job_id
        return True


def _matrix():
    samples = [
        ClassificationSample("one", "/tmp/one.npy", "one.npy", "A", "1D"),
        ClassificationSample("two", "/tmp/two.npy", "two.npy", "B", "1D"),
    ]
    return FeatureMatrix(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array(["A", "B"], dtype=object),
        samples,
        input_shape=(2, 2),
    )


def test_classification_array_serialization_preserves_nan_without_invalid_json():
    encoded = encode_array(np.array([[1.0, np.nan]]))

    assert encoded["data"] == [1.0, None]
    restored = decode_array(encoded)
    assert restored[0, 0] == 1.0
    assert np.isnan(restored[0, 1])


def test_training_adapter_builds_serializable_job_request_with_fake_runner(tmp_path):
    experiment = ExperimentResult(
        [], "macro_f1", ["A", "B"], ["one", "two"], np.array(["A", "B"], dtype=object)
    )
    value = serialize_experiment(experiment, tmp_path / "worker-output")
    runner = _Runner(value)
    trainer = JobRunnerClassifierTrainer(runner, tmp_path / "artifacts")
    progress = []

    restored = trainer.train(
        ClassificationTrainingRequest(
            feature_matrix=_matrix(),
            preprocessing=PreprocessingConfig(),
            algorithms=(AlgorithmConfig("fake", "Fake", True),),
            validation=ValidationConfig(),
            projection=ProjectionConfig(),
            ranking_metric="macro_f1",
        ),
        on_progress=lambda done, total, message: progress.append((done, total, message)),
    )

    assert runner.request.handler.endswith(":train_classifiers_job")
    assert runner.request.payload["algorithms"][0]["algorithm_id"] == "fake"
    assert restored.labels == ["A", "B"]
    assert progress == [(1, 1, "fake complete")]


def test_embedding_adapter_uses_job_runner_and_no_local_ml_import():
    runner = _Runner(
        {"method": "PCA 2D", "values": encode_array(np.array([[0.0, 1.0]]))}
    )
    adapter = JobRunnerEmbeddingAdapter(runner)

    result = adapter.embed(EmbeddingRequest(np.array([[1.0, 2.0]]), "PCA 2D"))

    assert runner.request.handler.endswith(":classification_embedding_job")
    np.testing.assert_array_equal(result.values, [[0.0, 1.0]])
