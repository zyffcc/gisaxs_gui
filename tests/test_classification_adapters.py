from pathlib import Path

import numpy as np

from src.gimap.app.jobs import JobProgress, JobResult
from src.gimap.features.classification.application import (
    ClassificationTrainingRequest,
    ClusteringRequest,
    EmbeddingRequest,
    GroupClassificationSamples,
)
from src.gimap.features.classification.domain import (
    AlgorithmConfig,
    ClassificationSample,
    DatasetSource,
    ExperimentResult,
    FeatureMatrix,
    PreprocessingConfig,
    ProjectionConfig,
    ValidationConfig,
)
from src.gimap.features.classification.infrastructure import (
    JobRunnerClassifierTrainer,
    JobRunnerClusteringAdapter,
    JobRunnerEmbeddingAdapter,
    LegacyClassificationDatasetAdapter,
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


def test_clustering_adapter_uses_job_runner_and_returns_integer_suggestions():
    runner = _Runner(
        {"method": "HDBSCAN", "labels": encode_array(np.array([0, 0, -1]))}
    )
    adapter = JobRunnerClusteringAdapter(runner)

    result = adapter.cluster(
        ClusteringRequest(np.ones((3, 2)), "HDBSCAN", min_cluster_size=2)
    )

    assert runner.request.handler.endswith(":classification_clustering_job")
    assert result.labels.tolist() == [0, 0, -1]


def test_unlabeled_mixed_files_import_then_build_separate_1d_and_2d_groups(tmp_path):
    paths = []
    for index in range(2):
        one_d = tmp_path / f"curve-{index}.npy"
        two_d = tmp_path / f"image-{index}.npy"
        np.save(one_d, np.linspace(0.0, 1.0, 12) + index)
        np.save(two_d, np.arange(36, dtype=float).reshape(6, 6) + index)
        paths.extend((str(one_d), str(two_d)))

    adapter = LegacyClassificationDatasetAdapter()
    imported = adapter.import_sources(
        (
            DatasetSource(
                "Dropped data",
                source_type="files",
                paths=paths,
                label_mode="unlabeled",
            ),
        )
    )
    groups = GroupClassificationSamples().execute(imported.samples)
    matrices = {
        group.key: adapter.build_feature_matrix(
            [sample for sample in imported.samples if sample.sample_id in group.sample_ids],
            PreprocessingConfig(),
            require_labels=False,
        )
        for group in groups
    }

    assert imported.summary.classes == 0
    assert imported.summary.unlabeled_samples == 4
    assert imported.summary.data_types == ["1D", "2D"]
    assert [group.key for group in groups] == ["1D", "2D"]
    assert matrices["1D"].X.shape[0] == 2
    assert matrices["2D"].X.shape[0] == 2
    assert matrices["1D"].y is None and matrices["2D"].y is None


def test_provisional_source_name_is_only_a_suggestion(tmp_path):
    path = tmp_path / "curve.npy"
    np.save(path, np.linspace(0.0, 1.0, 12))
    adapter = LegacyClassificationDatasetAdapter()

    imported = adapter.import_sources(
        (
            DatasetSource(
                "Folder guess",
                paths=[str(path)],
                label_mode="provisional",
            ),
        )
    )
    sample = imported.samples[0]

    assert sample.label == ""
    assert sample.label_status == "provisional"
    assert sample.suggested_label == "Folder guess"
    assert imported.summary.classes == 0
