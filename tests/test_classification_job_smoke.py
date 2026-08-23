import numpy as np

from src.gimap.features.classification.application import (
    ClassificationTrainingRequest,
    ClusteringRequest,
)
from src.gimap.features.classification.domain import (
    AlgorithmConfig,
    ClassificationSample,
    FeatureMatrix,
    PreprocessingConfig,
    ProjectionConfig,
    ValidationConfig,
)
from src.gimap.features.classification.infrastructure import (
    JobRunnerClassifierTrainer,
    JobRunnerClusteringAdapter,
    LazyJoblibPipeline,
)
from src.gimap.integrations.jobs import LocalProcessJobRunner


def test_minimal_classifier_training_runs_in_job_process(tmp_path):
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [1.0, 1.0],
            [0.9, 1.0],
            [1.0, 0.9],
            [0.9, 0.9],
        ],
        dtype=np.float64,
    )
    y = np.array(["A"] * 4 + ["B"] * 4, dtype=object)
    samples = [
        ClassificationSample(
            str(index),
            str(tmp_path / f"{index}.npy"),
            f"{index}.npy",
            str(label),
            "1D",
            raw_shape=(2,),
            load_status="loaded",
            qc_status="ready",
        )
        for index, label in enumerate(y)
    ]
    request = ClassificationTrainingRequest(
        feature_matrix=FeatureMatrix(X, y, samples, input_shape=X.shape),
        preprocessing=PreprocessingConfig(),
        algorithms=(AlgorithmConfig("gaussian_nb", "Gaussian NB", True),),
        validation=ValidationConfig(folds=2, random_state=42),
        projection=ProjectionConfig(),
        ranking_metric="macro_f1",
        timeout_seconds=60,
    )
    runner = LocalProcessJobRunner()
    try:
        experiment = JobRunnerClassifierTrainer(
            runner, tmp_path / "artifacts"
        ).train(request)
    finally:
        runner.shutdown()

    assert len(experiment.results) == 1
    assert experiment.results[0].status == "ok"
    assert experiment.results[0].metrics_mean["accuracy"] == 1.0
    assert isinstance(experiment.results[0].fitted_pipeline, LazyJoblibPipeline)
    assert experiment.results[0].fitted_pipeline.path.is_file()


def test_unlabeled_group_suggestions_run_in_job_process():
    X = np.array(
        [[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]], dtype=np.float64
    )
    runner = LocalProcessJobRunner()
    try:
        result = JobRunnerClusteringAdapter(runner).cluster(
            ClusteringRequest(X, "K-Means", n_clusters=2, timeout_seconds=60)
        )
    finally:
        runner.shutdown()

    assert result.method == "K-Means"
    assert len(set(result.labels.tolist())) == 2


def test_density_group_suggestions_run_in_job_process():
    X = np.array(
        [
            [0.0, 0.0],
            [0.05, 0.0],
            [0.0, 0.05],
            [3.0, 3.0],
            [3.05, 3.0],
            [3.0, 3.05],
        ],
        dtype=np.float64,
    )
    runner = LocalProcessJobRunner()
    try:
        result = JobRunnerClusteringAdapter(runner).cluster(
            ClusteringRequest(X, "HDBSCAN", min_cluster_size=2, timeout_seconds=60)
        )
    finally:
        runner.shutdown()

    assert result.method == "HDBSCAN"
    assert result.labels.shape == (6,)
