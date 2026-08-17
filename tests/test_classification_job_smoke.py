import numpy as np

from src.gimap.features.classification.application import ClassificationTrainingRequest
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
