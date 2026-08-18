from controllers.classification_workers import TrainingWorker as LegacyTrainingWorker
from src.gimap.features.classification.presentation.workers import TrainingWorker


def test_legacy_classification_worker_path_reexports_feature_owner() -> None:
    assert LegacyTrainingWorker is TrainingWorker
