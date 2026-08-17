"""Classification feature composition root。"""

from __future__ import annotations

from src.gimap.app import AppContext

from .application import (
    BuildClassificationFeatures,
    BuildClassificationModelPackage,
    ComputeClassificationEmbedding,
    ImportClassificationDataset,
    LoadClassificationModel,
    PredictClassification,
    SaveClassificationModel,
    TrainClassifiers,
)
from .infrastructure import (
    JobRunnerClassifierTrainer,
    JobRunnerEmbeddingAdapter,
    JoblibClassificationModelRepository,
    ImportlibRuntimeVersionAdapter,
    LegacyClassificationDatasetAdapter,
    LocalClassifierPredictorAdapter,
)
from .presentation import ClassificationViewModel


def create_classification_view_model(context: AppContext) -> ClassificationViewModel:
    if context.jobs is None:
        raise ValueError("ClassificationViewModel requires AppContext.jobs")
    datasets = LegacyClassificationDatasetAdapter()
    models = JoblibClassificationModelRepository()
    return ClassificationViewModel(
        context=context,
        import_dataset=ImportClassificationDataset(datasets),
        build_features=BuildClassificationFeatures(datasets),
        train_classifiers=TrainClassifiers(JobRunnerClassifierTrainer(context.jobs)),
        compute_embedding=ComputeClassificationEmbedding(
            JobRunnerEmbeddingAdapter(context.jobs)
        ),
        predict_classification=PredictClassification(
            LocalClassifierPredictorAdapter()
        ),
        save_model=SaveClassificationModel(models),
        load_model=LoadClassificationModel(models),
        build_model_package=BuildClassificationModelPackage(
            ImportlibRuntimeVersionAdapter()
        ),
    )
