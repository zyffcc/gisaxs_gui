"""Classification feature composition root。"""

from __future__ import annotations

from src.gimap.app import AppContext

from .application import (
    BuildClassificationFeatures,
    BuildClassificationModelPackage,
    ComputeClassificationEmbedding,
    EstimateClassificationFeatureMemory,
    ExportClassificationCsv,
    ImportClassificationDataset,
    LoadClassificationModel,
    LoadClassificationSession,
    ListClassificationAlgorithms,
    PredictClassification,
    SaveClassificationModel,
    SaveClassificationSession,
    TrainClassifiers,
    SummarizeClassificationDataset,
    ValidateClassificationDataset,
)
from .infrastructure import (
    JobRunnerClassifierTrainer,
    JobRunnerEmbeddingAdapter,
    JoblibClassificationModelRepository,
    ImportlibRuntimeVersionAdapter,
    LegacyClassificationDatasetAdapter,
    LocalClassificationArtifactRepository,
    LocalClassifierPredictorAdapter,
    ClassificationDataService,
    ClassificationTrainingService,
)
from .presentation import ClassificationViewModel


def create_classification_view_model(context: AppContext) -> ClassificationViewModel:
    if context.jobs is None:
        raise ValueError("ClassificationViewModel requires AppContext.jobs")
    datasets = LegacyClassificationDatasetAdapter(ClassificationDataService())
    algorithm_catalog = ClassificationTrainingService()
    models = JoblibClassificationModelRepository()
    artifacts = LocalClassificationArtifactRepository()
    return ClassificationViewModel(
        context=context,
        import_dataset=ImportClassificationDataset(datasets),
        build_features=BuildClassificationFeatures(datasets),
        validate_dataset=ValidateClassificationDataset(datasets),
        summarize_dataset=SummarizeClassificationDataset(datasets),
        estimate_feature_memory=EstimateClassificationFeatureMemory(),
        list_algorithms=ListClassificationAlgorithms(algorithm_catalog),
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
        save_session=SaveClassificationSession(artifacts),
        load_session=LoadClassificationSession(artifacts),
        export_csv=ExportClassificationCsv(artifacts),
    )
