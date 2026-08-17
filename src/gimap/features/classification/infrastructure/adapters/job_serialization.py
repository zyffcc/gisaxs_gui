"""Classification worker 的 JSON-safe serialization。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from ...domain import (
    ExperimentResult,
    MisclassifiedSample,
    ModelEvaluationResult,
    PreprocessingConfig,
    ProjectionConfig,
    ValidationConfig,
)
from .model_repository import LazyJoblibPipeline


ARRAY_MARKER = "gimap.classification.ndarray.v1"


def encode_array(value) -> dict:
    array = np.asarray(value)
    data = array.reshape(-1).tolist()
    if np.issubdtype(array.dtype, np.floating):
        data = [item if np.isfinite(item) else None for item in data]
    return {
        "format": ARRAY_MARKER,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": data,
    }


def decode_array(value: dict) -> np.ndarray:
    if value.get("format") != ARRAY_MARKER:
        raise ValueError("Unsupported classification array payload")
    return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])


def serialize_experiment(experiment: ExperimentResult, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    serialized_results = []
    for index, result in enumerate(experiment.results):
        pipeline_path = None
        if result.fitted_pipeline is not None:
            import joblib

            pipeline_path = artifact_dir / f"{index:02d}_{result.algorithm_id}.joblib"
            joblib.dump(result.fitted_pipeline, pipeline_path)
        serialized_results.append(
            {
                "algorithm_id": result.algorithm_id,
                "display_name": result.display_name,
                "status": result.status,
                "metrics_mean": result.metrics_mean,
                "metrics_std": result.metrics_std,
                "fold_metrics": result.fold_metrics,
                "confusion_matrix": encode_array(result.confusion_matrix) if result.confusion_matrix is not None else None,
                "classification_report": result.classification_report,
                "out_of_fold_predictions": encode_array(result.out_of_fold_predictions) if result.out_of_fold_predictions is not None else None,
                "probabilities": encode_array(result.probabilities) if result.probabilities is not None else None,
                "decision_scores": encode_array(result.decision_scores) if result.decision_scores is not None else None,
                "training_time": result.training_time,
                "prediction_time": result.prediction_time,
                "pipeline_path": str(pipeline_path) if pipeline_path else None,
                "error_message": result.error_message,
                "misclassified_samples": [asdict(item) for item in result.misclassified_samples],
                "labels": result.labels,
            }
        )
    return {
        "results": serialized_results,
        "ranking_metric": experiment.ranking_metric,
        "labels": experiment.labels,
        "sample_ids": experiment.sample_ids,
        "y_true": encode_array(experiment.y_true),
        "warnings": experiment.warnings,
        "input_shape": list(experiment.input_shape) if experiment.input_shape else None,
        "preprocessing_config": asdict(experiment.preprocessing_config),
        "projection_config": asdict(experiment.projection_config),
        "validation_config": asdict(experiment.validation_config),
        "created_at": experiment.created_at,
    }


def deserialize_experiment(value: dict) -> ExperimentResult:
    results = []
    for item in value["results"]:
        results.append(
            ModelEvaluationResult(
                algorithm_id=item["algorithm_id"],
                display_name=item["display_name"],
                status=item["status"],
                metrics_mean=item["metrics_mean"],
                metrics_std=item["metrics_std"],
                fold_metrics=item["fold_metrics"],
                confusion_matrix=decode_array(item["confusion_matrix"]) if item["confusion_matrix"] else None,
                classification_report=item["classification_report"],
                out_of_fold_predictions=decode_array(item["out_of_fold_predictions"]) if item["out_of_fold_predictions"] else None,
                probabilities=decode_array(item["probabilities"]) if item["probabilities"] else None,
                decision_scores=decode_array(item["decision_scores"]) if item["decision_scores"] else None,
                training_time=item["training_time"],
                prediction_time=item["prediction_time"],
                fitted_pipeline=LazyJoblibPipeline(Path(item["pipeline_path"])) if item["pipeline_path"] else None,
                error_message=item["error_message"],
                misclassified_samples=[MisclassifiedSample(**entry) for entry in item["misclassified_samples"]],
                labels=item["labels"],
            )
        )
    input_shape = value.get("input_shape")
    return ExperimentResult(
        results=results,
        ranking_metric=value["ranking_metric"],
        labels=value["labels"],
        sample_ids=value["sample_ids"],
        y_true=decode_array(value["y_true"]),
        warnings=value["warnings"],
        input_shape=tuple(input_shape) if input_shape else None,
        preprocessing_config=PreprocessingConfig(**value["preprocessing_config"]),
        projection_config=ProjectionConfig(**value["projection_config"]),
        validation_config=ValidationConfig(**value["validation_config"]),
        created_at=value["created_at"],
    )
