"""Shared data structures for the Classification workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np


class ClassificationPageState(str, Enum):
    """Lifecycle states for the Classification page."""

    EMPTY = "EMPTY"
    SCANNED = "SCANNED"
    IMPORTING = "IMPORTING"
    READY = "READY"
    TRAINING = "TRAINING"
    RESULTS_AVAILABLE = "RESULTS_AVAILABLE"
    PREDICTING = "PREDICTING"
    ERROR = "ERROR"


@dataclass
class DatasetSource:
    """A labeled source of training samples."""

    label: str
    source_type: str = "folder"
    paths: list[str] = field(default_factory=list)
    file_pattern: str = "*"
    color: str = "#3b82f6"
    recursive: bool = True


@dataclass
class ClassificationSample:
    """One file-backed sample in a Classification dataset."""

    sample_id: str
    file_path: str
    file_name: str
    label: str
    data_type: str
    raw_shape: Optional[tuple[int, ...]] = None
    included: bool = True
    load_status: str = "pending"
    qc_status: str = "pending"
    qc_messages: list[str] = field(default_factory=list)
    raw_data: Optional[np.ndarray] = None
    processed_data: Optional[np.ndarray] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    decision_score: Optional[float] = None


@dataclass
class DataQualityIssue:
    """Dataset or sample-level quality message."""

    severity: str
    message: str
    fix: str = ""
    sample_id: Optional[str] = None


@dataclass
class DatasetSummary:
    """Aggregate dataset state shown in the left panel and quality area."""

    classes: int = 0
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    included_samples: int = 0
    loaded_samples: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    valid_class_counts: dict[str, int] = field(default_factory=dict)
    data_types: list[str] = field(default_factory=list)
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    issues: list[DataQualityIssue] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(issue.severity == "error" for issue in self.issues):
            return "Error"
        if any(issue.severity == "warning" for issue in self.issues):
            return "Warning"
        if self.valid_samples >= 2 and self.classes >= 2:
            return "Ready"
        return "Warning"


@dataclass
class FeatureMatrix:
    """Feature matrix and sample mapping produced from loaded samples."""

    X: np.ndarray
    y: Optional[np.ndarray]
    samples: list[ClassificationSample]
    feature_names: list[str] = field(default_factory=list)
    data_type: str = "unknown"
    input_shape: Optional[tuple[int, int]] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessingConfig:
    """Raw-data preprocessing that is saved with the trained model package."""

    data_type: str = "auto"
    one_d_method: str = "Interpolate to common grid"
    two_d_method: str = "Center crop"
    normalize: str = "max"
    log_transform: bool = False
    smoothing_window: int = 0
    crop_min: Optional[float] = None
    crop_max: Optional[float] = None
    one_d_grid: Optional[list[float]] = None
    resize_shape: Optional[tuple[int, int]] = None
    flatten: bool = True


@dataclass
class ValidationConfig:
    """Validation strategy shared by every selected algorithm."""

    method: str = "Stratified K-fold"
    test_size: float = 0.2
    folds: int = 5
    repeats: int = 1
    shuffle: bool = True
    random_state: int = 42


@dataclass
class ProjectionConfig:
    """Optional feature projection inside the sklearn pipeline."""

    enabled: bool = False
    method: str = "None"
    n_components: int = 2
    explained_variance: float = 0.95
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1


@dataclass
class AlgorithmConfig:
    """A classifier and its independent parameter set."""

    algorithm_id: str
    display_name: str
    enabled: bool
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    requires_scaling: bool = False


@dataclass
class MisclassifiedSample:
    """One out-of-fold prediction mistake."""

    sample_id: str
    file_path: str
    file_name: str
    true_label: str
    predicted_label: str
    confidence: Optional[float]
    decision_score: Optional[float]
    data_shape: Optional[tuple[int, ...]]


@dataclass
class ModelEvaluationResult:
    """Complete evaluation payload for one algorithm."""

    algorithm_id: str
    display_name: str
    status: str
    metrics_mean: dict[str, float] = field(default_factory=dict)
    metrics_std: dict[str, float] = field(default_factory=dict)
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[dict[str, Any]] = None
    out_of_fold_predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    decision_scores: Optional[np.ndarray] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    fitted_pipeline: Any = None
    error_message: Optional[str] = None
    misclassified_samples: list[MisclassifiedSample] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def score(self, metric: str) -> float:
        return float(self.metrics_mean.get(metric, float("-inf")))


@dataclass
class ExperimentResult:
    """All model comparison outputs for a single Run Comparison action."""

    results: list[ModelEvaluationResult]
    ranking_metric: str
    labels: list[str]
    sample_ids: list[str]
    y_true: np.ndarray
    warnings: list[str] = field(default_factory=list)
    input_shape: Optional[tuple[int, int]] = None
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    projection_config: ProjectionConfig = field(default_factory=ProjectionConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def successful_results(self) -> list[ModelEvaluationResult]:
        return [result for result in self.results if result.status == "ok"]

    @property
    def best_result(self) -> Optional[ModelEvaluationResult]:
        successful = self.successful_results
        if not successful:
            return None
        return max(successful, key=lambda result: result.score(self.ranking_metric))


@dataclass
class SavedModelPackage:
    """Persisted model bundle used for future prediction."""

    pipeline: Any
    algorithm_id: str
    display_name: str
    class_names: list[str]
    data_type: str
    input_shape: Optional[tuple[int, int]]
    preprocessing_config: PreprocessingConfig
    projection_config: ProjectionConfig
    algorithm_parameters: dict[str, Any]
    sklearn_version: str
    numpy_version: str
    software_version: str
    training_date: str
    validation_config: ValidationConfig
    evaluation_metrics: dict[str, float]


@dataclass
class PredictionResult:
    """Prediction output for one unknown file."""

    file_path: str
    file_name: str
    predicted_label: Optional[str]
    confidence: Optional[float]
    decision_score: Optional[float]
    status: str
    message: str = ""
    data_shape: Optional[tuple[int, ...]] = None
