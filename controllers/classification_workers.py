"""Background workers for Classification import, training, prediction, and embedding."""

from __future__ import annotations

import traceback
from typing import Iterable

import numpy as np
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from controllers.classification_data_service import ClassificationDataService
from controllers.classification_models import (
    AlgorithmConfig,
    DatasetSource,
    ExperimentResult,
    PredictionResult,
    PreprocessingConfig,
    ProjectionConfig,
    SavedModelPackage,
    ValidationConfig,
)
from controllers.classification_training_service import ClassificationTrainingService


class WorkerSignals(QObject):
    """Common signal set used by Classification background tasks."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class CancellableWorker(QRunnable):
    """Base QRunnable with a cancellation flag."""

    def __init__(self) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class ImportWorker(CancellableWorker):
    """Scan and load files for all dataset sources."""

    def __init__(self, sources: list[DatasetSource], data_service: ClassificationDataService) -> None:
        super().__init__()
        self.sources = sources
        self.data_service = data_service

    def run(self) -> None:
        try:
            samples = self.data_service.scan_sources(self.sources)
            if self.is_cancelled():
                return

            def _progress(done: int, total: int, name: str) -> None:
                percent = int(done * 100 / max(1, total))
                self.signals.progress.emit(percent, f"Importing {name} ({done}/{total})")

            self.data_service.load_samples(samples, progress=_progress, is_cancelled=self.is_cancelled)
            summary = self.data_service.validate_dataset(samples)
            self.signals.finished.emit({"samples": samples, "summary": summary})
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class TrainingWorker(CancellableWorker):
    """Build features and run multi-model comparison."""

    def __init__(
        self,
        samples,
        preprocessing: PreprocessingConfig,
        algorithms: list[AlgorithmConfig],
        validation: ValidationConfig,
        projection: ProjectionConfig,
        ranking_metric: str,
        data_service: ClassificationDataService,
        training_service: ClassificationTrainingService,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.preprocessing = preprocessing
        self.algorithms = algorithms
        self.validation = validation
        self.projection = projection
        self.ranking_metric = ranking_metric
        self.data_service = data_service
        self.training_service = training_service

    def run(self) -> None:
        try:
            self.signals.progress.emit(0, "Building feature matrix")
            matrix = self.data_service.build_feature_matrix(self.samples, self.preprocessing, require_labels=True)
            if self.is_cancelled():
                return

            def _progress(done: int, total: int, name: str) -> None:
                percent = int(done * 100 / max(1, total))
                self.signals.progress.emit(percent, f"Training {name}")

            result: ExperimentResult = self.training_service.compare_algorithms(
                matrix.X,
                matrix.y,
                matrix.samples,
                self.algorithms,
                self.validation,
                self.projection,
                self.ranking_metric,
                progress=_progress,
                is_cancelled=self.is_cancelled,
            )
            result.preprocessing_config = self.preprocessing
            result.projection_config = self.projection
            result.input_shape = matrix.input_shape
            result.warnings.extend(matrix.warnings)
            self.signals.finished.emit({"result": result, "feature_matrix": matrix})
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class PredictionWorker(CancellableWorker):
    """Load unknown files, reuse saved preprocessing, and predict labels."""

    def __init__(
        self,
        paths: Iterable[str],
        package: SavedModelPackage,
        data_service: ClassificationDataService,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.package = package
        self.data_service = data_service

    def run(self) -> None:
        try:
            samples = self.data_service.samples_from_paths(self.paths, label="Unknown")
            if not samples:
                self.signals.finished.emit([])
                return

            def _progress(done: int, total: int, name: str) -> None:
                percent = int(done * 100 / max(1, total))
                self.signals.progress.emit(percent, f"Reading {name} ({done}/{total})")

            self.data_service.load_samples(samples, progress=_progress, is_cancelled=self.is_cancelled)
            if self.is_cancelled():
                return
            matrix = self.data_service.build_feature_matrix(
                samples,
                self.package.preprocessing_config,
                require_labels=False,
            )
            expected = self.package.input_shape[1] if self.package.input_shape else None
            if expected is not None and matrix.X.shape[1] != expected:
                raise ValueError(
                    f"Input feature count is {matrix.X.shape[1]}, but the saved model expects {expected}."
                )
            pipeline = self.package.pipeline
            y_pred = pipeline.predict(matrix.X)
            probabilities = None
            decision_scores = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    probabilities = pipeline.predict_proba(matrix.X)
                except Exception:
                    probabilities = None
            if probabilities is None and hasattr(pipeline, "decision_function"):
                try:
                    decision_scores = pipeline.decision_function(matrix.X)
                except Exception:
                    decision_scores = None

            results: list[PredictionResult] = []
            for index, sample in enumerate(matrix.samples):
                confidence = None
                if probabilities is not None:
                    confidence = float(np.max(probabilities[index]))
                decision_score = None
                if decision_scores is not None:
                    scores = np.asarray(decision_scores[index])
                    decision_score = float(np.max(scores)) if scores.ndim else float(scores)
                results.append(
                    PredictionResult(
                        file_path=sample.file_path,
                        file_name=sample.file_name,
                        predicted_label=str(y_pred[index]),
                        confidence=confidence,
                        decision_score=decision_score,
                        status="ok",
                        data_shape=sample.raw_shape,
                    )
                )
            self.signals.finished.emit(results)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class EmbeddingWorker(CancellableWorker):
    """Compute visualization-only 2D embeddings in the background."""

    def __init__(
        self,
        samples,
        preprocessing: PreprocessingConfig,
        method: str,
        data_service: ClassificationDataService,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.preprocessing = preprocessing
        self.method = method
        self.data_service = data_service

    def run(self) -> None:
        try:
            matrix = self.data_service.build_feature_matrix(self.samples, self.preprocessing, require_labels=True)
            X = matrix.X
            method = self.method
            self.signals.progress.emit(25, f"Computing {method} embedding")
            if method == "PCA 2D":
                from sklearn.decomposition import PCA

                model = PCA(n_components=min(2, X.shape[0], X.shape[1]), random_state=42)
                embedding = model.fit_transform(X)
            elif method == "UMAP 2D":
                try:
                    from umap import UMAP
                except Exception as exc:
                    raise RuntimeError("UMAP is not available. Install umap-learn or choose PCA/t-SNE.") from exc
                embedding = UMAP(
                    n_components=2,
                    n_neighbors=max(2, min(15, max(2, X.shape[0] - 1))),
                    random_state=42,
                ).fit_transform(X)
            else:
                from sklearn.manifold import TSNE

                perplexity = max(1.0, min(30.0, X.shape[0] - 1.0))
                embedding = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=42,
                    init="pca",
                    learning_rate="auto",
                ).fit_transform(X)
            if embedding.shape[1] == 1:
                embedding = np.column_stack([embedding[:, 0], np.zeros(embedding.shape[0])])
            self.signals.finished.emit({"embedding": embedding, "matrix": matrix, "method": method})
        except Exception:
            self.signals.error.emit(traceback.format_exc())
