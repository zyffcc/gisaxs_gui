"""Legacy Qt workers delegating Classification commands to the ViewModel。"""

from __future__ import annotations

import traceback

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from ..application import (
    AlgorithmConfig,
    DatasetSource,
    PreprocessingConfig,
    ProjectionConfig,
    SavedModelPackage,
    ValidationConfig,
)
from .view_model import ClassificationViewModel


class WorkerSignals(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class CancellableWorker(QRunnable):
    def __init__(self) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class ImportWorker(CancellableWorker):
    def __init__(
        self,
        sources: list[DatasetSource],
        view_model: ClassificationViewModel,
    ) -> None:
        super().__init__()
        self.sources = sources
        self.view_model = view_model

    def run(self) -> None:
        try:
            def progress(done: int, total: int, name: str) -> None:
                percent = int(done * 100 / max(1, total))
                self.signals.progress.emit(
                    percent, f"Importing {name} ({done}/{total})"
                )

            imported = self.view_model.import_sources(
                self.sources,
                on_progress=progress,
                is_cancelled=self.is_cancelled,
            )
            if imported is None:
                raise RuntimeError(
                    self.view_model.state.error_message or "Dataset import failed"
                )
            self.signals.finished.emit(
                {"samples": list(imported.samples), "summary": imported.summary}
            )
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class TrainingWorker(CancellableWorker):
    def __init__(
        self,
        samples,
        preprocessing: PreprocessingConfig,
        algorithms: list[AlgorithmConfig],
        validation: ValidationConfig,
        projection: ProjectionConfig,
        ranking_metric: str,
        view_model: ClassificationViewModel,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.preprocessing = preprocessing
        self.algorithms = algorithms
        self.validation = validation
        self.projection = projection
        self.ranking_metric = ranking_metric
        self.view_model = view_model

    def cancel(self) -> None:
        super().cancel()
        self.view_model.cancel()

    def run(self) -> None:
        try:
            def progress(done: int, total: int, name: str) -> None:
                percent = int(done * 100 / max(1, total))
                self.signals.progress.emit(percent, f"Training {name}")

            output = self.view_model.train(
                self.samples,
                self.preprocessing,
                self.algorithms,
                self.validation,
                self.projection,
                self.ranking_metric,
                on_progress=progress,
            )
            if output is None:
                raise RuntimeError(
                    self.view_model.state.error_message
                    or "Classification training failed"
                )
            self.signals.finished.emit(
                {
                    "result": output.experiment,
                    "feature_matrix": output.feature_matrix,
                }
            )
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class PredictionWorker(CancellableWorker):
    def __init__(
        self,
        paths,
        package: SavedModelPackage,
        view_model: ClassificationViewModel,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.package = package
        self.view_model = view_model

    def run(self) -> None:
        try:
            results = self.view_model.predict_paths(self.paths, self.package)
            if results is None:
                raise RuntimeError(
                    self.view_model.state.error_message
                    or "Classification prediction failed"
                )
            self.signals.finished.emit(results)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class EmbeddingWorker(CancellableWorker):
    def __init__(
        self,
        samples,
        preprocessing: PreprocessingConfig,
        method: str,
        view_model: ClassificationViewModel,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.preprocessing = preprocessing
        self.method = method
        self.view_model = view_model

    def cancel(self) -> None:
        super().cancel()
        self.view_model.cancel()

    def run(self) -> None:
        try:
            payload = self.view_model.compute_embedding(
                self.samples,
                self.preprocessing,
                self.method,
            )
            if payload is None:
                raise RuntimeError(
                    self.view_model.state.error_message or "Embedding failed"
                )
            result, matrix = payload
            self.signals.finished.emit(
                {
                    "embedding": result.values,
                    "matrix": matrix,
                    "method": result.method,
                }
            )
        except Exception:
            self.signals.error.emit(traceback.format_exc())
