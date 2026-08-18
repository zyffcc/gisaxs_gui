"""Classification ViewModel；不依赖 Qt 或 ML runtime。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.gimap.app import AppContext

from ..application import (
    BuildFeatureMatrixRequest,
    ClassificationCsvRequest,
    ClassificationSessionRequest,
    BuildClassificationModelPackageRequest,
    ClassificationPredictionRequest,
    ClassificationTrainingRequest,
    EmbeddingRequest,
    ImportDatasetRequest,
    SaveClassificationModelRequest,
)
from ..domain import DatasetSource
from .state import ClassificationState


class ClassificationViewModel:
    def __init__(
        self,
        *,
        context: AppContext,
        import_dataset,
        build_features,
        validate_dataset,
        summarize_dataset,
        estimate_feature_memory,
        list_algorithms,
        train_classifiers,
        compute_embedding,
        predict_classification,
        save_model,
        load_model,
        build_model_package,
        save_session,
        load_session,
        export_csv,
    ):
        self.context = context
        self._import_dataset = import_dataset
        self._build_features = build_features
        self._validate_dataset = validate_dataset
        self._summarize_dataset = summarize_dataset
        self._estimate_feature_memory = estimate_feature_memory
        self._list_algorithms = list_algorithms
        self._train_classifiers = train_classifiers
        self._compute_embedding = compute_embedding
        self._predict_classification = predict_classification
        self._save_model = save_model
        self._load_model = load_model
        self._build_model_package = build_model_package
        self._save_session = save_session
        self._load_session = load_session
        self._export_csv = export_csv
        self.state = ClassificationState()

    def load_settings(self) -> dict[str, object]:
        return dict(self.context.settings.get_section("classification"))

    def save_settings(self, values: dict[str, object]) -> None:
        self.context.settings.update_section("classification", dict(values))
        self.context.settings.save()

    def import_sources(self, sources, *, on_progress=None, is_cancelled=None):
        self.state = replace(self.state, dataset_status="loading", error_message=None)
        try:
            imported = self._import_dataset.execute(
                ImportDatasetRequest(tuple(sources)),
                on_progress=on_progress,
                is_cancelled=is_cancelled,
            )
        except Exception as exc:
            self.state = replace(
                self.state,
                dataset_status="error",
                error_message=str(exc),
                status_message=str(exc),
            )
            return None
        self.state = replace(
            self.state,
            dataset_status="ready",
            samples=imported.samples,
            summary=imported.summary,
            error_message=None,
            status_message=f"Imported {len(imported.samples)} samples",
        )
        return imported

    def build_features(self, samples, preprocessing, *, require_labels=True):
        try:
            matrix = self._build_features.execute(
                BuildFeatureMatrixRequest(
                    tuple(samples), preprocessing, require_labels=require_labels
                )
            )
        except Exception as exc:
            self.state = replace(self.state, error_message=str(exc), status_message=str(exc))
            return None
        self.state = replace(self.state, feature_matrix=matrix, error_message=None)
        return matrix

    def validate_dataset(self, samples):
        return self._validate_dataset.execute(tuple(samples))

    def summarize_dataset(self, samples):
        return self._summarize_dataset.execute(tuple(samples))

    def estimate_feature_memory(self, matrix) -> str:
        return self._estimate_feature_memory.execute(matrix)

    def default_algorithms(self):
        return self._list_algorithms.execute()

    def train(
        self,
        samples,
        preprocessing,
        algorithms,
        validation,
        projection,
        ranking_metric,
        *,
        on_progress=None,
        timeout_seconds=None,
    ):
        matrix = self.build_features(samples, preprocessing, require_labels=True)
        if matrix is None:
            return None
        self.state = replace(self.state, training_status="running", progress=0.0)

        def progress(done, total, message):
            fraction = done / total if total else 0.0
            self.state = replace(
                self.state, progress=fraction, status_message=message
            )
            if on_progress:
                on_progress(done, total, message)

        try:
            output = self._train_classifiers.execute(
                ClassificationTrainingRequest(
                    feature_matrix=matrix,
                    preprocessing=preprocessing,
                    algorithms=tuple(algorithms),
                    validation=validation,
                    projection=projection,
                    ranking_metric=ranking_metric,
                    timeout_seconds=timeout_seconds,
                ),
                on_progress=progress,
            )
        except Exception as exc:
            self.state = replace(
                self.state, training_status="error", error_message=str(exc)
            )
            return None
        self.state = replace(
            self.state,
            training_status="ready",
            experiment=output.experiment,
            feature_matrix=output.feature_matrix,
            progress=1.0,
            error_message=None,
        )
        return output

    def compute_embedding(self, samples, preprocessing, method, *, timeout_seconds=None):
        matrix = self.build_features(samples, preprocessing, require_labels=True)
        if matrix is None:
            return None
        self.state = replace(self.state, embedding_status="running")
        try:
            result = self._compute_embedding.execute(
                EmbeddingRequest(matrix.X, method, timeout_seconds)
            )
        except Exception as exc:
            self.state = replace(
                self.state, embedding_status="error", error_message=str(exc)
            )
            return None
        self.state = replace(
            self.state, embedding_status="ready", embedding=result, error_message=None
        )
        return result, matrix

    def predict_paths(self, paths, package):
        imported = self.import_sources(
            (DatasetSource("Unknown", paths=[str(path) for path in paths]),)
        )
        if imported is None:
            return None
        matrix = self.build_features(
            imported.samples, package.preprocessing_config, require_labels=False
        )
        if matrix is None:
            return None
        self.state = replace(self.state, prediction_status="running")
        try:
            output = self._predict_classification.execute(
                ClassificationPredictionRequest(matrix, package)
            )
        except Exception as exc:
            self.state = replace(
                self.state, prediction_status="error", error_message=str(exc)
            )
            return None
        self.state = replace(
            self.state,
            prediction_status="ready",
            predictions=output.items,
            error_message=None,
        )
        return output.items

    def save_model(self, path: Path, package) -> Path | None:
        try:
            saved = self._save_model.execute(
                SaveClassificationModelRequest(Path(path), package)
            )
        except Exception as exc:
            self.state = replace(self.state, error_message=str(exc))
            return None
        self.state = replace(self.state, active_package=package, error_message=None)
        return saved

    def load_model(self, path: Path):
        try:
            package = self._load_model.execute(Path(path))
        except Exception as exc:
            self.state = replace(self.state, error_message=str(exc))
            return None
        self.state = replace(self.state, active_package=package, error_message=None)
        return package

    def save_session(self, path: Path, values: dict) -> Path:
        return self._save_session.execute(
            ClassificationSessionRequest(Path(path), dict(values))
        )

    def load_session(self, path: Path) -> dict:
        return self._load_session.execute(Path(path))

    def export_csv(self, path: Path, columns, rows) -> Path:
        return self._export_csv.execute(
            ClassificationCsvRequest(
                Path(path),
                tuple(str(column) for column in columns),
                tuple(tuple(row) for row in rows),
            )
        )

    def build_model_package(
        self,
        result,
        *,
        class_names,
        data_type,
        input_shape,
        preprocessing,
        projection,
        algorithm_parameters,
        validation,
    ):
        return self._build_model_package.execute(
            BuildClassificationModelPackageRequest(
                pipeline=result.fitted_pipeline,
                algorithm_id=result.algorithm_id,
                display_name=result.display_name,
                class_names=tuple(class_names),
                data_type=data_type,
                input_shape=input_shape,
                preprocessing=preprocessing,
                projection=projection,
                algorithm_parameters=dict(algorithm_parameters),
                validation=validation,
                evaluation_metrics=dict(result.metrics_mean),
            )
        )

    def cancel(self) -> bool:
        return bool(
            self._train_classifiers.cancel() or self._compute_embedding.cancel()
        )
