"""Long-running model operations shared by the Classification ViewModel."""

from __future__ import annotations

from dataclasses import replace

from ..application import (
    ClassificationPredictionRequest,
    ClassificationTrainingRequest,
    ClusteringRequest,
    DatasetSource,
    EmbeddingRequest,
)


class ClassificationModelOperationsMixin:
    """Coordinate framework-neutral training, reduction, grouping, and prediction."""

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
            self.state = replace(self.state, progress=fraction, status_message=message)
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
        matrix = self.build_features(samples, preprocessing, require_labels=False)
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

    def suggest_clusters(
        self,
        samples,
        preprocessing,
        method,
        *,
        n_clusters=4,
        min_cluster_size=5,
        timeout_seconds=None,
    ):
        matrix = self.build_features(samples, preprocessing, require_labels=False)
        if matrix is None:
            return None
        self.state = replace(self.state, clustering_status="running")
        try:
            result = self._suggest_clusters.execute(
                ClusteringRequest(
                    matrix.X,
                    method,
                    n_clusters,
                    min_cluster_size,
                    timeout_seconds,
                )
            )
        except Exception as exc:
            self.state = replace(
                self.state, clustering_status="error", error_message=str(exc)
            )
            return None
        self.state = replace(
            self.state,
            clustering_status="ready",
            clustering=result,
            error_message=None,
        )
        return result, matrix

    def predict_paths(self, paths, package):
        imported = self.import_sources(
            (
                DatasetSource(
                    "Unknown",
                    paths=[str(path) for path in paths],
                    label_mode="unlabeled",
                ),
            )
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

    def cancel(self) -> bool:
        return bool(
            self._train_classifiers.cancel()
            or self._compute_embedding.cancel()
            or self._suggest_clusters.cancel()
        )


__all__ = ["ClassificationModelOperationsMixin"]
