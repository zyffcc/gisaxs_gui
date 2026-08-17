"""Classification JobRunner handlers；仅在 worker process 导入 ML runtime。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .adapters.job_serialization import decode_array, encode_array, serialize_experiment


def train_classifiers_job(payload, report, is_cancelled):
    from controllers.classification_training_service import ClassificationTrainingService
    from ..domain import (
        AlgorithmConfig,
        ClassificationSample,
        ProjectionConfig,
        ValidationConfig,
    )

    X = decode_array(payload["X"])
    y = decode_array(payload["y"])
    samples = [ClassificationSample(**item) for item in payload["samples"]]
    algorithms = [AlgorithmConfig(**item) for item in payload["algorithms"]]
    validation = ValidationConfig(**payload["validation"])
    projection = ProjectionConfig(**payload["projection"])

    def progress(done, total, name):
        report(done, total, f"Training {name}")

    experiment = ClassificationTrainingService().compare_algorithms(
        X,
        y,
        samples,
        algorithms,
        validation,
        projection,
        payload["ranking_metric"],
        progress=progress,
        is_cancelled=is_cancelled,
    )
    return serialize_experiment(experiment, Path(payload["artifact_dir"]))


def classification_embedding_job(payload, report, is_cancelled):
    X = decode_array(payload["X"])
    method = payload["method"]
    report(0, 1, f"Computing {method}")
    if is_cancelled():
        raise RuntimeError("Embedding was cancelled")
    if method == "PCA 2D":
        from sklearn.decomposition import PCA

        model = PCA(n_components=min(2, X.shape[0], X.shape[1]), random_state=42)
        embedding = model.fit_transform(X)
    elif method == "UMAP 2D":
        try:
            from umap import UMAP
        except Exception as exc:
            raise RuntimeError(
                "UMAP is not available. Install umap-learn or choose PCA/t-SNE."
            ) from exc
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
    report(1, 1, f"{method} complete")
    return {"method": method, "values": encode_array(embedding)}
