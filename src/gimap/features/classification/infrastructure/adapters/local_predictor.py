"""已加载 classification pipeline 的预测 adapter。"""

from __future__ import annotations

import numpy as np

from ...domain import PredictionResult


class LocalClassifierPredictorAdapter:
    def predict(self, request):
        matrix = request.feature_matrix
        package = request.package
        expected = package.input_shape[1] if package.input_shape else None
        if expected is not None and matrix.X.shape[1] != expected:
            raise ValueError(
                f"Input feature count is {matrix.X.shape[1]}, but the saved model expects {expected}."
            )
        pipeline = package.pipeline
        labels = pipeline.predict(matrix.X)
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
        items = []
        for index, sample in enumerate(matrix.samples):
            confidence = (
                float(np.max(probabilities[index]))
                if probabilities is not None
                else None
            )
            score = None
            if decision_scores is not None:
                values = np.asarray(decision_scores[index])
                score = float(np.max(values)) if values.ndim else float(values)
            items.append(
                PredictionResult(
                    file_path=sample.file_path,
                    file_name=sample.file_name,
                    predicted_label=str(labels[index]),
                    confidence=confidence,
                    decision_score=score,
                    status="ok",
                    data_shape=sample.raw_shape,
                )
            )
        return tuple(items)
