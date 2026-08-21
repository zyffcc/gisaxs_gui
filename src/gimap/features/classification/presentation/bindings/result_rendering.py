"""Result Rendering coordination for Classification."""

from __future__ import annotations


import numpy as np

from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import (
    QTableWidgetItem,
)

from src.gimap.features.classification.application import (
    ClassificationSample,
    ModelEvaluationResult,
)


class ResultRenderingMixin:
    """Own result rendering presentation behavior."""

    def _render_metric_chart(self) -> None:
        if self.experiment_result is None or self.page is None:
            return
        successful = self.experiment_result.successful_results
        if not successful:
            self.page.metricChartLabel.setText("No successful models.")
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [result.display_name for result in successful]
            metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
            x = np.arange(len(labels))
            width = 0.24
            fig, ax = plt.subplots(figsize=(7, 2.8), dpi=120)
            for offset, metric in enumerate(metrics):
                values = [result.metrics_mean.get(metric, 0.0) for result in successful]
                ax.bar(
                    x + (offset - 1) * width, values, width, label=metric.replace("_", " ").title()
                )
            ax.set_ylim(0, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, axis="y", alpha=0.2)
            fig.tight_layout()
            pixmap = self._figure_to_pixmap(fig)
            self.page.metricChartLabel.setPixmap(pixmap)
            self.page.metricChartLabel.setScaledContents(True)
            plt.close(fig)
        except Exception as exc:
            self.page.metricChartLabel.setText(str(exc))

    def _render_confusion_matrix(self, result: ModelEvaluationResult) -> None:
        table = self.page.confusionMatrixTable
        cm = result.confusion_matrix
        labels = result.labels
        if cm is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        matrix = cm.astype(float)
        mode = self.page.confusionNormalizeCombo.currentText()
        if mode == "Normalize by true class":
            denom = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)
        elif mode == "Normalize by predicted class":
            denom = matrix.sum(axis=0, keepdims=True)
            matrix = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)
        table.setRowCount(len(labels))
        table.setColumnCount(len(labels))
        table.setHorizontalHeaderLabels(labels)
        table.setVerticalHeaderLabels(labels)
        for row in range(len(labels)):
            for col in range(len(labels)):
                value = matrix[row, col]
                text = f"{int(value)}" if mode == "Raw counts" else f"{value:.2f}"
                table.setItem(row, col, QTableWidgetItem(text))

    def _render_per_class_metrics(self, result: ModelEvaluationResult) -> None:
        table = self.page.perClassTable
        report = result.classification_report or {}
        labels = result.labels
        table.setRowCount(len(labels))
        for row, label in enumerate(labels):
            row_data = report.get(label, {})
            values = [
                label,
                self._number_text(row_data.get("precision")),
                self._number_text(row_data.get("recall")),
                self._number_text(row_data.get("f1-score")),
                str(int(row_data.get("support", 0))),
            ]
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(value))

    def _render_misclassified_table(self, result: ModelEvaluationResult) -> None:
        table = self.page.misclassifiedTable
        table.setRowCount(len(result.misclassified_samples))
        for row, item in enumerate(result.misclassified_samples):
            values = [
                item.file_name,
                item.true_label,
                item.predicted_label,
                self._optional_float(
                    item.confidence if item.confidence is not None else item.decision_score
                ),
                self._shape_text(item.data_shape),
                "Preview",
            ]
            for col, value in enumerate(values):
                table_item = QTableWidgetItem(str(value))
                table_item.setData(Qt.UserRole, item.sample_id)
                table.setItem(row, col, table_item)

    def _render_prediction_table(self) -> None:
        table = self.page.predictionTable
        table.setRowCount(len(self.prediction_results))
        for row, result in enumerate(self.prediction_results):
            values = [
                result.file_name,
                result.predicted_label or "-",
                self._optional_float(
                    result.confidence if result.confidence is not None else result.decision_score
                ),
                result.status if result.status != "ok" else "OK",
            ]
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(str(value)))

    def _render_embedding(self, embedding: np.ndarray, samples: list[ClassificationSample]) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
            labels = [sample.label for sample in samples]
            unique = sorted(set(labels))
            for label in unique:
                mask = np.asarray([item == label for item in labels])
                ax.scatter(embedding[mask, 0], embedding[mask, 1], s=26, label=label, alpha=0.85)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.2)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            fig.tight_layout()
            self._set_graphics_pixmap(self.page.embeddingGraphicsView, self._figure_to_pixmap(fig))
            plt.close(fig)
        except Exception as exc:
            self._set_graphics_text(self.page.embeddingGraphicsView, str(exc))

    def _update_quality(self) -> None:
        page = self.page
        if page is None:
            return
        summary = self.summary
        page.summaryClassesLabel.setText(str(summary.classes))
        page.summaryTotalLabel.setText(str(summary.total_samples))
        page.summaryValidLabel.setText(str(summary.valid_samples))
        page.summaryInvalidLabel.setText(str(summary.invalid_samples))
        page.summaryBalanceLabel.setText(
            ", ".join(f"{k}:{v}" for k, v in summary.valid_class_counts.items()) or "-"
        )
        page.qualityListWidget.clear()
        if summary.total_samples == 0:
            page.qualityStatusLabel.setText("Waiting for data")
            page.qualityStatusLabel.setProperty("qualityState", "empty")
            page.qualityListWidget.addItem(
                "Add at least two labeled classes, then scan or drop their files here."
            )
        elif not summary.issues:
            page.qualityStatusLabel.setText(summary.status)
            page.qualityStatusLabel.setProperty("qualityState", "ready")
            page.qualityListWidget.addItem("Ready: dataset checks passed.")
        else:
            page.qualityStatusLabel.setText(summary.status)
            page.qualityStatusLabel.setProperty("qualityState", "attention")
            for issue in summary.issues[:20]:
                fix = f" Fix: {issue.fix}" if issue.fix else ""
                page.qualityListWidget.addItem(f"{issue.severity.title()}: {issue.message}{fix}")
        page.qualityStatusLabel.style().unpolish(page.qualityStatusLabel)
        page.qualityStatusLabel.style().polish(page.qualityStatusLabel)
        data_type = summary.data_types[0] if len(summary.data_types) == 1 else "auto"
        page.dataTypeBadgeLabel.setText(data_type)
