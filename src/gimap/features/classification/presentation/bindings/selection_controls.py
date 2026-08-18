"""Selection Controls coordination for Classification."""

from __future__ import annotations


from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import (
    QMessageBox,
)


class SelectionControlsMixin:
    """Own selection controls presentation behavior."""

    def _preview_selected_misclassification(self) -> None:
        row = self.page.misclassifiedTable.currentRow()
        if row < 0:
            return
        item = self.page.misclassifiedTable.item(row, 0)
        sample = self._sample_by_id(item.data(Qt.UserRole) if item else None)
        if sample is not None:
            self.current_preview_sample_id = sample.sample_id
            self._render_sample_preview(sample)

    def _update_class_filter(self) -> None:
        page = self.page
        combo = page.classFilterCombo
        current = combo.currentText()
        labels = ["All classes"] + list(self.sources.keys())
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(labels)
        combo.setCurrentText(current if current in labels else "All classes")
        combo.blockSignals(False)

    def _select_recommended_algorithms(self) -> None:
        recommended = {
            "logistic_regression",
            "linear_svm",
            "rbf_svm",
            "knn",
            "random_forest",
            "lda",
        }
        for config in self.algorithm_configs:
            config.enabled = config.algorithm_id in recommended
        self._populate_algorithm_table()
        self._on_algorithm_selection_changed()

    def _set_all_algorithms(self, enabled: bool) -> None:
        for config in self.algorithm_configs:
            config.enabled = enabled
        self._populate_algorithm_table()
        self._on_algorithm_selection_changed()

    def _reset_algorithm_defaults(self) -> None:
        self.algorithm_configs = self.classification_view_model.default_algorithms()
        self._populate_algorithm_table()
        self._on_algorithm_selection_changed()

    def _show_help(self) -> None:
        QMessageBox.information(
            self.main_window,
            "Classification",
            "Workflow: add at least two labeled classes, scan/import data, choose preprocessing and algorithms, then run a shared validation comparison.",
        )

    def _quality_message(self) -> str:
        if not self.summary.issues:
            return "Dataset is not ready."
        return "\n".join(
            f"{issue.severity.title()}: {issue.message} {issue.fix}".strip()
            for issue in self.summary.issues[:8]
        )
