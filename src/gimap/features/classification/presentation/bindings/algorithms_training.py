"""Algorithms Training coordination for Classification."""

from __future__ import annotations


from pathlib import Path

from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
)

from src.gimap.features.classification.application import (
    AlgorithmConfig,
    ClassificationPageState,
    ModelEvaluationResult,
)

from src.gimap.features.classification.presentation.workers import (
    EmbeddingWorker,
    PredictionWorker,
    TrainingWorker,
)


class AlgorithmsTrainingMixin:
    """Own algorithms training presentation behavior."""

    def _populate_algorithm_table(self) -> None:
        page = self.page
        if page is None:
            return
        table = page.algorithmTable
        table.blockSignals(True)
        try:
            table.setRowCount(len(self.algorithm_configs))
            for row, config in enumerate(self.algorithm_configs):
                use_item = QTableWidgetItem("")
                use_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                use_item.setCheckState(Qt.Checked if config.enabled else Qt.Unchecked)
                table.setItem(row, 0, use_item)
                name_item = QTableWidgetItem(config.display_name)
                name_item.setData(Qt.UserRole, config.algorithm_id)
                name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, 1, name_item)
                desc_item = QTableWidgetItem(config.description)
                desc_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, 2, desc_item)
                button = QPushButton("Parameters", table)
                button.clicked.connect(
                    lambda _checked=False, algorithm_id=config.algorithm_id: (
                        self._edit_algorithm_parameters(algorithm_id)
                    )
                )
                table.setCellWidget(row, 3, button)
        finally:
            table.blockSignals(False)
        self._update_run_summary()

    def _collect_algorithm_configs(self) -> list[AlgorithmConfig]:
        page = self.page
        table = page.algorithmTable
        enabled_by_id: dict[str, bool] = {}
        for row in range(table.rowCount()):
            name_item = table.item(row, 1)
            use_item = table.item(row, 0)
            if name_item is None or use_item is None:
                continue
            enabled_by_id[str(name_item.data(Qt.UserRole))] = use_item.checkState() == Qt.Checked
        for config in self.algorithm_configs:
            if config.algorithm_id in enabled_by_id:
                config.enabled = enabled_by_id[config.algorithm_id]
        return [config for config in self.algorithm_configs]

    def _edit_algorithm_parameters(self, algorithm_id: str) -> None:
        config = next(
            (item for item in self.algorithm_configs if item.algorithm_id == algorithm_id), None
        )
        if config is None:
            return
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle(f"{config.display_name} Parameters")
        layout = QFormLayout(dialog)
        editors: dict[str, QLineEdit] = {}
        for key, value in config.parameters.items():
            edit = QLineEdit(str(value), dialog)
            editors[key] = edit
            layout.addRow(key, edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        for key, edit in editors.items():
            config.parameters[key] = self._parse_parameter(edit.text(), config.parameters[key])
        self._mark_results_outdated()
        self._persist_parameters()
        self.log(f"[Algorithms] Updated parameters for {config.display_name}.")

    def _start_training(self) -> None:
        if self.current_worker is not None:
            QMessageBox.information(
                self.main_window, "Classification", "A Classification task is already running."
            )
            return
        self.summary = self.classification_view_model.validate_dataset(self.samples)
        if any(issue.severity == "error" for issue in self.summary.issues):
            QMessageBox.warning(self.main_window, "Classification", self._quality_message())
            self._refresh_everything()
            return
        algorithms = [config for config in self._collect_algorithm_configs() if config.enabled]
        if not algorithms:
            QMessageBox.warning(
                self.main_window, "Classification", "Select at least one algorithm."
            )
            return
        worker = TrainingWorker(
            self.samples,
            self._collect_preprocessing_config(),
            algorithms,
            self._collect_validation_config(),
            self._collect_projection_config(),
            self._ranking_metric(),
            self.classification_view_model,
        )
        self.current_worker = worker
        self._set_state(ClassificationPageState.TRAINING)
        self.page.taskProgressBar.setValue(0)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(self._on_training_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_training_finished(self, payload) -> None:
        self.current_worker = None
        if not isinstance(payload, dict) or "result" not in payload:
            self._on_worker_error("Training returned an invalid payload.")
            return
        self.experiment_result = payload["result"]
        self.feature_matrix = payload.get("feature_matrix")
        self._results_outdated = False
        self.active_result = self.experiment_result.best_result
        self._write_predictions_from_active_result()
        self._update_dataset_table()
        self._update_results_views()
        self._set_state(ClassificationPageState.RESULTS_AVAILABLE)
        best = self.active_result.display_name if self.active_result else "none"
        self.classification_completed.emit(
            {"best_model": best, "results": len(self.experiment_result.results)}
        )
        self.log(f"[Training] Comparison finished. Best model: {best}.")

    def _write_predictions_from_active_result(self) -> None:
        if self.active_result is None or self.feature_matrix is None:
            return
        predictions = self.active_result.out_of_fold_predictions
        if predictions is None:
            return
        probabilities = self.active_result.probabilities
        for index, sample in enumerate(self.feature_matrix.samples):
            if index >= len(predictions):
                continue
            sample.predicted_label = str(predictions[index])
            if (
                probabilities is not None
                and index < probabilities.shape[0]
                and np.any(np.isfinite(probabilities[index]))
            ):
                sample.confidence = float(np.nanmax(probabilities[index]))

    def _update_results_views(self) -> None:
        page = self.page
        if page is None:
            return
        result = self.experiment_result
        page.resultsOutdatedLabel.setText(
            "Outdated: settings changed after the last run." if self._results_outdated else ""
        )
        if result is None:
            page.bestModelLabel.setText("-")
            page.bestMacroF1Label.setText("-")
            page.bestBalancedAccuracyLabel.setText("-")
            page.bestAccuracyLabel.setText("-")
            page.resultSamplesLabel.setText("0")
            page.resultClassesLabel.setText("0")
            page.resultValidationLabel.setText("-")
            page.resultsTable.setRowCount(0)
            page.activeModelCombo.clear()
            page.metricChartLabel.setText("No metrics yet")
            page.confusionMatrixTable.setRowCount(0)
            page.confusionMatrixTable.setColumnCount(0)
            page.perClassTable.setRowCount(0)
            page.misclassifiedTable.setRowCount(0)
            return

        best = result.best_result
        page.bestModelLabel.setText(best.display_name if best else "-")
        page.bestMacroF1Label.setText(self._metric_text(best, "macro_f1"))
        page.bestBalancedAccuracyLabel.setText(self._metric_text(best, "balanced_accuracy"))
        page.bestAccuracyLabel.setText(self._metric_text(best, "accuracy"))
        page.resultSamplesLabel.setText(str(len(result.y_true)))
        page.resultClassesLabel.setText(str(len(result.labels)))
        page.resultValidationLabel.setText(result.validation_config.method)
        page.validationWarningLabel.setText(" ".join(result.warnings))

        table = page.resultsTable
        table.setSortingEnabled(False)
        table.setRowCount(len(result.results))
        for row, item in enumerate(result.results, start=1):
            values = [
                str(row),
                item.display_name,
                self._metric_text(item, "accuracy"),
                self._metric_text(item, "balanced_accuracy"),
                self._metric_text(item, "macro_f1"),
                self._metric_text(item, "weighted_f1"),
                f"{item.training_time:.3f}s" if item.status == "ok" else "-",
                f"{item.prediction_time:.3f}s" if item.status == "ok" else "-",
                "OK" if item.status == "ok" else f"Failed: {item.error_message}",
            ]
            for col, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if col == 1:
                    table_item.setData(Qt.UserRole, item.algorithm_id)
                table.setItem(row - 1, col, table_item)
        table.setSortingEnabled(True)
        table.selectRow(0)

        page.activeModelCombo.blockSignals(True)
        page.activeModelCombo.clear()
        for item in result.successful_results:
            page.activeModelCombo.addItem(item.display_name, item.algorithm_id)
        if best:
            page.activeModelCombo.setCurrentText(best.display_name)
        page.activeModelCombo.blockSignals(False)

        self._render_metric_chart()
        self._update_selected_result_details()

    def _update_selected_result_details(self) -> None:
        selected = self._selected_result()
        if selected is None:
            return
        self._render_confusion_matrix(selected)
        self._render_per_class_metrics(selected)
        self._render_misclassified_table(selected)

    def _selected_result(self) -> Optional[ModelEvaluationResult]:
        if self.experiment_result is None or self.page is None:
            return None
        row = self.page.resultsTable.currentRow()
        algorithm_id = None
        if row >= 0:
            item = self.page.resultsTable.item(row, 1)
            algorithm_id = item.data(Qt.UserRole) if item is not None else None
        if algorithm_id:
            for result in self.experiment_result.results:
                if result.algorithm_id == algorithm_id:
                    return result
        return self.experiment_result.best_result

    def _select_active_model_by_name(self, name: str) -> None:
        if self.experiment_result is None:
            return
        for result in self.experiment_result.successful_results:
            if result.display_name == name:
                self.active_result = result
                self._write_predictions_from_active_result()
                self._update_dataset_table()
                self._update_selected_result_details()
                self.log(f"[Model] Active model set to {name}.")
                return

    def _save_active_model(self) -> None:
        if (
            self.active_result is None
            or self.active_result.fitted_pipeline is None
            or self.experiment_result is None
        ):
            QMessageBox.warning(
                self.main_window, "Save Model", "No active trained model is available."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Active Model", "", "Joblib (*.joblib);;Pickle (*.pkl)"
        )
        if not path:
            return
        package = self._build_saved_model_package(self.active_result)
        saved_path = self.classification_view_model.save_model(Path(path), package)
        if saved_path is not None:
            self.active_model_package = package
            self.log(f"[Model] Saved active model to {path}")
            return
        QMessageBox.warning(
            self.main_window,
            "Save Model",
            self.classification_view_model.state.error_message or "Model save failed",
        )

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Load Model", "", "Joblib (*.joblib);;Pickle (*.pkl);;All files (*.*)"
        )
        if not path:
            return
        package = self.classification_view_model.load_model(Path(path))
        if package is not None:
            self.active_model_package = package
            self.log(f"[Model] Loaded model: {package.display_name}")
            return
        QMessageBox.warning(
            self.main_window,
            "Load Model",
            self.classification_view_model.state.error_message
            or "Unsupported legacy model package.",
        )

    def _predict_new_data_menu(self) -> None:
        menu = QMenu(self.page.predictNewDataButton)
        files_action = menu.addAction("Choose Files")
        folder_action = menu.addAction("Choose Folder")
        action = menu.exec_(
            self.page.predictNewDataButton.mapToGlobal(
                self.page.predictNewDataButton.rect().bottomLeft()
            )
        )
        if action == files_action:
            files, _ = QFileDialog.getOpenFileNames(
                self.main_window, "Choose unknown data", "", self._file_dialog_filter()
            )
            if files:
                self._start_prediction(files)
        elif action == folder_action:
            folder = QFileDialog.getExistingDirectory(
                self.main_window, "Choose unknown data folder"
            )
            if folder:
                self._start_prediction([folder])

    def _start_prediction(self, paths: list[str]) -> None:
        if self.current_worker is not None:
            QMessageBox.information(
                self.main_window, "Classification", "A Classification task is already running."
            )
            return
        package = self.active_model_package
        if package is None:
            if self.active_result is not None:
                package = self._build_saved_model_package(self.active_result)
                self.active_model_package = package
            else:
                QMessageBox.warning(
                    self.main_window, "Prediction", "Train, load, or save an active model first."
                )
                return
        worker = PredictionWorker(paths, package, self.classification_view_model)
        self.current_worker = worker
        self._set_state(ClassificationPageState.PREDICTING)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(self._on_prediction_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_prediction_finished(self, results) -> None:
        self.current_worker = None
        self.prediction_results = list(results or [])
        self._render_prediction_table()
        self._set_state(
            ClassificationPageState.RESULTS_AVAILABLE
            if self.experiment_result
            else ClassificationPageState.READY
        )
        self.log(f"[Prediction] Predicted {len(self.prediction_results)} file(s).")

    def _start_embedding(self) -> None:
        if self.current_worker is not None:
            QMessageBox.information(
                self.main_window, "Classification", "A Classification task is already running."
            )
            return
        if not self.samples:
            QMessageBox.warning(
                self.main_window, "Embedding", "Import data before running embedding visualization."
            )
            return
        worker = EmbeddingWorker(
            self.samples,
            self._collect_preprocessing_config(),
            self.page.embeddingMethodCombo.currentText(),
            self.classification_view_model,
        )
        self.current_worker = worker
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(self._on_embedding_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_embedding_finished(self, payload) -> None:
        self.current_worker = None
        if not isinstance(payload, dict):
            self._on_worker_error("Embedding returned an invalid payload.")
            return
        self._render_embedding(payload["embedding"], payload["matrix"].samples)
        self.log(f"[Embedding] {payload.get('method', 'Embedding')} complete.")
