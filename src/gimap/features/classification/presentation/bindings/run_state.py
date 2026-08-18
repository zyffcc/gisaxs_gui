"""Run State coordination for Classification."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.features.classification.application import (
    ClassificationPageState,
    PreprocessingConfig,
    ProjectionConfig,
    ValidationConfig,
)


from ..ranking_labels import RANKING_METRIC_BY_LABEL


class RunStateMixin:
    """Own run state presentation behavior."""

    def _update_input_summary(self) -> None:
        page = self.page
        if page is None:
            return
        try:
            matrix = self.classification_view_model.build_features(
                self.samples,
                self._collect_preprocessing_config(),
                require_labels=True,
            )
            if matrix is None:
                raise ValueError("Feature construction is not available")
            memory = self.classification_view_model.estimate_feature_memory(matrix)
            page.inputSummaryLabel.setText(
                f"Samples: {matrix.X.shape[0]} | Features: {matrix.X.shape[1]} | Input shape: {matrix.input_shape} | Memory: {memory}"
            )
        except Exception:
            page.inputSummaryLabel.setText("Samples: 0 | Features: 0 | Input shape: - | Memory: -")

    def _update_run_summary(self) -> None:
        page = self.page
        if page is None:
            return
        selected = (
            len([config for config in self._collect_algorithm_configs() if config.enabled])
            if page.algorithmTable.rowCount()
            else 0
        )
        valid = self.summary.valid_samples
        folds = page.foldsSpinBox.value()
        method = page.validationMethodCombo.currentText()
        runs = selected * (folds if "K-fold" in method else 1)
        page.runStatusLabel.setText(
            f"Selected algorithms: {selected} | Valid samples: {valid} | Estimated runs: {runs} | {self.state.value}"
        )

    def _set_state(self, state: ClassificationPageState) -> None:
        self.state = state
        page = self.page
        if page is None:
            return
        page.stateBadgeLabel.setText(state.value)
        if state in {ClassificationPageState.EMPTY, ClassificationPageState.SCANNED}:
            page.set_step("Dataset")
        elif state == ClassificationPageState.READY:
            page.set_step("Algorithms")
        elif state in {
            ClassificationPageState.IMPORTING,
            ClassificationPageState.TRAINING,
            ClassificationPageState.PREDICTING,
        }:
            page.set_step("Algorithms" if state == ClassificationPageState.TRAINING else "Dataset")
        elif state == ClassificationPageState.RESULTS_AVAILABLE:
            page.set_step("Results")
        busy = state in {
            ClassificationPageState.IMPORTING,
            ClassificationPageState.TRAINING,
            ClassificationPageState.PREDICTING,
        }
        page.cancelTaskButton.setEnabled(busy)
        page.runComparisonButton.setEnabled(not busy and self.summary.valid_samples >= 2)
        page.scanImportButton.setEnabled(not busy)
        page.addClassButton.setEnabled(not busy)
        page.predictNewDataButton.setEnabled(not busy)
        self.progress_updated.emit(0 if not busy else page.taskProgressBar.value())
        self._update_run_summary()
        job_state = {
            ClassificationPageState.IMPORTING: "running",
            ClassificationPageState.TRAINING: "running",
            ClassificationPageState.PREDICTING: "running",
            ClassificationPageState.RESULTS_AVAILABLE: "succeeded",
            ClassificationPageState.ERROR: "failed",
        }.get(state, "idle")
        page.set_job_state(
            job_state,
            progress=page.taskProgressBar.value()
            if busy
            else (100 if job_state == "succeeded" else 0),
        )

    def _on_worker_progress(self, percent: int, message: str) -> None:
        if self.page is not None:
            self.page.taskProgressBar.setValue(max(0, min(100, int(percent))))
        self.progress_updated.emit(max(0, min(100, int(percent))))
        self.status_updated.emit(message)

    def _on_worker_error(self, message: str) -> None:
        self.current_worker = None
        self._set_state(ClassificationPageState.ERROR)
        self.log(f"[Error] {message}")
        QMessageBox.warning(
            self.main_window,
            "Classification Error",
            str(message).splitlines()[-1] if message else "Unknown error",
        )

    def _cancel_current_task(self) -> None:
        if self.current_worker is not None and hasattr(self.current_worker, "cancel"):
            self.current_worker.cancel()
            self.log("[Task] Cancellation requested.")
            if self.page is not None:
                self.page.cancelTaskButton.setEnabled(False)

    def _collect_preprocessing_config(self) -> PreprocessingConfig:
        page = self.page
        resize_shape = None
        if page.twoDPreprocessingCombo.currentText() == "Resize":
            resize_shape = (page.resizeRowsSpinBox.value(), page.resizeColsSpinBox.value())
        return PreprocessingConfig(
            data_type=page.dataTypeBadgeLabel.text(),
            one_d_method=page.oneDPreprocessingCombo.currentText(),
            two_d_method=page.twoDPreprocessingCombo.currentText(),
            normalize=page.normalizeCombo.currentText(),
            log_transform=page.preprocessingLogCheckBox.isChecked(),
            smoothing_window=page.smoothingSpinBox.value(),
            resize_shape=resize_shape,
            flatten=True,
        )

    def _collect_validation_config(self) -> ValidationConfig:
        page = self.page
        return ValidationConfig(
            method=page.validationMethodCombo.currentText(),
            test_size=float(page.testSizeSpinBox.value()),
            folds=int(page.foldsSpinBox.value()),
            repeats=int(page.repeatsSpinBox.value()),
            shuffle=page.shuffleCheckBox.isChecked(),
            random_state=int(page.randomSeedSpinBox.value()),
        )

    def _collect_projection_config(self) -> ProjectionConfig:
        page = self.page
        return ProjectionConfig(
            enabled=page.useProjectionCheckBox.isChecked()
            and page.projectionMethodCombo.currentText() != "None",
            method=page.projectionMethodCombo.currentText(),
            n_components=int(page.projectionComponentsSpinBox.value()),
            explained_variance=float(page.pcaVarianceSpinBox.value()),
            umap_neighbors=int(page.umapNeighborsSpinBox.value()),
            umap_min_dist=float(page.umapMinDistSpinBox.value()),
        )

    def _ranking_metric(self) -> str:
        if self.page is None:
            return "macro_f1"
        return RANKING_METRIC_BY_LABEL.get(self.page.rankingMetricCombo.currentText(), "macro_f1")

    def _mark_results_outdated(self) -> None:
        if self.experiment_result is not None:
            self._results_outdated = True

    def _on_configuration_changed(self) -> None:
        self._mark_results_outdated()
        self._update_input_summary()
        self._update_run_summary()
        self._update_results_views()
        self._persist_parameters()

    def _on_algorithm_selection_changed(self) -> None:
        self._collect_algorithm_configs()
        self._mark_results_outdated()
        self._update_run_summary()
        self._update_results_views()
        self._persist_parameters()

    def _persist_parameters(self) -> None:
        try:
            params = self.get_parameters()
            self.classification_view_model.save_settings(params)
            self.parameters_changed.emit(params)
        except Exception as exc:
            self.log(f"[Session] Parameter persistence failed: {exc}")
