"""Lifecycle Config coordination for Classification."""

from __future__ import annotations


import time


from dataclasses import asdict


from src.gimap.features.classification.application import (
    AlgorithmConfig,
    ClassificationPageState,
    DatasetSource,
    DatasetSummary,
    PreprocessingConfig,
    ProjectionConfig,
    ValidationConfig,
)


from ..ranking_labels import RANKING_METRIC_BY_LABEL


class LifecycleConfigMixin:
    """Own lifecycle config presentation behavior."""

    def initialize(self):
        """Install the new Classification page and connect its workflow."""

        if self._initialized:
            return
        self._populate_algorithm_table()
        self._connect_signals()
        self._restore_global_parameters()
        self._refresh_everything()
        self._set_state(
            ClassificationPageState.EMPTY if not self.sources else ClassificationPageState.SCANNED
        )
        self._initialized = True
        self.log("[UI] Classification page ready.")

    def get_parameters(self):
        """Return session parameters for the application runtime."""

        return {
            "sources": [asdict(source) for source in self.sources.values()],
            "import_cache": {
                label: {
                    "path": ";".join(source.paths),
                    "rule": source.file_pattern,
                    "source_type": source.source_type,
                    "color": source.color,
                }
                for label, source in self.sources.items()
            },
            "preprocessing": asdict(self._collect_preprocessing_config())
            if self.page
            else asdict(PreprocessingConfig()),
            "validation": asdict(self._collect_validation_config())
            if self.page
            else asdict(ValidationConfig()),
            "projection": asdict(self._collect_projection_config())
            if self.page
            else asdict(ProjectionConfig()),
            "algorithms": [asdict(config) for config in self.algorithm_configs],
            "ranking_metric": self._ranking_metric(),
        }

    def set_parameters(self, parameters):
        """Restore sources and workflow configuration."""

        if not isinstance(parameters, dict):
            return
        self.sources = self._sources_from_parameters(parameters)
        algorithms = parameters.get("algorithms")
        if isinstance(algorithms, list):
            defaults = {
                config.algorithm_id: config
                for config in self.classification_view_model.default_algorithms()
            }
            restored: list[AlgorithmConfig] = []
            for raw in algorithms:
                if not isinstance(raw, dict):
                    continue
                base = defaults.get(raw.get("algorithm_id"))
                if base is None:
                    continue
                restored.append(
                    AlgorithmConfig(
                        algorithm_id=base.algorithm_id,
                        display_name=base.display_name,
                        enabled=bool(raw.get("enabled", base.enabled)),
                        parameters=dict(raw.get("parameters", base.parameters)),
                        description=base.description,
                        requires_scaling=base.requires_scaling,
                    )
                )
            if restored:
                self.algorithm_configs = restored
        if self.page is not None:
            self._apply_config_to_page(parameters)
            self._populate_algorithm_table()
            self._refresh_everything()
        self.parameters_changed.emit(self.get_parameters())

    def validate_parameters(self):
        if self.summary.classes < 2:
            return False, "At least two classes are required."
        if self.summary.valid_samples < 2:
            return False, "At least two valid samples are required."
        return True, "OK"

    def reset_to_defaults(self):
        self._cancel_current_task()
        self.sources.clear()
        self.samples.clear()
        self.summary = DatasetSummary()
        self.experiment_result = None
        self.feature_matrix = None
        self.active_result = None
        self.active_model_package = None
        self.prediction_results = []
        self._results_outdated = False
        self.algorithm_configs = self.classification_view_model.default_algorithms()
        if self.page is not None:
            self._populate_algorithm_table()
            self._refresh_everything()
        self._set_state(ClassificationPageState.EMPTY)
        self._persist_parameters()
        self.log("[Reset] Classification session reset.")

    def log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        browser = getattr(self.page, "logTextBrowser", None) if self.page else None
        if browser is not None:
            browser.append(line)
            bar = browser.verticalScrollBar()
            bar.setValue(bar.maximum())
        self.status_updated.emit(message)

    def _connect_signals(self) -> None:
        page = self.page
        if page is None:
            return
        page.newSessionButton.clicked.connect(self.reset_to_defaults)
        page.loadSessionButton.clicked.connect(self._load_session)
        page.saveSessionButton.clicked.connect(self._save_session)
        page.helpButton.clicked.connect(self._show_help)
        page.addClassButton.clicked.connect(self._add_class_dialog)
        page.scanImportButton.clicked.connect(lambda: self._start_import())
        page.filesDropped.connect(self._on_files_dropped)
        page.datasetTable.currentCellChanged.connect(
            lambda *_: self._preview_current_table_sample()
        )
        page.datasetTable.itemChanged.connect(self._on_dataset_item_changed)
        page.datasetSearchEdit.textChanged.connect(self._update_dataset_table)
        page.classFilterCombo.currentTextChanged.connect(self._update_dataset_table)
        page.qcFilterCombo.currentTextChanged.connect(self._update_dataset_table)
        page.excludeSelectedButton.clicked.connect(lambda: self._set_selected_included(False))
        page.includeSelectedButton.clicked.connect(lambda: self._set_selected_included(True))
        page.removeSelectedSamplesButton.clicked.connect(self._remove_selected_samples)
        page.openSelectedLocationButton.clicked.connect(self._open_selected_location)
        page.copySelectedPathsButton.clicked.connect(self._copy_selected_paths)
        page.exportSelectedFilesButton.clicked.connect(self._export_selected_file_list)
        page.prevSampleButton.clicked.connect(lambda: self._move_preview(-1))
        page.nextSampleButton.clicked.connect(lambda: self._move_preview(1))
        for widget in (
            page.previewLogScaleCheckBox,
            page.previewColormapCombo,
            page.previewAutoScaleCheckBox,
            page.previewVminSpinBox,
            page.previewVmaxSpinBox,
        ):
            signal = (
                getattr(widget, "stateChanged", None)
                or getattr(widget, "currentTextChanged", None)
                or getattr(widget, "valueChanged", None)
            )
            if signal is not None:
                signal.connect(lambda *_: self._render_current_preview())
        page.fitPreviewButton.clicked.connect(self._fit_preview)
        page.openFileLocationButton.clicked.connect(self._open_selected_location)

        config_widgets = [
            page.oneDPreprocessingCombo,
            page.twoDPreprocessingCombo,
            page.normalizeCombo,
            page.preprocessingLogCheckBox,
            page.smoothingSpinBox,
            page.resizeRowsSpinBox,
            page.resizeColsSpinBox,
            page.validationMethodCombo,
            page.testSizeSpinBox,
            page.foldsSpinBox,
            page.repeatsSpinBox,
            page.randomSeedSpinBox,
            page.shuffleCheckBox,
            page.rankingMetricCombo,
            page.useProjectionCheckBox,
            page.projectionMethodCombo,
            page.projectionComponentsSpinBox,
            page.pcaVarianceSpinBox,
            page.umapNeighborsSpinBox,
            page.umapMinDistSpinBox,
        ]
        for widget in config_widgets:
            for signal_name in ("currentTextChanged", "valueChanged", "stateChanged", "toggled"):
                signal = getattr(widget, signal_name, None)
                if signal is not None:
                    signal.connect(lambda *_: self._on_configuration_changed())
                    break

        page.selectRecommendedButton.clicked.connect(self._select_recommended_algorithms)
        page.selectAllAlgorithmsButton.clicked.connect(lambda: self._set_all_algorithms(True))
        page.clearAlgorithmsButton.clicked.connect(lambda: self._set_all_algorithms(False))
        page.resetAlgorithmDefaultsButton.clicked.connect(self._reset_algorithm_defaults)
        page.algorithmTable.itemChanged.connect(lambda *_: self._on_algorithm_selection_changed())
        page.runComparisonButton.clicked.connect(self._start_training)
        page.cancelTaskButton.clicked.connect(self._cancel_current_task)
        page.resultsTable.currentCellChanged.connect(
            lambda *_: self._update_selected_result_details()
        )
        page.resultsTable.cellDoubleClicked.connect(
            lambda *_: self._update_selected_result_details()
        )
        page.confusionNormalizeCombo.currentTextChanged.connect(
            lambda *_: self._update_selected_result_details()
        )
        page.activeModelCombo.currentTextChanged.connect(self._select_active_model_by_name)
        page.setActiveModelButton.clicked.connect(
            lambda: self._select_active_model_by_name(page.activeModelCombo.currentText())
        )
        page.saveActiveModelButton.clicked.connect(self._save_active_model)
        page.loadModelButton.clicked.connect(self._load_model)
        page.exportResultsButton.clicked.connect(self._export_results_csv)
        page.predictNewDataButton.clicked.connect(self._predict_new_data_menu)
        page.exportPredictionsButton.clicked.connect(self._export_predictions_csv)
        page.runEmbeddingButton.clicked.connect(self._start_embedding)
        page.misclassifiedTable.currentCellChanged.connect(
            lambda *_: self._preview_selected_misclassification()
        )

    def _restore_global_parameters(self) -> None:
        try:
            params = self.classification_view_model.load_settings()
            self.set_parameters(params)
        except Exception as exc:
            self.log(f"[Session] Failed to restore Classification parameters: {exc}")

    def _sources_from_parameters(self, parameters: dict) -> dict[str, DatasetSource]:
        sources: dict[str, DatasetSource] = {}
        raw_sources = parameters.get("sources")
        if isinstance(raw_sources, list):
            for raw in raw_sources:
                if not isinstance(raw, dict):
                    continue
                label = str(raw.get("label", "")).strip()
                if not label:
                    continue
                sources[label] = DatasetSource(
                    label=label,
                    source_type=str(raw.get("source_type", "folder")),
                    paths=[str(path) for path in raw.get("paths", [])],
                    file_pattern=str(raw.get("file_pattern", "*")),
                    color=str(raw.get("color", self._next_color(len(sources)))),
                    recursive=bool(raw.get("recursive", True)),
                )
        cache = parameters.get("import_cache")
        if not sources and isinstance(cache, dict):
            for label, raw in cache.items():
                if not isinstance(raw, dict):
                    continue
                path_value = raw.get("path", "")
                paths = [part for part in str(path_value).split(";") if part]
                sources[str(label)] = DatasetSource(
                    label=str(label),
                    source_type=str(raw.get("source_type", "folder")),
                    paths=paths,
                    file_pattern=str(raw.get("rule", "*")),
                    color=str(raw.get("color", self._next_color(len(sources)))),
                )
        return sources

    def _apply_config_to_page(self, parameters: dict) -> None:
        page = self.page
        if page is None:
            return
        preprocessing = parameters.get("preprocessing", {})
        if isinstance(preprocessing, dict):
            page.oneDPreprocessingCombo.setCurrentText(
                str(preprocessing.get("one_d_method", page.oneDPreprocessingCombo.currentText()))
            )
            page.twoDPreprocessingCombo.setCurrentText(
                str(preprocessing.get("two_d_method", page.twoDPreprocessingCombo.currentText()))
            )
            page.normalizeCombo.setCurrentText(
                str(preprocessing.get("normalize", page.normalizeCombo.currentText()))
            )
            page.preprocessingLogCheckBox.setChecked(
                bool(preprocessing.get("log_transform", False))
            )
            page.smoothingSpinBox.setValue(int(preprocessing.get("smoothing_window", 0) or 0))
            resize_shape = preprocessing.get("resize_shape")
            if isinstance(resize_shape, (list, tuple)) and len(resize_shape) == 2:
                page.resizeRowsSpinBox.setValue(int(resize_shape[0]))
                page.resizeColsSpinBox.setValue(int(resize_shape[1]))
        validation = parameters.get("validation", {})
        if isinstance(validation, dict):
            page.validationMethodCombo.setCurrentText(
                str(validation.get("method", page.validationMethodCombo.currentText()))
            )
            page.testSizeSpinBox.setValue(float(validation.get("test_size", 0.2)))
            page.foldsSpinBox.setValue(int(validation.get("folds", 5)))
            page.repeatsSpinBox.setValue(int(validation.get("repeats", 1)))
            page.randomSeedSpinBox.setValue(int(validation.get("random_state", 42)))
            page.shuffleCheckBox.setChecked(bool(validation.get("shuffle", True)))
        projection = parameters.get("projection", {})
        if isinstance(projection, dict):
            page.useProjectionCheckBox.setChecked(bool(projection.get("enabled", False)))
            page.projectionMethodCombo.setCurrentText(str(projection.get("method", "None")))
            page.projectionComponentsSpinBox.setValue(int(projection.get("n_components", 2)))
            page.pcaVarianceSpinBox.setValue(float(projection.get("explained_variance", 0.95)))
            page.umapNeighborsSpinBox.setValue(int(projection.get("umap_neighbors", 15)))
            page.umapMinDistSpinBox.setValue(float(projection.get("umap_min_dist", 0.1)))
        metric = str(parameters.get("ranking_metric", "macro_f1"))
        reverse = {value: key for key, value in RANKING_METRIC_BY_LABEL.items()}
        page.rankingMetricCombo.setCurrentText(reverse.get(metric, "Macro F1"))

    def _refresh_everything(self) -> None:
        self.summary = self.classification_view_model.validate_dataset(self.samples)
        self._render_dataset_cards()
        self._update_dataset_table()
        self._update_quality()
        self._update_input_summary()
        self._update_run_summary()
        self._update_results_views()
