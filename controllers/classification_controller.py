"""Controller for the refactored Classification workflow."""

from __future__ import annotations

import csv
import io
import json
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QThreadPool, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.global_params import global_params
from controllers.classification_data_service import ClassificationDataService
from controllers.classification_models import (
    AlgorithmConfig,
    ClassificationPageState,
    ClassificationSample,
    DatasetSource,
    DatasetSummary,
    ExperimentResult,
    ModelEvaluationResult,
    PredictionResult,
    PreprocessingConfig,
    ProjectionConfig,
    SavedModelPackage,
    ValidationConfig,
)
from controllers.classification_training_service import ClassificationTrainingService
from controllers.classification_workers import EmbeddingWorker, ImportWorker, PredictionWorker, TrainingWorker
from ui.classification_page import ClassificationPage


RANKING_METRIC_BY_LABEL = {
    "Macro F1": "macro_f1",
    "Balanced Accuracy": "balanced_accuracy",
    "Accuracy": "accuracy",
}


class ClassificationController(QObject):
    """Coordinate the Classification page, data service, workers, and results."""

    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    parameters_changed = pyqtSignal(dict)
    classification_completed = pyqtSignal(dict)

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent = parent
        self.main_window = getattr(parent, "parent", None)

        self.data_service = ClassificationDataService()
        self.training_service = ClassificationTrainingService()
        self.thread_pool = QThreadPool.globalInstance()
        self.algorithm_configs: list[AlgorithmConfig] = self.training_service.default_algorithm_configs()

        self.sources: dict[str, DatasetSource] = {}
        self.samples: list[ClassificationSample] = []
        self.summary = DatasetSummary()
        self.experiment_result: Optional[ExperimentResult] = None
        self.feature_matrix = None
        self.active_result: Optional[ModelEvaluationResult] = None
        self.active_model_package: Optional[SavedModelPackage] = None
        self.prediction_results: list[PredictionResult] = []
        self.current_preview_sample_id: Optional[str] = None
        self.current_worker = None
        self._initialized = False
        self._table_updating = False
        self._results_outdated = False
        self.state = ClassificationPageState.EMPTY
        self.page: Optional[ClassificationPage] = None

    def initialize(self):
        """Install the new Classification page and connect its workflow."""

        if self._initialized:
            return
        self._install_page()
        self._install_compatibility_aliases()
        self._populate_algorithm_table()
        self._connect_signals()
        self._restore_global_parameters()
        self._refresh_everything()
        self._set_state(ClassificationPageState.EMPTY if not self.sources else ClassificationPageState.SCANNED)
        self._initialized = True
        self.log("[UI] Classification page ready.")

    def get_parameters(self):
        """Return session parameters that can be saved by MainController."""

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
            "preprocessing": asdict(self._collect_preprocessing_config()) if self.page else asdict(PreprocessingConfig()),
            "validation": asdict(self._collect_validation_config()) if self.page else asdict(ValidationConfig()),
            "projection": asdict(self._collect_projection_config()) if self.page else asdict(ProjectionConfig()),
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
            defaults = {config.algorithm_id: config for config in self.training_service.default_algorithm_configs()}
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
        self.algorithm_configs = self.training_service.default_algorithm_configs()
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
        if browser is None:
            browser = getattr(self.ui, "classificationPageTextBrowser", None)
        if browser is not None:
            browser.append(line)
            bar = browser.verticalScrollBar()
            bar.setValue(bar.maximum())
        self.status_updated.emit(message)

    def _install_page(self) -> None:
        host = getattr(self.ui, "classificationPage", None)
        if host is None:
            raise RuntimeError("Generated UI does not expose classificationPage.")
        layout = host.layout()
        if layout is None:
            layout = QVBoxLayout(host)
            layout.setContentsMargins(0, 0, 0, 0)
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.page = ClassificationPage(host)
        layout.addWidget(self.page)

    def _install_compatibility_aliases(self) -> None:
        page = self.page
        if page is None:
            return
        compatibility_container = QWidget(page)
        compatibility_container.setObjectName("classificationCompatibilityContainer")
        compatibility_container.hide()
        self._compatibility_container = compatibility_container
        legacy_rule_label = QLabel("File pattern", compatibility_container)
        legacy_rule_value = QLineEdit("*", compatibility_container)
        legacy_knn_label = QLabel("KNeighbors", compatibility_container)
        legacy_knn_value = QLineEdit("5", compatibility_container)
        aliases = {
            "addClassButton": page.addClassButton,
            "datasetTable": page.datasetTable,
            "runComparisonButton": page.runComparisonButton,
            "algorithmList": page.algorithmTable,
            "validationMethodCombo": page.validationMethodCombo,
            "resultsTable": page.resultsTable,
            "confusionMatrixView": page.confusionMatrixTable,
            "misclassifiedTable": page.misclassifiedTable,
            "activeModelCombo": page.activeModelCombo,
            "predictNewDataButton": page.predictNewDataButton,
            "ClassificationImportTableWidget": page.datasetTable,
            "ClassificationGraphicsView": page.previewGraphicsView,
            "ClassificationPanelWidget": page,
            "ClassificationImportListWidget": page.legacyClassListWidget,
            "ClassificationImportPlusButton": page.addClassButton,
            "ClassificationImportMinusButton": page.removeSelectedSamplesButton,
            "ClassificationImportImportButton": page.scanImportButton,
            "ClassificationImportClassifyButton": page.predictNewDataButton,
            "ClassificationImportFolderPathLabel": page.scanImportButton,
            "ClassificationImportFolderPathValue": page.datasetSearchEdit,
            "ClassificationImportRuleLabel": legacy_rule_label,
            "ClassificationImportRuleValue": legacy_rule_value,
            "DimensionalityReductionMethodCombox": page.projectionMethodCombo,
            "DimensionalityReductionTargetDimValue": page.projectionComponentsSpinBox,
            "DimensionalityReductionNNeighborValue": page.umapNeighborsSpinBox,
            "DimensionalityReductionStartButton": page.runEmbeddingButton,
            "DimensionalityReductionShowResultButton": page.runEmbeddingButton,
            "ClassificationMethodCombox": page.activeModelCombo,
            "ClassificationKNnnNneighborsLabel": legacy_knn_label,
            "ClassificationKNnnNneighborsValue": legacy_knn_value,
            "ClassificationClassifyButton": page.runComparisonButton,
            "ClassificationSaveModelButton": page.saveActiveModelButton,
            "ClassificationLoadModelButton": page.loadModelButton,
            "classificationPageTextBrowser": page.logTextBrowser,
        }
        for name, widget in aliases.items():
            setattr(self.ui, name, widget)

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
        page.datasetTable.currentCellChanged.connect(lambda *_: self._preview_current_table_sample())
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
            signal = getattr(widget, "stateChanged", None) or getattr(widget, "currentTextChanged", None) or getattr(widget, "valueChanged", None)
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
        page.resultsTable.currentCellChanged.connect(lambda *_: self._update_selected_result_details())
        page.resultsTable.cellDoubleClicked.connect(lambda *_: self._update_selected_result_details())
        page.confusionNormalizeCombo.currentTextChanged.connect(lambda *_: self._update_selected_result_details())
        page.activeModelCombo.currentTextChanged.connect(self._select_active_model_by_name)
        page.setActiveModelButton.clicked.connect(lambda: self._select_active_model_by_name(page.activeModelCombo.currentText()))
        page.saveActiveModelButton.clicked.connect(self._save_active_model)
        page.loadModelButton.clicked.connect(self._load_model)
        page.exportResultsButton.clicked.connect(self._export_results_csv)
        page.predictNewDataButton.clicked.connect(self._predict_new_data_menu)
        page.exportPredictionsButton.clicked.connect(self._export_predictions_csv)
        page.runEmbeddingButton.clicked.connect(self._start_embedding)
        page.misclassifiedTable.currentCellChanged.connect(lambda *_: self._preview_selected_misclassification())

    def _restore_global_parameters(self) -> None:
        try:
            params = global_params.get_module_parameters("classification")
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
            page.oneDPreprocessingCombo.setCurrentText(str(preprocessing.get("one_d_method", page.oneDPreprocessingCombo.currentText())))
            page.twoDPreprocessingCombo.setCurrentText(str(preprocessing.get("two_d_method", page.twoDPreprocessingCombo.currentText())))
            page.normalizeCombo.setCurrentText(str(preprocessing.get("normalize", page.normalizeCombo.currentText())))
            page.preprocessingLogCheckBox.setChecked(bool(preprocessing.get("log_transform", False)))
            page.smoothingSpinBox.setValue(int(preprocessing.get("smoothing_window", 0) or 0))
            resize_shape = preprocessing.get("resize_shape")
            if isinstance(resize_shape, (list, tuple)) and len(resize_shape) == 2:
                page.resizeRowsSpinBox.setValue(int(resize_shape[0]))
                page.resizeColsSpinBox.setValue(int(resize_shape[1]))
        validation = parameters.get("validation", {})
        if isinstance(validation, dict):
            page.validationMethodCombo.setCurrentText(str(validation.get("method", page.validationMethodCombo.currentText())))
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
        self.summary = self.data_service.validate_dataset(self.samples)
        self._render_dataset_cards()
        self._update_dataset_table()
        self._update_quality()
        self._update_input_summary()
        self._update_run_summary()
        self._update_results_views()
        self._sync_legacy_class_list()

    def _render_dataset_cards(self) -> None:
        page = self.page
        if page is None:
            return
        page.clear_dataset_cards()
        label_summary = self.data_service.summarize_by_label(self.samples)
        for source in self.sources.values():
            card = self._create_dataset_card(source, label_summary.get(source.label, {}))
            page.add_dataset_card(card)

    def _create_dataset_card(self, source: DatasetSource, summary: dict[str, object]) -> QWidget:
        page = self.page
        card = QFrame(page)
        card.setProperty("classificationCard", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        color = QLabel(card)
        color.setFixedSize(14, 14)
        color.setStyleSheet(f"background: {source.color}; border-radius: 7px;")
        title = QLabel(source.label, card)
        title.setStyleSheet("font-weight: 700;")
        status = QLabel(str(summary.get("status", "Empty")), card)
        status.setProperty("classificationBadge", True)
        top.addWidget(color)
        top.addWidget(title, 1)
        top.addWidget(status)
        layout.addLayout(top)

        details = QGridLayout()
        details.setHorizontalSpacing(8)
        detail_items = [
            ("Files", summary.get("files", 0)),
            ("Loaded", summary.get("loaded", 0)),
            ("Failed", summary.get("failed", 0)),
            ("Type", summary.get("data_type", "-")),
            ("Shape", summary.get("shape", "-")),
            ("Pattern", source.file_pattern or "*"),
            ("Path", self._short_paths(source.paths)),
        ]
        for row, (name, value) in enumerate(detail_items):
            details.addWidget(QLabel(str(name), card), row, 0)
            value_label = QLabel(str(value), card)
            value_label.setWordWrap(True)
            details.addWidget(value_label, row, 1)
        layout.addLayout(details)

        buttons = QHBoxLayout()
        choose_folder = QPushButton("Choose Folder", card)
        choose_files = QPushButton("Choose Files", card)
        edit = QPushButton("Edit", card)
        remove = QPushButton("Remove", card)
        rescan = QPushButton("Rescan", card)
        choose_folder.clicked.connect(lambda: self._choose_source_folder(source.label))
        choose_files.clicked.connect(lambda: self._choose_source_files(source.label))
        edit.clicked.connect(lambda: self._edit_source_dialog(source.label))
        remove.clicked.connect(lambda: self._remove_source(source.label))
        rescan.clicked.connect(lambda: self._start_import([source.label]))
        for button in (choose_folder, choose_files, edit, remove, rescan):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        return card

    def _add_class_dialog(self) -> None:
        source = self._source_dialog()
        if source is None:
            return
        self.sources[source.label] = source
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _edit_source_dialog(self, label: str) -> None:
        current = self.sources.get(label)
        if current is None:
            return
        updated = self._source_dialog(current)
        if updated is None:
            return
        if updated.label != label:
            self.sources.pop(label, None)
            for sample in self.samples:
                if sample.label == label:
                    sample.label = updated.label
        self.sources[updated.label] = updated
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _source_dialog(self, source: Optional[DatasetSource] = None) -> Optional[DatasetSource]:
        page = self.page
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Dataset Class")
        layout = QFormLayout(dialog)
        name_edit = QLineEdit(dialog)
        type_combo = QComboBox(dialog)
        type_combo.addItems(["Folder", "Files"])
        path_edit = QLineEdit(dialog)
        pattern_edit = QLineEdit(dialog)
        pattern_edit.setText("*")
        color_button = QPushButton("Color", dialog)
        selected_color = {"value": self._next_color(len(self.sources))}
        if source is not None:
            name_edit.setText(source.label)
            type_combo.setCurrentText("Files" if source.source_type == "files" else "Folder")
            path_edit.setText(";".join(source.paths))
            pattern_edit.setText(source.file_pattern or "*")
            selected_color["value"] = source.color
        color_button.setStyleSheet(f"background: {selected_color['value']};")

        def browse() -> None:
            if type_combo.currentText() == "Folder":
                folder = QFileDialog.getExistingDirectory(self.main_window, "Choose dataset folder")
                if folder:
                    path_edit.setText(folder)
                    if not name_edit.text().strip():
                        name_edit.setText(os.path.basename(folder.rstrip("/\\")))
            else:
                files, _ = QFileDialog.getOpenFileNames(
                    self.main_window,
                    "Choose dataset files",
                    "",
                    self._file_dialog_filter(),
                )
                if files:
                    path_edit.setText(";".join(files))
                    if not name_edit.text().strip():
                        name_edit.setText(os.path.basename(os.path.dirname(files[0])))

        def choose_color() -> None:
            color = QColorDialog.getColor(parent=dialog)
            if color.isValid():
                selected_color["value"] = color.name()
                color_button.setStyleSheet(f"background: {selected_color['value']};")

        browse_button = QPushButton("Browse", dialog)
        browse_button.clicked.connect(browse)
        color_button.clicked.connect(choose_color)
        path_row = QWidget(dialog)
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(path_edit, 1)
        path_layout.addWidget(browse_button)
        layout.addRow("Class name", name_edit)
        layout.addRow("Source type", type_combo)
        layout.addRow("Path", path_row)
        layout.addRow("File pattern", pattern_edit)
        layout.addRow("Color", color_button)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return None
        label = name_edit.text().strip()
        paths = [path for path in path_edit.text().split(";") if path.strip()]
        if not label or not paths:
            QMessageBox.warning(self.main_window, "Dataset Class", "Class name and path are required.")
            return None
        label = self._unique_label(label, existing=source.label if source else None)
        return DatasetSource(
            label=label,
            source_type="files" if type_combo.currentText() == "Files" else "folder",
            paths=paths,
            file_pattern=pattern_edit.text().strip() or "*",
            color=selected_color["value"],
        )

    def _choose_source_folder(self, label: str) -> None:
        folder = QFileDialog.getExistingDirectory(self.main_window, "Choose dataset folder")
        if not folder or label not in self.sources:
            return
        source = self.sources[label]
        source.source_type = "folder"
        source.paths = [folder]
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _choose_source_files(self, label: str) -> None:
        files, _ = QFileDialog.getOpenFileNames(self.main_window, "Choose dataset files", "", self._file_dialog_filter())
        if not files or label not in self.sources:
            return
        source = self.sources[label]
        source.source_type = "files"
        source.paths = files
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _remove_source(self, label: str) -> None:
        if QMessageBox.question(self.main_window, "Remove Class", f"Remove class '{label}' and its samples?") != QMessageBox.Yes:
            return
        self.sources.pop(label, None)
        self.samples = [sample for sample in self.samples if sample.label != label]
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _on_files_dropped(self, paths: list[str]) -> None:
        for path in paths:
            if os.path.isdir(path):
                label = self._unique_label(os.path.basename(path.rstrip("/\\")) or "Class")
                self.sources[label] = DatasetSource(label=label, source_type="folder", paths=[path], color=self._next_color(len(self.sources)))
            elif os.path.isfile(path):
                folder = os.path.dirname(path)
                base_label = os.path.basename(folder) or "Files"
                label = base_label if base_label in self.sources else self._unique_label(base_label)
                if label not in self.sources:
                    self.sources[label] = DatasetSource(label=label, source_type="files", paths=[], color=self._next_color(len(self.sources)))
                self.sources[label].paths.append(path)
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _start_import(self, labels: Optional[list[str]] = None) -> None:
        if self.current_worker is not None:
            QMessageBox.information(self.main_window, "Classification", "A Classification task is already running.")
            return
        selected_sources = [self.sources[label] for label in labels or list(self.sources.keys()) if label in self.sources]
        if not selected_sources:
            QMessageBox.warning(self.main_window, "Classification", "Add at least one dataset class first.")
            return
        self._set_state(ClassificationPageState.IMPORTING)
        self.page.taskProgressBar.setValue(0)
        worker = ImportWorker(selected_sources, self.data_service)
        self.current_worker = worker
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(lambda payload: self._on_import_finished(payload, labels))
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_import_finished(self, payload, labels: Optional[list[str]]) -> None:
        self.current_worker = None
        if not isinstance(payload, dict):
            self._on_worker_error("Import returned an invalid payload.")
            return
        new_samples = payload.get("samples", [])
        if labels:
            label_set = set(labels)
            self.samples = [sample for sample in self.samples if sample.label not in label_set] + list(new_samples)
        else:
            self.samples = list(new_samples)
        self.summary = payload.get("summary") or self.data_service.validate_dataset(self.samples)
        self._mark_results_outdated()
        self._refresh_everything()
        state = ClassificationPageState.READY if self.summary.status == "Ready" else ClassificationPageState.SCANNED
        self._set_state(state)
        self._persist_parameters()
        self.log(f"[Import] Loaded {self.summary.loaded_samples}/{self.summary.total_samples} files.")

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
                button.clicked.connect(lambda _checked=False, algorithm_id=config.algorithm_id: self._edit_algorithm_parameters(algorithm_id))
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
        config = next((item for item in self.algorithm_configs if item.algorithm_id == algorithm_id), None)
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
            QMessageBox.information(self.main_window, "Classification", "A Classification task is already running.")
            return
        self.summary = self.data_service.validate_dataset(self.samples)
        if any(issue.severity == "error" for issue in self.summary.issues):
            QMessageBox.warning(self.main_window, "Classification", self._quality_message())
            self._refresh_everything()
            return
        algorithms = [config for config in self._collect_algorithm_configs() if config.enabled]
        if not algorithms:
            QMessageBox.warning(self.main_window, "Classification", "Select at least one algorithm.")
            return
        worker = TrainingWorker(
            self.samples,
            self._collect_preprocessing_config(),
            algorithms,
            self._collect_validation_config(),
            self._collect_projection_config(),
            self._ranking_metric(),
            self.data_service,
            self.training_service,
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
        self.classification_completed.emit({"best_model": best, "results": len(self.experiment_result.results)})
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
            if probabilities is not None and index < probabilities.shape[0] and np.any(np.isfinite(probabilities[index])):
                sample.confidence = float(np.nanmax(probabilities[index]))

    def _update_results_views(self) -> None:
        page = self.page
        if page is None:
            return
        result = self.experiment_result
        page.resultsOutdatedLabel.setText("Outdated: settings changed after the last run." if self._results_outdated else "")
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
        if self.active_result is None or self.active_result.fitted_pipeline is None or self.experiment_result is None:
            QMessageBox.warning(self.main_window, "Save Model", "No active trained model is available.")
            return
        path, _ = QFileDialog.getSaveFileName(self.main_window, "Save Active Model", "", "Joblib (*.joblib);;Pickle (*.pkl)")
        if not path:
            return
        package = self._build_saved_model_package(self.active_result)
        try:
            import joblib

            joblib.dump(package, path)
            self.active_model_package = package
            self.log(f"[Model] Saved active model to {path}")
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Save Model", str(exc))

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.main_window, "Load Model", "", "Joblib (*.joblib);;Pickle (*.pkl);;All files (*.*)")
        if not path:
            return
        try:
            import joblib

            package = joblib.load(path)
            if isinstance(package, SavedModelPackage):
                self.active_model_package = package
                self.log(f"[Model] Loaded model: {package.display_name}")
                return
            if isinstance(package, dict) and package.get("dr_type") == "t-SNE":
                QMessageBox.warning(
                    self.main_window,
                    "Load Model",
                    "This legacy model uses t-SNE as a classification feature and cannot transform new samples.",
                )
                return
            QMessageBox.warning(self.main_window, "Load Model", "Unsupported legacy model package.")
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Load Model", str(exc))

    def _predict_new_data_menu(self) -> None:
        menu = QMenu(self.page.predictNewDataButton)
        files_action = menu.addAction("Choose Files")
        folder_action = menu.addAction("Choose Folder")
        action = menu.exec_(self.page.predictNewDataButton.mapToGlobal(self.page.predictNewDataButton.rect().bottomLeft()))
        if action == files_action:
            files, _ = QFileDialog.getOpenFileNames(self.main_window, "Choose unknown data", "", self._file_dialog_filter())
            if files:
                self._start_prediction(files)
        elif action == folder_action:
            folder = QFileDialog.getExistingDirectory(self.main_window, "Choose unknown data folder")
            if folder:
                self._start_prediction([folder])

    def _start_prediction(self, paths: list[str]) -> None:
        if self.current_worker is not None:
            QMessageBox.information(self.main_window, "Classification", "A Classification task is already running.")
            return
        package = self.active_model_package
        if package is None:
            if self.active_result is not None:
                package = self._build_saved_model_package(self.active_result)
                self.active_model_package = package
            else:
                QMessageBox.warning(self.main_window, "Prediction", "Train, load, or save an active model first.")
                return
        worker = PredictionWorker(paths, package, self.data_service)
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
        self._set_state(ClassificationPageState.RESULTS_AVAILABLE if self.experiment_result else ClassificationPageState.READY)
        self.log(f"[Prediction] Predicted {len(self.prediction_results)} file(s).")

    def _start_embedding(self) -> None:
        if self.current_worker is not None:
            QMessageBox.information(self.main_window, "Classification", "A Classification task is already running.")
            return
        if not self.samples:
            QMessageBox.warning(self.main_window, "Embedding", "Import data before running embedding visualization.")
            return
        worker = EmbeddingWorker(
            self.samples,
            self._collect_preprocessing_config(),
            self.page.embeddingMethodCombo.currentText(),
            self.data_service,
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

    def _build_saved_model_package(self, result: ModelEvaluationResult) -> SavedModelPackage:
        import sklearn

        algorithm = next((config for config in self.algorithm_configs if config.algorithm_id == result.algorithm_id), None)
        data_type = self.feature_matrix.data_type if self.feature_matrix is not None else "unknown"
        input_shape = self.experiment_result.input_shape if self.experiment_result else None
        return SavedModelPackage(
            pipeline=result.fitted_pipeline,
            algorithm_id=result.algorithm_id,
            display_name=result.display_name,
            class_names=list(result.labels),
            data_type=data_type,
            input_shape=input_shape,
            preprocessing_config=self.experiment_result.preprocessing_config,
            projection_config=self.experiment_result.projection_config,
            algorithm_parameters=dict(algorithm.parameters if algorithm else {}),
            sklearn_version=sklearn.__version__,
            numpy_version=np.__version__,
            software_version="gisaxs_gui",
            training_date=datetime.now().isoformat(timespec="seconds"),
            validation_config=self.experiment_result.validation_config,
            evaluation_metrics=dict(result.metrics_mean),
        )

    def _update_dataset_table(self) -> None:
        page = self.page
        if page is None:
            return
        search = page.datasetSearchEdit.text().strip().lower()
        class_filter = page.classFilterCombo.currentText()
        qc_filter = page.qcFilterCombo.currentText()
        rows = []
        for sample in self.samples:
            if search and search not in sample.file_name.lower() and search not in sample.file_path.lower():
                continue
            if class_filter != "All classes" and sample.label != class_filter:
                continue
            if qc_filter != "All QC" and sample.qc_status.lower() != qc_filter.lower():
                continue
            rows.append(sample)
        table = page.datasetTable
        self._table_updating = True
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for row, sample in enumerate(rows):
            include_item = QTableWidgetItem("")
            include_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            include_item.setCheckState(Qt.Checked if sample.included else Qt.Unchecked)
            include_item.setData(Qt.UserRole, sample.sample_id)
            table.setItem(row, 0, include_item)
            values = [
                sample.label,
                sample.file_name,
                sample.data_type,
                self._shape_text(sample.raw_shape),
                sample.load_status,
                sample.qc_status,
                sample.predicted_label or "-",
                self._optional_float(sample.confidence),
            ]
            for offset, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, sample.sample_id)
                if offset != 1:
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, offset, item)
        table.setSortingEnabled(True)
        self._table_updating = False
        self._update_class_filter()
        self._update_run_summary()

    def _on_dataset_item_changed(self, item: QTableWidgetItem) -> None:
        if self._table_updating or item.column() != 0:
            return
        sample = self._sample_by_id(item.data(Qt.UserRole))
        if sample is None:
            return
        sample.included = item.checkState() == Qt.Checked
        self.summary = self.data_service.validate_dataset(self.samples)
        self._mark_results_outdated()
        self._refresh_everything()

    def _selected_sample_ids(self) -> list[str]:
        page = self.page
        ids: list[str] = []
        for index in page.datasetTable.selectionModel().selectedRows():
            item = page.datasetTable.item(index.row(), 0)
            if item is not None:
                ids.append(str(item.data(Qt.UserRole)))
        return ids

    def _preview_current_table_sample(self) -> None:
        page = self.page
        row = page.datasetTable.currentRow()
        if row < 0:
            return
        item = page.datasetTable.item(row, 0)
        if item is None:
            return
        sample = self._sample_by_id(item.data(Qt.UserRole))
        if sample is not None:
            self.current_preview_sample_id = sample.sample_id
            self._render_sample_preview(sample)

    def _render_current_preview(self) -> None:
        sample = self._sample_by_id(self.current_preview_sample_id)
        if sample is not None:
            self._render_sample_preview(sample)

    def _render_sample_preview(self, sample: ClassificationSample) -> None:
        page = self.page
        if page is None:
            return
        data = sample.raw_data
        page.sampleFileLabel.setText(sample.file_name)
        page.sampleShapeLabel.setText(self._shape_text(sample.raw_shape))
        loaded_samples = [item for item in self.samples if item.load_status == "loaded"]
        try:
            index = loaded_samples.index(sample) + 1
        except ValueError:
            index = 0
        page.sampleIndexLabel.setText(f"{index} / {len(loaded_samples)}")
        if data is None:
            self._set_graphics_text(page.previewGraphicsView, "Sample is not loaded.")
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
            if sample.data_type == "1D":
                arr = np.asarray(data)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    x, y = arr[:, 0], arr[:, 1]
                else:
                    y = arr.ravel()
                    x = np.arange(len(y))
                if page.previewLogScaleCheckBox.isChecked():
                    ax.semilogy(x, np.maximum(y, np.finfo(float).tiny), lw=1.0)
                else:
                    ax.plot(x, y, lw=1.0)
                ax.set_xlabel("q / index")
                ax.set_ylabel("Intensity")
                ax.grid(True, alpha=0.25)
            else:
                img = np.asarray(data, dtype=float)
                if page.previewLogScaleCheckBox.isChecked():
                    img = np.log1p(np.maximum(img, 0))
                vmin = vmax = None
                if page.previewAutoScaleCheckBox.isChecked():
                    vmin = float(np.nanpercentile(img, 0.5))
                    vmax = float(np.nanpercentile(img, 99.5))
                else:
                    vmin = float(page.previewVminSpinBox.value())
                    vmax = float(page.previewVmaxSpinBox.value())
                ax.imshow(img, cmap=page.previewColormapCombo.currentText(), vmin=vmin, vmax=vmax, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            self._set_graphics_pixmap(page.previewGraphicsView, self._figure_to_pixmap(fig))
            plt.close(fig)
        except Exception as exc:
            self._set_graphics_text(page.previewGraphicsView, str(exc))

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
                ax.bar(x + (offset - 1) * width, values, width, label=metric.replace("_", " ").title())
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
                self._optional_float(item.confidence if item.confidence is not None else item.decision_score),
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
                self._optional_float(result.confidence if result.confidence is not None else result.decision_score),
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
        page.summaryBalanceLabel.setText(", ".join(f"{k}:{v}" for k, v in summary.valid_class_counts.items()) or "-")
        page.qualityStatusLabel.setText(summary.status)
        page.qualityListWidget.clear()
        if not summary.issues:
            page.qualityListWidget.addItem("Ready: dataset checks passed.")
        else:
            for issue in summary.issues[:20]:
                fix = f" Fix: {issue.fix}" if issue.fix else ""
                page.qualityListWidget.addItem(f"{issue.severity.title()}: {issue.message}{fix}")
        data_type = summary.data_types[0] if len(summary.data_types) == 1 else "auto"
        page.dataTypeBadgeLabel.setText(data_type)

    def _update_input_summary(self) -> None:
        page = self.page
        if page is None:
            return
        try:
            matrix = self.data_service.build_feature_matrix(self.samples, self._collect_preprocessing_config(), require_labels=True)
            memory = self.data_service.estimate_feature_memory(matrix)
            page.inputSummaryLabel.setText(
                f"Samples: {matrix.X.shape[0]} | Features: {matrix.X.shape[1]} | Input shape: {matrix.input_shape} | Memory: {memory}"
            )
        except Exception:
            page.inputSummaryLabel.setText("Samples: 0 | Features: 0 | Input shape: - | Memory: -")

    def _update_run_summary(self) -> None:
        page = self.page
        if page is None:
            return
        selected = len([config for config in self._collect_algorithm_configs() if config.enabled]) if page.algorithmTable.rowCount() else 0
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
        elif state in {ClassificationPageState.IMPORTING, ClassificationPageState.TRAINING, ClassificationPageState.PREDICTING}:
            page.set_step("Algorithms" if state == ClassificationPageState.TRAINING else "Dataset")
        elif state == ClassificationPageState.RESULTS_AVAILABLE:
            page.set_step("Results")
        busy = state in {ClassificationPageState.IMPORTING, ClassificationPageState.TRAINING, ClassificationPageState.PREDICTING}
        page.cancelTaskButton.setEnabled(busy)
        page.runComparisonButton.setEnabled(not busy and self.summary.valid_samples >= 2)
        page.scanImportButton.setEnabled(not busy)
        page.addClassButton.setEnabled(not busy)
        page.predictNewDataButton.setEnabled(not busy)
        self.progress_updated.emit(0 if not busy else page.taskProgressBar.value())
        self._update_run_summary()

    def _on_worker_progress(self, percent: int, message: str) -> None:
        if self.page is not None:
            self.page.taskProgressBar.setValue(max(0, min(100, int(percent))))
        self.progress_updated.emit(max(0, min(100, int(percent))))
        self.status_updated.emit(message)

    def _on_worker_error(self, message: str) -> None:
        self.current_worker = None
        self._set_state(ClassificationPageState.ERROR)
        self.log(f"[Error] {message}")
        QMessageBox.warning(self.main_window, "Classification Error", str(message).splitlines()[-1] if message else "Unknown error")

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
            enabled=page.useProjectionCheckBox.isChecked() and page.projectionMethodCombo.currentText() != "None",
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
            global_params.set_parameter("classification", "import_cache", params["import_cache"])
            global_params.set_parameter("classification", "sources", params["sources"])
            global_params.set_parameter("classification", "preprocessing", params["preprocessing"])
            global_params.set_parameter("classification", "validation", params["validation"])
            global_params.set_parameter("classification", "projection", params["projection"])
            global_params.set_parameter("classification", "algorithms", params["algorithms"])
            global_params.set_parameter("classification", "ranking_metric", params["ranking_metric"])
            global_params.save_user_parameters()
            self.parameters_changed.emit(params)
        except Exception as exc:
            self.log(f"[Session] Parameter persistence failed: {exc}")

    def _save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self.main_window, "Save Classification Session", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.get_parameters(), handle, indent=2, ensure_ascii=False)
            self.log(f"[Session] Saved session to {path}")
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Save Session", str(exc))

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.main_window, "Load Classification Session", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                self.set_parameters(json.load(handle))
            self.log(f"[Session] Loaded session from {path}")
        except Exception as exc:
            QMessageBox.warning(self.main_window, "Load Session", str(exc))

    def _export_results_csv(self) -> None:
        if self.experiment_result is None:
            QMessageBox.information(self.main_window, "Export Results", "No results are available.")
            return
        path, _ = QFileDialog.getSaveFileName(self.main_window, "Export Results", "", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "algorithm", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "status", "error"])
            for rank, result in enumerate(self.experiment_result.results, start=1):
                writer.writerow(
                    [
                        rank,
                        result.display_name,
                        result.metrics_mean.get("accuracy", ""),
                        result.metrics_mean.get("balanced_accuracy", ""),
                        result.metrics_mean.get("macro_f1", ""),
                        result.metrics_mean.get("weighted_f1", ""),
                        result.status,
                        result.error_message or "",
                    ]
                )
        self.log(f"[Export] Results CSV written to {path}")

    def _export_predictions_csv(self) -> None:
        if not self.prediction_results:
            QMessageBox.information(self.main_window, "Export Predictions", "No prediction results are available.")
            return
        path, _ = QFileDialog.getSaveFileName(self.main_window, "Export Predictions", "", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "predicted_label", "confidence", "decision_score", "status", "message"])
            for result in self.prediction_results:
                writer.writerow(
                    [
                        result.file_path,
                        result.predicted_label or "",
                        result.confidence if result.confidence is not None else "",
                        result.decision_score if result.decision_score is not None else "",
                        result.status,
                        result.message,
                    ]
                )
        self.log(f"[Export] Prediction CSV written to {path}")

    def _export_selected_file_list(self) -> None:
        selected = [self._sample_by_id(sample_id) for sample_id in self._selected_sample_ids()]
        selected = [sample for sample in selected if sample is not None]
        if not selected:
            return
        path, _ = QFileDialog.getSaveFileName(self.main_window, "Export Selected Files", "", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["included", "class", "file", "path", "load_status", "qc_status"])
            for sample in selected:
                writer.writerow([sample.included, sample.label, sample.file_name, sample.file_path, sample.load_status, sample.qc_status])
        self.log(f"[Export] Selected file list written to {path}")

    def _set_selected_included(self, included: bool) -> None:
        selected_ids = set(self._selected_sample_ids())
        for sample in self.samples:
            if sample.sample_id in selected_ids:
                sample.included = included
        self.summary = self.data_service.validate_dataset(self.samples)
        self._mark_results_outdated()
        self._refresh_everything()

    def _remove_selected_samples(self) -> None:
        selected_ids = set(self._selected_sample_ids())
        if not selected_ids:
            return
        self.samples = [sample for sample in self.samples if sample.sample_id not in selected_ids]
        self._mark_results_outdated()
        self._refresh_everything()

    def _open_selected_location(self) -> None:
        sample = self._sample_by_id(self.current_preview_sample_id)
        if sample is None:
            ids = self._selected_sample_ids()
            sample = self._sample_by_id(ids[0]) if ids else None
        if sample is None:
            return
        folder = os.path.dirname(sample.file_path)
        if os.path.isdir(folder):
            os.startfile(folder)

    def _copy_selected_paths(self) -> None:
        selected = [self._sample_by_id(sample_id) for sample_id in self._selected_sample_ids()]
        paths = [sample.file_path for sample in selected if sample is not None]
        if paths:
            QApplication.clipboard().setText("\n".join(paths))
            self.log(f"[Dataset] Copied {len(paths)} path(s).")

    def _move_preview(self, delta: int) -> None:
        loaded = [sample for sample in self.samples if sample.load_status == "loaded"]
        if not loaded:
            return
        current = self._sample_by_id(self.current_preview_sample_id)
        try:
            index = loaded.index(current)
        except ValueError:
            index = 0
        index = max(0, min(len(loaded) - 1, index + delta))
        self.current_preview_sample_id = loaded[index].sample_id
        self._render_sample_preview(loaded[index])

    def _preview_selected_misclassification(self) -> None:
        row = self.page.misclassifiedTable.currentRow()
        if row < 0:
            return
        item = self.page.misclassifiedTable.item(row, 0)
        sample = self._sample_by_id(item.data(Qt.UserRole) if item else None)
        if sample is not None:
            self.current_preview_sample_id = sample.sample_id
            self._render_sample_preview(sample)

    def _sync_legacy_class_list(self) -> None:
        page = self.page
        if page is None:
            return
        page.legacyClassListWidget.clear()
        for label in self.sources:
            page.legacyClassListWidget.addItem(label)

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
        recommended = {"logistic_regression", "linear_svm", "rbf_svm", "knn", "random_forest", "lda"}
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
        self.algorithm_configs = self.training_service.default_algorithm_configs()
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
        return "\n".join(f"{issue.severity.title()}: {issue.message} {issue.fix}".strip() for issue in self.summary.issues[:8])

    def _set_graphics_text(self, view, text: str) -> None:
        scene = QGraphicsScene(view)
        scene.addText(text)
        view.setScene(scene)

    def _set_graphics_pixmap(self, view, pixmap: QPixmap) -> None:
        scene = QGraphicsScene(view)
        scene.addPixmap(pixmap)
        view.setScene(scene)
        view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def _figure_to_pixmap(self, fig) -> QPixmap:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        image = QImage.fromData(buffer.read(), "PNG")
        return QPixmap.fromImage(image)

    def _fit_preview(self) -> None:
        view = self.page.previewGraphicsView
        if view.scene() is not None:
            view.fitInView(view.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def _file_dialog_filter(self) -> str:
        return "Data files (*.dat *.txt *.csv *.xy *.chi *.tif *.tiff *.png *.jpg *.jpeg *.bmp *.cbf *.edf *.h5 *.hdf5 *.npy);;All files (*.*)"

    def _next_color(self, index: int) -> str:
        colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ca8a04", "#0891b2", "#db2777", "#4b5563"]
        return colors[index % len(colors)]

    def _unique_label(self, label: str, existing: Optional[str] = None) -> str:
        if label == existing:
            return label
        base = label
        counter = 2
        while label in self.sources:
            label = f"{base} {counter}"
            counter += 1
        return label

    def _short_paths(self, paths: list[str]) -> str:
        if not paths:
            return "-"
        if len(paths) == 1:
            return paths[0]
        return f"{paths[0]} (+{len(paths) - 1})"

    def _sample_by_id(self, sample_id) -> Optional[ClassificationSample]:
        if sample_id is None:
            return None
        for sample in self.samples:
            if sample.sample_id == str(sample_id):
                return sample
        return None

    def _shape_text(self, shape) -> str:
        if not shape:
            return "-"
        return "x".join(str(value) for value in shape)

    def _metric_text(self, result: Optional[ModelEvaluationResult], metric: str) -> str:
        if result is None or result.status != "ok":
            return "-"
        return f"{float(result.metrics_mean.get(metric, 0.0)):.3f}"

    def _number_text(self, value) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "-"

    def _optional_float(self, value) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "-"

    def _parse_parameter(self, text: str, original):
        stripped = text.strip()
        if isinstance(original, bool):
            return stripped.lower() in {"1", "true", "yes", "on"}
        if isinstance(original, int) and not isinstance(original, bool):
            try:
                return int(stripped)
            except ValueError:
                return original
        if isinstance(original, float):
            try:
                return float(stripped)
            except ValueError:
                return original
        if stripped in {"None", "none", ""}:
            return None if original is None else stripped
        return stripped

    # Compatibility entry points retained for older callers/tests.
    def _on_import_clicked(self):
        self._start_import()

    def _on_clf_start_clicked(self):
        self._start_training()

    def _on_clf_save_clicked(self):
        self._save_active_model()

    def _on_clf_load_clicked(self):
        self._load_model()

    def _on_import_classify_clicked(self):
        self._predict_new_data_menu()
