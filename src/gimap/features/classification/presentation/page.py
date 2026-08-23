"""Feature-owned Classification page with Python-owned view modules."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHeaderView,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from src.gimap.app.presentation import apply_design_system, install_safe_wheel_behavior
from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)

from .components import ClassificationEmptyState
from .panel_wiring import create_apply_panel, create_exploration_panel
from .views import (
    ClassificationDatasetPanelView,
    ClassificationExperimentPanelView,
    ClassificationInspectionPanelView,
    ClassificationPageView,
    ClassificationPreprocessingPanelView,
    ClassificationResultsPanelView,
)


STYLE_PATH = Path(__file__).resolve().parent / "styles" / "classification_page.qss"


class ClassificationPage(QWidget, ClassificationPageView):
    """Five-stage workbench for unlabeled exploration and supervised classification."""

    filesDropped = pyqtSignal(list)
    stepChanged = pyqtSignal(str)

    _STEP_NAMES = ("Data", "Prepare", "Explore", "Train", "Apply")
    _STEP_ALIASES = {
        "Dataset": "Data",
        "Preprocessing": "Prepare",
        "Algorithms": "Train",
        "Results": "Train",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self._responsive_mode = ""
        self._bind_form()
        self._load_stylesheet()
        install_safe_wheel_behavior(self)
        self.apply_responsive_mode()
        QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def clear_dataset_cards(self) -> None:
        while self.datasetCardsLayout.count():
            item = self.datasetCardsLayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.datasetCardsLayout.addStretch(1)

    def add_dataset_card(self, card: QWidget) -> None:
        stretch_index = max(0, self.datasetCardsLayout.count() - 1)
        self.datasetCardsLayout.insertWidget(stretch_index, card)

    def set_step(self, step: str) -> None:
        step = self._STEP_ALIASES.get(step, step)
        if step not in self._STEP_NAMES:
            return
        index = self._STEP_NAMES.index(step)
        if self.workflowStack.currentIndex() != index:
            self.workflowStack.setCurrentIndex(index)
        for name, button in self._step_buttons.items():
            button.setChecked(name == step)
        self.classificationWorkflowHeader.set_selected_step(step)
        self._current_step = step
        self.stepChanged.emit(step)

    def set_workflow_step_state(
        self, step: str, state: str, message: str = ""
    ) -> None:
        """Render verified progress without changing the user's current view."""
        if step in self._STEP_NAMES:
            self.classificationWorkflowHeader.set_step_state(step, state, message)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.apply_responsive_mode()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Pages inside the main stacked widget can receive their first useful
        # geometry only when selected, after construction-time sizing ran.
        self._responsive_mode = ""
        self.apply_responsive_mode()

    def apply_responsive_mode(self) -> None:
        width = max(1, self.width())
        height = max(1, self.height())
        if width >= 1500 and height >= 850:
            mode = "wide"
        elif width >= 1360 and height >= 760:
            mode = "medium"
        else:
            mode = "compact"
        if mode == self._responsive_mode:
            return
        self._responsive_mode = mode
        if mode == "compact":
            self.datasetInspectionSplitter.setOrientation(Qt.Vertical)
            self.explorationSplitter.setOrientation(Qt.Vertical)
            self.explorationSelectionPanel.setMaximumWidth(16777215)
            self.datasetPanel.setMinimumWidth(0)
            self.inspectionPanel.setMinimumWidth(0)
            self.datasetStepContent.setMinimumHeight(960)
            self.algorithmsStepContent.setMinimumHeight(980)
        else:
            self.datasetInspectionSplitter.setOrientation(Qt.Horizontal)
            self.explorationSplitter.setOrientation(Qt.Horizontal)
            self.explorationSelectionPanel.setMaximumWidth(460)
            self.datasetPanel.setMinimumWidth(320)
            self.inspectionPanel.setMinimumWidth(500 if mode == "wide" else 420)
            self.datasetStepContent.setMinimumHeight(560)
            self.algorithmsStepContent.setMinimumHeight(620)
        QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def _apply_initial_splitter_sizes(self) -> None:
        if not hasattr(self, "datasetInspectionSplitter"):
            return
        if self._responsive_mode == "compact":
            self.datasetInspectionSplitter.setSizes([430, 520])
            self.explorationSplitter.setSizes([560, 400])
        elif self._responsive_mode == "wide":
            self.datasetInspectionSplitter.setSizes([430, 900])
            self.explorationSplitter.setSizes([980, 420])
        else:
            self.datasetInspectionSplitter.setSizes([380, 680])
            self.explorationSplitter.setSizes([760, 360])
        self.overviewSplitter.setSizes([390, 260])

    def _bind_form(self) -> None:
        """Install legacy panel seams into the Designer-owned page shell."""
        self.classification_input_section = self.classificationInputSection
        self.classification_preview_panel = self.classificationPreviewPanel
        self.classification_configure_section = self.classificationConfigureSection
        self.classification_algorithm_section = self.classificationAlgorithmSection
        self.classification_results_section = self.classificationResultsSection
        self.classification_apply_section = self.classificationApplySection
        self.classification_export_section = self.classificationExportSection
        self.classification_log_section = self.classificationLogSection
        self.preprocessingStepContent = self.classificationConfigureSection
        self.algorithmsStepContent = self.algorithmsStepContent
        self.datasetPanel = self.classificationInputSection
        self.inspectionPanel = self.classificationPreviewPanel

        for section, title, description, content, layout in (
            (
                self.classification_input_section,
                self.classificationInputTitle,
                self.classificationInputDescription,
                self.classificationInputContent,
                self.classificationInputContentLayout,
            ),
            (
                self.classification_preview_panel,
                self.classificationPreviewTitle,
                self.classificationPreviewDescription,
                self.classificationPreviewContent,
                self.classificationPreviewContentLayout,
            ),
            (
                self.classification_configure_section,
                self.classificationConfigureTitle,
                self.classificationConfigureDescription,
                self.classificationConfigureContent,
                self.classificationConfigureContentLayout,
            ),
            (
                self.classification_algorithm_section,
                self.classificationAlgorithmTitle,
                self.classificationAlgorithmDescription,
                self.classificationAlgorithmContent,
                self.classificationAlgorithmContentLayout,
            ),
            (
                self.classification_results_section,
                self.classificationResultsTitle,
                self.classificationResultsDescription,
                self.classificationResultsContent,
                self.classificationResultsContentLayout,
            ),
            (
                self.classification_apply_section,
                self.classificationApplyTitle,
                self.classificationApplyDescription,
                self.classificationApplyContent,
                self.classificationApplyContentLayout,
            ),
            (
                self.classification_export_section,
                self.classificationExportTitle,
                self.classificationExportDescription,
                self.classificationExportContent,
                self.classificationExportContentLayout,
            ),
        ):
            bind_parameter_section(section, title, description, content, layout)
            apply_design_system(section)
        bind_advanced_section(
            self.classification_log_section,
            self.logToggleButton,
            self.classificationLogDescription,
            self.classificationLogContent,
            self.classificationLogContentLayout,
        )
        apply_design_system(self.classification_log_section)

        self.titleLabel.setObjectName("classificationTitle")
        self.subtitleLabel.setObjectName("classificationSubtitle")
        self.newSessionButton.setObjectName("NewSessionButton")
        self.loadSessionButton.setObjectName("LoadSessionButton")
        self.saveSessionButton.setObjectName("SaveSessionButton")
        self.workflowStack.setObjectName("classificationWorkflowStack")
        self.logTextBrowser.setObjectName("classificationPageTextBrowser")

        self.classificationInputContentLayout.addWidget(
            self._create_dataset_panel()
        )
        self.classificationPreviewContentLayout.addWidget(
            self._create_inspection_panel()
        )
        preprocessing_panel = self._create_preprocessing_panel()
        preprocessing_panel.setMaximumWidth(1100)
        preprocessing_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.classificationConfigureContentLayout.addWidget(preprocessing_panel)
        self.classificationAlgorithmContentLayout.addWidget(
            self._create_experiment_panel()
        )
        self.classificationExploreContentLayout.addWidget(
            create_exploration_panel(self)
        )
        results_panel = self._create_results_panel()
        self.classificationResultsContentLayout.addWidget(results_panel)
        apply_panel = create_apply_panel(self)
        self.classificationApplyContentLayout.addWidget(apply_panel)
        for button in (
            self.saveActiveModelButton,
            self.exportResultsButton,
            self.exportPredictionsButton,
        ):
            self._detach_widget(button)
            self.classificationExportContentLayout.addWidget(button)
        self.classificationExportContentLayout.addStretch(1)

        self._step_buttons = {
            "Data": self.datasetStepButton,
            "Prepare": self.preprocessingStepButton,
            "Explore": self.algorithmsStepButton,
            "Train": self.resultsStepButton,
            "Apply": self.applyStepButton,
        }
        self.classificationWorkflowHeader.step_requested.connect(self.set_step)

        self.datasetInspectionSplitter.setStretchFactor(0, 0)
        self.datasetInspectionSplitter.setStretchFactor(1, 1)
        self.logToggleButton = self.classification_log_section.toggle_button
        self.preview_empty_state = ClassificationEmptyState(self.previewGraphicsView)
        self.preprocessing_continue_button.clicked.connect(
            lambda _checked=False: self.set_step("Explore")
        )
        self.set_workflow_step_state("Data", "available", "Add files or folders")
        self.set_workflow_step_state("Prepare", "blocked", "Import data first")
        self.set_workflow_step_state("Explore", "blocked", "Prepare a compatible group")
        self.set_workflow_step_state("Train", "blocked", "Accept at least two classes")
        self.set_workflow_step_state("Apply", "blocked", "Train or load a model")
        self.set_step("Data")

    def _create_dataset_panel(self) -> QWidget:
        panel = QFrame(self)
        ui = ClassificationDatasetPanelView()
        ui.setupUi(panel)
        self._dataset_panel_ui = ui
        for name in (
            "addClassButton",
            "addDataButton",
            "scanImportButton",
            "datasetCardsScrollArea",
            "datasetCardsContainer",
            "datasetCardsLayout",
            "datasetSummaryFrame",
            "summaryClassesLabel",
            "summaryTotalLabel",
            "summaryValidLabel",
            "summaryInvalidLabel",
            "summaryBalanceLabel",
            "datasetSearchEdit",
            "classFilterCombo",
            "qcFilterCombo",
            "datasetTable",
            "excludeSelectedButton",
            "includeSelectedButton",
            "removeSelectedSamplesButton",
            "openSelectedLocationButton",
            "copySelectedPathsButton",
            "exportSelectedFilesButton",
        ):
            setattr(self, name, getattr(ui, name))
        self.datasetTable.verticalHeader().setVisible(False)
        self.datasetTable.horizontalHeader().setStretchLastSection(False)
        self.datasetTable.horizontalHeader().setSectionResizeMode(
            2,
            QHeaderView.Stretch,
        )
        self.addDataButton.setProperty("classificationPrimaryAction", True)
        self.addDataButton.setProperty("gimapPrimaryAction", True)
        return panel

    def _create_inspection_panel(self) -> QWidget:
        panel = QFrame(self)
        ui = ClassificationInspectionPanelView()
        ui.setupUi(panel)
        self._inspection_panel_ui = ui
        for name in (
            "prevSampleButton",
            "nextSampleButton",
            "sampleIndexLabel",
            "sampleFileLabel",
            "sampleShapeLabel",
            "previewGraphicsView",
            "previewLogScaleCheckBox",
            "previewColormapCombo",
            "previewAutoScaleCheckBox",
            "previewVminSpinBox",
            "previewVmaxSpinBox",
            "fitPreviewButton",
            "openFileLocationButton",
            "qualityFrame",
            "qualityStatusLabel",
            "qualityListWidget",
        ):
            setattr(self, name, getattr(ui, name))
        return panel

    def _create_preprocessing_panel(self) -> QWidget:
        panel = QFrame(self)
        ui = ClassificationPreprocessingPanelView()
        ui.setupUi(panel)
        self._preprocessing_panel_ui = ui
        for name in (
            "dataTypeBadgeLabel",
            "oneDPreprocessingCombo",
            "twoDPreprocessingCombo",
            "normalizeCombo",
            "preprocessingLogCheckBox",
            "smoothingSpinBox",
            "resizeRowsSpinBox",
            "resizeColsSpinBox",
            "inputSummaryLabel",
        ):
            setattr(self, name, getattr(ui, name))
        self.classification_preprocessing_advanced = (
            ui.classificationPreprocessingAdvancedSection
        )
        bind_advanced_section(
            self.classification_preprocessing_advanced,
            ui.preprocessingAdvancedToggle,
            ui.preprocessingAdvancedDescription,
            ui.preprocessingAdvancedContent,
            ui.preprocessingAdvancedContentLayout,
        )
        apply_design_system(self.classification_preprocessing_advanced)
        self.preprocessing_continue_button = QPushButton("Continue to explore", panel)
        self.preprocessing_continue_button.setObjectName("preprocessingContinueButton")
        self.preprocessing_continue_button.setProperty("classificationPrimaryAction", True)
        self.preprocessing_continue_button.setProperty("gimapPrimaryAction", True)
        ui.preprocessingPanelLayout.insertWidget(
            max(0, ui.preprocessingPanelLayout.count() - 1),
            self.preprocessing_continue_button,
            0,
            Qt.AlignRight,
        )
        return panel

    def _create_experiment_panel(self) -> QWidget:
        panel = QFrame(self)
        ui = ClassificationExperimentPanelView()
        ui.setupUi(panel)
        self._experiment_panel_ui = ui
        for name in (
            "algorithmConfigSplitter",
            "validationMethodCombo",
            "testSizeSpinBox",
            "foldsSpinBox",
            "repeatsSpinBox",
            "randomSeedSpinBox",
            "shuffleCheckBox",
            "rankingMetricCombo",
            "validationWarningLabel",
            "useProjectionCheckBox",
            "projectionMethodCombo",
            "projectionComponentsSpinBox",
            "pcaVarianceSpinBox",
            "umapNeighborsSpinBox",
            "umapMinDistSpinBox",
            "tsneNoteLabel",
            "selectRecommendedButton",
            "selectAllAlgorithmsButton",
            "clearAlgorithmsButton",
            "resetAlgorithmDefaultsButton",
            "algorithmTable",
            "runComparisonButton",
            "cancelTaskButton",
            "classification_job_status",
        ):
            setattr(self, name, getattr(ui, name))

        self.classification_algorithm_advanced = (
            ui.classificationAlgorithmAdvancedSection
        )
        bind_advanced_section(
            self.classification_algorithm_advanced,
            ui.algorithmAdvancedToggle,
            ui.algorithmAdvancedDescription,
            ui.algorithmAdvancedContent,
            ui.algorithmAdvancedContentLayout,
        )
        apply_design_system(self.classification_algorithm_advanced)
        self.classification_algorithm_advanced.setParent(panel)
        ui.experimentPanelLayout.insertWidget(1, self.classification_algorithm_advanced)
        self.algorithmConfigSplitter.setHandleWidth(0)
        self.algorithmConfigSplitter.setSizes([1200])
        self.classification_run_section = ui.classificationRunSection
        bind_parameter_section(
            self.classification_run_section,
            ui.classificationRunTitle,
            ui.classificationRunDescription,
            ui.classificationRunContent,
            ui.classificationRunContentLayout,
        )
        apply_design_system(self.classification_run_section)
        ui.experimentPanelLayout.removeWidget(self.classification_run_section)
        ui.experimentPanelLayout.insertWidget(1, self.classification_run_section)

        self.algorithmTable.setObjectName("algorithmList")
        self.algorithmTable.verticalHeader().setVisible(False)
        self.algorithmTable.horizontalHeader().setSectionResizeMode(
            2,
            QHeaderView.Stretch,
        )
        self.algorithmConfigSplitter.setStretchFactor(0, 0)
        self.algorithmConfigSplitter.setStretchFactor(1, 1)
        self.classification_job_status.set_actions_visible(
            pause=False,
            cancel=False,
            details=False,
        )
        self.runStatusLabel = self.classification_job_status.message_label
        self.runStatusLabel.setText(
            "Selected algorithms: 0 | Valid samples: 0 | Estimated runs: 0 | EMPTY"
        )
        self.taskProgressBar = self.classification_job_status.progress_bar
        self.taskProgressBar.setRange(0, 100)
        return panel

    def _create_results_panel(self) -> QWidget:
        panel = QFrame(self)
        ui = ClassificationResultsPanelView()
        ui.setupUi(panel)
        self._results_panel_ui = ui
        for name in (
            "activeModelCombo",
            "setActiveModelButton",
            "bestModelLabel",
            "bestMacroF1Label",
            "bestBalancedAccuracyLabel",
            "bestAccuracyLabel",
            "resultSamplesLabel",
            "resultClassesLabel",
            "resultValidationLabel",
            "resultsOutdatedLabel",
            "resultTabs",
            "overviewSplitter",
            "resultsTable",
            "metricChartLabel",
            "confusionNormalizeCombo",
            "confusionMatrixTable",
            "perClassTable",
            "misclassifiedTable",
        ):
            setattr(self, name, getattr(ui, name))

        self.resultTabs.setObjectName("classificationResultTabs")
        self.overviewSplitter.setObjectName("resultsOverviewSplitter")
        self.resultsTable.verticalHeader().setVisible(False)
        self.resultsTable.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.Stretch,
        )
        self.confusionMatrixTable.setObjectName("confusionMatrixView")
        self.confusionMatrixTable.verticalHeader().setVisible(True)
        self.perClassTable.verticalHeader().setVisible(False)
        self.perClassTable.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )
        self.misclassifiedTable.verticalHeader().setVisible(False)
        self.misclassifiedTable.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )
        return panel

    def _set_log_visible(self, visible: bool) -> None:
        self.classification_log_section.set_expanded(visible)

    def set_job_state(
        self,
        state: str,
        *,
        progress: int | None = None,
    ) -> None:
        """Update shared status presentation while keeping percent-based aliases。"""

        message = self.runStatusLabel.text()
        normalized_progress = None if progress is None else progress / 100.0
        self.classification_job_status.set_state(
            state,
            message,
            progress=normalized_progress,
        )
        if progress is not None:
            self.taskProgressBar.setRange(0, 100)
            self.taskProgressBar.setValue(max(0, min(100, int(progress))))

    @staticmethod
    def _detach_widget(widget: QWidget) -> None:
        parent = widget.parentWidget()
        if parent is None or parent.layout() is None:
            return

        def remove_from(layout) -> bool:
            index = layout.indexOf(widget)
            if index >= 0:
                layout.takeAt(index)
                return True
            for item_index in range(layout.count()):
                child_layout = layout.itemAt(item_index).layout()
                if child_layout is not None and remove_from(child_layout):
                    return True
            return False

        remove_from(parent.layout())

    def _load_stylesheet(self) -> None:
        if STYLE_PATH.exists():
            self.setStyleSheet(STYLE_PATH.read_text(encoding="utf-8"))
