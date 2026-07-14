"""Programmatic Classification page widget."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


STYLE_PATH = Path(__file__).resolve().parent / "styles" / "classification_page.qss"


class ClassificationPage(QWidget):
    """Single-page workflow UI for labeled dataset classification."""

    filesDropped = pyqtSignal(list)
    stepChanged = pyqtSignal(str)

    _STEP_NAMES = ("Dataset", "Preprocessing", "Algorithms", "Results")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ClassificationPageRoot")
        self.setAcceptDrops(True)
        self._responsive_mode = ""
        self._build_ui()
        self._load_stylesheet()
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
        if step not in self._STEP_NAMES:
            return
        index = self._STEP_NAMES.index(step)
        if self.workflowStack.currentIndex() != index:
            self.workflowStack.setCurrentIndex(index)
        for name, button in self._step_buttons.items():
            button.setChecked(name == step)
        self._current_step = step
        self.stepChanged.emit(step)

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
        elif width >= 1100 and height >= 760:
            mode = "medium"
        else:
            mode = "compact"
        if mode == self._responsive_mode:
            return
        self._responsive_mode = mode
        if mode == "compact":
            self.datasetInspectionSplitter.setOrientation(Qt.Vertical)
            self.algorithmConfigSplitter.setOrientation(Qt.Vertical)
            self.datasetPanel.setMinimumWidth(0)
            self.inspectionPanel.setMinimumWidth(0)
            self.datasetStepContent.setMinimumHeight(960)
            self.algorithmsStepContent.setMinimumHeight(980)
        else:
            self.datasetInspectionSplitter.setOrientation(Qt.Horizontal)
            self.algorithmConfigSplitter.setOrientation(Qt.Horizontal)
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
            self.algorithmConfigSplitter.setSizes([420, 540])
        elif self._responsive_mode == "wide":
            self.datasetInspectionSplitter.setSizes([430, 900])
            self.algorithmConfigSplitter.setSizes([520, 900])
        else:
            self.datasetInspectionSplitter.setSizes([380, 680])
            self.algorithmConfigSplitter.setSizes([440, 680])
        self.overviewSplitter.setSizes([390, 260])

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        root.addWidget(self._build_header())
        root.addWidget(self._build_stepper())

        self.workflowStack = QStackedWidget(self)
        self.workflowStack.setObjectName("classificationWorkflowStack")
        self.workflowStack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.datasetStepContent = QWidget(self)
        dataset_layout = QVBoxLayout(self.datasetStepContent)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        self.datasetInspectionSplitter = QSplitter(Qt.Horizontal, self.datasetStepContent)
        self.datasetInspectionSplitter.setObjectName("datasetInspectionSplitter")
        self.datasetPanel = self._build_dataset_panel()
        self.inspectionPanel = self._build_inspection_panel()
        self.datasetInspectionSplitter.addWidget(self.datasetPanel)
        self.datasetInspectionSplitter.addWidget(self.inspectionPanel)
        self.datasetInspectionSplitter.setChildrenCollapsible(False)
        self.datasetInspectionSplitter.setStretchFactor(0, 0)
        self.datasetInspectionSplitter.setStretchFactor(1, 1)
        dataset_layout.addWidget(self.datasetInspectionSplitter)
        self.workflowStack.addWidget(self._scroll_step(self.datasetStepContent, "datasetStepScrollArea"))

        self.preprocessingStepContent = self._build_preprocessing_panel()
        self.workflowStack.addWidget(self._scroll_step(self.preprocessingStepContent, "preprocessingStepScrollArea"))

        self.algorithmsStepContent = self._build_experiment_panel()
        self.workflowStack.addWidget(self._scroll_step(self.algorithmsStepContent, "algorithmsStepScrollArea"))

        self.resultsPanel = self._build_results_panel()
        self.workflowStack.addWidget(self._scroll_step(self.resultsPanel, "resultsStepScrollArea"))
        root.addWidget(self.workflowStack, 1)

        root.addWidget(self._build_log_panel())

        self.legacyClassListWidget = QListWidget(self)
        self.legacyClassListWidget.setObjectName("ClassificationImportListWidget")
        self.legacyClassListWidget.hide()
        self.set_step("Dataset")

    def _scroll_step(self, content: QWidget, object_name: str) -> QScrollArea:
        area = QScrollArea(self)
        area.setObjectName(object_name)
        area.setWidgetResizable(True)
        area.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        area.setMinimumSize(0, 0)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        area.setWidget(content)
        return area

    def _build_header(self) -> QWidget:
        header = QWidget(self)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self.titleLabel = QLabel("Classification", header)
        self.titleLabel.setObjectName("classificationTitle")
        self.subtitleLabel = QLabel(
            "Import labeled datasets, compare classifiers, and inspect model performance.",
            header,
        )
        self.subtitleLabel.setObjectName("classificationSubtitle")
        self.subtitleLabel.setWordWrap(True)
        self.subtitleLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        title_box.addWidget(self.titleLabel)
        title_box.addWidget(self.subtitleLabel)
        layout.addLayout(title_box, 1)

        self.newSessionButton = QPushButton("New Session", header)
        self.loadSessionButton = QPushButton("Load Session", header)
        self.saveSessionButton = QPushButton("Save Session", header)
        self.helpButton = QToolButton(header)
        self.helpButton.setText("?")
        self.helpButton.setToolTip("Classification workflow help")
        for button in (self.newSessionButton, self.loadSessionButton, self.saveSessionButton):
            button.setObjectName(button.text().replace(" ", "") + "Button")
            layout.addWidget(button)
        layout.addWidget(self.helpButton)
        return header

    def _build_stepper(self) -> QWidget:
        stepper = QFrame(self)
        stepper.setObjectName("classificationStepper")
        layout = QHBoxLayout(stepper)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        self.datasetStepButton = self._step_button("1  Dataset")
        self.preprocessingStepButton = self._step_button("2  Preprocessing")
        self.algorithmsStepButton = self._step_button("3  Algorithms")
        self.resultsStepButton = self._step_button("4  Results")
        self._step_buttons = {
            "Dataset": self.datasetStepButton,
            "Preprocessing": self.preprocessingStepButton,
            "Algorithms": self.algorithmsStepButton,
            "Results": self.resultsStepButton,
        }
        for name, button in self._step_buttons.items():
            layout.addWidget(button, 1)
            button.clicked.connect(lambda _checked=False, step=name: self.set_step(step))
        layout.addStretch(1)
        self.stateBadgeLabel = QLabel("EMPTY", stepper)
        self.stateBadgeLabel.setObjectName("stateBadgeLabel")
        layout.addWidget(self.stateBadgeLabel)
        return stepper

    def _step_button(self, text: str) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setCheckable(True)
        button.setAutoRaise(False)
        button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        button.setMinimumWidth(105)
        return button

    def _build_dataset_panel(self) -> QWidget:
        panel = self._section("Dataset")
        layout = panel.layout()

        top = QHBoxLayout()
        self.addClassButton = QPushButton("+ Add Class", panel)
        self.addClassButton.setObjectName("addClassButton")
        self.scanImportButton = QPushButton("Scan && Import", panel)
        self.scanImportButton.setObjectName("scanImportButton")
        top.addWidget(self.addClassButton)
        top.addWidget(self.scanImportButton)
        layout.addLayout(top)

        self.datasetCardsScrollArea = QScrollArea(panel)
        self.datasetCardsScrollArea.setWidgetResizable(True)
        self.datasetCardsScrollArea.setObjectName("datasetCardsScrollArea")
        self.datasetCardsContainer = QWidget(self.datasetCardsScrollArea)
        self.datasetCardsLayout = QVBoxLayout(self.datasetCardsContainer)
        self.datasetCardsLayout.setContentsMargins(0, 0, 0, 0)
        self.datasetCardsLayout.setSpacing(8)
        self.datasetCardsLayout.addStretch(1)
        self.datasetCardsScrollArea.setWidget(self.datasetCardsContainer)
        layout.addWidget(self.datasetCardsScrollArea, 2)

        self.datasetSummaryFrame = QFrame(panel)
        self.datasetSummaryFrame.setObjectName("datasetSummaryFrame")
        summary_grid = QGridLayout(self.datasetSummaryFrame)
        summary_grid.setContentsMargins(10, 8, 10, 8)
        summary_grid.setHorizontalSpacing(8)
        summary_grid.setVerticalSpacing(4)
        self.summaryClassesLabel = QLabel("0", self.datasetSummaryFrame)
        self.summaryTotalLabel = QLabel("0", self.datasetSummaryFrame)
        self.summaryValidLabel = QLabel("0", self.datasetSummaryFrame)
        self.summaryInvalidLabel = QLabel("0", self.datasetSummaryFrame)
        self.summaryBalanceLabel = QLabel("-", self.datasetSummaryFrame)
        for row, (name, value) in enumerate(
            (
                ("Classes", self.summaryClassesLabel),
                ("Total samples", self.summaryTotalLabel),
                ("Valid samples", self.summaryValidLabel),
                ("Invalid samples", self.summaryInvalidLabel),
                ("Class balance", self.summaryBalanceLabel),
            )
        ):
            summary_grid.addWidget(QLabel(name, self.datasetSummaryFrame), row, 0)
            summary_grid.addWidget(value, row, 1)
        layout.addWidget(self.datasetSummaryFrame)

        table_tools = QHBoxLayout()
        self.datasetSearchEdit = QLineEdit(panel)
        self.datasetSearchEdit.setPlaceholderText("Search file")
        self.classFilterCombo = QComboBox(panel)
        self.classFilterCombo.addItem("All classes")
        self.qcFilterCombo = QComboBox(panel)
        self.qcFilterCombo.addItems(["All QC", "Ready", "Warning", "Error", "Pending"])
        table_tools.addWidget(self.datasetSearchEdit, 2)
        table_tools.addWidget(self.classFilterCombo, 1)
        table_tools.addWidget(self.qcFilterCombo, 1)
        layout.addLayout(table_tools)

        self.datasetTable = QTableWidget(panel)
        self.datasetTable.setObjectName("datasetTable")
        self.datasetTable.setColumnCount(9)
        self.datasetTable.setHorizontalHeaderLabels(
            ["Include", "Class", "File", "Type", "Shape", "Load status", "QC status", "Prediction", "Confidence"]
        )
        self.datasetTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.datasetTable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.datasetTable.verticalHeader().setVisible(False)
        self.datasetTable.horizontalHeader().setStretchLastSection(False)
        self.datasetTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.datasetTable.setSortingEnabled(True)
        layout.addWidget(self.datasetTable, 3)

        sample_tools = QHBoxLayout()
        self.excludeSelectedButton = QPushButton("Exclude", panel)
        self.includeSelectedButton = QPushButton("Include", panel)
        self.removeSelectedSamplesButton = QPushButton("Remove", panel)
        self.openSelectedLocationButton = QPushButton("Open", panel)
        self.copySelectedPathsButton = QPushButton("Copy Paths", panel)
        self.exportSelectedFilesButton = QPushButton("Export List", panel)
        for button in (
            self.excludeSelectedButton,
            self.includeSelectedButton,
            self.removeSelectedSamplesButton,
            self.openSelectedLocationButton,
            self.copySelectedPathsButton,
            self.exportSelectedFilesButton,
        ):
            sample_tools.addWidget(button)
        layout.addLayout(sample_tools)
        return panel

    def _build_inspection_panel(self) -> QWidget:
        panel = self._section("Data Inspection")
        layout = panel.layout()

        preview_header = QHBoxLayout()
        self.prevSampleButton = QToolButton(panel)
        self.prevSampleButton.setText("<")
        self.nextSampleButton = QToolButton(panel)
        self.nextSampleButton.setText(">")
        self.sampleIndexLabel = QLabel("0 / 0", panel)
        self.sampleFileLabel = QLabel("No sample selected", panel)
        self.sampleShapeLabel = QLabel("-", panel)
        preview_header.addWidget(self.prevSampleButton)
        preview_header.addWidget(self.nextSampleButton)
        preview_header.addWidget(self.sampleIndexLabel)
        preview_header.addWidget(self.sampleFileLabel, 1)
        preview_header.addWidget(self.sampleShapeLabel)
        layout.addLayout(preview_header)

        self.previewGraphicsView = QGraphicsView(panel)
        self.previewGraphicsView.setObjectName("previewGraphicsView")
        self.previewGraphicsView.setMinimumHeight(220)
        layout.addWidget(self.previewGraphicsView, 3)

        preview_controls = QGridLayout()
        self.previewLogScaleCheckBox = QCheckBox("Log scale", panel)
        self.previewColormapCombo = QComboBox(panel)
        self.previewColormapCombo.addItems(["viridis", "magma", "plasma", "jet", "gray"])
        self.previewAutoScaleCheckBox = QCheckBox("Auto scale", panel)
        self.previewAutoScaleCheckBox.setChecked(True)
        self.previewVminSpinBox = QDoubleSpinBox(panel)
        self.previewVminSpinBox.setRange(-1e12, 1e12)
        self.previewVmaxSpinBox = QDoubleSpinBox(panel)
        self.previewVmaxSpinBox.setRange(-1e12, 1e12)
        self.fitPreviewButton = QPushButton("Fit", panel)
        self.openFileLocationButton = QPushButton("Open file location", panel)
        preview_controls.addWidget(self.previewLogScaleCheckBox, 0, 0)
        preview_controls.addWidget(QLabel("Colormap", panel), 0, 1)
        preview_controls.addWidget(self.previewColormapCombo, 0, 2)
        preview_controls.addWidget(self.previewAutoScaleCheckBox, 0, 3)
        preview_controls.addWidget(QLabel("vmin", panel), 1, 0)
        preview_controls.addWidget(self.previewVminSpinBox, 1, 1)
        preview_controls.addWidget(QLabel("vmax", panel), 1, 2)
        preview_controls.addWidget(self.previewVmaxSpinBox, 1, 3)
        preview_controls.addWidget(self.fitPreviewButton, 2, 0)
        preview_controls.addWidget(self.openFileLocationButton, 2, 1, 1, 3)
        layout.addLayout(preview_controls)

        self.qualityFrame = QFrame(panel)
        self.qualityFrame.setObjectName("qualityFrame")
        quality_layout = QVBoxLayout(self.qualityFrame)
        quality_layout.setContentsMargins(10, 8, 10, 8)
        quality_top = QHBoxLayout()
        self.qualityStatusLabel = QLabel("Warning", self.qualityFrame)
        self.qualityStatusLabel.setObjectName("qualityStatusLabel")
        quality_top.addWidget(QLabel("Data Quality", self.qualityFrame))
        quality_top.addStretch(1)
        quality_top.addWidget(self.qualityStatusLabel)
        quality_layout.addLayout(quality_top)
        self.qualityListWidget = QListWidget(self.qualityFrame)
        self.qualityListWidget.setMaximumHeight(110)
        quality_layout.addWidget(self.qualityListWidget)
        layout.addWidget(self.qualityFrame)

        return panel

    def _build_preprocessing_panel(self) -> QWidget:
        panel = self._section("Preprocessing")
        layout = panel.layout()
        intro = QLabel(
            "Configure a shared preprocessing pipeline. Changes apply to every selected algorithm.",
            panel,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        preprocessing = QFrame(panel)
        preprocessing.setObjectName("preprocessingFrame")
        grid = QFormLayout(preprocessing)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        grid.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.dataTypeBadgeLabel = QLabel("auto", preprocessing)
        self.oneDPreprocessingCombo = QComboBox(preprocessing)
        self.oneDPreprocessingCombo.addItems(
            [
                "None",
                "Interpolate to common grid",
                "Crop range",
                "Normalize by max",
                "Normalize by area",
                "Standard scaling",
                "Log transform",
                "Smoothing",
            ]
        )
        self.oneDPreprocessingCombo.setCurrentText("Interpolate to common grid")
        self.twoDPreprocessingCombo = QComboBox(preprocessing)
        self.twoDPreprocessingCombo.addItems(
            ["None", "Center crop", "Resize", "Normalize", "Log transform", "Mask invalid pixels", "Flatten"]
        )
        self.twoDPreprocessingCombo.setCurrentText("Center crop")
        self.normalizeCombo = QComboBox(preprocessing)
        self.normalizeCombo.addItems(["none", "max", "area"])
        self.normalizeCombo.setCurrentText("max")
        self.preprocessingLogCheckBox = QCheckBox("Log transform", preprocessing)
        self.smoothingSpinBox = QSpinBox(preprocessing)
        self.smoothingSpinBox.setRange(0, 99)
        self.resizeRowsSpinBox = QSpinBox(preprocessing)
        self.resizeRowsSpinBox.setRange(1, 10000)
        self.resizeRowsSpinBox.setValue(256)
        self.resizeColsSpinBox = QSpinBox(preprocessing)
        self.resizeColsSpinBox.setRange(1, 10000)
        self.resizeColsSpinBox.setValue(256)
        rows = (
            ("Data type", self.dataTypeBadgeLabel),
            ("1D preprocessing", self.oneDPreprocessingCombo),
            ("2D preprocessing", self.twoDPreprocessingCombo),
            ("Normalize", self.normalizeCombo),
            ("Smoothing", self.smoothingSpinBox),
            ("Resize rows", self.resizeRowsSpinBox),
            ("Resize cols", self.resizeColsSpinBox),
        )
        for label, widget in rows:
            grid.addRow(label, widget)
        grid.addRow("", self.preprocessingLogCheckBox)
        self.inputSummaryLabel = QLabel("Samples: 0 | Features: 0 | Input shape: - | Memory: -", preprocessing)
        self.inputSummaryLabel.setWordWrap(True)
        grid.addRow("Input summary", self.inputSummaryLabel)
        layout.addWidget(preprocessing)
        layout.addStretch(1)
        return panel

    def _build_experiment_panel(self) -> QWidget:
        panel = self._section("Experiment Setup")
        layout = panel.layout()
        layout.setContentsMargins(10, 8, 10, 8)

        self.algorithmConfigSplitter = QSplitter(Qt.Horizontal, panel)
        self.algorithmConfigSplitter.setObjectName("algorithmConfigSplitter")
        self.algorithmConfigSplitter.setChildrenCollapsible(False)
        self.algorithmConfigSplitter.setStretchFactor(0, 0)
        self.algorithmConfigSplitter.setStretchFactor(1, 1)

        config_column = QWidget(self.algorithmConfigSplitter)
        config_layout = QVBoxLayout(config_column)
        config_layout.setContentsMargins(0, 0, 6, 0)
        config_layout.setSpacing(8)

        validation = QFrame(config_column)
        validation.setObjectName("validationFrame")
        validation_form = QFormLayout(validation)
        validation_form.setContentsMargins(10, 8, 10, 8)
        validation_form.setHorizontalSpacing(12)
        validation_form.setVerticalSpacing(6)
        validation_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        validation_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.validationMethodCombo = QComboBox(validation)
        self.validationMethodCombo.addItems(
            ["Stratified K-fold", "Stratified train/test split", "Repeated stratified K-fold"]
        )
        self.testSizeSpinBox = QDoubleSpinBox(validation)
        self.testSizeSpinBox.setRange(0.05, 0.9)
        self.testSizeSpinBox.setSingleStep(0.05)
        self.testSizeSpinBox.setValue(0.2)
        self.foldsSpinBox = QSpinBox(validation)
        self.foldsSpinBox.setRange(2, 50)
        self.foldsSpinBox.setValue(5)
        self.repeatsSpinBox = QSpinBox(validation)
        self.repeatsSpinBox.setRange(1, 20)
        self.repeatsSpinBox.setValue(1)
        self.randomSeedSpinBox = QSpinBox(validation)
        self.randomSeedSpinBox.setRange(0, 999999)
        self.randomSeedSpinBox.setValue(42)
        self.shuffleCheckBox = QCheckBox("Shuffle", validation)
        self.shuffleCheckBox.setChecked(True)
        self.rankingMetricCombo = QComboBox(validation)
        self.rankingMetricCombo.addItems(["Macro F1", "Balanced Accuracy", "Accuracy"])
        for label, widget in (
            ("Validation", self.validationMethodCombo),
            ("Test size", self.testSizeSpinBox),
            ("Folds", self.foldsSpinBox),
            ("Repeats", self.repeatsSpinBox),
            ("Random seed", self.randomSeedSpinBox),
            ("Ranking metric", self.rankingMetricCombo),
        ):
            validation_form.addRow(label, widget)
        validation_form.addRow("", self.shuffleCheckBox)
        self.validationWarningLabel = QLabel("", validation)
        self.validationWarningLabel.setObjectName("validationWarningLabel")
        self.validationWarningLabel.setWordWrap(True)
        validation_form.addRow(self.validationWarningLabel)
        config_layout.addWidget(validation)

        projection = QFrame(config_column)
        projection.setObjectName("projectionFrame")
        projection_form = QFormLayout(projection)
        projection_form.setContentsMargins(10, 8, 10, 8)
        projection_form.setHorizontalSpacing(12)
        projection_form.setVerticalSpacing(6)
        projection_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        projection_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.useProjectionCheckBox = QCheckBox("Use projection for training", projection)
        self.projectionMethodCombo = QComboBox(projection)
        self.projectionMethodCombo.addItems(["None", "PCA", "UMAP"])
        self.projectionComponentsSpinBox = QSpinBox(projection)
        self.projectionComponentsSpinBox.setRange(1, 2048)
        self.projectionComponentsSpinBox.setValue(2)
        self.pcaVarianceSpinBox = QDoubleSpinBox(projection)
        self.pcaVarianceSpinBox.setRange(0.1, 1.0)
        self.pcaVarianceSpinBox.setSingleStep(0.05)
        self.pcaVarianceSpinBox.setValue(0.95)
        self.umapNeighborsSpinBox = QSpinBox(projection)
        self.umapNeighborsSpinBox.setRange(2, 200)
        self.umapNeighborsSpinBox.setValue(15)
        self.umapMinDistSpinBox = QDoubleSpinBox(projection)
        self.umapMinDistSpinBox.setRange(0.0, 1.0)
        self.umapMinDistSpinBox.setSingleStep(0.05)
        self.umapMinDistSpinBox.setValue(0.1)
        self.tsneNoteLabel = QLabel("t-SNE is visualization-only and is not saved as a prediction transform.", projection)
        self.tsneNoteLabel.setWordWrap(True)
        projection_form.addRow(self.useProjectionCheckBox)
        for label, widget in (
            ("Method", self.projectionMethodCombo),
            ("Components", self.projectionComponentsSpinBox),
            ("PCA variance", self.pcaVarianceSpinBox),
            ("UMAP neighbors", self.umapNeighborsSpinBox),
            ("UMAP min_dist", self.umapMinDistSpinBox),
        ):
            projection_form.addRow(label, widget)
        projection_form.addRow(self.tsneNoteLabel)
        config_layout.addWidget(projection)
        config_layout.addStretch(1)

        algorithm_column = QWidget(self.algorithmConfigSplitter)
        algorithm_layout = QVBoxLayout(algorithm_column)
        algorithm_layout.setContentsMargins(6, 0, 0, 0)
        algorithm_layout.setSpacing(8)
        algorithm_tools = QGridLayout()
        self.selectRecommendedButton = QPushButton("Select recommended", algorithm_column)
        self.selectAllAlgorithmsButton = QPushButton("Select all", algorithm_column)
        self.clearAlgorithmsButton = QPushButton("Clear", algorithm_column)
        self.resetAlgorithmDefaultsButton = QPushButton("Reset defaults", algorithm_column)
        for index, button in enumerate((
            self.selectRecommendedButton,
            self.selectAllAlgorithmsButton,
            self.clearAlgorithmsButton,
            self.resetAlgorithmDefaultsButton,
        )):
            algorithm_tools.addWidget(button, index // 2, index % 2)
        algorithm_layout.addLayout(algorithm_tools)

        self.algorithmTable = QTableWidget(algorithm_column)
        self.algorithmTable.setObjectName("algorithmList")
        self.algorithmTable.setColumnCount(4)
        self.algorithmTable.setHorizontalHeaderLabels(["Use", "Algorithm", "Description", "Parameters"])
        self.algorithmTable.verticalHeader().setVisible(False)
        self.algorithmTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.algorithmTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.algorithmTable.setMinimumHeight(360)
        algorithm_layout.addWidget(self.algorithmTable, 1)

        self.algorithmConfigSplitter.addWidget(config_column)
        self.algorithmConfigSplitter.addWidget(algorithm_column)
        layout.addWidget(self.algorithmConfigSplitter, 1)

        run_frame = QFrame(panel)
        run_frame.setObjectName("runFrame")
        run_layout = QGridLayout(run_frame)
        run_layout.setContentsMargins(10, 8, 10, 8)
        self.runComparisonButton = QPushButton("Run Comparison", run_frame)
        self.runComparisonButton.setObjectName("runComparisonButton")
        self.cancelTaskButton = QPushButton("Cancel", run_frame)
        self.cancelTaskButton.setEnabled(False)
        self.runStatusLabel = QLabel("Selected algorithms: 0 | Valid samples: 0 | Estimated runs: 0 | EMPTY", run_frame)
        self.taskProgressBar = QProgressBar(run_frame)
        self.taskProgressBar.setRange(0, 100)
        run_layout.addWidget(self.runComparisonButton, 0, 0)
        run_layout.addWidget(self.cancelTaskButton, 0, 1)
        run_layout.addWidget(self.runStatusLabel, 1, 0, 1, 2)
        run_layout.addWidget(self.taskProgressBar, 2, 0, 1, 2)
        layout.addWidget(run_frame)
        return panel

    def _build_results_panel(self) -> QWidget:
        panel = QFrame(self)
        panel.setObjectName("resultsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        top = QGridLayout()
        top.addWidget(QLabel("Results", panel), 0, 0)
        self.activeModelCombo = QComboBox(panel)
        self.setActiveModelButton = QPushButton("Set as Active Model", panel)
        self.saveActiveModelButton = QPushButton("Save Active Model", panel)
        self.loadModelButton = QPushButton("Load Model", panel)
        self.exportResultsButton = QPushButton("Export Results", panel)
        self.predictNewDataButton = QPushButton("Predict New Data", panel)
        top.addWidget(QLabel("Active model", panel), 0, 1)
        top.addWidget(self.activeModelCombo, 0, 2, 1, 2)
        top.addWidget(self.setActiveModelButton, 0, 4)
        for column, widget in enumerate((
            self.saveActiveModelButton,
            self.loadModelButton,
            self.exportResultsButton,
            self.predictNewDataButton,
        ), start=1):
            top.addWidget(widget, 1, column)
        top.setColumnStretch(2, 1)
        layout.addLayout(top)

        overview = QFrame(panel)
        overview.setObjectName("overviewFrame")
        overview_grid = QGridLayout(overview)
        overview_grid.setContentsMargins(8, 6, 8, 6)
        self.bestModelLabel = QLabel("-", overview)
        self.bestMacroF1Label = QLabel("-", overview)
        self.bestBalancedAccuracyLabel = QLabel("-", overview)
        self.bestAccuracyLabel = QLabel("-", overview)
        self.resultSamplesLabel = QLabel("0", overview)
        self.resultClassesLabel = QLabel("0", overview)
        self.resultValidationLabel = QLabel("-", overview)
        self.resultsOutdatedLabel = QLabel("", overview)
        overview_items = [
            ("Best model", self.bestModelLabel),
            ("Best Macro F1", self.bestMacroF1Label),
            ("Balanced Accuracy", self.bestBalancedAccuracyLabel),
            ("Accuracy", self.bestAccuracyLabel),
            ("Samples", self.resultSamplesLabel),
            ("Classes", self.resultClassesLabel),
            ("Validation", self.resultValidationLabel),
        ]
        for index, (label, widget) in enumerate(overview_items):
            row = index // 4
            col = (index % 4) * 2
            overview_grid.addWidget(QLabel(label, overview), row, col)
            overview_grid.addWidget(widget, row, col + 1)
        overview_grid.addWidget(self.resultsOutdatedLabel, 2, 0, 1, 8)
        layout.addWidget(overview)

        self.resultTabs = QTabWidget(panel)
        self.resultTabs.setObjectName("classificationResultTabs")
        self.resultTabs.setDocumentMode(True)

        overview_tab = QWidget(self.resultTabs)
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(4, 6, 4, 4)
        self.overviewSplitter = QSplitter(Qt.Vertical, overview_tab)
        self.overviewSplitter.setObjectName("resultsOverviewSplitter")
        self.overviewSplitter.setChildrenCollapsible(False)
        leaderboard = QWidget(self.overviewSplitter)
        leaderboard_layout = QVBoxLayout(leaderboard)
        leaderboard_layout.setContentsMargins(0, 0, 0, 0)
        leaderboard_layout.addWidget(QLabel("Leaderboard", leaderboard))
        self.resultsTable = QTableWidget(leaderboard)
        self.resultsTable.setObjectName("resultsTable")
        self.resultsTable.setColumnCount(9)
        self.resultsTable.setHorizontalHeaderLabels(
            [
                "Rank",
                "Algorithm",
                "Accuracy",
                "Balanced Accuracy",
                "Macro F1",
                "Weighted F1",
                "Training time",
                "Prediction time",
                "Status",
            ]
        )
        self.resultsTable.verticalHeader().setVisible(False)
        self.resultsTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.resultsTable.setSortingEnabled(True)
        self.resultsTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        leaderboard_layout.addWidget(self.resultsTable)
        chart = QWidget(self.overviewSplitter)
        chart_layout = QVBoxLayout(chart)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(QLabel("Metric comparison", chart))
        self.metricChartLabel = QLabel("No metrics yet", chart)
        self.metricChartLabel.setAlignment(Qt.AlignCenter)
        self.metricChartLabel.setObjectName("metricChartLabel")
        chart_layout.addWidget(self.metricChartLabel)
        self.overviewSplitter.addWidget(leaderboard)
        self.overviewSplitter.addWidget(chart)
        overview_layout.addWidget(self.overviewSplitter)
        self.resultTabs.addTab(overview_tab, "Overview")

        confusion_tab = QWidget(self.resultTabs)
        confusion_layout = QVBoxLayout(confusion_tab)
        confusion_layout.setContentsMargins(6, 8, 6, 6)
        confusion_top = QHBoxLayout()
        confusion_top.addWidget(QLabel("Confusion Matrix", confusion_tab))
        self.confusionNormalizeCombo = QComboBox(confusion_tab)
        self.confusionNormalizeCombo.addItems(["Raw counts", "Normalize by true class", "Normalize by predicted class"])
        confusion_top.addStretch(1)
        confusion_top.addWidget(self.confusionNormalizeCombo)
        confusion_layout.addLayout(confusion_top)
        self.confusionMatrixTable = QTableWidget(confusion_tab)
        self.confusionMatrixTable.setObjectName("confusionMatrixView")
        self.confusionMatrixTable.verticalHeader().setVisible(True)
        confusion_layout.addWidget(self.confusionMatrixTable)
        self.resultTabs.addTab(confusion_tab, "Confusion Matrix")

        metrics_tab = QWidget(self.resultTabs)
        metrics_layout = QVBoxLayout(metrics_tab)
        metrics_layout.setContentsMargins(6, 8, 6, 6)
        metrics_layout.addWidget(QLabel("Per-class metrics", metrics_tab))
        self.perClassTable = QTableWidget(metrics_tab)
        self.perClassTable.setColumnCount(5)
        self.perClassTable.setHorizontalHeaderLabels(["Class", "Precision", "Recall", "F1-score", "Support"])
        self.perClassTable.verticalHeader().setVisible(False)
        self.perClassTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        metrics_layout.addWidget(self.perClassTable)
        self.resultTabs.addTab(metrics_tab, "Per-class Metrics")

        misclassified_tab = QWidget(self.resultTabs)
        misclassified_layout = QVBoxLayout(misclassified_tab)
        misclassified_layout.setContentsMargins(6, 8, 6, 6)
        misclassified_layout.addWidget(QLabel("Misclassified samples", misclassified_tab))
        self.misclassifiedTable = QTableWidget(misclassified_tab)
        self.misclassifiedTable.setObjectName("misclassifiedTable")
        self.misclassifiedTable.setColumnCount(6)
        self.misclassifiedTable.setHorizontalHeaderLabels(["File", "True label", "Predicted label", "Confidence", "Shape", "Preview"])
        self.misclassifiedTable.verticalHeader().setVisible(False)
        self.misclassifiedTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        misclassified_layout.addWidget(self.misclassifiedTable)
        self.resultTabs.addTab(misclassified_tab, "Misclassified")

        embedding = QWidget(self.resultTabs)
        embedding_layout = QVBoxLayout(embedding)
        embedding_layout.setContentsMargins(6, 8, 6, 6)
        embedding_tools = QHBoxLayout()
        self.embeddingMethodCombo = QComboBox(embedding)
        self.embeddingMethodCombo.addItems(["PCA 2D", "UMAP 2D", "t-SNE 2D"])
        self.embeddingColorCombo = QComboBox(embedding)
        self.embeddingColorCombo.addItems(["True label", "Predicted label", "Correct / incorrect"])
        self.runEmbeddingButton = QPushButton("Run Embedding", embedding)
        embedding_tools.addWidget(self.embeddingMethodCombo)
        embedding_tools.addWidget(self.embeddingColorCombo)
        embedding_tools.addWidget(self.runEmbeddingButton)
        embedding_layout.addLayout(embedding_tools)
        self.embeddingGraphicsView = QGraphicsView(embedding)
        embedding_layout.addWidget(self.embeddingGraphicsView)
        self.resultTabs.addTab(embedding, "Embedding")

        prediction = QWidget(self.resultTabs)
        prediction_layout = QVBoxLayout(prediction)
        prediction_layout.setContentsMargins(6, 8, 6, 6)
        self.exportPredictionsButton = QPushButton("Export Prediction CSV", prediction)
        prediction_layout.addWidget(self.exportPredictionsButton, 0, Qt.AlignRight)
        self.predictionTable = QTableWidget(prediction)
        self.predictionTable.setColumnCount(4)
        self.predictionTable.setHorizontalHeaderLabels(["File", "Predicted label", "Confidence", "Status"])
        self.predictionTable.verticalHeader().setVisible(False)
        self.predictionTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        prediction_layout.addWidget(self.predictionTable)
        self.resultTabs.addTab(prediction, "Prediction")

        layout.addWidget(self.resultTabs, 1)
        return panel

    def _build_log_panel(self) -> QWidget:
        panel = QFrame(self)
        panel.setObjectName("logPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.logToggleButton = QToolButton(panel)
        self.logToggleButton.setText("Show Operation Log")
        self.logToggleButton.setCheckable(True)
        self.logToggleButton.setChecked(False)
        self.logTextBrowser = QTextBrowser(panel)
        self.logTextBrowser.setObjectName("classificationPageTextBrowser")
        self.logTextBrowser.setVisible(False)
        self.logTextBrowser.setMaximumHeight(220)
        self.logTextBrowser.setMinimumHeight(140)
        self.logToggleButton.toggled.connect(self._set_log_visible)
        layout.addWidget(self.logToggleButton)
        layout.addWidget(self.logTextBrowser)
        return panel

    def _set_log_visible(self, visible: bool) -> None:
        self.logTextBrowser.setVisible(visible)
        self.logToggleButton.setText("Hide Operation Log" if visible else "Show Operation Log")

    def _section(self, title: str) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName(title.replace(" ", "") + "Section")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        label = QLabel(title, frame)
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        return frame

    def _load_stylesheet(self) -> None:
        if STYLE_PATH.exists():
            self.setStyleSheet(STYLE_PATH.read_text(encoding="utf-8"))
