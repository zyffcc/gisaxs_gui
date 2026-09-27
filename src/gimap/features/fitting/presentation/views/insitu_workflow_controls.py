"""Controls for the feature-owned Fitting In-situ workflow page."""

from __future__ import annotations
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (
    QCheckBox,
    QTabWidget,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

class InSituWorkflowControls(QWidget):
    """Own layout widgets only; commands remain in page/binding classes."""

    STEP_KEYS = ("source", "preprocess", "geometry", "cut", "fit", "results")

    def __init__(self, owner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._owner = owner
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.stack = QStackedWidget(self)
        self.stack.setObjectName("fittingInsituParameterStack")
        self.stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        root.addWidget(self.stack)
        self.pages = {}
        self._build_source_page()
        self._build_preprocess_page()
        self._build_geometry_page()
        self._build_cut_page()
        self._build_fit_page()
        self._build_results_page()
        # The frequent path has three steps. Scientific setup lives in one inspector.
        self.setupTabs = QTabWidget()
        self.setupTabs.setDocumentMode(True)
        for key, title in (("preprocess", "Preprocess"), ("geometry", "Detector"), ("cut", "Cut")):
            widget = self.pages[key]
            self.stack.removeWidget(widget)
            self.setupTabs.addTab(widget, title)
        self.pages["fit"].layout().addWidget(self.setupTabs)
        self.setupTabs.hide()
        self.applyRecipeButton.setText("Save analysis settings")
        self.pages["fit"].layout().insertWidget(5, self.applyRecipeButton)
        self.setupButton = QPushButton("Detector / cut settings…")
        self.setupButton.setCheckable(True)
        self.setupButton.toggled.connect(self.setupTabs.setVisible)
        self.pages["fit"].layout().insertWidget(3, self.setupButton)

    def show_step(self, key: str) -> None:
        if key in self.STEP_KEYS:
            if key in ("preprocess", "geometry", "cut"):
                self.stack.setCurrentWidget(self.pages["fit"])
                self.setupTabs.show(); self.setupTabs.setCurrentWidget(self.pages[key])
            else:
                self.stack.setCurrentWidget(self.pages[key])

    def _page(self, key: str, title: str, description: str):
        page = QWidget(self.stack)
        page.setObjectName(f"fittingInsitu{key.title()}Parameters")
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(2, 2, 8, 8)
        layout.setSpacing(10)
        title_label = QLabel(title, page)
        title_label.setProperty("gimapSectionTitle", True)
        meta = QLabel(description, page)
        meta.setProperty("gimapMeta", True)
        meta.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(meta)
        self.stack.addWidget(page)
        self.pages[key] = page
        return page, layout

    def _build_source_page(self) -> None:
        page, layout = self._page(
            "source",
            "Source",
            "Choose live acquisition or an existing sequence. Both use the same Recipe.",
        )
        form = QFormLayout()
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.runModeCombo = QComboBox(page)
        self.runModeCombo.setObjectName("fittingInsituRunModeCombo")
        self.runModeCombo.addItems(("Process Existing Sequence", "Live Watch"))
        form.addRow("Mode", self.runModeCombo)

        self.sourceKindCombo = QComboBox(page)
        self.sourceKindCombo.setObjectName("fittingInsituSourceKindCombo")
        self.sourceKindCombo.addItems(("CBF images", "NXS module series"))
        self.sourceKindCombo.setToolTip(
            "CBF treats each file as one frame; NXS groups detector modules and watches internal frames."
        )
        form.addRow("Detector source", self.sourceKindCombo)

        folder_row = QWidget(page)
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        self.sequenceFolderEdit = QLineEdit(folder_row)
        self.sequenceFolderEdit.setObjectName("fittingInsituSequenceFolderEdit")
        self.sequenceFolderEdit.setPlaceholderText(
            "Acquisition root; child folders can be watched recursively"
        )
        self.sequenceBrowseButton = QPushButton("…", folder_row)
        self.sequenceBrowseButton.setObjectName("fittingInsituSequenceBrowseButton")
        self.sequenceBrowseButton.setToolTip("Choose acquisition root folder")
        folder_layout.addWidget(self.sequenceFolderEdit, 1)
        folder_layout.addWidget(self.sequenceBrowseButton)
        form.addRow("Root folder", folder_row)
        self.sequencePatternEdit = QLineEdit("*.cbf", page)
        self.sequencePatternEdit.setObjectName("fittingInsituSequencePatternEdit")
        form.addRow("File pattern", self.sequencePatternEdit)
        self.recursiveCheckBox = QCheckBox("Include child folders", page)
        self.recursiveCheckBox.setObjectName("fittingInsituRecursiveCheckBox")
        self.recursiveCheckBox.setChecked(True)
        form.addRow("", self.recursiveCheckBox)
        self.nxsModuleCountSpinBox = QSpinBox(page)
        self.nxsModuleCountSpinBox.setObjectName("fittingInsituNxsModuleCountSpinBox")
        self.nxsModuleCountSpinBox.setRange(1, 100)
        self.nxsModuleCountSpinBox.setValue(11)
        self.nxsModuleCountSpinBox.setToolTip(
            "Queue an NXS group only after at least this many module files exist."
        )
        form.addRow("NXS modules", self.nxsModuleCountSpinBox)
        self.nxsModuleCountLabel = form.labelForField(self.nxsModuleCountSpinBox)
        self.nxsModuleCountLabel.hide()
        self.nxsModuleCountSpinBox.hide()
        layout.addLayout(form)

        self.liveSettingsWidget = QWidget(page)
        live_form = QFormLayout(self.liveSettingsWidget)
        live_form.setContentsMargins(0, 0, 0, 0)
        self.pollSpinBox = self._double_spin(0.2, 3600.0, 2.0, 1)
        self.pollSpinBox.setSuffix(" s")
        self.stableCheckBox = QCheckBox("Wait until file size is stable", page)
        self.stableCheckBox.setChecked(True)
        live_form.addRow("Poll interval", self.pollSpinBox)
        live_form.addRow("", self.stableCheckBox)
        layout.addWidget(self.liveSettingsWidget)

        self.sequenceSettingsWidget = QWidget(page)
        range_grid = QGridLayout(self.sequenceSettingsWidget)
        range_grid.setContentsMargins(0, 0, 0, 0)
        self.sequenceStartSpinBox = self._range_spin()
        self.sequenceEndSpinBox = self._range_spin()
        self.sequenceStepSpinBox = QSpinBox(page)
        self.sequenceStepSpinBox.setRange(1, 1_000_000)
        self.sequenceStepSpinBox.setValue(1)
        for column, (label, editor) in enumerate(
            (("Start", self.sequenceStartSpinBox), ("End", self.sequenceEndSpinBox), ("Step", self.sequenceStepSpinBox))
        ):
            range_grid.addWidget(QLabel(label, page), 0, column)
            range_grid.addWidget(editor, 1, column)
        layout.addWidget(self.sequenceSettingsWidget)

        common = QFormLayout()
        self.fitEverySpinBox = QSpinBox(page)
        self.fitEverySpinBox.setRange(1, 100_000)
        self.fitEverySpinBox.setValue(1)
        self.uiEverySpinBox = QSpinBox(page)
        self.uiEverySpinBox.setRange(1, 100_000)
        self.uiEverySpinBox.setValue(5)
        self.uiEverySpinBox.setToolTip("Refresh heavy previews every N processed batches.")
        common.addRow("Images per analysis", self.fitEverySpinBox)
        common.addRow("Preview every", self.uiEverySpinBox)
        layout.addLayout(common)
        layout.addStretch(1)

    def _build_preprocess_page(self) -> None:
        page, layout = self._page(
            "preprocess",
            "Preprocess",
            "Scientific preprocessing is applied deterministically to every source frame.",
        )
        self.flipUdCheckBox = QCheckBox("Flip image vertically", page)
        self.thresholdCheckBox = QCheckBox("Apply intensity threshold", page)
        self.thresholdMinSpinBox = self._double_spin(-1e12, 1e12, -1e12, 4)
        self.thresholdMaxSpinBox = self._double_spin(-1e12, 1e12, 1e12, 4)
        self.mirrorFillCheckBox = QCheckBox("Mirror-fill detector gaps", page)
        self.mirrorMarginSpinBox = QSpinBox(page)
        self.mirrorMarginSpinBox.setRange(0, 10_000)
        self.invalidMarginSpinBox = QSpinBox(page)
        self.invalidMarginSpinBox.setRange(0, 20)
        self.invalidMarginSpinBox.setValue(3)
        self.invalidMarginSpinBox.setSuffix(" px")
        self.invalidMarginSpinBox.setToolTip("CBF invalid detector pixels and their neighbours are blocked before all analysis. Zero counts remain valid; masked gaps are not interpolated into observations.")
        form = QFormLayout()
        form.addRow("", self.flipUdCheckBox)
        form.addRow("", self.thresholdCheckBox)
        form.addRow("Threshold min", self.thresholdMinSpinBox)
        form.addRow("Threshold max", self.thresholdMaxSpinBox)
        form.addRow("", self.mirrorFillCheckBox)
        form.addRow("Gap margin", self.mirrorMarginSpinBox)
        form.addRow("CBF bad-pixel guard", self.invalidMarginSpinBox)
        layout.addLayout(form)
        layout.addStretch(1)

    def _build_geometry_page(self) -> None:
        page, layout = self._page(
            "geometry",
            "Geometry",
            "Experiment geometry normally stays fixed for one acquisition series.",
        )
        self.distanceSpinBox = self._double_spin(0.0, 1e9, 2000.0, 6)
        self.grazingSpinBox = self._double_spin(-360.0, 360.0, 0.2, 6)
        self.wavelengthSpinBox = self._double_spin(0.0, 1e6, 0.1, 8)
        self.centerXSpinBox = self._double_spin(-1e9, 1e9, 0.0, 4)
        self.centerYSpinBox = self._double_spin(-1e9, 1e9, 0.0, 4)
        self.pixelXSpinBox = self._double_spin(0.0, 1e6, 172.0, 6)
        self.pixelYSpinBox = self._double_spin(0.0, 1e6, 172.0, 6)
        form = QFormLayout()
        for label, editor in (
            ("Distance (mm)", self.distanceSpinBox),
            ("Grazing angle (°)", self.grazingSpinBox),
            ("Wavelength (nm)", self.wavelengthSpinBox),
            ("Beam center X (px)", self.centerXSpinBox),
            ("Beam center Y (px)", self.centerYSpinBox),
            ("Pixel X (µm)", self.pixelXSpinBox),
            ("Pixel Y (µm)", self.pixelYSpinBox),
        ):
            form.addRow(label, editor)
        layout.addLayout(form)
        layout.addStretch(1)

    def _build_cut_page(self) -> None:
        page, layout = self._page(
            "cut",
            "Yoneda & cut",
            "Define the region extracted from each preprocessed AnalysisImage.",
        )
        self.autoShowCheckBox = QCheckBox("Update preview while processing", page)
        self.autoShowCheckBox.setChecked(True)
        self.autoCutCheckBox = QCheckBox("Extract cut for every frame", page)
        self.autoCutCheckBox.setChecked(True)
        self.cutCenterVerticalSpinBox = self._double_spin(-1e9, 1e9, 0.0, 4)
        self.cutCenterParallelSpinBox = self._double_spin(-1e9, 1e9, 0.0, 4)
        self.cutVerticalSpinBox = self._double_spin(0.0, 1e9, 10.0, 4)
        self.cutParallelSpinBox = self._double_spin(0.0, 1e9, 10.0, 4)
        self.yonedaThicknessSpinBox = QSpinBox(page)
        self.yonedaThicknessSpinBox.setRange(1, 10_000)
        self.yonedaThicknessSpinBox.setValue(5)
        self.centerTrackingCombo = self._combo(
            ("Fixed", "Detect each frame", "Previous success")
        )
        self.yonedaTrackingCombo = self._combo(
            ("Fixed", "Detect each frame", "Previous success")
        )
        form = QFormLayout()
        form.addRow("", self.autoShowCheckBox)
        form.addRow("", self.autoCutCheckBox)
        form.addRow("Center vertical (px)", self.cutCenterVerticalSpinBox)
        form.addRow("Center parallel (px)", self.cutCenterParallelSpinBox)
        form.addRow("Cut vertical (px)", self.cutVerticalSpinBox)
        form.addRow("Cut parallel (px)", self.cutParallelSpinBox)
        form.addRow("Yoneda thickness (px)", self.yonedaThicknessSpinBox)
        form.addRow("Center tracking", self.centerTrackingCombo)
        form.addRow("Yoneda tracking", self.yonedaTrackingCombo)
        layout.addLayout(form)
        layout.addStretch(1)

    def _build_fit_page(self) -> None:
        page, layout = self._page(
            "fit",
            "Fit",
            "Run the current 1D workflow for every frame. Save settings before starting.",
        )
        self.autoFitCheckBox = QCheckBox("Fit every extracted curve", page)
        self.usePreviousCheckBox = QCheckBox("Use previous success as next initial guess", page)
        self.usePreviousCheckBox.setChecked(True)
        self.fullAutoFitCheckBox = QCheckBox("Run AI candidate generation", page)
        self.autoRefineCheckBox = QCheckBox("Refine candidate", page)
        self.profileCombo = self._combo(())
        self.fitInitializationCombo = self._combo(
            ("Previous success", "Recipe values", "AI each frame")
        )
        self.refinementCombo = self._combo(
            ("Plot only", "Every frame", "Every N frames", "On quality drop")
        )
        self.refineEverySpinBox = QSpinBox(page)
        self.refineEverySpinBox.setRange(1, 100_000)
        self.failurePolicyCombo = self._combo(
            ("Continue", "Stop")
        )
        self.autoFitCheckBox.setChecked(True)
        self.fullAutoFitCheckBox.setChecked(True)
        self.usePreviousCheckBox.setChecked(False)
        self.fitInitializationCombo.setCurrentText("AI each frame")
        self.workflowModeCombo = QComboBox(page)
        self.workflowModeCombo.addItems(("Fit curves · selected method", "Fit curves · legacy correction off", "Extract curves only"))
        self.workflowModeCombo.setToolTip(
            "Choose the fitting method in 1D parameters. Legacy correction off only disables "
            "the legacy V5 four-step correction. The single-RC specialist and its amplitude calibration "
            "use their own captured settings in both fitting modes."
        )
        self.failurePolicyCombo.setCurrentText("Continue")
        form = QFormLayout()
        form.addRow("Process", self.workflowModeCombo)
        form.addRow("On failure", self.failurePolicyCombo)
        layout.addLayout(form)
        label = QLabel("Uses the bundled 1D workflow and captured prediction parameters. Each q side keeps its own model and forward curve.", page)
        label.setWordWrap(True)
        layout.addWidget(label)
        self.predictionSettingsButton = QPushButton("1D parameters…", page)
        layout.addWidget(self.predictionSettingsButton)
        for widget in (self.autoFitCheckBox, self.usePreviousCheckBox, self.fullAutoFitCheckBox,
                       self.autoRefineCheckBox, self.profileCombo, self.fitInitializationCombo,
                       self.refinementCombo, self.refineEverySpinBox):
            widget.hide()
        self.workflowModeCombo.currentIndexChanged.connect(self._sync_workflow_mode)
        self._sync_workflow_mode(0)

    def _sync_workflow_mode(self, index):
        self.autoFitCheckBox.setChecked(index != 2)
        self.fullAutoFitCheckBox.setChecked(index != 2)
        self.usePreviousCheckBox.setChecked(False)
        self.autoRefineCheckBox.setChecked(False)
        self.fitInitializationCombo.setCurrentText("AI each frame")
        self.refinementCombo.setCurrentText("Plot only")

    def _build_results_page(self) -> None:
        page, layout = self._page(
            "results",
            "Results",
            "Inspect trends, heatmaps and the persistent session cache.",
        )
        self.changeScopeCombo = self._combo(
            ("Future frames", "Selected + future", "All frames (reprocess)")
        )
        self.applyRecipeButton = QPushButton("Create new Recipe version", page)
        self.applyRecipeButton.setObjectName("fittingInsituApplyPolicyButton")
        self.applyRecipeButton.setProperty("gimapPrimaryAction", True)
        form = QFormLayout()
        form.addRow("Apply edits to", self.changeScopeCombo)
        layout.addLayout(form)
        layout.addWidget(self.applyRecipeButton)
        self.trendButton = QPushButton("Open trend monitor", page)
        self.heatmapButton = QPushButton("Open cut heatmap", page)
        self.exportButton = QPushButton("Export results…", page)
        self.clearCacheButton = QPushButton("Clear session cache", page)
        self.openCacheButton = QPushButton("Open cache folder", page)
        for button in (
            self.trendButton,
            self.heatmapButton,
            self.exportButton,
            self.clearCacheButton,
            self.openCacheButton,
        ):
            layout.addWidget(button)
        layout.addStretch(1)

    @staticmethod
    def _combo(items: tuple[str, ...]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        return combo

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, decimals: int):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    @staticmethod
    def _range_spin() -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 100_000_000)
        spin.setSpecialValueText("Auto")
        return spin


__all__ = ["InSituWorkflowControls"]
