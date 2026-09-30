"""Controls for the Fitting In-situ series page: a series of 1D curves from Analyze."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
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

from ...application import DEFAULT_CURVE_PATTERN

FIT_MODES = (
    "Fit curves · selected method",
    "Fit curves · legacy correction off",
    "Plot curves only",
)
PLOT_ONLY = 2


class InSituWorkflowControls(QWidget):
    """Own layout widgets only; commands remain in page/binding classes."""

    STEP_KEYS = ("source", "fit", "results")

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
        self._build_fit_page()
        self._build_results_page()

    def show_step(self, key: str) -> None:
        if key in self.STEP_KEYS:
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
            "Curves",
            "A folder of 1D curves written by Analyze (…_fit_input.dat): "
            "Export ▸ Batch Export… (Each frame: the input for Fitting), or Send to Fitting ▸ Send Series to Fitting. "
            "Watching also fits curves Analyze writes while it watches the detector.",
        )
        form = QFormLayout()
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.runModeCombo = QComboBox(page)
        self.runModeCombo.setObjectName("fittingInsituRunModeCombo")
        self.runModeCombo.addItems(("Process Existing Sequence", "Live Watch"))
        form.addRow("Mode", self.runModeCombo)

        folder_row = QWidget(page)
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        self.sequenceFolderEdit = QLineEdit(folder_row)
        self.sequenceFolderEdit.setObjectName("fittingInsituSequenceFolderEdit")
        self.sequenceFolderEdit.setPlaceholderText("Folder of curves, usually …/gimap_analysis")
        self.sequenceBrowseButton = QPushButton("…", folder_row)
        self.sequenceBrowseButton.setObjectName("fittingInsituSequenceBrowseButton")
        self.sequenceBrowseButton.setToolTip("Choose the folder of curves")
        folder_layout.addWidget(self.sequenceFolderEdit, 1)
        folder_layout.addWidget(self.sequenceBrowseButton)
        form.addRow("Folder", folder_row)
        self.sequencePatternEdit = QLineEdit(DEFAULT_CURVE_PATTERN, page)
        self.sequencePatternEdit.setObjectName("fittingInsituSequencePatternEdit")
        self.sequencePatternEdit.setToolTip("Which files are curves (q, I[, σ[, pixels]] columns)")
        form.addRow("Files", self.sequencePatternEdit)
        self.recursiveCheckBox = QCheckBox("Include child folders", page)
        self.recursiveCheckBox.setObjectName("fittingInsituRecursiveCheckBox")
        self.recursiveCheckBox.setChecked(False)
        form.addRow("", self.recursiveCheckBox)
        layout.addLayout(form)

        self.liveSettingsWidget = QWidget(page)
        live_form = QFormLayout(self.liveSettingsWidget)
        live_form.setContentsMargins(0, 0, 0, 0)
        self.pollSpinBox = self._double_spin(0.2, 3600.0, 2.0, 1)
        self.pollSpinBox.setSuffix(" s")
        self.stableCheckBox = QCheckBox("Wait until the file is completely written", page)
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
        self.sequenceStartSpinBox.setToolTip("First file number (the last number in the file name)")
        layout.addWidget(self.sequenceSettingsWidget)

        common = QFormLayout()
        self.uiEverySpinBox = QSpinBox(page)
        self.uiEverySpinBox.setRange(1, 100_000)
        self.uiEverySpinBox.setValue(5)
        self.uiEverySpinBox.setToolTip("Redraw the preview every N processed curves.")
        common.addRow("Preview every", self.uiEverySpinBox)
        layout.addLayout(common)
        layout.addStretch(1)

    def _build_fit_page(self) -> None:
        page, layout = self._page(
            "fit",
            "Fit",
            "Fit every curve with the 1D workflow captured from Single analysis. "
            "Save the settings before starting.",
        )
        self.workflowModeCombo = QComboBox(page)
        self.workflowModeCombo.addItems(FIT_MODES)
        self.workflowModeCombo.setToolTip(
            "Choose the fitting method in 1D parameters. Legacy correction off only disables "
            "the legacy V5 four-step correction. The single-RC specialist and its amplitude "
            "calibration use their own captured settings in both fitting modes."
        )
        self.failurePolicyCombo = self._combo(("Continue", "Stop"))
        form = QFormLayout()
        form.addRow("Process", self.workflowModeCombo)
        form.addRow("On failure", self.failurePolicyCombo)
        layout.addLayout(form)
        label = QLabel(
            "Uses the bundled 1D workflow and the captured prediction parameters. "
            "Each q side keeps its own model and forward curve.",
            page,
        )
        label.setWordWrap(True)
        label.setProperty("gimapMeta", True)
        layout.addWidget(label)
        self.predictionSettingsButton = QPushButton("1D parameters…", page)
        layout.addWidget(self.predictionSettingsButton)
        self.applyRecipeButton = QPushButton("Save analysis settings", page)
        self.applyRecipeButton.setObjectName("fittingInsituApplyPolicyButton")
        self.applyRecipeButton.setProperty("gimapPrimaryAction", True)
        layout.addWidget(self.applyRecipeButton)
        layout.addStretch(1)

    def _build_results_page(self) -> None:
        page, layout = self._page(
            "results",
            "Results",
            "Inspect trends, the curve heatmap and the persistent session cache.",
        )
        self.changeScopeCombo = self._combo(
            ("Future frames", "Selected + future", "All frames (reprocess)")
        )
        form = QFormLayout()
        form.addRow("Apply edits to", self.changeScopeCombo)
        layout.addLayout(form)
        self.trendButton = QPushButton("Open trend monitor", page)
        self.heatmapButton = QPushButton("Open curve heatmap", page)
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

    def fit_mode(self) -> int:
        return self.workflowModeCombo.currentIndex()

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


__all__ = ["FIT_MODES", "InSituWorkflowControls", "PLOT_ONLY"]
