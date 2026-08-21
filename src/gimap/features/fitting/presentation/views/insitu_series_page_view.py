"""Hand-maintained Python View for the Fitting In-situ series context."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import JobStatus, ResultTable

from .insitu_workflow_controls import InSituWorkflowControls


class InSituSeriesPageView:
    """Create controls and layout only; commands are owned by the page binding."""

    STEP_DEFINITIONS = (
        ("source", "1", "Source"),
        ("preprocess", "2", "Preprocess"),
        ("geometry", "3", "Geometry"),
        ("cut", "4", "Yoneda & cut"),
        ("fit", "5", "Fit"),
        ("results", "6", "Results"),
    )

    def setupUi(self, page: QWidget) -> None:  # noqa: N802 - Qt View convention
        page.setObjectName("fittingInsituSeriesPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(10)
        self._build_header(page, root)
        self._build_workflow_bar(page, root)

        self.mainSplitter = QSplitter(Qt.Horizontal, page)
        self.mainSplitter.setObjectName("fittingInsituMainSplitter")
        self.mainSplitter.setChildrenCollapsible(False)
        root.addWidget(self.mainSplitter, 1)
        self._build_parameter_column(page)
        self._build_work_area(page)

        self.mainSplitter.setStretchFactor(0, 0)
        self.mainSplitter.setStretchFactor(1, 1)
        self.mainSplitter.setSizes([390, 960])
        page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _build_header(self, page: QWidget, root: QVBoxLayout) -> None:
        row = QHBoxLayout()
        text = QVBoxLayout()
        self.titleLabel = QLabel("In-situ series", page)
        self.titleLabel.setObjectName("fittingInsituTitleLabel")
        self.titleLabel.setProperty("gimapPageTitle", True)
        self.subtitleLabel = QLabel(
            "One versioned Recipe drives live acquisition and existing sequences.", page
        )
        self.subtitleLabel.setObjectName("fittingInsituSubtitleLabel")
        self.subtitleLabel.setProperty("gimapMeta", True)
        self.subtitleLabel.setWordWrap(True)
        text.addWidget(self.titleLabel)
        text.addWidget(self.subtitleLabel)
        row.addLayout(text, 1)
        self.recipeStatusLabel = QLabel("No Recipe", page)
        self.recipeStatusLabel.setObjectName("fittingInsituRecipeStatusLabel")
        self.recipeStatusLabel.setProperty("statusKind", "warning")
        self.captureRecipeButton = QPushButton("Use current Single setup", page)
        self.captureRecipeButton.setObjectName("fittingInsituCaptureRecipeButton")
        self.captureRecipeButton.setProperty("gimapPrimaryAction", True)
        row.addWidget(self.recipeStatusLabel)
        row.addWidget(self.captureRecipeButton)
        self.backToSingleButton = QPushButton("Back to Single analysis", page)
        self.backToSingleButton.setObjectName("fittingInsituBackToSingleButton")
        self.backToSingleButton.hide()
        root.addLayout(row)
        self.recipeMetaLabel = QLabel(
            "Analyze one representative frame, then explicitly transfer its setup.", page
        )
        self.recipeMetaLabel.setObjectName("fittingInsituRecipeMetaLabel")
        self.recipeMetaLabel.setProperty("gimapMeta", True)
        self.recipeMetaLabel.setWordWrap(True)
        root.addWidget(self.recipeMetaLabel)

    def _build_workflow_bar(self, page: QWidget, root: QVBoxLayout) -> None:
        bar = QFrame(page)
        bar.setObjectName("fittingInsituWorkflowBar")
        bar.setProperty("insituWorkflowBar", True)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(5)
        self.workflowButtonGroup = QButtonGroup(page)
        self.workflowButtonGroup.setExclusive(True)
        self.workflowButtons = {}
        for index, (key, number, title) in enumerate(self.STEP_DEFINITIONS):
            button = QToolButton(bar)
            button.setObjectName(f"fittingInsituWorkflow{key.title()}Button")
            button.setText(f"{number}  {title}")
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.setProperty("insituWorkflowStep", True)
            button.setProperty("workflowState", "ready" if key == "source" else "pending")
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.workflowButtonGroup.addButton(button, index)
            self.workflowButtons[key] = button
            layout.addWidget(button, 1)
            if index < len(self.STEP_DEFINITIONS) - 1:
                arrow = QLabel("→", bar)
                arrow.setProperty("gimapMeta", True)
                layout.addWidget(arrow)
        self.workflowButtons["source"].setChecked(True)
        root.addWidget(bar)

    def _build_parameter_column(self, page: QWidget) -> None:
        self.settingsScrollArea = QScrollArea(self.mainSplitter)
        self.settingsScrollArea.setObjectName("fittingInsituSettingsScrollArea")
        self.settingsScrollArea.setFrameShape(QScrollArea.NoFrame)
        self.settingsScrollArea.setWidgetResizable(True)
        self.settingsScrollArea.setMinimumWidth(340)
        self.settingsScrollArea.setMaximumWidth(500)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        self.workflowControls = InSituWorkflowControls(page, content)
        content_layout.addWidget(self.workflowControls, 1)
        self.settingsScrollArea.setWidget(content)

    def _build_work_area(self, page: QWidget) -> None:
        work_area = QWidget(self.mainSplitter)
        layout = QVBoxLayout(work_area)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(8)
        self.previewTabs = QTabWidget(work_area)
        self.previewTabs.setObjectName("fittingInsituPreviewTabs")
        self.previewTabs.setDocumentMode(True)
        self._build_preview_tab()
        self._build_frames_tab()
        self._build_log_tab()
        layout.addWidget(self.previewTabs, 1)

        status_line = QHBoxLayout()
        self.currentImageLabel = QLabel("Current image: -", work_area)
        self.currentImageLabel.setObjectName("fittingInsituCurrentImageLabel")
        self.currentImageLabel.setProperty("gimapMeta", True)
        self.currentImageLabel.setWordWrap(True)
        status_line.addWidget(self.currentImageLabel, 1)
        self.summaryLabels = {}
        for key, title in (("processed", "Done"), ("failed", "Failed"), ("queue", "Queue")):
            label = QLabel(f"{title}: 0", work_area)
            label.setObjectName(f"fittingInsitu{key.title()}Summary")
            self.summaryLabels[key] = label
            status_line.addWidget(label)
        layout.addLayout(status_line)

        action_row = QHBoxLayout()
        self.startWatchButton = QPushButton("Start live watch", work_area)
        self.startWatchButton.setObjectName("fittingInsituStartWatchButton")
        self.startWatchButton.setProperty("gimapPrimaryAction", True)
        self.startProcessButton = QPushButton("Process sequence", work_area)
        self.startProcessButton.setObjectName("fittingInsituStartProcessButton")
        self.startProcessButton.setProperty("gimapPrimaryAction", True)
        self.pauseButton = QPushButton("Pause", work_area)
        self.pauseButton.setObjectName("fittingInsituPauseButton")
        self.stopButton = QPushButton("Stop", work_area)
        self.stopButton.setObjectName("fittingInsituStopButton")
        self.stopButton.setProperty("gimapDangerAction", True)
        action_row.addWidget(self.startWatchButton)
        action_row.addWidget(self.startProcessButton)
        action_row.addStretch(1)
        action_row.addWidget(self.pauseButton)
        action_row.addWidget(self.stopButton)
        layout.addLayout(action_row)
        self.jobStatus = JobStatus(work_area)
        self.jobStatus.setObjectName("fittingInsituJobStatus")
        self.jobStatus.set_actions_visible(pause=False, cancel=False, details=False)
        layout.addWidget(self.jobStatus)

        self.statusValueLabels = {
            "run_mode": QLabel("-", work_area),
            "status": QLabel("Idle", work_area),
            "file": self.currentImageLabel,
            "processed": self.summaryLabels["processed"],
            "failed": self.summaryLabels["failed"],
            "queue": self.summaryLabels["queue"],
            "fit": QLabel("-", work_area),
            "chi": QLabel("-", work_area),
            "cache": QLabel("-", work_area),
        }
        for key in ("run_mode", "status", "fit", "chi", "cache"):
            self.statusValueLabels[key].hide()
        self.workflowControls.applyRecipeButton.setEnabled(False)

    def _build_preview_tab(self) -> None:
        page = QWidget(self.previewTabs)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        self.imageCanvas = self._make_canvas_holder(page, "Detector / processed image")
        self.curveCanvas = self._make_canvas_holder(page, "Cut / fitting curve")
        splitter = QSplitter(Qt.Vertical, page)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.imageCanvas)
        splitter.addWidget(self.curveCanvas)
        splitter.setSizes([420, 320])
        layout.addWidget(splitter, 1)
        self.previewTabs.addTab(page, "Preview")

    def _build_frames_tab(self) -> None:
        page = QWidget(self.previewTabs)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        self.resultsTable = ResultTable(
            (
                "Frame",
                "File",
                "Load",
                "Preprocess",
                "Geometry",
                "Cut",
                "Fit",
                "Recipe",
                "Fit quality",
            ),
            page,
            empty_message="No processed frames yet",
        )
        layout.addWidget(self.resultsTable)
        self.previewTabs.addTab(page, "Frames")

    def _build_log_tab(self) -> None:
        page = QWidget(self.previewTabs)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        self.logBrowser = QTextBrowser(page)
        self.logBrowser.setObjectName("fittingInsituLogBrowser")
        layout.addWidget(self.logBrowser)
        self.previewTabs.addTab(page, "Log")

    @staticmethod
    def _make_canvas_holder(parent: QWidget, fallback_text: str) -> QWidget:
        holder = QWidget(parent)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            figure = Figure(figsize=(5.5, 3.1), dpi=80)
            canvas = FigureCanvasQTAgg(figure)
            holder._insitu_figure = figure
            holder._insitu_canvas = canvas
            layout.addWidget(canvas)
        except (ImportError, RuntimeError):
            label = QLabel(f"{fallback_text} preview unavailable", holder)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
        return holder


__all__ = ["InSituSeriesPageView"]
