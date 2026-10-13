"""Static layout of the In-situ series page of Fitting (behaviour in ``single/series_page.py``).

A command bar — the folder, Start, Pause, Stop, Save — then the steps Curves → Start → Results
on the left, and on the right the selected frame with its fit over the trend of a parameter
through the series. A status line with a progress bar closes the page.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, CurvePlot, StepRail

from .fit_steps_view import info_card, muted, read_only_table

SERIES_STEPS = (("curves", "Curves"), ("start", "Start"), ("results", "Results"))
TREND_EMPTY = "The trend of the chosen value appears here after Start"
"""Shown over the trend plot while it has no points."""


class FitSeriesView:
    """Command bar, then steps | frame and trend, then the status line."""

    def setup_ui(self, page: QWidget) -> None:
        page.setObjectName("fitSeriesPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(12, 10, 12, 6)
        root.setSpacing(8)
        root.addWidget(self._command_bar(page))
        self.splitter = QSplitter(Qt.Horizontal, page)
        self.splitter.setObjectName("fitSeriesSplitter")
        self.splitter.setHandleWidth(8)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self._steps_panel(self.splitter))
        self.splitter.addWidget(self._plots_panel(self.splitter))
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([400, 900])
        root.addWidget(self.splitter, 1)
        row = QHBoxLayout()
        self.status_label = QLabel("", page)
        self.status_label.setObjectName("fitSeriesStatus")
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_progress = QProgressBar(page)
        self.status_progress.setObjectName("fitSeriesProgress")
        self.status_progress.setMaximumWidth(260)
        self.status_progress.hide()
        row.addWidget(self.status_label, 1)
        row.addWidget(self.status_progress)
        root.addLayout(row)

    def _command_bar(self, page: QWidget) -> QFrame:
        bar = QFrame(page)
        bar.setObjectName("fitCommandBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)
        self.folder_button = QPushButton("Choose Folder…", bar)
        self.folder_button.setObjectName("fitSeriesFolderButton")
        self.folder_button.setToolTip("The folder of the curves, e.g. the gimap_analysis folder Analyze writes")
        self.folder_chip = QLabel("No folder yet — choose one, or Send Series to Fitting in Analyze", bar)
        self.folder_chip.setObjectName("fitCurveChip")
        self.folder_chip.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.folder_chip.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.start_button = QPushButton("Start", bar)
        self.start_button.setObjectName("fitSeriesStart")
        self.start_button.setProperty("gimapPrimaryAction", True)
        self.start_button.setToolTip("Fit every listed curve with the model of Single analysis")
        self.pause_button = QPushButton("Pause", bar)
        self.pause_button.setObjectName("fitSeriesPause")
        self.pause_button.setCheckable(True)
        self.pause_button.setToolTip("Pause after the frame being fitted; press again to go on")
        self.pause_button.hide()
        self.stop_button = QPushButton("Stop", bar)
        self.stop_button.setObjectName("fitSeriesStop")
        self.stop_button.setProperty("gimapDangerAction", True)
        self.stop_button.setToolTip("Stop after the frame being fitted; the frames done are kept")
        self.stop_button.hide()
        self.save_button = QToolButton(bar)
        self.save_button.setObjectName("fitSeriesSave")
        self.save_button.setText("Save")
        self.save_button.setToolTip("Save the table of every frame (CSV + JSON record), the trend or the frame's plot")
        self.save_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.save_button)
        self.save_table_action = menu.addAction("Table of Every Frame…")
        self.save_table_action.setToolTip("CSV: every frame's parameters with their errors and fit quality, and a JSON record")
        self.save_trend_action = menu.addAction("Trend Plot…")
        self.save_frame_action = menu.addAction("Selected Frame's Plot…")
        self.save_button.setMenu(menu)
        self.save_button.hide()  # shown once frames are fitted
        row.addWidget(self.folder_button)
        row.addWidget(self.folder_chip, 1)
        row.addWidget(self.start_button)
        row.addWidget(self.pause_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.save_button)
        return bar

    # -- steps ----------------------------------------------------------------------------

    def _steps_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("fitProcessPanel")
        panel.setMinimumWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.step_rail = StepRail(SERIES_STEPS, panel)
        self.step_rail.setObjectName("fitSeriesRail")
        layout.addWidget(self.step_rail)
        divider = QFrame(panel)
        divider.setFrameShape(QFrame.HLine)
        divider.setProperty("gimapDivider", True)
        layout.addWidget(divider)
        self.step_stack = QStackedWidget(panel)
        self.step_pages: dict[str, QWidget] = {}
        self.step_intro: dict[str, QLabel] = {}
        for key, title in SERIES_STEPS:
            scroll = QScrollArea(self.step_stack)
            scroll.setObjectName(f"fitSeriesStep_{key}")
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            content = QWidget(scroll)
            page_layout = QVBoxLayout(content)
            page_layout.setContentsMargins(4, 2, 6, 4)
            page_layout.setSpacing(8)
            heading = QLabel(title, content)
            heading.setProperty("gimapInspectorTitle", True)
            intro = QLabel("", content)
            intro.setProperty("gimapInspectorIntro", True)
            intro.setWordWrap(True)
            page_layout.addWidget(heading)
            page_layout.addWidget(intro)
            getattr(self, f"_series_{key}_step")(content, page_layout)
            scroll.setWidget(content)
            self.step_stack.addWidget(scroll)
            self.step_pages[key] = scroll
            self.step_intro[key] = intro
        layout.addWidget(self.step_stack, 1)
        self.step_rail.set_current("curves")
        return panel

    def _series_curves_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel("Files", page))
        self.pattern_edit = QLineEdit("*_fit_input.dat", page)
        self.pattern_edit.setObjectName("fitSeriesPattern")
        self.pattern_edit.setToolTip("Which files of the folder are curves (* stands for any text)")
        row.addWidget(self.pattern_edit, 1)
        layout.addLayout(row)
        self.subfolders_check = QCheckBox("Also in subfolders", page)
        self.subfolders_check.setObjectName("fitSeriesSubfolders")
        layout.addWidget(self.subfolders_check)
        frames = QHBoxLayout()
        self.first_spin = QSpinBox(page)
        self.last_spin = QSpinBox(page)
        self.every_spin = QSpinBox(page)
        for spin, name, tip in ((self.first_spin, "fitSeriesFirst", "First frame to fit"),
                                (self.last_spin, "fitSeriesLast", "Last frame to fit"),
                                (self.every_spin, "fitSeriesEvery", "Fit every n-th frame (a quick look at a long series)")):
            spin.setObjectName(name)
            spin.setRange(1, 1)  # the page sets the range once the curves are listed
            spin.setToolTip(tip)
            spin.setMinimumWidth(56)
            spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # not as wide as the largest number
        every = QLabel("Every", page)
        every.setToolTip(self.every_spin.toolTip())
        frames.addWidget(QLabel("Frames", page))
        frames.addWidget(self.first_spin, 1)
        frames.addWidget(QLabel("–", page))
        frames.addWidget(self.last_spin, 1)
        frames.addSpacing(6)
        frames.addWidget(every)
        frames.addWidget(self.every_spin, 1)
        layout.addLayout(frames)
        self.watch_check = QCheckBox("Watch for new curves", page)
        self.watch_check.setObjectName("fitSeriesWatch")
        self.watch_check.setToolTip(
            "While the series runs, fit curves that appear in the folder (e.g. written by Analyze during a measurement)"
        )
        layout.addWidget(self.watch_check)
        self.stages_label = muted("", page)
        self.stages_label.setObjectName("fitSeriesStages")
        layout.addWidget(self.stages_label)
        self.skip_odd_check = QCheckBox("Leave out the odd frames", page)
        self.skip_odd_check.setObjectName("fitSeriesSkipOdd")
        self.skip_odd_check.setChecked(True)
        self.skip_odd_check.setToolTip(
            "Frames that match neither the frames before nor after them (a detector glitch, a shutter) are not fitted")
        self.skip_odd_check.hide()  # shown once odd frames are found
        layout.addWidget(self.skip_odd_check)
        self.frame_list = QListWidget(page)
        self.frame_list.setObjectName("fitSeriesFrames")
        self.frame_list.setMinimumHeight(180)
        layout.addWidget(self.frame_list, 1)
        layout.addWidget(muted("Select a frame to see it and its fit on the right.", page))

    def _series_start_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        self.model_card, self.model_summary = info_card(page)
        self.model_card.setObjectName("fitSeriesModelCard")
        layout.addWidget(self.model_card)
        self.edit_model_button = QPushButton("Edit in Single Analysis", page)
        self.edit_model_button.setObjectName("fitSeriesEditModel")
        self.edit_model_button.setToolTip(
            "The series uses the model, halves, fitting range and left-out points of Single analysis: set them "
            "there on one representative curve"
        )
        layout.addWidget(self.edit_model_button, 0, Qt.AlignLeft)
        title = QLabel("Each frame starts from", page)
        title.setProperty("gimapRole", "strong")
        layout.addWidget(title)
        self.start_group = QButtonGroup(page)
        self.start_previous = QRadioButton("The previous frame's result", page)
        self.start_previous.setObjectName("fitSeriesStartPrevious")
        self.start_previous.setToolTip("Follows a changing sample smoothly; a frame that does not converge restarts from the model")
        self.start_same = QRadioButton("The model in Single analysis", page)
        self.start_same.setObjectName("fitSeriesStartSame")
        self.start_same.setToolTip("Every frame independent of the others")
        self.start_stages = QRadioButton("The previous result; the Single model at each new stage", page)
        self.start_stages.setObjectName("fitSeriesStartStages")
        self.start_stages.setToolTip(
            "Follows the sample within a stage, and starts afresh where the curves change course (Curves step)")
        for index, button in enumerate((self.start_previous, self.start_same, self.start_stages)):
            self.start_group.addButton(button, index)
            layout.addWidget(button)
        self.start_previous.setChecked(True)
        method_title = QLabel("Method", page)
        method_title.setProperty("gimapRole", "strong")
        layout.addWidget(method_title)
        self.method_group = QButtonGroup(page)
        self.method_local = QRadioButton("Refine (fast)", page)
        self.method_local.setObjectName("fitSeriesRefine")
        self.method_local.setToolTip("Least squares from the start of each frame")
        self.method_global = QRadioButton("Search the ranges, then refine (slow)", page)
        self.method_global.setObjectName("fitSeriesSearch")
        self.method_global.setToolTip("For frames that change a lot from one to the next")
        for index, button in enumerate((self.method_local, self.method_global)):
            self.method_group.addButton(button, index)
            layout.addWidget(button)
        self.method_local.setChecked(True)
        layout.addStretch(1)

    def _series_results_step(self, page: QWidget, layout: QVBoxLayout) -> None:
        self.results_card, self.results_summary = info_card(page)
        layout.addWidget(self.results_card)
        self.results_table = read_only_table(("#", "χ²ᵣ", "parameter"), page, "fitSeriesTable",
                                             QAbstractItemView.ExtendedSelection)  # rows to copy; the current one is shown
        self.results_table.setProperty("gimapDataHeaders", True)  # parameter names with units
        self.results_table.setMinimumHeight(200)
        layout.addWidget(self.results_table, 1)
        self.to_single_button = QPushButton("Open Frame in Single Analysis", page)
        self.to_single_button.setObjectName("fitSeriesToSingle")
        self.to_single_button.setToolTip("The selected frame's curve and fitted model in Single analysis, to look closer")
        self.to_single_button.hide()
        layout.addWidget(self.to_single_button, 0, Qt.AlignLeft)
        log = AdvancedSection("Run Log", "", page)
        self.log_view = QPlainTextEdit(log)
        self.log_view.setObjectName("fitSeriesLog")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(120)
        log.add_widget(self.log_view)
        layout.addWidget(log)

    # -- plots ------------------------------------------------------------------------------

    def _plots_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("fitPlotPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.frame_plot = CurvePlot("", panel, log_y=True, log_x=False)
        self.frame_plot.setObjectName("fitSeriesFramePlot")
        self.frame_plot.set_labels("|q| (nm⁻¹)", "Intensity")
        self.frame_plot.set_empty_text("The selected frame and its fit appear here.")
        layout.addWidget(self.frame_plot, 3)
        self.trend_plot = CurvePlot("", panel, log_y=False, log_x=None)
        self.trend_plot.setObjectName("fitSeriesTrendPlot")
        self.trend_plot.set_labels("frame", "")
        self.trend_plot.set_empty_text(TREND_EMPTY)
        self.trend_combo = QComboBox(self.trend_plot)
        self.trend_combo.setObjectName("fitSeriesTrendParameter")
        self.trend_combo.setToolTip("Which parameter to follow through the series")
        self.trend_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.trend_plot.header_layout.insertWidget(0, self.trend_combo)
        layout.addWidget(self.trend_plot, 2)
        return panel


__all__ = ["FitSeriesView", "SERIES_STEPS", "TREND_EMPTY"]
