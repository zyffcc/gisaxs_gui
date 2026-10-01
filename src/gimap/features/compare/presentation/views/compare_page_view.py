"""Static layout of the Compare page (behaviour in ``presentation/page.py``).

A command bar — Add ▾, what is loaded, Save ▾ — then the steps Series → Compare → Results on the left
and three plots on the right: how far every series has changed frame by frame, their paths through
the two main changes, and the end state of each. A status line closes the page.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, CurvePlot, StepRail
from src.gimap.app.presentation.i18n import tr

COMPARE_STEPS = (("series", "Series"), ("compare", "Compare"), ("results", "Results"))
X_AXES = (("Frame", "frame"), ("Share of the series (0–1)", "share"))


def _muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    return label


def _card(parent: QWidget, name: str) -> tuple[QFrame, QLabel]:
    card = QFrame(parent)
    card.setProperty("gimapInfoCard", True)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 8, 10, 8)
    label = QLabel("", card)
    label.setObjectName(name)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(label)
    return card, label


def _table(columns, parent: QWidget, name: str, *, editable: bool = False) -> QTableWidget:
    table = QTableWidget(0, len(columns), parent)
    table.setObjectName(name)
    table.setHorizontalHeaderLabels([tr(column) for column in columns])  # headers: not reached by the translator
    table.verticalHeader().hide()
    if not editable:
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


class ComparePageView:
    def setup_ui(self, page: QWidget) -> None:
        page.setObjectName("comparePage")
        root = QVBoxLayout(page)
        root.setContentsMargins(12, 10, 12, 6)
        root.setSpacing(8)
        root.addWidget(self._command_bar(page))
        self.splitter = QSplitter(Qt.Horizontal, page)
        self.splitter.setObjectName("compareSplitter")
        self.splitter.setHandleWidth(8)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self._steps_panel(self.splitter))
        self.splitter.addWidget(self._plots_panel(self.splitter))
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([420, 900])
        root.addWidget(self.splitter, 1)
        row = QHBoxLayout()
        self.status_label = QLabel("", page)
        self.status_label.setObjectName("compareStatus")
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_progress = QProgressBar(page)
        self.status_progress.setObjectName("compareProgress")
        self.status_progress.setMaximumWidth(220)
        self.status_progress.setRange(0, 0)
        self.status_progress.hide()
        row.addWidget(self.status_label, 1)
        row.addWidget(self.status_progress)
        root.addLayout(row)

    def _command_bar(self, page: QWidget) -> QFrame:
        bar = QFrame(page)
        bar.setObjectName("compareCommandBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)
        self.add_button = QToolButton(bar)
        self.add_button.setObjectName("compareAdd")
        self.add_button.setText("Add Series")
        self.add_button.setToolTip("Add a series: Analyze's Series map, a folder of curve files, or curve files")
        self.add_button.setPopupMode(QToolButton.InstantPopup)
        self.add_button.setProperty("gimapPrimaryAction", True)
        menu = QMenu(self.add_button)
        self.add_map_action = menu.addAction("The Series Map of Analyze")
        self.add_map_action.setToolTip("The map built in Analyze ▸ Series (or use Send to Compare there)")
        self.add_folder_action = menu.addAction("Folder of Curves…")
        self.add_folder_action.setToolTip("Every curve file of a folder (q, I columns), in counting order: one series")
        self.add_files_action = menu.addAction("Curve Files…")
        self.add_files_action.setToolTip("Chosen curve files (q, I columns), in counting order: one series")
        self.add_button.setMenu(menu)
        self.chip = QLabel("Nothing to compare yet — add a series", bar)
        self.chip.setObjectName("compareChip")
        self.chip.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.save_button = QToolButton(bar)
        self.save_button.setObjectName("compareSave")
        self.save_button.setText("Save")
        self.save_button.setToolTip("Save the tables (CSV with a JSON record) or the plots")
        self.save_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.save_button)
        self.save_series_action = menu.addAction("Table of Every Series…")
        self.save_series_action.setToolTip("CSV: frames, odd frames, stages, how fast, group and differences of every series")
        self.save_frames_action = menu.addAction("Table of Every Frame…")
        self.save_frames_action.setToolTip("CSV: every frame of every series — odd and why, stage, place along the main changes")
        menu.addSeparator()
        self.save_change_action = menu.addAction("Change Plot…")
        self.save_paths_action = menu.addAction("Paths Plot…")
        self.save_end_action = menu.addAction("End States Plot…")
        self.save_button.setMenu(menu)
        self.save_button.hide()  # shown once there is a comparison
        row.addWidget(self.add_button)
        row.addWidget(self.chip, 1)
        row.addWidget(self.save_button)
        return bar

    # -- steps ------------------------------------------------------------------------------

    def _steps_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("compareProcessPanel")
        panel.setMinimumWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.step_rail = StepRail(COMPARE_STEPS, panel)
        self.step_rail.setObjectName("compareRail")
        layout.addWidget(self.step_rail)
        divider = QFrame(panel)
        divider.setFrameShape(QFrame.HLine)
        divider.setProperty("gimapDivider", True)
        layout.addWidget(divider)
        self.step_stack = QStackedWidget(panel)
        self.step_pages: dict[str, QWidget] = {}
        self.step_intro: dict[str, QLabel] = {}
        for key, title in COMPARE_STEPS:
            scroll = QScrollArea(self.step_stack)
            scroll.setObjectName(f"compareStep_{key}")
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
            getattr(self, f"_{key}_step")(content, page_layout)
            page_layout.addStretch(1)
            scroll.setWidget(content)
            self.step_stack.addWidget(scroll)
            self.step_pages[key] = scroll
            self.step_intro[key] = intro
        layout.addWidget(self.step_stack, 1)
        return panel

    def _series_step(self, parent: QWidget, layout: QVBoxLayout) -> None:
        self.series_table = _table(("Series", "Frames", "From"), parent, "compareSeriesTable", editable=True)
        self.series_table.setToolTip("Double-click a name to rename the series")
        self.series_table.setMinimumHeight(160)
        layout.addWidget(self.series_table)
        row = QHBoxLayout()
        self.remove_button = QPushButton("Remove Selected", parent)
        self.remove_button.setObjectName("compareRemove")
        self.remove_button.setToolTip("Take the selected series out of the comparison")
        self.clear_button = QPushButton("Remove All", parent)
        self.clear_button.setObjectName("compareClear")
        self.clear_button.setToolTip("Start again with no series")
        row.addWidget(self.remove_button)
        row.addWidget(self.clear_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.series_hint = _muted(
            "In Analyze, open a sample and build its map in the Series tab, then Send to Compare; repeat for every "
            "sample. Curve files (q and I columns, e.g. a Batch Export) can be added as a folder too.", parent)
        layout.addWidget(self.series_hint)

    def _compare_step(self, parent: QWidget, layout: QVBoxLayout) -> None:
        form = QFormLayout()
        form.setHorizontalSpacing(8)
        range_row = QHBoxLayout()
        self.q_low_spin = QDoubleSpinBox(parent)
        self.q_low_spin.setObjectName("compareQLow")
        self.q_high_spin = QDoubleSpinBox(parent)
        self.q_high_spin.setObjectName("compareQHigh")
        for spin in (self.q_low_spin, self.q_high_spin):
            spin.setDecimals(4)
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.01)
            spin.setKeyboardTracking(False)
            spin.setToolTip("The q range compared (Å⁻¹); leave out a noisy edge or a detector artefact")
        self.whole_range_button = QPushButton("Whole Range", parent)
        self.whole_range_button.setObjectName("compareWholeRange")
        self.whole_range_button.setToolTip("Compare every q the series share")
        range_row.addWidget(self.q_low_spin)
        range_row.addWidget(QLabel("–", parent))
        range_row.addWidget(self.q_high_spin)
        range_row.addWidget(self.whole_range_button)
        form.addRow("q range", range_row)
        self.end_spin = QSpinBox(parent)
        self.end_spin.setObjectName("compareEndFrames")
        self.end_spin.setRange(1, 1000)
        self.end_spin.setValue(10)
        self.end_spin.setPrefix("last ")
        self.end_spin.setSuffix(" frames")
        self.end_spin.setToolTip("The end state of a series: the mean of its last frames (odd frames left out)")
        form.addRow("End state", self.end_spin)
        layout.addLayout(form)
        self.shape_check = QCheckBox("Compare the shape only", parent)
        self.shape_check.setObjectName("compareShapeOnly")
        self.shape_check.setChecked(True)
        self.shape_check.setToolTip(
            "Each frame's mean level removed: a brighter beam, a thicker film or a longer exposure is not a new "
            "structure. Off: the overall intensity counts too")
        layout.addWidget(self.shape_check)
        self.method_label = _muted("", parent)
        self.method_label.setObjectName("compareMethod")
        layout.addWidget(self.method_label)

    def _results_step(self, parent: QWidget, layout: QVBoxLayout) -> None:
        card, self.summary_label = _card(parent, "compareSummary")
        layout.addWidget(card)
        self.results_table = _table(("Series", "Frames", "Odd", "Stages", "Half done", "90 % done", "Group"), parent,
                                    "compareResultsTable")
        self.results_table.setToolTip("Half / 90 % done: the frame by which half / 90 % of the series' change had happened")
        self.results_table.setMinimumHeight(140)
        layout.addWidget(self.results_table)
        row = QHBoxLayout()
        row.addWidget(_muted("How different (percent of intensity, shape):", parent), 1)
        self.distance_combo = QComboBox(parent)
        self.distance_combo.setObjectName("compareDistanceKind")
        self.distance_combo.addItem("At the end", "end")
        self.distance_combo.addItem("At the start", "start")
        self.distance_combo.setToolTip("Compare the series where they ended (the last frames) or where they began")
        row.addWidget(self.distance_combo)
        layout.addLayout(row)
        self.distance_table = _table((), parent, "compareDistanceTable")
        self.distance_table.setMinimumHeight(120)
        layout.addWidget(self.distance_table)
        self.details_section = AdvancedSection("Odd frames and stages of every series", "", parent)
        self.details_section.setObjectName("compareDetails")
        self.details_label = QLabel("", self.details_section)
        self.details_label.setObjectName("compareDetailsText")
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details_section.add_widget(self.details_label)
        layout.addWidget(self.details_section)

    # -- plots ------------------------------------------------------------------------------

    def _plots_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("comparePlotPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.change_plot = CurvePlot("How far each series has changed", panel, log_y=False, sides=False)
        self.change_plot.setObjectName("compareChangePlot")
        self.change_plot.log_check.hide()  # a component can be negative
        self.change_plot.set_labels("frame", "component 1")
        self.x_axis_combo = QComboBox(panel)
        self.x_axis_combo.setObjectName("compareXAxis")
        for text, key in X_AXES:
            self.x_axis_combo.addItem(text, key)
        self.x_axis_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.x_axis_combo.setToolTip("Frame number, or the share of each series (series of the same duration, "
                                     "recorded at different rates)")
        self.component_combo = QComboBox(panel)
        self.component_combo.setObjectName("compareComponent")
        self.component_combo.setToolTip("Which main change: component 1 explains the most")
        self.component_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.component_combo.setMinimumContentsLength(18)
        self.change_plot.header_layout.insertWidget(1, self.component_combo)
        self.change_plot.header_layout.insertWidget(1, self.x_axis_combo)
        layout.addWidget(self.change_plot, 3)
        lower = QHBoxLayout()
        self.paths_plot = CurvePlot("Paths through the two main changes", panel, log_y=False, sides=False)
        self.paths_plot.setObjectName("comparePathsPlot")
        self.paths_plot.log_check.hide()
        self.paths_plot.set_labels("component 1", "component 2")
        self.end_plot = CurvePlot("End states", panel, log_y=True)
        self.end_plot.setObjectName("compareEndPlot")
        self.end_plot.set_labels("q (Å⁻¹)", "Intensity")
        lower.addWidget(self.paths_plot, 1)
        lower.addWidget(self.end_plot, 1)
        layout.addLayout(lower, 2)
        return panel


__all__ = ["COMPARE_STEPS", "ComparePageView", "X_AXES"]
