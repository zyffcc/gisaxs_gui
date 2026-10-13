"""Static layout of the Compare page (behaviour in ``presentation/page.py``).

A command bar — Add ▾, what is loaded, Save ▾ — then the steps Series → Compare → Results on the left
and three plots on the right: how far every series has changed frame by frame, their paths through
the two main changes, and the end state of each (an empty state with Add Series until there is a
comparison). A status line closes the page.
"""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt
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

from src.gimap.app.presentation.components import AdvancedSection, CurvePlot, EmptyState, StepRail
from src.gimap.app.presentation.components.table_copy import enable_table_copy
from src.gimap.app.presentation.i18n import tr

COMPARE_STEPS = (("series", "Series"), ("compare", "Compare"), ("results", "Results"))
X_AXES = (("Frame", "frame"), ("Share of the series (0–1)", "share"))
RESULT_COLUMNS = ("Series", "Group", "Frames", "Odd", "Stages", "Half done", "90 % done")
GROUP_COLUMN = 1
RANGE_LIMITS = (-100.0, 100.0)
"""The range spins' limits before a comparison; then the common grid of the series (q, χ, …)."""
RANGE_DECIMALS = 4
NAME_WIDTH = 160
"""Widest the name column of the results and distance tables starts (it can be dragged wider)."""
LEAST_NAME_WIDTH = 48
COLUMNS_PROPERTY = "compareColumns"
"""The English titles of a result table's columns (its headers are made from them, on one line or two)."""
EMPTY_TITLE = "Nothing to compare yet"
EMPTY_TEXT = ("Add the Series map of a sample from Analyze (Series ▸ Send to Compare), a folder of curve files or "
              "chosen curve files (or drop them here). Two or more series are compared with each other; one alone "
              "is described.")
CHANGE_TIP = ("How far each series has changed, frame by frame, along the chosen main change; × marks an odd frame "
              "(left out of the comparison) in the colour of its series")
PATHS_TIP = ("Each series through the first two main changes: ○ its first kept frame, ■ its last; odd frames are "
             "left out")
PATHS_EMPTY = "A second main change is needed to draw the paths."
"""The paths plot without paths: every series has one main change only."""


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


def _table(columns, parent: QWidget, name: str, *, editable: bool = False, fitted: bool = False) -> QTableWidget:
    """``fitted``: a result table — names elided in the middle, the columns fitted to the width
    (``fit_columns``, again when the table is resized) and as tall as its rows (``fit_to_rows``) instead
    of a fixed height with blank rows."""
    table = QTableWidget(0, len(columns), parent)
    table.setObjectName(name)
    table.setHorizontalHeaderLabels([tr(column) for column in columns])  # headers: not reached by the translator
    table.verticalHeader().hide()
    if not editable:
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    enable_table_copy(table)  # Ctrl+C and Copy Rows / Copy Table: the rows with the column names
    if fitted:
        table.setTextElideMode(Qt.ElideMiddle)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if columns:  # headers that may break onto two lines (an elided header would hide its two lines)
            table.setProperty(COLUMNS_PROPERTY, list(columns))
            table.horizontalHeader().setTextElideMode(Qt.ElideNone)
        table.horizontalScrollBar().rangeChanged.connect(lambda *_range: fit_to_rows(table))
        # A new font or style (Settings ▸ font size, the theme) changes the header's height: fit again,
        # or the last row would be cut off with no scroll bar to reach it.
        table.horizontalHeader().geometriesChanged.connect(lambda: fit_to_rows(table))
        table.viewport().installEventFilter(_RefitOnResize(table))  # the viewport: resized after the table
        fit_to_rows(table)
    return table


class _RefitOnResize(QObject):
    """A result table (the parent) made wider or narrower — the window, the splitter: its columns fitted
    again to the new width of its viewport."""

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Resize and event.size().width() != event.oldSize().width():
            table = self.parent()
            if table is not None and table.rowCount():
                fit_columns(table)
                fit_to_rows(table)
        return False


def fit_to_rows(table: QTableWidget) -> None:
    """As tall as the header and the rows (and the horizontal scroll bar when the columns overflow)."""
    height = table.horizontalHeader().sizeHint().height() + 2 * table.frameWidth() + 2
    height += sum(table.rowHeight(row) for row in range(table.rowCount()))
    bar = table.horizontalScrollBar()
    if bar.maximum() > bar.minimum():
        height += bar.sizeHint().height()
    table.setFixedHeight(height)


def two_lines(text: str) -> str:
    """A column title on two lines, broken at the space nearest its middle — never between a number and its
    “%”: “90 % done” → “90 %” over “done”. A single word stays as it is."""
    spaces = [at for at, char in enumerate(text) if char == " " and not text[at + 1:].startswith("%")]
    if not spaces:
        return text
    at = min(spaces, key=lambda index: max(index, len(text) - index - 1))
    return text[:at] + "\n" + text[at + 1:]


def _headers(table: QTableWidget, *, wrapped: bool) -> None:
    """The column titles (``COLUMNS_PROPERTY``) in the interface language, on one line or on two."""
    for column, key in enumerate(table.property(COLUMNS_PROPERTY) or ()):
        item = table.horizontalHeaderItem(column)
        text = two_lines(tr(key)) if wrapped else tr(key)
        if item is not None and item.text() != text:
            item.setText(text)


def fit_columns(table: QTableWidget, column: int = 0) -> None:
    """Every column in view, as far as the width allows: the titles on one line, or on two when one line
    does not fit (“Half / done”, “90 % / done”: the values under them are short); the name column
    ``column`` as wide as its names, at most ``NAME_WIDTH``, and narrower — never below its title — when the
    others need the room (the names are elided in the middle; the full name is the tooltip). The user can
    drag it wider; a narrow panel that still cannot hold every column scrolls."""
    header = table.horizontalHeader()
    header.setSectionResizeMode(column, QHeaderView.Interactive)

    def others() -> int:
        return sum(max(table.sizeHintForColumn(index), header.sectionSizeHint(index))
                   for index in range(table.columnCount()) if index != column and not table.isColumnHidden(index))

    least = max(header.sectionSizeHint(column), LEAST_NAME_WIDTH)
    wanted = min(NAME_WIDTH, max(table.sizeHintForColumn(column) + 8, least))
    room = table.viewport().width()
    _headers(table, wrapped=False)
    if table.property(COLUMNS_PROPERTY) and wanted + others() > room:
        _headers(table, wrapped=True)
    table.setColumnWidth(column, max(least, min(wanted, room - others())))
    # The stretched last column never gets narrower than it was when stretching began (Qt): begin again
    # from its width now, or a title that once needed one line would keep the table too wide.
    header.setStretchLastSection(False)
    header.resizeSections()
    header.setStretchLastSection(True)


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
        self.splitter.setSizes([440, 900])  # room for every column of the result table (``fit_columns``)
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
            "sample. Curve files (q and I columns, e.g. a Batch Export) can be added as a folder too, or dropped on "
            "this page.", parent)
        layout.addWidget(self.series_hint)

    def _compare_step(self, parent: QWidget, layout: QVBoxLayout) -> None:
        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        range_box = QVBoxLayout()
        range_box.setSpacing(4)
        range_row = QHBoxLayout()
        self.q_low_spin = QDoubleSpinBox(parent)
        self.q_low_spin.setObjectName("compareQLow")
        self.q_high_spin = QDoubleSpinBox(parent)
        self.q_high_spin.setObjectName("compareQHigh")
        for spin in (self.q_low_spin, self.q_high_spin):
            spin.setDecimals(RANGE_DECIMALS)
            spin.setRange(*RANGE_LIMITS)  # then the series' own range (page: ``_show_range``)
            spin.setSingleStep(0.01)
            spin.setKeyboardTracking(False)
            spin.setMaximumWidth(110)
            spin.setToolTip("The q range compared (Å⁻¹); leave out a noisy edge or a detector artefact")  # the page names the axis
        self.whole_range_button = QPushButton("Whole Range", parent)
        self.whole_range_button.setObjectName("compareWholeRange")
        self.whole_range_button.setToolTip("Compare every q the series share")
        range_row.addWidget(self.q_low_spin)
        range_row.addWidget(QLabel("–", parent))
        range_row.addWidget(self.q_high_spin)
        range_row.addStretch(1)
        range_box.addLayout(range_row)
        button_row = QHBoxLayout()
        button_row.addWidget(self.whole_range_button)
        button_row.addStretch(1)
        range_box.addLayout(button_row)
        self.range_label = QLabel("q range", parent)  # the axis of the series (“χ range”): set by the page
        self.range_label.setObjectName("compareRangeLabel")
        form.addRow(self.range_label, range_box)
        self.end_spin = QSpinBox(parent)
        self.end_spin.setObjectName("compareEndFrames")
        self.end_spin.setRange(1, 1000)
        self.end_spin.setValue(10)
        self.end_spin.setPrefix("last ")
        self.end_spin.setSuffix(" frames")
        self.end_spin.setMaximumWidth(160)
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
        self.results_table = _table(RESULT_COLUMNS, parent, "compareResultsTable", fitted=True)
        self.results_table.setToolTip("Half / 90 % done: the frame by which half / 90 % of the series' change had happened")
        self.results_table.setColumnHidden(GROUP_COLUMN, True)  # groups exist from three series on
        layout.addWidget(self.results_table)
        self.distance_row = QWidget(parent)  # hidden below two series
        self.distance_row.setObjectName("compareDistanceRow")
        row = QHBoxLayout(self.distance_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.distance_label = _muted("How different (percent of intensity, shape):", self.distance_row)
        self.distance_label.setObjectName("compareDistanceLabel")
        row.addWidget(self.distance_label, 1)
        self.distance_combo = QComboBox(self.distance_row)
        self.distance_combo.setObjectName("compareDistanceKind")
        self.distance_combo.addItem("At the end", "end")
        self.distance_combo.addItem("At the start", "start")
        self.distance_combo.setToolTip("Compare the series where they ended (the last frames) or where they began")
        row.addWidget(self.distance_combo)
        layout.addWidget(self.distance_row)
        self.distance_table = _table((), parent, "compareDistanceTable", fitted=True)
        self.distance_table.setProperty("gimapDataHeaders", True)  # its headers are the series' names
        layout.addWidget(self.distance_table)
        self.distance_row.hide()
        self.distance_table.hide()
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
        frame = QFrame(parent)
        frame.setObjectName("comparePlotPanel")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(8, 8, 8, 8)
        self.plot_stack = QStackedWidget(frame)
        self.plot_stack.setObjectName("comparePlotStack")
        self.empty_state = EmptyState(EMPTY_TITLE, EMPTY_TEXT, self.plot_stack, action_text="Add Series")
        self.empty_state.setObjectName("compareEmpty")
        self.empty_state.action_button.setToolTip("Add a series: Analyze's Series map, a folder of curve files, or curve files")
        empty_page = QWidget(self.plot_stack)  # the card centred, not stretched over the whole panel
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setContentsMargins(24, 24, 24, 24)
        centre = QHBoxLayout()
        centre.addStretch(1)
        centre.addWidget(self.empty_state, 4)
        centre.addStretch(1)
        empty_layout.addStretch(1)
        empty_layout.addLayout(centre)
        empty_layout.addStretch(2)
        self.empty_state.setMaximumWidth(520)
        self.empty_state.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.plot_stack.addWidget(empty_page)
        self.empty_page = empty_page
        panel = QWidget(self.plot_stack)
        self.plots_page = panel
        self.plot_stack.addWidget(panel)
        outer.addWidget(self.plot_stack)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.change_plot = CurvePlot("How far each series has changed", panel, log_y=False, sides=False)
        self.change_plot.setObjectName("compareChangePlot")
        self.change_plot.log_check.hide()  # a component can be negative
        self.change_plot.set_labels("frame", "component 1")
        self.change_plot.legend.setOffset((-10, -10))  # bottom right: every curve rises to the top right
        self.change_plot.setToolTip(CHANGE_TIP)
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
        self.paths_plot.setToolTip(PATHS_TIP)
        self.paths_plot.set_empty_text(PATHS_EMPTY)  # shown when every series has one main change only
        self.end_plot = CurvePlot("End states", panel, log_y=True)
        self.end_plot.setObjectName("compareEndPlot")
        self.end_plot.set_labels("q (Å⁻¹)", "Intensity")
        lower.addWidget(self.paths_plot, 1)
        lower.addWidget(self.end_plot, 1)
        layout.addLayout(lower, 2)
        self.x_axis_combo.setEnabled(False)  # until there is a comparison
        self.component_combo.setEnabled(False)
        return frame


__all__ = ["CHANGE_TIP", "COMPARE_STEPS", "ComparePageView", "GROUP_COLUMN", "PATHS_EMPTY", "PATHS_TIP", "RANGE_DECIMALS",
           "RANGE_LIMITS", "RESULT_COLUMNS", "X_AXES", "fit_columns", "fit_to_rows", "two_lines"]
