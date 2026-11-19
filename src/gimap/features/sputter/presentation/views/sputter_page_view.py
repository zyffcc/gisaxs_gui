"""Static layout of the Sputter page (behaviour in ``presentation/page.py`` and its mixins).

From the top: the input bar (``sputter_main_view.py``: the scan folder, metal, substrate, deposition rate with its
unit — or, after the deposition, Add Samples… —, Analyze and its job status, the geometry in one line, Open Result…
and the switch of the view), then one of two views; with After Deposition chosen the sample list
(``sputter_samples_view.py``) stands left of them, in a splitter.

**The main screen** (``sputter_main_view.py``, shown first): one time window of a result — the measured image beside
the model image, the solutions to choose from, the value tiles and the evolution over the scan. While Analyze runs it
shows the live view instead (``sputter_live_view.py``): the steps and what each has found so far.

**Details**: left, from the top, Sample and geometry (X-rays, distance, incidence, beam centre, each with where its
value comes from); Analyze options (``sputter_main_view.py``); Static Frames (``sputter_static_view.py``, collapsed);
More options (collapsed): the pixel mask with the Masks of the Package, the sample roll and the folder results go to;
Storage (collapsed, ``sputter_steps_view.py``): the import caches of Analyze and their limit.
Right, about the result the main screen shows: Windows (one row per time window: the values of the solution shown),
Growth Process (the stages of the growth, its growth laws against the metal amount with the range of the same metal and
substrate, and the share of each physical term of the model per window), Report (the facts of the analysis, the time of each step with how the frames were read, a note when the disk was the
slow part, and the package's report), Logs (the logs of its steps) and Steps (the step tree of the run and what each
step found); each says in one line what to do while there is no result.
"""

from __future__ import annotations

from PyQt5.QtCore import QLocale, Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                             QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QTableWidget, QTabWidget,
                             QTextBrowser,
                             QVBoxLayout, QWidget)

from src.gimap.app.presentation.components import AdvancedSection, ErrorBanner, FilePicker, ParameterSection
from src.gimap.app.presentation.components.table_copy import enable_table_copy
from src.gimap.app.presentation.i18n import DATA_VALUES

from .sputter_live_view import SputterLiveView
from .sputter_main_view import SputterMainView
from .sputter_samples_view import LIST_WIDTH, SputterSamplesView
from .sputter_static_view import MASKS_BUTTON, MASKS_TIP, SputterStaticView
from .sputter_steps_view import TAB_STEPS

LEFT_WIDTH = 470
NO_RESULT = "Analyze a scan or open a result"
"""What each tab of Details says while the main screen shows no result."""
TAB_WINDOWS, TAB_REPORT, TAB_LOG, TAB_PROCESS = "Windows", "Report", "Logs", "Growth Process"
"""('Log' alone is the logarithmic scale elsewhere in the table of translations.)"""


def _number_edit(parent: QWidget, placeholder: str = "") -> QLineEdit:
    edit = QLineEdit(parent)
    validator = QDoubleValidator(edit)
    validator.setNotation(QDoubleValidator.StandardNotation)
    validator.setLocale(QLocale.c())  # a decimal point in every interface locale
    edit.setValidator(validator)
    edit.setPlaceholderText(placeholder)
    return edit


class _Line(QLabel):
    """A wrapped line as tall as its text at its width. In a form inside a scroll area Qt sizes a wrapped label for
    a guessed, narrower width: a one-line source would get an empty second line under it."""

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        super().setText(text)
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        """The text measured at the width it has (``QLabel.heightForWidth`` would count the height set before)."""
        if not self.text() or self.width() <= 1:
            if self.maximumHeight() != 16777215:
                self.setMinimumHeight(0)
                self.setMaximumHeight(16777215)
            return
        margins = self.contentsMargins()
        room = max(1, self.width() - margins.left() - margins.right() - 2 * self.margin())
        text = self.fontMetrics().boundingRect(0, 0, room, 100_000, int(Qt.AlignLeft | Qt.TextWordWrap), self.text())
        needed = text.height() + margins.top() + margins.bottom() + 2 * self.margin()
        if (self.minimumHeight(), self.maximumHeight()) != (needed, needed):
            self.setFixedHeight(needed)


def _meta(parent: QWidget) -> QLabel:
    label = _Line("", parent)
    label.setProperty("gimapMeta", True)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # right under its field when the form row is taller than one line
    return label


def _row(parent: QWidget, *widgets, stretch_last: bool = False) -> QWidget:
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if (stretch_last and index == len(widgets) - 1) else 0)
    return row


class SputterPageView(SputterMainView, SputterLiveView, SputterStaticView, SputterSamplesView):
    def setup_ui(self, page: QWidget) -> None:
        page.setObjectName("sputterPage")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)
        outer.addWidget(self._input_bar(page, _number_edit, _meta))
        self.view_stack = QStackedWidget(page)
        self.view_stack.setObjectName("sputterViews")
        self.view_stack.addWidget(self._main_view(self.view_stack))
        self.view_stack.addWidget(self._details_view(self.view_stack))
        # the sample list of After Deposition beside both views (hidden for an in-situ scan)
        self.body_splitter = QSplitter(Qt.Horizontal, page)
        self.body_splitter.setObjectName("sputterBody")
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.addWidget(self._samples_panel(self.body_splitter))
        self.body_splitter.addWidget(self.view_stack)
        self.body_splitter.setStretchFactor(0, 0)
        self.body_splitter.setStretchFactor(1, 1)
        self.body_splitter.setSizes([LIST_WIDTH, 4 * LIST_WIDTH])
        outer.addWidget(self.body_splitter, 1)

    def _details_view(self, parent: QWidget) -> QWidget:
        details = QWidget(parent)
        self.details_view = details
        outer = QHBoxLayout(details)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        scroll = QScrollArea(details)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(LEFT_WIDTH)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details_scroll = scroll
        left = QWidget()
        scroll.setWidget(left)
        column = QVBoxLayout(left)
        column.setSpacing(10)
        column.addWidget(self._geometry_section(left))
        column.addWidget(self._analysis_section(left, _number_edit, _meta))
        column.addWidget(self._static_section(left, _number_edit, _meta))
        column.addWidget(self._options_section(left))
        column.addWidget(self._storage_section(left))  # sputter_steps_view.py
        column.addStretch(1)
        outer.addWidget(scroll)
        outer.addWidget(self._tabs(details), 1)
        return details

    def _geometry_section(self, parent: QWidget) -> ParameterSection:
        section = ParameterSection("Sample and geometry", "", parent)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(4)
        self.energy_kind = QComboBox(section)
        self.energy_kind.addItem("Energy (keV)", "energy")
        self.energy_kind.addItem("Wavelength (nm)", "wavelength")
        self.energy_value = _number_edit(section, "required")
        self.sdd = _number_edit(section, "required")
        self.alpha = _number_edit(section, "required")
        self.beam_col = _number_edit(section, "auto")
        self.beam_row = _number_edit(section, "auto")
        self.beam_col.setToolTip("Beam centre in pixels as Analyze shows it (pixel corners on whole numbers, row 0 at "
                                 "the top). Empty: estimated from the frames.")
        self.beam_row.setToolTip(self.beam_col.toolTip())
        self.source_labels = {group: _meta(section) for group in ("xray", "sdd", "alpha", "beam", "roll")}
        form.addRow("X-rays", _row(section, self.energy_kind, self.energy_value, stretch_last=True))
        form.addRow("", self.source_labels["xray"])
        form.addRow("Distance (mm)", self.sdd)
        form.addRow("", self.source_labels["sdd"])
        form.addRow("Incidence αi (°)", self.alpha)
        form.addRow("", self.source_labels["alpha"])
        form.addRow("Beam centre (px)", _row(section, QLabel("col", section), self.beam_col, QLabel("row", section),
                                             self.beam_row))
        form.addRow("", self.source_labels["beam"])
        section.add_layout(form)
        self.preset_button = QPushButton("Save Geometry as Preset…", section)
        self.preset_button.setToolTip("Keep this geometry for every scan of this detector and flat field in a date "
                                      "range (a beamtime): such scans are filled in from it.")
        section.add_widget(_row(section, self.preset_button))
        self.geometry_section = section
        return section

    def _options_section(self, parent: QWidget) -> AdvancedSection:
        """What Analyze uses that is rarely changed: the pixel mask, the sample roll and where results go."""
        section = AdvancedSection("More options", "", parent, expanded=False)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(4)
        self.roll = _number_edit(section, "0")
        self.mask_picker = FilePicker(section, placeholder="optional detector mask (.npy)", browse_text="…")
        self.package_masks_button = QPushButton(MASKS_BUTTON, section)
        self.package_masks_button.setToolTip(MASKS_TIP)
        self.results_picker = FilePicker(section, placeholder="", browse_text="…")
        self.results_picker.setToolTip("Each scan and geometry gets its own folder in here; the same request shows the "
                                       "result it has.")
        form.addRow("Mask", self.mask_picker)
        form.addRow("", _row(section, self.package_masks_button))
        form.addRow("Sample roll (°)", self.roll)
        form.addRow("", self.source_labels["roll"])
        form.addRow("Save results in", self.results_picker)
        section.add_layout(form)
        self.options_section = section
        return section

    # -- the tabs about the result shown -------------------------------------------------------------------------
    def _tabs(self, page: QWidget) -> QTabWidget:
        self.tabs = QTabWidget(page)
        self.tabs.setObjectName("sputterDetailsTabs")
        self.windows_table = QTableWidget(0, 0)
        self.windows_table.setObjectName("sputterWindowsTable")
        self.windows_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.windows_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.windows_table.setSortingEnabled(False)
        self.windows_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.windows_table.verticalHeader().setVisible(False)
        enable_table_copy(self.windows_table)
        self.tab_windows = self._empty_or(self.windows_table)
        report = QWidget()
        layout = QVBoxLayout(report)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        self.report_facts = QLabel("", report)
        self.report_facts.setObjectName("sputterReportFacts")
        self.report_facts.setWordWrap(True)
        self.report_facts.setTextFormat(Qt.RichText)
        self.report_facts.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.report_steps = QLabel("", report)  # Time per step: a small table (the page fills it)
        self.report_steps.setObjectName("sputterReportSteps")
        self.report_steps.setTextFormat(Qt.RichText)
        self.report_steps.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.report_disk_note = ErrorBanner("", "", report, level="warning", dismissible=False)
        self.report_disk_note.setObjectName("sputterReportDiskNote")
        self.report_disk_note.title_label.setVisible(False)
        self.report_disk_note.setVisible(False)  # only when reading the frames from disk was the slow part
        self.report_view = QTextBrowser(report)
        self.report_view.setObjectName("sputterReport")
        self.report_view.setOpenExternalLinks(True)
        layout.addWidget(self.report_facts)
        layout.addWidget(self.report_steps)
        layout.addWidget(self.report_disk_note)
        layout.addWidget(self.report_view, 1)
        self.tab_report = self._empty_or(report)
        logs = QWidget()
        layout = QVBoxLayout(logs)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        self.log_choice = QComboBox(logs)
        self.log_choice.setObjectName("sputterLogChoice")
        self.log_choice.setProperty(DATA_VALUES, True)  # file names
        self.log_choice.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.log_view = QPlainTextEdit(logs)
        self.log_view.setObjectName("sputterLogView")
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.log_choice, 0, Qt.AlignLeft)
        layout.addWidget(self.log_view, 1)
        self.tab_log = self._empty_or(logs)
        self.tab_steps = self._empty_or(self._steps_tab())  # sputter_steps_view.py
        self.process_view = QTextBrowser()  # the growth process of the result (process_tab.py fills it)
        self.process_view.setObjectName("sputterProcess")
        self.tab_process = self._empty_or(self.process_view)
        self.tabs.addTab(self.tab_windows, TAB_WINDOWS)
        self.tabs.addTab(self.tab_process, TAB_PROCESS)
        self.tabs.addTab(self.tab_report, TAB_REPORT)
        self.tabs.addTab(self.tab_log, TAB_LOG)
        self.tabs.addTab(self.tab_steps, TAB_STEPS)
        return self.tabs

    @staticmethod
    def _empty_or(content: QWidget) -> QStackedWidget:
        """A tab: page 0 the one line said without a result, page 1 ``content``."""
        stack = QStackedWidget()
        empty = QLabel(NO_RESULT, stack)
        empty.setAlignment(Qt.AlignCenter)
        empty.setProperty("gimapRole", "muted")
        stack.addWidget(empty)
        stack.addWidget(content)
        return stack


__all__ = ["LEFT_WIDTH", "NO_RESULT", "SputterPageView", "TAB_LOG", "TAB_PROCESS", "TAB_REPORT", "TAB_WINDOWS"]
