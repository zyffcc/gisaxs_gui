"""Static layout of the Sputter page (behaviour in ``presentation/page.py`` and its mixins).

Left, from the top: the title; Data (the scan folder and one line about it); Sample and geometry (metal,
X-rays, distance, incidence, beam centre, each with where its value comes from); the Predict button with its
job status and the log under Details; a line about the model's Python and PyTorch; the result in one line;
More options (collapsed): polymer, sample roll, mask, time window, deposition rate and t0, the results folder,
Open Results Folder…, Add Refined Windows… and the explanation shown. Right: Size Trend, Literature (added by the
page), Structure Sketch, Windows and Reliability.
"""

from __future__ import annotations

from PyQt5.QtCore import QLocale, Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTabWidget,
                             QTextBrowser, QVBoxLayout, QWidget)

from src.gimap.app.presentation.components import AdvancedSection, FilePicker, JobStatus, ParameterSection
from src.gimap.app.presentation.components.table_copy import enable_table_copy

LEFT_WIDTH = 470
METAL_CHOICES = (("Ag", "Ag"), ("Au", "Au"), ("Cu — approximated with the Ag model, reference only", "Cu"))
CU_NOTE = ("Cu is read with the Ag model (the networks were trained for Ag and Au): its sizes are for reference "
           "only.")
POLYMERS = ("PSBMA", "PS", "PEO", "polymer")
TITLE = ("<b>Sputter series</b><br>Particle size R and spacing D over time, the metal thickness where it can be "
         "stated (island-volume equivalent; nominal, from a deposition rate), and how far each value can be trusted.")
RESULT_EMPTY = "No result yet: choose a scan folder and press Predict."
REPORT_EMPTY = ("<p>Choose a scan folder and press Predict, or open a results folder (More options). Each R and D "
                "value gets a grade: <b>Lit</b> literature method (R from the nominal thickness and 2π/q*; the default "
                "with a deposition rate), <b>P</b> network estimate on the physical path (the default without a rate), "
                "<b>A</b> refined (unique), <b>B</b> refined (an equally good alternative exists), <b>C</b> network "
                "estimate only.</p>")


def _number_edit(parent: QWidget, placeholder: str = "") -> QLineEdit:
    edit = QLineEdit(parent)
    validator = QDoubleValidator(edit)
    validator.setNotation(QDoubleValidator.StandardNotation)
    validator.setLocale(QLocale.c())  # a decimal point in every interface locale
    edit.setValidator(validator)
    edit.setPlaceholderText(placeholder)
    return edit


def _narrow(combo: QComboBox, characters: int = 12) -> QComboBox:
    """A combo box that fits the column however long its items are (the list shows them in full)."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(characters)
    combo.view().setMinimumWidth(combo.view().sizeHintForColumn(0))
    return combo


def _meta(parent: QWidget) -> QLabel:
    label = QLabel("", parent)
    label.setProperty("gimapMeta", True)
    label.setWordWrap(True)
    return label


def _row(parent: QWidget, *widgets, stretch_last: bool = False) -> QWidget:
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if (stretch_last and index == len(widgets) - 1) else 0)
    return row


class SputterPageView:
    def setup_ui(self, page: QWidget) -> None:
        page.setObjectName("sputterPage")
        outer = QHBoxLayout(page)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(LEFT_WIDTH)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left = QWidget()
        scroll.setWidget(left)
        column = QVBoxLayout(left)
        column.setSpacing(10)
        title = QLabel(TITLE, left)
        title.setWordWrap(True)
        column.addWidget(title)
        column.addWidget(self._data_section(left))
        column.addWidget(self._geometry_section(left))
        column.addWidget(self._predict_block(left))
        column.addWidget(self._options_section(left))
        column.addStretch(1)
        outer.addWidget(scroll)
        outer.addWidget(self._tabs(page), 1)

    def _data_section(self, parent: QWidget) -> ParameterSection:
        section = ParameterSection("Data", "", parent)
        self.scan_picker = FilePicker(section, placeholder="Folder with the CBF frames of one scan",
                                      browse_text="Choose…", clearable=False)
        self.scan_picker.setObjectName("sputterScanFolder")
        self.scan_summary = _meta(section)
        self.scan_summary.setText("Choose the folder of a scan: its first frame tells the detector and the time.")
        section.add_widget(self.scan_picker)
        section.add_widget(self.scan_summary)
        return section

    def _geometry_section(self, parent: QWidget) -> ParameterSection:
        section = ParameterSection("Sample and geometry", "", parent)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(4)
        self.metal = QComboBox(section)
        for label, key in METAL_CHOICES:
            self.metal.addItem(label, key)
        _narrow(self.metal)
        self.cu_note = _meta(section)
        self.cu_note.setText(CU_NOTE)
        self.cu_note.setVisible(False)
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
        form.addRow("Metal", self.metal)
        form.addRow("", self.cu_note)
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
        return section

    def _predict_block(self, parent: QWidget) -> QWidget:
        block = QWidget(parent)
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.predict_button = QPushButton("Predict", block)
        self.predict_button.setObjectName("sputterPredict")
        self.predict_button.setProperty("gimapPrimaryAction", True)
        self.predict_button.setMinimumHeight(36)
        self.predict_button.setToolTip("Read the frames and predict the structure of every time window. The first "
                                       "time, PyTorch is installed into GIMaP's Python before (about 1.2 GB).")
        self.job_status = JobStatus(block)
        self.job_status.set_actions_visible(pause=False, cancel=True, details=True)
        self.log = QPlainTextEdit(block)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(600)
        self.log.setMinimumHeight(120)
        self.log.setPlaceholderText("Messages of the model and of the installation")
        self.log.setVisible(False)
        self.environment_label = _meta(block)
        self.environment_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_label = QLabel(RESULT_EMPTY, block)
        self.result_label.setWordWrap(True)
        self.result_label.setTextFormat(Qt.RichText)
        for widget in (self.predict_button, self.job_status, self.log, self.environment_label, self.result_label):
            layout.addWidget(widget)
        return block

    def _options_section(self, parent: QWidget) -> AdvancedSection:
        section = AdvancedSection("More options", "", parent, expanded=False)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(4)
        self.polymer = QComboBox(section)
        self.polymer.setEditable(True)
        self.polymer.addItems(POLYMERS)
        _narrow(self.polymer, 8)
        self.roll = _number_edit(section, "0")
        self.mask_picker = FilePicker(section, placeholder="optional detector mask (.npy)", browse_text="…")
        self.bin_seconds = QDoubleSpinBox(section)
        self.bin_seconds.setRange(0.05, 60.0)
        self.bin_seconds.setDecimals(2)
        self.bin_seconds.setSingleStep(0.1)
        self.bin_seconds.setValue(0.5)
        self.rate = _number_edit(section, "optional")
        self.rate.setToolTip("Deposition rate (for example from a quartz balance): the nominal thickness "
                             "rate × (t − t0) is drawn with the metal thickness, and the literature method and the "
                             "Literature tab use it. Changing it later does not read the frames again.")
        self.t0 = _number_edit(section, "optional")
        self.t0.setToolTip("Time (s) of the scan at which the deposition started (the nominal thickness is 0 there).")
        self.results_picker = FilePicker(section, placeholder="", browse_text="…")
        self.results_picker.setToolTip("Each scan and geometry gets its own folder in here; the same request reuses it.")
        form.addRow("Polymer", self.polymer)
        form.addRow("Sample roll (°)", self.roll)
        form.addRow("", self.source_labels["roll"])
        form.addRow("Mask", self.mask_picker)
        form.addRow("Time window (s)", self.bin_seconds)
        form.addRow("Deposition rate (nm/s)", self.rate)
        form.addRow("Start of deposition t0 (s)", self.t0)
        form.addRow("Save results in", self.results_picker)
        section.add_layout(form)
        self.open_button = QPushButton("Open Results Folder…", section)
        self.merge_button = QPushButton("Add Refined Windows…", section)
        self.merge_button.setToolTip("Merge physical-refinement windows (one JSON per window, e.g. from the cluster) "
                                     "into the current result.")
        self.merge_button.setEnabled(False)
        section.add_widget(_row(section, self.open_button, self.merge_button))
        self.choice_box = _narrow(QComboBox(section))
        self.choice_box.setEnabled(False)
        self.choice_box.setToolTip(
            "Which explanation of the windows is shown.\n"
            "Literature method (the default with a deposition rate): R from the nominal thickness and D = 2π/q* by mass "
            "balance, as in the P03 in-situ GISAXS studies (grade Lit).\n"
            "Physical path (network; the default without a rate): one path of stages through time, network sizes in "
            "every island window, no switching between sources (grade P).\n"
            "Per window: highest network weight, or the converged refinement (grades A/B/C).\n"
            "Learned selector: trained on simulated growth series (grade L; its tested accuracy is on the Reliability "
            "tab).\n"
            "Time-consistent path (exploratory): registered physical rules; not more accurate in simulation (grades "
            "S/T).")
        explanation = QFormLayout()
        explanation.addRow("Explanation", self.choice_box)
        section.add_layout(explanation)
        self.options_section = section
        return section

    def _tabs(self, page: QWidget) -> QTabWidget:
        self.tabs = QTabWidget(page)
        trend = QWidget()
        layout = QVBoxLayout(trend)
        layout.setContentsMargins(4, 4, 4, 4)
        self.log_scale = QCheckBox("Logarithmic size axis", trend)
        self.log_scale.setChecked(True)
        layout.addWidget(self.log_scale)
        self.trend_layout = layout
        self.tabs.addTab(trend, "Size Trend")
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        enable_table_copy(self.table)
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(False)
        self.report.setHtml(REPORT_EMPTY)
        return self.tabs


__all__ = ["CU_NOTE", "METAL_CHOICES", "REPORT_EMPTY", "RESULT_EMPTY", "SputterPageView", "TITLE"]
