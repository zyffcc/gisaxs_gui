"""Static layout of the Sputter page's main screen and of its input bar (behaviour in ``presentation/client_flow.py``
and ``client_panel.py``).

**The input bar** (always shown, above the two views): the switch In Situ | After Deposition and its inputs — an
in-situ scan: the scan folder, Metal, Substrate, the deposition rate with its unit (Å/s or nm/s) and its QCM reading
box; samples measured after
their deposition: Add Samples… (a folder of samples, one sample's folder or one CBF file: ``static_list.py``), a line
that sums up the sample list and the switch that shows or hides that list — and the primary button Analyze;
one line about the scan; one line with the geometry used and Edit; Open Result…, Export… (``export_flow.py``) and
the switch between the main
screen and Details; at the right the job status of Analyze (of both kinds).

**The main screen** shows one time window of a result: the window selector (previous, slider, next, its time and
nominal deposit) with the switch Patterns | Real Space; a host for the measured image beside the model image and a
host for the solution drawn in real space (one shown at a time); a host for the selector of the solution
(candidates and the refined one, each with its fit deviation); the value tiles — radius, height, distance, gap, size
distribution width, coverage, and for a porous substrate the island, cap and film shares and the two correlation
lengths: only the tiles of the quantities a result has are shown — and the fit deviation of the solution shown; a
host for the evolution over the scan. A series of static samples is shown the same way, one sample in the place of a
time window. Before there is a result an empty state says what to do; while Analyze runs, the live view
(``sputter_live_view.py``) is the third page of the same stack.

**Analyze options** (a section of Details): the mode, the parallel processes (Automatic or a number), the optional
onset and duration of the deposition and polymer film thickness, the analysis package and its folder, Open Report,
Open Folder and Export… (of the result shown), and the messages of the run (the job status's Details shows them).
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                             QPushButton, QSizePolicy, QSlider, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from src.gimap.app.presentation.components import (EmptyState, FilePicker, JobStatus, ParameterSection,
                                                   SafeWheelComboBox, SegmentedControl)
from src.gimap.app.presentation.i18n import DATA_VALUES


STATUS_WIDTH = 400
EMPTY_WIDTH = 720
SAMPLE_METALS = ("Ag", "Au", "Cu")
SUBSTRATE_CHOICES = (("Si", "Si"), ("TiO₂", "TiO2"), ("PS", "PS"), ("PET", "PET"), ("PSBMA", "PSBMA"), ("PNO", "PNO"),
                     ("PEO", "PEO"), ("Polymer", "Polymer"))
MODE_CHOICES = (("Auto", "auto", "Quick for Si and TiO₂, Refine for polymer substrates."),
                ("Quick", "quick", "The network's candidates, each drawn by the physical model: about a second per "
                                   "time window."),
                ("Refine", "refine", "Also refines the solution of every window (minutes per window on a polymer)."),
                ("Full", "full", "A longer refinement of every window and of the whole scan together: hours."))
PRIOR_CHOICES = (("Auto", "auto", "On where it passed a held-out test (Ag on PS), off for the other systems."),
                 ("On", "on", "Before the film forms, the whole-scan refinement keeps the size and the distance "
                              "near the growth laws found for the metal on this substrate."),
                 ("Off", "off", "The whole-scan refinement without the growth laws."))
VIEW_CHOICES = (("Main Screen", "main"), ("Details", "details"))
ADD_SAMPLES = "Add Samples…"
ADD_SAMPLES_TIP = ("A folder of samples (each folder in it with CBF frames is one sample), one sample's folder or one CBF "
                   "file. The nominal thickness, HiPIMS and the bare substrate are read from the folder names.")
KIND_CHOICES = (("In Situ", "insitu", "Frames recorded during the deposition: the deposition rate gives the metal "
                                      "deposited at each time."),
                ("After Deposition", "static", "Samples measured after their deposition: each one's nominal thickness "
                                               "takes the place of the rate, and every series is shown against it."))
"""What the bar analyses: an in-situ scan, or a sample measured after its deposition."""
IMAGE_CHOICES = (("Patterns", "patterns", "The measured image beside the model image of the solution shown."),
                 ("Real Space", "space", "The solution shown drawn in real space: a cut through a row of grains and "
                                         "the view from above, in nm."),
                 ("Components", "components", "The model image of the solution shown split into its physical terms "
                                              "(grains, film, substrate, …), and how each term's share changes."))
SCAN_FOLDER_TIP = ("The folder of one scan (its CBF frames, or a scan folder with them in one subfolder such as p2m/): "
                   "its first frame tells the detector, the exposure and the time.")
RATE_TIP = ("Deposition rate, in the unit chosen beside it: needed, the solutions depend on the metal deposited at "
            "each time.")
SUBSTRATE_TIP = ("Polymer: a polymer film of another or unknown kind, a block copolymer for instance. The model describes "
                 "every polymer as one film layer; which polymer it is changes the metal's sizes by a few percent.")
RATE_UNITS = (("Å/s", "A/s", 0.1), ("nm/s", "nm/s", 1.0))
"""(shown, stored, nm/s per unit) of the deposition rate's unit; a quartz balance shows Å/s, the first."""
RATE_UNIT_TIP = "Unit of the deposition rate: a quartz balance (QCM) shows Å/s; 1 nm/s = 10 Å/s."
QCM_TIP = ("Ticked: the number is the reading of the quartz balance (QCM), which the analysis converts to the rate at "
           "the sample. Not ticked: the rate at the sample itself.")
ANALYZE_TIP = ("Read the frames, find the solutions of every time window and draw each as an image beside the "
               "measured one.")
TILES = (("R_nm", "Radius R", "nm", "Radius of the grains."),
         ("H_nm", "Height H", "nm", "Height of the grains."),
         ("base_nm", "Continuous layer t_b", "nm", "Thickness t_b of the continuous metal layer under the grains (0: "
                                                   "separate islands)."),
         ("D_nm", "Distance D", "nm", "Centre-to-centre distance of neighbouring grains."),
         ("gap_nm", "Gap", "nm", "Gap between neighbouring grains: D − 2R."),
         ("width", "Size distribution width", "", "Width σ of the log-normal distribution of the grain radius."),
         ("coverage", "Coverage", "", "Share of the surface that the metal covers (0 to 1)."),
         ("f_islands", "Island share", "", "Share of the metal that is in resolvable islands. Early in the "
                                           "deposition it can be above 1."),
         ("cap_share", "Cap share", "", "Share of the optically active metal that is in the cap layer on the "
                                        "particles of the substrate (0 to 1)."),
         ("depth_nm", "Vertical correlation", "nm", "Vertical correlation length of the metal network in the pores."),
         ("xi_nm", "Lateral correlation", "nm", "Lateral correlation length of the metal network in the pores."),
         ("film_share", "Film share", "", "Share of the scattering of the layer that leaves through the metal film "
                                          "(0 to 1)."))
"""(quantity, caption, unit, tooltip) of the value tiles, in their order. A result shows the tiles of the quantities
it has: a flat substrate the first seven (the continuous layer only when the result lists it), a porous one the
radius (as a relative size: the page says so in its caption), the width and the last five."""
CONNECTED_TILES = {
    "R_nm": ("Relief radius R", "Radius of the grains that make up the relief of the joined film."),
    "H_nm": ("Film height H", "Height of the film's top above the substrate: continuous layer and relief."),
    "D_nm": ("Relief spacing D", "Spacing of the relief of the joined film (the centre distance of the grains it is "
                                 "made of): the lateral correlation length of its surface, not of separate particles."),
}
"""Captions and tooltips of the tiles while the solution shown is a joined film (a continuous layer, or grains joined
into a network across the picture): the grains are then its relief. The gap D − 2R is not shown then."""
CONNECTED_HIDDEN = ("gap_nm",)
DEVIATION_CAPTION = "Fit deviation (lower is better)"
DEVIATION_TIP = ("How far the model image is from the measured image, per cell and with an allowance for the model: "
                 "it compares the solutions of one measured image.")
ENTRY_HINT = "The number beside each solution is its fit deviation: lower is better."
FILM_TIP = "Thickness of the polymer film on the wafer. Empty: the analysis uses its own value for this polymer."
ONSET_TIP = "Time of the scan (s) at which the deposition started. Empty: found in the data."
DURATION_TIP = ("How long the deposition lasted after its start (s): the metal amount stays constant afterwards. "
                "Empty: to the end of the scan.")
STEPS_BUTTON = "Steps"
STEPS_TIP = "The steps of the analysis at work and what each has found (the live view)."
WORKERS_TIP = ("How many processes the analysis runs at once (Analyze and Static Frames). Automatic: one per core, two "
               "cores left free and at most 16, fewer when the free memory is short; worked out again when the "
               "analysis starts. More processes do not help when reading the frames from the disk is the slow part.")


class ShownComboBox(SafeWheelComboBox):
    """A combo box that says when it is shown or opened (``aboutToShow``): its choices can be worked out again."""

    aboutToShow = pyqtSignal()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self.aboutToShow.emit()

    def showPopup(self) -> None:  # noqa: N802 - Qt API
        self.aboutToShow.emit()
        super().showPopup()


class ElidedLabel(QLabel):
    """One line that never widens its window: text that does not fit ends in “…” and is whole in the tooltip.
    ``full_text`` is what was set."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)

    def set_full_text(self, text: str) -> None:
        self._full = str(text)
        self.setToolTip(self._full)
        self._fit()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(20, self.width() - 2))
        if shown != self.text():
            self.setText(shown)


def _row(parent: QWidget, *widgets, stretch_last: bool = False) -> QWidget:
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if (stretch_last and index == len(widgets) - 1) else 0)
    return row


def _caption(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setProperty("gimapRole", "caption")
    return label


class SputterMainView:
    """The input bar, the main screen and the Analyze options. ``_number_edit`` and ``_meta`` are the page view's."""

    def _input_bar(self, page: QWidget, number_edit, meta) -> QWidget:
        bar = QWidget(page)
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        inputs = QHBoxLayout()
        inputs.setSpacing(6)
        self.kind_switch = SegmentedControl(bar)  # in situ: a series during the deposition; after: one sample
        self.kind_switch.setObjectName("sputterKind")
        for index, (label, key, tip) in enumerate(KIND_CHOICES):
            self.kind_switch.addItem(label, key)
            self.kind_switch.setItemToolTip(index, tip)
        self.input_stack = QStackedWidget(bar)  # 0: an in-situ scan, 1: a sample measured after the deposition
        self.input_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        insitu, after = QWidget(self.input_stack), QWidget(self.input_stack)
        self.input_stack.addWidget(insitu)
        self.input_stack.addWidget(after)
        self._static_inputs(after, number_edit)
        self.scan_picker = FilePicker(insitu, placeholder="Folder with the CBF frames of one scan",
                                      browse_text="Choose…", clearable=False)
        self.scan_picker.setObjectName("sputterScanFolder")
        self.scan_picker.setToolTip(SCAN_FOLDER_TIP)
        self.scan_picker.path_edit.setToolTip(SCAN_FOLDER_TIP)
        self.sample_metal = SafeWheelComboBox(bar)  # the wheel scrolls the page: it never changes the sample
        self.sample_metal.setObjectName("sputterSampleMetal")
        self.sample_metal.setProperty(DATA_VALUES, True)  # element symbols
        for key in SAMPLE_METALS:
            self.sample_metal.addItem(key, key)
        self.substrate = SafeWheelComboBox(bar)
        self.substrate.setObjectName("sputterSubstrate")
        self.substrate.setProperty(DATA_VALUES, True)  # material names
        for label, key in SUBSTRATE_CHOICES:
            self.substrate.addItem(label, key)
        self.substrate.setToolTip(SUBSTRATE_TIP)
        self.sample_rate = number_edit(bar, "required")
        self.sample_rate.setObjectName("sputterSampleRate")
        self.sample_rate.setFixedWidth(84)
        self.sample_rate.setToolTip(RATE_TIP)
        self.rate_unit = SafeWheelComboBox(insitu)  # a quartz balance shows Å/s: the first, the default
        self.rate_unit.setObjectName("sputterRateUnit")
        self.rate_unit.setProperty(DATA_VALUES, True)  # units
        for shown, key, _scale in RATE_UNITS:
            self.rate_unit.addItem(shown, key)
        self.rate_unit.setToolTip(RATE_UNIT_TIP)
        self.qcm_check = QCheckBox("QCM reading", insitu)
        self.qcm_check.setToolTip(QCM_TIP)
        self.analyze_button = QPushButton("Analyze", bar)
        self.analyze_button.setObjectName("sputterAnalyze")
        self.analyze_button.setProperty("gimapPrimaryAction", True)
        self.analyze_button.setMinimumWidth(96)
        self.analyze_button.setToolTip(ANALYZE_TIP)
        # the bar fits a 1280 x 800 window with macOS's larger font too (tight spacing: a gap of 8 before a caption)
        row = QHBoxLayout(insitu)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.scan_picker, 1)
        for caption, widget in (("Metal", self.sample_metal), ("Substrate", self.substrate),
                                ("Rate", self.sample_rate)):
            label = QLabel(caption, insitu)
            label.setBuddy(widget)
            row.addSpacing(4)
            row.addWidget(label)
            row.addWidget(widget)
        row.addWidget(self.rate_unit)
        row.addWidget(self.qcm_check)
        inputs.addWidget(self.kind_switch)
        inputs.addSpacing(4)
        inputs.addWidget(self.input_stack, 1)
        inputs.addSpacing(4)
        inputs.addWidget(self.analyze_button)
        outer.addLayout(inputs)

        lower = QHBoxLayout()
        lower.setSpacing(12)
        lines = QVBoxLayout()
        lines.setSpacing(4)
        first = QHBoxLayout()
        first.setSpacing(6)
        self.geometry_summary = ElidedLabel(bar)
        self.geometry_summary.setObjectName("sputterGeometrySummary")
        self.geometry_summary.setProperty("gimapMeta", True)
        self.geometry_edit_button = QPushButton("Edit", bar)
        self.geometry_edit_button.setProperty("gimapRole", "link")
        self.geometry_edit_button.setCursor(Qt.PointingHandCursor)
        self.geometry_edit_button.setToolTip("Show the geometry in Details to enter or change it.")
        self.open_result_button = QPushButton("Open Result…", bar)
        self.open_result_button.setToolTip("Show the result in a folder analysed before.")
        self.export_button = QPushButton("Export…", bar)
        self.export_button.setObjectName("sputterExport")
        self.export_button.setEnabled(False)
        self.view_switch = SegmentedControl(bar)
        self.view_switch.setObjectName("sputterViewSwitch")
        for label, key in VIEW_CHOICES:
            self.view_switch.addItem(label, key)
        self.view_switch.setItemToolTip(1, "The geometry and the options of Analyze, and the windows, the report and "
                                           "the log of the result shown.")
        first.addWidget(self.geometry_summary, 1)
        first.addWidget(self.geometry_edit_button)
        first.addSpacing(12)
        first.addWidget(self.open_result_button)
        first.addWidget(self.export_button)
        first.addWidget(self.view_switch)
        lines.addLayout(first)
        self.scan_summary = meta(bar)  # one line about the scan once a folder is chosen
        self.scan_summary.setVisible(False)
        lines.addWidget(self.scan_summary)
        lines.addStretch(1)
        lower.addLayout(lines, 1)
        self.analyze_status = JobStatus(bar)
        self.analyze_status.setObjectName("sputterAnalyzeStatus")
        self.analyze_status.set_actions_visible(pause=False, cancel=True, details=True)
        self.analyze_status.setFixedWidth(STATUS_WIDTH)
        lower.addWidget(self.analyze_status, 0, Qt.AlignTop)
        outer.addLayout(lower)
        return bar

    def _static_inputs(self, page: QWidget, number_edit) -> None:
        """The inputs of samples measured after their deposition: Add Samples… (its menu: ``static_list.py``), one
        line about the sample list and the switch Sample List that shows or hides the list beside the main screen (the
        series folder, the mode and the geometry are in Details)."""
        self.static_add_button = QPushButton(ADD_SAMPLES, page)
        self.static_add_button.setObjectName("sputterStaticAdd")
        self.static_add_button.setToolTip(ADD_SAMPLES_TIP)
        self.static_summary = ElidedLabel(page)
        self.static_summary.setObjectName("sputterStaticSummary")
        self.static_summary.setProperty("gimapMeta", True)
        self.static_list_button = QToolButton(page)
        self.static_list_button.setObjectName("sputterStaticList")
        self.static_list_button.setText("Sample List")
        self.static_list_button.setCheckable(True)
        self.static_list_button.setChecked(True)
        self.static_list_button.setToolTip("Show or hide the list of samples beside the main screen.")
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.static_add_button)
        row.addWidget(self.static_summary, 1)
        row.addWidget(self.static_list_button)

    # -- the main screen -----------------------------------------------------------------------------------------
    def _main_view(self, page: QWidget) -> QWidget:
        self.result_stack = QStackedWidget(page)
        empty = QWidget(self.result_stack)  # the empty state in the middle, as wide as its sentence needs
        centred = QHBoxLayout(empty)
        centred.setContentsMargins(0, 0, 0, 0)
        self.client_empty = EmptyState(parent=empty)
        self.client_empty.setObjectName("sputterClientEmpty")
        self.client_empty.setMaximumWidth(EMPTY_WIDTH)
        self.client_empty.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        centred.addStretch(1)
        centred.addWidget(self.client_empty, 6, Qt.AlignVCenter)
        centred.addStretch(1)
        self.result_stack.addWidget(empty)
        self.result_stack.addWidget(self._result_view(self.result_stack))
        self.result_stack.addWidget(self._live_view(self.result_stack))  # while Analyze runs (sputter_live_view.py)
        return self.result_stack

    def _result_view(self, parent: QWidget) -> QWidget:
        view = QWidget(parent)
        self.client_view = view
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        selector = QHBoxLayout()
        selector.setSpacing(6)
        self.window_previous = QToolButton(view)
        self.window_previous.setArrowType(Qt.LeftArrow)
        self.window_previous.setToolTip("Previous time window")
        self.window_next = QToolButton(view)
        self.window_next.setArrowType(Qt.RightArrow)
        self.window_next.setToolTip("Next time window")
        self.window_slider = QSlider(Qt.Horizontal, view)
        self.window_slider.setObjectName("sputterWindowSlider")
        self.window_slider.setRange(0, 0)
        self.window_slider.setPageStep(1)
        self.window_slider.setTracking(True)
        self.window_slider.setToolTip("Time window shown")
        self.window_label = QLabel("", view)
        self.window_label.setObjectName("sputterWindowLabel")
        self.window_label.setProperty("gimapRole", "heading")
        self.window_label.setMinimumWidth(250)
        self.result_name = QLabel("", view)  # the scan, the metal and the model family: data, not translated
        self.result_name.setProperty("gimapRole", "muted")
        self.result_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        selector.addWidget(self.window_previous)
        selector.addWidget(self.window_slider, 1)
        selector.addWidget(self.window_next)
        selector.addSpacing(8)
        selector.addWidget(self.window_label)
        selector.addWidget(self.result_name)
        self.image_switch = SegmentedControl(view)  # what the picture shows: the patterns or the solution in real space
        self.image_switch.setObjectName("sputterImageSwitch")
        for index, (label, key, tip) in enumerate(IMAGE_CHOICES):
            self.image_switch.addItem(label, key)
            self.image_switch.setItemToolTip(index, tip)
        selector.addSpacing(8)
        selector.addWidget(self.image_switch)
        layout.addLayout(selector)
        self.run_note = ElidedLabel(view)  # while Analyze goes on after its preliminary result: what it does now
        self.run_note.setObjectName("sputterRunNote")
        self.run_note.setProperty("gimapRole", "info")
        self.run_note.setVisible(False)
        self.steps_button = QPushButton(STEPS_BUTTON, view)  # the steps of the run at work (its live view)
        self.steps_button.setObjectName("sputterRunSteps")
        self.steps_button.setProperty("gimapRole", "link")
        self.steps_button.setCursor(Qt.PointingHandCursor)
        self.steps_button.setToolTip(STEPS_TIP)
        self.steps_button.setVisible(False)
        note = QHBoxLayout()
        note.setSpacing(8)
        note.addWidget(self.run_note, 1)
        note.addWidget(self.steps_button)
        layout.addLayout(note)

        self.pattern_host = QStackedWidget(view)  # 0: the two images, 1: the real-space picture, 2: the model's terms
        patterns_page, space_page = QWidget(self.pattern_host), QWidget(self.pattern_host)
        components_page = QWidget(self.pattern_host)
        self.pattern_layout = QVBoxLayout(patterns_page)
        self.pattern_layout.setContentsMargins(0, 0, 0, 0)
        self.space_layout = QVBoxLayout(space_page)
        self.space_layout.setContentsMargins(0, 0, 0, 0)
        self.components_layout = QVBoxLayout(components_page)
        self.components_layout.setContentsMargins(0, 0, 0, 0)
        self.pattern_host.addWidget(patterns_page)
        self.pattern_host.addWidget(space_page)
        self.pattern_host.addWidget(components_page)
        layout.addWidget(self.pattern_host, 5)

        entries = QHBoxLayout()
        entries.setSpacing(8)
        self.entry_caption = QLabel("Solution shown", view)
        self.entry_caption.setProperty("gimapRole", "strong")
        self.entry_layout = QHBoxLayout()  # the page puts the selector of this window's entries here
        self.entry_layout.setContentsMargins(0, 0, 0, 0)
        self.default_button = QPushButton("Back to Default", view)
        self.default_button.setProperty("gimapRole", "link")
        self.default_button.setCursor(Qt.PointingHandCursor)
        self.default_button.setToolTip("Show again the solution the smooth evolution over the scan selects for this "
                                       "window.")
        self.default_button.setVisible(False)
        self.entry_hint = ElidedLabel(view)  # the page sets ENTRY_HINT (in its language); it takes the room left
        self.entry_hint.setProperty("gimapMeta", True)
        entries.addWidget(self.entry_caption)
        entries.addLayout(self.entry_layout)
        entries.addWidget(self.default_button)
        entries.addWidget(self.entry_hint, 1)
        layout.addLayout(entries)

        layout.addLayout(self._tiles(view))

        self.evolution_host = QWidget(view)
        self.evolution_layout = QVBoxLayout(self.evolution_host)
        self.evolution_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.evolution_host, 4)
        return view

    def _tiles(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.tiles: dict = {}
        self.tile_values: dict = {}
        self.tile_captions: dict = {}
        for key, caption, unit, tip in TILES:
            tile = QFrame(parent)
            tile.setObjectName(f"sputterTile_{key}")
            tile.setProperty("gimapSection", True)
            tile.setToolTip(tip)
            inner = QVBoxLayout(tile)
            inner.setContentsMargins(12, 8, 12, 8)
            inner.setSpacing(2)
            name = ElidedLabel(tile)  # the page sets the caption (in its language): a long one never widens the row
            name.setProperty("gimapRole", "caption")
            name.set_full_text(caption)
            name.setToolTip("")  # the tile's own tooltip explains the quantity
            inner.addWidget(name)
            self.tile_captions[key] = name
            value = QLabel("", tile)
            value.setProperty("gimapRole", "display")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            line = QHBoxLayout()
            line.setSpacing(4)
            line.addWidget(value, 0, Qt.AlignBottom)
            if unit:
                unit_label = QLabel(unit, tile)
                unit_label.setProperty("gimapRole", "muted")
                line.addWidget(unit_label, 0, Qt.AlignBottom)
            line.addStretch(1)
            inner.addLayout(line)
            tile.setVisible(False)
            row.addWidget(tile, 1)
            self.tiles[key], self.tile_values[key] = tile, value
        deviation = QVBoxLayout()
        deviation.setContentsMargins(8, 8, 0, 8)
        deviation.setSpacing(2)
        self.deviation_caption = _caption(DEVIATION_CAPTION, parent)
        self.deviation_caption.setToolTip(DEVIATION_TIP)
        self.deviation_value = QLabel("", parent)
        self.deviation_value.setProperty("gimapRole", "heading")
        self.deviation_value.setToolTip(DEVIATION_TIP)
        self.deviation_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        deviation.addWidget(self.deviation_caption)
        deviation.addWidget(self.deviation_value)
        deviation.addStretch(1)
        row.addLayout(deviation)
        return row

    # -- Details: the options of Analyze -------------------------------------------------------------------------
    def _analysis_section(self, parent: QWidget, number_edit, meta) -> ParameterSection:
        section = ParameterSection("Analyze options", "", parent)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(4)
        self.mode_box = SegmentedControl(section)
        self.mode_box.setObjectName("sputterMode")
        for index, (label, key, tip) in enumerate(MODE_CHOICES):
            self.mode_box.addItem(label, key)
            self.mode_box.setItemToolTip(index, tip)
        self.prior_box = SegmentedControl(section)
        self.prior_box.setObjectName("sputterPrior")
        for index, (label, key, tip) in enumerate(PRIOR_CHOICES):
            self.prior_box.addItem(label, key)
            self.prior_box.setItemToolTip(index, tip)
        self.onset = number_edit(section, "from the data")
        self.onset.setToolTip(ONSET_TIP)
        self.duration = number_edit(section, "end of the scan")
        self.duration.setToolTip(DURATION_TIP)
        self.film_thickness = number_edit(section, "optional")
        self.film_thickness.setToolTip(FILM_TIP)
        self.film_label = QLabel("Polymer film thickness (nm)", section)
        self.workers_box = ShownComboBox(section)  # the page fills it (Automatic with its number, then 1 … cores)
        self.workers_box.setObjectName("sputterWorkers")
        self.workers_box.setProperty(DATA_VALUES, True)  # numbers; the page says Automatic in its language
        self.workers_box.setSizeAdjustPolicy(SafeWheelComboBox.AdjustToContents)
        self.workers_box.setToolTip(WORKERS_TIP)
        self.package_label = meta(section)
        self.package_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.package_button = QPushButton("Choose Folder…", section)
        self.package_button.setToolTip("The folder of the analysis package (it holds code and models).")
        form.addRow("Mode", self.mode_box)
        form.addRow("Growth-law prior", self.prior_box)
        form.addRow("Parallel processes", self.workers_box)
        form.addRow("Start of deposition (s)", self.onset)
        form.addRow("Deposition duration (s)", self.duration)
        form.addRow(self.film_label, self.film_thickness)
        form.addRow("Analysis package", self.package_button)
        form.addRow("", self.package_label)
        section.add_layout(form)
        self.report_button = QPushButton("Open Report", section)
        self.report_button.setToolTip("The report of the result shown: every quantity with its range and where it "
                                      "comes from.")
        self.report_button.setEnabled(False)
        self.result_folder_button = QPushButton("Open Folder", section)
        self.result_folder_button.setToolTip("Open the folder of the result shown in the file manager.")
        self.result_folder_button.setEnabled(False)
        self.details_export_button = QPushButton("Export…", section)
        self.details_export_button.setEnabled(False)
        section.add_widget(_row(section, self.report_button, self.result_folder_button, self.details_export_button))
        self.client_log = QPlainTextEdit(section)
        self.client_log.setReadOnly(True)
        self.client_log.setMaximumBlockCount(600)
        self.client_log.setMinimumHeight(120)
        self.client_log.setPlaceholderText("Messages of the analysis")
        self.client_log.setVisible(False)
        section.add_widget(self.client_log)
        self.analysis_section = section
        return section


__all__ = ["CONNECTED_HIDDEN", "CONNECTED_TILES", "RATE_UNITS", "RATE_UNIT_TIP", "DEVIATION_CAPTION", "ENTRY_HINT", "ElidedLabel", "IMAGE_CHOICES", "KIND_CHOICES", "MODE_CHOICES", "PRIOR_CHOICES", "SAMPLE_METALS", "SUBSTRATE_CHOICES",
           "STEPS_BUTTON", "STEPS_TIP", "ShownComboBox", "SputterMainView", "TILES", "VIEW_CHOICES", "WORKERS_TIP"]
