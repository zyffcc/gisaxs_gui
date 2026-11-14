"""Static layout of Details ▸ Static Frames (after deposition) (behaviour in ``presentation/static_flow.py``).

A collapsed section for frames measured after the deposition, one sample at a time: a line that says what it is for;
the frames of the sample (a folder — Folder… — or one CBF file or a zip archive — File… —, the folder inside an
archive after ``::``); the sample's name and nominal thickness; optional frames of the bare substrate; the folder of
the series the sample goes to; the mode; the geometry of these frames (X-ray energy, distance, beam centre, sample
roll, pixel mask with the masks of the package) with Package Values, which fills in the package's own; the button
Analyze Sample with its job status; and the log.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QWidget

from src.gimap.app.presentation.components import AdvancedSection, FilePicker, JobStatus, SegmentedControl

STATIC_TITLE = "Static Frames (after deposition)"
STATIC_TIP = ("Frames of a sample measured after its deposition: each sample is analysed into a series, which the "
              "main screen shows against the nominal thickness.")
STATIC_MODE_CHOICES = (("Quick", "quick", "The network's candidates, each drawn by the physical model, on a few "
                                          "frames of the sample; the best is kept."),
                       ("Refine", "refine", "Also refines the solution with a short fit."),
                       ("Full", "full", "A longer fit, on every usable frame of the sample."))
FRAMES_TIP = ("The GISAXS frames of one sample: a folder (every CBF file in it), one CBF file, or a zip archive. For "
              "an archive, the folder inside it follows after two colons: archive.zip::folder/")
NAME_TIP = ("The name of the sample in the series: letters, digits, - and _ (anything else becomes _). Analysing a "
            "name again with other frames or values replaces that sample.")
NOMINAL_TIP = "The nominal thickness of the film (nm): the position of the sample on the axis of the series."
REFERENCE_TIP = ("Frames of the bare substrate, measured like the sample: they are used as the substrate part of its "
                 "image. Empty: the film is taken to lie directly on Si.")
SERIES_TIP = ("The folder of the series. Each sample analysed into it is added to its report and to what the main "
              "screen shows; choosing the folder of an existing series adds to that series.")
GEOMETRY_NOTE = "Geometry of these frames (an empty field leaves the value of the package)"
CENTRE_TIP = ("Beam centre in pixels as Analyze shows it (pixel corners on whole numbers, row 0 at the top). Empty: "
              "the value of the package.")
MASKS_BUTTON = "Masks of the Package"
MASKS_TIP = "The pixel masks the analysis package brings, by file name: choosing one fills the field."


def _row(parent: QWidget, *widgets, stretch_first: bool = False) -> QWidget:
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if (stretch_first and index == 0) else 0)
    return row


class SputterStaticView:
    """The section of static frames. ``number_edit`` and ``meta`` are the page view's."""

    def _static_section(self, parent: QWidget, number_edit, meta) -> AdvancedSection:
        section = AdvancedSection(STATIC_TITLE, STATIC_TIP, parent, expanded=False)
        section.setObjectName("sputterStaticSection")
        self.static_note = meta(section)  # what the package analyses, or why it cannot
        section.add_widget(self.static_note)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(4)
        self.static_frames = FilePicker(section, placeholder="folder, CBF file or zip archive",
                                        browse_text="Folder…", clearable=False)
        self.static_frames.setObjectName("sputterStaticFrames")
        self.static_frames.path_edit.setToolTip(FRAMES_TIP)
        self.static_frames_file = QPushButton("File…", section)
        self.static_frames_file.setToolTip("Choose one CBF file or a zip archive.")
        self.static_name = QLineEdit(section)
        self.static_name.setObjectName("sputterStaticName")
        self.static_name.setPlaceholderText("required")
        self.static_name.setToolTip(NAME_TIP)
        self.static_nominal = number_edit(section, "required")
        self.static_nominal.setToolTip(NOMINAL_TIP)
        self.static_reference = FilePicker(section, placeholder="optional", browse_text="Folder…", clearable=False)
        self.static_reference.path_edit.setToolTip(REFERENCE_TIP)
        self.static_reference_file = QPushButton("File…", section)
        self.static_reference_file.setToolTip(self.static_frames_file.toolTip())
        self.static_series = FilePicker(section, placeholder="", browse_text="…", clearable=False)
        self.static_series.setObjectName("sputterStaticSeries")
        self.static_series.path_edit.setToolTip(SERIES_TIP)
        self.static_mode = SegmentedControl(section)
        self.static_mode.setObjectName("sputterStaticMode")
        for index, (label, key, tip) in enumerate(STATIC_MODE_CHOICES):
            self.static_mode.addItem(label, key)
            self.static_mode.setItemToolTip(index, tip)
        form.addRow("Frames", _row(section, self.static_frames, self.static_frames_file, stretch_first=True))
        form.addRow("Sample name", self.static_name)
        form.addRow("Nominal thickness (nm)", self.static_nominal)
        form.addRow("Bare substrate", _row(section, self.static_reference, self.static_reference_file,
                                           stretch_first=True))
        form.addRow("Series folder", self.static_series)
        form.addRow("Mode", self.static_mode)
        section.add_layout(form)

        note = QLabel(GEOMETRY_NOTE, section)
        note.setProperty("gimapRole", "caption")
        note.setWordWrap(True)
        section.add_widget(note)
        geometry = QFormLayout()
        geometry.setLabelAlignment(Qt.AlignRight)
        geometry.setVerticalSpacing(4)
        self.static_energy = number_edit(section, "of the package")
        self.static_sdd = number_edit(section, "of the package")
        self.static_beam_col = number_edit(section, "of the package")
        self.static_beam_row = number_edit(section, "of the package")
        self.static_beam_col.setToolTip(CENTRE_TIP)
        self.static_beam_row.setToolTip(CENTRE_TIP)
        self.static_roll = number_edit(section, "of the package")
        self.static_mask = FilePicker(section, placeholder="of the package (.npy)", browse_text="…")
        self.static_masks_button = QPushButton(MASKS_BUTTON, section)
        self.static_masks_button.setToolTip(MASKS_TIP)
        self.static_defaults_button = QPushButton("Package Values", section)
        self.static_defaults_button.setProperty("gimapRole", "link")
        self.static_defaults_button.setCursor(Qt.PointingHandCursor)
        self.static_defaults_button.setToolTip("Fill the geometry fields with the values the analysis package uses "
                                               "for static frames.")
        geometry.addRow("X-ray energy (keV)", self.static_energy)
        geometry.addRow("Distance (mm)", self.static_sdd)
        geometry.addRow("Beam centre (px)", _row(section, QLabel("col", section), self.static_beam_col,
                                                 QLabel("row", section), self.static_beam_row))
        geometry.addRow("Sample roll (°)", self.static_roll)
        geometry.addRow("Mask", self.static_mask)
        geometry.addRow("", _row(section, self.static_masks_button, self.static_defaults_button))
        section.add_layout(geometry)

        self.static_button = QPushButton("Analyze Sample", section)  # Analyze, in the bar, is the primary action
        self.static_button.setObjectName("sputterStaticAnalyze")
        self.static_button.setMinimumHeight(32)
        self.static_button.setToolTip("Read the frames of this sample, find its solution and add it to the series; "
                                      "the main screen then shows the series.")
        section.add_widget(self.static_button)
        self.static_status = JobStatus(section)
        self.static_status.setObjectName("sputterStaticStatus")
        self.static_status.set_actions_visible(pause=False, cancel=True, details=True)
        section.add_widget(self.static_status)
        self.static_log = QPlainTextEdit(section)
        self.static_log.setReadOnly(True)
        self.static_log.setMaximumBlockCount(600)
        self.static_log.setMinimumHeight(120)
        self.static_log.setPlaceholderText("Messages of the analysis")
        self.static_log.setVisible(False)
        section.add_widget(self.static_log)
        self.static_section = section
        return section


__all__ = ["MASKS_BUTTON", "MASKS_TIP", "STATIC_MODE_CHOICES", "STATIC_TITLE", "SputterStaticView"]
