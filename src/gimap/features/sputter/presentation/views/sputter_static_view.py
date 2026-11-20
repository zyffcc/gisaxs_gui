"""Static layout of Details ▸ Static Frames (after deposition) (behaviour in ``presentation/static_flow.py``).

The samples themselves are in the sample list beside the main screen (``sputter_samples_view.py``), filled by Add
Samples… of the input bar with After Deposition chosen (``sputter_main_view.py``), and Analyze runs those ticked. This
collapsed section holds the rest: a line that says what it is for; the folder the series go to (one folder for each
series in it); the mode; the geometry of these frames (X-ray energy, distance, beam centre, sample roll, pixel mask
with the masks of the package) with Package Values, which fills in the package's own; and the log. The job status is
the bar's (``static_status`` is ``analyze_status``).
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QWidget

from src.gimap.app.presentation.components import AdvancedSection, FilePicker, SegmentedControl

STATIC_TITLE = "Static Frames (after deposition)"
STATIC_TIP = ("Samples measured after their deposition (choose After Deposition in the bar and add them there): the "
              "folder their series go to, the mode and the geometry of their frames. Each series is shown on the main "
              "screen against the nominal thickness.")
STATIC_MODE_CHOICES = (("Quick", "quick", "The network's candidates, each drawn by the physical model, on a few "
                                          "frames of the sample; the best is kept."),
                       ("Refine", "refine", "Also refines the solution with a short fit."),
                       ("Full", "full", "A longer fit, on every usable frame of the sample."))
REFERENCE_TIP = ("Frames of another piece of the same substrate without metal, measured like the samples: they give "
                 "the substrate part of their images. An in-situ scan takes this from its own frames before the "
                 "deposition; a sample measured afterwards has none, so it needs another piece. None: the film is taken "
                 "to lie directly on Si.")
SERIES_TIP = ("The folder the series go to: each series of the list (the part of the names before the thickness, "
              "and the deposition method) is analysed into a folder of its own in it, named after the series. A "
              "series already there is added to.")
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
    """The section of static frames. ``number_edit`` and ``meta`` are the page view's; the input bar is built first
    (its job status is the one of static frames)."""

    def _static_section(self, parent: QWidget, number_edit, meta) -> AdvancedSection:
        section = AdvancedSection(STATIC_TITLE, STATIC_TIP, parent, expanded=False)
        section.setObjectName("sputterStaticSection")
        self.static_note = meta(section)  # what the package analyses, or why it cannot
        section.add_widget(self.static_note)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(4)
        self.static_series = FilePicker(section, placeholder="", browse_text="…", clearable=False)
        self.static_series.setObjectName("sputterStaticSeries")
        self.static_series.path_edit.setToolTip(SERIES_TIP)
        self.static_mode = SegmentedControl(section)
        self.static_mode.setObjectName("sputterStaticMode")
        for index, (label, key, tip) in enumerate(STATIC_MODE_CHOICES):
            self.static_mode.addItem(label, key)
            self.static_mode.setItemToolTip(index, tip)
        form.addRow("Results folder", self.static_series)
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

        self.static_status = self.analyze_status  # one job status for both kinds of analysis (the bar's)
        self.static_log = QPlainTextEdit(section)
        self.static_log.setReadOnly(True)
        self.static_log.setMaximumBlockCount(600)
        self.static_log.setMinimumHeight(120)
        self.static_log.setPlaceholderText("Messages of the analysis")
        self.static_log.setVisible(False)
        section.add_widget(self.static_log)
        self.static_section = section
        return section


__all__ = ["MASKS_BUTTON", "MASKS_TIP", "REFERENCE_TIP", "SERIES_TIP", "STATIC_MODE_CHOICES", "STATIC_TITLE",
           "SputterStaticView"]
