"""Rank Incidence Angles (static layout; behaviour in ``single/angle_rank_dialog.py``).

Top: the candidate angles and the exposure of each, what was measured so far, the model-error floor and the beam's
footprint; folded below them the particles buried in a film. Middle: the table of the candidates (gain, counts, the
relative error each parameter could reach). Bottom: the note and Record Choice.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, SafeWheelDoubleSpinBox

from .step_widgets import read_only_table

DEFAULT_ANGLES = "0.1, 0.12, 0.15, 0.2, 0.25, 0.4, 0.6"


def _spin(parent: QWidget, name: str, low: float, high: float, decimals: int, step: float, value: float, tip: str):
    box = SafeWheelDoubleSpinBox(parent)
    box.setObjectName(name)
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setToolTip(tip)
    return box


def build_angle_dialog(owner, parent: QWidget) -> QDialog:
    dialog = QDialog(parent)
    dialog.setObjectName("fitAngleDialog")
    dialog.setWindowTitle("Rank Incidence Angles")
    dialog.resize(860, 620)
    owner.angle_dialog = dialog
    layout = QVBoxLayout(dialog)
    owner.angle_intro = QLabel(
        "Which incidence angle to measure next: what an exposure at each angle would add to what the fit in Single "
        "analysis already knows (Physics and Methods 9.20). The score is only as good as the model.", dialog)
    owner.angle_intro.setWordWrap(True)
    owner.angle_intro.setProperty("gimapRole", "muted")
    layout.addWidget(owner.angle_intro)
    form = QFormLayout()
    owner.angle_list_edit = QLineEdit(DEFAULT_ANGLES, dialog)
    owner.angle_list_edit.setObjectName("fitAngleList")
    owner.angle_list_edit.setToolTip("The candidate incidence angles in degrees, separated by commas")
    owner.angle_exposure_spin = _spin(dialog, "fitAngleExposure", 0.001, 1e5, 3, 0.1, 1.0,
                                      "The exposure of one measurement at a candidate angle (s)")
    owner.angle_made_edit = QLineEdit("", dialog)
    owner.angle_made_edit.setObjectName("fitAngleMade")
    owner.angle_made_edit.setPlaceholderText("the curve's own angle and exposure")
    owner.angle_made_edit.setToolTip(
        "What was measured so far, as angle:exposure pairs (0.4:10, 0.1:1); empty: the angle and exposure of the "
        "curve fitted")
    owner.angle_curve_exposure_spin = _spin(dialog, "fitAngleCurveExposure", 0.001, 1e5, 3, 0.1, 1.0,
                                            "The exposure the fitted curve's intensity is per (s): its counts per pixel "
                                            "were collected in this time")
    owner.angle_floor_spin = _spin(dialog, "fitAngleFloor", 0.0, 50.0, 1, 0.5, 5.0,
                                   "The model-error floor f in % of each point: beyond 1/f² counts the model, not "
                                   "the photons, limits the error")
    owner.angle_beam_spin = _spin(dialog, "fitAngleBeam", 0.001, 10.0, 3, 0.01, 0.03,
                                  "The height of the beam at the sample (mm): the footprint is height / sin αi")
    owner.angle_length_spin = _spin(dialog, "fitAngleLength", 0.1, 500.0, 1, 1.0, 20.0,
                                    "The length of the sample along the beam (mm): the footprint ends there")
    footprint = QHBoxLayout()
    footprint.addWidget(owner.angle_beam_spin)
    footprint.addWidget(QLabel("mm beam on", dialog))
    footprint.addWidget(owner.angle_length_spin)
    footprint.addWidget(QLabel("mm sample", dialog))
    footprint.addStretch(1)
    form.addRow("Candidate angles (°)", owner.angle_list_edit)
    form.addRow("Exposure per angle (s)", owner.angle_exposure_spin)
    form.addRow("Measured so far", owner.angle_made_edit)
    form.addRow("Exposure of the curve (s)", owner.angle_curve_exposure_spin)
    form.addRow("Model-error floor (%)", owner.angle_floor_spin)
    form.addRow("Footprint", footprint)
    layout.addLayout(form)
    section = AdvancedSection(
        "Particles buried in a film",
        "A fraction of the particles lies in a film on the substrate (a cellulose film under a spray), fewer with "
        "depth as exp(−z/ζ): the angles below and above the film's critical angle tell how deep.",
        dialog,
    )
    section.setObjectName("fitAngleBurialSection")
    owner.angle_burial_check = QCheckBox("Some particles are buried in a film", section)
    owner.angle_burial_check.setObjectName("fitAngleBurial")
    owner.angle_film_delta_edit = QLineEdit("3.0e-6", section)
    owner.angle_film_delta_edit.setObjectName("fitAngleFilmDelta")
    owner.angle_film_delta_edit.setToolTip("δ of the film at the wavelength of the curve (cellulose at 10 keV: 3.0e-6)")
    owner.angle_film_beta_edit = QLineEdit("2.0e-9", section)
    owner.angle_film_beta_edit.setObjectName("fitAngleFilmBeta")
    owner.angle_film_beta_edit.setToolTip("β of the film at the wavelength of the curve")
    owner.angle_fraction_spin = _spin(section, "fitAngleFraction", 0.0, 100.0, 1, 5.0, 20.0,
                                      "The share of the particles in the film (%)")
    owner.angle_depth_spin = _spin(section, "fitAngleDepth", 0.1, 10000.0, 1, 5.0, 50.0,
                                   "ζ: how deep the buried particles lie, the depth where their number falls to 1/e (nm)")
    burial = QFormLayout()
    burial.addRow(owner.angle_burial_check)
    burial.addRow("Film δ", owner.angle_film_delta_edit)
    burial.addRow("Film β", owner.angle_film_beta_edit)
    burial.addRow("Buried share (%)", owner.angle_fraction_spin)
    burial.addRow("Depth ζ (nm)", owner.angle_depth_spin)
    section.add_layout(burial)
    layout.addWidget(section)
    buttons = QHBoxLayout()
    owner.angle_rank_button = QPushButton("Rank Angles", dialog)
    owner.angle_rank_button.setObjectName("fitAngleRank")
    owner.angle_rank_button.setProperty("gimapRole", "primary")
    owner.angle_rank_button.setToolTip("Score every candidate angle with the fit in Single analysis")
    buttons.addWidget(owner.angle_rank_button)
    buttons.addStretch(1)
    layout.addLayout(buttons)
    owner.angle_table = read_only_table(("Rank", "αi (°)", "Gain", "Counts"), dialog, "fitAngleTable")
    layout.addWidget(owner.angle_table, 1)
    owner.angle_status = QLabel("", dialog)
    owner.angle_status.setObjectName("fitAngleStatus")
    owner.angle_status.setWordWrap(True)
    owner.angle_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(owner.angle_status)
    record = QHBoxLayout()
    owner.angle_note_edit = QLineEdit("", dialog)
    owner.angle_note_edit.setObjectName("fitAngleNote")
    owner.angle_note_edit.setPlaceholderText("Note of the operator (optional)")
    owner.angle_record_button = QPushButton("Record Choice", dialog)
    owner.angle_record_button.setObjectName("fitAngleRecord")
    owner.angle_record_button.setToolTip(
        "Write the angle chosen in the table, every candidate's scores and the fit into angle_decisions.jsonl next to "
        "the curve (one line per choice)")
    owner.angle_record_button.setEnabled(False)
    record.addWidget(owner.angle_note_edit, 1)
    record.addWidget(owner.angle_record_button)
    layout.addLayout(record)
    return dialog


__all__ = ["DEFAULT_ANGLES", "build_angle_dialog"]
