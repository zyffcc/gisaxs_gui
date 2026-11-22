"""Fit Several Curves Together (static layout; behaviour in ``single/joint_fit_dialog.py``).

Left: the curves (each its kind and geometry), the substrate, the parameters (shared or each curve's own), the method
with Run and Stop. Right: the plot of one curve with its model, the summary, Use in Model and Save Record….
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import CurvePlot, SafeWheelDoubleSpinBox

from .step_widgets import read_only_table


def _strong(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setProperty("gimapRole", "strong")
    return label


def _note(parent: QWidget, name: str) -> QLabel:
    label = QLabel("", parent)
    label.setObjectName(name)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _spin(parent: QWidget, name: str, low: float, high: float, decimals: int, step: float, tip: str):
    box = SafeWheelDoubleSpinBox(parent)
    box.setObjectName(name)
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setToolTip(tip)
    return box


def build_joint_dialog(owner, parent: QWidget) -> QDialog:
    dialog = QDialog(parent)
    dialog.setObjectName("fitJointDialog")
    dialog.setWindowTitle("Fit Several Curves Together")
    dialog.resize(1080, 740)
    owner.joint_dialog = dialog
    outer = QVBoxLayout(dialog)
    splitter = QSplitter(Qt.Horizontal, dialog)
    outer.addWidget(splitter)

    left = QWidget(splitter)
    column = QVBoxLayout(left)
    column.setContentsMargins(0, 0, 6, 0)
    column.addWidget(_strong("Curves", left))
    owner.joint_curves = read_only_table(("Curve", "Kind", "Points", "Geometry"), left, "fitJointCurves")
    owner.joint_curves.setSelectionMode(QAbstractItemView.SingleSelection)
    column.addWidget(owner.joint_curves, 2)
    buttons = QHBoxLayout()
    owner.joint_add_files = QPushButton("Add Files…", left)
    owner.joint_add_files.setObjectName("fitJointAddFiles")
    owner.joint_add_files.setToolTip("Curve files: a horizontal cut, a vertical cut (Analyze writes its geometry in "
                                     "the file), or any other curve")
    owner.joint_add_current = QPushButton("Add Curve of the Page", left)
    owner.joint_add_current.setObjectName("fitJointAddCurrent")
    owner.joint_add_current.setToolTip("The curve open in Single analysis, with the angles of its Model")
    owner.joint_remove = QPushButton("Remove", left)
    owner.joint_remove.setObjectName("fitJointRemove")
    for button in (owner.joint_add_files, owner.joint_add_current, owner.joint_remove):
        buttons.addWidget(button)
    buttons.addStretch(1)
    column.addLayout(buttons)

    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    owner.joint_kind = QComboBox(left)
    owner.joint_kind.setObjectName("fitJointKind")
    owner.joint_kind.setToolTip("Horizontal cut: I(|q_y|) at one exit angle. Vertical cut: I(q_z) at one in-plane q, "
                                "in the four-term DWBA (needs the substrate). Other curve: at q_z = 0.")
    form.addRow("Kind", owner.joint_kind)
    owner.joint_alpha_i = _spin(left, "fitJointAlphaI", 0.0, 5.0, 4, 0.01, "Incidence angle of the frame (°)")
    owner.joint_alpha_f = _spin(left, "fitJointAlphaF", 0.0, 5.0, 4, 0.01,
                                "Exit angle of the horizontal cut's band (°); 0 with αi = 0: q_z = 0")
    owner.joint_wavelength = _spin(left, "fitJointWavelength", 0.005, 1.0, 5, 0.0001, "Wavelength (nm)")
    owner.joint_q_parallel = _spin(left, "fitJointQParallel", 0.0, 50.0, 4, 0.01,
                                   "In-plane q of the vertical cut's column band (nm⁻¹), on the sample horizon")
    owner.joint_critical = _spin(left, "fitJointCritical", 0.0, 2.0, 4, 0.01,
                                 "The film's critical angle when Analyze refracted the cut's q_z (°); 0: vacuum q_z")
    form.addRow("αi (°)", owner.joint_alpha_i)
    form.addRow("αf (°)", owner.joint_alpha_f)
    form.addRow("λ (nm)", owner.joint_wavelength)
    form.addRow("q∥ (nm⁻¹)", owner.joint_q_parallel)
    form.addRow("αc of q_z (°)", owner.joint_critical)
    substrate = QHBoxLayout()
    owner.joint_substrate = QComboBox(left)
    owner.joint_substrate.setObjectName("fitJointSubstrate")
    owner.joint_substrate.setToolTip("The substrate under the particles (all curves): a vertical cut needs it; a "
                                     "horizontal cut with both angles is then in the four-term DWBA as well")
    owner.joint_delta = QLineEdit(left)
    owner.joint_delta.setObjectName("fitJointDelta")
    owner.joint_beta = QLineEdit(left)
    owner.joint_beta.setObjectName("fitJointBeta")
    for field in (owner.joint_delta, owner.joint_beta):
        field.setValidator(QDoubleValidator(0.0, 1.0, 12, field))
        field.setMaximumWidth(84)
    owner.joint_delta_label, owner.joint_beta_label = QLabel("δ", left), QLabel("β", left)
    substrate.addWidget(owner.joint_substrate, 1)
    for widget in (owner.joint_delta_label, owner.joint_delta, owner.joint_beta_label, owner.joint_beta):
        substrate.addWidget(widget)
    form.addRow("Substrate", substrate)
    owner.joint_min_exit = _spin(left, "fitJointMinExit", 0.0, 1000.0, 2, 0.25,
                                 "Multiple of the substrate critical angle at each cut’s wavelength. "
                                 "Set to 0 to keep all valid exit angles.")
    owner.joint_min_exit.setSuffix(" αc")
    form.addRow("Lowest exit angle of the vertical cut", owner.joint_min_exit)
    column.addLayout(form)

    parameters = QHBoxLayout()
    parameters.addWidget(_strong("Parameters", left))
    parameters.addStretch()
    owner.joint_family_label = QLabel("Particle shape", left)
    parameters.addWidget(owner.joint_family_label)
    owner.joint_family = QComboBox(left)
    owner.joint_family.setObjectName("fitJointFamily")
    owner.joint_family.setToolTip("Changes the particle in Single analysis. Choose Vertical cylinder to fit its radius and height.")
    parameters.addWidget(owner.joint_family)
    column.addLayout(parameters)
    owner.joint_parameters = read_only_table(("Parameter", "Shared", "Value", "± 1σ"), left, "fitJointParameters")
    owner.joint_parameters.setEditTriggers(QAbstractItemView.NoEditTriggers)
    column.addWidget(owner.joint_parameters, 3)
    owner.joint_scales_note = QLabel("Each curve has its own intensity scale, background and beam tail. "
                                     "Upright cylinders share their height and height spread.", left)
    owner.joint_scales_note.setObjectName("fitJointScalesNote")
    owner.joint_scales_note.setWordWrap(True)
    owner.joint_scales_note.setProperty("gimapRole", "muted")
    column.addWidget(owner.joint_scales_note)
    run = QHBoxLayout()
    owner.joint_method = QComboBox(left)
    owner.joint_method.setObjectName("fitJointMethod")
    owner.joint_run = QPushButton("Fit Together", left)
    owner.joint_run.setObjectName("fitJointRun")
    owner.joint_run.setProperty("gimapPrimaryAction", True)
    owner.joint_stop = QPushButton("Stop", left)
    owner.joint_stop.setObjectName("fitJointStop")
    owner.joint_stop.hide()
    run.addWidget(owner.joint_method, 1)
    run.addWidget(owner.joint_run)
    run.addWidget(owner.joint_stop)
    column.addLayout(run)
    owner.joint_progress = QProgressBar(left)
    owner.joint_progress.setObjectName("fitJointProgress")
    owner.joint_progress.hide()
    column.addWidget(owner.joint_progress)
    owner.joint_status = _note(left, "fitJointStatus")
    column.addWidget(owner.joint_status)

    right = QWidget(splitter)
    column = QVBoxLayout(right)
    column.setContentsMargins(6, 0, 0, 0)
    shown = QHBoxLayout()
    shown.addWidget(QLabel("Show", right))
    owner.joint_shown = QComboBox(right)
    owner.joint_shown.setObjectName("fitJointShown")
    shown.addWidget(owner.joint_shown, 1)
    column.addLayout(shown)
    owner.joint_dimensions = _note(right, "fitJointDimensions")
    owner.joint_dimensions.setProperty("gimapRole", "strong")
    column.addWidget(owner.joint_dimensions)
    owner.joint_plot = CurvePlot("", right, log_y=True, sides=False)
    owner.joint_plot.setObjectName("fitJointPlot")
    column.addWidget(owner.joint_plot, 1)
    owner.joint_summary = _note(right, "fitJointSummary")
    column.addWidget(owner.joint_summary)
    actions = QHBoxLayout()
    owner.joint_use = QPushButton("Use in Model", right)
    owner.joint_use.setObjectName("fitJointUse")
    owner.joint_use.setToolTip("The fitted values of the first horizontal cut (shared ones included) into the Model "
                               "of Single analysis")
    owner.joint_save = QPushButton("Save Record…", right)
    owner.joint_save.setObjectName("fitJointSave")
    owner.joint_save.setToolTip("Every curve, its model and errors, the shared values, χ² and the covariance (JSON)")
    owner.joint_report = QPushButton("Save Report…", right)
    owner.joint_close = QPushButton("Close", right)
    owner.joint_close.setObjectName("fitJointClose")
    owner.joint_close.clicked.connect(dialog.close)
    actions.addWidget(owner.joint_use)
    actions.addWidget(owner.joint_save)
    actions.addWidget(owner.joint_report)
    actions.addStretch(1)
    actions.addWidget(owner.joint_close)
    column.addLayout(actions)
    splitter.setStretchFactor(0, 2)
    splitter.setStretchFactor(1, 3)
    return dialog


__all__ = ["build_joint_dialog"]
