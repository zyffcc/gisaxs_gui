"""The Fit step of the single-curve page (static layout; behaviour in ``single/run.py`` and ``single/predict.py``).

One choice and one button: the method — Predict parameters (GIMaP's network, then least squares), Refine (least
squares from Model) or Search (the ranges, then least squares); the page selects Predict for a GISAXS cut not fitted
yet and says why — what Predict needs right under it (the particles), then Fit. Parameters that stop at an edge of
their range are reported (Fit details), never widened. Everything changed rarely is in Options: the evaluation budget, many
curves.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget

from src.gimap.app.presentation.components import AdvancedSection

from .step_widgets import muted

METHODS = (
    ("predict", "Predict parameters",
     "The network proposes particles and sizes. Least squares refines the values. Evidence compares the solutions, "
     "with BIC weights if sampling fails. Refine afterwards to use the limits in Model."),
    ("local", "Refine the values in Model",
     "Least squares from the values in Model, with Fitting's own model: fast; finds the nearest good fit."),
    ("global", "Search the ranges, then refine",
     "Tries values across the min–max range of every free parameter first; slower, for a poor start."),
)
PREDICT_PARTICLES_TIP = ("The particles in Model uses the families in radius order, with their paracrystals. "
                         "For a choice of particles, the network proposes compositions. Evidence or BIC weights rank the fits.")


def build_fit_step(owner, page: QWidget, layout) -> None:
    row = QHBoxLayout()
    row.addWidget(QLabel("Method", page))
    owner.method_combo = QComboBox(page)
    owner.method_combo.setObjectName("fitMethod")
    for key, title, description in METHODS:
        owner.method_combo.addItem(title, key)
        owner.method_combo.setItemData(owner.method_combo.count() - 1, description, Qt.ToolTipRole)
    owner.method_combo.setCurrentIndex(owner.method_combo.findData("local"))  # the page recommends another per curve
    owner.method_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    owner.method_combo.setMinimumContentsLength(14)
    row.addWidget(owner.method_combo, 1)
    layout.addLayout(row)
    owner.method_hint = muted("", page)  # only why a method is recommended, or that the network is missing
    owner.method_hint.setObjectName("fitMethodHint")
    owner.method_hint.hide()
    layout.addWidget(owner.method_hint)

    owner.predict_row = QWidget(page)
    group = QVBoxLayout(owner.predict_row)
    group.setContentsMargins(0, 0, 0, 0)
    predict = QHBoxLayout()
    group.addLayout(predict)
    predict.setContentsMargins(0, 0, 0, 0)
    predict.addWidget(QLabel("Particles", owner.predict_row))
    owner.predict_particles = QComboBox(owner.predict_row)
    owner.predict_particles.setObjectName("fitPredictParticles")
    owner.predict_particles.setToolTip(PREDICT_PARTICLES_TIP)
    owner.predict_particles.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    owner.predict_particles.setMinimumContentsLength(12)
    predict.addWidget(owner.predict_particles, 1)
    owner.predict_substrate_button = QPushButton("Cut geometry and substrate…", owner.predict_row)
    group.addWidget(owner.predict_substrate_button)
    owner.predict_substrate_note = muted("", owner.predict_row)
    group.addWidget(owner.predict_substrate_note)
    owner.reflected_beam = QCheckBox("Reflected beam present in the data", owner.predict_row)
    group.addWidget(owner.reflected_beam)
    owner.measured_qz = QDoubleSpinBox(owner.predict_row)
    owner.measured_qz.setDecimals(5)
    owner.measured_qz.setRange(0, 10)
    owner.measured_qz.setSingleStep(0.01)
    owner.measured_qz.setSpecialValueText("Unknown")
    qz_row = owner.predict_qz_row = QWidget(owner.predict_row)
    qz_layout = QHBoxLayout(qz_row)
    qz_layout.setContentsMargins(0, 0, 0, 0)
    qz_layout.addWidget(QLabel("Measured q_z for level-1 flags (nm⁻¹)", qz_row))
    qz_layout.addWidget(owner.measured_qz)
    qz_row.hide()
    owner.reflected_beam.toggled.connect(qz_row.setVisible)
    group.addWidget(qz_row)
    layout.addWidget(owner.predict_row)

    owner.fit_step_button = QPushButton("Fit", page)
    owner.fit_step_button.setObjectName("fitRunStep")
    owner.fit_step_button.setProperty("gimapPrimaryAction", True)
    owner.fit_step_button.setMinimumHeight(34)
    layout.addWidget(owner.fit_step_button)
    owner.fit_progress = QProgressBar(page)
    owner.fit_progress.setObjectName("fitProgress")
    owner.fit_progress.setRange(0, 100)
    owner.fit_progress.hide()
    owner.fit_progress_text = muted("", page)
    owner.stop_step_button = QPushButton("Stop", page)
    owner.stop_step_button.setObjectName("fitStopStep")
    owner.stop_step_button.setProperty("gimapDangerAction", True)
    owner.stop_step_button.setToolTip("Stop the fit; the best values so far are kept")
    owner.stop_step_button.hide()
    layout.addWidget(owner.fit_progress)
    layout.addWidget(owner.fit_progress_text)
    layout.addWidget(owner.stop_step_button, 0, Qt.AlignLeft)
    owner.weighting_note = muted("", page)  # a warning when the file has no σ (with σ: the residuals' tooltip)
    owner.weighting_note.setObjectName("fitWeightingNote")
    owner.weighting_note.setProperty("gimapRole", "warning")
    owner.weighting_note.hide()
    layout.addWidget(owner.weighting_note)

    options = AdvancedSection("Options", "", page)
    options.setObjectName("fitAdvanced")
    budget = QHBoxLayout()
    budget.addWidget(QLabel("Evaluations at most (Refine, Search)", options))
    owner.budget_spin = QSpinBox(options)
    owner.budget_spin.setObjectName("fitBudget")
    owner.budget_spin.setRange(0, 1_000_000)
    owner.budget_spin.setSingleStep(500)
    owner.budget_spin.setSpecialValueText("automatic")
    owner.budget_spin.setToolTip("How many model evaluations a fit may use (automatic: by the method and free parameters)")
    budget.addWidget(owner.budget_spin, 1)
    options.add_layout(budget)
    owner.batch_button = QPushButton("1D Predict Batch Window…", options)
    owner.batch_button.setObjectName("fitManyCurves")
    owner.batch_button.setToolTip("1D Predict: a list of curve files fitted one after another, with their results. "
                                  "To fit every curve of a folder with this model, use Fit Every Curve in This Folder…")
    options.add_widget(owner.batch_button)
    layout.addWidget(options)


__all__ = ["METHODS", "PREDICT_PARTICLES_TIP", "build_fit_step"]
