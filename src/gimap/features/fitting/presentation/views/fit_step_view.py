"""The Fit step of the single-curve page (static layout; behaviour in ``single/run.py`` and ``single/predict.py``).

One choice and one button: the method — Predict parameters (GIMaP's network, then least squares), Refine (least
squares from Model) or Search (the ranges, then least squares); the page selects Predict for a GISAXS cut not fitted
yet and says why — what Predict needs right under it (the particles), then Fit. Parameters that stop at an edge of
their range are reported (Fit details), never widened. Everything changed rarely is in Options: the evaluation budget, many
curves.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox, QWidget

from src.gimap.app.presentation.components import AdvancedSection

from .step_widgets import muted

METHODS = (
    ("predict", "Predict parameters",
     "GIMaP's network proposes the particles and their sizes from the curve; the proposals are weighted against the "
     "model (a posterior with 68 % intervals and the probability of each solution), and each solution is refined by "
     "least squares. Values fixed in Model are not kept: Refine afterwards keeps them."),
    ("local", "Refine the values in Model",
     "Least squares from the values in Model, with Fitting's own model: fast; finds the nearest good fit."),
    ("global", "Search the ranges, then refine",
     "Tries values across the min–max range of every free parameter first; slower, for a poor start."),
)
PREDICT_PARTICLES_TIP = ("The particles in Model: their families, with or without a paracrystal, in the order of their "
                         "radii. A choice: the network proposes the compositions it finds most probable among these, "
                         "and their posterior probabilities decide.")


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
    predict = QHBoxLayout(owner.predict_row)
    predict.setContentsMargins(0, 0, 0, 0)
    predict.addWidget(QLabel("Particles", owner.predict_row))
    owner.predict_particles = QComboBox(owner.predict_row)
    owner.predict_particles.setObjectName("fitPredictParticles")
    owner.predict_particles.setToolTip(PREDICT_PARTICLES_TIP)
    owner.predict_particles.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    owner.predict_particles.setMinimumContentsLength(12)
    predict.addWidget(owner.predict_particles, 1)
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
