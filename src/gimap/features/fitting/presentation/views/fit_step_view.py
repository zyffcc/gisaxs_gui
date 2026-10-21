"""The Fit step of the single-curve page (static layout; behaviour in ``single/run.py`` and ``single/fit1d*.py``).

One choice and one button: the method (the page selects the one recommended for the curve and says why), what it
needs right under it (the particles for Predict, the families for the shape search), then Fit. Parameters that
stop at an edge of their range are dealt with by the fit itself (the check box). Everything changed rarely is in
Options: fit1d's network, search, curve type, qz and systematic error; the evaluation budget; many curves.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox, QWidget

from src.gimap.app.presentation.components import AdvancedSection

from .fit1d_settings_view import setup_fit1d_choice, setup_fit1d_options
from .step_widgets import muted

METHODS = (
    ("fit1d", "Predict parameters (fit1d)",
     "Finds the particles and their sizes from the curve (network proposals and a physical fit), with errors and "
     "equivalent solutions. The first step for a GISAXS cut."),
    ("local", "Refine the values in Model",
     "Least squares from the values in Model, with Fitting's own model: fast; finds the nearest good fit."),
    ("global", "Search the ranges, then refine",
     "Tries values across the min–max range of every free parameter first; slower, for a poor start."),
    ("shapes", "Find the particle shape (no AI)",
     "Fits sphere, random cylinder and vertical cylinder from several starts and lists the solutions."),
    ("ai", "AI proposal (1D Predict)",
     "The V5 model proposes compositions and parameters, then corrects them numerically."),
)
QUICK_FAMILIES = (("Try every family", "auto"), ("The families in Model", "model"))
RESOLVE_TIP = ("When a value stops at the edge of its range, widen that side (or fix a size spread at its lower "
               "limit), fit again and keep it if the fit is not worse. Results lists what was done.")


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
    owner.method_hint = muted("", page)
    owner.method_hint.setObjectName("fitMethodHint")
    layout.addWidget(owner.method_hint)

    setup_fit1d_choice(owner, page, layout)
    owner.families_row = QWidget(page)
    families = QHBoxLayout(owner.families_row)
    families.setContentsMargins(0, 0, 0, 0)
    families.addWidget(QLabel("Families", owner.families_row))
    owner.families_combo = QComboBox(owner.families_row)
    owner.families_combo.setObjectName("fitShapeFamilies")
    for text, value in QUICK_FAMILIES:
        owner.families_combo.addItem(text, value)
    families.addWidget(owner.families_combo, 1)
    layout.addWidget(owner.families_row)

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
    owner.resolve_edges_check = QCheckBox("Resolve values at the edge of their range automatically", page)
    owner.resolve_edges_check.setObjectName("fitResolveEdges")
    owner.resolve_edges_check.setToolTip(RESOLVE_TIP)
    owner.resolve_edges_check.setChecked(True)
    layout.addWidget(owner.resolve_edges_check)
    owner.weighting_note = muted("", page)
    layout.addWidget(owner.weighting_note)

    options = AdvancedSection("Options", "", page)
    options.setObjectName("fitAdvanced")
    options.add_widget(muted("Predict (fit1d)", options))
    setup_fit1d_options(owner, options)
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


__all__ = ["METHODS", "QUICK_FAMILIES", "RESOLVE_TIP", "build_fit_step"]
