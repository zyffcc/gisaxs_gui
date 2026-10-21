"""The Results step of the single-curve page (static layout; behaviour in ``single/results.py``,
``single/fit1d.py`` and ``single/result_actions.py``).

From the top: how good the fit is; what to do next (Refine these values, Fit again, every curve of the folder);
what the fit did by itself (values at an edge it resolved); what is still to know; the values — select some to fix
or free them, or to change their range in Model — then the folds: other solutions or compositions, the constraints
and ranges used. Saving and the log at the end.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

from src.gimap.app.presentation.components import AdvancedSection, FlowLayout

from .step_widgets import fit_to_rows, info_card, muted, read_only_table

CANDIDATE_COLUMNS = ("#", "composition", "χ²/point", "ΔBIC", "note")
DETAIL_COLUMNS = ("parameter", "value ± error", "equivalent range", "state")


def _label(parent: QWidget, name: str, role: str) -> QLabel:
    label = QLabel("", parent)
    label.setObjectName(name)
    label.setWordWrap(True)
    label.setProperty("gimapRole", role)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.hide()
    return label


def _stretching(table, column: int):
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(column, QHeaderView.Stretch)  # cut short; the tooltip has it whole
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    fit_to_rows(table)  # every row shown: only the step's page scrolls
    return table


def build_results_step(owner, page: QWidget, layout) -> None:
    owner.quality_card, owner.quality_label = info_card(page)
    owner.quality_card.setObjectName("fitQualityCard")
    layout.addWidget(owner.quality_card)
    next_row = FlowLayout()
    owner.next_refine_button = QPushButton("Refine These Values", page)
    owner.next_refine_button.setObjectName("fitNextRefine")
    owner.next_refine_button.setProperty("gimapPrimaryAction", True)
    owner.next_refine_button.setToolTip("Refine the values now in Model with Fitting's own model (least squares)")
    owner.fit_again_button = QPushButton("Fit Again", page)
    owner.fit_again_button.setObjectName("fitAgain")
    owner.fit_again_button.setToolTip("Run the same method again, with the values, fixed values and ranges now in Model")
    owner.series_button = QPushButton("Fit Every Curve in This Folder…", page)
    owner.series_button.setObjectName("fitSeriesFromHere")
    owner.series_button.setToolTip("In-situ series: every curve of this folder, each fitted from this model")
    for button in (owner.next_refine_button, owner.fit_again_button, owner.series_button):
        button.hide()
        next_row.addWidget(button)
    layout.addLayout(next_row)
    owner.actions_label = _label(page, "fitAutoActions", "success")
    layout.addWidget(owner.actions_label)
    owner.warnings_label = _label(page, "fitWarnings", "warning")
    owner.warnings_label.show()
    layout.addWidget(owner.warnings_label)

    owner.parameters_table = _stretching(read_only_table(("", "value", "±"), page, "fitParametersTable",
                                                         QAbstractItemView.ExtendedSelection), 0)
    layout.addWidget(owner.parameters_table)
    tools = QHBoxLayout()
    tools.setSpacing(6)
    owner.fix_button = QPushButton("Fix", page)
    owner.fix_button.setObjectName("fitFixSelected")
    owner.fix_button.setToolTip("Keep the selected values as they are in the next fit")
    owner.free_button = QPushButton("Let It Fit", page)
    owner.free_button.setObjectName("fitFreeSelected")
    owner.free_button.setToolTip("Let the next fit change the selected values")
    owner.range_button = QPushButton("Range…", page)
    owner.range_button.setObjectName("fitRangeSelected")
    owner.range_button.setToolTip("Show the min–max range of the selected value in Model")
    for button in (owner.fix_button, owner.free_button, owner.range_button):
        tools.addWidget(button)
    tools.addStretch(1)
    layout.addLayout(tools)
    owner.values_hint = muted("Select values to fix them or let them fit, then Fit Again; right-click for the same.",
                              page)
    layout.addWidget(owner.values_hint)

    owner.alternatives = AdvancedSection("Other solutions", "", page)
    owner.alternatives.setObjectName("fitAlternatives")
    owner.solutions_title = QLabel("Solutions", owner.alternatives)
    owner.solutions_title.setProperty("gimapRole", "strong")
    owner.solutions_table = _stretching(read_only_table(("#", "model", "R (nm)", "D (nm)", "χ²"), owner.alternatives,
                                                        "fitSolutionsTable"), 1)
    owner.use_solution_button = QPushButton("Use This Solution", owner.alternatives)
    owner.use_solution_button.setObjectName("fitUseSolution")
    owner.use_solution_button.setToolTip("Put the selected solution into Model (Undo brings the previous model back)")
    owner.fit1d_candidates_intro = muted(
        "Every composition fit1d tried, best (lowest BIC) first. Compositions within ΔBIC ≤ 10 of the best cannot be "
        "told apart on this curve: choose from what you know of the sample.", owner.alternatives)
    owner.fit1d_candidates_table = _stretching(read_only_table(CANDIDATE_COLUMNS, owner.alternatives,
                                                               "fit1dCandidates"), 4)
    owner.fit1d_candidate_details = _stretching(read_only_table(DETAIL_COLUMNS, owner.alternatives,
                                                                "fit1dCandidateDetails",
                                                                QAbstractItemView.ExtendedSelection), 0)
    owner.fit1d_use_candidate_button = QPushButton("Use This Composition", owner.alternatives)
    owner.fit1d_use_candidate_button.setObjectName("fit1dUseCandidate")
    owner.fit1d_use_candidate_button.setToolTip(
        "Predict again with this composition (its values as the start; values at an edge resolved) — Undo brings "
        "the previous model back")
    for widget in (owner.solutions_title, owner.solutions_table, owner.use_solution_button,
                   owner.fit1d_candidates_intro, owner.fit1d_candidates_table, owner.fit1d_candidate_details,
                   owner.fit1d_use_candidate_button):
        owner.alternatives.add_widget(widget)
    owner.alternatives.hide()
    layout.addWidget(owner.alternatives)

    owner.constraints_section = AdvancedSection("Constraints and ranges", "", page)
    owner.constraints_section.setObjectName("fitConstraintsSection")
    owner.fit1d_constraints_label = _label(owner.constraints_section, "fit1dConstraints", "muted")
    owner.fit1d_constraints_label.show()
    owner.constraints_section.add_widget(owner.fit1d_constraints_label)
    owner.constraints_section.hide()
    layout.addWidget(owner.constraints_section)

    exports = FlowLayout()
    owner.export_data_step_button = QPushButton("Save Data and Fit…", page)
    owner.export_data_step_button.setObjectName("fitExportData")
    owner.export_data_step_button.setToolTip(
        "CSV: q, I, σ, the model, the residuals and each term; a JSON record of the model and the fit next to it")
    owner.export_plot_step_button = QPushButton("Save Plot…", page)
    owner.export_plot_step_button.setObjectName("fitExportPlot")
    owner.export_plot_step_button.setToolTip("The plot as shown: PNG (image) or SVG (vector)")
    owner.save_model_button = QPushButton("Save Model…", page)
    owner.save_model_button.setObjectName("fitSaveModel")
    owner.save_model_button.setToolTip("The model (values, fit/fixed, ranges) as JSON, to load for another curve")
    for button in (owner.export_data_step_button, owner.export_plot_step_button, owner.save_model_button):
        exports.addWidget(button)
    layout.addLayout(exports)

    log = AdvancedSection("Run Log", "", page)
    log.setObjectName("fitLogSection")
    owner.log_view = QPlainTextEdit(log)
    owner.log_view.setObjectName("fitLog")
    owner.log_view.setReadOnly(True)
    owner.log_view.setMinimumHeight(140)
    log.add_widget(owner.log_view)
    layout.addWidget(log)


__all__ = ["CANDIDATE_COLUMNS", "DETAIL_COLUMNS", "build_results_step"]
