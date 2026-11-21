"""The Uncertainty and Report window of the single-curve page (static layout; behaviour in ``single/uncertainty.py``).

Three tabs: the errors and the correlation matrix of the last Refine or Search (a heat map with the values, the
strongly correlated pairs), the χ² profile of one parameter, and the fit report to read, copy or save.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import CurvePlot

from .step_widgets import read_only_table


def _note(parent: QWidget, name: str) -> QLabel:
    label = QLabel("", parent)
    label.setObjectName(name)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def build_uncertainty_dialog(owner, parent: QWidget) -> None:
    dialog = QDialog(parent)
    dialog.setObjectName("fitUncertaintyDialog")
    dialog.setWindowTitle("Uncertainty and Report")
    dialog.resize(860, 640)
    owner.uncertainty_dialog = dialog
    layout = QVBoxLayout(dialog)
    owner.uncertainty_tabs = QTabWidget(dialog)
    layout.addWidget(owner.uncertainty_tabs)

    errors = QWidget(dialog)
    column = QVBoxLayout(errors)
    owner.uncertainty_summary = _note(errors, "fitUncertaintySummary")
    column.addWidget(owner.uncertainty_summary)
    owner.uncertainty_values = read_only_table(("Parameter", "Name in links", "value", "± 1σ", "relative"), errors,
                                               "fitUncertaintyValues")
    owner.uncertainty_values.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    column.addWidget(owner.uncertainty_values, 2)
    title = QLabel("Correlation matrix", errors)
    title.setProperty("gimapRole", "strong")
    column.addWidget(title)
    owner.correlation_table = QTableWidget(errors)
    owner.correlation_table.setObjectName("fitCorrelationMatrix")
    owner.correlation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    owner.correlation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    owner.correlation_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    column.addWidget(owner.correlation_table, 3)
    owner.correlation_pairs = _note(errors, "fitCorrelatedPairs")
    column.addWidget(owner.correlation_pairs)
    owner.uncertainty_tabs.addTab(errors, "Errors and correlations")

    profile = QWidget(dialog)
    column = QVBoxLayout(profile)
    controls = QHBoxLayout()
    owner.profile_parameter = QComboBox(profile)
    owner.profile_parameter.setObjectName("fitProfileParameter")
    owner.profile_points = QSpinBox(profile)
    owner.profile_points.setObjectName("fitProfilePoints")
    owner.profile_points.setRange(5, 41)
    owner.profile_points.setValue(15)
    owner.profile_points.setToolTip("Values of the grid: ±3 errors around the fitted value, inside its range")
    owner.profile_run = QPushButton("Run Profile", profile)
    owner.profile_run.setObjectName("fitProfileRun")
    owner.profile_run.setProperty("gimapPrimaryAction", True)
    owner.profile_stop = QPushButton("Stop", profile)
    owner.profile_stop.setObjectName("fitProfileStop")
    owner.profile_stop.hide()
    points_label = QLabel("Points", profile)
    for widget, stretch in ((owner.profile_parameter, 1), (points_label, 0), (owner.profile_points, 0),
                            (owner.profile_run, 0), (owner.profile_stop, 0)):
        controls.addWidget(widget, stretch)
    column.addLayout(controls)
    owner.profile_progress = QProgressBar(profile)
    owner.profile_progress.setObjectName("fitProfileProgress")
    owner.profile_progress.hide()
    column.addWidget(owner.profile_progress)
    owner.profile_plot = CurvePlot("", profile, log_y=False, sides=False)
    owner.profile_plot.setObjectName("fitProfilePlot")
    owner.profile_plot.log_check.hide()  # Δχ² is drawn on a linear axis
    column.addWidget(owner.profile_plot, 1)
    owner.profile_result = _note(profile, "fitProfileResult")
    column.addWidget(owner.profile_result)
    owner.uncertainty_tabs.addTab(profile, "χ² profile")

    report = QWidget(dialog)
    column = QVBoxLayout(report)
    controls = QHBoxLayout()
    owner.report_format = QComboBox(report)
    owner.report_format.setObjectName("fitReportFormat")
    owner.report_copy = QPushButton("Copy", report)
    owner.report_copy.setObjectName("fitReportCopy")
    owner.report_save = QPushButton("Save Report…", report)
    owner.report_save.setObjectName("fitReportSave")
    owner.report_save.setProperty("gimapPrimaryAction", True)
    controls.addWidget(QLabel("Format", report))
    controls.addWidget(owner.report_format)
    controls.addStretch(1)
    controls.addWidget(owner.report_copy)
    controls.addWidget(owner.report_save)
    column.addLayout(controls)
    owner.report_view = QTextBrowser(report)
    owner.report_view.setObjectName("fitReportView")
    column.addWidget(owner.report_view, 1)
    owner.uncertainty_tabs.addTab(report, "Fit report")


__all__ = ["build_uncertainty_dialog"]
