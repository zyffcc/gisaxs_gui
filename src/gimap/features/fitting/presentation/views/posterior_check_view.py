"""The Posterior check tab of the Uncertainty and Report window (static layout; behaviour in
``single/posterior_check.py``): the run's settings, Run Check and Stop, the values checked beside the run's intervals,
the trace of one value and the verdict on convergence."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
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


def _spin(parent: QWidget, name: str, low: int, high: int, value: int, tip: str) -> QSpinBox:
    box = QSpinBox(parent)
    box.setObjectName(name)
    box.setRange(low, high)
    box.setValue(value)
    box.setToolTip(tip)
    return box


def build_check_tab(owner, tabs: QTabWidget) -> QWidget:
    tab = QWidget(tabs)
    column = QVBoxLayout(tab)
    owner.check_what = _note(tab, "fitCheckWhat")
    column.addWidget(owner.check_what)
    controls = QHBoxLayout()
    owner.check_steps = _spin(tab, "fitCheckSteps", 200, 100000, 3000,
                              "Steps of every walker; the first half is discarded (burn-in)")
    owner.check_steps.setSingleStep(500)
    owner.check_walkers = _spin(tab, "fitCheckWalkers", 0, 400, 0,
                                "Walkers in each independent ensemble (even). Auto uses at least 16 and four per value.")
    owner.check_walkers.setSpecialValueText("auto")
    owner.check_seed = _spin(tab, "fitCheckSeed", 0, 999999, 0, "Seed of the random numbers (the same seed, the same run)")
    owner.check_run = QPushButton("Run Check", tab)
    owner.check_run.setObjectName("fitCheckRun")
    owner.check_run.setProperty("gimapPrimaryAction", True)
    owner.check_stop = QPushButton("Stop", tab)
    owner.check_stop.setObjectName("fitCheckStop")
    owner.check_stop.hide()
    for text, widget in (("Steps", owner.check_steps), ("Walkers", owner.check_walkers), ("Seed", owner.check_seed)):
        controls.addWidget(QLabel(text, tab))
        controls.addWidget(widget)
    controls.addStretch(1)
    controls.addWidget(owner.check_run)
    controls.addWidget(owner.check_stop)
    column.addLayout(controls)
    owner.check_progress = QProgressBar(tab)
    owner.check_progress.setObjectName("fitCheckProgress")
    owner.check_progress.hide()
    column.addWidget(owner.check_progress)
    owner.check_table = read_only_table(("Value", "Checked", "Reference 68 %", "MCMC median", "MCMC 68 %", "MCMC 95 %",
                                         "τ (steps)", "R̂"), tab, "fitCheckTable")
    owner.check_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    column.addWidget(owner.check_table, 2)
    owner.check_verdict = _note(tab, "fitCheckVerdict")
    column.addWidget(owner.check_verdict)
    trace = QHBoxLayout()
    trace.addWidget(QLabel("Trace of", tab))
    owner.check_trace_value = QComboBox(tab)
    owner.check_trace_value.setObjectName("fitCheckTraceValue")
    trace.addWidget(owner.check_trace_value, 1)
    owner.check_save = QPushButton("Save Check…", tab)
    owner.check_save.setObjectName("fitCheckSave")
    owner.check_save.setToolTip("The run's settings, intervals and diagnostics beside the values it checks (JSON)")
    trace.addStretch(1)
    trace.addWidget(owner.check_save)
    column.addLayout(trace)
    owner.check_trace = CurvePlot("", tab, log_y=False, sides=False)
    owner.check_trace.setObjectName("fitCheckTrace")
    owner.check_trace.log_check.hide()
    column.addWidget(owner.check_trace, 2)
    tabs.addTab(tab, "Posterior check (MCMC)")
    return tab


__all__ = ["build_check_tab"]
