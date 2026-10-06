"""Static layout of the single-curve page of Fitting (behaviour in ``single/``).

A command bar — open a curve, undo, the curve, Fit, export — then two panels: the steps with
their controls (``fit_steps_view.py``) and the plot of the curve with the model, its terms and
the fitting range, over the residuals. A status line with a progress bar closes the page.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import CurvePlot

from .fit_steps_view import FitStepsView

RESIDUAL_HEIGHT = 130


def _separator(parent: QWidget) -> QFrame:
    line = QFrame(parent)
    line.setObjectName("toolbarSeparator")
    line.setFrameShape(QFrame.VLine)
    line.setFixedWidth(1)
    return line


class FitPageView(FitStepsView):
    """Command bar, then steps | plot, then the status line."""

    def setup_ui(self, page: QWidget) -> None:
        page.setObjectName("fitPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(12, 10, 12, 6)
        root.setSpacing(8)
        root.addWidget(self._command_bar(page))
        self.splitter = QSplitter(Qt.Horizontal, page)
        self.splitter.setObjectName("fitSplitter")
        self.splitter.setHandleWidth(8)
        self.splitter.setChildrenCollapsible(False)
        self.steps_panel = self.setup_steps_panel(self.splitter)
        self.steps_panel.setMinimumWidth(340)
        self.splitter.addWidget(self.steps_panel)
        self.splitter.addWidget(self._plot_panel(self.splitter))
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([400, 900])
        root.addWidget(self.splitter, 1)
        root.addLayout(self._status_row(page))

    def _command_bar(self, page: QWidget) -> QFrame:
        bar = QFrame(page)
        bar.setObjectName("fitCommandBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)
        self.open_button = QToolButton(bar)
        self.open_button.setObjectName("fitOpenButton")
        self.open_button.setText("Open Curve…")
        self.open_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.open_button.setToolTip("Open a 1D curve (q, I, σ) — Analyze ▸ Send to Fitting opens its cut here")
        open_menu = QMenu(self.open_button)
        self.load_model_action = QAction("Load Model…", page)
        self.load_model_action.setToolTip("A model saved from Fitting (JSON), for this curve")
        open_menu.addAction(self.load_model_action)
        self.open_button.setMenu(open_menu)
        self.undo_button = QToolButton(bar)
        self.undo_button.setObjectName("fitUndoButton")
        self.undo_button.setText("Undo")
        self.undo_button.setAutoRaise(True)
        self.undo_button.setToolTip(
            "Undo the last change of the model, the fitting range or the left-out points, or the last fit (Ctrl+Z)")
        self.redo_button = QToolButton(bar)
        self.redo_button.setObjectName("fitRedoButton")
        self.redo_button.setText("Redo")
        self.redo_button.setAutoRaise(True)
        self.redo_button.setToolTip("Redo (Ctrl+Shift+Z)")
        self.curve_chip = QLabel("No curve yet — open one, or send a cut from Analyze", bar)
        self.curve_chip.setObjectName("fitCurveChip")
        self.curve_chip.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.curve_chip.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.fit_button = QPushButton("Fit", bar)
        self.fit_button.setObjectName("fitRunButton")
        self.fit_button.setProperty("gimapPrimaryAction", True)
        self.fit_button.setToolTip("Run the method chosen in the Fit step (Ctrl+Return)")
        self.stop_button = QPushButton("Stop", bar)
        self.stop_button.setObjectName("fitStopButton")
        self.stop_button.setProperty("gimapDangerAction", True)
        self.stop_button.setToolTip("Stop the fit; the best values so far are kept")
        self.stop_button.hide()
        self.export_button = QToolButton(bar)
        self.export_button.setObjectName("fitExportButton")
        self.export_button.setText("Save")
        self.export_button.setToolTip("Save the data and fit, the plot, or the model")
        self.export_button.setPopupMode(QToolButton.InstantPopup)
        export_menu = QMenu(self.export_button)
        self.export_data_action = export_menu.addAction("Data and Fit…")
        self.export_plot_action = export_menu.addAction("Plot…")
        self.save_model_action = export_menu.addAction("Model…")
        self.export_button.setMenu(export_menu)
        for widget in (self.open_button, self.undo_button, self.redo_button):
            row.addWidget(widget)
        row.addWidget(_separator(bar))
        row.addWidget(self.curve_chip, 1)
        row.addWidget(self.fit_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.export_button)
        return bar

    def _plot_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("fitPlotPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.plot = CurvePlot("", panel, log_y=True, log_x=False)
        self.plot.setObjectName("fitCurvePlot")
        self.plot.set_labels("|q| (nm⁻¹)", "Intensity")
        self.plot.set_empty_text("Open a curve (q, I, σ), or send a cut from Analyze ▸ Send to Fitting.")
        self.terms_check = QCheckBox("Terms", self.plot)
        self.terms_check.setObjectName("fitShowTerms")
        self.terms_check.setToolTip("Also draw each term of the model: the particles, the background, the resolution peak")
        self.plot.header_layout.insertWidget(1, self.terms_check)
        self.exclude_button = QToolButton(self.plot)
        self.exclude_button.setObjectName("fitExcludeButton")
        self.exclude_button.setText("Exclude")
        self.exclude_button.setCheckable(True)
        self.exclude_button.setToolTip(
            "Leave points out of the fit: click a point (again to take it back) or drag a box around several"
        )
        self.plot.header_layout.insertWidget(2, self.exclude_button)
        layout.addWidget(self.plot, 1)
        self.residual_plot = CurvePlot("", panel, log_y=False, log_x=None)
        self.residual_plot.setObjectName("fitResidualPlot")
        self.residual_plot.log_check.hide()
        self.residual_plot.setFixedHeight(RESIDUAL_HEIGHT)
        self.residual_plot.set_labels("|q| (nm⁻¹)", "Δ/σ")  # the formula is in the tooltip (the page sets it)
        self.residual_plot.set_empty_text("Open a curve: the residuals of the model appear here.")
        layout.addWidget(self.residual_plot)
        return panel

    def _status_row(self, page: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.status_label = QLabel("", page)
        self.status_label.setObjectName("fitStatusLabel")
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_progress = QProgressBar(page)
        self.status_progress.setObjectName("fitStatusProgress")
        self.status_progress.setRange(0, 100)
        self.status_progress.setMaximumWidth(220)
        self.status_progress.hide()
        row.addWidget(self.status_label, 1)
        row.addWidget(self.status_progress)
        return row


__all__ = ["FitPageView", "RESIDUAL_HEIGHT"]
