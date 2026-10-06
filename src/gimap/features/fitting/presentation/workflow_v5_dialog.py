"""One-click fitting and text batch workspace; all heavy work runs out of process.

The window is in the interface language: its static texts are translated by the i18n walker when it
is shown, and what it writes at run time — the status line, the input line, the table's headers,
tooltips and stage names, the placeholders and the figure's placeholder — goes through ``tr`` and is
made again after a switch of the language (``_say``, ``_retranslate``). The prediction settings and the
options they give are in ``workflow_v5_settings.py``; their values never depend on the language.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QEvent
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
    QTextBrowser,
    QProgressBar,
    QTabWidget,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from src.gimap.app.presentation.i18n import language_changed, tr, trf
from src.gimap.app.presentation.theme import set_role, theme_manager
from src.gimap.app.jobs import JobRequest
from src.gimap.integrations.jobs import LocalProcessJobRunner
from ..application.workflow_v5 import validate_options, bundled_workflow
from .curve_rendering import follow_plot_theme, text_font_family, theme_figure
from .workflow_v5_settings import WorkflowV5SettingsMixin

HEADERS = ("Curve / side", "#", "Components", "logRMSE", "RMS/σ", "Stage")
HEADER_TIPS = {
    2: "S: sphere; RC: random cylinder; VC: vertical cylinder. Repeated types are distinct components.",
    3: "Natural-log RMSE on original positive-intensity observations only.",
    4: "RMS of (forward − observed)/sigma, including negative observations; not a calibrated probability.",
}
FIGURE_EMPTY = "Add curves (or Use current cut), then Fit curve.\nSelect a candidate to see its fit here."
PARAMETERS_EMPTY = "Select a candidate on the left: its components and parameters, with units."
LOG_EMPTY = "What each curve's fit did, while the batch runs."
STAGE_LABELS = {
    "stable_amplitude_calibrated": "Model + amplitude",
    "stable_neural": "Neural model",
    "stable_numerical_fallback": "Numerical fallback",
}
RMSE_TIP = ("Natural-log RMSE over the measured curve, including measurement noise. "
            "Review peak positions, overall shape and residuals; no mandatory cutoff is applied.")
UNITS_NOTE = ("Lengths: nm. sigma_R/h/D: relative standard deviations.\nMixture weights are not posterior "
              "probabilities.\nResolution sigma: nm^-1; nu: dimensionless.")


class WorkflowJobThread(QThread):
    progress = pyqtSignal(object)
    completed = pyqtSignal(object)

    def __init__(self, payload, parent=None, runner=None):
        super().__init__(parent)
        self.runner = runner or LocalProcessJobRunner()
        self.request = JobRequest(
            handler="src.gimap.features.fitting.infrastructure.adapters.workflow_v5:run_workflow_job",
            payload=payload,
            timeout_seconds=None,
        )

    def run(self):
        try:
            self.completed.emit(self.runner.run(self.request, self.progress.emit))
        except Exception as exc:
            self.completed.emit(exc)

    def cancel(self):
        self.runner.cancel(self.request.job_id)


class WorkflowV5Dialog(QDialog, WorkflowV5SettingsMixin):
    settings_changed = pyqtSignal(dict)
    candidate_selected = pyqtSignal(dict)

    def __init__(
        self, current_curve=None, options=None, parent=None, runner=None, settings_only=False
    ):
        super().__init__(parent)
        self.setWindowTitle("1D Predict · fit curves & batch")
        self.setObjectName("workflowV5Dialog")
        self.resize(1180, 800)
        self.setFont(QFont("Segoe UI", 10))
        self.current_curve = current_curve
        self.can_start = lambda: True
        self.sigma_estimated = lambda: False
        self.observation_metadata = lambda: {}
        self.runner = runner
        self._close_owner_when_finished = False
        if parent is not None:
            parent.installEventFilter(self)
        self.files = []
        self.rows = []
        self.job = None
        self.output_dir = None
        self._folder = ""
        """Where Add curves… opens: the folder of the curves added last."""
        self._texts = {}
        """Run-time texts (widget → the function that makes it), made again after a switch of the language."""
        self._options = validate_options(options or {})
        root = QVBoxLayout(self)
        title = QLabel("1D Predict", self)
        set_role(title, "display")
        root.addWidget(title)
        hint = QLabel(
            "Load curves → Fit → compare candidates. General V5 proposes multiple compositions. "
            "The single-RC specialist requires a known single random cylinder and is experimental."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)
        actions = QHBoxLayout()
        self.load_button = QPushButton("Add curves…")
        self.current_button = QPushButton("Use current cut")
        self.settings_button = QPushButton("Parameters…")
        self.settings_button.setCheckable(True)
        self.run_button = QPushButton("Fit curve")
        self.run_button.setObjectName("workflowV5RunButton")
        set_role(self.run_button, "primary")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.export_button = QPushButton("Open results")
        self.export_button.setEnabled(False)
        for b in (self.load_button, self.current_button, self.settings_button):
            actions.addWidget(b)
        actions.addStretch(1)
        for b in (self.run_button, self.cancel_button, self.export_button):
            actions.addWidget(b)
        root.addLayout(actions)
        self.input_label = QLabel(
            "Current cut — original measured points; positive and negative sides are fitted separately."
        )
        self.input_label.setWordWrap(True)
        root.addWidget(self.input_label)
        self.settings_panel = self._build_settings()
        root.addWidget(self.settings_panel)
        self._follow_language()
        if settings_only:
            self.setWindowTitle("In-situ · 1D prediction parameters")
            title.setText("In-situ prediction parameters")
            hint.setText(
                "Save creates a new settings snapshot for future frames. Completed frames are unchanged."
            )
            for button in (
                self.load_button,
                self.current_button,
                self.settings_button,
                self.run_button,
                self.cancel_button,
                self.export_button,
            ):
                button.hide()
            self.input_label.hide()
            self.status = QLabel("Edit known components / resolution, or leave them automatic.")
            root.addWidget(self.status)
            close = QPushButton("Close")
            close.clicked.connect(self.close)
            root.addWidget(close)
            self.resize(1040, 430)
            return
        self.settings_panel.hide()
        split = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, len(HEADERS))
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.table.setColumnWidth(0, 130)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        split.addWidget(self.table)
        tabs = QTabWidget()
        # Every tab label bold, not only the selected one, set as the font so that the tab widths fit it.
        bold = QFont(tabs.tabBar().font())
        bold.setWeight(QFont.DemiBold)
        tabs.tabBar().setFont(bold)
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._figure_empty = self.figure.text(0.5, 0.5, tr(FIGURE_EMPTY), ha="center", va="center",
                                              color=theme_manager().color("plot_fg").name(),
                                              fontfamily=text_font_family())  # Chinese too, not empty boxes
        follow_plot_theme(self, self.figure, self.canvas)
        self.parameters = QTextBrowser()
        self.log = QTextBrowser()
        tabs.addTab(self.canvas, "Fit plot")
        tabs.addTab(self.parameters, "Parameters and units")
        tabs.addTab(self.log, "Batch log")
        split.addWidget(tabs)
        split.setSizes([520, 640])
        root.addWidget(split, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self._say(self.status, lambda: tr(
            "Ready. Automatic conditions are estimates; candidates are not calibrated probabilities."))
        root.addWidget(self.status)
        self.progress = QProgressBar()
        root.addWidget(self.progress)
        self._static_texts()
        self.load_button.clicked.connect(self._choose_files)
        self.current_button.clicked.connect(self._use_current)
        self.settings_button.toggled.connect(self.settings_panel.setVisible)
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self._cancel)
        self.export_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))
        )
        self.table.currentCellChanged.connect(lambda row, *_: self._preview(row))

    # -- texts in the interface language ------------------------------------------------------

    def _say(self, widget, text) -> None:
        """``text`` on ``widget`` (a label or a button); a function that makes it is made again after a
        switch of the interface language (a message from the job is shown as it came)."""
        self._texts[widget] = text if callable(text) else None
        widget.setText(text() if callable(text) else str(text))

    def show_status(self, text) -> None:
        """The status line: a text, or a function that makes it in the interface language."""
        self._say(self.status, text)

    def _static_texts(self) -> None:
        """What the walker cannot reach: the headers' tooltips, the placeholders of the text views, the
        figure's placeholder (the header labels too, so that they follow a switch while the window is open)."""
        self.table.setHorizontalHeaderLabels([tr(text) for text in HEADERS])
        for column, tip in HEADER_TIPS.items():
            self.table.horizontalHeaderItem(column).setToolTip(tr(tip))
        self.parameters.setPlaceholderText(tr(PARAMETERS_EMPTY))
        self.log.setPlaceholderText(tr(LOG_EMPTY))
        if self._figure_empty in self.figure.texts:
            self._figure_empty.set_text(tr(FIGURE_EMPTY))
            self.canvas.draw_idle()

    def _follow_language(self) -> None:
        changed = language_changed()

        def retranslate(*_args) -> None:
            try:
                self._retranslate()
            except RuntimeError:  # the window is gone
                pass

        def disconnect(*_args) -> None:
            try:
                changed.disconnect(retranslate)
            except TypeError:
                pass

        changed.connect(retranslate)
        self.destroyed.connect(disconnect)

    def _retranslate(self) -> None:
        for widget, make in list(self._texts.items()):
            if make is not None:
                widget.setText(make())
        if hasattr(self, "table"):
            self._static_texts()
            if self.rows:
                self._fill_table(keep=True)  # the stage names and tooltips

    # -- input --------------------------------------------------------------------------------

    def _choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, tr("Select one or more 1D curves"), self._folder, "Curves (*.txt *.dat *.csv);;All files (*)"
        )
        if files:
            self.files = files
            self._folder = str(Path(files[0]).parent)
            names = ", ".join(Path(f).name for f in files[:5])
            self._say(self.input_label, lambda: trf("{count} file(s): {names}", count=len(files), names=names))
            self._say(self.run_button, (lambda: trf("Fit {count} files", count=len(files))) if len(files) > 1
                      else (lambda: tr("Fit curve")))

    def _use_current(self):
        self.files = []
        self._say(self.input_label, lambda: tr(
            "Current cut — native points, q converted to nm⁻¹ by the fitting workspace."))
        self._say(self.run_button, lambda: tr("Fit curve"))

    # -- the run --------------------------------------------------------------------------------

    def start(self):
        if self.job is not None:
            return
        if not self.can_start():
            self.show_status(lambda: tr("A fitting / in-situ job is already running. Finish or cancel it first."))
            return
        try:
            self.save_settings()
            self.output_dir = (
                Path.cwd()
                / "AI_Fitting_Output"
                / ("v5_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            )
            payload = dict(
                model_path=str(bundled_workflow()),
                output_dir=str(self.output_dir),
                options=self.options(),
            )
            if self.files:
                payload["files"] = self.files
            else:
                arrays = self.current_curve() if self.current_curve else None
                if arrays is None:
                    raise ValueError(tr("Load a curve or select files first"))
                payload.update(
                    {
                        k: None if v is None else np.asarray(v).tolist()
                        for k, v in zip(("q", "intensity", "sigma"), arrays)
                    }
                )
                payload["options"]["q_unit"] = "nm^-1"
                payload["sigma_estimated"] = self.sigma_estimated()
                payload["observation_metadata"] = self.observation_metadata()
            self.rows = []
            self.table.setRowCount(0)
            self.figure.clear()
            self.canvas.draw_idle()
            self.job = WorkflowJobThread(payload, self, runner=self.runner)
            self.job.progress.connect(self._progress)
            self.job.completed.connect(self._completed)
            self.job.finished.connect(self._finished)
            self._set_running(True)
            method = payload["options"]["method"]
            starting = (
                "Starting the experimental single-RC specialist… Checking the known composition and input scope."
                if method == "stable"
                else "Starting numerical physical fitting…"
                if method == "experimental"
                else "Loading experimental General V5… First run includes loading and compilation."
            )
            self.show_status(lambda: tr(starting))
            self.job.start()
        except Exception as exc:
            QMessageBox.warning(self, tr("1D Predict"), str(exc))

    def _set_running(self, running):
        for widget in (self.run_button, self.load_button, self.current_button, self.settings_panel):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _progress(self, progress):
        self.show_status(progress.message)  # the job's own words
        self.log.append(progress.message)
        self.progress.setValue(int(100 * progress.fraction))

    def _completed(self, result):
        if isinstance(result, Exception) or not result.succeeded:
            message = (
                str(result)
                if isinstance(result, Exception)
                else (result.error.message if result.error else result.status)
            )
            self.show_status(message)
            self.log.append(message)
            return
        value = result.value
        self.set_results(value["candidates"], value["output_dir"])
        saved = self._texts.get(self.status)
        quality = saved if saved is not None else (lambda text=self.status.text(): text)
        failures = sum(r["status"] == "failed" for r in value["records"])
        seconds = f"{value['summary']['runtime_seconds']:.2f}"
        self.show_status(lambda: trf("Finished in {seconds} s · {failures} failed files. {quality}",
                                     seconds=seconds, failures=failures, quality=quality()))
        self.log.append(json.dumps(value["records"], indent=2))

    def _finished(self):
        job = self.job
        self.job = None
        if job:
            job.deleteLater()
        self._set_running(False)
        self.export_button.setEnabled(bool(self.output_dir and self.output_dir.exists()))
        if self._close_owner_when_finished:
            self.parentWidget().close()

    # -- the candidates -------------------------------------------------------------------------

    def set_results(self, rows, output_dir):
        self.rows = list(rows)
        self.output_dir = Path(output_dir)
        self._fill_table()
        if rows:
            if self.table.currentRow() == 0:
                self._preview(0)  # already the current row: no signal, drawn here
            else:
                self.table.selectRow(0)
        self.export_button.setEnabled(True)
        self.progress.setValue(100)
        count = len(rows)
        self.show_status(lambda: trf(
            "{count} candidates saved. Review curve shape and residuals; "
            "observed-data scores include noise and are not probabilities.", count=count))

    def _fill_table(self, *, keep: bool = False) -> None:
        """The candidates' rows; ``keep``: the same rows again in another language (the selection kept, the
        selected candidate not sent again)."""
        current = self.table.currentRow()
        if keep:
            self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            error = r.get("best_log_rmse")
            stage = r["best_source"]
            short_file = tr("Current") if r.get("file") == "Current curve" else r.get("file", "")
            combination = (
                r["combination"]
                .replace("random_cylinder", "RC")
                .replace("vertical_cylinder", "VC")
                .replace("sphere", "S")
            )
            values = [
                f"{short_file} / {'+' if r['side'] == 'positive' else '−'}",
                r["rank"],
                combination,
                "—" if error is None else f"{error:.4f}",
                f"{r['signed_weighted_rms']:.3f}",
                tr(STAGE_LABELS.get(stage, stage)),
            ]
            for j, v in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(str(v)))
            self.table.item(i, 0).setToolTip(f"{r.get('file', '')} / {r['side']}")
            self.table.item(i, 2).setToolTip(r["combination"])
            stage_detail = [trf("Stage: {stage}", stage=stage)]
            if r.get("fallback_reason"):
                stage_detail.append(trf("Reason: {reason}", reason=r["fallback_reason"]))
            warnings = r.get("warnings") or []
            if isinstance(warnings, str):
                warnings = [warnings]
            stage_detail.extend(str(warning) for warning in warnings)
            self.table.item(i, 5).setToolTip("\n".join(stage_detail))
            self.table.item(i, 3).setToolTip(tr(RMSE_TIP))
        if keep:
            if 0 <= current < len(self.rows):
                self.table.setCurrentCell(current, 0)
            self.table.blockSignals(False)
            if 0 <= current < len(self.rows):
                self._show_parameters(self.rows[current])

    def _preview(self, index):
        if not 0 <= index < len(self.rows):
            return
        r = self.rows[index]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.errorbar(
            r["native_q"],
            r["observed"],
            yerr=r["sigma"],
            fmt="o",
            ms=4,
            color="#64748b",
            alpha=0.7,
            label="Measured ± σ",
        )
        ax.plot(
            r["display_q"],
            r["display_fit"],
            color="#2563eb",
            lw=1.8,
            label=("Physical fit" if r.get("physics_backend") == "experimental_calibrated" else "Model forward") + f" · {len(r['display_q'])} display points",
        )
        ax.set_yscale("symlog", linthresh=max(min(r["sigma"]), 1e-12))
        ax.set_xlabel("q (nm⁻¹)")
        ax.set_ylabel("Intensity (input units)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
        theme_figure(self.figure)  # screen colours only: the data and the fit are drawn as they are
        self.canvas.draw_idle()
        self._show_parameters(r)
        self.candidate_selected.emit(r)

    def _show_parameters(self, r) -> None:
        detail = {
            k: v
            for k, v in r.items()
            if k not in ("native_q", "native_fit", "observed", "sigma", "display_q", "display_fit")
        }
        self.parameters.setPlainText(tr(UNITS_NOTE) + "\n\n" + json.dumps(detail, indent=2, ensure_ascii=False))

    def _cancel(self):
        if self.job:
            self.job.cancel()
            self.show_status(lambda: tr("Cancelling… completed file results remain saved."))

    def closeEvent(self, event):
        if self.job is not None:
            self._cancel()
            event.ignore()
        else:
            super().closeEvent(event)

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() == QEvent.Close and self.job is not None:
            self._close_owner_when_finished = True
            self._cancel()
            event.ignore()
            return True
        return super().eventFilter(watched, event)
