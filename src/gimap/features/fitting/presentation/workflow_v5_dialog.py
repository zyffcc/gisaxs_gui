"""One-click fitting and text batch workspace; all heavy work runs out of process."""

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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QFormLayout,
    QGroupBox,
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

from src.gimap.app.presentation.theme import set_role
from src.gimap.app.jobs import JobRequest
from src.gimap.integrations.jobs import LocalProcessJobRunner
from ..application.workflow_v5 import validate_options, bundled_workflow


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


class WorkflowV5Dialog(QDialog):
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
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Curve / side", "#", "Components", "logRMSE", "RMS/σ", "Stage"]
        )
        self.table.horizontalHeaderItem(2).setToolTip(
            "S: sphere; RC: random cylinder; VC: vertical cylinder. Repeated types are distinct components."
        )
        self.table.horizontalHeaderItem(3).setToolTip(
            "Natural-log RMSE on original positive-intensity observations only."
        )
        self.table.horizontalHeaderItem(4).setToolTip(
            "RMS of (forward − observed)/sigma, including negative observations; not a calibrated probability."
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.table.setColumnWidth(0, 130)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        split.addWidget(self.table)
        tabs = QTabWidget()
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.parameters = QTextBrowser()
        self.log = QTextBrowser()
        tabs.addTab(self.canvas, "Fitting")
        tabs.addTab(self.parameters, "Parameters & units")
        tabs.addTab(self.log, "Batch log")
        split.addWidget(tabs)
        split.setSizes([520, 640])
        root.addWidget(split, 1)
        self.status = QLabel(
            "Ready. Automatic conditions are estimates; candidates are not calibrated probabilities."
        )
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.progress = QProgressBar()
        root.addWidget(self.progress)
        self.load_button.clicked.connect(self._choose_files)
        self.current_button.clicked.connect(self._use_current)
        self.settings_button.toggled.connect(self.settings_panel.setVisible)
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self._cancel)
        self.export_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))
        )
        self.table.currentCellChanged.connect(lambda row, *_: self._preview(row))

    def _build_settings(self):
        box = QGroupBox(
            "Prediction settings — saved for single curves and future batch / in-situ runs"
        )
        outer = QHBoxLayout(box)
        left, right = QFormLayout(), QFormLayout()
        self.component_boxes = []
        row = QHBoxLayout()
        for index in range(4):
            combo = QComboBox()
            combo.addItems(["Auto / unused", "Sphere", "Random cylinder", "Vertical cylinder"])
            combo.setCurrentIndex(
                self._options["components"][index]
                if index < len(self._options["components"])
                else 0
            )
            row.addWidget(combo)
            self.component_boxes.append(combo)
        left.addRow("Complete composition", row)
        self.unit = QComboBox()
        self.unit.addItems(["nm^-1", "A^-1"])
        self.unit.setCurrentText(self._options["q_unit"])
        self.side = QComboBox()
        self.side.addItems(["both", "positive", "negative"])
        self.side.setCurrentText(self._options["side"])
        self.numerical = QCheckBox("General V5: improve fit with four numerical steps")
        self.numerical.setChecked(self._options["numerical"])
        self.amplitude_calibration = QCheckBox("Calibrate intensity amplitudes")
        self.amplitude_calibration.setChecked(self._options.get("amplitude_calibration", True))
        self.amplitude_calibration.setToolTip(
            "The single-RC specialist can adjust particle, background and resolution amplitudes while keeping "
            "the neural shape parameters fixed. Broader fitting may still run when curve agreement is poor."
        )
        self.method = QComboBox()
        self.method.addItem("General V5 (experimental)", "model")
        self.method.addItem("Single RC specialist (experimental)", "stable")
        self.method.addItem("Physical fit (numerical)", "experimental")
        self.method.setCurrentIndex(self.method.findData(self._options["method"]))
        self.method.setToolTip(
            "General V5 proposes multiple compositions but remains experimental. "
            "The specialist requires Complete composition = one Random cylinder and eligible native CBF counts. "
            "Other inputs, fixed resolution and poor curve agreement use numerical fallback; "
            "that fallback does not make the specialist a general model. Scores are not probabilities."
        )
        left.addRow("Fit method", self.method)
        left.addRow("Text-file q unit", self.unit)
        left.addRow("q sides", self.side)
        left.addRow(self.amplitude_calibration)
        left.addRow(self.numerical)
        self.fix_sigma = QCheckBox("Fix σ res (nm⁻¹)")
        self.fix_sigma.setChecked(self._options["sigma_res"] is not None)
        self.sigma_res = self._spin(0.001, 0.1, self._options["sigma_res"] or 0.01, 8)
        self.fix_nu = QCheckBox("Fix ν res")
        self.fix_nu.setChecked(self._options["nu_res"] is not None)
        self.nu_res = self._spin(1, 20, self._options["nu_res"] or 7, 6)
        self.sigma_res.setToolTip("General V5: 0.007–0.013 nm⁻¹. RC specialist / physical fit: 0.001–0.1 nm⁻¹; fixed resolution uses numerical fallback.")
        self.nu_res.setToolTip("General V5: 5–10. RC specialist / physical fit: 1–20; fixed resolution uses numerical fallback.")
        right.addRow(self.fix_sigma, self.sigma_res)
        right.addRow(self.fix_nu, self.nu_res)
        self.relative_noise = self._spin(0, 10, self._options["relative_noise"], 4)
        self.noise_floor = self._spin(0, 1e12, self._options["absolute_noise"], 6)
        self.noise_floor.setSpecialValueText("Auto: 0.1% peak")
        self.normalizer = self._spin(0, 1e20, self._options["normalizer"] or 0, 6)
        self.normalizer.setSpecialValueText("Auto: measured max")
        right.addRow("Relative σ (if missing)", self.relative_noise)
        right.addRow("Absolute σ floor", self.noise_floor)
        right.addRow("Intensity normalizer", self.normalizer)
        self.search = QSpinBox()
        self.search.setRange(1, 34)
        self.search.setValue(self._options["search_combinations"])
        self.conditions = QSpinBox()
        self.conditions.setRange(1, 34)
        self.conditions.setValue(self._options["condition_combinations"])
        left.addRow("Discover combinations", self.search)
        left.addRow("Condition best combinations", self.conditions)
        self.method.currentIndexChanged.connect(self._update_method_controls)
        self._update_method_controls()
        save = QPushButton("Save settings")
        save.clicked.connect(self.save_settings)
        right.addRow(save)
        outer.addLayout(left, 1)
        outer.addLayout(right, 1)
        return box

    def _update_method_controls(self):
        legacy = self.method.currentData() == "model"
        self.amplitude_calibration.setEnabled(self.method.currentData() == "stable")
        for widget in (self.numerical, self.normalizer, self.search, self.conditions):
            widget.setEnabled(legacy)

    @staticmethod
    def _spin(lo, hi, value, decimals):
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(lo, hi)
        s.setValue(value)
        return s

    def options(self):
        return validate_options(
            {
                **self._options,
                "method": self.method.currentData(),
                "components": [b.currentIndex() for b in self.component_boxes if b.currentIndex()],
                "q_unit": self.unit.currentText(),
                "side": self.side.currentText(),
                "sigma_res": self.sigma_res.value() if self.fix_sigma.isChecked() else None,
                "nu_res": self.nu_res.value() if self.fix_nu.isChecked() else None,
                "relative_noise": self.relative_noise.value(),
                "absolute_noise": self.noise_floor.value(),
                "normalizer": self.normalizer.value() or None,
                "numerical": self.numerical.isChecked(),
                "amplitude_calibration": self.amplitude_calibration.isChecked(),
                "search_combinations": self.search.value(),
                "condition_combinations": self.conditions.value(),
            }
        )

    def save_settings(self):
        try:
            self._options = self.options()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText(
            "Settings saved. Existing in-situ recipes keep their captured settings."
        )
        self.settings_changed.emit(self._options)

    def _choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select one or more 1D curves", "", "Curves (*.txt *.dat *.csv);;All files (*)"
        )
        if files:
            self.files = files
            self.input_label.setText(
                f"{len(files)} file(s): " + ", ".join(Path(f).name for f in files[:5])
            )
            self.run_button.setText(f"Fit {len(files)} files" if len(files) > 1 else "Fit curve")

    def _use_current(self):
        self.files = []
        self.input_label.setText(
            "Current cut — native points, q converted to nm⁻¹ by the fitting workspace."
        )
        self.run_button.setText("Fit curve")

    def start(self):
        if self.job is not None:
            return
        if not self.can_start():
            self.status.setText(
                "A fitting / in-situ job is already running. Finish or cancel it first."
            )
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
                    raise ValueError("Load a curve or select files first")
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
            self.status.setText(
                "Starting the experimental single-RC specialist… Checking the known composition and input scope."
                if method == "stable"
                else "Starting numerical physical fitting…"
                if method == "experimental"
                else "Loading experimental General V5… First run includes loading and compilation."
            )
            self.job.start()
        except Exception as exc:
            QMessageBox.warning(self, "1D fitting", str(exc))

    def _set_running(self, running):
        for widget in (self.run_button, self.load_button, self.current_button, self.settings_panel):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _progress(self, progress):
        self.status.setText(progress.message)
        self.log.append(progress.message)
        self.progress.setValue(int(100 * progress.fraction))

    def _completed(self, result):
        if isinstance(result, Exception) or not result.succeeded:
            message = (
                str(result)
                if isinstance(result, Exception)
                else (result.error.message if result.error else result.status)
            )
            self.status.setText(message)
            self.log.append(message)
            return
        value = result.value
        self.set_results(value["candidates"], value["output_dir"])
        quality_status = self.status.text()
        failures = sum(r["status"] == "failed" for r in value["records"])
        self.status.setText(
            f"Finished in {value['summary']['runtime_seconds']:.2f} s · {failures} failed files. {quality_status}"
        )
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

    def set_results(self, rows, output_dir):
        self.rows = list(rows)
        self.output_dir = Path(output_dir)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            error = r.get("best_log_rmse")
            stage = r["best_source"]
            stage_label = {
                "stable_amplitude_calibrated": "Model + amplitude",
                "stable_neural": "Neural model",
                "stable_numerical_fallback": "Numerical fallback",
            }.get(stage, stage)
            short_file = "Current" if r.get("file") == "Current curve" else r.get("file", "")
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
                stage_label,
            ]
            for j, v in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(str(v)))
            self.table.item(i, 0).setToolTip(f"{r.get('file', '')} / {r['side']}")
            self.table.item(i, 2).setToolTip(r["combination"])
            stage_detail = [f"Stage: {stage}"]
            if r.get("fallback_reason"):
                stage_detail.append(f"Reason: {r['fallback_reason']}")
            warnings = r.get("warnings") or []
            if isinstance(warnings, str):
                warnings = [warnings]
            stage_detail.extend(str(warning) for warning in warnings)
            self.table.item(i, 5).setToolTip("\n".join(stage_detail))
            self.table.item(i, 3).setToolTip(
                "Natural-log RMSE over the measured curve, including measurement noise. "
                "Review peak positions, overall shape and residuals; no mandatory cutoff is applied."
            )
        if rows:
            self.table.selectRow(0)
        self.export_button.setEnabled(True)
        self.progress.setValue(100)
        self.status.setText(
            f"{len(rows)} candidates saved. Review curve shape and residuals; "
            "observed-data scores include noise and are not probabilities."
        )

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
        self.canvas.draw_idle()
        detail = {
            k: v
            for k, v in r.items()
            if k not in ("native_q", "native_fit", "observed", "sigma", "display_q", "display_fit")
        }
        self.parameters.setPlainText(
            "Lengths: nm. sigma_R/h/D: relative standard deviations.\nMixture weights are not posterior probabilities.\nResolution sigma: nm^-1; nu: dimensionless.\n\n"
            + json.dumps(detail, indent=2, ensure_ascii=False)
        )
        self.candidate_selected.emit(r)

    def _cancel(self):
        if self.job:
            self.job.cancel()
            self.status.setText("Cancelling… completed file results remain saved.")

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
