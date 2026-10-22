"""Sputter workspace: particle size R and spacing D over an in-situ sputter series, and how far to trust them.

Left: open a finished result folder, or analyze a new scan (the model runs in
the external sputter toolkit, see Toolkit). Right: Size Trend (R, D and the
TS length L with simulation-based error bars, coloured by grade), Windows
(every time window, value, source and reason when missing) and Reliability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QTabWidget,
                             QTableWidget, QTableWidgetItem, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
                             QCheckBox, QSizePolicy)

from src.gimap.app.presentation.components import show_toast

from ..application import (GRADES, AnalyzeRequest, SputterService, StructureResult, overall_fraction, parse_progress,
                           reliability_report)

PREFERENCES_KEY = "sputter_page"
TASK_KEY = "sputter_analyze"
COLORS = {"A": "#1f5fa8", "B": "#7fa3d6", "C": "#d07a10", "S": "#2a8a3e", "T": "#c0392b", "L": "#6a3d9a"}
CHOICES = {"window": "Per window (network / refined)", "selected": "Learned selector",
           "sequence": "Physics path (exploratory)"}
TABLE_COLUMNS = (
    ("time_s", "Time (s)"), ("stage", "Stage"), ("module", "Model"),
    ("R_nm", "R (nm)"), ("R_grade", "R grade"), ("R_typical_rel_error", "R typ. error"),
    ("D_nm", "D (nm)"), ("D_grade", "D grade"), ("D_typical_rel_error", "D typ. error"), ("D_corr_nm", "2π/q* (nm)"),
    ("D_over_direct", "D ÷ 2π/q*"), ("D_direct_check", "D check"),
    ("L_TS_nm", "TS length L (nm)"), ("refine_status", "Refinement"), ("coequal_alternatives", "Alternatives"),
    ("sequence_status", "Path"), ("sequence_data_cost_nats", "Data cost (nats)"),
    ("selector_probability", "Selector P"), ("selector_choice", "Component"),
    ("R_reason", "R note"), ("D_reason", "D note"),
)


def _float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (AttributeError, ValueError):
        return None


class _PathField(QWidget):
    def __init__(self, placeholder: str, *, folder: bool, filter_: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)
        self.edit = QLineEdit(self); self.edit.setPlaceholderText(placeholder)
        self.button = QToolButton(self); self.button.setText("…")
        lay.addWidget(self.edit, 1); lay.addWidget(self.button)
        self._folder, self._filter = folder, filter_
        self.button.clicked.connect(self._browse)

    def _browse(self):
        start = self.edit.text() or str(Path.home())
        if self._folder:
            path = QFileDialog.getExistingDirectory(self, "Choose Folder", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose File", start, self._filter)
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, text: str) -> None:
        self.edit.setText(text or "")


class SputterPage(QWidget):
    lineReceived = pyqtSignal(str)

    def __init__(self, service: SputterService, *, task_runner=None, preferences=None, parent=None):
        super().__init__(parent)
        self.setObjectName("sputterPage")
        self.service, self.task_runner, self.preferences = service, task_runner, preferences
        self.result: Optional[StructureResult] = None
        self._stop = False
        self._build()
        self._restore()
        self.lineReceived.connect(self._on_line)

    # -- layout ---------------------------------------------------------------
    def _build(self):
        outer = QHBoxLayout(self); outer.setContentsMargins(12, 12, 12, 12); outer.setSpacing(12)
        scroll = QScrollArea(self); scroll.setWidgetResizable(True); scroll.setFixedWidth(460)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left = QWidget(); scroll.setWidget(left); col = QVBoxLayout(left); col.setSpacing(10)
        title = QLabel("<b>Sputter series</b><br>Particle size R and spacing D over time, and how far each value can be "
                       "trusted.", left)
        title.setWordWrap(True); col.addWidget(title)

        results = QGroupBox("Results", left); rl = QVBoxLayout(results)
        self.open_button = QPushButton("Open Results Folder…", results)
        self.merge_button = QPushButton("Add Refined Windows…", results)
        self.merge_button.setToolTip("Merge physical-refinement windows (one JSON per window, e.g. from the cluster) "
                                     "into the current result.")
        self.result_label = QLabel("No result open.", results); self.result_label.setWordWrap(True)
        choice_row = QWidget(results); cr = QHBoxLayout(choice_row); cr.setContentsMargins(0, 0, 0, 0)
        cr.addWidget(QLabel("Explanation", choice_row))
        self.choice_box = QComboBox(choice_row); self.choice_box.setEnabled(False)
        self.choice_box.setToolTip(
            "How one of the network's explanations is chosen per window.\n"
            "Per window: highest network weight, or the converged refinement (grades A/B/C).\n"
            "Learned selector: trained on simulated growth series; uses the weights, agreement of D with 2π/q* and "
            "agreement with neighbouring windows (grade L; its tested accuracy is on the Reliability tab).\n"
            "Physics path (exploratory): registered physical rules; not more accurate in simulation (grades S/T).")
        cr.addWidget(self.choice_box, 1)
        rl.addWidget(self.open_button); rl.addWidget(self.merge_button); rl.addWidget(choice_row)
        rl.addWidget(self.result_label)
        col.addWidget(results)

        scan = QGroupBox("Analyze a Scan", left); form = QFormLayout(scan); form.setLabelAlignment(Qt.AlignRight)
        self.scan_field = _PathField("folder with the CBF frames", folder=True)
        self.out_field = _PathField("new output folder", folder=True)
        self.energy_kind = QComboBox(); self.energy_kind.addItems(["Energy (keV)", "Wavelength (nm)"])
        self.energy_value = QDoubleSpinBox(); self.energy_value.setDecimals(5); self.energy_value.setRange(0.001, 200.)
        self.energy_value.setValue(12.0)
        energy_row = QWidget(); er = QHBoxLayout(energy_row); er.setContentsMargins(0, 0, 0, 0)
        er.addWidget(self.energy_kind); er.addWidget(self.energy_value, 1)
        self.sdd = QDoubleSpinBox(); self.sdd.setRange(1., 100000.); self.sdd.setDecimals(1); self.sdd.setValue(3387.)
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0.001, 10.); self.alpha.setDecimals(3); self.alpha.setValue(0.4)
        self.beam_col = QLineEdit(); self.beam_col.setPlaceholderText("auto")
        self.beam_row = QLineEdit(); self.beam_row.setPlaceholderText("auto")
        beam_row = QWidget(); br = QHBoxLayout(beam_row); br.setContentsMargins(0, 0, 0, 0)
        br.addWidget(QLabel("col")); br.addWidget(self.beam_col); br.addWidget(QLabel("row")); br.addWidget(self.beam_row)
        self.roll = QDoubleSpinBox(); self.roll.setRange(-10., 10.); self.roll.setDecimals(3)
        self.metal = QComboBox(); self.metal.addItems(["Ag", "Au", "Cu"])
        self.polymer = QComboBox(); self.polymer.setEditable(True); self.polymer.addItems(["PSBMA", "PS", "PEO", "polymer"])
        self.mask_field = _PathField("optional detector mask (.npy)", folder=False, filter_="NumPy mask (*.npy);;All files (*)")
        self.window = QDoubleSpinBox(); self.window.setRange(0.05, 60.); self.window.setDecimals(2); self.window.setValue(0.5)
        self.device = QComboBox(); self.device.addItems(["cpu", "cuda"])
        self.workers = QSpinBox(); self.workers.setRange(1, 64); self.workers.setValue(4)
        form.addRow("Scan", self.scan_field); form.addRow("Output", self.out_field)
        form.addRow("X-rays", energy_row); form.addRow("Distance (mm)", self.sdd); form.addRow("Incidence αi (°)", self.alpha)
        form.addRow("Beam centre (px)", beam_row); form.addRow("Sample roll (°)", self.roll)
        form.addRow("Metal", self.metal); form.addRow("Polymer", self.polymer); form.addRow("Mask", self.mask_field)
        form.addRow("Time window (s)", self.window); form.addRow("Device", self.device); form.addRow("Workers", self.workers)
        buttons = QWidget(); bl = QHBoxLayout(buttons); bl.setContentsMargins(0, 0, 0, 0)
        self.run_button = QPushButton("Analyze"); self.stop_button = QPushButton("Stop"); self.stop_button.setEnabled(False)
        bl.addWidget(self.run_button, 1); bl.addWidget(self.stop_button)
        form.addRow(buttons)
        self.progress = QProgressBar(); self.progress.setRange(0, 1000); self.progress.setValue(0)
        self.status = QLabel(""); self.status.setWordWrap(True)
        form.addRow(self.progress); form.addRow(self.status)
        col.addWidget(scan)

        tool = QGroupBox("Toolkit", left); tf = QFormLayout(tool)
        self.python_field = _PathField("python.exe with PyTorch", folder=False, filter_="Python (python*.exe python*);;All files (*)")
        self.toolkit_field = _PathField("folder of sputter_struct_v1", folder=True)
        note = QLabel("The model runs in its own Python (PyTorch, NumPy, SciPy, fabio, Matplotlib), so GIMaP itself needs "
                      "no PyTorch. Reading the frames takes most of the time.")
        note.setWordWrap(True)
        tf.addRow("Python", self.python_field); tf.addRow("Folder", self.toolkit_field); tf.addRow(note)
        col.addWidget(tool)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(400)
        self.log.setMinimumHeight(110); self.log.setPlaceholderText("Toolkit messages")
        col.addWidget(self.log); col.addStretch(1)
        outer.addWidget(scroll)

        self.tabs = QTabWidget(self)
        trend = QWidget(); tl = QVBoxLayout(trend); tl.setContentsMargins(4, 4, 4, 4)
        self.log_scale = QCheckBox("Logarithmic size axis"); self.log_scale.setChecked(True)
        tl.addWidget(self.log_scale)
        self.canvas = None
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.figure = Figure(figsize=(7, 6), dpi=100, constrained_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            tl.addWidget(self.canvas, 1)
        except Exception as exc:  # matplotlib missing: keep the table and the report
            tl.addWidget(QLabel(f"Plotting is unavailable: {exc}"))
        self.tabs.addTab(trend, "Size Trend")
        from .sketch_view import SketchView
        self.sketch = SketchView()
        self.tabs.addTab(self.sketch, "Structure Sketch")
        self.table = QTableWidget(0, len(TABLE_COLUMNS)); self.table.setHorizontalHeaderLabels([c[1] for c in TABLE_COLUMNS])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers); self.table.setSortingEnabled(False)
        self.tabs.addTab(self.table, "Windows")
        self.report = QTextBrowser(); self.report.setOpenExternalLinks(False)
        self.report.setHtml("<p>Open a result folder or analyze a scan. Each R and D value gets a grade: "
                            "<b>A</b> refined (unique), <b>B</b> refined (an equally good alternative exists), "
                            "<b>C</b> network estimate only.</p>")
        self.tabs.addTab(self.report, "Reliability")
        outer.addWidget(self.tabs, 1)

        self.open_button.clicked.connect(self._choose_results)
        self.merge_button.clicked.connect(self._choose_refined)
        self.choice_box.currentIndexChanged.connect(self._choice_changed)
        self.run_button.clicked.connect(self._analyze)
        self.stop_button.clicked.connect(self._request_stop)
        self.log_scale.toggled.connect(lambda _: self._plot())
        self.merge_button.setEnabled(False)

    # -- preferences ------------------------------------------------------------
    def _settings(self) -> tuple[str, str]:
        return self.python_field.text(), self.toolkit_field.text()

    def _restore(self):
        stored = self.preferences.get(PREFERENCES_KEY, {}) if self.preferences is not None else {}
        stored = stored if isinstance(stored, dict) else {}
        self.python_field.setText(stored.get("python", "")); self.toolkit_field.setText(stored.get("toolkit", ""))
        self.mask_field.setText(stored.get("mask", ""))
        for key, spin in (("sdd", self.sdd), ("alpha", self.alpha), ("energy", self.energy_value), ("window", self.window)):
            if isinstance(stored.get(key), (int, float)):
                spin.setValue(float(stored[key]))
        if stored.get("energy_kind") in (0, 1):
            self.energy_kind.setCurrentIndex(int(stored["energy_kind"]))
        if stored.get("device") in ("cpu", "cuda"):
            self.device.setCurrentText(stored["device"])

    def _remember(self):
        if self.preferences is None:
            return
        self.preferences.set(PREFERENCES_KEY, dict(
            python=self.python_field.text(), toolkit=self.toolkit_field.text(), mask=self.mask_field.text(),
            sdd=self.sdd.value(), alpha=self.alpha.value(), energy=self.energy_value.value(),
            energy_kind=self.energy_kind.currentIndex(), window=self.window.value(), device=self.device.currentText()))

    def toolkit_settings(self) -> tuple[str, str]:
        """For the service: read when a run starts."""
        return self._settings()

    # -- actions ----------------------------------------------------------------
    def _choose_results(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Sputter Result Folder", self.out_field.text() or str(Path.home()))
        if folder:
            self.open_results(Path(folder))

    def open_results(self, folder: Path, variant: Optional[str] = None) -> None:
        try:
            available = self.service.variants(folder)
            wanted = variant or self.choice_box.currentData() or "window"
            self.choice_box.blockSignals(True); self.choice_box.clear()
            for key in available:
                self.choice_box.addItem(CHOICES.get(key, key), key)
            index = max(0, self.choice_box.findData(wanted if wanted in available else "window"))
            self.choice_box.setCurrentIndex(index); self.choice_box.setEnabled(len(available) > 1)
            self.choice_box.blockSignals(False)
            self.show_result(self.service.open(folder, self.choice_box.currentData() or "window"))
        except (OSError, ValueError) as exc:
            self._notify(str(exc), error=True)

    def _choice_changed(self, _index: int) -> None:
        if self.result is not None:
            self.open_results(self.result.folder, self.choice_box.currentData())

    def _request(self) -> AnalyzeRequest:
        value = self.energy_value.value(); is_energy = self.energy_kind.currentIndex() == 0
        return AnalyzeRequest(
            scan_folder=Path(self.scan_field.text()), out_folder=Path(self.out_field.text()), sdd_mm=self.sdd.value(),
            alpha_i_deg=self.alpha.value(), metal=self.metal.currentText(),
            energy_ev=value * 1000. if is_energy else None, wavelength_nm=None if is_energy else value,
            polymer_type=self.polymer.currentText().strip() or "polymer",
            beam_col=_float(self.beam_col.text()), beam_row=_float(self.beam_row.text()),
            sample_roll_deg=self.roll.value(), mask_file=Path(self.mask_field.text()) if self.mask_field.text() else None,
            bin_seconds=self.window.value(), device=self.device.currentText(), workers=self.workers.value())

    def _analyze(self):
        request = self._request()
        problems = request.problems()
        if problems:
            self._notify(" ".join(problems), error=True); return
        self._remember(); self._stop = False
        self.run_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.progress.setValue(0); self.status.setText("Starting the sputter toolkit…")
        job = lambda: self.service.analyze(request, self.lineReceived.emit, lambda: self._stop)
        self._submit(job)

    def _choose_refined(self):
        if self.result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder of Refined Window Files", str(self.result.folder))
        if not folder:
            return
        self._remember(); self.run_button.setEnabled(False); self.status.setText("Merging refined windows…")
        result, metal = self.result, self.result.metal or self.metal.currentText()
        self._submit(lambda: self.service.merge_refined(result.folder, Path(folder), metal, self.lineReceived.emit))

    def _submit(self, job):
        if self.task_runner is None:
            try:
                self._done(job())
            except Exception as exc:  # noqa: BLE001 - shown to the user
                self._failed(str(exc))
            return
        self.task_runner.submit(TASK_KEY, job, on_done=self._done, on_error=self._task_failed)

    def _task_failed(self, message: str, details: str = "") -> None:
        if details:
            self.log.appendPlainText(details)
        self._failed(message)

    def _request_stop(self):
        self._stop = True; self.status.setText("Stopping…")

    def _done(self, result: StructureResult):
        self.run_button.setEnabled(True); self.stop_button.setEnabled(False)
        self.progress.setValue(1000); self.status.setText("Done.")
        self.show_result(result)

    def _failed(self, message: str):
        self.run_button.setEnabled(True); self.stop_button.setEnabled(False)
        self.status.setText(message.splitlines()[0] if message else "Failed")
        self.log.appendPlainText(message)
        self._notify(message.splitlines()[0] if message else "The analysis failed.", error=True)

    def _on_line(self, line: str):
        progress = parse_progress(line)
        if progress is None:
            self.log.appendPlainText(line); return
        self.progress.setValue(int(1000 * overall_fraction(progress)))
        self.status.setText(f"{progress.stage}: {progress.message}")
        self.log.appendPlainText(f"[{progress.stage}] {progress.message}")

    def _notify(self, text: str, *, error: bool = False):
        try:
            show_toast(self, text, level="error" if error else "info")
        except Exception:  # noqa: BLE001 - a missing toast host must not hide the message
            self.status.setText(text)

    # -- showing a result ---------------------------------------------------------
    def show_result(self, result: StructureResult) -> None:
        self.result = result
        self.merge_button.setEnabled(True)
        cR, cD = result.counts("R"), result.counts("D")
        self.result_label.setText(f"<b>{result.folder.name}</b> ({result.metal})<br>{len(result.rows)} windows · R in "
                                  f"{sum(cR[g] for g in GRADES)} · D in {sum(cD[g] for g in GRADES)}<br>refined A+B: "
                                  f"R {cR['A'] + cR['B']}, D {cD['A'] + cD['B']}"
                                  + (f"<br>time-consistent path: S {cR['S']}, T {cR['T']} (R)" if result.variant == "sequence" else "")
                                  + (f"<br>learned selector: L {cR['L']} (R), {cD['L']} (D)" if result.variant == "selected" else ""))
        self.result_label.setToolTip(str(result.folder))
        self._plot(); self._fill_table(); self.report.setHtml(reliability_report(result))
        self.sketch.set_result(result)

    def _plot(self):
        if self.canvas is None:
            return
        fig = self.figure; fig.clear()
        result = self.result
        if result is None:
            self.canvas.draw_idle(); return
        pal = self.palette(); fg = pal.text().color().name(); bg = pal.base().color().name()
        fig.set_facecolor(bg)
        ts = result.ts_length()
        panels = [("R", "R (nm), hemisphere-equivalent"), ("D", "D (nm), peak windows")] + ([("L", "TS length L (nm)")] if ts.time else [])
        axes = fig.subplots(len(panels), 1, sharex=True)
        for ax, (name, label) in zip(axes, panels):
            ax.set_facecolor(bg)
            for spine in ax.spines.values():
                spine.set_color(fg)
            ax.tick_params(colors=fg); ax.yaxis.label.set_color(fg); ax.xaxis.label.set_color(fg)
            if name == "L":
                ax.plot(ts.time, ts.value, "o", ms=2.5, color="#5a8a2a", label="TS length L (network)")
            else:
                for grade in ("C", "B", "A", "S", "T", "L"):
                    s = result.series(name, grade)
                    if not s.time:
                        continue
                    lo = [v - l for v, l in zip(s.value, s.low)]; hi = [h - v for v, h in zip(s.value, s.high)]
                    ax.errorbar(s.time, s.value, yerr=[lo, hi], fmt="o", ms=2.6, lw=.5, color=COLORS[grade], alpha=.85,
                                label=f"{grade}: {GRADES[grade]}")
                if name == "D":
                    bad = result.inconsistent_spacing()
                    if bad.time:
                        ax.plot(bad.time, bad.value, "o", ms=7, mfc="none", mec="#d62728", mew=.9,
                                label="D vs 2π/q* outside simulated 90% range")
                    d = result.direct_spacing()
                    if d.time:
                        ax.plot(d.time, d.value, "x", ms=3, color="0.5", label="2π/q* (direct)")
            if self.log_scale.isChecked():
                ax.set_yscale("log")
            ax.set_ylabel(label); ax.grid(alpha=.25)
            leg = ax.legend(fontsize=7, loc="best")
            if leg is not None:
                leg.get_frame().set_alpha(.7)
        axes[-1].set_xlabel("time (s)")
        self.canvas.draw_idle()

    def _fill_table(self):
        rows = self.result.rows if self.result is not None else []
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, (key, _) in enumerate(TABLE_COLUMNS):
                v = r.get(key)
                if isinstance(v, float):
                    text = f"±{100 * v:.0f}%" if key.endswith("typical_rel_error") else f"{v:.3g}"
                else:
                    text = "" if v is None else str(v)
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()


__all__ = ["SputterPage"]
