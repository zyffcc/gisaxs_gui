"""The automatic analysis (no AI) inside the Analyze workspace.

``GuidedAnalysis`` runs GIMaP's standard procedure (``StandardPipeline`` — the same tools the AI
uses) on the frame shown in Analyze, in a worker thread. It owns two widgets that the workspace
hosts: ``controls`` for the Results step (beamtime notes, Run, the questions only a person can
answer) and ``results`` for the Results tab (``GuidedResultsPanel``). ``find_geometry()`` runs only
the calibration part. Signals tell the workspace when a run starts, progresses and ends.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..application import (
    GOALS,
    PERMISSION_AUTO,
    SERIES_SUM,
    AnalysisGoals,
    PipelineOptions,
    RunResults,
    StandardPipeline,
    ToolCall,
    ToolCatalog,
    energy_from_notes,
    incidence_from_notes,
    pixel_size_from_notes,
    series_changes,
)
from .gui_bridge import GuiBridge
from .gui_workbench import GuiWorkbench
from .guided_report import report_markdown, report_page
from .guided_progress import GuidedProgressPanel
from .guided_results import GuidedResultsPanel
from .guided_text import OPTION_FIELDS, label


class _Worker(QObject):
    progressed = pyqtSignal(str)
    stepped = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self, job: Callable[[], dict]) -> None:
        try:
            self.finished.emit(job())
        except Exception as exc:  # shown on the page, never raised into Qt
            self.failed.emit(str(exc) or type(exc).__name__)


class GuidedAnalysis(QObject):
    started = pyqtSignal(str)
    progressed = pyqtSignal(str)
    finished = pyqtSignal(object)
    """The report (a dict); ``report["ok"]`` says whether there are results."""
    failed = pyqtSignal(str)
    refineRequested = pyqtSignal()
    """GISAXS: open the prepared curve in Fitting (the page's Send to Fitting)."""
    solutionRequested = pyqtSignal(dict)
    """GISAXS: open the prepared curve in Fitting with this solution (a ``native_v5`` candidate) drawn."""
    stopping = pyqtSignal()
    """Stop was asked for: the run ends before its next step."""

    def __init__(
        self,
        automation: Callable[[], object],
        *,
        explorer=None,
        calibrator=None,
        fitter=None,
        save_text: Optional[Callable[[str, str], str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._automation = automation
        self._explorer = explorer
        self._calibrator = calibrator
        self._fitter = fitter
        self._save_text = save_text
        self._bridge = GuiBridge(self)
        self._thread: Optional[threading.Thread] = None
        self._worker: Optional[_Worker] = None
        self._stop_event: Optional[threading.Event] = None
        self._answers: dict[str, str] = {}
        """Answers of earlier rounds: kept when the next question replaces the fields."""
        self.report: Optional[dict] = None
        self.start_report: Optional[dict] = None
        self.results_record: Optional[RunResults] = None
        self.question_fields: dict[str, QLineEdit] = {}
        self.controls = self._build_controls(parent)
        self.progress_panel = GuidedProgressPanel(parent)
        self.progress_panel.stopRequested.connect(self.stop)
        self.progress_panel.saveRequested.connect(self.save_report)
        self.progress_panel.discardRequested.connect(self.discard)
        self.results = GuidedResultsPanel(automation, parent)
        self.results.compareRequested.connect(self.compare_start)
        self.results.saveRequested.connect(self.save_report)
        self.results.refineRequested.connect(self.refineRequested)
        self.results.solutionRequested.connect(self.solutionRequested)

    # -- the controls in the Results step --------------------------------------------------

    def _build_controls(self, parent: Optional[QWidget]) -> QWidget:
        box = QWidget(parent)
        box.setObjectName("guidedControls")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(label(
            "Beamtime notes (optional): αi, energy, pixel size and where the calibration is — whatever you know.",
            box, role="muted",
        ))
        self.notes_edit = QPlainTextEdit(box)
        self.notes_edit.setObjectName("guidedNotes")
        self.notes_edit.setPlaceholderText("e.g. P03, GIWAXS, alpha_i = 0.4 deg, 11.8 keV, calibration in D:\\beamtime\\calib")
        self.notes_edit.setFixedHeight(74)
        layout.addWidget(self.notes_edit)
        self.notes_found = label("", box, role="muted")
        self.notes_found.setObjectName("guidedNotesFound")
        layout.addWidget(self.notes_found)
        self.run_button = QPushButton("Run Automatic Analysis", box)
        self.run_button.setObjectName("guidedRunButton")
        self.run_button.setProperty("gimapRole", "primary")
        layout.addWidget(self.run_button, 0, Qt.AlignLeft)
        self.status_label = label("No AI needed: the standard procedure, each decision with its reason.", box, role="muted")
        self.status_label.setObjectName("guidedStatus")
        layout.addWidget(self.status_label)
        self.questions = QFrame(box)
        self.questions.setObjectName("guidedQuestions")
        self.questions.setProperty("gimapInfoCard", True)
        questions = QVBoxLayout(self.questions)
        questions.addWidget(label("Only you can answer these — then run again:", self.questions, bold=True))
        self.question_form = QFormLayout()
        questions.addLayout(self.question_form)
        self.again_button = QPushButton("Run Again with These Answers", self.questions)
        self.again_button.setObjectName("guidedAgainButton")
        questions.addWidget(self.again_button, 0, Qt.AlignLeft)
        self.answers_label = label("", self.questions, role="muted")
        questions.addWidget(self.answers_label)
        self.questions.hide()
        layout.addWidget(self.questions)
        self.notes_edit.textChanged.connect(self._read_notes)
        self.run_button.clicked.connect(self.run)
        self.again_button.clicked.connect(self.run)
        return box

    def _read_notes(self) -> None:
        text = self.notes_edit.toPlainText()
        found = []
        for name, value, unit in (
            ("αi", incidence_from_notes(text), "°"), ("energy", energy_from_notes(text), " keV"),
            ("pixel", pixel_size_from_notes(text), " µm"),
        ):
            if value is not None:
                found.append(f"{name} = {value:g}{unit}")
        self.notes_found.setText("Found in the notes: " + ", ".join(found) if found else "")

    # -- running -------------------------------------------------------------------------

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _remember_answers(self) -> None:
        for option, field in self.question_fields.items():
            if field.text().strip():
                self._answers[option] = field.text().strip()

    def options(self) -> PipelineOptions:
        self._remember_answers()
        values: dict = {"notes": self.notes_edit.toPlainText()}
        for option, text in self._answers.items():
            if option in ("calibration", "standard"):
                values[option] = text
            else:
                try:
                    values[option] = float(text)
                except ValueError:
                    continue
        return PipelineOptions(**values)

    def run(self) -> None:
        if self.running():
            return
        self.start_report = None
        self._launch(self.options(), self._finished, "Working… (finding a calibration can take a minute)")

    def find_geometry(self) -> None:
        """Only the geometry: look for a calibration near the data, fit and check it, save the profile."""
        if self.running():
            return
        options = dataclasses.replace(self.options(), stop_after_geometry=True)
        self._launch(options, self._finished, "Looking for a calibration near the data…")

    def compare_start(self) -> None:
        """The start of the series (frames 1–10) with the same settings, then Analyze back at the end."""
        if self.running() or self.report is None:
            return
        end = self.report.get("frames") or {}
        options = dataclasses.replace(self.options(), frame=1, sum_frames=SERIES_SUM)
        restore = {"frame": -1, "sum": int(end.get("summed") or SERIES_SUM)}
        self._launch(options, self._compared, "Analysing the start of the series…", restore=restore)

    def _launch(self, options: PipelineOptions, on_finished, message: str, *, restore: Optional[dict] = None) -> None:
        goals = AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO, instructions=options.notes)
        self.results_record = RunResults()
        catalog = ToolCatalog(
            GuiWorkbench(self._automation(), self._bridge), goals, self.results_record,
            explorer=self._explorer, calibrator=self._calibrator, fitter=self._fitter,
        )
        worker = _Worker()
        worker.progressed.connect(self._progress)
        worker.stepped.connect(self.progress_panel.step)
        worker.finished.connect(on_finished)
        worker.failed.connect(self._failed)
        self._stop_event = threading.Event()
        pipeline = StandardPipeline(
            catalog, options, worker.progressed.emit, stop=self._stop_event, events=worker.stepped.emit,
        )

        def job() -> dict:
            report = pipeline.run()
            if restore is not None:
                catalog.execute(ToolCall("restore", "set_frame", restore))
            return report

        self._worker = worker
        self.run_button.setEnabled(False)
        self.again_button.setEnabled(False)
        self.status_label.setText(message)
        self.progress_panel.start(message, geometry_only=options.stop_after_geometry)
        self.started.emit(message)
        self._thread = threading.Thread(target=worker.run, args=(job,), name="gimap-guided", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """End the run before its next step (a step in progress finishes first); results so far are kept."""
        if not self.running() or self._stop_event is None or self._stop_event.is_set():
            return
        self._stop_event.set()
        self.progress_panel.stopping()
        self.status_label.setText("Stopping after the current step …")
        self.stopping.emit()

    def discard(self) -> None:
        """Forget the results of a stopped run (the Results tab is empty again)."""
        self.report = None
        self.results.hide()
        self.progress_panel.dismiss()
        self.status_label.setText("The results of the stopped run were discarded.")

    def _progress(self, text: str) -> None:
        self.status_label.setText(text)
        self.progressed.emit(text)

    def _done(self) -> None:
        self._thread = None
        self.run_button.setEnabled(True)
        self.again_button.setEnabled(True)

    def _failed(self, message: str) -> None:
        self._done()
        self.progress_panel.finish(None, failed=message)
        self.status_label.setText(f"The analysis stopped: {message}")
        self.failed.emit(message)

    def _finished(self, report: dict) -> None:
        self._done()
        self.report = report
        self._show_questions(report.get("needs_attention") or [])
        attention = report.get("needs_attention") or []
        self.progress_panel.finish(report)
        if report.get("stopped"):
            self.status_label.setText("Stopped by you — what was found so far is in the Results tab.")
        elif report.get("ok"):
            self.status_label.setText("Done — see the Results tab." + (f" {len(attention)} question(s) below." if attention else ""))
        else:
            self.status_label.setText("Needs your answers below before it can give results.")
        self.results.show_report(report, self.start_report)
        self.finished.emit(report)

    def _compared(self, report: dict) -> None:
        self._done()
        self.start_report = report
        if self.report is not None:
            self.results.show_report(self.report, self.start_report)
        kinds: dict[str, int] = {}
        for row in series_changes(report, self.report or {}):
            if row["change"] != "present at both":
                kind = row["change"].split(" ")[0]
                kinds[kind] = kinds.get(kind, 0) + 1
        summary = ", ".join(f"{count} {kind}" for kind, count in kinds.items()) or "no line changed"
        self.status_label.setText(
            f"Start versus end: {summary}." if report.get("ok") else "The start of the series could not be analysed."
        )
        if self.report is not None:
            self.finished.emit(self.report)

    def _show_questions(self, attention: list) -> None:
        self._remember_answers()
        while self.question_form.rowCount():
            self.question_form.removeRow(0)
        self.question_fields = {}
        for item in attention:
            option = item.get("option")
            if option not in OPTION_FIELDS or option in self.question_fields:
                continue
            title, placeholder = OPTION_FIELDS[option]
            field = QLineEdit(self.questions)
            field.setObjectName(f"guidedAnswer_{option}")
            field.setPlaceholderText(placeholder)
            field.setToolTip(f"{item['why']}\n{item.get('hint', '')}")
            field.setText(self._answers.get(option, ""))
            self.question_form.addRow(title, field)
            self.question_fields[option] = field
        given = [f"{OPTION_FIELDS[key][0]}: {value}" for key, value in self._answers.items() if key not in self.question_fields]
        self.answers_label.setText("Your earlier answers are kept: " + "; ".join(given) if given else "")
        self.questions.setVisible(bool(self.question_fields))

    # -- the report --------------------------------------------------------------------------

    def report_markdown(self) -> str:
        return report_markdown(self.report or {}, self.start_report)

    def report_page(self) -> str:
        """The report as one web page with the q map (rings drawn) and I(q) (peaks marked)."""
        return report_page(self.report or {}, self.start_report, self._automation)

    def save_report(self) -> Optional[str]:
        if self.report is None or self._save_text is None:
            return None
        stem = Path(str(self.report.get("frame") or "giwaxs")).stem
        path, chosen = QFileDialog.getSaveFileName(
            self.controls, "Save Report", f"{stem}_report.html", "Web page with pictures (*.html);;Markdown text (*.md)",
        )
        if not path:
            return None
        markdown = path.lower().endswith(".md") or (chosen.startswith("Markdown") and not path.lower().endswith(".html"))
        return self._save_text(path, self.report_markdown() if markdown else self.report_page())


__all__ = ["GuidedAnalysis"]
