"""The automatic analysis (no AI) inside the Analyze workspace.

``GuidedAnalysis`` runs GIMaP's standard procedure (``StandardPipeline`` — the same tools the AI
uses) on the frame shown in Analyze, in a worker thread. It owns two widgets that the workspace
hosts: ``controls`` for the Results step (beamtime notes, Run, the questions only a person can
answer: ``AnswerForm``) and ``results`` for the Results tab (``GuidedResultsPanel``). ``find_geometry()``
runs only the calibration part. Signals tell the workspace when a run starts, progresses and ends;
``refresh_language()`` composes what it shows again after a switch of the interface language.

The results belong to one file and, in a series, to the frames the run analysed: ``frame_shown(path)``
puts away the results of another file (they come back with it) and turns off what acts on the frame
while Analyze shows other frames of the series, so a report is never saved or sent to Fitting next to
another frame. ``files_cleared()`` (Clear, a project opened) forgets every report and earlier answer
(``GuidedFramesMixin``); a run acts only on the file it started on (``pin_frame``), so a step still in
progress never touches the next one.
"""

from __future__ import annotations

import dataclasses
import re
import threading
from typing import Callable, Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

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
    series_changes,
    step_text,
)
from src.gimap.app.presentation.i18n import current_language, tr, trf

from .gui_bridge import GuiBridge
from .gui_workbench import GuiWorkbench
from .guided_frames import GuidedFramesMixin, frame_key
from .guided_progress import GuidedProgressPanel
from .guided_questions import AnswerForm
from .guided_report import report_markdown, report_page
from .guided_results import GuidedResultsPanel
from .guided_text import (
    GEOMETRY_MESSAGE,
    IDLE_TEXT,
    RUN_MESSAGE,
    START_MESSAGE,
    ask_save_path,
    changes_summary,
    detected_technique,
    label,
    notes_found,
    run_status,
    save_failed_toast,
    saved_toast,
    step_summary,
)

KEPT_REPORTS = 24
"""Reports kept per file for this session (a file shown again gets its results back)."""
_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]*")


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


class GuidedAnalysis(GuidedFramesMixin, QObject):
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
        self.report: Optional[dict] = None
        self.start_report: Optional[dict] = None
        self.results_record: Optional[RunResults] = None
        self._saying: Callable[[], str] = lambda: tr(IDLE_TEXT)
        """What the line under Run says, composed again after a switch of the interface language (``_say``)."""
        self._reports: dict[str, tuple[dict, Optional[dict]]] = {}
        """(report, start-of-series report) per file (``frame_key``), the newest last."""
        self._shown_path: Optional[str] = None
        """The file Analyze shows (``frame_shown``); None until it says."""
        self._pending_path: Optional[str] = None
        """A file shown while a run was active: looked at when the run ends."""
        self._frames_given: Optional[tuple[str, int, int]] = None  # (file, first, summed) said by frame_shown
        self._frames_elsewhere = False  # the status says the results are for other frames of the series
        self._clear_after_run = False  # Analyze was cleared during a run: all is forgotten when it ends
        self._step_arguments: dict[str, dict] = {}
        self.controls = self._build_controls(parent)
        self.progress_panel = GuidedProgressPanel(parent)
        # A narrow Results tab wraps the title ("Automatic analysis — done") instead of pushing the tab wider
        # than its view, which clipped every line of the results on the right.
        self.progress_panel.title_label.setWordWrap(True)
        self.progress_panel.stopRequested.connect(self.stop)
        self.progress_panel.saveRequested.connect(self.save_report)
        self.progress_panel.discardRequested.connect(self.discard)
        self.results = GuidedResultsPanel(automation, parent)
        self.results.compareRequested.connect(self.compare_start)
        self.results.saveRequested.connect(self.save_report)
        self.results.refineRequested.connect(lambda: self._for_this_frame() and self.refineRequested.emit())
        self.results.solutionRequested.connect(lambda row: self._for_this_frame() and self.solutionRequested.emit(row))

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
        self.run_button.setToolTip("The standard procedure without AI: geometry, mask, cuts and results, each step "
                                   "with what it found")
        self.run_button.setProperty("gimapRole", "primary")
        layout.addWidget(self.run_button, 0, Qt.AlignLeft)
        self.status_label = label(IDLE_TEXT, box, role="muted")
        self.status_label.setObjectName("guidedStatus")
        layout.addWidget(self.status_label)
        self.questions = AnswerForm(self._frame_folder, box)  # Run Again, or Enter in a field
        self.again_button, self.answers_label = self.questions.again_button, self.questions.answers_label
        layout.addWidget(self.questions)
        layout.addStretch(1)  # spare height below the controls, not between them
        self.notes_edit.textChanged.connect(self._read_notes)
        self.run_button.clicked.connect(self.run)
        self.questions.runRequested.connect(self.run)
        return box

    @property
    def question_fields(self) -> dict:
        """The answer field of every question shown (``AnswerForm.fields``)."""
        return self.questions.fields

    @property
    def _answers(self) -> dict:
        """Answers of earlier rounds, kept for this session (``AnswerForm.answers``)."""
        return self.questions.answers

    @_answers.setter
    def _answers(self, answers: dict) -> None:
        self.questions.answers = answers

    def _read_notes(self) -> None:
        self.notes_found.setText(notes_found(self.notes_edit.toPlainText()))

    # -- running -------------------------------------------------------------------------

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _busy(self) -> bool:
        """A run was started and its end is not handled yet: the worker thread may already be gone while
        its result still waits in the event queue (``_done`` clears this on the GUI thread)."""
        return self._thread is not None

    def options(self) -> PipelineOptions:
        self.questions.remember()
        # follow_detection: Auto still undecided (no geometry yet) → the technique it detects once the run applied one.
        values: dict = {"notes": self.notes_edit.toPlainText(), "follow_detection": True}
        for option, text in self._answers.items():
            if option in ("calibration", "standard"):
                values[option] = text
            else:
                try:
                    values[option] = float(text)
                except ValueError:
                    continue
        technique = detected_technique(self._status(), bool(values.get("calibration")))
        if technique is not None:  # Analyze on Auto: the technique it detected, without switching its mode
            values["technique"] = technique
        return PipelineOptions(**values)

    def _status(self) -> dict:
        """What Analyze shows (``AnalyzeAutomation.status``); empty when there is no Analyze to ask."""
        try:
            status = self._automation().status()
        except Exception:  # no page (tests, a window being closed): nothing is known
            return {}
        return status if isinstance(status, dict) else {}

    def run(self) -> None:
        if self._busy():
            return
        self.start_report = None
        self._launch(self.options(), self._finished, RUN_MESSAGE)

    def find_geometry(self) -> None:
        """Only the geometry: look for a calibration near the data, fit and check it, save the profile."""
        if self._busy():
            return
        options = dataclasses.replace(self.options(), stop_after_geometry=True)
        self._launch(options, self._finished, GEOMETRY_MESSAGE)

    def compare_start(self) -> None:
        """The start of the series (frames 1–10) with the same settings, then Analyze back at the end."""
        if self._busy() or self.report is None:
            return
        end = self.report.get("frames") or {}
        options = dataclasses.replace(self.options(), frame=1, sum_frames=SERIES_SUM)
        restore = {"frame": -1, "sum": int(end.get("summed") or SERIES_SUM)}
        self._launch(options, self._compared, START_MESSAGE, restore=restore)

    def _launch(self, options: PipelineOptions, on_finished, message: str, *, restore: Optional[dict] = None) -> None:
        goals = AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO, instructions=options.notes)
        self.results_record = RunResults()
        # pin_frame: only the run's file is acted on; a Clear ends a wait for a re-analysis it dropped (never done).
        workbench = GuiWorkbench(self._automation(), self._bridge, pin_frame=True, cancelled=lambda: self._clear_after_run)
        catalog = ToolCatalog(workbench, goals, self.results_record, explorer=self._explorer,
                              calibrator=self._calibrator, fitter=self._fitter)
        worker = _Worker()
        worker.stepped.connect(self._stepped)
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
        self._step_arguments = {}
        self._frames_elsewhere = False
        self.run_button.setEnabled(False)
        self.again_button.setEnabled(False)
        text = self._say(lambda: tr(message))
        self.progress_panel.start(message, geometry_only=options.stop_after_geometry)  # translates it itself
        self.started.emit(text)
        self._thread = threading.Thread(target=worker.run, args=(job,), name="gimap-guided", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """End the run before its next step (a step in progress finishes first); results so far are kept."""
        if not self.running() or self._stop_event is None or self._stop_event.is_set():
            return
        self._stop_event.set()
        self.progress_panel.stopping()
        self._say(lambda: tr("Stopping after the current step …"))
        self.stopping.emit()

    def discard(self) -> None:
        """Forget the results of a stopped run (the Results tab is empty again)."""
        if self.report is not None:
            self._reports.pop(frame_key(self.report.get("frame")), None)
        self.report = None
        self._frames_elsewhere = False
        self.results.hide()
        self.progress_panel.dismiss()
        self._say(lambda: tr("The results of the stopped run were discarded."))
        self._sync_actions()

    def _stepped(self, event: dict) -> None:
        if event.get("state") == "start":
            self._step_arguments[str(event.get("tool") or "")] = dict(event.get("arguments") or {})

    def _progress(self, text: str) -> None:
        self.progressed.emit(self._say(lambda: self._readable(text)))

    def _readable(self, text: str) -> str:
        """``"fit_horizontal_cut: …"`` (the pipeline's line, also used by the command line) in words."""
        name, sep, summary = str(text).partition(": ")
        if not sep or not _TOOL_NAME.fullmatch(name):
            return text
        arguments = self._step_arguments.get(name)
        step = step_text(name, arguments, tr)
        summary = step_summary(name, arguments, summary, current_language())
        return f"{step} — {summary}" if summary else step

    def _done(self) -> None:
        self._thread = None
        self.run_button.setEnabled(True)
        self.again_button.setEnabled(True)

    def _after_run(self) -> None:
        """The file shown during the run is looked at now (or all is forgotten: Analyze was cleared meanwhile;
        a file shown after the Clear, such as a project's, is then the one shown, without another frame_shown)."""
        pending, self._pending_path = self._pending_path, None
        if self._clear_after_run:
            given = self._frames_given
            self._forget_all()
            self._shown_path = pending
            self._frames_given = given if given is not None and given[0] == pending else None
            return
        if pending is not None and pending != self._shown_path:
            self.frame_shown(pending)
        else:
            self._follow_frames()

    def _failed(self, message: str) -> None:
        self._done()
        self.progress_panel.finish(None, failed=message)
        self._say(lambda: trf("The analysis stopped: {message}", message=message))
        self.failed.emit(message)
        self._after_run()

    def _status_text(self, report: dict) -> str:
        """The line under Run: what the run found and where to look (questions only when there are fields below)."""
        return run_status(report, len(self.question_fields))

    def _keep(self, report: dict, start_report: Optional[dict]) -> None:
        key = frame_key(report.get("frame"))
        if key is None:
            return
        self._reports.pop(key, None)
        self._reports[key] = (report, start_report)
        while len(self._reports) > KEPT_REPORTS:
            self._reports.pop(next(iter(self._reports)))

    def _finished(self, report: dict) -> None:
        self._done()
        try:
            self.report = report
            self._keep(report, self.start_report)
            if self._shown_path is None:
                self._shown_path = frame_key(report.get("frame"))
            self._show_questions(report.get("needs_attention") or [])
            self.progress_panel.finish(report, failed=str(report.get("failed") or ""))  # the frame changed: an error
            self._say(lambda: self._status_text(report))
            self._frames_elsewhere = False
            self.results.show_report(report, self.start_report)
            self._sync_actions()
        finally:  # Analyze always hears how a run it was told about ended (it re-enables its buttons then)
            self.finished.emit(report)
            self._after_run()

    def _compared(self, report: dict) -> None:
        self._done()
        self.start_report = report
        try:
            if self.report is not None:
                self._keep(self.report, self.start_report)
                self.results.show_report(self.report, self.start_report)
                self._sync_actions()
            changes = series_changes(report, self.report or {})
            self._say(lambda: trf("Start versus end: {summary}.", summary=changes_summary(changes)) if report.get("ok")
                      else tr("The start of the series could not be analysed."))
        finally:  # as in _finished; without the end's report (discarded meanwhile) the run counts as failed
            if self.report is not None:
                self.finished.emit(self.report)
            else:
                self.failed.emit(tr("The start of the series could not be analysed."))
            self._after_run()

    def _show_questions(self, attention: list) -> None:
        """One field per value only a person knows (``AnswerForm``); answers of earlier rounds are kept."""
        self.questions.show_questions(attention)

    # -- the interface language ----------------------------------------------------------------

    def _say(self, compose: Callable[[], str]) -> str:
        """Show ``compose()`` under Run; it is composed again after a switch of the interface language."""
        self._saying = compose
        text = compose()
        self.status_label.setText(text)
        return text

    def refresh_language(self) -> None:
        """After a switch of the interface language: the line under Run, what the notes give, the questions
        (what is typed stays), the progress panel and the Results tab (the rows selected stay)."""
        self.status_label.setText(self._saying())
        self._read_notes()
        self.questions.refresh_language()
        self.progress_panel.refresh_language()
        if self.report is not None and not self.results.isHidden():
            self.results.refresh_language()

    # -- the report --------------------------------------------------------------------------

    def report_markdown(self) -> str:
        return report_markdown(self.report or {}, self.start_report)

    def report_page(self) -> str:
        """The report as one web page with its pictures: the q map and I(q) with its peaks (GIWAXS) or the fit (GISAXS)."""
        return report_page(self.report or {}, self.start_report, self._automation)

    def save_report(self) -> Optional[str]:
        """The report as a web page (pictures drawn from the frame in Analyze) or Markdown, next to the data."""
        if self.report is None or self._save_text is None or not self._for_this_frame():
            return None
        path, chosen = ask_save_path(
            self.controls, "Save Report", self.report.get("frame"), "report.html",
            f"{tr('Web page with pictures')} (*.html);;{tr('Markdown text')} (*.md)",
        )
        if not path:
            return None
        markdown = path.lower().endswith(".md") or (chosen.endswith("(*.md)") and not path.lower().endswith(".html"))
        try:
            written = self._save_text(path, self.report_markdown() if markdown else self.report_page())
        except OSError as exc:
            save_failed_toast(self.controls, path, exc.strerror or str(exc))
            return None
        saved_toast(self.controls, written or path)
        return written


__all__ = ["GuidedAnalysis"]
