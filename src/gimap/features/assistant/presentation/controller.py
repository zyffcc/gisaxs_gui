"""Runs the AI assistant on the frame open in Analyze and shows the run in a dock panel.

The brain is Claude Code on the user's Claude plan (``RunAgentTask``), the
Claude API with a key, or another provider through an OpenAI-compatible API
(DeepSeek, Qwen, OpenAI, Ollama …; both ``RunAssistantTask``).  A run happens on a daemon
thread so the window stays responsive and a stuck request can never keep GIMaP
from closing; every action on the Analyze page reaches the GUI thread through
``GuiBridge``.
"""

from __future__ import annotations

import dataclasses
import html
import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QMessageBox, QVBoxLayout

from src.gimap.app.presentation.task_runner import TaskRunner

from ..application import (
    BACKEND_API,
    BACKEND_CLAUDE_CODE,
    BACKEND_PROVIDER,
    BACKENDS,
    RUN_FAILED,
    AnalysisGoals,
    LlmError,
    LlmUsage,
    RunAgentTask,
    RunAssistantTask,
    RunOutcome,
    RunResults,
    ToolCatalog,
    clean,
    results_payload,
)
from . import preferences
from .choice_dialog import GuiChooser
from .code_section import describe_code_status
from .gui_bridge import GuiBridge
from .gui_workbench import GuiConfirmer, GuiWorkbench
from .panel import AssistantPanel
from .provider_section import chosen_provider, describe_provider
from .report_view import report_html
from .services import AssistantServices
from .settings_page import AssistantSettingsPage
from .start_dialog import AssistantStartDialog

TITLE = "Process with AI"
SHUTDOWN_WAIT_S = 3.0


class _RunWorker(QObject):
    """Runs one task off the GUI thread; its signals arrive on the GUI thread (queued)."""

    stepStarted = pyqtSignal(object)
    stepFinished = pyqtSignal(object)
    modelText = pyqtSignal(str)
    modelProgress = pyqtSignal(str, str)
    noticed = pyqtSignal(str)
    usageChanged = pyqtSignal(object)
    finished = pyqtSignal(object)

    def run(self, task: Callable[..., RunOutcome], goals: AnalysisGoals, cancelled: Callable[[], bool]) -> None:
        try:
            outcome = task(goals, cancelled=cancelled)
        except Exception as exc:  # reported in the panel, never lost in the thread
            outcome = RunOutcome(RUN_FAILED, f"The run stopped unexpectedly: {exc}", [], RunResults(), LlmUsage())
        self._emit(self.finished, outcome)

    # RunEvents, called on the run's thread
    def step_started(self, step) -> None:
        self._emit(self.stepStarted, dataclasses.replace(step))

    def step_finished(self, step) -> None:
        self._emit(self.stepFinished, dataclasses.replace(step))

    def model_text(self, text: str) -> None:
        self._emit(self.modelText, text)

    def model_progress(self, kind: str, text: str) -> None:
        self._emit(self.modelProgress, kind, text)

    def notice(self, text: str) -> None:
        self._emit(self.noticed, text)

    def usage(self, _turn_usage, total_usage) -> None:
        self._emit(self.usageChanged, total_usage)

    @staticmethod
    def _emit(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:  # the window was closed while the run finished
            pass


class AssistantController(QObject):
    def __init__(
        self,
        window,
        services: AssistantServices,
        *,
        settings,
        automation: Callable[[], object],
        show_analyze: Optional[Callable[[], None]] = None,
    ):
        super().__init__(window)
        self.window = window
        self.services = services
        self.settings = settings
        self._automation = automation
        self._show_analyze = show_analyze or (lambda: None)
        self.tasks = TaskRunner(self)
        self._bridge = GuiBridge(self)
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._worker: Optional[_RunWorker] = None
        self._dock: Optional[QDockWidget] = None
        self.panel: Optional[AssistantPanel] = None
        self._goals: Optional[AnalysisGoals] = None
        self._frame: Optional[str] = None
        self._model = ""
        self._started = 0.0
        self.outcome: Optional[RunOutcome] = None
        self._operation_tools: Optional[ToolCatalog] = None

    # -- entry points ------------------------------------------------------------------

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, notes: str = "") -> None:
        """Ask what to find out about the frame in Analyze (``notes`` pre-filled), then run."""
        if self.running():
            self.show_panel()
            return
        try:
            self._show_analyze()
            status = self._automation().status()
        except Exception as exc:  # a slot must never raise
            QMessageBox.warning(self.window, TITLE, f"Analyze is not ready: {exc}")
            return
        dialog = AssistantStartDialog(
            self.settings,
            status=status,
            configure=lambda: self.configure(dialog),
            parent=self.window,
        )
        if notes.strip():
            dialog.notes_edit.setPlainText(notes.strip())
        dialog.backendChanged.connect(lambda backend: self.check_brain(dialog, backend))
        self.check_brain(dialog, dialog.backend())
        if dialog.exec_() == QDialog.Accepted:
            self.run(dialog.goals(), status)

    def check_brain(self, dialog: AssistantStartDialog, backend: str) -> None:
        """Tell the start dialog whether ``backend`` can run (Claude Code is asked in the background)."""
        if backend == BACKEND_PROVIDER:
            ready, text, _label = describe_provider(self.settings, self.services.providers)
            dialog.set_status(ready, text)
            return
        if backend == BACKEND_API:
            source = self.services.credentials()
            text = (
                f"Claude API · {self.model()} · credentials: {source}." if source
                else "No Claude API key yet: use Set Up AI… to add one."
            )
            dialog.set_status(bool(source), text)
            return
        dialog.set_status(False, "Checking Claude Code…")
        cli, model = self.code_cli(), self.code_model()

        def checked(info: dict) -> None:
            if dialog.backend() == BACKEND_CLAUDE_CODE:
                dialog.set_status(*describe_code_status(info, model))

        self.tasks.submit(
            "assistant-brain-check",
            lambda: self.services.code_status(cli),
            on_done=checked,
            on_error=lambda message, _details: checked({"error": message}),
        )

    def configure(self, parent=None) -> None:
        """The Assistant settings in a small dialog (brain, sign-in, API key)."""
        dialog = QDialog(parent or self.window)
        dialog.setWindowTitle("Set Up AI")
        dialog.setMinimumWidth(640)
        layout = QVBoxLayout(dialog)
        page = self.settings_page(dialog)
        layout.addWidget(page)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, dialog)
        buttons.rejected.connect(dialog.accept)
        layout.addWidget(buttons)
        if self.backend() == BACKEND_CLAUDE_CODE:
            page.code_section.check()
        dialog.exec_()

    def settings_page(self, parent=None) -> AssistantSettingsPage:
        return AssistantSettingsPage(self.settings, self.services, self.tasks, parent)

    def backend(self) -> str:
        chosen = preferences.read(self.settings, "backend")
        return chosen if chosen in BACKENDS else BACKEND_CLAUDE_CODE

    def model(self) -> str:
        return str(preferences.read(self.settings, "model")).strip()

    def code_model(self) -> str:
        return str(preferences.read(self.settings, "code_model") or "").strip()

    def code_cli(self) -> str:
        return str(preferences.read(self.settings, "code_cli") or "").strip()

    def _brain(self):
        """The model client of the chosen brain, its label and the run to build around it."""
        effort = str(preferences.read(self.settings, "effort"))
        if self.backend() == BACKEND_PROVIDER:
            if self.services.providers is None:
                raise LlmError("Other AI providers need the openai package (python -m pip install openai).")
            key, model, url = chosen_provider(self.settings)
            _ready, _text, label = describe_provider(self.settings, self.services.providers)
            return self.services.providers.create(key, model, url), label, RunAssistantTask
        if self.backend() == BACKEND_CLAUDE_CODE:
            agent = self.services.create_agent(self.code_cli(), self.code_model(), effort)
            label = f"Claude Code · {self.code_model() or 'default model'}"
            return agent, label, RunAgentTask
        return self.services.create_llm(self.model(), effort), self.model(), RunAssistantTask

    def run(self, goals: AnalysisGoals, status: Optional[dict] = None) -> bool:
        if self.running():
            return False
        try:
            brain, model, task_type = self._brain()
        except LlmError as exc:
            QMessageBox.warning(self.window, TITLE, exc.message)
            return False
        except Exception as exc:  # a slot must never raise
            QMessageBox.warning(self.window, TITLE, f"Claude could not be started: {exc}")
            return False
        standing = str(preferences.read(self.settings, "standing_instructions") or "").strip()
        if standing:
            goals = dataclasses.replace(goals, standing_instructions=standing)
        automation = self._automation()
        status = status or automation.status()
        self._cancel = threading.Event()
        cancelled = self._cancel.is_set
        worker = _RunWorker()
        task = task_type(
            brain,
            GuiWorkbench(automation, self._bridge, cancelled=cancelled),
            confirmer=GuiConfirmer(self._bridge, self.window, cancelled=cancelled),
            store=self.services.store,
            events=worker,
            explorer=self.services.explorer,
            calibrator=self.services.calibrator,
            fitter=self.services.fitter,
            chooser=GuiChooser(self._bridge, self.window, cancelled=cancelled),
            max_turns=int(preferences.read(self.settings, "max_turns")),
        )
        panel = self.show_panel()
        worker.stepStarted.connect(panel.step_started)
        worker.stepFinished.connect(panel.step_finished)
        worker.modelText.connect(panel.model_text)
        worker.modelProgress.connect(panel.model_progress)
        worker.noticed.connect(panel.notice)
        worker.usageChanged.connect(self._usage)
        worker.finished.connect(lambda outcome, source=worker: self._finished(outcome, source))
        self._worker, self._goals, self._model = worker, goals, model
        self._frame = status.get("path")
        self.outcome = None
        self._started = time.monotonic()
        panel.begin(status.get("file") or "frame", goals, model, goals.permission)
        self._thread = threading.Thread(
            target=worker.run, args=(task, goals, cancelled), name="gimap-claude", daemon=True
        )
        self._thread.start()
        return True

    def follow_up(self, question: str) -> bool:
        """A new run on the same frame and settings: the person's question, with the last report as context."""
        if self.running() or self._goals is None or not question.strip():
            return False
        context = ""
        report = self.outcome.results.report if self.outcome is not None else None
        if report is not None:
            findings = "; ".join(f"{item.item}: {item.findings}" for item in report.items)
            context = f"\n\nWhat you found in the previous run on this frame: {report.summary} {findings}"[:2400]
        goals = dataclasses.replace(self._goals, instructions=f"Follow-up question: {question.strip()}{context}")
        return self.run(goals)

    def stop(self) -> None:
        if self.running():
            self._cancel.set()
            if self.panel is not None:
                self.panel.stopping()

    def shutdown(self) -> None:
        """Stop a run before the window closes (a stuck request is left to the daemon thread)."""
        if self.running():
            self._cancel.set()
            self._thread.join(SHUTDOWN_WAIT_S)
        self.tasks.shutdown()

    # -- panel -------------------------------------------------------------------------

    def show_panel(self) -> AssistantPanel:
        if self._dock is None:
            self.panel = AssistantPanel()
            dock = QDockWidget("AI Assistant", self.window)
            dock.setObjectName("assistantDock")
            dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
            dock.setWidget(self.panel)
            self.window.addDockWidget(Qt.RightDockWidgetArea, dock)
            self.window.resizeDocks([dock], [440], Qt.Horizontal)
            self.panel.stopRequested.connect(self.stop)
            self.panel.runAgainRequested.connect(self.start)
            self.panel.saveRequested.connect(self.save_report)
            self.panel.operationRequested.connect(self.operation_action)
            self.panel.undoAllRequested.connect(self.undo_all_operations)
            self.panel.followUpRequested.connect(self.follow_up)
            self._dock = dock
        self._dock.show()
        self._dock.raise_()
        return self.panel

    def _usage(self, total: LlmUsage) -> None:
        if self.panel is not None:
            self.panel.usage(total, self.services.cost(self._model, total), time.monotonic() - self._started)

    def _cost(self, outcome: RunOutcome) -> Optional[float]:
        if outcome.cost_usd is not None:  # Claude Code reports its own estimate
            return outcome.cost_usd
        return self.services.cost(self._model, outcome.usage)

    def _finished(self, outcome: RunOutcome, source: Optional[_RunWorker] = None) -> None:
        if source is not None and source is not self._worker:
            return  # a late signal of an earlier run
        elapsed = time.monotonic() - self._started
        self.outcome = outcome
        self._thread = None
        if self.panel is not None:
            self.panel.finish(outcome, cost=self._cost(outcome), elapsed=elapsed, language=self._language())
        try:
            self.services.save_run(self._record(outcome, transcript=True))
        except OSError:
            pass
        status_bar = getattr(self.window, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(f"AI: {outcome.message}", 8000)

    # -- the changes (operation cards) ----------------------------------------------------

    def _operations(self) -> Optional[ToolCatalog]:
        """Applies and undoes the run's changes on the GUI thread (no model involved)."""
        if self.outcome is None or self._goals is None:
            return None
        if self._operation_tools is None or self._operation_tools.results is not self.outcome.results:
            self._operation_tools = ToolCatalog(GuiWorkbench(self._automation(), self._bridge), self._goals, self.outcome.results)
        return self._operation_tools

    def operation_action(self, identifier: str, action: str) -> None:
        tools = self._operations()
        if tools is None:
            return
        try:
            if action == "dismiss":
                tools.dismiss_operation(identifier)
                message = "Dismissed."
            else:
                outcome = tools.apply_operation(identifier) if action == "apply" else tools.undo_operation(identifier)
                message = outcome.summary if not outcome.is_error else f"Could not {action}: {outcome.summary}"
        except KeyError:
            return
        self._after_operation(message)

    def undo_all_operations(self) -> None:
        tools = self._operations()
        if tools is None:
            return
        count = tools.undo_all_operations()
        self._after_operation(f"Undid {count} change(s).")

    def _after_operation(self, message: str) -> None:
        if self.panel is not None and self.outcome is not None:
            self.panel.show_operations(self.outcome.results.operations)
        status_bar = getattr(self.window, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(f"AI change: {message}", 6000)

    def _language(self) -> str:
        return self._goals.language if self._goals is not None else "English"

    # -- saving ------------------------------------------------------------------------

    def _record(self, outcome: RunOutcome, *, transcript: bool = False) -> dict:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame": self._frame,
            "model": outcome.model or self._model,
            "brain": self.backend(),
            "billing": outcome.billing,
            "cost_usd": self._cost(outcome),
            "state": outcome.state,
            "message": outcome.message,
            "goals": clean(self._goals),
            "usage": clean(outcome.usage),
            "steps": [clean(step) for step in outcome.steps],
            "results": results_payload(outcome.results),
        }
        if transcript:
            record["transcript"] = outcome.transcript
        return record

    def report_document(self, outcome: RunOutcome) -> str:
        name = Path(self._frame).name if self._frame else "frame"
        cost = self._cost(outcome)
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>Claude report – {html.escape(name)}</title></head>"
            f"<body style='font-family: sans-serif; max-width: 60em'><h2>{html.escape(name)}</h2>"
            f"{report_html(outcome, cost=cost, language=self._language())}</body></html>\n"
        )

    def save_report(self) -> Optional[str]:
        outcome = self.outcome
        if outcome is None:
            return None
        source = Path(self._frame) if self._frame else Path.home() / "frame"
        suggested = source.parent / "gimap_analysis" / f"{source.stem}_claude_report.html"
        path, _ = QFileDialog.getSaveFileName(self.window, "Save Claude Report", str(suggested), "HTML report (*.html)")
        if not path:
            return None
        try:
            written = self.services.save_text(path, self.report_document(outcome))
            self.services.save_text(
                str(Path(written).with_suffix(".json")),
                json.dumps(self._record(outcome), indent=2, ensure_ascii=False) + "\n",
            )
        except OSError as exc:
            QMessageBox.warning(self.window, "Save Claude Report", f"The report could not be saved: {exc}")
            return None
        return written


__all__ = ["AssistantController"]
