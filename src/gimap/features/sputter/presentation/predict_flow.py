"""Predict, Stop, a new deposition rate and refined windows (a mixin of ``SputterPage``).

Predict is one background task (``SputterService.predict``): check PyTorch in GIMaP's Python, install it when
it is missing (pip's progress on the bar; Stop works until pip writes the packages), run the model (the bar
follows the frames read), show the result. Its events reach the page through a signal (GUI thread). Stop
ends the model's whole process tree within 0.2 s. A failure says why, with what to do, and keeps pip's or the
model's last lines under Details.

Predict uses the values of the folder in the picker: pressed while that folder's header is still read, it
starts once the form is filled from it; a folder whose header says it has no frames is refused. A new
deposition rate or t0 rewrites only the physical files of the result shown (``rebuild_physical``: no frame is
read again) — when that result is of the scan in the picker; otherwise the fields are what the next Predict
uses. A rate given where there was none (or taken away) shows the default explanation afterwards: the
literature method needs a rate. Predict pressed while those files are rewritten follows them.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from ..application import (DOWNLOAD_SIZE, EXCLUDED_FRAMES, INSTALL_STEP, STEP_CHECK, STOP_NOW, InstallError,
                           InstallProgress, PredictStep, ProgressTracker, Stopped, StructureResult, ToolkitFailed)
from .results_panel import thickness_pair
from .texts import install_text, message_text, text

PREDICT_KEY = "sputter_predict"
STOPPING = "Stopping…"
STOPPING_LATER = "Stopping once pip has written the packages…"
WAITING_FOR_HEADER = "Predict starts once the header of the first frame is read…"
OTHER_SCAN = ("The result shown is of another scan: the deposition rate and t0 are kept for the next Predict of the "
              "chosen folder.")


@dataclass(frozen=True)
class Failure:
    message: str
    details: str = ""
    kind: str = "error"


def guarded(work):
    """Run ``work``; a failure comes back as a ``Failure`` (its details are lost on the way otherwise)."""
    try:
        return work()
    except InstallError as exc:
        return Failure(str(exc), exc.details, "cancelled" if exc.kind == "cancelled" else "install")
    except Stopped as exc:
        return Failure(str(exc), "", "cancelled")
    except ToolkitFailed as exc:
        return Failure(str(exc), exc.details, "toolkit")
    except (OSError, ValueError, RuntimeError) as exc:
        return Failure(str(exc) or type(exc).__name__, "", "error")
    except Exception as exc:  # noqa: BLE001 - shown to the user, with the traceback under Details
        return Failure(str(exc) or type(exc).__name__, traceback.format_exc(), "error")


def _float(value: str) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _shown(key: str, value):
    """A value of the status line in the interface language: the stage and the download size are table
    entries, an error is made from a template, an installation step from its stage (all English in the state)."""
    if key == "error":
        return message_text(value)
    if key in ("stage", "size"):
        return tr(str(value)) if value else ""
    if isinstance(value, InstallProgress):
        return install_text(value)
    return value


class SputterPredictMixin:
    def _connect_predict(self) -> None:
        self.predict_button.clicked.connect(self.predict)
        self.job_status.cancelRequested.connect(self.stop)
        self.job_status.detailsRequested.connect(lambda: self.log.setVisible(not self.log.isVisible()))
        self.eventReceived.connect(self._on_event)
        for field in (self.rate, self.t0):
            field.editingFinished.connect(self._thickness_changed)
            field.textEdited.connect(self._thickness_typed_now)
        self.merge_button.clicked.connect(self._choose_refined)

    # -- state -----------------------------------------------------------------------------------------------
    def running(self) -> bool:
        return self._busy is not None

    def _should_stop(self):
        return self._stop

    def _status(self, state: str, parts: list, progress: Optional[float] = 0.0, *, cancel: bool = True) -> None:
        """``parts``: (template, values) pairs joined by " — " (composed again after a language switch)."""
        self._status_parts = (state, parts, progress, cancel)
        self._show_status()

    def _show_status(self) -> None:
        state, parts, progress, cancel = self._status_parts
        words = [text(template, **{key: _shown(key, value) for key, value in values.items()})
                 for template, values in parts if template]
        words = " — ".join(word for word in words if word)
        self.job_status.set_state(state, words, progress=progress)
        self.job_status.cancel_button.setEnabled(state == "running" and cancel and self._stop is False)

    def _set_busy(self, busy: Optional[str]) -> None:
        self._busy = busy
        self.predict_button.setEnabled(busy in (None, "physical"))  # pressed during a quick rewrite: it follows
        for widget in (self.open_button, self.preset_button):
            widget.setEnabled(busy is None)
        self.merge_button.setEnabled(busy is None and self.result is not None)
        self.choice_box.setEnabled(busy is None and self.choice_box.count() > 1)

    # -- Predict ---------------------------------------------------------------------------------------------
    def predict(self) -> None:
        if self._busy is not None:
            if self._busy == "physical":
                self._predict_queued = True
            return
        picker = self.scan_picker.path()
        if picker and not self.header_ready():
            if not self._inspecting:  # a folder typed in but not looked at yet
                self.set_scan_folder(picker)
            self._predict_queued = True
            self._status("running", [(WAITING_FOR_HEADER, {})], None, cancel=False)
            return
        if picker and self.scan_info.problem:
            self._status("failed", [("{error}", {"error": self.scan_info.problem})], 0.0)
            self._notify(message_text(self.scan_info.problem), error=True)
            return
        request = self._request()
        problems = request.problems()
        if problems:
            self._status("failed", [(problem, {}) for problem in problems], 0.0)
            self._notify(" ".join(tr(problem) for problem in problems), error=True)
            return
        missing = self.service.toolkit_problem()
        if missing:
            self._status("failed", [("{error}", {"error": missing})], 0.0)
            self._notify(message_text(missing), error=True)
            return
        self._remember(request)
        self._stop = False
        self._tracker = ProgressTracker()
        self._step = ""
        self._thickness_typed = False
        self.log.appendPlainText("— " + str(request.scan_folder))
        self._set_busy("predict")
        self._status("running", [(STEP_CHECK, {})], None)
        service, emit, should_stop = self.service, self.eventReceived.emit, self._should_stop
        self.tasks.submit(PREDICT_KEY, lambda: guarded(lambda: service.predict(request, emit, should_stop)),
                          on_done=self._finished, on_error=lambda message, details: self._finished(
                              Failure(message, details)))

    def _run_queued(self) -> None:
        """Predict pressed while the header was read or the physical files were rewritten: now."""
        if self._predict_queued and self._busy is None:
            self._predict_queued = False
            self.predict()

    def _forget_queued(self) -> None:
        """Another folder was chosen: a Predict pressed for the earlier one does not run for it."""
        if self._predict_queued:
            self._predict_queued = False
            if self._busy is None:
                self._status("idle", [("Ready", {})], 0.0)

    def stop(self) -> None:
        if self._busy is None or self._stop:
            return
        self._stop = True
        state, parts, progress, _cancel = self._status_parts
        later = self._installing_writing
        self._status("running", parts[:1] + [(STOPPING_LATER if later else STOPPING, {})], progress, cancel=False)

    def dispose(self) -> None:
        """The window closes: stop at once (also pip while it writes), forget the page's background tasks."""
        self._predict_queued = False
        if self._busy is not None:
            self._stop = STOP_NOW
        self.tasks.cancel(PREDICT_KEY)

    # -- events (GUI thread) -------------------------------------------------------------------------------
    def _on_event(self, event) -> None:
        if isinstance(event, PredictStep):
            if event.installing:
                self._status("running", [(INSTALL_STEP, {"size": DOWNLOAD_SIZE})], None)
            elif event.text:
                self._installing_writing = False  # the installation is over: Stop ends the model at once
                self._step = event.text
                self._status("running", [(event.text, {})], self._tracker.fraction if event.folder else None)
            return
        if isinstance(event, InstallProgress):
            self._installing_writing = not event.cancellable and event.stage != "done"
            words = install_text(event)
            if words:
                self.log.appendPlainText(words)
            self._status("running", [(INSTALL_STEP, {"size": DOWNLOAD_SIZE}), ("{progress}", {"progress": event})],
                         event.fraction, cancel=event.cancellable)
            return
        line = str(event)
        tracker = self._tracker
        if tracker is None or not tracker.feed(line):
            self.log.appendPlainText(line)
            return
        if line.startswith("SPUTTER_PROGRESS"):
            fields = line.split("\t")
            self.log.appendPlainText(f"[{fields[1]}] " + "\t".join(fields[3:]))
        elif line.startswith("INTAKE_EXCLUDED") or line.startswith("INTAKE_PLAN"):
            self.log.appendPlainText(line)
        template, values = tracker.message()
        parts = [(self._step, {}), (template, values)]
        if tracker.excluded:
            parts.append((EXCLUDED_FRAMES, {"count": len(tracker.excluded)}))
        if self._stop:
            parts.append((STOPPING, {}))
        self._status("running", parts, tracker.fraction, cancel=not self._stop)

    def _finished(self, outcome) -> None:
        busy = self._busy
        self._set_busy(None)
        self._installing_writing = False
        self._show_environment()
        try:
            if isinstance(outcome, Failure):
                self._failed(outcome)
            elif isinstance(outcome, StructureResult):
                self._succeeded(busy, outcome)
        finally:
            self._run_queued()

    def _succeeded(self, busy: Optional[str], outcome: StructureResult) -> None:
        excluded = len(self._tracker.excluded) if (busy == "predict" and self._tracker is not None) else 0
        parts = [("Done: {windows} time windows.", {"windows": len(outcome.rows)})]
        if excluded:
            parts.append((EXCLUDED_FRAMES, {"count": excluded}))
        self._status("succeeded", parts, 1.0)
        # a rate typed while the model ran is not lost (only what was typed, compared as numbers)
        typed = (self.rate.text(), self.t0.text()) if busy == "predict" and self._thickness_typed else None
        self._thickness_typed = False
        self.show_result(outcome)
        if busy == "physical" and self._rebuild_values is not None:
            self._drawn = self._rebuild_values
        self._remember_result(outcome.folder)
        folder = outcome.folder
        show_toast(self, text("The structure of {name} is ready.", name=folder.name), level="success",
                   action=(tr("Open Folder"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))))
        if typed is not None and self._result_is_for_picker() and \
                thickness_pair(_float(typed[0]), _float(typed[1])) != self._drawn:
            self.rate.setText(typed[0])
            self.t0.setText(typed[1])
            self._thickness_changed()

    def _failed(self, failure: Failure) -> None:
        message = message_text(failure.message)
        if failure.details:
            self.log.appendPlainText(failure.details)
        state = "cancelled" if failure.kind == "cancelled" else "failed"
        self._status(state, [("{error}", {"error": failure.message})], 0.0)
        self._notify(message.splitlines()[0] if message else tr("The prediction failed."),
                     error=state == "failed")

    # -- a new deposition rate, refined windows --------------------------------------------------------------
    def _thickness_typed_now(self, _text: str = "") -> None:
        if self._busy == "predict":
            self._thickness_typed = True

    def _thickness_changed(self) -> None:
        """A rate or t0 other than the one the result shown was drawn with: rewrite its physical files (fast)
        and show them — when it is the result of the scan in the picker."""
        result = self.result
        if result is None or self._busy is not None:
            return
        values = self._values()
        if values["rate"] is not None and values["rate"] <= 0:
            self._notify(tr("The deposition rate must be positive (or empty)."), error=True)
            return
        wanted = thickness_pair(values["rate"], values["t0"])
        if wanted == self._drawn:
            return
        if not self._result_is_for_picker():
            self._status("idle", [(OTHER_SCAN, {})], 0.0)
            return
        folder, variant = result.folder, self.choice_box.currentData()
        if variant in (None, "physical", "literature") and (self._drawn is None or self._drawn[0] is None
                                                            or values["rate"] is None):
            variant = None  # a rate given or taken away: the default explanation (the literature method needs one)
        self._remember_thickness(result.scan_folder, values["rate"], values["t0"])
        self._rebuild_values = wanted
        self._set_busy("physical")
        self._tracker = None
        self._status("running", [("Drawing the nominal thickness again (no frame is read)…", {})], None,
                     cancel=False)
        service, emit = self.service, self.eventReceived.emit
        self.tasks.submit(PREDICT_KEY, lambda: guarded(lambda: service.rebuild_physical(
            folder, values["rate"], values["t0"], emit, variant)), on_done=self._finished,
            on_error=lambda message, details: self._finished(Failure(message, details)))

    def _choose_refined(self) -> None:
        if self.result is None or self._busy is not None:
            return
        refined = QFileDialog.getExistingDirectory(self, tr("Folder of Refined Window Files"), str(self.result.folder))
        if not refined:
            return
        result, values = self.result, self._values()
        metal = result.metal or values["metal"]
        variant = self.choice_box.currentData()
        rate, t0 = (values["rate"], values["t0"]) if self._result_is_for_picker() else \
            (self._drawn if self._drawn is not None else (None, None))
        self._set_busy("merge")
        self._tracker = None
        self._rebuild_values = None
        self._status("running", [("Merging the refined windows…", {})], None, cancel=False)
        service, emit = self.service, self.eventReceived.emit
        self.tasks.submit(PREDICT_KEY, lambda: guarded(lambda: service.merge_refined(
            result.folder, Path(refined), metal, emit, rate, t0, variant)),
            on_done=self._finished, on_error=lambda message, details: self._finished(Failure(message, details)))


__all__ = ["Failure", "OTHER_SCAN", "PREDICT_KEY", "SputterPredictMixin", "WAITING_FOR_HEADER", "guarded"]
