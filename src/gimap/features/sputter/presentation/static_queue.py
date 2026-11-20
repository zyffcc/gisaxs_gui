"""Analyze with After Deposition chosen: the ticked samples of the list one after another (a mixin of
``SputterPage``).

**Analyze** takes the ticked samples that have a nominal thickness, in the order the list shows them (a right-click
on a row analyses one sample or one series). Values every sample shares — the folder of the series, the mode, the
geometry — are checked first; a sample whose own values are wrong (its folder is gone, say) is marked failed and left
out. When
samples analysed before would be replaced (the same name from other frames or values), one question names them all.

Each sample is one run of the package (``static_flow.py``): the job status says which sample of how many is at work
and its step, the bar moves over the whole list, the sample's row says it is at work and the main screen shows the
live view of that sample. A sample that fails is marked failed with its reason (in the row's tooltip and the log) and
the next one starts. Cancel stops the sample at work and the ones after it; the samples analysed stay in their series.
When the samples are done, each series analysed in the run is fitted together as one growing film (the package's
``iampred.static --series``: the samples by nominal thickness, radius, spacing, base layer, film height, amount and
coverage may not decrease); a package without it, or a series of one sample, skips this. At the end the job status sums
up, the main screen shows the series of the last sample analysed at that sample, and a toast offers the folder of the
series. A sample analysed before with the same request is shown as it is, at once.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QMessageBox

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from ..application import STATIC_PROBLEMS, ClientProgress, ClientResult, series_of
from .client_flow import Failure, _guarded
from .client_texts import STARTING
from .static_flow import STATIC_KEY, STATIC_REPLACE, STATIC_REPLACE_TITLE
from .static_list import STATE_RUNNING
from .texts import message_text, text

SAMPLE_OF = "Sample {index} of {count}: {name}"
BATCH_DONE = "Done: {done} samples analysed."
BATCH_DONE_FAILED = "Done: {done} samples analysed, {failed} failed (the reason of each is in its row's tooltip)."
BATCH_FAILED = "No sample was analysed: {failed} failed (the reason of each is in its row's tooltip)."
BATCH_STOPPED = "Stopped after {done} samples. The samples analysed stay in their series."
NO_SAMPLES = "Add samples first: Add Samples… takes a folder of samples."
NOTHING_TICKED = "Tick the samples to analyse in the list (each needs its nominal thickness)."
DONE_TOAST = "{done} samples analysed into {series} series."
SERIES_STEP = "Fitting the samples of {series} together as one growing film"
QUEUE_TEMPLATES = (SAMPLE_OF, BATCH_DONE, BATCH_DONE_FAILED, BATCH_FAILED, BATCH_STOPPED, NO_SAMPLES, NOTHING_TICKED,
                   DONE_TOAST, SERIES_STEP)
"""The run-time texts of Analyze over the list that need Chinese."""
SHARED_PROBLEMS = STATIC_PROBLEMS[5:]
"""Problems of the values every sample shares (the series folder, the mode, the geometry)."""


class StaticQueueMixin:
    """Needs ``static_flow.py`` (one sample's run) and ``static_list.py`` (the list and the states of its rows)."""

    def _init_static_queue(self) -> None:
        self._static_queue: list = []
        self._static_count = 0
        self._static_index = 0
        self._static_current = None
        self._static_outcomes: dict = {}
        self._static_last = None
        """(result, label) of the last sample analysed."""
        self._static_series: Optional[list] = None
        """The series folders still to fit together after the samples (None until the samples are done)."""

    def _confirm_static_replace(self, names: list) -> bool:
        answer = QMessageBox.question(self, tr(STATIC_REPLACE_TITLE), text(STATIC_REPLACE, names=", ".join(names)),
                                      QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        return answer == QMessageBox.Yes

    def _static_refused(self, message: str) -> None:
        self._static_status("failed", [(message, {})], 0.0)
        self._notify(message_text(message), error=True)

    def analyze_static(self, samples: Optional[list] = None) -> None:
        """Analyse ``samples`` (None: the ticked samples of the list that have a thickness) one after another."""
        static = self._static_service()
        if static is None or self._static_busy or self._client_busy:
            return
        unsupported = static.defaults().problem
        if unsupported:
            self._static_refused(unsupported)
            return
        chosen = list(samples) if samples is not None else [sample for sample in self.static_samples if sample.ready()]
        order = [sample.source for _series, group in series_of(self.static_samples) for sample in group]
        chosen.sort(key=lambda sample: order.index(sample.source) if sample.source in order else len(order))
        if not chosen:
            self._static_refused(NOTHING_TICKED if self.static_samples else NO_SAMPLES)
            return
        workers = self.run_workers()  # Automatic: worked out now
        requests = [(sample, replace(self.static_request_for(sample), workers=workers)) for sample in chosen]
        shared = [problem for problem in requests[0][1].problems() if problem in SHARED_PROBLEMS]
        if shared:
            self._static_status("failed", [(problem, {}) for problem in shared], 0.0)
            self._notify(" ".join(message_text(problem) for problem in shared), error=True)
            return
        todo, refused = [], []
        for sample, request in requests:
            problems = request.problems()
            if problems:
                refused.append(sample)
                self.set_sample_state(sample, "failed", " ".join(message_text(problem) for problem in problems))
            else:
                todo.append((sample, request))
        if not todo:
            self._static_status("failed", [(BATCH_FAILED, {"failed": len(requests)})], 0.0)
            return
        replacing = [request.label() for _sample, request in todo if static.replaces(request)]
        if replacing and not self._confirm_static_replace(replacing):
            return
        self._remember_static()
        self._static_queue = todo
        self._static_count, self._static_index = len(todo), 0
        self._static_outcomes = {sample.source: "failed" for sample in refused}
        self._static_last = None
        self._static_series = None
        for sample, _request in todo:
            self.set_sample_state(sample, "waiting")
        self._static_stop = False
        self._static_busy = True
        self._sync_run_buttons()
        self._run_next_static()

    def _run_next_static(self) -> None:
        if self._static_stop or not self._static_queue:
            if not self._static_stop and self._static_series is None:   # all samples done: fit each series together
                self._static_series = list(dict.fromkeys(
                    str(self.series_folder(sample.series)) for sample in self.static_samples
                    if self._static_outcomes.get(sample.source) == "done"))
            if not self._static_stop and self._static_series:
                self._run_next_series()
                return
            self._static_batch_done()
            return
        static = self._static_service()
        sample, request = self._static_queue.pop(0)
        self._static_index += 1
        self._static_current = sample
        self._static_tracker = ClientProgress.static()
        self._static_label = request.label()
        self._static_request_shown = request
        self.static_log.appendPlainText("— " + request.frames)
        self.set_sample_state(sample, "running", tr(STATE_RUNNING))
        self._static_status("running", [self._static_which(), (STARTING, {})], self._static_overall(0.0))
        emit, should_stop = self.staticEvent.emit, self._static_should_stop
        self.tasks.submit(STATIC_KEY, lambda: _guarded(lambda: static.analyze(request, emit, should_stop)),
                          on_done=self._static_sample_finished,
                          on_error=lambda message, details: self._static_sample_finished(Failure(message, details)))

    def _run_next_series(self) -> None:
        """The samples of the next series analysed in this run fitted together as one growing film (the package's
        ``--series``; skipped by a package without it or for a series of one sample)."""
        static = self._static_service()
        folder = self._static_series.pop(0)
        self._static_tracker, self._static_request_shown = None, None
        name = Path(folder).name
        self.static_log.appendPlainText("— " + text(SERIES_STEP, series=name))
        self._static_status("running", [(SERIES_STEP, {"series": name})], 1.0)
        emit, should_stop = self.staticEvent.emit, self._static_should_stop
        lines = lambda event: emit(event) if isinstance(event, str) else None  # noqa: E731 - no live view here
        workers = self.run_workers()
        self.tasks.submit(STATIC_KEY, lambda: _guarded(lambda: static.fit_series(folder, lines, should_stop, workers)),
                          on_done=lambda outcome: self._series_finished(folder, outcome),
                          on_error=lambda message, details: self._series_finished(folder, Failure(message, details)))

    def _series_finished(self, folder: str, outcome) -> None:
        if isinstance(outcome, ClientResult) and outcome.static:
            self._series_results[folder] = outcome
            if self._static_last is not None and str(self._static_last[0].folder) == folder:
                self._static_last = (outcome, self._static_last[1])
        elif isinstance(outcome, Failure):          # the samples keep their own solutions
            self.static_log.appendPlainText(message_text(outcome.message))
            if outcome.details:
                self.static_log.appendPlainText(outcome.details)
            if outcome.kind == "cancelled":
                self._static_stop = True
        self._run_next_static()

    def _static_which(self) -> tuple:
        name = self._static_current.name if self._static_current is not None else ""
        return SAMPLE_OF, {"index": self._static_index, "count": self._static_count, "name": name}

    def _static_overall(self, fraction: Optional[float]) -> float:
        return (self._static_index - 1 + min(max(fraction or 0.0, 0.0), 1.0)) / max(1, self._static_count)

    def _static_step(self, template: str, values: dict, fraction: Optional[float]) -> None:
        """The step of the sample at work, said in the job status (called by ``_static_event``)."""
        self._static_status("running", [self._static_which(), (template, values)], self._static_overall(fraction))

    def _static_sample_finished(self, outcome) -> None:
        self._live_end()
        sample, self._static_current = self._static_current, None
        self._static_tracker = None
        if sample is None:
            return
        if isinstance(outcome, ClientResult):
            self._static_outcomes[sample.source] = "done"
            self._series_results[str(self.series_folder(sample.series))] = outcome if outcome.static else None
            self._static_last = (outcome, self.sample_label(sample))
            self.set_sample_state(sample, "")
            self._show_sample_texts()  # the series row and the bar's line count it now
        elif isinstance(outcome, Failure):
            if outcome.details:
                self.static_log.appendPlainText(outcome.details)
            stopped = outcome.kind == "cancelled" or self._static_stop
            self._static_outcomes[sample.source] = "stopped" if stopped else "failed"
            self.set_sample_state(sample, "stopped" if stopped else "failed", message_text(outcome.message))
            if not stopped and self._static_count == 1:  # one sample: its reason in the job status, as before
                self._static_queue = []
                self._static_finish("failed", [("{error}", {"error": outcome.message})])
                message = message_text(outcome.message)
                self._notify(message.splitlines()[0] if message else outcome.message, error=True)
                return
        self._run_next_static()

    def _static_batch_done(self) -> None:
        outcomes = list(self._static_outcomes.values())
        done, failed = outcomes.count("done"), outcomes.count("failed")
        if self._static_stop:
            self._static_finish("cancelled", [(BATCH_STOPPED, {"done": done})])
        elif failed and not done:
            self._static_finish("failed", [(BATCH_FAILED, {"failed": failed})])
        else:
            self._static_finish("succeeded", [(BATCH_DONE_FAILED if failed else BATCH_DONE,
                                               {"done": done, "failed": failed})])
        if self._static_last is not None:
            result, label = self._static_last
            self.show_client_result(result, sample=label)
            self._remember_client_result(result.folder)
            self.show_details(False)
            self._choose_row_of_window()
            if not self._static_stop and done:
                folder = self.static_results_folder()
                series = len({sample.series for sample in self.static_samples
                              if self._static_outcomes.get(sample.source) == "done"})
                show_toast(self, text(DONE_TOAST, done=done, series=series), level="success",
                           action=(tr("Open Folder"),
                                   lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))))

    def _static_finish(self, state: str, parts: list) -> None:
        for sample, _request in self._static_queue:  # the samples that waited: as their series say again
            self.set_sample_state(sample, "")
        self._static_queue = []
        self._static_busy = False
        self._sync_run_buttons()
        self._static_status(state, parts, 1.0 if state == "succeeded" else 0.0)
        self._show_sample_texts()
        self._run_queued()  # an Analyze that waited for this analysis


__all__ = ["NOTHING_TICKED", "NO_SAMPLES", "QUEUE_TEMPLATES", "SAMPLE_OF", "SERIES_STEP", "StaticQueueMixin"]
