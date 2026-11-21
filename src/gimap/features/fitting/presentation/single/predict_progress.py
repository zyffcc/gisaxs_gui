"""How far Predict parameters is, and what Stop keeps.

The job reports its stages (``infrastructure/adapters/posterior_job.StageReport``): step 1, the network's proposals
for each composition; step 2, the least-squares refinement of their modes (and other starts, a search of the
ranges); step 3, the sampling of each solution's posterior. The line under the bars says the step, the item under
way, the time since the start and, once an item of the step is done, about how long the step still takes (from the
items done so far: a guide, the items differ). The bar is determinate from the proposals on.

After each solution's posterior is sampled the job sends the solutions finished so far, their probabilities over
those alone. Stop keeps them — marked incomplete — when there are any; otherwise nothing changes.
"""

from __future__ import annotations

import time

from PyQt5.QtCore import QTimer

from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_predict import predicted
from .predict_texts import count_solutions, points_tuple
from .session import Solution

STAGES = {
    "loading": "Loading the proposal network",
    "proposals": "Network proposals: composition {number} of {total}",
    "refinement": "Refining the modes: composition {number} of {total}",
    "starts": "Refining from other starts: composition {number} of {total}",
    "search": "Searching the ranges for a composition the proposals miss",
    "sampling": "Sampling the posterior: solution {number} of {total}",
}
STEPS = {"proposals": 1, "refinement": 2, "starts": 2, "search": 2, "sampling": 3}
STEP = "Step {step} of 3 · {stage}"
ELAPSED = "{time} elapsed"
LEFT = "about {time} left in this step"
STOPPING_KEEPS = "Stopping … the solutions finished so far are kept."
KEPT = ("Stopped: {solutions} kept, of the {finished} of {total} solutions sampled before Stop — marked incomplete; "
        "their probabilities are over those alone.")
NONE_KEPT = "Stopped before any solution was finished: no new solutions."


def clock(seconds: float) -> str:
    """``1:05`` (minutes:seconds), ``1:02:05`` from an hour."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def stage_line(details: dict, now: float | None = None) -> str:
    """The line of a stage report, in the interface language, its times counted on to ``now`` (``time.monotonic``)."""
    stage = str(details.get("stage") or "")
    done, total = int(details.get("done") or 0), int(details.get("total") or 0)
    since = max(0.0, (time.monotonic() if now is None else now) - float(details.get("received") or 0.0)) \
        if details.get("received") is not None else 0.0
    text = trf(STAGES.get(stage, stage), number=min(done + 1, max(total, 1)), total=total)
    if stage in STEPS:
        text = trf(STEP, step=STEPS[stage], stage=text)
    parts = [text, trf(ELAPSED, time=clock(float(details.get("elapsed") or 0.0) + since))]
    if details.get("left") is not None:
        parts.append(trf(LEFT, time=clock(max(0.0, float(details["left"]) - since))))
    return " · ".join(parts)


class PredictProgressMixin:
    """Needs the page's ``_progress``, ``_stop_event``, ``_running``, ``_progress_shown``, ``_show_progress``,
    ``session``, ``_finish``, ``_status``, ``_solutions_found``."""

    def setup_predict_progress(self) -> None:
        self._predict_partial = None
        """(encoded solutions, finished, total) of the last report that carried them."""
        self._predict_tried = None
        self._predict_clock = QTimer(self)
        self._predict_clock.setInterval(1000)  # the time since the start goes on between the job's reports
        self._predict_clock.timeout.connect(self._tick_predict)

    def _predict_started(self, tried) -> None:
        self._predict_partial, self._predict_tried = None, tried
        self._predict_clock.start()

    def _tick_predict(self) -> None:
        if self._running != "predict":
            self._predict_clock.stop()
            return
        if self._progress_shown is not None and isinstance(self._progress_shown[1], dict):
            self._show_progress(*self._progress_shown)

    def _predict_report(self, progress) -> None:
        """A report of the job (its thread): the stage line, and the solutions finished so far."""
        details = dict(progress.details or {})
        if details.get("partial"):
            self._predict_partial = (details.pop("partial"), int(details.get("done") or 0), int(details.get("total") or 0))
        if details.get("stage"):
            details["received"] = time.monotonic()
            self._progress.changed.emit(float(progress.fraction), details)
        else:
            self._progress.changed.emit(float(progress.fraction), str(progress.message))

    def stopping_line(self) -> str:
        return tr(STOPPING_KEEPS) if self._predict_partial else tr("Stopping…")

    def _predict_stopped(self, curve, selection) -> None:
        """Stop during Predict: the solutions finished so far, if any, marked incomplete (else nothing changes)."""
        self._predict_clock.stop()
        kept = self._predict_partial
        self._predict_partial = None
        solutions = [] if not kept else partial_solutions(kept[0], self._predict_tried)
        if not solutions:
            self._finish()
            self._status(lambda: tr(NONE_KEPT), "warning")
            return
        finished, total = kept[1], kept[2]
        self.session.predict_run = {"stopped": True, "finished": finished, "total": total}
        self._solutions_found(solutions, curve, selection, message=lambda: trf(
            KEPT, solutions=count_solutions(len(solutions)), finished=finished, total=total))


def as_solutions(items, *, incomplete: bool = False) -> list[Solution]:
    """``Predicted`` solutions of the job as the session keeps them."""
    return [Solution(p.model, p.chi2, f"Predict parameters · {p.name}", 0.0, probability=p.probability,
                     intervals=tuple(p.intervals.items()), flags=p.flags, points=points_tuple(p.points), tried=p.tried,
                     composition=p.name, ess=p.ess, incomplete=incomplete) for p in items]


def partial_solutions(encoded, tried) -> list[Solution]:
    try:
        return as_solutions(predicted({"solutions": encoded, "tried": tried}), incomplete=True)
    except (KeyError, TypeError, ValueError):
        return []


__all__ = ["ELAPSED", "KEPT", "LEFT", "NONE_KEPT", "STAGES", "STEP", "STOPPING_KEEPS", "PredictProgressMixin",
           "as_solutions", "clock", "partial_solutions", "stage_line"]
