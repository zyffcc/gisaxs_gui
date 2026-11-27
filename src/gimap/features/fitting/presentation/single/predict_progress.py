"""How far Predict parameters is, and what Stop keeps.

The job reports its stages (``infrastructure/adapters/posterior_job.StageReport``): step 1, the network's proposals
for each composition; step 2, the least-squares refinement of their modes (and other starts, a search of the
ranges); step 3, the sampling of each solution's posterior. The line under the bars says the step, the item under
way, the time since the start and, once an item of the step is done, about how long the step still takes (from the
items done so far: a guide, the items differ). The bar is determinate from the proposals on.

After each solution's posterior is sampled the job sends the solutions finished so far, their probabilities over
those alone. Stop keeps them — marked incomplete — when there are any; otherwise nothing changes.

While it runs, the plot shows what it does: after step 1 the network's first guess, then after each least-squares
fit of step 2 the fit just refined and the best so far, with their χ²ᵣ (``preview``). The curves are computed in
the thread that waits for the job, not in the window's, and go when Predict ends.
"""

from __future__ import annotations

import json
import time

import numpy as np
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal

from src.gimap.app.presentation.components.curve_plot import LineStyle
from src.gimap.app.presentation.i18n import tr, trf

from ...application.single_fit import evaluate, model_from_dict
from ...application.single_predict import predicted
from .predict_texts import count_solutions, points_tuple
from .session import Solution, model_label

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
BEST_SO_FAR = "Predict, best so far: {label} · χ²ᵣ {chi2}"
JUST_REFINED = "Predict, just refined: {label} · χ²ᵣ {chi2}"
FIRST_GUESS = "Predict, the network's first guess: {label}"
BEST_COLOR, LATEST_COLOR = "#16a34a", "#64748b"
PREVIEW_POINTS = 400
PREVIEW_CACHE = 64
"""Curves kept by model while Predict runs (the best so far comes again with every fit)."""


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


def preview_grid(q):
    """The |q| of the preview curves: ``PREVIEW_POINTS`` over the points Predict fits; None without any."""
    q = np.abs(np.asarray(q, float))
    q = q[np.isfinite(q) & (q > 0)]
    return np.geomspace(q.min(), q.max(), PREVIEW_POINTS) if q.size else None


def preview_curves(preview, grid, cache) -> dict:
    """``{"best" | "latest": (model, χ²ᵣ or None, values, key)}`` of a report's ``preview`` on ``grid``; a model
    that cannot be evaluated is left out, ``latest`` too when it is the best. ``cache``: the curves by model."""
    out = {}
    for name in ("best", "latest"):
        item = (preview or {}).get(name)
        if not item or not item.get("model"):
            continue
        key = json.dumps(item["model"], sort_keys=True)
        if key not in cache:
            try:
                model = model_from_dict(item["model"])
                with np.errstate(all="ignore"):
                    values = np.asarray(evaluate(model, grid), float)
            except (KeyError, TypeError, ValueError, FloatingPointError, ArithmeticError):
                cache[key] = None
            else:
                usable = values.shape == grid.shape and bool(np.isfinite(values).any())
                cache[key] = (model, values) if usable else None
            while len(cache) > PREVIEW_CACHE:
                cache.pop(next(iter(cache)))
        found = cache[key]
        if found is not None:
            out[name] = (found[0], item.get("chi2"), found[1], key)
    if "best" in out and "latest" in out and out["best"][3] == out["latest"][3]:
        del out["latest"]
    return out


def preview_series(preview, grid) -> list:
    """``(label, q, values, colour, LineStyle)`` of the preview curves, in the interface language."""
    rows = []
    for name in ("latest", "best"):
        if name not in preview:
            continue
        model, chi2, values, _key = preview[name]
        label = model_label(model)
        if chi2 is None:
            rows.append((trf(FIRST_GUESS, label=label), grid, values, LATEST_COLOR,
                         LineStyle(width=1.6, style=Qt.DotLine, z=3)))
        elif name == "best":
            rows.append((trf(BEST_SO_FAR, label=label, chi2=f"{chi2:.3g}"), grid, values, BEST_COLOR,
                         LineStyle(width=2.2, z=4)))
        else:
            rows.append((trf(JUST_REFINED, label=label, chi2=f"{chi2:.3g}"), grid, values, LATEST_COLOR,
                         LineStyle(width=1.4, style=Qt.DashLine, z=3)))
    return rows


class _PreviewSignal(QObject):
    shown = pyqtSignal(object)
    """(job id, curves of ``preview_curves``): sent from the thread that waits for the job."""


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
        self._predict_preview = None
        """The curves of the fits shown while Predict runs (``preview_curves``)."""
        self._predict_grid = None
        self._preview_cache = {}
        self._preview_signal = _PreviewSignal(self)
        self._preview_signal.shown.connect(self._show_predict_preview)

    def _predict_started(self, tried) -> None:
        self._predict_partial, self._predict_tried = None, tried
        self._partial_plan = None
        self._predict_preview, self._preview_cache = None, {}
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
        preview, grid = details.pop("preview", None), self._predict_grid
        if preview and grid is not None:
            curves = preview_curves(preview, grid, self._preview_cache)
            if curves:
                self._preview_signal.shown.emit((progress.job_id, curves))
        if details.get("partial"):
            self._partial_plan = details.pop("prediction_plan", None)
            self._predict_partial = (details.pop("partial"), int(details.get("done") or 0), int(details.get("total") or 0))
        if details.get("stage"):
            details["received"] = time.monotonic()
            self._progress.changed.emit(float(progress.fraction), details)
        else:
            self._progress.changed.emit(float(progress.fraction), str(progress.message))

    def _show_predict_preview(self, payload) -> None:
        """The fits of a report on the plot (the window's thread), while this Predict runs."""
        job_id, curves = payload
        if self._running != "predict" or self._stop_event.is_set() or job_id != self._predict_job_id:
            return
        self._predict_preview = curves
        self._redraw(preserve_view=True)

    def clear_predict_preview(self) -> None:
        if self._predict_preview is not None:
            self._predict_preview = None
            self._redraw(preserve_view=True)

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
        if self._partial_plan:
            from ...application.fit_reproduction import prediction_record
            self.session.predict_run["reproduction"] = prediction_record(
                {**self._partial_plan, "partial_after": finished}, kept[0])
        message = ("Stopped. {solutions} kept after sampling {finished} of {total} solutions. "
                   "BIC weights refer to the finished solutions.") if any("bic" in s.flags for s in solutions) else KEPT
        self._solutions_found(solutions, curve, selection, message=lambda: trf(
            message, solutions=count_solutions(len(solutions)), finished=finished, total=total))


def as_solutions(items, *, incomplete: bool = False) -> list[Solution]:
    """``Predicted`` solutions of the job as the session keeps them."""
    return [Solution(p.model, p.chi2, f"Predict parameters · {p.name}", 0.0, probability=p.probability,
                     intervals=tuple(p.intervals.items()), flags=p.flags, points=points_tuple(p.points), tried=p.tried,
                     composition=p.name, ess=p.ess, incomplete=incomplete, diagnostics=p.diagnostics,
                     posterior_summary=p.posterior_summary, importance_starts=p.importance_starts) for p in items]


def partial_solutions(encoded, tried) -> list[Solution]:
    try:
        return as_solutions(predicted({"solutions": encoded, "tried": tried}), incomplete=True)
    except (KeyError, TypeError, ValueError):
        return []


__all__ = ["BEST_SO_FAR", "ELAPSED", "FIRST_GUESS", "JUST_REFINED", "KEPT", "LEFT", "NONE_KEPT", "STAGES", "STEP",
           "STOPPING_KEEPS", "PredictProgressMixin", "as_solutions", "clock", "partial_solutions", "preview_curves",
           "preview_grid", "preview_series", "stage_line"]
