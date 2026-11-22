"""The chained fit of the In-situ series page (``application/series_chain``) and two more ways to save its results.

* **The previous result; Predict parameters where the chain breaks** (Start step): every frame is refined from the
  result of the frame before it; where that chain breaks (a failed or unconverged fit, a new stage, a jump of χ²ᵣ;
  never a value at a bound) the frame is fitted again by Predict parameters (the page's ``predict_solutions``,
  unchanged), from the solution closest to the frame before among the probable ones, and the better fit is kept. A
  model Predict parameters cannot take (no particle, more than three) is refined only.
* **The Started from column** of the table says where each frame's kept fit started: “continued from frame n”, “restarted
  from Predict (why the chain broke)”, the backward pass, the model of Single analysis; the rule that chose Predict's
  solution is on hover (``start_texts.py``).
* **Then a backward pass**: once every frame is done, from the last but one to the first, each frame is refined from
  the kept result of the frame after it; the better fit of the two passes is kept, and its row says “backward pass”.
* **Save ▸ Fits as NXcanSAS…**: for every fitted frame the points fitted and the model (one file, an entry each).
* **Save ▸ Results Database…**: every frame's fit — state, χ²ᵣ, values and errors, flags, where it started — into
  the series' results database (``gimap_series.sqlite``; one found in the folder of the curves or up to two folders
  above it is offered first, so the fits join the curves Analyze recorded).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.series_store import STORE_NAME, SeriesStore

from ...application.fit_state import frame_to_dict
from ...application.series_chain import (
    BACKWARD,
    CHAIN,
    FORWARD,
    backward_order,
    better,
    chained_fit,
    frame_flags,
    observation_time,
)
from ...application.series_fit import file_name, parameter_columns
from ...application.single_predict import MAX_PARTICLES
from .results import folder_action, start_folder

BACKWARD_STARTED = "Backward pass: {count} frames, each from the result of the frame after it."
BACKWARD_KEPT = "Backward pass: {count} frames fitted better from the frame after them."
CHAIN_REFINE_ONLY = "Predict parameters needs 1–3 particles: where the chain breaks the frame is refined only."
NOTHING_TO_SAVE = "No fitted frame to save."
SAVED_NXCANSAS = "Saved the fits of {count} frames to {name}."
SAVED_STORE = "Saved {count} frames' fits to {name}."
STORE_LEVELS = 2
"""A results database is looked for in the folder of the curves and up to this many folders above it."""


def nearby_store(folder) -> Optional[Path]:
    """The results database in ``folder`` or up to ``STORE_LEVELS`` folders above it (Analyze's, next to its curves)."""
    if not folder:
        return None
    here = Path(folder)
    for place in [here, *list(here.parents)[:STORE_LEVELS]]:
        if (place / STORE_NAME).is_file():
            return place / STORE_NAME
    return None


class FitSeriesChainMixin:
    """Needs the series page: ``fits``, ``paths``, ``queue``, ``_previous``, ``_settings``, ``_start_model``,
    ``_curves``, ``_stop``, ``predictor``, ``_listed``, ``_log``, ``_status``, ``start_chain``, ``backward_check``."""

    _fit_pass = FORWARD

    # -- one frame -----------------------------------------------------------------------------------

    def _chain_predicts(self) -> bool:
        model = self._start_model
        return model is not None and 1 <= len(model.components) <= MAX_PARTICLES

    def fit_marks(self, settings, previous, start) -> dict:
        """What a frame fitted by the usual way says of itself: where its fit started (and from which frame's result)
        and in which pass."""
        follows = previous is not None and previous.ok and start is previous.result.model
        origin = "predict" if settings.method == "predict" else (
            ("next" if self._fit_pass == BACKWARD else "previous") if follows else "single")
        return {"start_from": origin, "fit_pass": self._fit_pass,
                "start_frame": previous.index + 1 if follows and origin != "predict" else 0}

    def chain_fitter(self, settings, previous, new_stage: bool):
        """``fit(index, path, curve, start)`` of a chained series for the worker of one frame (thread-safe: no widget is
        touched); ``None`` for the other starts and for Predict each frame (the usual ``fit_frame``)."""
        if settings.method == "predict" or settings.start != CHAIN:
            return None
        stop, single, fit_pass = self._stop.is_set, self._start_model, self._fit_pass
        solutions = self.predict_solutions() if self._chain_predicts() else None
        return lambda index, path, curve, start: chained_fit(
            index, path, curve, start, single, settings, previous, new_stage=new_stage, stop=stop, solutions=solutions,
            fit_pass=fit_pass)

    # -- the backward pass ---------------------------------------------------------------------------

    def _chain_reset(self) -> None:
        """At Start: the forward pass first."""
        self._fit_pass = FORWARD
        self._backward_better = 0
        if self._settings.start == CHAIN and not self._chain_predicts():
            self._log(tr(CHAIN_REFINE_ONLY))

    def _begin_backward(self) -> bool:
        """After the frames in order (no watch, not stopped): the backward pass, when it was asked for."""
        if (not self._settings.backward or self._fit_pass == BACKWARD or self._stop.is_set()
                or self._watch.isActive()):
            return False
        done = [index for index in self._listed() if index in self.fits]
        order = backward_order(done)
        if not order:
            return False
        self._fit_pass = BACKWARD
        self.queue = list(order)
        self._previous = self.fits[max(done)]
        self._log(trf(BACKWARD_STARTED, count=len(order)))
        return True

    def _chain_kept(self, frame):
        """The fit a frame keeps: in the backward pass the better of its two fits (the forward one on equal terms)."""
        if self._fit_pass != BACKWARD:
            return frame
        before = self.fits.get(frame.index)
        if before is None:
            return frame
        kept = better(before, frame)
        if kept is frame:
            self._backward_better += 1
        return kept

    def _chain_finished(self) -> None:
        if self._fit_pass == BACKWARD:
            self._log(trf(BACKWARD_KEPT, count=self._backward_better))

    # -- the trend against time ---------------------------------------------------------------------

    def forget_times(self) -> None:
        """Other curves are listed: no time is known of them yet (the axis box hides until one says it)."""
        self.__dict__["_frame_times"] = {}
        self.trend_axis_combo.hide()

    def note_time(self, index: int, observation) -> None:
        """A listed curve was read: when its frame was measured (``observation_time``); the trend's axis box appears
        once a curve says it."""
        times = self.__dict__.setdefault("_frame_times", {})
        moment = observation_time(observation)
        if moment is None:
            times.pop(index, None)
            return
        times[index] = moment
        if self.trend_axis_combo.isHidden():
            self.trend_axis_combo.show()

    def trend_x(self, indices) -> tuple:
        """``(x of each frame, axis label)`` of the trend: the frame numbers, or — Time chosen and every frame drawn
        with a time — the seconds since the earliest of them."""
        times = getattr(self, "_frame_times", {})
        indices = list(indices)
        if self.trend_axis_combo.currentData() == "time" and indices and all(index in times for index in indices):
            first = min(times[index] for index in indices)
            return [times[index] - first for index in indices], "time (s)"
        return [index + 1 for index in indices], "frame"

    def flags_cell(self, frame) -> str:
        """The Flags cell of a frame's row, in the interface language (details are data: names, numbers). Where the
        chain broke and the backward pass are said in the Started from cell (``start_texts.start_text``), not here again."""
        return "; ".join(tr(kind) + (f": {detail}" if detail else "") for kind, detail in frame_flags(frame)
                         if kind not in ("chain broke", "backward"))

    # -- saving ----------------------------------------------------------------------------------------

    def _frame_record(self, frame) -> dict:
        return {"frame": frame.index + 1, "file": file_name(frame.path, self.folder), **frame_to_dict(frame)}

    def save_fits_nxcansas(self, path=None):
        """Every fitted frame's points and model as one NXcanSAS file (Save ▸ Fits as NXcanSAS…)."""
        from ...application.series_chain import series_nxcansas
        from src.gimap.shared.nxcansas import write_nxcansas

        frames = [frame for frame in self.fits.values() if frame.fitted and self._curves.get(frame.index) is not None]
        if not frames:
            self._status(lambda: tr(NOTHING_TO_SAVE), "warning")
            return None
        path = path or QFileDialog.getSaveFileName(self, tr("Save Fits as NXcanSAS"),
                                                   str(start_folder(self.folder) / "series_fits.h5"),
                                                   "NXcanSAS (*.h5)")[0]
        if not path:
            return None
        curves = series_nxcansas(frames, self._curves, self._settings,
                                 lambda frame: f"frame {frame.index + 1} {file_name(frame.path, self.folder)}",
                                 self._frame_record)
        try:
            write_nxcansas(path, curves)
        except (OSError, ValueError) as exc:
            self._status(lambda error=str(exc): tr("Could not save: {error}").format(error=error), "error")
            return None
        self._status(lambda count=len(frames), name=Path(path).name: trf(SAVED_NXCANSAS, count=count, name=name), "ok",
                     action=folder_action(path))
        return path

    def save_results_database(self, path=None):
        """Every frame's fit into the series' results database (Save ▸ Results Database…)."""
        if not self.fits or self._start_model is None:
            self._status(lambda: tr(NOTHING_TO_SAVE), "warning")
            return None
        if path is None:
            offered = nearby_store(self.folder) or start_folder(self.folder) / STORE_NAME
            path = QFileDialog.getSaveFileName(self, tr("Save Results Database"), str(offered),
                                               tr("Results database") + " (*.sqlite)")[0]
            if not path:
                return None
        columns = parameter_columns(self._start_model)
        rows = []
        for frame in sorted(self.fits.values(), key=lambda item: item.index):
            result = frame.result if frame.ok else None
            rows.append(dict(
                file=str(frame.path), frame=frame.index + 1, state=frame.state,
                chi2=None if result is None else result.chi2_reduced,
                values={} if result is None else {name: result.model.get(key).value for key, name in columns},
                errors={} if result is None else {name: result.errors.get(key) for key, name in columns},
                flags=[f"{kind}: {detail}" if detail else kind for kind, detail in frame_flags(frame)],
                start_from=frame.start_from, record=self._frame_record(frame)))
        name = self._folder_name() or Path(self.folder or ".").name or "series"
        try:
            with SeriesStore(path) as store:
                count = store.put_fits(name, "model", rows)
                store.set_info(name, {"fit_folder": self.folder, "fit_settings": {
                    "start": self._settings.start, "method": self._settings.method, "side": self._settings.side,
                    "backward": self._settings.backward}})
        except (OSError, ValueError) as exc:
            self._status(lambda error=str(exc): tr("Could not save: {error}").format(error=error), "error")
            return None
        self._status(lambda: trf(SAVED_STORE, count=count, name=Path(path).name), "ok", action=folder_action(path))
        return path


__all__ = ["BACKWARD_KEPT", "BACKWARD_STARTED", "CHAIN_REFINE_ONLY", "FitSeriesChainMixin", "NOTHING_TO_SAVE",
           "SAVED_NXCANSAS", "SAVED_STORE", "nearby_store"]
