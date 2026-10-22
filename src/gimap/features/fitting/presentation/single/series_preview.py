"""The In-situ series before (and beside) its run: the selected frame and the trend's choices.

A listed frame is drawn as soon as it is selected — the first one when the curves are listed — on the
points the series would fit (its halves, the range and left-out points of Single analysis), and drawn again
when those change; once the series is started, every frame on the points (and in the unit of q) the run
fits (a fitted one with its model), whatever is chosen since. The points of the same curve that are not
fitted stay on the plot — paler outside the fitting range (the orange band), crosses for those left out —
and a line above the plot says what the frame is fitted on: the halves, the range, how many points. A
frame that cannot be fitted as the series stands (a file that cannot be read, no points on the chosen
half) says why instead of an empty plot. The curves the stage search reads are kept, so selecting a frame
seldom reads its file again. Before Start, the trend's list offers the values of the model in Single
analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import set_role

from ...application import LoadCurveRequest
from ...application.models import CURVE_NOT_FOUND
from ...application.series_fit import TOO_FEW_POINTS, UNREADABLE, parameter_columns, prepare_frame
from ...application.single_fit import SERIES_SIDES, Curve, evaluate, left_out_points, prepare_points
from ..views.fit_series_view import FRAME_EMPTY
from .series_stages import CHI2
from .series_text import ONE_HALF
from .session import path_name

DATA_COLOR, MODEL_COLOR, OUTSIDE_COLOR, LEFT_OUT_COLOR = "#2563eb", "#f97316", "#94a3b8", "#dc2626"
"""As in Single analysis: the points fitted, the model, the points outside the range, those left out."""


class FitSeriesPreviewMixin:
    """Needs the series page: ``paths``, ``fits``, ``_curves``, ``_settings``, ``_stage_settings()``,
    ``single``, ``view_model``, ``frame_list``, ``frame_plot``, ``trend_combo``, ``trend_plot``."""

    _preview_curves: dict
    _trend_labels: Optional[tuple] = None
    _unit: str = "angstrom"
    _drawn_with: Optional[tuple] = None
    """What the frame plot shows: the frame, the points chosen and the unit of q."""
    _read_errors: Optional[dict] = None
    """Why a listed file could not be read, by its path (English, as the reader said it)."""

    # -- the curves -------------------------------------------------------------------------

    def _started(self) -> bool:
        return self.running or self._start_model is not None

    def _preview_unit(self) -> str:
        """The unit of q of the files whose header states none: the run's once started, else Single analysis'
        setting (a file whose header states one is read in that one: ``application.curve_units``)."""
        if self._started():
            return self._unit
        return self.single.unit_setting()

    def _keep_preview_curves(self, curves) -> None:
        """Curves read elsewhere (the stage search), for drawing their frames without reading them again."""
        for curve in curves:
            if curve is not None and curve.path:
                self._preview_curves[(curve.path, curve.source_unit)] = curve  # read in this unit: so is it with it

    def _preview_curve(self, index: int) -> Optional[Curve]:
        path = self.paths[index]
        key = (path, self._preview_unit())
        curve = self._preview_curves.get(key)
        if curve is None:
            errors = self._read_errors = self._read_errors if self._read_errors is not None else {}
            try:
                outcome = self.load_series_curve(LoadCurveRequest(Path(path), key[1]))  # a detector frame is cut
            except (OSError, ValueError) as exc:  # the file cannot be read: the frame says so
                errors[path] = str(exc)
                return None
            if outcome.error is not None:
                errors[path] = outcome.error.message
                return None
            errors.pop(path, None)
            value = outcome.value
            self._note_curve(index, value.q, value.observation)  # two halves? sent with which? (series_halves.py)
            curve = Curve.from_arrays(value.q, value.intensity, value.error, name=Path(path).name, path=path,
                                      unit=value.q_source_unit, unit_from=value.q_unit_from)
            self._preview_curves[key] = curve
        return curve

    # -- the selected frame -----------------------------------------------------------------

    def _select(self, index: Optional[int], *, follow: bool = False) -> None:
        if index is None or not 0 <= index < len(self.paths):
            return
        listed = self._listed()
        if follow and index in listed:
            self.frame_list.blockSignals(True)
            self.frame_list.setCurrentRow(listed.index(index))
            self.frame_list.blockSignals(False)
        self._selected = index
        self._mark_table_row(index)
        self._show_set_up()  # “Set Up on Frame n …” names the frame that is shown
        frame = self.fits.get(index)
        curve = self._curves.get(index)
        if curve is None and (frame is None or frame.ok):  # (a frame the run could not read is not drawn from an
            # earlier look at its file: it says why it has no fit)
            curve = self._preview_curve(index)  # read first: what it holds may decide the halves of the series
        started = frame is not None or self._started()
        settings = self._settings if started else self._stage_settings()  # before Start: as it would be fitted
        self._drawn_with = (index, settings, self._preview_unit())
        path = self.paths[index]
        self.frame_plot.set_title(f"{index + 1} · {Path(path).name}")
        if curve is None:
            self._frame_not_drawn(index, (frame.error if frame is not None and frame.error else None)
                                  or (self._read_errors or {}).get(path) or UNREADABLE)
            return
        try:
            prepared = prepare_frame(curve, settings)
        except ValueError as exc:  # no points on the chosen half: the other half is not shown in its place
            self._frame_not_drawn(index, str(exc))
            return
        data = prepared.data
        if not data.q.size:
            self._frame_not_drawn(index, TOO_FEW_POINTS)
            return
        curves, colors, markers = [(tr("measured"), data.q, data.intensity)], [DATA_COLOR], [True]
        if frame is not None and frame.ok and frame.native:  # fit1d: its own v2 curve, not Fitting's model of it
            native = frame.native["curves"]
            curves.append((tr("fit1d v2 model"), np.asarray(native["q_nm_inv"], float),
                           np.asarray(native["model"], float)))
            colors.append(MODEL_COLOR)
            markers.append(False)
        elif frame is not None and frame.ok:
            q = np.geomspace(max(data.q.min(), 1e-6), data.q.max(), 400)
            curves.append((tr("model"), q, evaluate(frame.result.model, q)))
            colors.append(MODEL_COLOR)
            markers.append(False)
        # The other points of the same curve (the same halves): not fitted, but there to be seen.
        every = prepare_points(curve, settings.side, None, (), strict=True).data
        left_out = left_out_points(every.q, settings.excluded) if settings.excluded else np.zeros(every.q.size, bool)
        if settings.q_range is not None:
            left_out &= (every.q >= min(settings.q_range)) & (every.q <= max(settings.q_range))
        outside = ~np.isin(every.q, data.q) & ~left_out
        for name, chosen, color, marker in (("outside the range", outside, OUTSIDE_COLOR, True),
                                            ("left out", left_out, LEFT_OUT_COLOR, "x")):
            if chosen.any():
                curves.append((tr(name), every.q[chosen], every.intensity[chosen]))
                colors.append(color)
                markers.append(marker)
        self.frame_plot.set_empty_text(FRAME_EMPTY)
        self.frame_plot.set_curves(curves, colors, markers=markers)
        if settings.q_range is None:
            self.frame_plot.hide_window()
        else:
            self.frame_plot.show_window(*settings.q_range)  # the fitting range, as Single analysis shows it
        self._show_frame_note(self._fitted_on(curve, settings, prepared))

    def _fitted_on(self, curve: Curve, settings, prepared) -> str:
        """“Mean of both halves · 771 points · |q| 0.1–1.2 nm⁻¹ · 3 left out”: what a frame is fitted on."""
        parts = []
        single = prepared.points - prepared.averaged if curve.signed and settings.side == "mean" else 0
        if curve.signed:
            parts.append(tr(dict(SERIES_SIDES)[settings.side]))
        elif prepared.halves_used in ONE_HALF and self.curves_signed():  # a frame of a series of cuts with one half only:
            # under the mean it is that half alone; under a chosen half it holds the one that was chosen
            parts.append(tr(ONE_HALF[prepared.halves_used] if settings.side == "mean" else dict(SERIES_SIDES)[settings.side]))
        parts.append(trf("{points} points, {single} of them from one half alone", points=prepared.points, single=single)
                     if single else trf("{points} points", points=prepared.points))
        if settings.q_range is not None:
            parts.append(f"|q| {settings.q_range[0]:.4g}–{settings.q_range[1]:.4g} nm⁻¹")
        if prepared.left_out:
            parts.append(trf("{count} left out", count=prepared.left_out))
        return " · ".join(parts)

    def _show_frame_note(self, text: str, role: str = "muted") -> None:
        self.frame_note.setText(text)
        self.frame_note.setVisible(bool(text))
        set_role(self.frame_note, role)

    def _reason_text(self, reason: str) -> str:
        """Why a frame has no fit, in the interface language: the reason as the application gave it (English,
        kept in the record); of a file that is no longer there, the sentence with its path as it is."""
        head = CURVE_NOT_FOUND.partition("{")[0]
        return trf(CURVE_NOT_FOUND, path=reason[len(head):]) if reason.startswith(head) else tr(reason)

    def _frame_not_drawn(self, index: int, reason: str) -> None:
        """A frame that cannot be fitted as the series stands: why, over the empty plot and above it."""
        text = trf("Frame {frame} cannot be fitted: {reason}", frame=index + 1, reason=self._reason_text(reason))
        self.frame_plot.hide_window()
        self.frame_plot.set_curves([])
        self.frame_plot.set_empty_text(text)
        self._show_frame_note(text, "warning")

    def _clear_frame(self) -> None:
        """No frame is shown (no curves listed): the plot as it starts."""
        self._drawn_with = None
        self.frame_plot.set_title("")
        self.frame_plot.hide_window()
        self.frame_plot.set_curves([])
        self.frame_plot.set_empty_text(FRAME_EMPTY)
        self._show_frame_note("")

    def _select_first(self) -> None:
        """The first listed frame, drawn (after the curves are listed)."""
        if self._listed():
            self.frame_list.setCurrentRow(0)

    def _keep_selection(self) -> None:
        """After the list is filled again: the selected frame stays selected (signals blocked by the caller)."""
        listed = self._listed()
        if self._selected in listed:
            self.frame_list.setCurrentRow(listed.index(self._selected))

    def _redraw_preview(self) -> None:
        """Before Start: the selected frame again when Single analysis now fits other points (its halves,
        range or left-out points) or reads q in another unit than when it was drawn."""
        if self._started() or self._selected is None:
            return
        if (self._selected, self._stage_settings(), self._preview_unit()) != self._drawn_with:
            self._select(self._selected)

    # -- the trend's choices ----------------------------------------------------------------

    def _fill_trend_choices(self, model=None) -> None:
        """χ²ᵣ and every value of ``model`` (the started model by default); R first when there is one."""
        model = self._start_model if model is None else model
        self.trend_combo.blockSignals(True)
        self.trend_combo.clear()
        self.trend_combo.addItem("χ²ᵣ", CHI2)
        labels = []
        for path, _name in parameter_columns(model):
            labels.append(path_name(model, path, unit=True))
            self.trend_combo.addItem(labels[-1], path)
            self.trend_combo.setItemData(self.trend_combo.count() - 1, path_name(model, path, unit=True, full=True),
                                         Qt.ToolTipRole)  # “1·Sphere R (nm)” when the list says “R (nm)”
        index = next((i for i in range(self.trend_combo.count()) if isinstance(self.trend_combo.itemData(i), tuple)
                      and self.trend_combo.itemData(i)[1] == "R"), 0)
        self.trend_combo.setCurrentIndex(index)
        self.trend_combo.blockSignals(False)
        self._trend_labels = tuple(labels)

    def _preview_trend(self) -> None:
        """Before Start: the values of the model in Single analysis to choose from, and an empty trend that says so."""
        if self.running or self._start_model is not None:
            return
        model = self.single.session.model
        labels = tuple(path_name(model, path, unit=True) for path, _name in parameter_columns(model))
        if labels != self._trend_labels or not self.trend_combo.count():
            self._fill_trend_choices(model)
        if not self.fits:
            self._draw_trend()  # empty: the plot says when it fills (``TREND_EMPTY``)


__all__ = ["FitSeriesPreviewMixin"]
