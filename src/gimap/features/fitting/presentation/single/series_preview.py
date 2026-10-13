"""The In-situ series before (and beside) its run: the selected frame and the trend's choices.

A listed frame is drawn as soon as it is selected — the first one when the curves are listed — on the
points Single analysis would fit (its halves, range and left-out points), and drawn again when those
change; once the series is started, every frame on the points (and in the unit of q) the run fits (a
fitted one with its model), whatever Single shows since. The curves the stage search reads are kept, so
selecting a frame seldom reads its file again. Before Start, the trend's list offers the values of the
model in Single analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt

from src.gimap.app.presentation.i18n import tr

from ...application import LoadCurveRequest
from ...application.series_fit import parameter_columns
from ...application.single_fit import Curve, evaluate, prepare_curve
from .series_stages import CHI2
from .session import path_name

DATA_COLOR, MODEL_COLOR = "#2563eb", "#f97316"


class FitSeriesPreviewMixin:
    """Needs the series page: ``paths``, ``fits``, ``_curves``, ``_settings``, ``_stage_settings()``,
    ``single``, ``view_model``, ``frame_list``, ``frame_plot``, ``trend_combo``, ``trend_plot``."""

    _preview_curves: dict
    _trend_labels: Optional[tuple] = None
    _unit: str = "angstrom"
    _drawn_with: Optional[tuple] = None
    """What the frame plot shows: the frame, the points chosen and the unit of q."""

    # -- the curves -------------------------------------------------------------------------

    def _started(self) -> bool:
        return self.running or self._start_model is not None

    def _preview_unit(self) -> str:
        """The unit of q the files are read in: the run's once started, else Single analysis's."""
        if self._started():
            return self._unit
        curve = self.single.session.curve
        return curve.source_unit if curve is not None else "angstrom"

    def _keep_preview_curves(self, curves) -> None:
        """Curves read elsewhere (the stage search), for drawing their frames without reading them again."""
        for curve in curves:
            if curve is not None and curve.path:
                self._preview_curves[(curve.path, curve.source_unit)] = curve  # as read (the unit of q in the file)

    def _preview_curve(self, index: int) -> Optional[Curve]:
        path = self.paths[index]
        key = (path, self._preview_unit())
        curve = self._preview_curves.get(key)
        if curve is None:
            try:
                outcome = self.view_model.load_curve(LoadCurveRequest(Path(path), key[1]))
            except (OSError, ValueError) as exc:  # the file cannot be read: the frame stays empty
                self._log(tr("Could not open {name}: {reason}").format(name=Path(path).name, reason=exc))
                return None
            if outcome.error is not None:
                return None
            value = outcome.value
            curve = Curve.from_arrays(value.q, value.intensity, value.error, name=Path(path).name, path=path,
                                      unit=value.q_source_unit)
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
        frame = self.fits.get(index)
        curve = self._curves.get(index)
        started = frame is not None or self._started()
        settings = self._settings if started else self._stage_settings()  # before Start: as Single would fit it
        self._drawn_with = (index, settings, self._preview_unit())
        if curve is None:
            curve = self._preview_curve(index)
        data = None
        if curve is not None:
            try:
                data = prepare_curve(curve, settings.side, settings.q_range, settings.excluded)
            except ValueError:
                data = None
        if data is None or not data.q.size:
            self.frame_plot.set_title(f"{index + 1} · {Path(self.paths[index]).name}")
            self.frame_plot.set_curves([])
            return
        curves = [(tr("measured"), data.q, data.intensity)]
        colors, markers = [DATA_COLOR], [True]
        if frame is not None and frame.ok:
            q = np.geomspace(max(data.q.min(), 1e-6), data.q.max(), 400)
            curves.append((tr("model"), q, evaluate(frame.result.model, q)))
            colors.append(MODEL_COLOR)
            markers.append(False)
        self.frame_plot.set_title(f"{index + 1} · {curve.name}")
        self.frame_plot.set_curves(curves, colors, markers=markers)

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
