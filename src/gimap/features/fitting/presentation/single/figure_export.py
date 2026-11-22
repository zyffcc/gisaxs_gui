"""Save Plot: the curve, the model, its terms as shown and the residuals in one figure — PNG, SVG or PDF, on white,
with the points' σ as error bars (``infrastructure/adapters/curve_figures.py``, ``shared/figures``).

The figure shows what the plot shows: the points outside the fitting range (grey) and left out (red ×), the fitted
points with their error bars, the model, the terms that are switched on (Components), the fitting range shaded, the
axes as they are (log q, log I) — and under it the residuals (Δ/σ, or Δ ln I without σ). A file of another type is
refused before anything is written, with what to choose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.gimap.app.presentation.i18n import tr

from ...application.models import ExportCurveFigureRequest, FigureSeries
from ...application.single_fit import left_out_points, prepare_curve, residuals
from .plotting import LEFT_OUT_COLOR, MODEL_COLOR, OUTSIDE_COLOR, component_style, plotted_model
from .points import BOTH_HALVES, DATA_COLOR, measured_series

FIGURE_SUFFIXES = (".png", ".svg", ".pdf")
WRONG_TYPE = "Choose a .png, .svg or .pdf file: {name} is none of them."
NO_FOLDER = "the folder {folder} does not exist"
INTENSITY = "Intensity (a.u.)"


def _part_name(name: str) -> str:
    """A term's name for the figure's legend (not translated: a figure keeps its own words)."""
    return {"background_B": "constant background", "resolution peak": "beam-tail term"}.get(name, name)


class FigureExportMixin:
    """Needs the page's ``session``, ``_data``, ``plot``, ``terms_check``, ``_component_choices``, ``view_model``,
    ``_status``."""

    def figure_request(self, path) -> ExportCurveFigureRequest:
        session = self.session
        full, data = session.all_points(), self._data()
        series, below = [], []
        inside = np.zeros(full.q.size, bool) if data is None else np.isin(full.q, data.q)
        left_out = left_out_points(full.q, session.excluded) if session.excluded else np.zeros(full.q.size, bool)
        outside = ~inside & ~left_out
        if outside.any():
            series.append(FigureSeries("outside the range", full.q[outside], full.intensity[outside], OUTSIDE_COLOR))
        if left_out.any():
            series.append(FigureSeries("left out", full.q[left_out], full.intensity[left_out], LEFT_OUT_COLOR,
                                       marker="x"))
        halves = {name: side for name, side, _color in BOTH_HALVES}
        for name, q, intensity, color in measured_series(session, data):
            sigma = None
            if session.model.weighting == "file" and data is not None:
                # Only an uncertainty read from the file is an experimental error bar.
                sigma = (prepare_curve(session.curve, halves[name], session.q_range, session.excluded).sigma
                         if name in halves else data.sigma)
            series.append(FigureSeries(name, q, intensity, color, error=sigma))
        q_model, total, parts = plotted_model(session.model, full, data)
        series.append(FigureSeries("model", q_model, total, MODEL_COLOR, style="line"))
        if self.terms_check.isChecked():
            for index, (name, values) in enumerate(parts.items()):
                if self._component_choices.get(name, True):
                    series.append(FigureSeries(_part_name(name), q_model, values, component_style(name, index)[0],
                                               style="line", dashed=True))
        if data is not None and data.q.size:
            below = [FigureSeries("0", data.q[[0, -1]], np.zeros(2), OUTSIDE_COLOR, style="line"),
                     FigureSeries("residual", data.q, residuals(session.model, data), DATA_COLOR)]
        band = session.q_range or (float(full.q.min()), float(full.q.max()))
        weighted = data is not None and data.sigma is not None
        return ExportCurveFigureRequest(
            Path(path), tuple(series), "|q| (nm⁻¹)", INTENSITY, "log" if self.plot.log_x_check.isChecked() else "linear",
            log_y=self.plot.log_check.isChecked(), residuals=tuple(below),
            residual_label="Δ/σ" if weighted else "Δ ln I", band=tuple(band))

    def write_figure(self, path):
        from .results import folder_action

        path = Path(path)
        if path.suffix.lower() not in FIGURE_SUFFIXES:
            self._status(lambda: tr(WRONG_TYPE).format(name=path.name), "error")
            return None
        if self.session.curve is None:
            self._status(tr("Open a curve first."), "warning")
            return None
        if not path.parent.is_dir():  # as every save of the page: a folder is not made behind the person's back
            self._status(lambda: tr("Could not save: {error}").format(error=tr(NO_FOLDER).format(folder=path.parent)),
                         "error")
            return None
        try:
            request = self.figure_request(path)
        except ValueError as exc:
            self._status(tr(str(exc)), "error")
            return None
        outcome = self.view_model.export_curve_figure(request)
        if outcome.error is not None:
            self._status(lambda: tr("Could not save: {error}").format(error=outcome.error.message), "error")
            return None
        self._status(lambda: tr("Saved {name}.").format(name=path.name), "ok", action=folder_action(path))
        return str(path)


__all__ = ["FIGURE_SUFFIXES", "INTENSITY", "NO_FOLDER", "WRONG_TYPE", "FigureExportMixin"]
