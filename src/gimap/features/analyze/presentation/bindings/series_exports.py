"""What the Series tab saves: the map (table and figure), the frame picked, the trace, the peak table.

Every file is offered under the name of the series, its curve and — for a signed cut — its halves
(``<series>_<curve>[_<halves>]_<what>``), not under the frame that happens to be on screen; the curve of the
frame picked under that frame's name as Batch Export gives it (``<frame>_<curve>[_<halves>]``). The tables say
which curve, halves, frames and cut they hold (``application/series_map.series_header``); the table of the
map has its settings in a JSON record next to it. The frame picked and the trace are saved as a figure or
as a table of their numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import tr, trf

from ...application import frame_stem, track_peak
from ..series_texts import TRACE_NO_POINT, halves_words, row_fact
from ..texts import message_text

TABLE_FILTER = "CSV (*.csv)"
ROW_AXIS = "row of the map (frames not evenly spaced: their numbers are in the table)"
"""The frame axis of a saved figure whose rows are not evenly spaced in frame number or time."""


def figure_rows(values) -> Optional[tuple[float, float]]:
    """``(top, bottom)`` of a picture whose rows are named by ``values`` in their middle: the first row begins
    half a step above its value, the last ends half a step below its own. ``None`` when the values are not
    evenly spaced (rows left out), where no linear axis names every row."""
    values = np.asarray(list(values), dtype=float)
    if values.size == 1:
        return float(values[0]) - 0.5, float(values[0]) + 0.5
    steps = np.diff(values)
    if values.size < 2 or steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
        return None
    return float(values[0] - 0.5 * steps[0]), float(values[-1] + 0.5 * steps[0])


class SeriesExportsMixin:
    """Needs the Series widgets, ``_series_map`` and its state, ``view_model``, ``_status`` and ``notify_written``."""

    def _connect_series_exports(self) -> None:
        self.series_export_csv_action.triggered.connect(lambda: self.export_series_csv())
        self.series_export_figure_action.triggered.connect(lambda: self.export_series_figure())
        self.series_export_profile_action.triggered.connect(lambda: self.export_series_frame_figure())
        self.series_export_profile_data_action.triggered.connect(lambda: self.export_series_frame_table())
        self.series_export_trace_action.triggered.connect(lambda: self.export_series_trace_figure())
        self.series_export_trace_data_action.triggered.connect(lambda: self.export_series_trace_table())
        self.series_export_track_action.triggered.connect(lambda: self.export_series_track())

    def _series_halves(self) -> str:
        series = self._series_map
        return "" if series is None or series.view == "both" else f"_{series.view}"

    def _series_stem(self) -> str:
        """``<series>_<curve>[_<halves>]``: what every file of this map is named after."""
        return f"{self._series_name or 'series'}_{self._series_key}{self._series_halves()}"

    def _series_path(self, title: str, suffix: str, filters: str, *, name: Optional[str] = None) -> Optional[Path]:
        """Ask where to save; offered: ``<series stem>_<suffix>``, or ``name`` as it is."""
        folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
        offered = name or f"{self._series_stem()}_{suffix}"
        path, _ = QFileDialog.getSaveFileName(self, tr(title), str(folder / offered), filters)
        return Path(path) if path else None

    def _series_saved(self, write, path: Optional[Path], failed: str) -> Optional[Path]:
        """Run ``write(path)``; say where it went, or — with ``failed`` (a template with ``{error}``) — why not."""
        if path is None:
            return None
        try:
            written = write(Path(path))
        except (ValueError, OSError) as exc:
            self._status(trf(failed, error=message_text(exc)), "error")
            return None
        self.notify_written(trf("Saved {name}", name=written.name), written.parent)
        return written

    # -- the map ---------------------------------------------------------------------------------------

    def export_series_csv(self, path: Optional[Path] = None) -> Optional[Path]:
        series = self._series_map
        if series is None:
            return None
        path = path or self._series_path("Export Series Map", "map.csv", TABLE_FILTER)
        return self._series_saved(lambda target: self.view_model.export_series_map(series, target), path,
                                  "Could not export the map: {error}")

    def export_series_figure(self, path: Optional[Path] = None) -> Optional[Path]:
        """The map as shown (colours, log, limits), frame 1 at the top as on screen, the frame axis in the
        numbers (or seconds) of the rows."""
        series, state = self._series_map, self.series_map_view.display_state()
        if series is None or state is None:
            return None
        path = path or self._series_path("Save Series Map", "map.png", self.FIGURE_FILTER)
        image = state.pop("image")
        left, right = state["extent"][:2]
        values = self._series_axis_values(series)
        rows = figure_rows(values)
        state["y_label"] = self._series_axis_label(series)
        if rows is None:
            rows = (0.5, series.rows + 0.5)
            ticks = np.unique(np.linspace(0, series.rows - 1, min(series.rows, 6), dtype=int))
            state["y_ticks"] = [(float(row + 1), f"{values[row]:g}") for row in ticks]
        state["extent"] = (left, right, rows[1], rows[0])  # (left, right, bottom, top): the first row on top
        if halves_words(series):  # a short title (a long one is cut at the picture's edges): the curve, the halves
            state["title"] = " · ".join(part for part in (state.get("title", ""), halves_words(series)) if part)
        return self._series_saved(lambda target: self.view_model.save_image_figure(target, image, **state), path,
                                  "Could not save the figure: {error}")

    # -- the frame picked and the trace ----------------------------------------------------------------

    def _series_figure(self, plot, suffix: str, path: Optional[Path], *, name: Optional[str] = None) -> Optional[Path]:
        if self._series_map is None:
            return None
        path = path or self._series_path("Save Plot", f"{suffix}.png", self.FIGURE_FILTER,
                                         name=f"{name}.png" if name else None)
        state = plot.figure_state()
        curves = state.pop("curves")
        return self._series_saved(lambda target: self.view_model.save_curves_figure(target, curves, **state), path,
                                  "Could not save the plot: {error}")

    def _series_frame_name(self) -> str:
        """``<frame>_<curve>[_<halves>]`` of the row picked, the frame named as Batch Export names it
        (``frame_stem``): its file — a set of module files by its scan —, ``_frameNNNN`` (the 0-based index) for
        a frame of a multi-frame source, ``_sumK`` for a row of K summed frames. The 1-based number people
        count in is in the table's header and the figure's title."""
        series, row = self._series_map, self._series_row
        fact = row_fact(series, row)
        if fact is None:  # (nothing known of the row but its place)
            frame = f"{self._series_name or 'series'}_frame{int(row):04d}"
        else:
            frame = frame_stem(fact.source, self.view_model.describe(Path(fact.source)), frame_index=fact.frame_index,
                               frame_count=fact.frame_count, frame_total=fact.summed)
        return f"{frame}_{self._series_key}{self._series_halves()}"

    def export_series_frame_figure(self, path: Optional[Path] = None) -> Optional[Path]:
        if self._series_map is None:
            return None
        return self._series_figure(self.series_profile_plot, "", path, name=self._series_frame_name())

    def export_series_frame_table(self, path: Optional[Path] = None) -> Optional[Path]:
        """The curve of the frame picked, as it is in the map: x and intensity."""
        series, row = self._series_map, self._series_row
        if series is None:
            return None
        path = path or self._series_path("Export the Frame's Curve", "", TABLE_FILTER,
                                         name=f"{self._series_frame_name()}.csv")
        return self._series_saved(lambda target: self.view_model.write_series_curve(series, row, target), path,
                                  "Could not export the curve: {error}")

    def _series_trace_suffix(self) -> str:
        return f"trace_{self.series_trace_combo.currentData() or 'intensity'}"

    def export_series_trace_figure(self, path: Optional[Path] = None) -> Optional[Path]:
        return self._series_figure(self.series_trace_plot, self._series_trace_suffix(), path)

    def export_series_trace_table(self, path: Optional[Path] = None) -> Optional[Path]:
        """The value the trace plot shows for every frame."""
        series, found = self._series_map, self._series_trace_values()
        if series is None or found is None:
            self._status(tr("The trace has no values yet."), "warning")
            return None
        if found[0] != "change" and self._series_window()[1] == "none":  # nothing measured in the window
            self._status(tr(TRACE_NO_POINT), "warning")
            return None
        kind, values, words = found
        path = path or self._series_path("Export the Trace", f"{self._series_trace_suffix()}.csv", TABLE_FILTER)
        return self._series_saved(
            lambda target: self.view_model.write_series_trace(series, kind, values, target, what=words), path,
            "Could not export the trace: {error}")

    def export_series_track(self, path: Optional[Path] = None) -> Optional[Path]:
        """Frame, file and the peak in the x window (position, moment width, area, height) as CSV."""
        series = self._series_map
        if series is None or not hasattr(self, "_series_q"):
            return None
        path = path or self._series_path("Export Peak Table", "peaks.csv", TABLE_FILTER)
        track = track_peak(series, *self._series_q)
        return self._series_saved(lambda target: self.view_model.export_series_track(series, track, target), path,
                                  "Could not export the peak table: {error}")


__all__ = ["ROW_AXIS", "SeriesExportsMixin", "TABLE_FILTER", "figure_rows"]
