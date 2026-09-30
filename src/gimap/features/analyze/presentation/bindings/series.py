"""Behaviour of the Series tab: build the frame × q map, pick rows and q, export.

Build Map reduces every frame (or group of summed frames) with the current settings through the
batch runner (``bindings/batch_run.py``) without writing files: several frames at once when the
series is long, the batch panel with progress, Pause and Stop, and the map redrawn as rows arrive.
A Batch Export fills the same map live. Rows and q windows are picked with draggable bands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.components import show_toast

from ...application import GISAXS, BatchChoices, SeriesMap, build_series_map, track_peak

DEFAULT_CURVE = {GISAXS: "horizontal"}
"""The curve a series map starts with; GIWAXS and others: the radial I(q)."""
REDRAW_EVERY = 5
"""Rows added between redraws of the growing map."""


class SeriesMixin:
    """Needs the Series widgets (``views/series_view.py``), ``view_model``, ``tasks`` and ``_status``."""

    def _connect_series(self) -> None:
        self._series_map: Optional[SeriesMap] = None
        self._series_rows: list = []
        self._series_failures: list[str] = []
        self._series_total = 0
        self._series_row = 0
        self.series_build_button.clicked.connect(self.build_series_map)
        self.series_open_button.clicked.connect(lambda: self.open_series_row())
        self.series_export_csv_action.triggered.connect(self.export_series_csv)
        self.series_export_figure_action.triggered.connect(self.export_series_figure)
        self.series_export_profile_action.triggered.connect(lambda: self._export_series_plot(self.series_profile_plot, "frame"))
        self.series_export_trace_action.triggered.connect(lambda: self._export_series_plot(self.series_trace_plot, "trace"))
        self.series_export_track_action.triggered.connect(lambda: self.export_series_track())
        self.series_trace_combo.currentIndexChanged.connect(lambda _index: self._series_redraw_trace())
        view = self.series_map_view
        view.horizontalBandChanged.connect(lambda low, high: self._series_pick_row(0.5 * (low + high), move_band=False))
        view.verticalBandChanged.connect(lambda low, high: self._series_pick_q(low, high, move_band=False))
        view.positionClicked.connect(self._series_clicked)
        view.set_readout(self._series_readout)

    # -- curves offered ----------------------------------------------------------------

    def _refresh_series_curves(self, analysis) -> None:
        """The curves of the frame on screen, the mode's usual one first chosen."""
        reduction = analysis.reduction if analysis is not None else None
        curves = [curve for curve in (reduction.curves if reduction is not None else ()) if not curve.is_empty]
        current = self.series_curve_combo.currentData()
        with QSignalBlocker(self.series_curve_combo):
            self.series_curve_combo.clear()
            for curve in curves:
                self.series_curve_combo.addItem(curve.title, curve.key)
            keys = [curve.key for curve in curves]
            wanted = current if current in keys else DEFAULT_CURVE.get(getattr(analysis, "kind", None), "radial")
            if wanted in keys:
                self.series_curve_combo.setCurrentIndex(keys.index(wanted))
        files = len(self.view_model.state.files)
        frames = int(getattr(analysis, "frame_count", 1) or 1)
        ready = bool(curves) and (files > 1 or frames > 1)
        self.series_build_button.setEnabled(ready and not self._series_queue and not self.batch_running())
        if self._series_map is None:
            if not curves:
                self.series_info_label.setText("The frame needs a geometry before its curves can be stacked.")
            elif ready:
                count = f"{files} files" if files > 1 else f"{frames} frames"
                self.series_info_label.setText(f"{count} listed: Build Map stacks the chosen curve of every frame.")
            else:
                self.series_info_label.setText("")

    # -- building ----------------------------------------------------------------------

    @property
    def _series_queue(self) -> list:
        """The frames of a map being built still to start (a map-only batch, see ``bindings/batch_run.py``)."""
        return self._batch if getattr(self, "_batch_map_only", False) else []

    def build_series_map(self) -> None:
        """Reduce every listed frame (every n-th) and stack the chosen curve — through the batch runner, so
        a long series uses several processes, shows its progress and can be paused or stopped."""
        key = self.series_curve_combo.currentData()
        requests = self.view_model.batch_requests()[:: max(1, self.series_step_spin.value())]
        if key is None or len(requests) < 2:
            self._status("A series map needs at least two frames with a geometry.", "warning")
            return
        if self.batch_running():
            self._status("A batch is running: its frames appear in this map as they are done.", "warning")
            return
        remembered = self.view_model.batch_preferences()[0]
        choices = BatchChoices(tables=False, every=self.series_step_spin.value(), speed=remembered.speed)
        self.series_export_button.setEnabled(False)
        self.run_batch(Path("."), choices, stem="series", map_only=True, live_key=key)

    def cancel_series_map(self) -> None:
        """Stop after the frames being reduced; the rows so far stay in the map."""
        if getattr(self, "_batch_map_only", False):
            self.cancel_batch()

    def _series_finished(self, failures) -> None:
        """A map-only batch ended: the whole map, a note of what failed, Export."""
        self.series_build_button.setEnabled(True)
        if not self._series_rows:
            self._status("No frame gave the curve: " + "; ".join(failures[:3]), "error")
            self.series_info_label.setText("No frame gave the curve.")
            return
        self._show_series_map(keep_view=False)
        failed = f"; {len(failures)} failed: {', '.join(failures[:3])}" if failures else ""
        text = f"Series map: {len(self._series_rows)} frames of {self.series_curve_combo.currentText()}{failed}"
        self._status(text, "warning" if failed else "ok")
        show_toast(self.window(), text, level="warning" if failed else "ok", action=("Export CSV…", self.export_series_csv))

    def _show_series_map(self, *, keep_view: bool, keep_q: Optional[bool] = None) -> None:
        try:
            series = build_series_map(self._series_rows, self._series_key)
        except ValueError as exc:
            self.series_info_label.setText(str(exc))
            return
        self._series_map = series
        width = float(series.x[-1] - series.x[0]) if series.x.size > 1 else 1.0
        self.series_map_view.set_image(
            series.image, valid=np.isfinite(series.image), rect=(float(series.x[0]), 0.0, width, float(series.rows)),
            y_down=True, title=f"{self.series_curve_combo.currentText()} — {series.rows} frames",
            x_label=series.x_label, y_label="frame", keep_view=keep_view,
            context=f"series:{self._series_key}",  # each curve its own limits (I(q) and I(χ) differ)
        )
        for widget in (self.series_map_view, self.series_plots):
            widget.show()
        self.series_empty.hide()
        stretch = self.series_host.itemAt(self.series_stretch_index)
        if stretch is not None and stretch.spacerItem() is not None:
            self.series_host.setStretch(self.series_stretch_index, 0)
        self.series_export_button.setEnabled(True)
        self.series_info_label.setText(
            f"{series.rows} frames × {series.x.size} points. Click or drag the horizontal band to pick a frame, "
            "drag the vertical band (and its edges) to pick a q window; Open shows the frame in Analyze."
        )
        self._series_pick_row(min(self._series_row, series.rows - 1) + 0.5)
        if not (keep_view if keep_q is None else keep_q) or not hasattr(self, "_series_q"):  # where the series changes most
            centre = series.most_changing_x()
            self._series_pick_q(centre - self._series_step(), centre + self._series_step())
        else:
            self._series_pick_q(*self._series_q)

    def _series_step(self) -> float:
        x = self._series_map.x
        return float(np.nanmedian(np.abs(np.diff(x)))) * 2 if x.size > 1 else 0.01

    # -- picking -----------------------------------------------------------------------

    def _series_pick_row(self, y: float, *, move_band: bool = True) -> None:
        series = self._series_map
        if series is None:
            return
        row = int(np.clip(np.floor(y), 0, series.rows - 1))
        self._series_row = row
        if move_band:
            self.series_map_view.show_horizontal_band(row, row + 1)
        self.series_profile_plot.set_title(f"Frame {row + 1}: {series.labels[row]}")
        self.series_profile_plot.set_labels(series.x_label, "Intensity")
        self.series_profile_plot.set_curves([(f"frame {row + 1}", series.x, series.profile(row))])

    def _series_pick_q(self, low: float, high: float, *, move_band: bool = True) -> None:
        series = self._series_map
        if series is None:
            return
        low, high = sorted((float(low), float(high)))
        self._series_q = (low, high)
        if move_band:
            self.series_map_view.show_vertical_band(low, high)
        self._series_redraw_trace()

    def _series_redraw_trace(self) -> None:
        """The lower-right plot: mean intensity in the q window, or a property of its peak, against frame."""
        series = self._series_map
        if series is None or not hasattr(self, "_series_q"):
            return
        low, high = self._series_q
        centre, half = 0.5 * (low + high), 0.5 * (high - low)
        frames = np.arange(1, series.rows + 1, dtype=float)
        kind = self.series_trace_combo.currentData() or "intensity"
        name = series.x_label.split(" (")[0]
        if kind == "intensity":
            self.series_trace_plot.set_title(f"I at {name} = {centre:.3g} ± {half:.1g}")
            self.series_trace_plot.set_labels("frame", "Intensity")
            self.series_trace_plot.set_curves([(f"{centre:.4g}", frames, series.trace(centre, half))])
            return
        track = track_peak(series, low, high)
        self._series_track = track
        unit = series.x_label[series.x_label.find("("):] if "(" in series.x_label else ""
        labels = {"position": f"peak position {unit}", "fwhm": f"FWHM {unit}", "area": "area", "height": "height"}
        self.series_trace_plot.set_title(f"Peak in {low:.3g}–{high:.3g}")
        self.series_trace_plot.set_labels("frame", labels[kind])
        self.series_trace_plot.set_curves([(kind, frames, dict(track.table())[kind])])

    def export_series_track(self, path: Optional[Path] = None) -> Optional[Path]:
        """Frame, file and the peak in the q window (position, FWHM, area, height) as CSV."""
        series = self._series_map
        if series is None or not hasattr(self, "_series_q"):
            return None
        path = path or self._series_path("Export Peak Table", "peaks.csv", "CSV (*.csv)")
        if path is None:
            return None
        try:
            written = self.view_model.export_series_track(series, track_peak(series, *self._series_q), Path(path))
        except (ValueError, OSError) as exc:
            self._status(f"Could not export the peak table: {exc}", "error")
            return None
        self.notify_written(f"Saved {written.name}", written.parent)
        return written

    def _series_clicked(self, x: float, y: float) -> None:
        self._series_pick_row(y)

    def _series_readout(self, x: float, y: float) -> str:
        series = self._series_map
        if series is None or not 0 <= y < series.rows:
            return ""
        row = int(y)
        column = int(np.argmin(np.abs(series.x - x)))
        value = series.image[row, column]
        text = "—" if not np.isfinite(value) else f"{value:.4g}"
        return f"frame {row + 1} ({series.labels[row]}) · {series.x_label.split(' (')[0]} = {series.x[column]:.4g} · I = {text}"

    def open_series_row(self, row: Optional[int] = None) -> bool:
        """Show the frame of a map row in Analyze (its file, and its frame of a series)."""
        series = self._series_map
        row = self._series_row if row is None else int(row)
        if series is None or not 0 <= row < len(series.refs):
            return False
        path, frame_index = series.refs[row]
        keys = [str(item).casefold() for item in self.view_model.state.files]
        if str(path).casefold() not in keys:
            return False
        index = keys.index(str(path).casefold())
        if self.file_list.currentRow() != index:
            self.file_list.setCurrentRow(index)
        if int(frame_index) != self.view_model.state.frame_index:
            self.view_model.set_frame(int(frame_index))
            self.run_analysis()
        return True

    # -- export ------------------------------------------------------------------------

    def _series_path(self, title: str, suffix: str, filters: str) -> Optional[Path]:
        folder = self.view_model.default_export_dir() or Path(self._last_folder or ".")
        path, _ = QFileDialog.getSaveFileName(self, title, str(folder / f"series_{self._series_key}_{suffix}"), filters)
        return Path(path) if path else None

    def export_series_csv(self, path: Optional[Path] = None) -> Optional[Path]:
        if self._series_map is None:
            return None
        path = path or self._series_path("Export Series Map", "map.csv", "CSV (*.csv)")
        if path is None:
            return None
        try:
            written = self.view_model.export_series_map(self._series_map, Path(path))
        except (ValueError, OSError) as exc:
            self._status(f"Could not export the map: {exc}", "error")
            return None
        self.notify_written(f"Saved {written.name}", written.parent)
        return written

    def export_series_figure(self, path: Optional[Path] = None) -> Optional[Path]:
        state = self.series_map_view.display_state()
        if state is None:
            return None
        path = path or self._series_path("Save Series Map", "map.png", self.FIGURE_FILTER)
        if path is None:
            return None
        image = state.pop("image")
        try:
            written = self.view_model.save_image_figure(Path(path), image, **state)
        except (ValueError, OSError) as exc:
            self._status(f"Could not save the figure: {exc}", "error")
            return None
        self.notify_written(f"Saved {written.name}", written.parent)
        return written

    def _export_series_plot(self, plot, suffix: str) -> None:
        if self._series_map is not None:
            self.save_plot(plot, f"series_{suffix}")


__all__ = ["SeriesMixin"]
