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
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.stage_text import axis_symbol

from ...application import GISAXS, BatchChoices, SeriesMap, build_series_map, track_peak
from ..texts import curve_title, message_text
from ..views.series_view import CHANGE_EMPTY, CHANGE_FEW
from .series_stages import series_name

DEFAULT_CURVE = {GISAXS: "horizontal"}
"""The curve a series map starts with; GIWAXS and others: the radial I(q)."""
REDRAW_EVERY = 5
"""Rows added between redraws of the growing map."""


class SeriesMixin:
    """Needs the Series widgets (``views/series_view.py``), ``view_model``, ``tasks`` and ``_status``."""

    def _connect_series(self) -> None:
        self._series_map: Optional[SeriesMap] = None
        self._series_name: Optional[str] = None
        """The map's own name (from the files of its rows), kept when the file list changes."""
        self._series_rows: list = []
        self._series_failures: list[str] = []
        self._series_total = 0
        self._series_row = 0
        # Every way files are listed (opened, dropped, folder watching) tells a map of the earlier list.
        self.file_list.model().rowsInserted.connect(lambda *_args: self._series_list_grew())
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
                self.series_curve_combo.addItem(curve_title(curve.title), curve.key)
            keys = [curve.key for curve in curves]
            wanted = current if current in keys else DEFAULT_CURVE.get(getattr(analysis, "kind", None), "radial")
            if wanted in keys:
                self.series_curve_combo.setCurrentIndex(keys.index(wanted))
        popup = self.series_curve_combo.view()  # the combo may be narrower than a name: its list shows them whole
        popup.setMinimumWidth(popup.sizeHintForColumn(0) + 2 * popup.frameWidth() + 24 if curves else 0)
        files = len(self.view_model.state.files)
        frames = int(getattr(analysis, "frame_count", 1) or 1)
        ready = bool(curves) and (files > 1 or frames > 1)
        self.series_controls.setVisible(files > 1 or frames > 1)  # one frame: only the explanation
        self.series_build_button.setEnabled(ready and not self._series_queue and not self.batch_running())
        if self._series_map is None:
            if not curves:
                self._series_say("The frame needs a geometry before its curves can be stacked.")
            elif ready and files > 1:
                self._series_say("{n} files listed: Build Map stacks the chosen curve of every frame.", n=files)
            elif ready:
                self._series_say("{n} frames listed: Build Map stacks the chosen curve of every frame.", n=frames)
            else:
                self._series_say("")

    def _series_say(self, template: str, **values) -> None:
        """The sentence under the Series controls: ``template`` (English) filled with ``values`` in the interface
        language, kept so a switch of the language says it again (``_series_language``)."""
        self._series_info = (template, values)
        self.series_info_label.setText(trf(template, **values) if values else tr(template))

    def _series_language(self, analysis) -> None:
        """After a switch of the interface language: the curve names, the sentence under the controls, the map's
        title and the titles of the frame and trace plots again (nothing is reduced again)."""
        self._refresh_series_curves(analysis)
        info = getattr(self, "_series_info", None)
        if info is not None and info[0]:
            self.series_info_label.setText(trf(info[0], **info[1]) if info[1] else tr(info[0]))
        series = self._series_map
        if series is None:
            return
        self.series_map_view.title_label.setText(self._series_title(series))
        self._series_pick_row(self._series_row + 0.5, move_band=False)
        self._series_redraw_trace()

    def _series_title(self, series) -> str:
        return trf("{curve} — {n} frames", curve=self.series_curve_combo.currentText(), n=series.rows)

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
            self._status(tr("A series map needs at least two frames with a geometry."), "warning")
            return
        if self.batch_running():
            self._status(tr("A batch is running: its frames appear in this map as they are done."), "warning")
            return
        remembered = self.view_model.batch_preferences()[0]
        choices = BatchChoices(tables=False, every=self.series_step_spin.value(), speed=remembered.speed)
        self.series_export_button.hide()
        self._clear_stages()
        self.run_batch(Path("."), choices, stem="series", map_only=True, live_key=key)

    def cancel_series_map(self) -> None:
        """Stop after the frames being reduced; the rows so far stay in the map."""
        if getattr(self, "_batch_map_only", False):
            self.cancel_batch()

    def _reset_series(self) -> None:
        """The file list was cleared: no map, the explanation of the empty tab again.

        A map being built is stopped and its late rows are dropped; a running Batch Export goes on
        writing its files, but no longer draws its frames into the map."""
        if getattr(self, "_batch_map_only", False) and self.batch_running():
            self._series_dropped = True  # ``_series_finished`` then says nothing
            self.cancel_series_map()
        self._batch_live = False
        self._series_map, self._series_name = None, None
        self._series_rows = []
        self._series_row = 0
        self._clear_stages()
        for widget in (self.series_map_view, self.series_plots, self.series_export_button, self.series_compare_button):
            widget.hide()
        self.series_empty.show()
        stretch = self.series_host.itemAt(self.series_stretch_index)
        if stretch is not None and stretch.spacerItem() is not None:
            self.series_host.setStretch(self.series_stretch_index, 1)
        self._series_say("")

    def _series_list_grew(self) -> None:
        """Files were added: a map built before still holds for its own frames, and says so."""
        series = self._series_map
        if series is not None and not getattr(self, "_batch_live", False):
            self._series_say("Map of the earlier list ({n} frames) — Build Map again to include the new files", n=series.rows)

    def _series_drop_file(self, path, *, summed_across: bool = False) -> None:
        """A file left the list (not while a batch runs): its rows leave the map, which is drawn again, and its
        stages are found again; fewer than two rows left: no map. With frames summed across files
        (``summed_across``) every later group changes, so the map stays and says it is of the earlier list."""
        series = self._series_map
        if series is None:
            return
        key = str(path).casefold()
        kept = [row for row in self._series_rows if str(row[4][0]).casefold() != key]
        if summed_across:
            self._series_say("Map of the earlier list ({n} frames) — Build Map again without the removed file",
                             n=series.rows)
            return
        if len(kept) == len(self._series_rows):
            return  # none of its frames is in the map (every n-th frame)
        if len(kept) < 2:
            self._reset_series()
            return
        self._series_rows = kept
        self._clear_stages()
        self._show_series_map(keep_view=False, keep_q=True)
        self._find_stages()

    def _series_finished(self, failures) -> None:
        """A map-only batch ended: the whole map, a note of what failed, Export."""
        if getattr(self, "_series_dropped", False):  # the list was cleared while it ran: nothing to show
            self._series_dropped = False
            self._series_rows = []
            self.series_build_button.setEnabled(len(self.view_model.state.files) > 1)
            return
        self.series_build_button.setEnabled(True)
        if not self._series_rows:
            self._status(trf("No frame gave the curve: {reasons}", reasons="; ".join(failures[:3])), "error")
            self._series_say("No frame gave the curve.")
            return
        self._show_series_map(keep_view=False)
        values = dict(n=len(self._series_rows), curve=self.series_curve_combo.currentText())
        text = (trf("Series map: {n} frames of {curve}; {failed} failed: {names}", failed=len(failures),
                    names=", ".join(failures[:3]), **values) if failures else
                trf("Series map: {n} frames of {curve}", **values))
        self._status(text, "warning" if failures else "ok")
        show_toast(self.window(), text, level="warning" if failures else "ok",
                   action=(tr("Export CSV…"), self.export_series_csv))
        self._find_stages()

    def _show_series_map(self, *, keep_view: bool, keep_q: Optional[bool] = None) -> None:
        try:
            series = build_series_map(self._series_rows, self._series_key)
        except ValueError as exc:
            self._series_say(str(exc))
            return
        self._series_map = series
        self._series_name = series_name(_ref_files(series))
        width = float(series.x[-1] - series.x[0]) if series.x.size > 1 else 1.0
        self.series_map_view.set_image(
            series.image, valid=np.isfinite(series.image), rect=(float(series.x[0]), 0.0, width, float(series.rows)),
            y_down=True, title=self._series_title(series),
            x_label=series.x_label, y_label="frame", keep_view=keep_view,
            context=f"series:{self._series_key}",  # each curve its own limits (I(q) and I(χ) differ)
        )
        for widget in (self.series_map_view, self.series_plots):
            widget.show()
        self.series_empty.hide()
        stretch = self.series_host.itemAt(self.series_stretch_index)
        if stretch is not None and stretch.spacerItem() is not None:
            self.series_host.setStretch(self.series_stretch_index, 0)
        self.series_export_button.show()
        self._series_say(
            "{rows} frames × {points} points. Click or drag the horizontal band to pick a frame, drag the vertical "
            "band (and its edges) to pick a {axis} window; Open shows the frame in Analyze.",
            rows=series.rows, points=series.x.size, axis=axis_symbol(series.x_label))
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
        self.series_profile_plot.set_title(trf("Frame {n}: {label}", n=row + 1, label=series.labels[row]))
        self.series_profile_plot.set_labels(series.x_label, "Intensity")
        self.series_profile_plot.set_curves([(trf("frame {n}", n=row + 1), series.x, series.profile(row))])

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
        # The stages are looked for in a map of three frames or more (``_find_stages``).
        self.series_trace_plot.set_empty_text("" if kind != "change" else CHANGE_EMPTY if series.rows >= 3 else CHANGE_FEW)
        if kind == "change" and self._draw_change_trace():
            return
        name = series.x_label.split(" (")[0]
        if kind == "intensity":
            self.series_trace_plot.set_title(trf("I at {axis} = {centre} ± {half}", axis=name, centre=f"{centre:.3g}",
                                                 half=f"{half:.1g}"))
            self.series_trace_plot.set_labels("frame", "Intensity")
            self.series_trace_plot.set_curves([(f"{centre:.4g}", frames, series.trace(centre, half))])
            return
        track = track_peak(series, low, high)
        self._series_track = track
        unit = series.x_label[series.x_label.find("("):] if "(" in series.x_label else ""
        labels = {"position": f"peak position {unit}", "fwhm": f"FWHM {unit}", "area": "area", "height": "height"}
        self.series_trace_plot.set_title(trf("Peak in {low}–{high}", low=f"{low:.3g}", high=f"{high:.3g}"))
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
            self._status(trf("Could not export the peak table: {error}", error=message_text(exc)), "error")
            return None
        self.notify_written(trf("Saved {name}", name=written.name), written.parent)
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
        return trf("frame {n} ({label})", n=row + 1, label=series.labels[row]) + (
            f" · {series.x_label.split(' (')[0]} = {series.x[column]:.4g} · I = {text}")

    def open_series_row(self, row: Optional[int] = None) -> bool:
        """Show the frame of a map row in Analyze (its file, and its frame of a series)."""
        series = self._series_map
        row = self._series_row if row is None else int(row)
        if series is None or not 0 <= row < len(series.refs):
            return False
        path, frame_index = series.refs[row]
        keys = [str(item).casefold() for item in self.view_model.state.files]
        if str(path).casefold() not in keys:
            self._status(tr("That frame is no longer listed"), "warning")
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
        path, _ = QFileDialog.getSaveFileName(self, tr(title), str(folder / f"series_{self._series_key}_{suffix}"), filters)
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
            self._status(trf("Could not export the map: {error}", error=message_text(exc)), "error")
            return None
        self.notify_written(trf("Saved {name}", name=written.name), written.parent)
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
            self._status(trf("Could not save the figure: {error}", error=message_text(exc)), "error")
            return None
        self.notify_written(trf("Saved {name}", name=written.name), written.parent)
        return written

    def _export_series_plot(self, plot, suffix: str) -> None:
        if self._series_map is not None:
            self.save_plot(plot, f"series_{suffix}")


def _ref_files(series) -> list[Path]:
    """The files of a map's rows, each once, in row order (a multi-frame file gives several rows)."""
    files: dict[str, Path] = {}
    for path, _frame in getattr(series, "refs", None) or ():
        files.setdefault(str(path).casefold(), Path(path))
    return list(files.values())


__all__ = ["SeriesMixin"]
