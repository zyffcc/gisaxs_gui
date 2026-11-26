"""The picture of the Series map and the line that says what it is.

* **Where a line is drawn.** The x of a curve is not evenly spaced (one point per valid detector column,
  merged bins at low q, holes at detector gaps), the columns of an image are: the map is drawn from
  ``SeriesMap.display()`` — its picture on an even grid, each column at its true x, gaps left blank. The
  values and the x of the map itself (bands, read-out, traces, exports) are untouched.
* **The frame axis** counts frames as people do — the frame in its file, or its place in the list, from 1 —
  with a tick in the middle of a row, never a fraction; or the time since the first row when the frames
  carry one (Options ▸ Frame axis). Its ticks are those of the rows in view, again at every zoom and pan.
* **A live map of many rows** is stacked on a thread of its own and shown when done (``series_later.py``):
  ``_series_stacking`` is the work, ``_series_show_built`` the showing.
* **The halves** of a signed cut (qy on both sides of the beam): as measured, one half, both on |qy|, or
  their mean. The map is stacked again from the rows kept, nothing is reduced again; the stages, the
  traces and the exports follow.
* **What the map is** stands above it: its curve, halves and frames, the cut of the series, and — when the
  Curve box or the settings no longer are what it was built with — that it must be built again.
* **Wide** gives the tab the room of the image beside it (and gives it back).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QSignalBlocker, Qt

from src.gimap.app.presentation.components.detector_view import display_image
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.stage_text import axis_symbol

from ...application import GISAXS, build_series_map
from ..frame_axis import MAJOR_TICKS, MINOR_ROWS, frame_ticks, nice_step
from ..series_texts import (
    AXIS_LABELS, HOW_TO, HOW_TO_SUMS, MAP_TIME, MAP_WAITING, NO_TIME, STALE_CURVE, STALE_SETTINGS, cut_sentence,
    map_group, map_words, sum_note,
)
from ..views.series_view import VIEW_TIPS
from .series_stages import series_name

RING_CURVE = "azimuthal"
"""The GIWAXS curve the cut of a series is about (I(χ) of the ring); every GISAXS curve is a cut."""
HALVES_CURVE = "horizontal"
"""The curve that has halves to choose: the horizontal GISAXS cut, I(qy) on both sides of the beam."""
SUM_MARKS = ("sums", "Incomplete sums")
"""The kind of mark (key, name in the map's Marks menu) on a row that sums fewer frames than the others."""
SUM_PLATE = (0, 0, 0, 170)
"""RGBA of the plate under such a mark (white text: read over the bright and the dark end of every colour map)."""


def row_files(rows) -> list[Path]:
    """The files of a map's rows, each once, in row order (a multi-frame file gives several rows)."""
    files: dict[str, Path] = {}
    for row in rows:
        source = row[4][0]
        text = source if isinstance(source, str) else str(source)
        key = text.casefold()
        if key not in files:  # (a row's own path as it is: a live map of 20 000 rows made as many again)
            files[key] = source if isinstance(source, Path) else Path(text)
    return list(files.values())


class SeriesDisplayMixin:
    """Needs the Series widgets, ``view_model``, ``splitter``, ``even_split`` and the state of ``SeriesMixin``."""

    def _connect_series_display(self) -> None:
        self._series_view = "both"
        """The halves a map of a signed cut is stacked in (``series_view_combo``)."""
        self._series_built: dict = {}
        """What the map was built with (``view_model.series_built_with`` when its run began)."""
        self._series_run = None
        self._series_extent: Optional[tuple[float, float]] = None
        """The x the picture of the map spans (its left and right edge)."""
        self._series_wide_before: Optional[tuple] = None
        self._series_sum_marks: list = []
        self.series_map_view.marks.add_layer(*SUM_MARKS)
        self.series_view_combo.activated.connect(self._series_view_chosen)
        self.series_axis_combo.activated.connect(lambda _index: self._series_axis_chosen())
        self.series_wide_button.toggled.connect(self._series_wide_toggled)
        self.series_curve_combo.currentIndexChanged.connect(lambda _index: self._series_mark_stale())
        self.series_map_view.view_box.sigYRangeChanged.connect(lambda *_args: self._series_ticks_in_view())

    # -- the map ---------------------------------------------------------------------------------------

    def _show_series_map(self, *, keep_view: bool, keep_q: Optional[bool] = None, live: bool = False) -> None:
        """Stack the map of the rows now and show it, here and at once (a map stacked meanwhile on the map's own
        thread is not shown: ``series_later.py``). ``live``: the map of a live series, which grows from the map before
        (``series_later.LiveStack``); else it is made whole."""
        self._series_later_forget()
        if not live:
            self._series_live_forget()
        self._series_split_angles()  # one incidence angle per map, the others aside (``series_mixed.py``)
        began = time.monotonic()
        try:
            built = self._series_stacking(live=live)()
        except ValueError as exc:
            self._series_say(str(exc))
            return
        self._series_build_s = time.monotonic() - began
        self._series_show_built(built, keep_view=keep_view, keep_q=keep_q)

    def _series_stacking(self, *, live: bool = False):
        """The work of stacking the map of the rows now, as a function of nothing that touches no widget (to run on
        any thread): it returns ``(map, name, (picture, left, width), (valid, prepared))`` — ``prepared`` the
        picture as the map view draws it (``DetectorView.set_image``). Raises ``ValueError`` for rows that make no
        map. ``live``: grown from the map before when it can be (``series_later.LiveStack``): the same map."""
        built = self._series_built
        rows, key = list(self._series_rows), self._series_key
        numbers = built.get("numbers")  # (the frame list's cannot change; a dictionary might, while the map is made)
        options = dict(
            view=self._series_view if key == HALVES_CURVE else "both",
            title=getattr(self, "_series_curve_title", ""), numbers=dict(numbers) if isinstance(numbers, dict) else numbers,
            group=int(built.get("group") or 1),
        )
        built_with = {
            "listed": built.get("listed"), "every": built.get("every"), "cut": built.get("cut"),
            "cut_text": built.get("cut_text"), "settings": built.get("settings"),
            "first_frame": getattr(self._series_run, "first_metadata", None), "incidence": built.get("incidence"),
        }

        log = self.series_map_view.log_check.isChecked()
        stacker = self._series_live_stack() if live else None

        def stack():
            if stacker is not None:
                built = stacker.stack(rows, key, options, built_with, log)
                if built is not None:
                    return built
            name = series_name(row_files(rows))
            series = build_series_map(rows, key, built_with={"name": name, **built_with}, **options)
            picture, left, width = series.display()
            valid = np.isfinite(picture)
            return series, name, (picture, left, width), (valid, (log, *display_image(picture, valid, log)))

        return stack

    def _series_show_built(self, built, *, keep_view: bool, keep_q: Optional[bool] = None) -> None:
        """Show a stacked map (``_series_stacking``)."""
        began = time.monotonic()
        try:
            self._draw_series_map(built, keep_view=keep_view, keep_q=keep_q)
        finally:
            self._series_draw_s = time.monotonic() - began  # (a live map waits accordingly: ``redraw_wait_s``)

    def _draw_series_map(self, built, *, keep_view: bool, keep_q: Optional[bool] = None) -> None:
        series, name, (picture, left, width), (valid, prepared) = built
        self._series_map, self._series_name = series, name
        self._series_extent = (left, left + width)
        self.series_map_view.set_image(
            picture, valid=valid, prepared=prepared, rect=(left, 0.0, width, float(series.rows)),
            y_down=True, title=self._series_title(series),
            x_label=series.x_label, y_label=self._series_axis_label(series), keep_view=keep_view,
            context=f"series:{self._series_key}",  # each curve its own limits (I(q) and I(χ) differ)
        )
        self._series_draw_frame_axis(series)
        self._series_mark_short_sums(series)
        signed = series.view != "both" or (series.x.size > 1 and float(series.x[0]) < 0.0 < float(series.x[-1]))
        self.series_view_combo.setVisible(signed and self._series_key == HALVES_CURVE)
        for widget in (self.series_map_view, self.series_plots, self.series_title_label, self.series_wide_button):
            widget.show()
        self.series_empty.hide()
        stretch = self.series_host.itemAt(self.series_stretch_index)
        if stretch is not None and stretch.spacerItem() is not None:
            self.series_host.setStretch(self.series_stretch_index, 0)
        self.series_export_button.show()
        # The map is the thing now: its size and how to pick a frame or a window are its tooltip (the line above it
        # says what it holds); rows of sums are no frames.
        group = map_group(series)
        values = dict(rows=series.rows, points=series.x.size, axis=axis_symbol(series.x_label))
        self._series_how = (HOW_TO_SUMS, dict(n=group, **values)) if group > 1 else (HOW_TO, values)
        self._series_show_how()
        self._series_say("")
        self._series_describe()
        self._series_check_exposures()
        self._series_reload_events()
        self._series_mark_events()
        self._series_pick_row(min(self._series_row, series.rows - 1) + 0.5)
        if not (keep_view if keep_q is None else keep_q) or not hasattr(self, "_series_q"):  # where the series changes most
            centre = series.most_changing_x()
            self._series_pick_q(centre - self._series_step(), centre + self._series_step())
        else:
            self._series_pick_q(*self._series_q)

    def _series_show_how(self) -> None:
        """The map's tooltip: its rows and points, how to pick a row and an x window (again after a language switch)."""
        template, values = getattr(self, "_series_how", ("", {})) if self._series_map is not None else ("", {})
        self.series_map_view.setToolTip(trf(template, **values) if template else "")

    def _series_step(self) -> float:
        x = self._series_map.x
        return float(np.nanmedian(np.abs(np.diff(x)))) * 2 if x.size > 1 else 0.01

    def _series_mark_short_sums(self, series=None) -> None:
        """A plate on every row that sums fewer frames than the others (“sum of 2 of 5”): the last group of a
        Sum N is no row like the others — its intensity is the sum of fewer frames. ``None``: no marks."""
        import pyqtgraph as pg

        view = self.series_map_view
        for item in self._series_sum_marks:
            view.marks.forget(SUM_MARKS[0], item)
            try:
                view.plot.removeItem(item)
            except RuntimeError:  # the scene is gone
                pass
        self._series_sum_marks = []
        if series is None or self._series_extent is None:
            return
        low, high = self._series_extent
        for row in np.flatnonzero(series.short_rows):
            fact = series.facts[int(row)]
            mark = pg.TextItem(sum_note(fact), color="#ffffff", anchor=(1.0, 0.5), fill=pg.mkBrush(*SUM_PLATE))
            mark.setPos(high - 0.05 * (high - low), row + 0.5)  # left of the strip of the stages
            mark.setZValue(13)
            view.plot.addItem(mark, ignoreBounds=True)
            view.marks.track(SUM_MARKS[0], mark, True)
            self._series_sum_marks.append(mark)

    # -- the frame axis --------------------------------------------------------------------------------

    def _series_in_time(self, series) -> bool:
        return self.series_axis_combo.currentData() == "time" and getattr(series, "times", None) is not None

    def _series_axis_values(self, series) -> np.ndarray:
        """What the frame axis counts for every row: the frame numbers, or the seconds since the first row."""
        if self._series_in_time(series):
            return np.asarray(series.times, dtype=float)
        numbers = getattr(series, "numbers", None) or range(1, series.rows + 1)
        return np.asarray(list(numbers), dtype=float)

    def _series_axis_label(self, series) -> str:
        return AXIS_LABELS["time" if self._series_in_time(series) else "frame"]

    def _series_draw_frame_axis(self, series) -> None:
        """Frame numbers (or seconds) in the middle of the rows; “Time” is offered when the frames carry one."""
        timed = getattr(series, "times", None) is not None
        index = self.series_axis_combo.findData("time")
        item = self.series_axis_combo.model().item(index) if index >= 0 else None
        if item is not None:
            item.setEnabled(timed)
            item.setToolTip(tr(MAP_TIME[series.time_from]) if timed and series.time_from in MAP_TIME else tr(NO_TIME))
        if not timed and self.series_axis_combo.currentData() == "time":
            with QSignalBlocker(self.series_axis_combo):
                self.series_axis_combo.setCurrentIndex(max(0, self.series_axis_combo.findData("frame")))
        values = np.array(self._series_axis_values(series), dtype=float)
        values.setflags(write=False)  # (kept by the axis for every zoom and pan: ``frame_axis``)
        self._series_ticks = (values, not self._series_in_time(series))
        self._series_ticks_in_view()
        self.series_map_view.plot.setLabel("left", self._series_axis_label(series))

    def _series_ticks_in_view(self) -> None:
        """The ticks of the rows in view (``frame_ticks``): again at every zoom and pan."""
        ticks = getattr(self, "_series_ticks", None)
        if ticks is None or self._series_map is None:
            return
        values, whole = ticks
        shown = self.series_map_view.view_box.viewRange()[1]
        self.series_map_view.plot.getAxis("left").setTicks(frame_ticks(values, whole=whole, shown=shown))

    def _series_axis_chosen(self) -> None:
        series = self._series_map
        if series is None:
            return
        self._series_draw_frame_axis(series)
        self._series_describe()  # (where the time is from)
        self._series_redraw_trace()

    # -- the halves ------------------------------------------------------------------------------------

    def _series_view_tips(self) -> None:
        """What each of the halves is, at its item (in the interface language: the walker knows no item tips)."""
        for index, tip in enumerate(VIEW_TIPS[: self.series_view_combo.count()]):
            self.series_view_combo.setItemData(index, tr(tip), Qt.ToolTipRole)

    def _series_view_chosen(self, index: int) -> None:
        """Another half of the signed cut: the map stacked again from the rows kept (nothing is reduced)."""
        view = self.series_view_combo.itemData(index) or "both"
        if view == self._series_view or not self._series_rows:
            self._series_view = view
            return
        self._series_view = view
        self._clear_stages()
        self._show_series_map(keep_view=False, keep_q=False)
        if not self.batch_running():
            self._find_stages()

    # -- what the map is -------------------------------------------------------------------------------

    def _series_cut_matters(self, kind, key) -> bool:
        """The cut of the series changes this curve: every GISAXS cut, and I(χ) of the ring."""
        return kind == GISAXS or key == RING_CURVE

    def _series_describe(self) -> None:
        """The line above the map: curve, halves, frames and list, what a row sums, the cut of the series."""
        series = self._series_map
        if series is None:
            self.series_title_label.hide()
            self.series_stale_label.hide()
            return
        built = self._series_built
        parts = [self._series_curve_name(), *map_words(series, self._series_name or "", int(built.get("listed") or 0))]
        waiting = len(getattr(self, "_watch_series_jobs", ()) or ()) if self.view_model.watch.active else 0
        if waiting:  # (the map is behind the detector: said, not left to be guessed)
            parts.append(trf(MAP_WAITING, n=waiting))
        cut = cut_sentence(built.get("cut_shown"))
        if cut and self._series_cut_matters(built.get("kind"), self._series_key):
            parts.append(cut)
        if self._series_in_time(series) and series.time_from in MAP_TIME:
            parts.append(tr(MAP_TIME[series.time_from]))
        parts.append(self._series_angle_words())
        self.series_title_label.setText(" · ".join(part for part in parts if part))
        self.series_title_label.show()
        self._series_mark_stale()

    def _series_mark_stale(self) -> None:
        """Say when the map is no longer what Build Map would give: another curve chosen, other settings."""
        label = self.series_stale_label
        series = self._series_map
        if series is None or getattr(self, "_batch_live", False):
            label.hide()
            return
        notes = []
        if self.view_model.state.analysis is not None and self.view_model.series_settings_changed(self._series_built):
            notes.append(tr(STALE_SETTINGS))
        key = self.series_curve_combo.currentData()
        if key is not None and key != getattr(self, "_series_key", key):
            notes.append(trf(STALE_CURVE, curve=self.series_curve_combo.currentText()))
        label.setText(" ".join(notes))
        label.setVisible(bool(notes))

    # -- room ------------------------------------------------------------------------------------------

    def _series_wide_toggled(self, on: bool) -> None:
        """Wide: the image beside the tab shrinks to its least width; off: the split as it was before."""
        splitter, split = self.splitter, self.even_split
        sizes = splitter.sizes()
        if len(sizes) != 3:
            return
        if on:
            self._series_wide_before = (sizes, split.by_hand)
            image = splitter.widget(1)
            least = max(image.minimumSizeHint().width(), image.minimumWidth())
            split.by_hand = True  # (the even split would take the room back at the next resize)
            splitter.setSizes([sizes[0], least, max(sizes[2], sizes[1] + sizes[2] - least)])
        elif self._series_wide_before is not None:
            before, by_hand = self._series_wide_before
            self._series_wide_before = None
            split.by_hand = by_hand
            if by_hand and sum(before) == sum(sizes):
                splitter.setSizes(before)
            else:
                image = max(1, math.floor(0.5 * (sizes[1] + sizes[2])))
                splitter.setSizes([sizes[0], image, sizes[1] + sizes[2] - image])
                split.balance()


__all__ = ["MAJOR_TICKS", "MINOR_ROWS", "SeriesDisplayMixin", "frame_ticks", "nice_step", "row_files"]
