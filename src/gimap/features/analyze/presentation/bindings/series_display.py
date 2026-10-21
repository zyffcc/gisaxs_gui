"""The picture of the Series map and the line that says what it is.

* **Where a line is drawn.** The x of a curve is not evenly spaced (one point per valid detector column,
  merged bins at low q, holes at detector gaps), the columns of an image are: the map is drawn from
  ``SeriesMap.display()`` — its picture on an even grid, each column at its true x, gaps left blank. The
  values and the x of the map itself (bands, read-out, traces, exports) are untouched.
* **The frame axis** counts frames as people do — the frame in its file, or its place in the list, from 1 —
  with a tick in the middle of a row, never a fraction; or the time since the first row when the frames
  carry one (Options ▸ Frame axis).
* **The halves** of a signed cut (qy on both sides of the beam): as measured, one half, both on |qy|, or
  their mean. The map is stacked again from the rows kept, nothing is reduced again; the stages, the
  traces and the exports follow.
* **What the map is** stands above it: its curve, halves and frames, the cut of the series, and — when the
  Curve box or the settings no longer are what it was built with — that it must be built again.
* **Wide** gives the tab the room of the image beside it (and gives it back).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QSignalBlocker, Qt

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.stage_text import axis_symbol

from ...application import GISAXS, build_series_map
from ..series_texts import (
    AXIS_LABELS, HOW_TO, HOW_TO_SUMS, MAP_TIME, NO_TIME, STALE_CURVE, STALE_SETTINGS, cut_sentence, map_group,
    map_words, sum_note,
)
from ..views.series_view import VIEW_TIPS
from .series_stages import series_name

MAJOR_TICKS = 8
"""About this many numbered ticks on the frame axis; the other rows are numbered once there is room (zoom)."""
RING_CURVE = "azimuthal"
"""The GIWAXS curve the cut of a series is about (I(χ) of the ring); every GISAXS curve is a cut."""
HALVES_CURVE = "horizontal"
"""The curve that has halves to choose: the horizontal GISAXS cut, I(qy) on both sides of the beam."""
SUM_MARKS = ("sums", "Incomplete sums")
"""The kind of mark (key, name in the map's Marks menu) on a row that sums fewer frames than the others."""
SUM_PLATE = (0, 0, 0, 170)
"""RGBA of the plate under such a mark (white text: read over the bright and the dark end of every colour map)."""


def nice_step(span: float, most: int = MAJOR_TICKS) -> int:
    """1, 2, 5, 10, 20, 50 …: the smallest of them that leaves at most ``most`` ticks over ``span``."""
    step = 1
    for power in range(0, 9):
        for base in (1, 2, 5):
            step = base * 10**power
            if span / step <= most:
                return step
    return step


def frame_ticks(values, *, whole: bool = True) -> list[list[tuple[float, str]]]:
    """Ticks for the frame axis of a map of ``len(values)`` rows: each in the middle of its row, named by
    ``values`` (frame numbers, or seconds). Two levels: about ``MAJOR_TICKS`` named rows — round frame numbers
    where the numbers are whole —, and the rows between them, which are named when the axis has room."""
    values = [float(value) for value in values]
    rows = len(values)
    if not rows:
        return [[], []]
    if whole:
        names = [f"{int(round(value))}" for value in values]
    else:  # seconds: two digits of their usual step, the same decimals for every row (“0.0, 1.5, 3.0”)
        steps = sorted(abs(b - a) for a, b in zip(values, values[1:]) if b != a)
        usual = steps[len(steps) // 2] if steps else 1.0
        decimals = int(min(4, max(0, 1 - math.floor(math.log10(usual))))) if usual > 0 else 0
        names = [f"{value:.{decimals}f}" for value in values]
        while decimals > 0 and all(name.endswith("0") for name in names):  # (“0.50, 1.00” says no more than “0.5, 1.0”)
            decimals -= 1
            names = [f"{value:.{decimals}f}" for value in values]
    step = nice_step((values[-1] - values[0] + 1) if whole else rows)
    major = [row for row in range(rows) if (int(round(values[row])) % step == 0 if whole else row % step == 0)]
    if len(major) < min(rows, 2):  # numbers that never are round (every 3rd frame from 1): by row
        step = nice_step(rows)
        major = list(range(0, rows, step))
    minor = [row for row in range(rows) if row not in set(major)]
    return [[(row + 0.5, names[row]) for row in major], [(row + 0.5, names[row]) for row in minor]]


def row_files(rows) -> list[Path]:
    """The files of a map's rows, each once, in row order (a multi-frame file gives several rows)."""
    files: dict[str, Path] = {}
    for row in rows:
        files.setdefault(str(row[4][0]).casefold(), Path(row[4][0]))
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

    # -- the map ---------------------------------------------------------------------------------------

    def _show_series_map(self, *, keep_view: bool, keep_q: Optional[bool] = None) -> None:
        built = self._series_built
        name = series_name(row_files(self._series_rows))
        try:
            series = build_series_map(
                self._series_rows, self._series_key,
                view=self._series_view if self._series_key == HALVES_CURVE else "both",
                title=getattr(self, "_series_curve_title", ""), numbers=built.get("numbers"),
                group=int(built.get("group") or 1),
                built_with={
                    "name": name, "listed": built.get("listed"), "every": built.get("every"), "cut": built.get("cut"),
                    "cut_text": built.get("cut_text"), "settings": built.get("settings"),
                    "first_frame": getattr(self._series_run, "first_metadata", None),
                },
            )
        except ValueError as exc:
            self._series_say(str(exc))
            return
        self._series_map, self._series_name = series, name
        picture, left, width = series.display()
        self._series_extent = (left, left + width)
        self.series_map_view.set_image(
            picture, valid=np.isfinite(picture), rect=(left, 0.0, width, float(series.rows)),
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
        # (the map is the thing now: its how-to takes one line, see ``_series_say``); rows of sums are no frames
        group = map_group(series)
        values = dict(rows=series.rows, points=series.x.size, axis=axis_symbol(series.x_label))
        if group > 1:
            self._series_say(HOW_TO_SUMS, n=group, **values)
        else:
            self._series_say(HOW_TO, **values)
        self._series_describe()
        self._series_pick_row(min(self._series_row, series.rows - 1) + 0.5)
        if not (keep_view if keep_q is None else keep_q) or not hasattr(self, "_series_q"):  # where the series changes most
            centre = series.most_changing_x()
            self._series_pick_q(centre - self._series_step(), centre + self._series_step())
        else:
            self._series_pick_q(*self._series_q)

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
        for row, fact in enumerate(series.facts):
            if not fact.incomplete:
                continue
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
        axis = self.series_map_view.plot.getAxis("left")
        axis.setTicks(frame_ticks(self._series_axis_values(series), whole=not self._series_in_time(series)))
        self.series_map_view.plot.setLabel("left", self._series_axis_label(series))

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
        cut = cut_sentence(built.get("cut_shown"))
        if cut and self._series_cut_matters(built.get("kind"), self._series_key):
            parts.append(cut)
        if self._series_in_time(series) and series.time_from in MAP_TIME:
            parts.append(tr(MAP_TIME[series.time_from]))
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


__all__ = ["MAJOR_TICKS", "SeriesDisplayMixin", "frame_ticks", "nice_step", "row_files"]
