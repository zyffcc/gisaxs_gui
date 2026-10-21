"""Picking in the Series map: a frame (the horizontal band) and an x window (the vertical band).

The frame picked is drawn in the lower-left plot; the lower-right plot shows, for every frame, the mean
intensity in the window, a property of the peak found there, or the change along the series. Both are
named in the frame numbers of the map's frame axis (or its seconds), and a row that sums fewer frames
than the others — the last group of a Sum N — is drawn apart from the line, so a rising series does not
seem to fall at its end. With “Show the picked frame's image” a picked row is also shown in Analyze.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import QTimer

from src.gimap.app.presentation.i18n import trf

from ...application import track_peak
from ...application.series_stages import STAGES_LEAST_FRAMES
from ..series_texts import (
    FRAME_NAME, PEAK_LABELS, PEAK_NAMES, READOUT, TRACE_INTENSITY, TRACE_NEAREST, TRACE_NO_POINT, TRACE_PEAK,
    TRACE_WORDS, frame_title, row_fact, sum_note,
)
from ..views.series_view import CHANGE_EMPTY, CHANGE_FEW
from .series_display import nice_step

FOLLOW_MS = 300
"""A picked row is shown in Analyze this long after the last pick (dragging the band reads no frame)."""


def unit_of(x_label: str) -> str:
    return x_label[x_label.find("("):] if "(" in x_label else ""


class SeriesPickMixin:
    """Needs the Series widgets, ``_series_map``, ``_series_stages`` and ``open_series_row``."""

    def _connect_series_pick(self) -> None:
        self._series_follow_timer = QTimer(self)
        self._series_follow_timer.setSingleShot(True)
        self._series_follow_timer.setInterval(FOLLOW_MS)
        self._series_follow_timer.timeout.connect(self._series_follow_row)
        self.series_follow_check.toggled.connect(lambda on: self._series_follow_timer.start() if on else None)
        view = self.series_map_view
        view.horizontalBandChanged.connect(
            lambda low, high: self._series_pick_row(0.5 * (low + high), move_band=False, by_hand=True))
        view.verticalBandChanged.connect(lambda low, high: self._series_pick_q(low, high, move_band=False))
        view.positionClicked.connect(self._series_clicked)
        view.set_readout(self._series_readout)
        self.series_trace_combo.currentIndexChanged.connect(lambda _index: self._series_redraw_trace())

    # -- a frame ---------------------------------------------------------------------------------------

    def _series_pick_row(self, y: float, *, move_band: bool = True, by_hand: bool = False) -> None:
        series = self._series_map
        if series is None:
            return
        row = int(np.clip(np.floor(y), 0, series.rows - 1))
        self._series_row = row
        if move_band:
            self.series_map_view.show_horizontal_band(row, row + 1)
        self.series_profile_plot.set_title(frame_title(series, row))
        self.series_profile_plot.title_label.setToolTip(str(series.labels[row]))
        self.series_profile_plot.set_labels(series.x_label, "Intensity")
        self.series_profile_plot.set_curves([(trf(FRAME_NAME, n=series.number(row)), series.x, series.profile(row))])
        if by_hand and self.series_follow_check.isChecked():
            self._series_follow_timer.start()

    def _series_clicked(self, x: float, y: float) -> None:
        self._series_pick_row(y, by_hand=True)

    def _series_follow_row(self) -> None:
        """Show the picked row's frame in Analyze (not while a batch reduces the frames)."""
        if self.series_follow_check.isChecked() and self._series_map is not None and not self.batch_running():
            self.open_series_row()

    def _series_readout(self, x: float, y: float) -> str:
        series = self._series_map
        if series is None or not 0 <= y < series.rows:
            return ""
        row = int(y)
        column = int(np.argmin(np.abs(series.x - x)))
        value = series.image[row, column]
        text = "—" if not np.isfinite(value) else f"{value:.4g}"
        return trf(READOUT, n=series.number(row), label=series.labels[row]) + (
            f" · {series.x_label.split(' (')[0]} = {series.x[column]:.4g} · I = {text}")

    # -- an x window, and its trace -------------------------------------------------------------------

    def _series_pick_q(self, low: float, high: float, *, move_band: bool = True) -> None:
        series = self._series_map
        if series is None:
            return
        low, high = sorted((float(low), float(high)))
        self._series_q = (low, high)
        if move_band:
            self.series_map_view.show_vertical_band(low, high)
        self._series_redraw_trace()

    def _series_window(self) -> tuple[np.ndarray, str]:
        """``SeriesMap.trace_columns`` of the x window: ``none`` when nothing was measured in it."""
        low, high = self._series_q
        return self._series_map.trace_columns(0.5 * (low + high), 0.5 * (high - low))

    def _series_trace_values(self) -> Optional[tuple[str, np.ndarray, str]]:
        """``(kind, value of every row, what it is in English)`` of the trace chosen, or ``None`` when there is
        none (no map, no window, the change before the stages are found)."""
        series = self._series_map
        if series is None or not hasattr(self, "_series_q"):
            return None
        low, high = self._series_q
        kind = self.series_trace_combo.currentData() or "intensity"
        words = TRACE_WORDS[kind].format(low=f"{low:.6g}", high=f"{high:.6g}", unit=unit_of(series.x_label))
        if kind == "change":
            stages = self._series_stages
            return None if stages is None else (kind, np.asarray(stages.scores[:, 0], dtype=float), words)
        if kind == "intensity":
            columns, how = self._series_window()
            if how == "nearest":
                words = TRACE_WORDS["nearest"].format(low=f"{low:.6g}", high=f"{high:.6g}", unit=unit_of(series.x_label),
                                                      x=f"{float(series.x[columns][0]):.6g}")
            return kind, series.trace(0.5 * (low + high), 0.5 * (high - low)), words
        track = track_peak(series, low, high)
        self._series_track = track
        return kind, dict(track.table())[kind], words

    def _series_redraw_trace(self) -> None:
        """The lower-right plot: mean intensity in the x window, or a property of its peak, against frame."""
        series = self._series_map
        if series is None or not hasattr(self, "_series_q"):
            return
        low, high = self._series_q
        centre, half = 0.5 * (low + high), 0.5 * (high - low)
        kind = self.series_trace_combo.currentData() or "intensity"
        few = series.rows < STAGES_LEAST_FRAMES  # no stages are looked for in so short a map
        columns, how = self._series_window()
        empty = TRACE_NO_POINT if kind != "change" and how == "none" else ""  # nothing measured there: no value
        self.series_trace_plot.set_empty_text(empty if kind != "change" else CHANGE_FEW if few else CHANGE_EMPTY)
        if kind == "change" and self._draw_change_trace():
            return
        found = self._series_trace_values()
        if found is None:
            return
        _kind, values, _words = found
        axis = self._series_axis_values(series)
        frame_axis = self._series_axis_label(series)
        name = series.x_label.split(" (")[0]
        if kind == "intensity" and how == "nearest":  # a window between two merged bins: the point nearest to it
            self.series_trace_plot.set_title(trf(TRACE_NEAREST, axis=name, x=f"{float(series.x[columns][0]):.4g}"))
            self.series_trace_plot.set_labels(frame_axis, "Intensity")
            legend = f"{float(series.x[columns][0]):.4g}"
        elif kind == "intensity":
            self.series_trace_plot.set_title(trf(TRACE_INTENSITY, axis=name, centre=f"{centre:.3g}", half=f"{half:.1g}"))
            self.series_trace_plot.set_labels(frame_axis, "Intensity")
            legend = f"{centre:.4g}"
        else:
            self.series_trace_plot.set_title(trf(TRACE_PEAK, low=f"{low:.3g}", high=f"{high:.3g}"))
            self.series_trace_plot.set_labels(frame_axis, PEAK_LABELS[kind].format(unit=unit_of(series.x_label)).strip())
            legend = PEAK_NAMES[kind]
        if how == "none":  # nothing measured in the window: no curve (nor a legend of one), the sentence says why
            self.series_trace_plot.set_curves([])
            return
        # A row that sums fewer frames than the others is no point of the line: drawn apart, and named.
        short = np.array([fact.incomplete for fact in series.facts], dtype=bool) if len(series.facts) == series.rows else (
            np.zeros(series.rows, dtype=bool))
        # (one point alone draws no line: a dot)
        curves, markers = [(legend, axis[~short], np.asarray(values)[~short])], [True if (~short).sum() == 1 else False]
        if short.any():
            last = int(np.flatnonzero(short)[-1])
            curves.append((sum_note(row_fact(series, last)), axis[short], np.asarray(values)[short]))
            markers.append("x")
        self._series_whole_frames(axis)
        self.series_trace_plot.set_curves(curves, markers=markers)

    def _series_whole_frames(self, axis) -> None:
        """The trace's frame axis has ticks at whole frames only (never “frame 1.5”); seconds: as they fall."""
        bottom = self.series_trace_plot.plot.getAxis("bottom")
        series = self._series_map
        if series is None or self._series_in_time(series) or not len(axis):
            bottom.setTickSpacing()
            return
        major = nice_step(float(axis[-1] - axis[0]) + 1.0)
        bottom.setTickSpacing(major=major, minor=max(1, major // 5))


__all__ = ["FOLLOW_MS", "SeriesPickMixin", "unit_of"]
