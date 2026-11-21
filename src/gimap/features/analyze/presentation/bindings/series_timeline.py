"""The timeline under the Series map: how far each frame lies from the one before it, or from a reference frame
(``application.timeline_of``), with the stage boundaries, the odd frames and the jumps marked — where the series
changes, at a glance. A click on the plot under the map (any of its traces) picks the nearest frame and shows it.

The reference frame is chosen in Options (its number on the frame axis, the first by default). The timeline
compares every x of the map, or — Options ▸ Stages compare — the x range the map shows, as the stages do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt

from src.gimap.app.presentation.i18n import tr, trf

from ...application import timeline_of

TIMELINE_KINDS = ("neighbour", "reference")
"""The kinds of the trace plot that are the timeline (``views/series_view.TRACE_ITEMS``)."""
TIMELINE_TITLES = {"neighbour": "χ² to the frame before", "reference": "χ² to frame {n}"}
TIMELINE_WORDS = {
    "neighbour": "chi2 of every frame to the frame before it (the noise of each x column from the series itself)",
    "reference": "chi2 of every frame to the reference frame {n} (the noise of each x column from the series itself)",
}
"""What the timeline is, for the table of a trace (English: a record)."""
TIMELINE_FEW = "The timeline needs a map of two frames or more"
TIMELINE_TIP = ("χ² of two frames: their difference in units of the noise of each x column (taken from the series), "
                "about 1 for frames that differ by noise alone. Dashed lines: stage boundaries; ▲ jumps; × odd frames. "
                "Click a point to show that frame.")
JUMP_COLOR = "#f59e0b"
ODD_COLOR = "#ef4444"
BOUNDARY_COLOR = "#64748b"


class SeriesTimelineMixin:
    """Needs the Series widgets, ``_series_map``, ``_series_stages``, ``_series_axis_values``, ``open_series_row``."""

    def _connect_series_timeline(self) -> None:
        self._series_timeline_cache = None
        self.series_trace_plot.positionClicked.connect(self._series_trace_clicked)
        self.series_reference_spin.valueChanged.connect(lambda _value: self._series_timeline_changed())
        self.series_export_timeline_action.triggered.connect(lambda: self.export_series_timeline())

    # -- the values ------------------------------------------------------------------------------------

    def _series_reference_row(self, series) -> int:
        """The row of the reference frame: the frame number of Options on the map's frame axis (nearest)."""
        numbers = np.asarray(series.numbers, dtype=float)
        return int(np.argmin(np.abs(numbers - float(self.series_reference_spin.value()))))

    def _series_timeline(self):
        """The timeline of the map shown (kept until the map, the reference or the x range compared change)."""
        series = self._series_map
        if series is None or series.rows < 2:
            return None
        reference = self._series_reference_row(series)
        x_range = self._stages_wanted_range() if hasattr(self, "_stages_wanted_range") else None
        key = (id(series), series.rows, reference, x_range)
        cached = self._series_timeline_cache
        if cached is not None and cached[0] == key and cached[1] is series:
            return cached[2]
        try:
            timeline = timeline_of(series, reference, x_range)
        except ValueError:
            return None
        self._series_timeline_cache = (key, series, timeline)
        return timeline

    def _series_timeline_values(self, kind: str):
        """``(kind, values, words)`` of a timeline trace (the table of the trace), or ``None``."""
        timeline = self._series_timeline()
        if timeline is None:
            return None
        series = self._series_map
        values = timeline.neighbour if kind == "neighbour" else timeline.reference
        return kind, values, TIMELINE_WORDS[kind].format(n=series.number(timeline.reference_row))

    def _series_timeline_changed(self) -> None:
        if (self.series_trace_combo.currentData() or "") in TIMELINE_KINDS:
            self._series_redraw_trace()

    # -- the plot --------------------------------------------------------------------------------------

    def _draw_timeline_trace(self, kind: str) -> bool:
        """The plot under the map as the timeline (``True``: drawn, also when it only says why it is empty)."""
        import pyqtgraph as pg

        plot, series = self.series_trace_plot, self._series_map
        plot.set_empty_text(TIMELINE_FEW)
        timeline = self._series_timeline()
        frame_axis = self._series_axis_label(series)
        if timeline is None:
            plot.set_title(tr(TIMELINE_TITLES["neighbour"]))
            plot.set_curves([])
            return True
        values = timeline.neighbour if kind == "neighbour" else timeline.reference
        axis = self._series_axis_values(series)
        title = trf(TIMELINE_TITLES[kind], n=series.number(timeline.reference_row))
        plot.set_title(title)
        plot.title_label.setToolTip(tr(TIMELINE_TIP))
        plot.set_labels(frame_axis, "χ²")
        curves, colors, markers = [(title, axis, values)], ["#2563eb"], [False]
        stages = self._series_stages if getattr(self, "_series_stages", None) is not None else None
        stages = stages if stages is not None and stages.rows == series.rows else None
        jumps = list(timeline.jumps) if kind == "neighbour" else []
        if jumps:
            curves.append((tr("jumps"), axis[jumps], values[jumps]))
            colors.append(JUMP_COLOR)
            markers.append("t1")
        odd = sorted(stages.odd_rows) if stages is not None else []
        if odd:
            curves.append((tr("odd frames"), axis[odd], values[odd]))
            colors.append(ODD_COLOR)
            markers.append("x")
        self._series_whole_frames(axis)
        plot.set_curves(curves, colors, markers=markers)
        if stages is not None:  # a dashed line between the last frame of a stage and the first of the next
            for edge in stages.edges[1:-1]:
                if 0 < edge < series.rows:
                    line = pg.InfiniteLine(pos=0.5 * (axis[edge - 1] + axis[edge]), angle=90,
                                           pen=pg.mkPen(BOUNDARY_COLOR, width=1, style=Qt.DashLine))
                    plot.add_extra(line)
        return True

    def _series_trace_clicked(self, x: float, _y: float) -> None:
        """A click on the plot under the map: the nearest frame (on the frame axis) is picked and shown."""
        series = self._series_map
        if series is None or self.batch_running():
            return
        axis = self._series_axis_values(series)
        if not len(axis) or not np.isfinite(x):
            return
        row = int(np.nanargmin(np.abs(axis - float(x))))
        self._series_pick_row(row + 0.5)
        self.open_series_row(row)

    # -- the table -------------------------------------------------------------------------------------

    def export_series_timeline(self, path: Optional[Path] = None) -> Optional[Path]:
        """Per frame its χ² to the frame before and to the reference frame, a jump or not, its stage, odd or not."""
        series, timeline = self._series_map, self._series_timeline()
        if series is None or timeline is None:
            self._status(tr(TIMELINE_FEW), "warning")
            return None
        path = path or self._series_path("Export the Timeline", "timeline.csv", "CSV table (*.csv)")
        stages = getattr(self, "_series_stages", None)
        stages = stages if stages is not None and stages.rows == series.rows else None
        return self._series_saved(
            lambda target: self.view_model.export_series_timeline(series, timeline, target, stages=stages), path,
            "Could not export the timeline: {error}")


__all__ = ["SeriesTimelineMixin", "TIMELINE_FEW", "TIMELINE_KINDS", "TIMELINE_TIP", "TIMELINE_TITLES", "TIMELINE_WORDS"]
