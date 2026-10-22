"""Stages and odd frames of the Series map (``application/series_stages.py``), and Send to Compare.

Found in the background once a map is complete (a Build Map or a Batch Export that filled it): a colour
strip with dashed boundaries and red arrows at odd frames on the map (“Stages” in its Marks menu), the
count (Auto or chosen) and the frames of every stage under the controls, and — folded — what changes
between stages, the typical frame of each, and the odd frames with the reason. The odd frames can be left
out of a Batch Export.

Frames are named by their number on the map's frame axis (the frame in its file, or its place in the list),
not by the row. A map of fewer than ``STAGES_LEAST_FRAMES`` frames has no stages, and says so. The stages
compare every x of the map, or — Options ▸ Stages compare — only the x range the map shows, and then follow
its zoom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer

from src.gimap.app.presentation.components import RowGroups, stage_color
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.stage_text import axis_symbol, change_text, odd_reason, stages_summary

from ...application import change_curve, odd_frame_refs, stages_of
from ...application.series_stages import STAGES_LEAST_FRAMES
from ..series_texts import AXIS_LABELS, NO_STAGES, STAGES_NEED, STAGES_NEED_EXPORT
from ..excluded_frames import ExcludedFrames
from ..views.series_view import CHANGE_TITLE

ZOOM_SETTLES_MS = 700
"""The stages of “the x range shown” are found again this long after the map's view last moved."""


def frame_number(series, row: int) -> int:
    """The frame of a map row as its frame axis counts it (the row from 1 for a map that has no numbers)."""
    number = getattr(series, "number", None)
    return int(number(row)) if callable(number) else int(row) + 1


class SeriesStagesMixin:
    """Needs the Series widgets (``views/series_view.py``), ``_series_map``, ``tasks``, ``_status``."""

    def _connect_stages(self) -> None:
        self._series_stages = None
        self._send_to_compare: Optional[Callable[[object, str], None]] = None
        self._stage_overlay = RowGroups(self.series_map_view)
        self._stages_x_range = None
        """The x range the stages shown were found in (``None``: every x of the map)."""
        self._stages_failure: Optional[str] = None
        """Why the last search found no stages (English, as the application says it), while that holds."""
        self._stages_zoom_timer = QTimer(self)
        self._stages_zoom_timer.setSingleShot(True)
        self._stages_zoom_timer.setInterval(ZOOM_SETTLES_MS)
        self._stages_zoom_timer.timeout.connect(self._stages_follow_zoom)
        self.series_map_view.view_box.sigXRangeChanged.connect(lambda *_args: self._stages_view_moved())
        self.series_stages_range_combo.activated.connect(lambda _index: self._find_stages())
        self.series_stages_combo.activated.connect(self._stages_count_chosen)
        self.series_export_stages_action.triggered.connect(lambda: self.export_series_stages())
        self.series_compare_button.clicked.connect(self.send_series_to_compare)

    def set_compare_target(self, send: Optional[Callable[[object, str], None]]) -> None:
        """``send(series_map, name)``: where Send to Compare puts the map (set by the application)."""
        self._send_to_compare = send

    # -- finding ---------------------------------------------------------------------------

    def _clear_stages(self) -> None:
        self._series_stages = None
        self._stages_failure = None
        self._stage_overlay.clear()
        self.series_stages_row.hide()
        self.series_stages_details.hide()
        self.series_compare_button.hide()
        self._change_item_tip()

    def _stages_wanted_range(self) -> Optional[tuple[float, float]]:
        """The x range the stages are to compare: what the map shows (Options ▸ Stages compare), or all of it."""
        if self.series_stages_range_combo.currentData() != "shown":
            return None
        low, high = self.series_map_view.view_box.viewRange()[0]
        return (float(low), float(high))

    def _stages_view_moved(self) -> None:
        if self.series_stages_range_combo.currentData() == "shown" and self._series_map is not None:
            self._stages_zoom_timer.start()  # once the zoom has come to rest

    def _stages_follow_zoom(self) -> None:
        """The map shows another x range: the stages of “the x range shown” are found in it."""
        wanted = self._stages_wanted_range()
        if wanted is None or self._series_map is None or self.batch_running():
            return
        before = self._stages_x_range
        if before is None or not np.allclose(before, wanted, rtol=1e-3, atol=0.0):
            self._find_stages()

    def _find_stages(self) -> None:
        series = self._series_map
        self.series_compare_button.setVisible(series is not None and self._send_to_compare is not None)
        self.series_export_stages_action.setEnabled(False)  # until there are stages to write
        if series is None or series.rows < 2:
            return
        self.series_stages_row.show()
        if series.rows < STAGES_LEAST_FRAMES:  # too short for stages: said, with no empty box beside it
            self._series_stages = None
            self._stage_overlay.clear()
            self.series_stages_combo.hide()
            self.series_stages_details.hide()
            self.series_stages_label.setText(trf(STAGES_NEED, n=STAGES_LEAST_FRAMES, rows=series.rows))
            return
        self.series_stages_combo.show()
        self._stages_failure = None
        self.series_stages_label.setText(tr("Looking for stages and odd frames …"))
        x_range = self._stages_wanted_range()
        self.tasks.submit(
            "series-stages", lambda: stages_of(series, x_range),
            on_done=lambda stages: self._stages_found(series, stages, x_range),
            on_error=lambda message, _trace: self._stages_failed(series, message, x_range),
        )

    def _stages_failed(self, series, message: str, x_range=None) -> None:
        """No stages in this map (or in the x range it shows): nothing of stages found before stays — not on
        the map, not in the details, not after a switch of the language (``_stages_language``)."""
        if series is not self._series_map:
            return
        self._clear_stages()
        self._stages_failure, self._stages_x_range = message, x_range
        self.series_stages_row.show()
        self.series_compare_button.setVisible(self._send_to_compare is not None)
        self.series_stages_label.setText(trf(NO_STAGES, reason=tr(message)))
        self.series_stages_label.setToolTip("")
        self.series_changes_label.setText("")
        if (self.series_trace_combo.currentData() or "") == "change":
            self._series_redraw_trace()

    def _stages_found(self, series, stages, x_range=None) -> None:
        if series is not self._series_map:  # a newer map replaced it meanwhile
            return
        self._series_stages, self._stages_failure = stages, None
        self._stages_x_range = x_range
        self.series_export_stages_action.setEnabled(True)
        self.series_stages_combo.blockSignals(True)
        self.series_stages_combo.clear()
        self.series_stages_combo.addItem(tr("Auto ({count})").format(count=stages.suggested), None)
        for count in sorted(stages.boundaries):
            self.series_stages_combo.addItem(str(count), count)
        self.series_stages_combo.blockSignals(False)
        self._show_stages()

    def _stages_count_chosen(self, index: int) -> None:
        stages = self._series_stages
        if stages is None:
            return
        count = self.series_stages_combo.itemData(index)
        self._series_stages = stages.with_count(stages.suggested if count is None else int(count))
        self._show_stages()

    # -- showing ---------------------------------------------------------------------------

    def _show_stages(self) -> None:
        series, stages = self._series_map, self._series_stages
        if series is None or stages is None:
            return
        ranges = stages.ranges()
        typical = stages.stage_representatives()
        tips = [tr("Stage {n}: frames {a}–{b}; typical frame {t}").format(
                    n=i + 1, a=frame_number(series, a), b=frame_number(series, b), t=frame_number(series, typical[i]))
                for i, (a, b) in enumerate(ranges)]
        # Over the picture of the map (its even grid begins and ends half a step beyond the first and last x).
        extent = getattr(self, "_series_extent", None) or (float(series.x[0]), float(series.x[-1]))
        self._stage_overlay.show(stages.edges, extent, odd_rows=[frame.row for frame in stages.odd], tips=tips)
        self.series_stages_label.setText(stages_summary(stages, lambda row: str(frame_number(series, row))))
        self.series_stages_label.setToolTip("\n".join(tips))
        self.series_changes_label.setText(self._stages_text(series, stages))
        self.series_skip_odd_check.setVisible(bool(stages.odd))
        self.series_stages_row.show()
        self.series_stages_details.show()
        self._change_item_tip()
        if (self.series_trace_combo.currentData() or "") == "change":
            self._series_redraw_trace()

    def _stages_language(self) -> None:
        """The stages' sentences again in the interface language (after a language switch)."""
        stages, series = self._series_stages, self._series_map
        if series is not None and series.rows < STAGES_LEAST_FRAMES and not self.series_stages_row.isHidden():
            self.series_stages_label.setText(trf(STAGES_NEED, n=STAGES_LEAST_FRAMES, rows=series.rows))
        failure = getattr(self, "_stages_failure", None)
        if stages is None and series is not None and failure:  # what failed, in the new language
            self.series_stages_label.setText(trf(NO_STAGES, reason=tr(failure)))
        if stages is None or series is None:
            return
        if self.series_stages_combo.count() and self.series_stages_combo.itemData(0) is None:
            self.series_stages_combo.setItemText(0, trf("Auto ({count})", count=stages.suggested))
        self._show_stages()

    def _stages_text(self, series, stages) -> str:
        """The stages, what changes between them and the odd frames, on the map's own axis (q, χ …)."""
        axis = axis_symbol(series.x_label)
        lines = []
        typical = stages.stage_representatives()
        for index, (first, last) in enumerate(stages.ranges()):
            lines.append(tr("Stage {n}: frames {a}–{b} (typical: frame {t})").format(
                n=index + 1, a=frame_number(series, first), b=frame_number(series, last),
                t=frame_number(series, typical[index])))
        lines += [change_text(change, axis) for change in stages.stage_changes()]
        if stages.half_row is not None and stages.count > 1:
            lines.append(tr("Half of the change by frame {half}, 90 % by frame {ninety}.").format(
                half=frame_number(series, stages.half_row),
                ninety="—" if stages.ninety_row is None else frame_number(series, stages.ninety_row)))
        span = f"{stages.q.min():.4g}–{stages.q.max():.4g}"
        lines.append(tr("Compared: the shape of log I over {span} ({points} points); the overall level separately.").format(
            span=span, points=stages.q.size))
        for frame in stages.odd:
            lines.append(tr("Odd frame {n} ({label}): {why}").format(
                n=frame_number(series, frame.row), label=series.labels[frame.row], why=odd_reason(frame, axis)))
        if any(frame.narrow for frame in stages.odd):
            lines.append(trf("A difference near one {axis} in a few points is the detector: mask it in the Mask step.",
                             axis=axis))
        return "\n".join(lines)

    def _draw_change_trace(self) -> bool:
        """The lower-right plot as “Change along the series”: the first component, one colour per stage."""
        stages, series = self._series_stages, self._series_map
        # The frames as the map's frame axis counts them (numbers, or seconds).
        timed = series is not None and stages is not None and series.rows == stages.rows
        frame_axis = self._series_axis_label(series) if timed else AXIS_LABELS["frame"]
        if stages is None:  # the plot's empty text says when it fills (``views/series_view.py``)
            self.series_trace_plot.set_title(CHANGE_TITLE)
            self.series_trace_plot.set_labels(frame_axis, "component 1")
            self.series_trace_plot.set_curves([])
            return True
        values = change_curve(stages)
        frames = self._series_axis_values(series) if timed else np.arange(1, stages.rows + 1, dtype=float)
        odd = stages.odd_rows
        curves, colors, markers = [], [], []
        for index, (first, last) in enumerate(stages.ranges(), start=1):
            rows = [row for row in range(first, last + 1) if row not in odd]
            curves.append((tr("stage {n}").format(n=index), frames[rows], values[rows]))
            colors.append(stage_color(index))
            markers.append(False)
        if odd:
            rows = sorted(odd)
            curves.append((tr("odd frames"), frames[rows], values[rows]))
            colors.append("#ef4444")
            markers.append("x")
        # A short title (the header is shared with the trace combo and the plot's buttons); the share of the
        # change in the y label, and as a sentence in the title's tooltip and the combo item's.
        share = change_share(stages)
        self.series_trace_plot.set_title(CHANGE_TITLE)
        self.series_trace_plot.title_label.setToolTip(trf("Main component: {share:.0f} % of the change", share=share))
        self.series_trace_plot.set_labels(frame_axis, f"component 1 ({share:.0f} %)")
        if timed:
            self._series_whole_frames(frames)
        self.series_trace_plot.set_curves(curves, colors, markers=markers)
        return True

    def _change_item_tip(self) -> None:
        """The “Change along the series” item of the trace combo says how much of the change it shows."""
        index = self.series_trace_combo.findData("change")
        if index < 0:
            return
        stages = self._series_stages
        tip = trf("Main component: {share:.0f} % of the change", share=change_share(stages)) if stages is not None else None
        self.series_trace_combo.setItemData(index, tip, Qt.ToolTipRole)

    # -- export, Batch Export, Compare -------------------------------------------------------

    def export_series_stages(self, path: Optional[Path] = None) -> Optional[Path]:
        series, stages = self._series_map, self._series_stages
        if series is not None and series.rows < STAGES_LEAST_FRAMES:  # a short map never has stages
            self._status(trf(STAGES_NEED_EXPORT, n=STAGES_LEAST_FRAMES, rows=series.rows), "warning")
            return None
        if series is not None and stages is None and self._stages_failure:  # why there are none
            self._status(trf(NO_STAGES, reason=tr(self._stages_failure)), "warning")
            return None
        if series is None or stages is None:
            self._status(tr("The stages are found once a map is built."), "warning")
            return None
        path = path or self._series_path("Export Stages", "stages.csv", "CSV (*.csv)")  # the title is translated there
        if path is None:
            return None
        try:
            written = self.view_model.export_series_stages(series, stages, Path(path))
        except (ValueError, OSError) as exc:
            self._status(tr("Could not export the stages: {error}").format(error=exc), "error")
            return None
        self.notify_written(tr("Saved {name} and its record").format(name=written.name), written.parent)
        return written

    def _without_odd_frames(self, requests: list) -> list:
        """Batch Export: the odd frames of the map left out when that is chosen."""
        if not self.series_skip_odd_check.isChecked() or self._series_map is None:
            return requests
        odd = odd_frame_refs(self._series_map, self._series_stages)
        if not odd:
            return requests
        indices = getattr(requests, "indices", None)
        if indices is not None:
            excluded = []
            for path, frame_index in odd:
                group = self.view_model.batch_group_index(path, frame_index)
                if group is None or group not in indices:
                    continue
                index = indices.index(group)
                request = requests[index]
                if (str(request.path).casefold(), int(request.frame_index)) in odd:
                    excluded.append(index)
            kept = ExcludedFrames(requests, excluded)
        else:
            kept = [request for request in requests
                    if (str(request.path).casefold(), int(request.frame_index)) not in odd]
        if len(kept) < len(requests):
            left_out = len(requests) - len(kept)
            self._status(trf("1 odd frame left out of the Batch Export (Series ▸ Stages)." if left_out == 1
                             else "{n} odd frames left out of the Batch Export (Series ▸ Stages).", n=left_out))
        return kept

    def current_series(self):
        """``(series_map, name)`` of the map shown, or ``None`` (Compare ▸ Add ▸ The Series Map of Analyze)."""
        series = self._series_map
        if series is None or series.rows < 2:
            return None
        return series, self._map_name(series)

    def send_series_to_compare(self) -> bool:
        series = self._series_map
        if series is None or self._send_to_compare is None:
            return False
        self._send_to_compare(series, self._map_name(series))
        return True

    def _map_name(self, series) -> str:
        """The name kept with the map when it was built (its own frames, not the files listed now)."""
        name = getattr(self, "_series_name", None)
        if name:
            return name
        refs = getattr(series, "refs", None) or ()
        return series_name(list({str(path).casefold(): Path(path) for path, _frame in refs}.values())
                           or [Path(item) for item in self.view_model.state.files])


def change_share(stages) -> float:
    """The share (%) of the change along the series that its main component carries."""
    return 100.0 * float(stages.explained[0]) if stages.explained.size else 0.0


def series_name(files: list) -> str:
    """A short name for the series: what the file names share (or the one file's stem), without
    frame numbers and NeXus module suffixes."""
    import os
    import re

    stems = [path.stem for path in files] or ["series"]
    common = os.path.commonprefix(stems) if len(stems) > 1 else stems[0]
    name = re.sub(r"(_m\d{2})$", "", common)
    name = re.sub(r"[_\-.]*\d*$", "", name) if len(stems) > 1 else re.sub(r"_\d{5}$", "", name)
    return name.strip("_-. ") or (files[0].parent.name if files else "series")


__all__ = ["SeriesStagesMixin", "ZOOM_SETTLES_MS", "change_share", "frame_number", "series_name"]
