"""Stages and odd frames of the In-situ series (``application/series_fit.stages_of_curves``).

Once the curves are listed, all of them are read in the background and compared as they will be fitted
(the halves of the series, the range and left-out points of Single analysis): the Curves step says how many
stages and odd frames there are, the frame list shows each frame's stage (a small square in its colour,
hollow for an odd frame; the names keep the list's colour, a frame that did not converge or failed is in the
warning or danger colour), the odd frames can be left out of the run, a new stage can restart from the
Single model, and the trend is drawn in the colours of the stages (“stage n” in a legend at the bottom right).

A curve that cannot be used — a file that cannot be read, a frame without the chosen half — does not end
the search: the stages are found on the others, and that frame says why in the list before the run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap

from src.gimap.app.presentation.components import stage_color
from src.gimap.app.presentation.components.curve_plot import Marker
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.stage_text import change_text
from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.app.presentation.theme import theme_manager

from ...application import LoadCurveRequest
from ...application.series_fit import SeriesSettings, stages_of_curves, unusable_curves
from ...application.single_fit import Curve
from ..views.fit_series_view import TREND_EMPTY
from .series_text import STATE_COLOR
from .session import path_name

CHI2 = "chi2"
ICON_SIZE = 10
TREND_MARKER = Marker("o", 6.0)
"""A frame's value in the trend: large enough to see on either theme."""
LOOKING = "Looking for stages and odd frames …"
NOT_FOR_THIS_RUN = "No stages for this run: it started before they were found."
NO_STAGES_AT_START = "No stages are known: every frame starts from the previous frame's result."
"""Said at Start when “afresh at each stage” is chosen and no stages are there to go by (the search still
reads the curves, or found none): the run does not wait for them."""


def stage_icon(stage: int, odd: bool = False) -> QIcon:
    """A small square in the colour of ``stage`` (the shared stage palette); hollow for an odd frame."""
    pixmap = QPixmap(ICON_SIZE, ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    color = QColor(stage_color(stage))
    painter.setPen(QPen(color, 1.6))
    painter.setBrush(Qt.NoBrush if odd else QBrush(color))
    painter.drawRect(QRectF(0.8, 0.8, ICON_SIZE - 1.6, ICON_SIZE - 1.6))
    painter.end()
    icon = QIcon()
    for mode in (QIcon.Normal, QIcon.Selected, QIcon.Active):  # the stage's colour also on a selected row
        icon.addPixmap(pixmap, mode)
    return icon


def _failure_text(message: str) -> str:
    """Why no stages were found, in the interface language: “too few frames”, or “<file>: <why it is no curve>”
    (the file name as it is)."""
    name, separator, reason = message.partition(": ")
    if separator and tr(reason) != reason:
        return f"{name}: {tr(reason)}"
    return tr(message)


class FitSeriesStagesMixin:
    """Needs the series page: ``paths``, ``_listed()``, ``single``, ``view_model``, ``tasks``, the widgets."""

    def _connect_stages(self) -> None:
        self._series_stages = None
        self._stage_rows: dict[int, int] = {}
        self._stages_key = None
        self._stages_failure: Optional[str] = None
        """Why the last search found no stages (shown in the interface language)."""
        self._stages_wait = ""
        """What the line says while there is no result (``LOOKING``, ``NOT_FOR_THIS_RUN``; English)."""
        self._skipped: dict[int, str] = {}
        """The listed frames that cannot be used as the series stands, with the reason (English): the file
        cannot be read, no points on the chosen half, too few points in the range."""
        self._stages_timer = QTimer(self)
        self._stages_timer.setSingleShot(True)
        self._stages_timer.setInterval(400)
        self._stages_timer.timeout.connect(self._find_stages)
        self.stage_tasks = TaskRunner(self)  # apart from the fits: Pause and Stop look at those alone
        self.skip_odd_check.toggled.connect(lambda _on: self.refresh())
        theme_manager().changed.connect(self._restyle_stages)  # the names' colours are made for the theme

    def _dispose_stages(self) -> None:
        try:
            theme_manager().changed.disconnect(self._restyle_stages)
        except (TypeError, RuntimeError):  # not connected
            pass
        try:  # no search starts, and none ends, on a page that is gone
            self._stages_timer.stop()
            self.stage_tasks.shutdown(2000)
        except (AttributeError, RuntimeError):
            pass

    def _restyle_stages(self, *_args) -> None:
        """Another theme: the warning and danger colours of the frames' names again (the theme's own)."""
        try:
            self._mark_stages()
            self._rebuild_results()  # the rows of the table too
            self._draw_trend()  # the plot drew its own curves again and cleared the ±1σ bars added to them
        except RuntimeError:  # the page is already gone
            self._dispose_stages()

    def _stage_settings(self) -> SeriesSettings:
        """The points a frame would be fitted on now: the halves of the series (``series_halves.py``), the
        range and left-out points of Single analysis."""
        session = self.single.session
        return SeriesSettings(side=self.series_side(), q_range=session.q_range, excluded=frozenset(session.excluded))

    def _schedule_stages(self) -> None:
        """Find the stages again when the listed curves or the way they are fitted changed."""
        settings = self._stage_settings()
        key = (tuple(self.paths[index] for index in self._listed()), settings.side, settings.q_range, settings.excluded)
        if key != self._stages_key and not self.running:
            self._stages_key = key
            self._stages_timer.start()

    def _forget_stages(self) -> None:
        """No stages for what is listed now (another folder, a new search): no stage colours, “· odd” or
        Leave-out check left from before — nor odd frames of other curves left out at Start."""
        self._series_stages, self._stage_rows, self._stages_failure, self._skipped = None, {}, None, {}
        self.skip_odd_check.hide()
        self._say_stages("")  # nor the line about them
        self._mark_stages()

    def _say_stages(self, sentence: str) -> None:
        """The Curves step's line while there is no result of a search (English; said in the interface language)."""
        self._stages_wait = sentence
        self.stages_label.setText(tr(sentence) if sentence else "")
        self.stages_label.setToolTip("")

    def _search_after_run(self, say: bool = True) -> None:
        """A search that would begin or end while the series runs is put off: the run keeps the stages it
        started with — none, when it started before they were found — and they are looked for again when
        it has ended (``_schedule_stages``, from ``_frames_changed`` at the end of the run)."""
        self._stages_key = None
        if say and self._series_stages is None:
            self._stages_failure = None
            self._say_stages(NOT_FOR_THIS_RUN)

    def _find_stages(self) -> None:
        listed = self._listed()
        if self.running:
            self._search_after_run(len(listed) >= 3)
            return
        self._forget_stages()
        if len(listed) < 3:
            return
        paths = [self.paths[index] for index in listed]
        settings = self._stage_settings()
        unit = self._preview_unit()  # of files whose header states none
        load = self.load_series_curve  # (a detector frame is cut as in Analyze)

        def work():
            curves, unreadable, read = [], {}, []
            for row, path in enumerate(paths):
                outcome = load(LoadCurveRequest(Path(path), unit))
                if outcome.error is not None:  # this file is left out; the search goes on
                    curves.append(None)
                    unreadable[row] = outcome.error.message
                    continue
                value = outcome.value
                curves.append(Curve.from_arrays(value.q, value.intensity, value.error, name=Path(path).name, path=path,
                                                unit=value.q_source_unit, unit_from=value.q_unit_from))
                read.append((listed[row], value.q, value.observation))
            try:
                return stages_of_curves(curves, settings, unreadable), "", curves, unreadable, read
            except ValueError as exc:  # too few frames, or no curve that can be used: still said of each curve
                return None, str(exc), curves, dict(unusable_curves(curves, settings, unreadable)), read

        self._say_stages(LOOKING)
        self.stage_tasks.submit("series-stages", work, on_done=lambda found: self._stages_found(listed, settings, *found),
                                on_error=lambda message, _trace: self._stages_failed(listed, message))

    def _stages_failed(self, listed: list, message: str) -> None:
        if self.running:
            self._search_after_run()
        if listed != self._listed() or self.running:
            return
        self._stages_failure = str(message)
        self._show_stages_text(listed)

    def _stages_found(self, listed: list, settings, stages, failure: str = "", curves=(), unreadable=None, read=()) -> None:
        keep = getattr(self, "_keep_preview_curves", None)
        if keep is not None:
            keep(curves)  # read once: selecting a frame draws it without reading it again
        if self.running:  # Start was pressed while the curves were read
            self._search_after_run()
        if listed != self._listed() or self.running or not self._listed_curves(listed, curves):
            return
        for index, q, observation in read:  # whether these curves have two halves, and what they were sent with
            self._note_curve(index, q, observation)
        if settings != self._stage_settings():  # now known to have no halves (or to have some): found again
            self.refresh()
            return
        skipped = dict(stages.skipped) if stages is not None else dict(unreadable or {})
        self._skipped = {listed[row]: reason for row, reason in skipped.items()}
        self._series_stages, self._stages_failure = stages, failure or None
        self._stage_rows = {index: row for row, index in enumerate(listed)} if stages is not None else {}
        self._show_stages_text(listed)
        self.skip_odd_check.setVisible(stages is not None and bool(stages.odd))
        self._frames_changed()
        if self.fits:
            self._draw_trend()  # in the colours of the stages (a run that started before they were found)

    def _listed_curves(self, listed: list, curves) -> bool:
        """Whether ``curves`` are the files listed now — not those of a folder listed before that holds as
        many (its search may end after the next folder is listed)."""
        return all(curve is None or not curve.path or curve.path == self.paths[listed[row]]
                   for row, curve in enumerate(curves))

    def _show_stages_text(self, listed: list) -> None:
        """The Curves step's line about the stages — or why there are none — the frames that cannot be used,
        and the Leave-out check, in the interface language."""
        stages = self._series_stages
        if stages is None:
            text = tr("No stages: {reason}").format(reason=_failure_text(self._stages_failure or ""))
            self.stages_label.setToolTip("")
        else:
            ranges = ", ".join(f"{listed[first] + 1}–{listed[last] + 1}" for first, last in stages.ranges())
            text = (tr("{count} stages: frames {ranges}").format(count=stages.count, ranges=ranges) if stages.count > 1
                    else tr("One stage: no change of course"))
            if stages.odd:
                text += " · " + tr("odd frames: {frames}").format(frames=", ".join(str(listed[f.row] + 1) for f in stages.odd))
            self.stages_label.setToolTip("\n".join(change_text(change) for change in stages.stage_changes()))
            self.skip_odd_check.setText(tr("Leave out the odd frames ({count})").format(count=len(stages.odd)))
        if self._skipped:
            text += " · " + trf("cannot be used: frames {frames}", frames=", ".join(str(index + 1) for index in sorted(self._skipped)))
        self._stages_wait = ""
        self.stages_label.setText(text)

    def _refresh_stages_language(self) -> None:
        """After a switch of the interface language: the stages line, the names and the trend's legend again."""
        if self._series_stages is not None and self._stage_rows:
            self._show_stages_text(sorted(self._stage_rows, key=self._stage_rows.get))
        elif self._stages_failure is not None:
            self._show_stages_text(self._listed())
        elif self._stages_wait:
            self.stages_label.setText(tr(self._stages_wait))
        self._mark_stages()
        self._draw_trend()

    def _listing_record(self, fitted: list) -> dict:
        """Which curves of the folder a run fits, for the series record: the frames chosen, the odd frames
        left out and the stages (frame numbers), and whether new curves are watched for."""
        stages, kept = self._series_stages, set(fitted)
        searched = sorted(self._stage_rows, key=self._stage_rows.get)
        return {
            "first": self.first_spin.value(), "last": self.last_spin.value(), "every": self.every_spin.value(),
            "subfolders": self.subfolders_check.isChecked(), "curves_in_folder": len(self.paths), "frames_to_fit": len(fitted),
            "odd_frames_left_out": [index + 1 for index in self._listed() if index not in kept],
            "stages": None if stages is None or not searched else
            [[searched[first] + 1, searched[last] + 1] for first, last in stages.ranges()],
            "watch": self.watch_check.isChecked(),
            **self.frames_record(),
        }

    def _start_choice(self) -> str:
        """``previous``, ``same`` or ``stages``: where each frame starts."""
        return "same" if self.start_same.isChecked() else "stages" if self.start_stages.isChecked() else "previous"

    def _set_start(self, choice) -> None:
        {"same": self.start_same, "stages": self.start_stages}.get(choice, self.start_previous).setChecked(True)

    # -- per frame -----------------------------------------------------------------------------

    def stage_of_frame(self, index: int) -> Optional[int]:
        row = self._stage_rows.get(index)
        return None if row is None or self._series_stages is None else self._series_stages.stage_of(row)

    def is_odd_frame(self, index: int) -> bool:
        row = self._stage_rows.get(index)
        return row is not None and self._series_stages is not None and row in self._series_stages.odd_rows

    def _frame_text(self, index: int) -> str:
        """A frame's line in the list: its state and file, and “· odd” for an odd frame."""
        text = self._item_text(index)
        return text + "  · " + tr("odd") if self.is_odd_frame(index) else text

    def _mark_stages(self) -> None:
        """Every listed frame's line: its stage as a small square, its state's colour, “· odd” for an odd
        frame. Can be called again (a new theme, another language): every line is made afresh."""
        for position, index in enumerate(self._listed()):
            item = self.frame_list.item(position)
            if item is not None:
                self._style_item(item, index)

    def _style_item(self, item, index: int) -> None:
        """One frame's line: the text, the stage square and tooltip, the colour of its state. A frame that is
        not a plain result says why on hover too (its line may be longer than the list is wide)."""
        stage = self.stage_of_frame(index)
        odd = self.is_odd_frame(index)
        state, detail = self._frame_state(index)
        why = detail if state in STATE_COLOR else ""
        if stage is None:  # no stages (yet): no square, no stage tooltip
            item.setIcon(QIcon())
            item.setToolTip(why)
        else:
            item.setIcon(stage_icon(stage, odd))
            tip = tr("Odd frame (left out of the stages)") if odd else tr("Stage {n}").format(n=stage)
            item.setToolTip(tip + ("\n" + why if why else ""))
        token = STATE_COLOR.get(state)
        if token is None:
            item.setData(Qt.ForegroundRole, None)  # the list's own colour (light and dark)
        else:
            item.setForeground(QBrush(theme_manager().color(token)))
        item.setText(self._frame_text(index))

    def _without_odd(self, listed: list) -> list:
        if not self.skip_odd_check.isChecked() or self._series_stages is None:
            return listed
        kept = [index for index in listed if not self.is_odd_frame(index)]
        if len(kept) < len(listed):
            left_out = len(listed) - len(kept)
            self._log(trf("1 odd frame left out." if left_out == 1 else "{n} odd frames left out.", n=left_out))
        return kept

    def _new_stage(self, index: int) -> bool:
        previous = self._previous
        if previous is None:
            return False
        return self.stage_of_frame(index) != self.stage_of_frame(previous.index)

    # -- the trend ------------------------------------------------------------------------------

    def _draw_trend(self) -> None:
        """A value (± 1σ) or χ²ᵣ against frame, one colour per stage."""
        import pyqtgraph as pg

        key = self.trend_combo.currentData()
        frames = [self.fits[index] for index in sorted(self.fits)
                  if self.fits[index].ok and not getattr(self.fits[index].result, "stopped", False)]  # results only
        self.trend_plot.set_title("")  # while it is empty, the plot says when it fills (``TREND_EMPTY``)
        if key is None or not frames or self._start_model is None:
            self.trend_plot.set_curves([])
            return
        groups: dict[Optional[int], list] = {}
        for frame in frames:
            groups.setdefault(self.stage_of_frame(frame.index), []).append(frame)
        name = tr("χ²ᵣ") if key == CHI2 else self.trend_combo.currentText()
        curves, colors, bars = [], [], []
        for stage, members in groups.items():
            x = np.array([frame.index + 1 for frame in members], float)
            if key == CHI2:
                y, error = np.array([frame.result.chi2_reduced for frame in members]), np.full(len(members), math.nan)
            else:
                y = np.array([frame.result.model.get(key).value for frame in members])
                error = np.array([frame.result.errors.get(key, math.nan) for frame in members])
            staged = stage is not None and len(groups) > 1
            color = stage_color(stage) if staged else "#2563eb"
            label = tr("stage {n}").format(n=stage) if staged else name  # the value: in the list and on the axis
            curves.append((label, x, y))
            colors.append(color)
            bars.append((x, y, error, color))
        # The stages' legend in the corner with the fewest points — the bottom right for a value that grows,
        # off the late frames; one value needs no legend (the list above the plot and the axis name it).
        self.trend_plot.legend.setOffset(self._legend_corner(curves) if len(groups) > 1 else (-10, 10))
        self.trend_plot.legend.setVisible(len(groups) > 1)
        self.trend_plot.set_labels("frame", "χ²ᵣ" if key == CHI2 else path_name(self._start_model, key, unit=True, translate=False))
        self.trend_plot.set_curves(curves, colors, markers=[TREND_MARKER] * len(curves))
        first, last = frames[0].index + 1, frames[-1].index + 1
        self.trend_plot.plot.setXRange(first - 0.5, last + 0.5, padding=0)  # no point on the edge of the plot
        step = next(step for step in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10 ** 9)
                    if (last - first) / step <= 10)
        self.trend_plot.plot.getAxis("bottom").setTickSpacing(major=step, minor=max(1, step // 5))  # whole frames
        for x, y, error, color in bars:  # ±1σ as vertical bars, in the plot's coordinates
            keep = np.isfinite(error)
            if not keep.any():
                continue
            low, high = y[keep] - error[keep], y[keep] + error[keep]
            if self.trend_plot.log_check.isChecked():
                low = np.where(low > 0, low, y[keep] / 10.0)
            item = self.trend_plot.plot.plot(np.repeat(x[keep], 2), np.column_stack([low, high]).ravel(),
                                             pen=pg.mkPen(color, width=1.2), connect="pairs")
            self.trend_plot._items.append(item)  # cleared with the curves


    def _legend_corner(self, curves) -> tuple:
        """Where the trend's legend covers the fewest points: its offset, the bottom right first."""
        x = np.concatenate([np.asarray(points, dtype=float) for _name, points, _values in curves])
        y = np.concatenate([np.asarray(values, dtype=float) for _name, _points, values in curves])
        if self.trend_plot.log_check.isChecked():
            y = np.log10(np.where(y > 0, y, np.nan))
        keep = np.isfinite(x) & np.isfinite(y)
        if not keep.any():
            return -10, -10
        x, y = x[keep], y[keep]
        across = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
        up = (y - y.min()) / max(float(np.ptp(y)), 1e-12)
        wide, high = 0.3, 0.45  # about what a legend of a few stages takes of the plot
        corners = (((-10, -10), (across > 1 - wide) & (up < high)), ((10, 10), (across < wide) & (up > 1 - high)),
                   ((10, -10), (across < wide) & (up < high)), ((-10, 10), (across > 1 - wide) & (up > 1 - high)))
        return min(corners, key=lambda corner: int(np.count_nonzero(corner[1])))[0]


__all__ = ["FitSeriesStagesMixin", "LOOKING", "NOT_FOR_THIS_RUN", "NO_STAGES_AT_START", "TREND_EMPTY", "stage_icon"]
