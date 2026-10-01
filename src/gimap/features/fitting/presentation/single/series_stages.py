"""Stages and odd frames of the In-situ series (``application/series_fit.stages_of_curves``).

Once the curves are listed, all of them are read in the background and compared as they will be fitted
(the halves, range and left-out points of Single analysis): the Curves step says how many stages and odd
frames there are, the frame list shows each frame's stage, the odd frames can be left out of the run, a
new stage can restart from the Single model, and the trend is drawn in the colours of the stages.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QBrush, QColor

from src.gimap.app.presentation.components import stage_color
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.stage_text import change_text
from src.gimap.app.presentation.task_runner import TaskRunner

from ...application import LoadCurveRequest
from ...application.series_fit import SeriesSettings, stages_of_curves
from ...application.single_fit import Curve
from .session import path_name

CHI2 = "chi2"


class FitSeriesStagesMixin:
    """Needs the series page: ``paths``, ``_listed()``, ``single``, ``view_model``, ``tasks``, the widgets."""

    def _connect_stages(self) -> None:
        self._series_stages = None
        self._stage_rows: dict[int, int] = {}
        self._stages_key = None
        self._stages_timer = QTimer(self)
        self._stages_timer.setSingleShot(True)
        self._stages_timer.setInterval(400)
        self._stages_timer.timeout.connect(self._find_stages)
        self.stage_tasks = TaskRunner(self)  # apart from the fits: Pause and Stop look at those alone
        self.skip_odd_check.toggled.connect(lambda _on: self.refresh())

    def _stage_settings(self) -> SeriesSettings:
        session = self.single.session
        return SeriesSettings(side=session.side, q_range=session.q_range, excluded=frozenset(session.excluded))

    def _schedule_stages(self) -> None:
        """Find the stages again when the listed curves or the way they are fitted changed."""
        settings = self._stage_settings()
        key = (tuple(self.paths[index] for index in self._listed()), settings.side, settings.q_range, settings.excluded)
        if key != self._stages_key and not self.running:
            self._stages_key = key
            self._stages_timer.start()

    def _find_stages(self) -> None:
        listed = self._listed()
        self._series_stages, self._stage_rows = None, {}
        if len(listed) < 3 or self.running:
            self.stages_label.setText("")
            self.skip_odd_check.hide()
            return
        paths = [self.paths[index] for index in listed]
        settings = self._stage_settings()
        curve = self.single.session.curve
        unit = curve.source_unit if curve is not None else "angstrom"
        load = self.view_model.load_curve

        def work():
            curves = []
            for path in paths:
                outcome = load(LoadCurveRequest(Path(path), unit))
                if outcome.error is not None:
                    raise ValueError(f"{Path(path).name}: {outcome.error.message}")
                value = outcome.value
                curves.append(Curve.from_arrays(value.q, value.intensity, value.error, name=Path(path).name, path=path,
                                                unit=value.q_source_unit))
            return stages_of_curves(curves, settings)

        self.stages_label.setText(tr("Looking for stages and odd frames …"))
        self.stage_tasks.submit("series-stages", work, on_done=lambda stages: self._stages_found(listed, stages),
                          on_error=lambda message, _trace: self.stages_label.setText(
                              tr("No stages: {reason}").format(reason=message)))

    def _stages_found(self, listed: list, stages) -> None:
        if listed != self._listed() or self.running:
            return
        self._series_stages = stages
        self._stage_rows = {index: row for row, index in enumerate(listed)}
        ranges = ", ".join(f"{listed[first] + 1}–{listed[last] + 1}" for first, last in stages.ranges())
        text = (tr("{count} stages: frames {ranges}").format(count=stages.count, ranges=ranges) if stages.count > 1
                else tr("One stage: no change of course"))
        if stages.odd:
            text += " · " + tr("odd frames: {frames}").format(frames=", ".join(str(listed[f.row] + 1) for f in stages.odd))
        self.stages_label.setText(text)
        self.stages_label.setToolTip("\n".join(change_text(change) for change in stages.stage_changes()))
        self.skip_odd_check.setText(tr("Leave out the odd frames ({count})").format(count=len(stages.odd)))
        self.skip_odd_check.setVisible(bool(stages.odd))
        self._frames_changed()

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

    def _mark_stages(self) -> None:
        """Colour every listed frame by its stage; odd frames say so."""
        for position, index in enumerate(self._listed()):
            item = self.frame_list.item(position)
            stage = self.stage_of_frame(index)
            if item is None or stage is None:
                continue
            item.setForeground(QBrush(QColor(stage_color(stage))))
            item.setToolTip(tr("Odd frame (left out of the stages)") if self.is_odd_frame(index)
                            else tr("Stage {n}").format(n=stage))
            if self.is_odd_frame(index):
                item.setText(item.text() + "  · " + tr("odd"))

    def _without_odd(self, listed: list) -> list:
        if not self.skip_odd_check.isChecked() or self._series_stages is None:
            return listed
        kept = [index for index in listed if not self.is_odd_frame(index)]
        if len(kept) < len(listed):
            self._log(tr("{n} odd frames left out.").format(n=len(listed) - len(kept)))
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
        frames = [self.fits[index] for index in sorted(self.fits) if self.fits[index].ok]
        if key is None or not frames:
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
            color = "#2563eb" if stage is None or len(groups) == 1 else stage_color(stage)
            label = name if stage is None or len(groups) == 1 else tr("{name}, stage {n}").format(name=name, n=stage)
            curves.append((label, x, y))
            colors.append(color)
            bars.append((x, y, error, color))
        self.trend_plot.set_labels("frame", "χ²ᵣ" if key == CHI2 else path_name(self._start_model, key, unit=True, translate=False))
        self.trend_plot.set_curves(curves, colors, markers=["o"] * len(curves))
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


__all__ = ["FitSeriesStagesMixin"]
