"""A running batch shown live in the Series tab: the map of every frame done so far (redrawn at most once a
second) and the newest frame's curve below it. Part of ``BatchRunMixin`` (``bindings/batch_run.py``)."""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore import QSignalBlocker

from ...application import GISAXS

REDRAW_S = 1.0
"""The live map is redrawn at most this often."""
PACE = 4.0
"""…and, a map of many rows taking long to draw, not before ``PACE`` times its last drawing has passed: the window
spends at most a fifth of its time drawing the map."""
STACK_PACE = 2.0
"""…nor before twice its last stacking (on the map's own thread for a map of many rows, ``series_later.py``): that
work takes at most half the time of one processor core from the frames being reduced (20 000 rows: about a second
per stacking)."""


def redraw_wait_s(last_draw_s: float, last_stack_s: float = 0.0) -> float:
    """How long the live map waits between two drawings after one that took ``last_draw_s`` to draw on the window's
    thread and ``last_stack_s`` to stack."""
    return max(REDRAW_S, PACE * max(0.0, float(last_draw_s)), STACK_PACE * max(0.0, float(last_stack_s)))

LIVE_CURVE = {GISAXS: "horizontal"}


class BatchLiveMixin:
    """Needs ``view_model``, ``batch_panel``, the Series tab and the state of ``BatchRunMixin``."""

    def _batch_live_key(self) -> Optional[str]:
        analysis = self.view_model.state.analysis
        reduction = analysis.reduction if analysis is not None else None
        keys = [curve.key for curve in (reduction.curves if reduction is not None else ()) if not curve.is_empty]
        current = self.series_curve_combo.currentData()
        if current in keys:
            return current
        wanted = LIVE_CURVE.get(getattr(analysis, "kind", None), "radial")
        return wanted if wanted in keys else (keys[0] if keys else None)

    def _batch_live_start(self, key: Optional[str]) -> None:
        self._batch_live = bool(key) and (self._batch_total > 1 or self._batch_map_only)
        if not self._batch_live:
            return
        self._series_later_forget()  # (a map of the run before, being stacked, is not shown)
        self._series_live_forget()
        self._series_rows, self._series_failures = [], []
        self._series_aside = []  # (an angle chosen in the αi list is kept for the map built again)
        # A late frame can still be writing a recipe. Reuse the folder until those workers finish.
        self.view_model.begin_series_spool(replace_existing=not self.tasks.is_busy() and not self._batch_inflight)
        self._series_keep_curve(key)  # the key and the name the map goes by (``bindings/series.py``)
        self._series_map = None
        self._clear_stages()
        self._series_row = 0
        self._batch_last_draw, self._batch_rows_drawn = 0.0, 0
        index = self.series_curve_combo.findData(key)
        if index >= 0:
            with QSignalBlocker(self.series_curve_combo):
                self.series_curve_combo.setCurrentIndex(index)
        with QSignalBlocker(self.batch_panel.follow_check):
            self.batch_panel.follow_check.setChecked(True)
        if self._batch_map_only:
            self._series_say("Reducing {n} frames: each appears here as soon as it is done.", n=self._batch_total)
        else:
            self._series_say("Batch Export: every frame appears here as soon as it is done.")
        self.show_right("series")

    def _batch_live_redraw(self, *, force: bool = False) -> None:
        """Draw the map of the rows so far again, paced (``redraw_wait_s``); ``force``: now, here (the last map of a
        run, before its stages are looked for; Follow turned on or off)."""
        if not self._batch_live or not self._series_rows:
            return
        if not force and self._series_later_busy():
            return  # the map of the rows before is being stacked: these rows at a redraw after it
        self._watch_series_normalize()
        if not force and len(self._series_rows) == self._batch_rows_drawn:
            return
        now = time.monotonic()
        wait = redraw_wait_s(getattr(self, "_series_draw_s", 0.0), getattr(self, "_series_build_s", 0.0))
        if not force and now - self._batch_last_draw < wait:
            return
        self._batch_last_draw, self._batch_rows_drawn = now, len(self._series_rows)
        follow = self.batch_panel.follow_check.isChecked()
        if follow:
            self._series_row = len(self._series_rows) - 1
        had_map = self._series_map is not None
        if force:
            self._show_series_map(keep_view=not follow and had_map, keep_q=had_map, live=True)
        else:
            self._series_redraw_live(keep_view=not follow and had_map, keep_q=had_map)

    def _batch_unfollow(self) -> None:
        streaming = self.view_model.watch.active and self._series_map is not None
        if (streaming or (self._batch_live and self.batch_running())) and self.batch_panel.follow_check.isChecked():
            with QSignalBlocker(self.batch_panel.follow_check):
                self.batch_panel.follow_check.setChecked(False)
            with QSignalBlocker(self.series_watch_follow_check):
                self.series_watch_follow_check.setChecked(False)

    def _batch_live_finish(self) -> None:
        if self._batch_live:
            self._batch_live_redraw(force=True)
            self._find_stages()
        self._batch_live = False
        self.series_build_button.setEnabled(self.series_curve_combo.count() > 0)


__all__ = ["BatchLiveMixin", "LIVE_CURVE", "PACE", "REDRAW_S", "STACK_PACE", "redraw_wait_s"]
