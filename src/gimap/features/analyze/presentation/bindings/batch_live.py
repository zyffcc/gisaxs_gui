"""A running batch shown live in the Series tab: the map of every frame done so far (redrawn at most once a
second) and the newest frame's curve below it. Part of ``BatchRunMixin`` (``bindings/batch_run.py``)."""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore import QSignalBlocker

from ...application import GISAXS

REDRAW_S = 1.0
"""The live map is redrawn at most this often."""
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
        self._series_rows, self._series_failures = [], []
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
        if not self._batch_live or not self._series_rows:
            return
        self._watch_series_normalize()
        if not force and len(self._series_rows) == self._batch_rows_drawn:
            return
        now = time.monotonic()
        if not force and now - self._batch_last_draw < REDRAW_S:
            return
        self._batch_last_draw, self._batch_rows_drawn = now, len(self._series_rows)
        follow = self.batch_panel.follow_check.isChecked()
        if follow:
            self._series_row = len(self._series_rows) - 1
        had_map = self._series_map is not None
        self._show_series_map(keep_view=not follow and had_map, keep_q=had_map)

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


__all__ = ["BatchLiveMixin", "LIVE_CURVE", "REDRAW_S"]
