"""Show in Analyze: the detector frames of one time window, summed, in the Analyze workspace (a mixin of
``SputterPage``).

A right click on a row of the Windows table, on the Structure Sketch (the window it shows) or on the Size Trend (the
window nearest in time to the click) offers **Show in Analyze** under the window's time. The page reads which raw
frames the window summed and the geometry of the result through the service (``SputterService.window``; a problem —
the scan folder moved, frames missing, a result without binned windows — is said here) and hands the window to the
composition root (``set_analyze_target``), which shows it in Analyze. A page without a target (on its own, in a
test) offers the entry disabled.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMenu

from src.gimap.app.presentation.i18n import tr

from .texts import message_text

SHOW_IN_ANALYZE = "Show in Analyze"


class WindowMenuMixin:
    """Needs ``table``, ``sketch``, ``trend``, ``result``, ``service`` and ``_notify``."""

    _analyze_target: Optional[Callable[[object], object]] = None

    def set_analyze_target(self, show: Optional[Callable[[object], object]]) -> None:
        """``show(window)`` shows a window's frames (``WindowFrames``) in Analyze; ``None``: not offered."""
        self._analyze_target = show

    def _connect_window_menu(self) -> None:
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)  # its Copy Rows / Copy Table stay in the menu
        self.table.customContextMenuRequested.connect(self._table_menu)
        self.sketch.windowMenuRequested.connect(lambda row, where: self._popup(self._window_menu(row, self.sketch), where))
        canvas = self.trend.canvas
        if canvas is not None:
            canvas.setContextMenuPolicy(Qt.CustomContextMenu)
            canvas.customContextMenuRequested.connect(self._trend_menu)

    # -- which window ------------------------------------------------------------------------------------------
    def _row(self, index) -> Optional[dict]:
        rows = self.result.rows if self.result is not None else []
        return rows[index] if index is not None and 0 <= index < len(rows) else None

    def nearest_row(self, time_s: float) -> Optional[int]:
        """The row of the result whose window is nearest in time to ``time_s`` (None: no window has a time)."""
        rows = self.result.rows if self.result is not None else []
        timed = [(abs(row["time_s"] - time_s), index) for index, row in enumerate(rows)
                 if row.get("time_s") is not None and row.get("frame")]
        return min(timed)[1] if timed else None

    def _trend_time(self, position) -> Optional[float]:
        """The time under a point of the Size Trend (Qt widget coordinates), None outside its panels."""
        canvas, figure = self.trend.canvas, self.trend.figure
        if canvas is None or figure is None or not figure.axes:
            return None
        x, y = canvas.mouseEventCoords(position)
        if not any(ax.bbox.contains(x, y) for ax in figure.axes):
            return None
        return float(figure.axes[0].transData.inverted().transform((x, y))[0])  # every panel shares the time axis

    # -- the menus -----------------------------------------------------------------------------------------------
    def _window_menu(self, index, parent) -> QMenu:
        """The menu of one window (row ``index`` of the result): its time on top (greyed), then Show in Analyze."""
        row = self._row(index)
        menu = QMenu(parent)
        if row is not None and row.get("time_s") is not None:
            menu.addAction(f"t = {row['time_s']:.2f} s").setEnabled(False)  # which window (a time: not translated)
        frame = str(row.get("frame") or "") if row is not None else ""
        show = menu.addAction(tr(SHOW_IN_ANALYZE), lambda: self.show_window_in_analyze(frame))
        show.setEnabled(bool(frame) and self._analyze_target is not None)
        menu.aboutToHide.connect(menu.deleteLater)  # after its action ran (deleted on the next turn of the event loop)
        return menu

    @staticmethod
    def _popup(menu: QMenu, where) -> None:
        menu.popup(where)

    def _table_menu(self, position) -> None:
        index = self.table.indexAt(position)
        menu = self._window_menu(index.row() if index.isValid() else None, self.table)
        copies = self.table.actions()  # Copy Rows, Copy Table (``enable_table_copy``)
        if copies:
            menu.addSeparator()
            menu.addActions(copies)
        self._popup(menu, self.table.viewport().mapToGlobal(position))

    def _trend_menu(self, position) -> None:
        time_s = self._trend_time(position)
        index = self.nearest_row(time_s) if time_s is not None else None
        if index is not None:
            self._popup(self._window_menu(index, self.trend.canvas), self.trend.canvas.mapToGlobal(position))

    # -- the hand-over -------------------------------------------------------------------------------------------
    def show_window_in_analyze(self, frame: str) -> bool:
        """Read the window ``frame`` (its id) of the result shown and hand it to Analyze; ``False``: not offered, or
        the window cannot be read (said here) or Analyze refused it (said there)."""
        if self._analyze_target is None or self.result is None or not frame:
            return False
        try:
            window = self.service.window(self.result.folder, frame)
        except (OSError, ValueError) as exc:
            self._notify(message_text(exc), error=True)
            return False
        return bool(self._analyze_target(window))


__all__ = ["SHOW_IN_ANALYZE", "WindowMenuMixin"]
