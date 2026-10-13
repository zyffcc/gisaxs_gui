"""A CurvePlot selection band whose public interval always stays in data units."""

from __future__ import annotations

import numpy as np

from .curve_plot_extras import style_window


class CurveWindow:
    """Keep a data-space interval while the plot switches between linear and log x.

    A window reaching zero or negative x cannot be represented on a log axis. Hide
    it there and retain the original interval, so changing display scale never
    changes a scientific selection. The Marks layer still owns its visibility
    preference.
    """

    def __init__(self, plot):
        import pyqtgraph as pg

        self.plot = plot
        self.interval = None
        self._wanted = False
        self._updating = False
        self.item = pg.LinearRegionItem(orientation="vertical")
        style_window(self.item)
        self.item.setZValue(10)
        plot.plot.addItem(self.item, ignoreBounds=True)
        self.item.hide()
        self.item.sigRegionChangeFinished.connect(self._finished)

    def show(self, low: float, high: float, color=None) -> None:
        self.interval = tuple(sorted((float(low), float(high))))
        self._wanted = True
        style_window(self.item, color)
        self.refresh()

    def hide(self) -> None:
        self._wanted = False
        self.plot.marks.track("window", self.item, False)

    def refresh(self) -> None:
        """Render the retained interval without emitting a new data selection."""
        region = self.interval
        represented = region is not None and np.isfinite(region).all()
        if represented and self.plot.log_x_check.isChecked():
            represented = min(region) > 0
            if represented:
                region = tuple(np.log10(region))
        self._updating = True
        try:
            if represented:
                self.item.setRegion(region)
            self.plot.marks.track("window", self.item, self._wanted and represented)
        finally:
            self._updating = False

    def _finished(self) -> None:
        if self._updating or not self._wanted or not self.plot.marks.wanted(self.item):
            return
        region = self.item.getRegion()
        if self.plot.log_x_check.isChecked():
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                region = tuple(np.power(10.0, region))
            if not np.isfinite(region).all() or min(region) <= 0:
                self.refresh()
                return
        if not np.isfinite(region).all():
            self.refresh()
            return
        self.interval = tuple(map(float, region))
        self.plot.windowChanged.emit(*self.interval)
