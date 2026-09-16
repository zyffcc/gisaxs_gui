"""Reusable pixel detector inspection; scientific transforms remain with the caller."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLabel


class ScientificImageViewer(QWidget):
    """Persistent pyqtgraph items with explicit display and interaction intents.

    Input arrays are read-only from this component's perspective. Display pixels
    may be downsampled; cursor intensity always comes from the supplied full array.
    Nonuniform q grids must use the feature's Matplotlib projection instead.
    """

    region_selected = pyqtSignal(tuple)
    log_changed = pyqtSignal(bool)
    levels_changed = pyqtSignal(float, float)
    frame_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        import pyqtgraph as pg

        super().__init__(parent, Qt.Window)

        self.setWindowTitle("Interactive detector · pixel view")
        self.resize(900, 650)
        self._updating = False
        self._intensity = None
        self._extent = None
        self._geometry = None
        self._colormap = None
        self._frame_index = 0
        self._frame_count = 1
        self._frame_pending = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._next_frame)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.reset_button = QPushButton("Fit image")
        self.roi_button = QPushButton("ROI")
        self.roi_button.setCheckable(True)
        self.apply_button = QPushButton("Apply ROI")
        self.apply_button.setEnabled(False)
        self.log_button = QCheckBox("Log intensity")
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setVisible(False)
        for widget in (
            self.reset_button,
            self.roi_button,
            self.apply_button,
            self.log_button,
            self.play_button,
        ):
            controls.addWidget(widget)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.graphics = pg.GraphicsLayoutWidget()
        self.plot = self.graphics.addPlot(row=0, col=0)
        self.plot.setLabel("bottom", "X (pixel)")
        self.plot.setLabel("left", "Y (pixel)")
        self.view_box = self.plot.getViewBox()
        self.view_box.setAspectLocked(True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setAutoDownsample(True)
        self.plot.addItem(self.image_item)
        self.histogram = pg.HistogramLUTItem()
        self.histogram.setImageItem(self.image_item)
        self.histogram.setMinimumWidth(155)
        self.histogram.gradient.setToolTip(
            "Right-click to choose an inspection color map. Exports use the workspace color map."
        )
        self.graphics.addItem(self.histogram, row=0, col=1)
        self.roi = pg.RectROI([0, 0], [1, 1], pen="#f97316")
        self.roi.setZValue(20)
        self.plot.addItem(self.roi)
        self.roi.hide()
        self.cross_x = pg.InfiniteLine(angle=90, movable=False, pen="#22d3ee")
        self.cross_y = pg.InfiniteLine(angle=0, movable=False, pen="#22d3ee")
        for item in (self.cross_x, self.cross_y):
            self.plot.addItem(item, ignoreBounds=True)
            item.hide()
        self.center_item = pg.ScatterPlotItem(symbol="+", size=16, pen="#22d3ee")
        self.plot.addItem(self.center_item, ignoreBounds=True)
        self.overlay_item = pg.PlotDataItem(pen="#f97316")
        self.plot.addItem(self.overlay_item, ignoreBounds=True)
        layout.addWidget(self.graphics, 1)
        self.readout = QLabel("Open a detector image to inspect pixels")
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.readout)
        self.reset_button.clicked.connect(self.view_box.autoRange)
        self.roi_button.toggled.connect(self.roi.setVisible)
        self.roi_button.toggled.connect(self.apply_button.setEnabled)
        self.apply_button.clicked.connect(self._apply_roi)
        self.log_button.toggled.connect(self._log_changed)
        self.histogram.sigLevelChangeFinished.connect(self._levels_changed)
        self.play_button.toggled.connect(self._play_changed)
        self._mouse_proxy = pg.SignalProxy(
            self.graphics.scene().sigMouseMoved,
            rateLimit=60,
            slot=lambda event: self._mouse_moved(event[0]),
        )

    def set_frame(
        self,
        display,
        *,
        intensity,
        extent,
        origin,
        levels,
        colormap,
        log_scale,
        no_data_color="white",
    ):
        """Project caller-prepared pixels without applying any scientific transform."""
        from matplotlib import colormaps
        import pyqtgraph as pg

        values = np.asarray(display)
        raw = np.asarray(intensity)
        if values.ndim != 2 or raw.ndim != 2 or not values.size or not raw.size:
            raise ValueError("Detector images must be nonempty 2D arrays")
        x0, x1, y0, y1 = map(float, extent)
        if x0 == x1 or y0 == y1 or origin not in {"upper", "lower"}:
            raise ValueError("Invalid detector extent or origin")
        geometry = (raw.shape, (x0, x1, y0, y1), origin)
        self._updating = True
        try:
            self._intensity = raw
            self._extent = (x0, x1, y0, y1)
            self._origin = origin
            self.graphics.setVisible(True)
            self.graphics.setBackground(no_data_color)
            self.roi_button.setEnabled(True)
            self.log_button.setEnabled(True)
            self.apply_button.setEnabled(self.roi_button.isChecked())
            self.image_item.setImage(np.ascontiguousarray(values), autoLevels=False, levels=levels)
            # Matplotlib extent is left/right/bottom/top; row 0 depends on origin.
            first_y, last_y = (y1, y0) if origin == "upper" else (y0, y1)
            self.image_item.setRect(QRectF(x0, first_y, x1 - x0, last_y - first_y))
            self.view_box.invertY(y0 > y1)
            if self._colormap != colormap:
                colors = colormaps.get_cmap(colormap)(np.linspace(0, 1, 256)) * 255
                self.histogram.gradient.setColorMap(pg.ColorMap(np.linspace(0, 1, 256), colors))
                self.histogram.gradient.showTicks(False)
                self._colormap = colormap
            self.histogram.setLevels(*levels)
            self.log_button.setChecked(log_scale)
            if geometry != self._geometry:
                self._geometry = geometry
                self.roi.setPos((min(x0, x1), min(y0, y1)))
                self.roi.setSize((abs(x1 - x0) / 3, abs(y1 - y0) / 3))
                self.roi.maxBounds = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
                self.view_box.autoRange()
            self.readout.setText("Move over pixels for coordinates and intensity")
        finally:
            self._updating = False

    def set_unavailable(self, message):
        """Do not leave stale pixel data actionable when the source changes mode."""
        self.stop_playback()
        self._intensity = None
        self.graphics.hide()
        self.apply_button.setEnabled(False)
        self.roi_button.setEnabled(False)
        self.log_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.readout.setText(message)

    def set_overlay(self, *, bounds=None, center=None, paths=()):
        self.center_item.setData(
            [] if center is None else [center[0]], [] if center is None else [center[1]]
        )
        outlines = list(paths)
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            outlines.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]))
        if outlines:
            points = np.concatenate([np.vstack((path, [np.nan, np.nan])) for path in outlines])
            self.overlay_item.setData(points[:, 0], points[:, 1], connect="finite")
        else:
            self.overlay_item.setData([], [])

    def intensity_at(self, x, y):
        if self._intensity is None:
            return None
        x0, x1, y0, y1 = self._extent
        first_y, last_y = (y1, y0) if self._origin == "upper" else (y0, y1)
        height, width = self._intensity.shape
        column = int(np.floor((x - x0) / (x1 - x0) * width))
        row = int(np.floor((y - first_y) / (last_y - first_y) * height))
        if not (0 <= row < height and 0 <= column < width):
            return None
        return row, column, float(self._intensity[row, column])

    def _mouse_moved(self, position):
        point = self.view_box.mapSceneToView(position)
        sample = self.intensity_at(point.x(), point.y())
        for item in (self.cross_x, self.cross_y):
            item.setVisible(sample is not None)
        if sample is None:
            self.readout.setText("Outside detector")
            return
        self.cross_x.setPos(point.x())
        self.cross_y.setPos(point.y())
        row, column, value = sample
        self.readout.setText(
            f"X {point.x():.3f} · Y {point.y():.3f} · "
            f"row {row}, column {column} · intensity {value:.6g}"
        )

    def _apply_roi(self):
        if self._intensity is None:
            return
        position, size = self.roi.pos(), self.roi.size()
        self.region_selected.emit(
            (position.x(), position.x() + size.x(), position.y(), position.y() + size.y())
        )

    def _log_changed(self, checked):
        if not self._updating:
            self.log_changed.emit(checked)

    def _levels_changed(self):
        if not self._updating and self._intensity is not None:
            self.levels_changed.emit(*self.histogram.getLevels())

    def set_frame_position(self, index, count):
        self._frame_index, self._frame_count = int(index), max(1, int(count))
        self._frame_pending = False
        self.play_button.setVisible(count > 1)
        self.play_button.setEnabled(count > 1 and self._intensity is not None)
        if count <= 1:
            self.stop_playback()

    def _play_changed(self, playing):
        self.play_button.setText("Pause" if playing else "Play")
        if playing and self._frame_count > 1 and self._intensity is not None:
            self._timer.start()
        else:
            self._timer.stop()

    def _next_frame(self):
        if not self._frame_pending and self._frame_count > 1:
            self._frame_pending = True
            self.frame_requested.emit((self._frame_index + 1) % self._frame_count)

    def stop_playback(self):
        self.play_button.setChecked(False)
        self._timer.stop()
        self._frame_pending = False

    def hideEvent(self, event):
        self.stop_playback()
        super().hideEvent(event)
