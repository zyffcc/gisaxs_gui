"""Embeddable pyqtgraph detector viewer with its own display controls.

Pixel frames are shown in the canonical frame of ``DetectorGeometry``: pixel
``(row i, column j)`` covers ``[j, j+1] x [i, i+1]`` and row 0 is at the top.
Reciprocal-space maps are shown with the y axis pointing up.  The viewer never
changes the scientific values it receives: log scaling, colour levels and
down-sampling only affect the display, and the cursor readout reports the
value of the full-resolution array.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .levels import LevelControl, auto_levels, level_bar
from .marks import DETECTOR_MARKS, MarkLayers
from .box_zoom import zoom_button
from .curve_plot_extras import IconTools, roomy_ticks
from .flow_layout import FlowLayout

COMPACT_WIDTH = 520
"""Below this width the toolbar shows icons, the colour bar is slim and the x axis has fewer labels."""
COLORMAPS = ("viridis", "inferno", "magma", "plasma", "cividis", "turbo", "gray")
LEVEL_PERCENTILES = (1.0, 99.7)


_IMAGE_ITEM_CLASS = None


def _image_item_class():
    """``pg.ImageItem`` whose NaN mask follows the current down-sampling factor.

    pyqtgraph 0.13 caches the NaN positions of the first rendered (possibly
    down-sampled) image and reuses them after zooming changes the factor,
    which raises IndexError.  Dropping the cache before each render keeps NaN
    pixels transparent at every zoom level.
    """
    global _IMAGE_ITEM_CLASS
    if _IMAGE_ITEM_CLASS is None:
        import pyqtgraph as pg

        class NanSafeImageItem(pg.ImageItem):
            def render(self):
                self._imageNanLocations = None
                super().render()

        _IMAGE_ITEM_CLASS = NanSafeImageItem
    return _IMAGE_ITEM_CLASS


def _colormap(name: str):
    import pyqtgraph as pg

    if name == "gray":
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
    try:
        return pg.colormap.get(name)
    except Exception:
        return pg.colormap.getFromMatplotlib(name)


def display_levels(values: np.ndarray) -> Optional[tuple[float, float]]:
    """Robust colour limits (1–99.7 % of a sample of the finite displayed values)."""
    return auto_levels(values)


def display_array(data: np.ndarray, valid: Optional[np.ndarray], log_scale: bool) -> np.ndarray:
    """float32 array to draw: NaN where invalid (drawn transparent), log10 if asked."""
    array = np.asarray(data, dtype=np.float32)
    shown = np.array(array, dtype=np.float32, copy=True)
    if valid is not None:
        shown[~np.asarray(valid, dtype=bool)] = np.nan
    if log_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            shown = np.where(shown > 0, np.log10(shown), np.nan).astype(np.float32)
    return shown


class DetectorView(QWidget):
    """Detector image + colour bar + display controls + draggable cut bands."""

    horizontalBandChanged = pyqtSignal(float, float)
    verticalBandChanged = pyqtSignal(float, float)
    positionClicked = pyqtSignal(float, float)
    beamCenterMoved = pyqtSignal(float, float)
    """The user dragged the beam-centre target or picked a position in pick mode."""
    boxChanged = pyqtSignal(float, float, float, float)
    """The user moved or resized the rectangle: ``(x0, y0, x1, y1)`` in view units."""
    pickModeChanged = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        import pyqtgraph as pg

        super().__init__(parent)
        self.setObjectName("detectorView")
        self._data: Optional[np.ndarray] = None
        self._valid: Optional[np.ndarray] = None
        self._rect: Optional[tuple[float, float, float, float]] = None
        self._y_down = True
        self._readout: Optional[Callable[[float, float], str]] = None
        self._updating_bands = False
        self._updating_target = False
        self._pick_mode = False
        self._beam_in_view = False
        self._drawing = False
        """Set by a ``ShapeLayer`` while a shape is drawn: the view's own clicks are held back."""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        controls = FlowLayout()  # wraps to a second line in a narrow panel instead of cutting texts
        self.toolbar_layout = controls
        self.log_check = QCheckBox("Log", self)
        self.log_check.setChecked(True)
        self.log_check.setToolTip("Display log10 of the intensity (display only)")
        self.colormap_combo = QComboBox(self)
        self.colormap_combo.addItems(COLORMAPS)
        self.colormap_combo.setToolTip("Colour map")
        self.levels = LevelControl(self)
        self.auto_levels_button = self.levels.make_button(self, self.auto_levels)
        self.fit_button = QToolButton(self)
        self.fit_button.setText("Fit view")
        self.fit_button.setToolTip("Show the whole image again")
        self.title_label = QLabel(self)
        self.title_label.setObjectName("detectorViewTitle")
        self.title_label.setProperty("gimapRole", "muted")
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.graphics = pg.GraphicsLayoutWidget(self)
        self.plot = self.graphics.addPlot(row=0, col=0)
        self.view_box = self.plot.getViewBox()
        self.zoom_button = zoom_button(self.view_box, self)
        self.marks = MarkLayers(self)
        for key, title in DETECTOR_MARKS:
            self.marks.add_layer(key, title)
        self.marks_button = self.marks.menu_button(self)
        for widget in (self.log_check, self.colormap_combo, self.auto_levels_button, self.zoom_button, self.fit_button,
                       self.marks_button):
            controls.addWidget(widget)
        self._compact: Optional[bool] = None
        self._icons = IconTools(self, {"zoom": self.zoom_button, "fit": self.fit_button, "marks": self.marks_button})
        controls.addStretch(1)
        controls.addWidget(self.title_label)
        layout.addLayout(controls)

        self.plot.setMenuEnabled(False)
        self.view_box.setAspectLocked(True)
        for side in ("left", "bottom"):
            # Units are in the axis label (Å⁻¹, pixel); never rescale them to "x0.001".
            self.plot.getAxis(side).enableAutoSIPrefix(False)
        self.plot.getAxis("bottom").tickSpacing = roomy_ticks(self.plot.getAxis("bottom"))  # labels never run together
        self.image_item = _image_item_class()(axisOrder="row-major")
        self.image_item.setAutoDownsample(True)
        self.plot.addItem(self.image_item)
        self.color_bar = level_bar(self.image_item)  # a histogram with absolute handles (components/levels.py)
        self.graphics.addItem(self.color_bar, row=0, col=1)
        self.color_bar.edited.levels.connect(lambda low, high: self.levels.edited(low, high, self.log_check.isChecked()))
        self.levels.changed.connect(self._apply_levels)
        self.marks.track("colorbar", self.color_bar)
        # Label overlay (masked pixels, curve sources): uint8 labels through a colour table.
        self.overlay_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item.setAutoDownsample(True)
        self.overlay_item.setZValue(5)
        self.plot.addItem(self.overlay_item, ignoreBounds=True)
        self.overlay_item.hide()
        self.horizontal_band = pg.LinearRegionItem(
            orientation="horizontal", brush=(249, 115, 22, 40), pen=pg.mkPen("#f97316", width=1.5)
        )
        self.vertical_band = pg.LinearRegionItem(
            orientation="vertical", brush=(34, 211, 238, 35), pen=pg.mkPen("#22d3ee", width=1.5)
        )
        self.horizon_line = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("#e5e7eb", width=1, style=Qt.DashLine)
        )
        self.marker_item = pg.ScatterPlotItem(
            symbol="+", size=16, pen=pg.mkPen("#22d3ee", width=2), brush=None
        )
        self.beam_target = pg.TargetItem(
            pos=(0.0, 0.0),
            size=22,
            symbol="crosshair",
            pen=pg.mkPen("#22d3ee", width=2),
            hoverPen=pg.mkPen("#f97316", width=2),
            movable=True,
        )
        self.beam_target.setToolTip("Beam centre: drag to move it")
        self.box_roi = pg.RectROI((0.0, 0.0), (1.0, 1.0), pen=pg.mkPen("#facc15", width=1.5))
        self.box_roi.setZValue(15)
        self.plot.addItem(self.box_roi, ignoreBounds=True)
        self.box_roi.hide()
        self._updating_box = False
        for item in (
            self.horizontal_band,
            self.vertical_band,
            self.horizon_line,
            self.marker_item,
            self.beam_target,
        ):
            item.setZValue(10)
            self.plot.addItem(item, ignoreBounds=True)
            item.hide()
        self.beam_target.setZValue(20)
        layout.addWidget(self.graphics, 1)

        self.readout_label = QLabel("Open a detector frame", self)
        self.readout_label.setObjectName("detectorViewReadout")
        self.readout_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # The readout changes with every mouse move: it must never change the size of the view
        # (a longer text would widen the panel, and the image would jump). Too long: cut, whole in the tooltip.
        self.readout_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.readout_label.setMinimumWidth(0)
        layout.addWidget(self.readout_label)

        self.log_check.toggled.connect(lambda _checked: self._redraw(keep_view=True))
        self.colormap_combo.currentTextChanged.connect(self.set_colormap)
        self.fit_button.clicked.connect(self.fit_view)
        self.horizontal_band.sigRegionChangeFinished.connect(self._horizontal_finished)
        self.vertical_band.sigRegionChangeFinished.connect(self._vertical_finished)
        self.beam_target.sigPositionChangeFinished.connect(self._beam_target_finished)
        self.box_roi.sigRegionChangeFinished.connect(self._box_finished)
        self._mouse_proxy = pg.SignalProxy(
            self.graphics.scene().sigMouseMoved, rateLimit=30, slot=lambda args: self._mouse_moved(args[0])
        )
        self.graphics.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.set_colormap(self.colormap_combo.currentText())

    # -- image -----------------------------------------------------------------------

    def set_image(
        self,
        data: np.ndarray,
        *,
        valid: Optional[np.ndarray] = None,
        rect: Optional[tuple[float, float, float, float]] = None,
        y_down: bool = True,
        title: str = "",
        x_label: str = "x (pixel)",
        y_label: str = "y (pixel)",
        keep_view: bool = False,
        context: str = "image",
    ) -> None:
        """Show ``data``; ``rect = (x, y, width, height)`` places row 0 at ``y``. ``context`` (detector, q map
        …): each keeps its own colour limits.

        With ``y_down`` (pixel frames) row 0 is drawn at the top; otherwise the
        y axis points up and row 0 is the bottom row.
        """
        array = np.asarray(data)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("DetectorView needs a non-empty 2D array.")
        self._data = array
        self._valid = None if valid is None else np.asarray(valid, dtype=bool)
        self._rect = None if rect is None else tuple(float(value) for value in rect)
        self._y_down = bool(y_down)
        self.levels.set_context(context)
        self.view_box.invertY(self._y_down)
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.title_label.setText(title)
        from ..i18n import tr

        self.readout_label.setText(tr("Move the cursor over the image to read values"))
        self._redraw(keep_view=keep_view)

    def clear(self) -> None:
        self._data = None
        self._valid = None
        self.image_item.clear()
        self.clear_overlays()
        self.hide_labels()
        self.title_label.setText("")
        from ..i18n import tr

        self.readout_label.setText(tr("Open a detector frame"))

    def has_image(self) -> bool:
        return self._data is not None

    def display_state(self) -> Optional[dict]:
        """What is shown, for exporting it as a figure: the displayed array and its styling."""
        if self._data is None:
            return None
        shown = display_array(self._data, self._valid, self.log_check.isChecked())
        rows, columns = shown.shape
        if self._rect is None:
            extent = (0.0, float(columns), float(rows), 0.0) if self._y_down else None
        else:
            left, bottom, width, height = self._rect
            extent = (left, left + width, bottom, bottom + height)
        return {
            "image": shown,
            "extent": extent,
            "origin_upper": self._y_down,
            "x_label": self.plot.getAxis("bottom").labelText,
            "y_label": self.plot.getAxis("left").labelText,
            "colormap": self.colormap_combo.currentText(),
            "log_scale": self.log_check.isChecked(),
            "levels": tuple(float(value) for value in self.color_bar.levels()),
            "title": self.title_label.text(),
        }

    def dispose(self) -> None:
        """Delete the pyqtgraph view before this widget's parent is destroyed.

        pyqtgraph views torn down as part of a parent widget, while Python still
        references them, can abort the interpreter; deleting them first is safe.
        A view that is already gone is left alone: its scene went with it, and
        disconnecting from a deleted scene's signal crashes.
        """
        self._data = None
        self._valid = None
        if sip.isdeleted(self.graphics):
            return
        self._mouse_proxy.disconnect()
        self.graphics.deleteLater()

    def _redraw(self, *, keep_view: bool) -> None:
        if self._data is None:
            return
        shown = display_array(self._data, self._valid, self.log_check.isChecked())
        levels = self.levels.levels_for(shown, self.log_check.isChecked())
        self.image_item.setImage(shown, autoLevels=False, levels=levels or (0.0, 1.0))
        if self._rect is not None:
            self.image_item.setRect(QRectF(*self._rect))
        else:
            self.image_item.setRect(QRectF(0.0, 0.0, float(shown.shape[1]), float(shown.shape[0])))
        if levels is not None:
            self.color_bar.set_levels(levels)
        self.color_bar.set_label("log₁₀ I" if self.log_check.isChecked() else "I")
        if not keep_view:
            self.fit_view()

    def auto_levels(self) -> None:
        """The rule's limits for the image shown now, kept for the next frames (napari's “once”)."""
        self.levels.auto_once(self.image_item.image, self.log_check.isChecked())

    def _apply_levels(self) -> None:
        image = self.image_item.image
        levels = None if image is None else self.levels.levels_for(image, self.log_check.isChecked())
        if levels is not None:
            self.image_item.setLevels(levels)
            self.color_bar.set_levels(levels)

    def set_colormap(self, name: str) -> None:
        colormap = _colormap(str(name))
        self.color_bar.setColorMap(colormap)
        if self.colormap_combo.currentText() != name:
            index = self.colormap_combo.findText(name)
            if index >= 0:
                self.colormap_combo.setCurrentIndex(index)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Narrow views keep the image large: icon buttons, no title, a slim colour bar, fewer x labels."""
        super().resizeEvent(event)
        compact = event.size().width() < COMPACT_WIDTH
        if compact != self._compact:
            self._compact = compact
            from ..i18n import tr

            self.fit_button.setText(tr("Reset" if compact else "Fit view"))
            self.title_label.setVisible(not compact)
            self._icons.set_on(compact)
            self.color_bar.set_compact(compact)  # slim, but its handles still drag
            self.plot.getAxis("bottom").setTickDensity(0.5 if compact else 1.0)  # labels that do not run together

    def fit_view(self) -> None:
        """The whole image — and the beam centre when it is to stay in view (``show_beam_center``)."""
        self.view_box.autoRange(items=[self.image_item], padding=0.02)
        if not self._beam_in_view or not (self.marks.wanted(self.beam_target) and self.marks.is_visible("beam")):
            return
        (x0, x1), (y0, y1) = self.view_box.viewRange()
        point = self.beam_target.pos()
        x, y = float(point.x()), float(point.y())
        if not (x0 <= x <= x1 and y0 <= y <= y1):  # widen just enough, with a small margin
            margin_x, margin_y = 0.03 * (x1 - x0), 0.03 * (y1 - y0)
            self.view_box.setRange(xRange=(min(x0, x - margin_x), max(x1, x + margin_x)),
                                   yRange=(min(y0, y - margin_y), max(y1, y + margin_y)), padding=0)

    def set_aspect_locked(self, locked: bool) -> None:
        """Square pixels (detector frames, q maps) or free axes (a frame × q series map)."""
        self.view_box.setAspectLocked(bool(locked))

    # -- overlays ------------------------------------------------------------------

    def show_labels(self, labels: np.ndarray, colors, *, alpha: int = 120) -> None:
        """Colour pixels by label: 0 transparent, ``i`` in ``colors[i - 1]`` (``"#rrggbb"``).

        ``labels`` has the frame's shape and pixel frame (the detector view only).
        """
        from PyQt5.QtGui import QColor

        labels = np.asarray(labels, dtype=np.uint8)
        table = np.zeros((256, 4), dtype=np.uint8)
        for index, color in enumerate(colors[:255], start=1):
            value = QColor(color)
            table[index] = (value.red(), value.green(), value.blue(), int(alpha))
        self.overlay_item.setImage(labels, autoLevels=False, levels=(0, 255), lut=table)
        rows, columns = labels.shape
        self.overlay_item.setRect(QRectF(0.0, 0.0, float(columns), float(rows)))
        self.marks.track("pixels", self.overlay_item, True)

    def hide_labels(self) -> None:
        self.marks.track("pixels", self.overlay_item, False)

    def labels_shown(self) -> bool:
        return self.marks.wanted(self.overlay_item)

    def clear_overlays(self) -> None:
        for key, item in (
            ("bands", self.horizontal_band),
            ("bands", self.vertical_band),
            ("horizon", self.horizon_line),
            ("points", self.marker_item),
            ("beam", self.beam_target),
            ("box", self.box_roi),
        ):
            self.marks.track(key, item, False)

    def show_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Show the draggable rectangle between two corners (view units)."""
        left, right = sorted((float(x0), float(x1)))
        bottom, top = sorted((float(y0), float(y1)))
        self._updating_box = True
        try:
            self.box_roi.setPos((left, bottom), update=False, finish=False)
            self.box_roi.setSize((max(right - left, 1e-9), max(top - bottom, 1e-9)), finish=False)
            self.marks.track("box", self.box_roi, True)
        finally:
            self._updating_box = False

    def hide_box(self) -> None:
        self.marks.track("box", self.box_roi, False)

    def _box_finished(self, *_args) -> None:
        if self._updating_box or not self.box_roi.isVisible():
            return
        x, y = self.box_roi.pos()
        width, height = self.box_roi.size()
        self.boxChanged.emit(float(x), float(y), float(x + width), float(y + height))

    def show_beam_center(self, x: float, y: float, *, movable: bool = True, keep_in_view: bool = False) -> None:
        """Show the beam-centre target; dragging it emits ``beamCenterMoved``. ``keep_in_view``: Fit view
        includes it (a q map, whose direct beam lies just outside the map)."""
        from ..i18n import tr

        self._updating_target = True
        self._beam_in_view = bool(keep_in_view)
        try:
            self.beam_target.setPos(float(x), float(y))
            self.beam_target.movable = bool(movable)
            self.beam_target.setToolTip(tr("Beam centre: drag to move it") if movable else tr(
                "Beam centre (the direct beam); move it on the detector image"))
            self.marks.track("beam", self.beam_target, True)
        finally:
            self._updating_target = False

    def set_pick_mode(self, enabled: bool) -> None:
        """While on, the next left click on the image sets the beam centre."""
        enabled = bool(enabled) and self._data is not None
        if enabled == self._pick_mode:
            return
        self._pick_mode = enabled
        self.graphics.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.pickModeChanged.emit(enabled)

    def is_pick_mode(self) -> bool:
        return self._pick_mode

    def _beam_target_finished(self, *_args) -> None:
        if not self._updating_target:
            position = self.beam_target.pos()
            self.beamCenterMoved.emit(float(position.x()), float(position.y()))

    def show_horizontal_band(self, low: float, high: float, *, movable: bool = True) -> None:
        self._updating_bands = True
        try:
            self.horizontal_band.setRegion((float(low), float(high)))
            self.horizontal_band.setMovable(movable)
            self.marks.track("bands", self.horizontal_band, True)
        finally:
            self._updating_bands = False

    def show_vertical_band(self, low: float, high: float, *, movable: bool = True) -> None:
        self._updating_bands = True
        try:
            self.vertical_band.setRegion((float(low), float(high)))
            self.vertical_band.setMovable(movable)
            self.marks.track("bands", self.vertical_band, True)
        finally:
            self._updating_bands = False

    def show_horizon(self, y: Optional[float]) -> None:
        if y is None or not np.isfinite(y):
            self.marks.track("horizon", self.horizon_line, False)
            return
        self.horizon_line.setValue(float(y))
        self.marks.track("horizon", self.horizon_line, True)

    def show_markers(self, points, *, symbol: str = "+", color: str = "#22d3ee", size: int = 16) -> None:
        """Symbols at image positions (x, y), e.g. single pixels too small to see when zoomed out."""
        import pyqtgraph as pg

        points = [(float(x), float(y)) for x, y in points]
        if not points:
            self.marks.track("points", self.marker_item, False)
            return
        self.marker_item.setData(pos=points, symbol=symbol, size=size, pen=pg.mkPen(color, width=2), brush=None)
        self.marks.track("points", self.marker_item, True)

    def hide_markers(self) -> None:
        self.marks.track("points", self.marker_item, False)

    def _horizontal_finished(self) -> None:
        if not self._updating_bands:
            low, high = self.horizontal_band.getRegion()
            self.horizontalBandChanged.emit(float(low), float(high))

    def _vertical_finished(self) -> None:
        if not self._updating_bands:
            low, high = self.vertical_band.getRegion()
            self.verticalBandChanged.emit(float(low), float(high))

    # -- cursor ----------------------------------------------------------------------

    def set_readout(self, formatter: Optional[Callable[[float, float], str]]) -> None:
        """``formatter(x, y)`` returns extra text (for example q) for a view position."""
        self._readout = formatter

    def value_at(self, x: float, y: float) -> Optional[float]:
        """Full-resolution value under a view position, ``None`` outside or invalid."""
        if self._data is None:
            return None
        rows, columns = self._data.shape
        if self._rect is None:
            column, row = int(np.floor(x)), int(np.floor(y))
        else:
            left, bottom, width, height = self._rect
            column = int(np.floor((x - left) / width * columns))
            row = int(np.floor((y - bottom) / height * rows))
        if not (0 <= row < rows and 0 <= column < columns):
            return None
        if self._valid is not None and not self._valid[row, column]:
            return None
        value = float(self._data[row, column])
        return value if np.isfinite(value) else None

    def _mouse_moved(self, position) -> None:
        if self._data is None or not self.plot.sceneBoundingRect().contains(position):
            return
        point = self.view_box.mapSceneToView(position)
        x, y = float(point.x()), float(point.y())
        value = self.value_at(x, y)
        text = f"x = {x:.1f}, y = {y:.1f}   I = {value:.4g}" if value is not None else f"x = {x:.1f}, y = {y:.1f}"
        if self._readout is not None:
            extra = self._readout(x, y)
            if extra:
                text += "   " + extra
        self._show_readout(text)

    def _show_readout(self, text: str) -> None:
        """The cursor readout, cut to the width of the view (whole in the tooltip)."""
        room = max(40, self.readout_label.width() - 4)
        shown = self.readout_label.fontMetrics().elidedText(text, Qt.ElideRight, room)
        self.readout_label.setText(shown)
        self.readout_label.setToolTip(text if shown != text else "")

    def _mouse_clicked(self, event) -> None:
        if self._data is None or event.button() != Qt.LeftButton or self._drawing:
            return
        point = self.view_box.mapSceneToView(event.scenePos())
        if self._pick_mode:
            self.set_pick_mode(False)
            self.beamCenterMoved.emit(float(point.x()), float(point.y()))
            return
        if event.double():
            self.positionClicked.emit(float(point.x()), float(point.y()))


__all__ = ["COLORMAPS", "DetectorView", "display_array", "display_levels"]
