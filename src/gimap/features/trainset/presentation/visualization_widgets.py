"""Reusable visualization widgets for Trainset presentation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from matplotlib import colormaps

from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal

from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap

from PyQt5.QtWidgets import (
    QSizePolicy,
    QWidget,
)


class ArrayCanvas(QWidget):
    region_created = pyqtSignal(str, dict)
    position_changed = pyqtSignal(dict)

    def __init__(
        self,
        empty_text: str = "Load a real scattering file to begin",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setMinimumSize(300, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.empty_text = empty_text
        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.roi: Optional[Dict[str, int]] = None
        self.beam_center: Optional[tuple[float, float]] = None
        self.binary_mode = False
        self.display_colormap = "gray"
        self.display_log = False
        self.display_auto_scale = True
        self.display_vmin = 0.0
        self.display_vmax = 1.0
        self.mode = ""
        self._press: Optional[QPoint] = None
        self._current: Optional[QPoint] = None
        self._draw_rect = QRect()

    def set_data(
        self,
        image: Optional[np.ndarray],
        mask: Optional[np.ndarray] = None,
        roi: Optional[Dict[str, int]] = None,
        binary: bool = False,
        beam_center: Optional[tuple[float, float]] = None,
    ) -> None:
        self.image = None if image is None else np.asarray(image)
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.roi = roi
        self.beam_center = beam_center
        self.binary_mode = binary
        self.update()

    def set_draw_mode(self, mode: str) -> None:
        self.mode = mode
        self.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)

    def set_display_options(
        self,
        colormap: str,
        log_scale: bool,
        auto_scale: bool,
        vmin: float,
        vmax: float,
    ) -> None:
        self.display_colormap = colormap if colormap in colormaps else "gray"
        self.display_log = bool(log_scale)
        self.display_auto_scale = bool(auto_scale)
        self.display_vmin = float(vmin)
        self.display_vmax = float(vmax)
        self.update()

    def _image_rect(self) -> QRect:
        if self.image is None or not self.image.size:
            return QRect()
        height, width = self.image.shape[:2]
        colorbar_space = 54 if not self.binary_mode else 0
        available_width = max(1, self.width() - colorbar_space)
        scale = min(available_width / max(width, 1), self.height() / max(height, 1))
        draw_width, draw_height = int(width * scale), int(height * scale)
        return QRect(
            (available_width - draw_width) // 2,
            (self.height() - draw_height) // 2,
            draw_width,
            draw_height,
        )

    def _to_image(self, point: QPoint, clip: bool = True) -> QPoint:
        rect = self._image_rect()
        if rect.isEmpty() or self.image is None:
            return QPoint()
        x = (point.x() - rect.left()) / max(rect.width(), 1)
        y = (point.y() - rect.top()) / max(rect.height(), 1)
        if clip:
            x = float(np.clip(x, 0.0, 1.0))
            y = float(np.clip(y, 0.0, 1.0))
        return QPoint(
            int(round(x * max(self.image.shape[1] - 1, 0))),
            int(round(y * max(self.image.shape[0] - 1, 0))),
        )

    def mousePressEvent(self, event) -> None:
        if (
            self.mode
            and event.button() == Qt.LeftButton
            and self._image_rect().contains(event.pos())
        ):
            if self.mode == "beam_center":
                point = self._to_image(event.pos())
                self.region_created.emit(
                    "beam_center", {"x": float(point.x()), "y": float(point.y())}
                )
                self.set_draw_mode("")
                return
            self._press = event.pos()
            self._current = event.pos()
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.image is not None and self._image_rect().contains(event.pos()):
            point = self._to_image(event.pos())
            x = max(0, min(point.x(), self.image.shape[1] - 1))
            y = max(0, min(point.y(), self.image.shape[0] - 1))
            self.position_changed.emit({"x": x, "y": y, "intensity": float(self.image[y, x])})
        if self._press is not None:
            self._current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._press is None or self._current is None or self.image is None:
            return
        clip_to_image = self.mode in {"roi", "roi_ellipse"}
        first = self._to_image(self._press, clip=clip_to_image)
        second = self._to_image(event.pos(), clip=clip_to_image)
        x0, x1 = sorted((first.x(), second.x()))
        y0, y1 = sorted((first.y(), second.y()))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            if self.mode in {"circle", "ellipse"}:
                payload = {
                    "type": "ellipse",
                    "cx": (x0 + x1) / 2.0,
                    "cy": (y0 + y1) / 2.0,
                    "radius_x": max(1.0, (x1 - x0) / 2.0),
                    "radius_y": max(1.0, (y1 - y0) / 2.0),
                }
            else:
                payload = {
                    "type": "rectangle",
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                }
            self.region_created.emit(self.mode, payload)
        self._press = self._current = None
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(248, 250, 252))
        painter.setPen(QPen(QColor(215, 222, 232), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self.image is None or not self.image.size:
            painter.setPen(QColor(100, 116, 139))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.empty_text)
            return
        data = np.asarray(self.image, dtype=np.float64)
        finite = data[np.isfinite(data)]
        if self.binary_mode:
            normalized = (np.nan_to_num(data) > 0).astype(np.float64)
        elif finite.size:
            if self.display_log:
                positive = finite[finite > 0]
                if positive.size:
                    if self.display_auto_scale:
                        low, high = np.percentile(positive, [1.0, 99.5])
                    else:
                        low = self.display_vmin if self.display_vmin > 0 else float(positive.min())
                        high = self.display_vmax
                    high = max(float(high), float(low) * (1.0 + 1e-12))
                    transformed = np.log10(np.clip(data, float(low), float(high)))
                    log_low, log_high = np.log10(float(low)), np.log10(float(high))
                    normalized = np.clip(
                        (transformed - log_low) / max(log_high - log_low, 1e-12), 0.0, 1.0
                    )
                else:
                    normalized = np.zeros(data.shape)
            else:
                if self.display_auto_scale:
                    low, high = np.percentile(finite, [1.0, 99.5])
                else:
                    low, high = self.display_vmin, self.display_vmax
                normalized = np.clip((data - low) / max(float(high) - float(low), 1e-12), 0.0, 1.0)
        else:
            normalized = np.zeros(data.shape)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        if self.binary_mode:
            gray = np.ascontiguousarray((normalized * 255).astype(np.uint8))
            qimage = QImage(
                gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format_Grayscale8
            ).copy()
        else:
            rgba = np.ascontiguousarray(
                colormaps[self.display_colormap](normalized, bytes=True), dtype=np.uint8
            )
            qimage = QImage(
                rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888
            ).copy()
        target = self._image_rect()
        painter.drawPixmap(target, QPixmap.fromImage(qimage))
        if not self.binary_mode and target.width() > 40:
            bar_width = 13
            bar_x = min(self.width() - bar_width - 28, target.right() + 10)
            bar_rect = QRect(bar_x, target.top(), bar_width, target.height())
            gradient_values = np.linspace(1.0, 0.0, max(2, bar_rect.height()), dtype=np.float64)[
                :, None
            ]
            gradient_rgba = np.ascontiguousarray(
                colormaps[self.display_colormap](gradient_values, bytes=True),
                dtype=np.uint8,
            )
            gradient_image = QImage(
                gradient_rgba.data,
                1,
                gradient_rgba.shape[0],
                gradient_rgba.strides[0],
                QImage.Format_RGBA8888,
            ).copy()
            painter.drawPixmap(bar_rect, QPixmap.fromImage(gradient_image))
            painter.setPen(QPen(QColor(71, 85, 105), 1))
            painter.drawRect(bar_rect)
            painter.drawText(
                QRect(bar_rect.right() + 4, bar_rect.top() - 2, 24, 16), Qt.AlignLeft, "max"
            )
            painter.drawText(
                QRect(bar_rect.right() + 4, bar_rect.bottom() - 14, 24, 16), Qt.AlignLeft, "min"
            )
        if self.beam_center is not None:
            center_x, center_y = self.beam_center
            sx = target.width() / max(data.shape[1], 1)
            sy = target.height() / max(data.shape[0], 1)
            px = target.left() + int(center_x * sx)
            py = target.top() + int(center_y * sy)
            if target.adjusted(-1, -1, 1, 1).contains(px, py):
                painter.setPen(QPen(QColor(255, 196, 73), 2))
                painter.drawLine(px - 10, py, px + 10, py)
                painter.drawLine(px, py - 10, px, py + 10)
        if self.mask is not None and self.mask.shape == data.shape:
            overlay = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
            overlay[self.mask] = (235, 82, 82, 125)
            overlay = np.ascontiguousarray(overlay)
            mask_image = QImage(
                overlay.data,
                overlay.shape[1],
                overlay.shape[0],
                overlay.strides[0],
                QImage.Format_RGBA8888,
            ).copy()
            painter.drawPixmap(target, QPixmap.fromImage(mask_image))
        if self.roi:
            sx = target.width() / max(data.shape[1], 1)
            sy = target.height() / max(data.shape[0], 1)
            roi_rect = QRect(
                target.left() + int(self.roi["x"] * sx),
                target.top() + int(self.roi["y"] * sy),
                int(self.roi["width"] * sx),
                int(self.roi["height"] * sy),
            )
            painter.setPen(QPen(QColor(88, 180, 255), 2))
            painter.drawRect(roi_rect)
        if self._press is not None and self._current is not None:
            painter.setPen(QPen(QColor(255, 196, 73), 2, Qt.DashLine))
            rect = QRect(self._press, self._current).normalized()
            if self.mode in {"circle", "ellipse", "roi_ellipse"}:
                painter.drawEllipse(rect)
            else:
                painter.drawRect(rect)


class HistogramWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.x = np.array([])
        self.y = np.array([])
        self.setMinimumHeight(150)

    def set_data(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        painter.setPen(QColor(71, 85, 105))
        painter.drawText(12, 20, "Processed simulated-pixel intensity")
        if not self.y.size or self.y.max() <= 0:
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Run a simulated comparison to calculate this distribution.",
            )
            return
        plot = self.rect().adjusted(68, 34, -18, -48)
        painter.setPen(QPen(QColor(100, 116, 139), 1))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        painter.setPen(QPen(QColor(37, 99, 235), 2))
        points = []
        x_min = float(np.nanmin(self.x)) if self.x.size else 0.0
        x_max = float(np.nanmax(self.x)) if self.x.size else float(len(self.y) - 1)
        for index, value in enumerate(self.y):
            x_value = float(self.x[index]) if index < self.x.size else float(index)
            x = plot.left() + int((x_value - x_min) / max(x_max - x_min, 1e-12) * plot.width())
            y = plot.bottom() - int(float(value) / float(self.y.max()) * plot.height())
            points.append(QPoint(x, y))
        for first, second in zip(points, points[1:]):
            painter.drawLine(first, second)
        painter.setPen(QColor(71, 85, 105))
        for fraction in (0.0, 0.5, 1.0):
            x = plot.left() + int(fraction * plot.width())
            value = x_min + fraction * (x_max - x_min)
            painter.drawLine(x, plot.bottom(), x, plot.bottom() + 4)
            painter.drawText(
                QRect(x - 48, plot.bottom() + 7, 96, 18), Qt.AlignHCenter, f"{value:.4g}"
            )
            y = plot.bottom() - int(fraction * plot.height())
            count = fraction * float(self.y.max())
            painter.drawLine(plot.left() - 4, y, plot.left(), y)
            painter.drawText(
                QRect(4, y - 9, 58, 18), Qt.AlignRight | Qt.AlignVCenter, f"{count:.3g}"
            )
        painter.drawText(
            QRect(plot.left(), self.height() - 24, plot.width(), 18),
            Qt.AlignHCenter,
            "Processed intensity (after enabled pipeline)",
        )
        painter.save()
        painter.translate(16, plot.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRect(-plot.height() // 2, -10, plot.height(), 20), Qt.AlignCenter, "Pixel count"
        )
        painter.restore()


class ParameterCoverageWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.x = np.array([])
        self.y = np.array([])
        self.x_name = "Parameter 1"
        self.y_name = "Parameter 2"
        self.parameter_names = []
        self.ranges: Dict[str, tuple[float, float]] = {}
        self.histogram = np.array([])
        self.histogram_edges = np.array([])
        self.setMinimumHeight(150)

    @staticmethod
    @staticmethod
    def _axis_label(name: str) -> str:
        labels = {
            "radius_nm": "Radius R (nm)",
            "height_nm": "Height h (nm)",
            "length_nm": "Length (nm)",
            "width_nm": "Width (nm)",
            "D_nm": "Spacing D (nm)",
            "sigma_D_ratio": "Paracrystal σ/D",
        }
        return labels.get(name, name.replace("_", " "))

    def set_samples(self, samples, parameter_names=None, parameter_specs=None) -> None:
        if not samples:
            self.x = self.y = np.array([])
            self.histogram = self.histogram_edges = np.array([])
            self.update()
            return
        requested_names = list(samples[0]) if parameter_names is None else list(parameter_names)
        names = [name for name in requested_names if name in samples[0]]
        if not names:
            names = list(samples[0])[:2]
        self.parameter_names = names
        specs = parameter_specs or {}
        self.ranges = {
            name: (
                float(specs.get(name, {}).get("minimum", min(row[name] for row in samples))),
                float(specs.get(name, {}).get("maximum", max(row[name] for row in samples))),
            )
            for name in names
        }
        self.x_name = names[0] if names else "Parameter 1"
        self.y_name = names[1] if len(names) > 1 else self.x_name
        self.x = np.asarray([row[self.x_name] for row in samples], dtype=float)
        self.y = np.asarray([row[self.y_name] for row in samples], dtype=float)
        if len(names) == 1:
            x_range = self.ranges[self.x_name]
            bins = max(5, min(24, int(np.sqrt(max(1, self.x.size))) + 1))
            self.histogram, self.histogram_edges = np.histogram(self.x, bins=bins, range=x_range)
        else:
            self.histogram = self.histogram_edges = np.array([])
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        painter.setPen(QColor(71, 85, 105))
        if not self.x.size:
            painter.drawText(self.rect(), Qt.AlignCenter, "No ground-truth samples yet.")
            return
        dimensions = len(self.parameter_names)
        title = (
            f"Ground truth: 1D distribution of {self._axis_label(self.x_name)}"
            if dimensions == 1
            else f"Ground truth: 2D joint distribution ({self._axis_label(self.x_name)} × {self._axis_label(self.y_name)})"
        )
        if dimensions > 2:
            title += f" · first 2 of {dimensions} form-factor dimensions"
        painter.drawText(12, 20, title)
        plot = self.rect().adjusted(70, 38, -22, -52)
        painter.setPen(QPen(QColor(100, 116, 139), 1))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())

        x_min, x_max = self.ranges.get(self.x_name, (float(self.x.min()), float(self.x.max())))
        if dimensions == 1:
            maximum = max(1.0, float(self.histogram.max()) if self.histogram.size else 1.0)
            width = plot.width() / max(1, len(self.histogram))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(37, 99, 235, 190))
            for index, value in enumerate(self.histogram):
                height = int(float(value) / maximum * plot.height())
                painter.drawRect(
                    QRect(
                        plot.left() + int(index * width) + 1,
                        plot.bottom() - height,
                        max(1, int(width) - 2),
                        height,
                    )
                )
            y_min, y_max = 0.0, maximum
            y_label = "Sample count"
        else:
            y_min, y_max = self.ranges.get(self.y_name, (float(self.y.min()), float(self.y.max())))
            bins = max(6, min(22, int(np.sqrt(max(1, self.x.size))) + 1))
            heat, _, _ = np.histogram2d(
                self.x,
                self.y,
                bins=(bins, bins),
                range=((x_min, x_max), (y_min, y_max)),
            )
            peak = max(1.0, float(heat.max()))
            cell_w = plot.width() / bins
            cell_h = plot.height() / bins
            painter.setPen(Qt.NoPen)
            for ix in range(bins):
                for iy in range(bins):
                    fraction = float(heat[ix, iy]) / peak
                    color = QColor(
                        int(239 - 202 * fraction),
                        int(246 - 147 * fraction),
                        int(255 - 20 * fraction),
                    )
                    painter.setBrush(color)
                    painter.drawRect(
                        QRect(
                            plot.left() + int(ix * cell_w),
                            plot.bottom() - int((iy + 1) * cell_h),
                            max(1, int(cell_w) + 1),
                            max(1, int(cell_h) + 1),
                        )
                    )
            y_label = self._axis_label(self.y_name)

        painter.setPen(QColor(71, 85, 105))
        for fraction in (0.0, 0.5, 1.0):
            x = plot.left() + int(fraction * plot.width())
            x_value = x_min + fraction * (x_max - x_min)
            painter.drawLine(x, plot.bottom(), x, plot.bottom() + 4)
            painter.drawText(
                QRect(x - 46, plot.bottom() + 7, 92, 18), Qt.AlignHCenter, f"{x_value:.4g}"
            )
            y = plot.bottom() - int(fraction * plot.height())
            y_value = y_min + fraction * (y_max - y_min)
            painter.drawLine(plot.left() - 4, y, plot.left(), y)
            painter.drawText(
                QRect(3, y - 9, 60, 18), Qt.AlignRight | Qt.AlignVCenter, f"{y_value:.4g}"
            )
        painter.drawText(
            QRect(plot.left(), self.height() - 25, plot.width(), 18),
            Qt.AlignHCenter,
            self._axis_label(self.x_name),
        )
        painter.save()
        painter.translate(16, plot.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRect(-plot.height() // 2, -10, plot.height(), 20), Qt.AlignCenter, y_label
        )
        painter.restore()
