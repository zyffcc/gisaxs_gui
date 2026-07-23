from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from matplotlib import colormaps
from PyQt5.QtCore import QPoint, QRect, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from trainset.config import PHYSICAL_BACKGROUND_PARAMETERS
from trainset.plugins import REGISTRY


class ArrayCanvas(QWidget):
    region_created = pyqtSignal(str, dict)
    position_changed = pyqtSignal(dict)

    def __init__(self, empty_text: str = "Load a real scattering file to begin", parent: Optional[QWidget] = None):
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
        scale = min(self.width() / max(width, 1), self.height() / max(height, 1))
        draw_width, draw_height = int(width * scale), int(height * scale)
        return QRect((self.width() - draw_width) // 2, (self.height() - draw_height) // 2, draw_width, draw_height)

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
        if self.mode and event.button() == Qt.LeftButton and self._image_rect().contains(event.pos()):
            if self.mode == "beam_center":
                point = self._to_image(event.pos())
                self.region_created.emit("beam_center", {"x": float(point.x()), "y": float(point.y())})
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
                payload = {"type": "rectangle", "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
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
                    normalized = np.clip((transformed - log_low) / max(log_high - log_low, 1e-12), 0.0, 1.0)
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
            qimage = QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format_Grayscale8).copy()
        else:
            rgba = np.ascontiguousarray(colormaps[self.display_colormap](normalized, bytes=True), dtype=np.uint8)
            qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()
        target = self._image_rect()
        painter.drawPixmap(target, QPixmap.fromImage(qimage))
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
            mask_image = QImage(overlay.data, overlay.shape[1], overlay.shape[0], overlay.strides[0], QImage.Format_RGBA8888).copy()
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
            painter.drawText(self.rect(), Qt.AlignCenter, "Run a simulated comparison to calculate this distribution.")
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
            painter.drawText(QRect(x - 48, plot.bottom() + 7, 96, 18), Qt.AlignHCenter, f"{value:.4g}")
            y = plot.bottom() - int(fraction * plot.height())
            count = fraction * float(self.y.max())
            painter.drawLine(plot.left() - 4, y, plot.left(), y)
            painter.drawText(QRect(4, y - 9, 58, 18), Qt.AlignRight | Qt.AlignVCenter, f"{count:.3g}")
        painter.drawText(
            QRect(plot.left(), self.height() - 24, plot.width(), 18),
            Qt.AlignHCenter,
            "Processed intensity (after enabled pipeline)",
        )
        painter.save()
        painter.translate(16, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRect(-plot.height() // 2, -10, plot.height(), 20), Qt.AlignCenter, "Pixel count")
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
            painter.drawText(QRect(x - 46, plot.bottom() + 7, 92, 18), Qt.AlignHCenter, f"{x_value:.4g}")
            y = plot.bottom() - int(fraction * plot.height())
            y_value = y_min + fraction * (y_max - y_min)
            painter.drawLine(plot.left() - 4, y, plot.left(), y)
            painter.drawText(QRect(3, y - 9, 60, 18), Qt.AlignRight | Qt.AlignVCenter, f"{y_value:.4g}")
        painter.drawText(
            QRect(plot.left(), self.height() - 25, plot.width(), 18),
            Qt.AlignHCenter,
            self._axis_label(self.x_name),
        )
        painter.save()
        painter.translate(16, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRect(-plot.height() // 2, -10, plot.height(), 20), Qt.AlignCenter, y_label)
        painter.restore()


class TrainsetBuildPage(QWidget):
    step_changed = pyqtSignal(int)
    mask_region_created = pyqtSignal(str, dict)
    configuration_edited = pyqtSignal()
    what_if_requested = pyqtSignal(dict)

    STEPS = ("Dataset Design", "Local Preview", "Model Design", "Local Run", "Monitor & Results")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fields: Dict[str, QWidget] = {}
        self.preview_canvases: Dict[str, ArrayCanvas] = {}
        self._display_controls: Dict[str, Dict[str, QWidget]] = {}
        self._comparison_details: Dict[str, Any] = {}
        self._comparison_parameter_specs: Dict[str, Any] = {}
        self._parameter_dialog: Optional[QDialog] = None
        self._step_states = ["Not started"] * len(self.STEPS)
        self._design_stage_ready = [False, False, False, False]
        self.setObjectName("freshTrainsetBuildPage")
        self._build()
        self._apply_style()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        header = QGridLayout()
        titles = QVBoxLayout()
        title = QLabel("2D Trainset & Model Workspace")
        title.setObjectName("pageTitle")
        title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        subtitle = QLabel("Design locally · validate with real scattering data · run a lightweight test · export for Maxwell later")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 0, 0, 1, 4)
        self.project_name = QLineEdit("gisaxs_2d_project")
        self.project_name.setPlaceholderText("Project name")
        self.project_name.setMinimumWidth(130)
        self.project_name.setMaximumWidth(220)
        header.addWidget(QLabel("Project"), 1, 0)
        header.addWidget(self.project_name, 1, 1)
        self.validation_badge = QLabel("Not validated")
        self.validation_badge.setObjectName("validationBadge")
        header.addWidget(self.validation_badge, 1, 2, Qt.AlignLeft)
        self.auto_remember_check = QCheckBox("Remember changes")
        self.auto_remember_check.setChecked(True)
        self.auto_remember_check.setToolTip(
            "Automatically save TrainSet settings after edits and restore them at the next GUI launch. "
            "Turn this off to experiment without replacing the remembered configuration."
        )
        self.reset_defaults_button = QPushButton("Reset defaults")
        self.reset_defaults_button.setToolTip(
            "Restore built-in TrainSet defaults. Generated datasets are not deleted."
        )
        header.addWidget(self.auto_remember_check, 2, 0, 1, 2)
        header.addWidget(self.reset_defaults_button, 2, 2)
        header.setColumnStretch(3, 1)
        root.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.step_list = QListWidget()
        self.step_list.setObjectName("trainsetStepList")
        self.step_list.setWordWrap(True)
        self.step_list.setMinimumWidth(176)
        self.step_list.setMaximumWidth(218)
        for number, text in enumerate(self.STEPS, start=1):
            item = QListWidgetItem(f"{number}.  {text}")
            item.setData(Qt.UserRole, number - 1)
            self.step_list.addItem(item)
        self.step_list.currentRowChanged.connect(self._step_selected)
        splitter.addWidget(self.step_list)
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.stack.addWidget(self._dataset_page())
        self.stack.addWidget(self._preview_page())
        self.stack.addWidget(self._model_page())
        self.stack.addWidget(self._hpc_page())
        self.stack.addWidget(self._monitor_page())
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        actions = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.validate_button = QPushButton("Validate")
        self.load_button = QPushButton("Load")
        self.load_button.setToolTip("Load trainset project")
        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("Save trainset project")
        self.preview_button = QPushButton("Preview")
        self.prepare_button = QPushButton("Prepare job")
        self.prepare_button.setToolTip("Prepare a portable local/Maxwell job package")
        self.submit_button = QPushButton("Maxwell")
        self.submit_button.setEnabled(False)
        self.submit_button.setToolTip("SSH submission is intentionally disabled in the local demo. The portable Slurm package interface is retained.")
        for button in (
            self.back_button,
            self.validate_button,
            self.load_button,
            self.save_button,
            self.preview_button,
            self.prepare_button,
            self.submit_button,
        ):
            button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            button.setMinimumWidth(76)
        action_grid = QGridLayout()
        action_grid.setContentsMargins(0, 0, 0, 0)
        action_grid.setHorizontalSpacing(6)
        action_grid.setVerticalSpacing(4)
        action_grid.addWidget(self.validate_button, 0, 0)
        action_grid.addWidget(self.load_button, 0, 1)
        action_grid.addWidget(self.save_button, 0, 2)
        action_grid.addWidget(self.preview_button, 1, 0)
        action_grid.addWidget(self.prepare_button, 1, 1)
        action_grid.addWidget(self.submit_button, 1, 2)
        actions.addWidget(self.back_button)
        actions.addLayout(action_grid)
        actions.addStretch(1)
        root.addLayout(actions)
        self.back_button.clicked.connect(lambda: self.step_list.setCurrentRow(max(0, self.step_list.currentRow() - 1)))
        self.step_list.setCurrentRow(0)

    def set_step_state(self, index: int, state: str) -> None:
        if not 0 <= index < len(self.STEPS):
            return
        self._step_states[index] = state
        item = self.step_list.item(index)
        item.setText(f"{index + 1}.  {self.STEPS[index]}\n{state}")
        item.setToolTip(state)

    def set_design_stage_ready(self, index: int, ready: bool = True) -> None:
        if not 0 <= index < len(self._design_stage_ready):
            return
        self._design_stage_ready[index] = ready
        labels = ("Full detector", "ROI", "Masked image", "Mask only")
        self.design_tabs.setTabText(index, f"{'✓ ' if ready else ''}{labels[index]}")

    def _scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(content)
        return area

    def _spin(self, path: str, value: int, minimum: int = 0, maximum: int = 100000000) -> QSpinBox:
        widget = QSpinBox()
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        widget.setMinimumWidth(72)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        self.fields[path] = widget
        return widget

    def _double(self, path: str, value: float, minimum: float = -1e12, maximum: float = 1e12, decimals: int = 6) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        widget.setMinimumWidth(82)
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        self.fields[path] = widget
        return widget

    def _line(self, path: str, value: str = "") -> QLineEdit:
        widget = QLineEdit(value)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        widget.setMinimumWidth(90)
        self.fields[path] = widget
        return widget

    def _combo(self, path: str, values, current: Optional[str] = None) -> QComboBox:
        widget = QComboBox()
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        widget.setMinimumWidth(90)
        widget.addItems(list(values))
        if current and widget.findText(current) >= 0:
            widget.setCurrentText(current)
        self.fields[path] = widget
        return widget

    def _check(self, path: str, checked: bool = False, text: str = "") -> QCheckBox:
        widget = QCheckBox(text)
        widget.setChecked(checked)
        self.fields[path] = widget
        return widget

    def _make_display_bar(self, key: str) -> QWidget:
        bar = QWidget()
        bar.setProperty("displayBar", True)
        layout = QGridLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(7)
        layout.setVerticalSpacing(5)

        colormap = QComboBox()
        colormap.addItems(("gray", "viridis", "magma", "inferno", "plasma", "cividis", "turbo"))
        log_scale = QCheckBox("Log")
        auto_scale = QCheckBox("Auto")
        auto_scale.setChecked(True)
        vmin = QDoubleSpinBox()
        vmax = QDoubleSpinBox()
        for control, value in ((vmin, 0.0), (vmax, 1.0)):
            control.setRange(-1e30, 1e30)
            control.setDecimals(6)
            control.setValue(value)
            control.setMinimumWidth(92)
            control.setEnabled(False)

        # Keep the image controls on one compact row. This preserves useful
        # canvas height on 720p screens while still fitting narrow workspaces.
        layout.addWidget(QLabel("Colormap"), 0, 0)
        layout.addWidget(colormap, 0, 1)
        layout.addWidget(log_scale, 0, 2)
        layout.addWidget(auto_scale, 0, 3)
        layout.addWidget(QLabel("Vmin"), 0, 4)
        layout.addWidget(vmin, 0, 5)
        layout.addWidget(QLabel("Vmax"), 0, 6)
        layout.addWidget(vmax, 0, 7)
        layout.setColumnStretch(8, 1)

        controls: Dict[str, QWidget] = {
            "colormap": colormap,
            "log": log_scale,
            "auto": auto_scale,
            "vmin": vmin,
            "vmax": vmax,
        }
        self._display_controls[key] = controls
        setattr(self, f"{key}_display_colormap", colormap)
        setattr(self, f"{key}_display_log", log_scale)
        setattr(self, f"{key}_display_auto", auto_scale)
        setattr(self, f"{key}_display_vmin", vmin)
        setattr(self, f"{key}_display_vmax", vmax)

        def apply_display(*_args) -> None:
            automatic = auto_scale.isChecked()
            vmin.setEnabled(not automatic)
            vmax.setEnabled(not automatic)
            self._apply_display_settings(key)

        colormap.currentTextChanged.connect(apply_display)
        log_scale.toggled.connect(apply_display)
        auto_scale.toggled.connect(apply_display)
        vmin.valueChanged.connect(apply_display)
        vmax.valueChanged.connect(apply_display)
        self._apply_display_settings(key)
        return bar

    def _apply_display_settings(self, key: str) -> None:
        controls = self._display_controls.get(key)
        if not controls:
            return
        if key == "design":
            canvases = [
                self.full_detector_canvas,
                self.roi_design_canvas,
                self.masked_design_canvas,
                self.mask_only_canvas,
            ]
        else:
            canvases = list(self.preview_canvases.values())
            for copies in getattr(self, "impact_canvases", {}).values():
                canvases.extend(copies)
        for canvas in canvases:
            canvas.set_display_options(
                controls["colormap"].currentText(),
                controls["log"].isChecked(),
                controls["auto"].isChecked(),
                controls["vmin"].value(),
                controls["vmax"].value(),
            )

    def _dataset_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.dataset_splitter = QSplitter(Qt.Horizontal)
        self.dataset_splitter.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        content = QWidget()
        form_stack = QVBoxLayout(content)
        form_stack.setContentsMargins(0, 0, 8, 0)

        reference = QGroupBox("1 · Real scattering reference")
        ref_form = QGridLayout(reference)
        self.reference_path = self._line("project.reference_file")
        self.reference_button = QPushButton("Load file...")
        ref_form.addWidget(QLabel("Reference file"), 0, 0)
        ref_form.addWidget(self.reference_path, 0, 1)
        ref_form.addWidget(self.reference_button, 1, 1)
        reference_help = QLabel(
            "This image is geometry guidance only: use it to select beam center, ROI, thresholds and fixed masks. "
            "Training images and previews are generated by BornAgain."
        )
        reference_help.setWordWrap(True)
        ref_form.addWidget(reference_help, 2, 0, 1, 2)
        ref_form.setColumnStretch(1, 1)
        form_stack.addWidget(reference)

        beam_detector = QGroupBox("2 · Beam and detector geometry")
        grid = QGridLayout(beam_detector)
        entries = (
            ("Wavelength (nm)", self._double("beam.wavelength_nm", 0.105, 1e-5, 10.0)),
            ("Grazing angle (°)", self._double("beam.grazing_angle_deg", 0.4, -10.0, 10.0)),
            ("Detector preset", self._combo("detector.preset", ("Custom", "PILATUS3 X 2M", "EIGER2 X 4M"), "Custom")),
            ("Distance (mm)", self._double("detector.distance_mm", 3230.0, 1.0, 1e6, 3)),
            ("Pixels X", self._spin("detector.pixels_x", 1475, 1)),
            ("Pixels Y", self._spin("detector.pixels_y", 1679, 1)),
            ("Pixel X (mm)", self._double("detector.pixel_size_x_mm", 0.172, 1e-6, 100.0)),
            ("Pixel Y (mm)", self._double("detector.pixel_size_y_mm", 0.172, 1e-6, 100.0)),
            ("Beam center X (px)", self._double("detector.beam_center_x_px", 804.0, -1e6, 1e6, 2)),
            ("Beam center Y (px)", self._double("detector.beam_center_y_px", 305.0, -1e6, 1e6, 2)),
        )
        for row, (label, widget) in enumerate(entries):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        self.pick_beam_center_button = QPushButton("Pick beam center on detector")
        self.beam_cursor_label = QLabel("Move over the full detector to inspect x, y and intensity")
        self.beam_cursor_label.setWordWrap(True)
        grid.addWidget(self.pick_beam_center_button, len(entries), 0, 1, 2)
        grid.addWidget(self.beam_cursor_label, len(entries) + 1, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        form_stack.addWidget(beam_detector)

        roi_group = QGroupBox("3 · ROI and simulation angular range")
        roi_grid = QGridLayout(roi_group)
        for row, (label, path, value) in enumerate((
            ("X", "roi.x", 600), ("Y", "roi.y", 180), ("Width", "roi.width", 256), ("Height", "roi.height", 256)
        )):
            roi_grid.addWidget(QLabel(label), row, 0)
            roi_grid.addWidget(self._spin(path, value, 0), row, 1)
        self.draw_roi_button = QPushButton("Draw rectangle ROI")
        self.roi_range_label = QLabel("phi / alpha range will be calculated from the detector geometry")
        self.roi_range_label.setWordWrap(True)
        roi_grid.addWidget(self.draw_roi_button, 4, 0, 1, 2)
        roi_help = QLabel("ROI is rectangular so tensor shape and angular limits stay explicit.")
        roi_help.setWordWrap(True)
        roi_grid.addWidget(roi_help, 5, 0, 1, 2)
        roi_grid.addWidget(self.roi_range_label, 6, 0, 1, 2)
        roi_grid.setColumnStretch(1, 1)
        form_stack.addWidget(roi_group)

        mask_group = QGroupBox("4 · Mask design")
        mask_grid = QGridLayout(mask_group)
        mask_grid.addWidget(QLabel("Mode"), 0, 0)
        mask_grid.addWidget(self._combo("mask.mode", ("fixed", "random"), "fixed"), 0, 1)
        threshold_toggle = self._check(
            "mask.threshold.enabled",
            True,
            "Mask detector defects from reference intensity",
        )
        threshold_toggle.setToolTip(
            "Converts out-of-range pixels in the experimental ROI into a fixed spatial mask. "
            "The same gap/hot-pixel locations are then applied to every BornAgain simulation."
        )
        mask_grid.addWidget(threshold_toggle, 1, 0, 1, 2)
        threshold_min_label = QLabel("Reference keep ≥")
        threshold_min_label.setToolTip("Reference ROI pixels below this value become masked detector locations.")
        threshold_max_label = QLabel("Reference keep ≤")
        threshold_max_label.setToolTip("Reference ROI pixels above this value become masked detector locations.")
        threshold_min = self._double("mask.threshold.minimum", 0.0)
        threshold_max = self._double("mask.threshold.maximum", 1e12)
        threshold_min.setToolTip("Lower inclusive reference intensity. A value of 0 masks CBF gap values such as -1 and -2.")
        threshold_max.setToolTip("Upper inclusive reference intensity used to locate saturated/hot pixels.")
        self.auto_reference_threshold_check = self._check(
            "mask.threshold.auto_reference_upper",
            True,
            "Auto-detect hot-pixel upper limit",
        )
        self.auto_reference_threshold_check.setToolTip(
            "After loading/changing the ROI, use the 99.999th finite-intensity percentile as the upper limit. "
            "This normally isolates a few extreme detector pixels while retaining the scattering pattern."
        )
        self.threshold_summary = QLabel("Load a reference to calculate detector-gap and hot-pixel locations.")
        self.threshold_summary.setWordWrap(True)
        self.threshold_summary.setProperty("cardBody", True)
        threshold_toggle.toggled.connect(threshold_min.setEnabled)
        mask_grid.addWidget(threshold_min_label, 2, 0)
        mask_grid.addWidget(threshold_min, 2, 1)
        mask_grid.addWidget(threshold_max_label, 3, 0)
        mask_grid.addWidget(threshold_max, 3, 1)
        mask_grid.addWidget(self.auto_reference_threshold_check, 4, 0, 1, 2)
        mask_grid.addWidget(self.threshold_summary, 5, 0, 1, 2)
        self.draw_rectangle_button = QPushButton("Add rectangle")
        self.draw_circle_button = QPushButton("Add elliptical mask")
        self.remove_mask_button = QPushButton("Remove selected")
        self.clear_masks_button = QPushButton("Clear all shapes")
        mask_actions = QGridLayout()
        mask_actions.addWidget(self.draw_rectangle_button, 0, 0)
        mask_actions.addWidget(self.draw_circle_button, 0, 1)
        mask_actions.addWidget(self.remove_mask_button, 1, 0)
        mask_actions.addWidget(self.clear_masks_button, 1, 1)
        mask_grid.addLayout(mask_actions, 6, 0, 1, 2)
        self.mask_shape_table = QTableWidget(0, 5)
        self.mask_shape_table.setHorizontalHeaderLabels(("Type", "X / CX", "Y / CY", "Width / Radius X", "Height / Radius Y"))
        self.mask_shape_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mask_shape_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mask_shape_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mask_shape_table.setMaximumHeight(135)
        mask_grid.addWidget(self.mask_shape_table, 7, 0, 1, 2)
        self.random_mask_panel = QWidget()
        random_grid = QGridLayout(self.random_mask_panel)
        random_grid.setContentsMargins(0, 4, 0, 0)
        random_definitions = (
            (
                "Vertical bars",
                "mask.random.vertical_bars",
                2,
                "Random full-height masked columns, representing module gaps or dead detector strips.",
            ),
            (
                "Horizontal bars",
                "mask.random.horizontal_bars",
                1,
                "Random full-width masked rows, representing detector gaps or invalid readout bands.",
            ),
            (
                "Circles",
                "mask.random.circles",
                1,
                "Random circular dead spots placed inside the ROI.",
            ),
        )
        for row, (label, path, value, help_text) in enumerate(random_definitions):
            label_widget = QLabel(label)
            control = self._spin(path, value, 0, 100)
            label_widget.setToolTip(help_text)
            control.setToolTip(help_text)
            random_grid.addWidget(label_widget, row, 0)
            random_grid.addWidget(control, row, 1)
        for row, (label, path, value, minimum, maximum, help_text) in enumerate(
            (
                ("Bar width min (px)", "mask.random.bar_width_min", 2, 1, 1000, "Smallest random bar thickness in ROI pixels."),
                ("Bar width max (px)", "mask.random.bar_width_max", 6, 1, 1000, "Largest random bar thickness in ROI pixels."),
                ("Circle radius min (px)", "mask.random.circle_radius_min", 4, 1, 1000, "Smallest random dead-spot radius."),
                ("Circle radius max (px)", "mask.random.circle_radius_max", 12, 1, 1000, "Largest random dead-spot radius."),
            ),
            start=3,
        ):
            label_widget = QLabel(label)
            control = self._spin(path, value, minimum, maximum)
            label_widget.setToolTip(help_text)
            control.setToolTip(help_text)
            random_grid.addWidget(label_widget, row, 0)
            random_grid.addWidget(control, row, 1)
        beamstop = self._check("mask.random.beamstop", True, "Beamstop + support arm")
        beamstop.setToolTip("Masks a central circular beamstop and one randomly directed horizontal support arm.")
        random_grid.addWidget(beamstop, 7, 0, 1, 2)
        self.random_mask_preview_button = QPushButton("New random mask example")
        self.random_mask_preview_button.setToolTip(
            "Draw a fresh unseeded example in Masked image and Mask only. Dataset generation remains reproducible from the project seed."
        )
        random_grid.addWidget(self.random_mask_preview_button, 8, 0, 1, 2)
        mask_grid.addWidget(self.random_mask_panel, 8, 0, 1, 2)
        mask_grid.setColumnStretch(1, 1)
        form_stack.addWidget(mask_group)

        form_factor = QGroupBox("5 · Particle form factor and population")
        sample_grid = QGridLayout(form_factor)
        particle_labels = [item.label for item in REGISTRY.list("particle")]
        sample_grid.addWidget(QLabel("Particle shape"), 0, 0)
        self.particle_combo = self._combo("sample.particle_label", particle_labels, "Spherical segment")
        sample_grid.addWidget(self.particle_combo, 0, 1)
        sample_grid.addWidget(QLabel("Particle material"), 1, 0)
        sample_grid.addWidget(self._combo("sample.particle_material", ("Copper", "Gold", "Silicon", "Polymer"), "Copper"), 1, 1)
        self.particle_help = QLabel("Choose a shape first; only parameters used by that form factor are shown below.")
        self.particle_help.setWordWrap(True)
        sample_grid.addWidget(self.particle_help, 2, 0, 1, 2)
        self.particle_parameter_table = QTableWidget(0, 5)
        self.particle_parameter_table.setHorizontalHeaderLabels(("Parameter", "Distribution", "Minimum", "Maximum", "Meaning / unit"))
        self.particle_parameter_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.particle_parameter_table.setMaximumHeight(150)
        sample_grid.addWidget(self.particle_parameter_table, 3, 0, 1, 2)
        sample_grid.addWidget(QLabel("Population model"), 4, 0)
        sample_grid.addWidget(self._combo("sample.mixture.mode", ("single", "gaussian_mixture"), "gaussian_mixture"), 4, 1)
        sample_grid.addWidget(QLabel("Gaussian components"), 5, 0)
        sample_grid.addWidget(self._spin("sample.mixture.components", 5, 1, 20), 5, 1)
        width_help = (
            "Each Gaussian component width is this fraction of the full selected particle-parameter range. "
            "Example: 0.10 with R = 2…12 nm gives σ = 1 nm."
        )
        width_min_label = QLabel("Component width fraction min")
        width_min_control = self._double("sample.mixture.sigma_fraction_min", 0.01, 0.0, 1.0, 3)
        width_max_label = QLabel("Component width fraction max")
        width_max_control = self._double("sample.mixture.sigma_fraction_max", 0.30, 0.0, 1.0, 3)
        for widget in (width_min_label, width_min_control, width_max_label, width_max_control):
            widget.setToolTip(width_help)
        sample_grid.addWidget(width_min_label, 6, 0)
        sample_grid.addWidget(width_min_control, 6, 1)
        sample_grid.addWidget(width_max_label, 7, 0)
        sample_grid.addWidget(width_max_control, 7, 1)
        random_weights = self._check("sample.mixture.random_weights", True, "Random component weights")
        random_weights.setToolTip(
            "On: draw non-negative component contributions that sum to 1 for every sample. Off: all Gaussian components contribute equally."
        )
        sample_grid.addWidget(random_weights, 8, 0, 1, 2)
        sample_grid.addWidget(QLabel("Surface density (nm⁻²)"), 9, 0)
        sample_grid.addWidget(self._double("sample.surface_density_per_nm2", 0.01, 0.0, 1e6, 6), 9, 1)
        self.segment_constraint_check = self._check("sample.constraints.segment_height_le_2r", True, "Enforce h ≤ 2R for spherical segments")
        sample_grid.addWidget(self.segment_constraint_check, 10, 0, 1, 2)
        sample_grid.setColumnStretch(1, 1)
        form_stack.addWidget(form_factor)

        structure_factor = QGroupBox("6 · Interference / structure factor")
        structure_grid = QGridLayout(structure_factor)
        interference_labels = [item.label for item in REGISTRY.list("interference")]
        structure_grid.addWidget(QLabel("Interference model"), 0, 0)
        self.interference_combo = self._combo("sample.interference_label", interference_labels, "Paracrystal")
        structure_grid.addWidget(self.interference_combo, 0, 1)
        self.interference_help = QLabel("The selected structure factor defines its own parameters.")
        self.interference_help.setWordWrap(True)
        structure_grid.addWidget(self.interference_help, 1, 0, 1, 2)
        self.interference_parameter_table = QTableWidget(0, 5)
        self.interference_parameter_table.setHorizontalHeaderLabels(("Parameter", "Distribution", "Minimum", "Maximum", "Meaning / unit"))
        self.interference_parameter_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.interference_parameter_table.setMaximumHeight(135)
        structure_grid.addWidget(self.interference_parameter_table, 2, 0, 1, 2)
        self.spacing_constraint_check = self._check("sample.constraints.interparticle_spacing_gt_2r", True, "Enforce D > 2R (no particle overlap)")
        structure_grid.addWidget(self.spacing_constraint_check, 3, 0, 1, 2)
        structure_grid.setColumnStretch(1, 1)
        form_stack.addWidget(structure_factor)

        layer_group = QGroupBox("7 · Layers and substrate")
        layer_grid = QGridLayout(layer_group)
        layer_help = QLabel("Use equal min/max for a fixed value; use different values to sample a training range.")
        layer_help.setWordWrap(True)
        layer_grid.addWidget(layer_help, 0, 0, 1, 2)
        self.layer_table = QTableWidget(2, 6)
        self.layer_table.setHorizontalHeaderLabels(("Enabled", "Material", "Thickness min", "Thickness max", "Roughness min", "Roughness max"))
        for row, values in enumerate((("1", "Copper", "20.0", "20.0", "0.0", "0.0"), ("1", "Polymer", "50.0", "50.0", "0.0", "0.0"))):
            for column, value in enumerate(values):
                self.layer_table.setItem(row, column, QTableWidgetItem(value))
        self.layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layer_table.setMaximumHeight(140)
        layer_grid.addWidget(self.layer_table, 1, 0, 1, 2)
        layer_actions = QHBoxLayout()
        self.add_layer_button = QPushButton("Add layer")
        self.remove_layer_button = QPushButton("Remove selected")
        layer_actions.addWidget(self.add_layer_button)
        layer_actions.addWidget(self.remove_layer_button)
        layer_actions.addStretch(1)
        layer_grid.addLayout(layer_actions, 2, 0, 1, 2)
        layer_grid.addWidget(QLabel("Substrate"), 3, 0)
        layer_grid.addWidget(self._combo("sample.substrate.material", ("Silicon", "Copper", "Gold", "Polymer"), "Silicon"), 3, 1)
        layer_grid.addWidget(QLabel("Substrate roughness (nm)"), 4, 0)
        layer_grid.addWidget(self._double("sample.substrate.roughness_nm", 0.0, 0.0, 1e6), 4, 1)
        layer_grid.setColumnStretch(1, 1)
        self.add_layer_button.clicked.connect(self._add_layer_row)
        self.remove_layer_button.clicked.connect(lambda: self._remove_selected_rows(self.layer_table))
        form_stack.addWidget(layer_group)

        dataset_group = QGroupBox("8 · Dataset sampling, files and split")
        dataset_grid = QGridLayout(dataset_group)
        dataset_grid.addWidget(QLabel("Sampling"), 0, 0)
        dataset_grid.addWidget(self._combo("dataset.sampling", ("latin_hypercube", "uniform", "log_uniform", "grid"), "latin_hypercube"), 0, 1)
        dataset_grid.addWidget(QLabel("Samples"), 1, 0)
        dataset_grid.addWidget(self._spin("dataset.number_of_samples", 200000, 1, 1000000000), 1, 1)
        shard_label = QLabel("Samples per output file")
        shard_label.setToolTip("A shard is one HDF5 dataset file. Multiple smaller files are easier to generate in parallel and resume later.")
        dataset_grid.addWidget(shard_label, 2, 0)
        dataset_grid.addWidget(self._spin("dataset.samples_per_shard", 2000, 1, 10000000), 2, 1)
        shard_help = QLabel("One shard = one HDF5 file. This controls file size, not how samples are shared between train/validation/test.")
        shard_help.setWordWrap(True)
        dataset_grid.addWidget(shard_help, 3, 0, 1, 2)
        for row, (name, value) in enumerate((("Train", 0.8), ("Validation", 0.1), ("Test", 0.1)), start=4):
            dataset_grid.addWidget(QLabel(name), row, 0)
            dataset_grid.addWidget(self._double(f"dataset.split.{name.lower()}", value, 0.0, 1.0, 3), row, 1)
        dataset_grid.setColumnStretch(1, 1)
        form_stack.addWidget(dataset_group)
        form_stack.addStretch(1)

        self.dataset_splitter.addWidget(self._scroll(content))
        preview_side = QVBoxLayout()
        preview_title = QLabel("Design preview")
        preview_title.setProperty("sectionTitle", True)
        preview_hint = QLabel("Hover for detector coordinates and intensity. Pick the beam center, then draw a rectangular ROI and inspect its mask.")
        preview_hint.setWordWrap(True)
        preview_hint.setProperty("cardBody", True)
        preview_side.addWidget(preview_title)
        preview_side.addWidget(preview_hint)
        self.design_tabs = QTabWidget()
        self.design_tabs.setObjectName("designStageTabs")
        self.design_tabs.tabBar().setExpanding(True)
        self.design_tabs.tabBar().setUsesScrollButtons(False)
        self.design_tabs.tabBar().setElideMode(Qt.ElideRight)
        self.full_detector_canvas = ArrayCanvas("Load a real scattering file to begin")
        self.roi_design_canvas = ArrayCanvas("Define an ROI to inspect the cropped detector region")
        self.masked_design_canvas = ArrayCanvas("Configure a mask to see it overlaid on the ROI")
        self.mask_only_canvas = ArrayCanvas("The binary mask will appear here (white = masked)")
        self.full_detector_canvas.region_created.connect(self.mask_region_created)
        self.roi_design_canvas.region_created.connect(self.mask_region_created)
        self.full_detector_canvas.position_changed.connect(
            lambda position: self.beam_cursor_label.setText(
                f"x={position['x']} px · y={position['y']} px · I={position['intensity']:.6g}"
            )
        )
        for label, canvas in (
            ("Full detector", self.full_detector_canvas),
            ("ROI", self.roi_design_canvas),
            ("Masked image", self.masked_design_canvas),
            ("Mask only", self.mask_only_canvas),
        ):
            self.design_tabs.addTab(canvas, label)
        self.design_canvas = self.full_detector_canvas
        preview_side.addWidget(self._make_display_bar("design"))
        preview_side.addWidget(self.design_tabs, 1)
        self.design_info = QLabel("No reference loaded")
        self.design_info.setWordWrap(True)
        self.design_info.setProperty("infoPanel", True)
        preview_side.addWidget(self.design_info)
        side = QWidget()
        side.setObjectName("designPreviewCard")
        side.setLayout(preview_side)
        self.dataset_splitter.addWidget(side)
        self.dataset_splitter.setStretchFactor(0, 7)
        self.dataset_splitter.setStretchFactor(1, 3)
        self.dataset_splitter.setSizes((720, 420))
        self._dataset_splitter_orientation = Qt.Horizontal
        layout.addWidget(self.dataset_splitter)
        return page

    def _preview_page(self) -> QWidget:
        page = QWidget()
        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        layout = QVBoxLayout(content)
        intro = QLabel(
            "Simulation-first preview: BornAgain creates the scattering pattern. Choose any sampled physics, background or noise range "
            "to compare its minimum, midpoint and maximum without using the experimental image as training data."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        orientation_hint = QLabel(
            "Orientation: detector x increases to the right and detector y downward; exit angle/qz is higher at the top. "
            "BornAgain is flipped vertically once to match the experimental detector display."
        )
        orientation_hint.setWordWrap(True)
        orientation_hint.setProperty("infoPanel", True)
        layout.addWidget(orientation_hint)

        controls = QVBoxLayout()
        selection_controls = QGridLayout()
        selection_controls.addWidget(QLabel("Compare range"), 0, 0)
        self.impact_parameter_combo = QComboBox()
        self.impact_parameter_combo.setMinimumWidth(180)
        self.impact_parameter_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        selection_controls.addWidget(self.impact_parameter_combo, 0, 1)
        selection_controls.addWidget(QLabel("Coverage samples"), 1, 0)
        self.preview_count = QSpinBox()
        self.preview_count.setRange(3, 1000)
        self.preview_count.setValue(16)
        self.preview_count.setToolTip("Number of label samples drawn for the parameter-coverage diagnostic; it does not rerun BornAgain.")
        selection_controls.addWidget(self.preview_count, 1, 1)
        selection_controls.setColumnStretch(1, 1)
        controls.addLayout(selection_controls)

        action_controls = QGridLayout()
        self.generate_preview_button = QPushButton("Update simulated comparison")
        self.generate_preview_button.setObjectName("primaryAction")
        self.generate_preview_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.generate_preview_button.setMinimumWidth(190)
        self.generate_preview_button.setMinimumHeight(38)
        action_controls.addWidget(self.generate_preview_button, 0, 0, 1, 2)
        self.force_simulation_button = QPushButton("Recompute BornAgain")
        self.force_simulation_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.force_simulation_button.setMinimumWidth(160)
        self.force_simulation_button.setMinimumHeight(38)
        self.force_simulation_button.setToolTip("Clear the in-memory physics cache and explicitly rerun BornAgain.")
        action_controls.addWidget(self.force_simulation_button, 1, 0)
        self.new_realization_button = QPushButton("New noise / mask realization")
        self.new_realization_button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.new_realization_button.setMinimumWidth(180)
        self.new_realization_button.setMinimumHeight(38)
        self.new_realization_button.setToolTip("Keep cached BornAgain images, but draw fresh stochastic background, noise, mask and edge-crop values.")
        action_controls.addWidget(self.new_realization_button, 1, 1)
        self.preview_parameters_button = QPushButton("View parameters used…")
        self.preview_parameters_button.setEnabled(False)
        self.preview_parameters_button.setToolTip(
            "Open a roomy read-only window showing the exact physics, preprocessing realization, beam, detector and ROI values for all three images."
        )
        self.preview_parameters_button.clicked.connect(self.show_comparison_parameters)
        action_controls.addWidget(self.preview_parameters_button, 2, 0, 1, 2)
        action_controls.setColumnStretch(0, 1)
        action_controls.setColumnStretch(1, 1)
        controls.addLayout(action_controls)
        self.preview_cache_status = QLabel("BornAgain cache: empty")
        self.preview_cache_status.setWordWrap(True)
        controls.addWidget(self.preview_cache_status)
        self.preview_progress = QProgressBar()
        self.preview_progress.setRange(0, 100)
        self.preview_progress.setValue(0)
        self.preview_progress.setTextVisible(True)
        self.preview_progress.setVisible(False)
        controls.addWidget(self.preview_progress)
        self.preview_activity = QLabel("")
        self.preview_activity.setWordWrap(True)
        self.preview_activity.setVisible(False)
        controls.addWidget(self.preview_activity)
        layout.addLayout(controls)

        # Kept as an internal compatibility field for local-run code. Local
        # Preview itself is always simulated.
        self.preview_mode = QComboBox()
        self.preview_mode.addItem("Simulated impact")

        self.preprocessing_tabs = QTabWidget()
        self.preprocessing_tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        background_page = QWidget()
        background_layout = QVBoxLayout(background_page)
        background_enable = self._check("pre.background.enabled", False, "Add physical background")
        background_enable.setToolTip(
            "Synthetic GISAXS background based on the former Yuxin_train model. Every component can be ranged and inspected below."
        )
        background_layout.addWidget(background_enable)
        background_groups = (
            (
                "General",
                {
                    "target_fraction",
                    "constant_fraction",
                    "plane_qy_slope",
                    "plane_qz_slope",
                    "low_qz_cut_fraction",
                    "blur_sigma_px",
                },
            ),
            (
                "Specular ridge",
                {
                    "specular_amplitude",
                    "specular_width_fraction",
                    "specular_widening",
                    "specular_decay_fraction",
                },
            ),
            (
                "Yoneda band",
                {
                    "yoneda_amplitude",
                    "yoneda_center_fraction",
                    "yoneda_width_fraction",
                    "yoneda_center_hole",
                },
            ),
            (
                "Diffuse wedge",
                {
                    "wedge_amplitude",
                    "wedge_anisotropy",
                    "wedge_porod_exponent",
                    "wedge_rg_fraction",
                },
            ),
        )
        background_component_tabs = QTabWidget()
        self.background_component_tabs = background_component_tabs
        self.background_parameter_tables = []
        for group_name, keys in background_groups:
            definitions = [item for item in PHYSICAL_BACKGROUND_PARAMETERS if item["key"] in keys]
            background_table = QTableWidget(len(definitions), 3)
            background_table.setHorizontalHeaderLabels(("Parameter", "Minimum", "Maximum"))
            background_table.verticalHeader().setVisible(False)
            background_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            background_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
            background_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
            background_table.setColumnWidth(1, 126)
            background_table.setColumnWidth(2, 126)
            for row, definition in enumerate(definitions):
                label = QTableWidgetItem(str(definition["label"]))
                label.setToolTip(str(definition["help"]))
                label.setFlags(label.flags() & ~Qt.ItemIsEditable)
                background_table.setItem(row, 0, label)
                key = str(definition["key"])
                minimum = self._double(
                    f"pre.background.{key}.min",
                    float(definition["minimum"]),
                    -1e6,
                    1e6,
                    int(definition["decimals"]),
                )
                maximum = self._double(
                    f"pre.background.{key}.max",
                    float(definition["maximum"]),
                    -1e6,
                    1e6,
                    int(definition["decimals"]),
                )
                minimum.setToolTip(str(definition["help"]))
                maximum.setToolTip(str(definition["help"]))
                background_table.setCellWidget(row, 1, minimum)
                background_table.setCellWidget(row, 2, maximum)
            visible_rows = max(4, len(definitions))
            background_table.setMinimumHeight(34 + visible_rows * 31)
            self.background_parameter_tables.append(background_table)
            background_component_tabs.addTab(background_table, group_name)
        background_layout.addWidget(background_component_tabs)
        self.background_parameter_table = self.background_parameter_tables[0]
        self.preprocessing_tabs.addTab(background_page, "Physical background")

        noise_page = QWidget()
        noise_layout = QGridLayout(noise_page)
        gaussian = self._check("pre.gaussian.enabled", True, "Gaussian readout noise")
        gaussian.setToolTip("Adds zero-mean Gaussian noise. Lower SNR means stronger noise; SNR is computed from mean signal power.")
        noise_layout.addWidget(gaussian, 0, 0, 1, 3)
        snr_label = QLabel("SNR range (dB)")
        snr_label.setToolTip("Signal-to-noise ratio in dB. Lower values add more Gaussian noise.")
        noise_layout.addWidget(snr_label, 1, 0)
        noise_layout.addWidget(self._double("pre.gaussian.min", 80.0, -100.0, 300.0, 2), 1, 1)
        noise_layout.addWidget(self._double("pre.gaussian.max", 110.0, -100.0, 300.0, 2), 1, 2)
        poisson = self._check("pre.poisson.enabled", False, "Poisson photon-count noise")
        poisson.setToolTip("Converts intensity to expected photon counts, draws Poisson counts, then converts back.")
        noise_layout.addWidget(poisson, 2, 0, 1, 3)
        poisson_label = QLabel("Photon-count scale")
        poisson_help = (
            "Intensity multiplier before Poisson sampling. Low scale means few counts and stronger relative shot noise; "
            "high scale means more counts and weaker relative noise."
        )
        poisson_label.setToolTip(poisson_help)
        poisson_min = self._double("pre.poisson.min", 1.0, 1e-6, 1e9, 3)
        poisson_max = self._double("pre.poisson.max", 20.0, 1e-6, 1e9, 3)
        poisson_min.setToolTip(poisson_help)
        poisson_max.setToolTip(poisson_help)
        noise_layout.addWidget(poisson_label, 3, 0)
        noise_layout.addWidget(poisson_min, 3, 1)
        noise_layout.addWidget(poisson_max, 3, 2)
        independent_help = QLabel("Gaussian and Poisson are independent: enable neither, either one, or both. Applied order: Gaussian → Poisson.")
        independent_help.setWordWrap(True)
        noise_layout.addWidget(independent_help, 4, 0, 1, 3)
        noise_layout.setColumnStretch(1, 1)
        noise_layout.setColumnStretch(2, 1)
        self.preprocessing_tabs.addTab(noise_page, "Noise")

        transform_page = QWidget()
        chain = QGridLayout(transform_page)
        chain.addWidget(self._check("pre.mask.enabled", True), 0, 0)
        mask_stage_label = QLabel("1. Apply configured detector mask")
        mask_stage_label.setWordWrap(True)
        mask_stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chain.addWidget(mask_stage_label, 0, 1, 1, 3)
        chain.addWidget(self._check("pre.log.enabled", True), 1, 0)
        log_stage_label = QLabel("2. Log transform compresses scattering dynamic range")
        log_stage_label.setWordWrap(True)
        log_stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chain.addWidget(log_stage_label, 1, 1, 1, 3)
        chain.addWidget(self._check("pre.normalize.enabled", True), 2, 0)
        normalize_stage_label = QLabel("3. Normalize valid pixels")
        normalize_stage_label.setWordWrap(True)
        normalize_stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chain.addWidget(normalize_stage_label, 2, 1)
        chain.addWidget(self._combo("pre.normalize.mode", ("range", "upper", "lower"), "range"), 2, 2)
        normalize_bounds = QHBoxLayout()
        normalize_bounds.addWidget(self._double("pre.normalize.lower", 0.0))
        normalize_bounds.addWidget(self._double("pre.normalize.upper", 1.0))
        chain.addLayout(normalize_bounds, 2, 3)
        chain.addWidget(self._check("pre.edge.enabled", False), 3, 0)
        crop_stage_label = QLabel("4. Random edge crop then resize back")
        crop_stage_label.setWordWrap(True)
        crop_stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        chain.addWidget(crop_stage_label, 3, 1)
        chain.addWidget(QLabel("Maximum px"), 3, 2)
        chain.addWidget(self._spin("pre.edge.maximum", 4, 0, 128), 3, 3)
        transform_help = QLabel("Each enabled stage is shown separately in the Pipeline stages view.")
        transform_help.setWordWrap(True)
        chain.addWidget(transform_help, 4, 0, 1, 4)
        chain.setColumnStretch(1, 1)
        self.preprocessing_tabs.addTab(transform_page, "Mask & transforms")
        self.preprocessing_tabs.setMinimumHeight(330)
        layout.addWidget(self.preprocessing_tabs)

        self.preview_views = QTabWidget()
        self.preview_views.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        comparison_page = QWidget()
        comparison_layout = QVBoxLayout(comparison_page)
        self.impact_canvases: Dict[str, list[ArrayCanvas]] = {"minimum": [], "midpoint": [], "maximum": []}
        self.impact_value_labels: Dict[str, list[QLabel]] = {"minimum": [], "midpoint": [], "maximum": []}
        self.impact_responsive_stack = QStackedWidget()
        self.impact_responsive_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        wide = QWidget()
        wide_layout = QHBoxLayout(wide)
        for key, heading in (("minimum", "Minimum"), ("midpoint", "Midpoint"), ("maximum", "Maximum")):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            value_label = QLabel(heading)
            value_label.setAlignment(Qt.AlignCenter)
            canvas = ArrayCanvas(f"{heading} simulated result")
            canvas.setMinimumSize(180, 210)
            panel_layout.addWidget(value_label)
            panel_layout.addWidget(canvas, 1)
            wide_layout.addWidget(panel, 1)
            self.impact_canvases[key].append(canvas)
            self.impact_value_labels[key].append(value_label)
        self.impact_responsive_stack.addWidget(wide)
        compact = QTabWidget()
        for key, heading in (("minimum", "Minimum"), ("midpoint", "Midpoint"), ("maximum", "Maximum")):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            value_label = QLabel(heading)
            value_label.setAlignment(Qt.AlignCenter)
            canvas = ArrayCanvas(f"{heading} simulated result")
            canvas.setMinimumSize(180, 210)
            panel_layout.addWidget(value_label)
            panel_layout.addWidget(canvas, 1)
            compact.addTab(panel, heading)
            self.impact_canvases[key].append(canvas)
            self.impact_value_labels[key].append(value_label)
        self.impact_responsive_stack.addWidget(compact)
        comparison_layout.addWidget(self._make_display_bar("preview"))
        comparison_layout.addWidget(self.impact_responsive_stack, 1)
        self.preview_views.addTab(comparison_page, "Range impact")

        pipeline_page = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_page)
        self.preview_tabs = QTabWidget()
        placeholder = QLabel(
            "Run Update simulated comparison. Only BornAgain Raw and the preprocessing stages you enabled will appear here, in execution order."
        )
        placeholder.setWordWrap(True)
        placeholder.setAlignment(Qt.AlignCenter)
        self.preview_tabs.addTab(placeholder, "No result yet")
        pipeline_layout.addWidget(self.preview_tabs)
        self.preview_views.addTab(pipeline_page, "Pipeline stages")

        diagnostics = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics)
        self.preview_stats = QLabel("Update the simulated comparison to inspect cache use, tensor shape and dynamic range.")
        self.preview_stats.setWordWrap(True)
        self.preview_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.diagnostic_tabs = QTabWidget()
        self.histogram = HistogramWidget()
        self.parameter_coverage = ParameterCoverageWidget()
        self.diagnostic_tabs.addTab(self.histogram, "Intensity")
        self.diagnostic_tabs.addTab(self.parameter_coverage, "Ground-truth distribution")
        self.preview_gate_table = QTableWidget(4, 2)
        self.preview_gate_table.setHorizontalHeaderLabels(("Local readiness check", "State"))
        for row, gate in enumerate(("Configuration valid", "Local samples generated", "Tensor shapes compatible", "Storage estimate accepted")):
            self.preview_gate_table.setItem(row, 0, QTableWidgetItem(gate))
            self.preview_gate_table.setItem(row, 1, QTableWidgetItem("Pending"))
        self.preview_gate_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_gate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.storage_accept_check = QCheckBox("Storage estimate reviewed")
        self.storage_accept_check.setToolTip("Confirm that you reviewed the estimated local storage for the configured full dataset.")
        diagnostics_layout.addWidget(self.preview_stats)
        diagnostics_layout.addWidget(self.diagnostic_tabs, 1)
        diagnostics_layout.addWidget(self.storage_accept_check)
        diagnostics_layout.addWidget(self.preview_gate_table)
        self.preview_views.addTab(diagnostics, "Diagnostics")
        layout.addWidget(self.preview_views, 1)
        self.preview_capability = self.preview_cache_status
        outer_layout.addWidget(self._scroll(content))
        return page

    def _model_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel("Build the feature extractor row by row. The regression output size is added automatically from the variable physics parameters.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.model_layer_table = QTableWidget(0, 5)
        self.model_layer_table.setHorizontalHeaderLabels(("Layer type", "Filters / units", "Kernel / pool", "Activation", "Dropout rate"))
        self.model_layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.model_layer_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.model_layer_table.setMinimumHeight(280)
        layout.addWidget(self.model_layer_table, 2)
        layer_actions = QHBoxLayout()
        self.add_model_layer_button = QPushButton("Add layer")
        self.remove_model_layer_button = QPushButton("Remove selected")
        self.move_model_layer_up_button = QPushButton("Move up")
        self.move_model_layer_down_button = QPushButton("Move down")
        for button in (self.add_model_layer_button, self.remove_model_layer_button, self.move_model_layer_up_button, self.move_model_layer_down_button):
            layer_actions.addWidget(button)
        layer_actions.addStretch(1)
        layout.addLayout(layer_actions)

        lower = QHBoxLayout()
        training = QGroupBox("Training controls")
        training_form = QFormLayout(training)
        training_form.addRow("Output mode", self._combo("model.output_mode", ("regression",), "regression"))
        training_form.addRow("Batch size", self._spin("training.batch_size", 64, 1, 100000))
        training_form.addRow("Epochs", self._spin("training.epochs", 100, 1, 100000))
        training_form.addRow("Optimizer", self._combo("training.optimizer", ("adam", "adamw", "sgd"), "adam"))
        training_form.addRow("Learning rate", self._double("training.learning_rate", 0.0001, 1e-9, 10.0, 8))
        training_form.addRow("Scheduler", self._combo("training.scheduler", ("cosine", "plateau", "constant"), "cosine"))
        lower.addWidget(training, 1)
        summary = QGroupBox("Model contract")
        summary_layout = QVBoxLayout(summary)
        self.model_summary = QTextEdit()
        self.model_summary.setReadOnly(True)
        self.model_summary.setPlainText("Add, remove and reorder layers, then validate the real tensor contract.\n\nSupported: Conv2D, MaxPool2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Flatten and Dense.")
        summary_layout.addWidget(self.model_summary)
        lower.addWidget(summary, 2)
        layout.addLayout(lower, 2)
        self.model_validate_button = QPushButton("Build model and validate one forward pass")
        self.model_validate_button.setObjectName("primaryAction")
        layout.addWidget(self.model_validate_button)
        self.add_model_layer_button.clicked.connect(lambda: self.add_model_layer({"type": "conv2d", "units": 32, "kernel": 3, "activation": "relu"}))
        self.remove_model_layer_button.clicked.connect(lambda: self._remove_selected_rows(self.model_layer_table))
        self.move_model_layer_up_button.clicked.connect(lambda: self._move_model_layer(-1))
        self.move_model_layer_down_button.clicked.connect(lambda: self._move_model_layer(1))
        return self._scroll(page)

    def _hpc_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        tabs = QTabWidget()
        local = QWidget()
        local_form = QFormLayout(local)
        local_intro = QLabel(
            "Local physical workflow: choose an output folder/Python, prepare the reproducible package, "
            "generate a small BornAgain test first, then generate the full dataset and train."
        )
        local_intro.setWordWrap(True)
        local_form.addRow(local_intro)
        local_form.addRow("Output folder", self._line("project.workspace", ""))
        local_form.addRow("Python executable", self._line("training.local_python", ""))
        self.local_python_button = QPushButton("Choose Python executable…")
        local_form.addRow(self.local_python_button)
        self.local_folder_button = QPushButton("Choose output folder…")
        local_form.addRow(self.local_folder_button)
        local_form.addRow("Test samples", self._spin("training.smoke_samples", 64, 8, 10000))
        local_form.addRow("Test epochs", self._spin("training.smoke_epochs", 2, 1, 20))
        self.local_prepare_button = QPushButton("1 · Prepare local job package")
        self.local_generate_test_button = QPushButton("2 · Generate small physical BornAgain test")
        self.local_generate_test_button.setToolTip("Generate Test samples with the real BornAgain pipeline.")
        self.local_generate_button = QPushButton("3 · Generate full physical dataset")
        self.local_train_button = QPushButton("4 · Train on generated dataset")
        self.local_smoke_button = QPushButton("Optional · Reference-based I/O smoke test")
        self.local_smoke_button.setToolTip(
            "Fast non-physical I/O/model check using a loaded reference image. It is not a replacement for the small BornAgain physical test."
        )
        self.local_generate_test_button.setObjectName("primaryAction")
        for button in (
            self.local_prepare_button,
            self.local_generate_test_button,
            self.local_generate_button,
            self.local_train_button,
            self.local_smoke_button,
        ):
            local_form.addRow(button)
        output_help = QLabel(
            "Generation writes HDF5 shards under <output>/<project name>/dataset. Progress and errors appear in Monitor & Results."
        )
        output_help.setWordWrap(True)
        local_form.addRow(output_help)
        tabs.addTab(local, "Local")

        maxwell = QWidget()
        maxwell_form = QFormLayout(maxwell)
        maxwell_form.addRow("Host", self._line("hpc.host", "maxwell.desy.de"))
        maxwell_form.addRow("User", self._line("hpc.user", ""))
        maxwell_form.addRow("Remote project path", self._line("hpc.remote_path", ""))
        maxwell_form.addRow("Partition", self._line("hpc.partition", "allgpu"))
        maxwell_form.addRow("GPUs", self._spin("hpc.gpus", 1, 0, 64))
        maxwell_form.addRow("CPUs", self._spin("hpc.cpus", 8, 1, 1024))
        maxwell_form.addRow("Memory", self._line("hpc.memory", "64G"))
        maxwell_form.addRow("Run time", self._line("hpc.time", "24:00:00"))
        maxwell_form.addRow("Remote Python command", self._line("hpc.python_command", "python"))
        maxwell_form.addRow("Job array", self._check("hpc.job_array", True, "Generate HDF5 shards in parallel"))
        self.connection_button = QPushButton("Test SSH and remote path")
        self.hpc_prepare_button = QPushButton("Prepare reproducible HPC job")
        self.hpc_submit_button = QPushButton("Upload and submit dependent jobs")
        self.hpc_submit_button.setObjectName("primaryAction")
        reserved = QLabel("Reserved interface: Maxwell SSH upload/submission is disabled for this local demo. Job packaging and Slurm scripts remain exportable.")
        reserved.setWordWrap(True)
        maxwell_form.addRow(reserved)
        maxwell_form.addRow(self.connection_button)
        maxwell_form.addRow(self.hpc_prepare_button)
        maxwell_form.addRow(self.hpc_submit_button)
        self.connection_button.setEnabled(False)
        self.hpc_prepare_button.setEnabled(False)
        self.hpc_submit_button.setEnabled(False)
        tabs.addTab(maxwell, "Maxwell")
        layout.addWidget(tabs)
        self.package_tree = QTextEdit()
        self.package_tree.setReadOnly(True)
        self.package_tree.setPlainText("Prepare a job package to see its path and reproducibility manifest.")
        layout.addWidget(self.package_tree, 1)
        return self._scroll(page)

    def _monitor_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        status = QHBoxLayout()
        self.job_state = QLabel("NO JOB")
        self.job_state.setObjectName("jobState")
        self.job_id_label = QLabel("Job ID: —")
        self.refresh_job_button = QPushButton("Refresh")
        self.sync_results_button = QPushButton("Sync results")
        status.addWidget(self.job_state)
        status.addWidget(self.job_id_label)
        status.addWidget(self.refresh_job_button)
        status.addWidget(self.sync_results_button)
        status.addStretch(1)
        layout.addLayout(status)
        splitter = QSplitter(Qt.Horizontal)
        self.monitor_splitter = splitter
        self.job_log = QTextEdit()
        self.job_log.setReadOnly(True)
        self.job_log.setPlaceholderText("Slurm/local process output will appear here. Closing the GUI does not stop remote jobs.")
        splitter.addWidget(self.job_log)
        result = QWidget()
        result_layout = QVBoxLayout(result)
        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(("Epoch", "Train loss", "Validation loss", "Learning rate"))
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.metrics_table)
        self.register_model_button = QPushButton("Register best model in 2D Prediction")
        result_layout.addWidget(self.register_model_button)
        splitter.addWidget(result)
        layout.addWidget(splitter, 1)
        return page

    def _step_selected(self, index: int) -> None:
        if index < 0:
            return
        self.stack.setCurrentIndex(index)
        self.back_button.setEnabled(index > 0)
        self.step_changed.emit(index)

    def add_mask_shape(self, shape: Dict[str, Any]) -> None:
        row = self.mask_shape_table.rowCount()
        self.mask_shape_table.insertRow(row)
        shape_type = str(shape.get("type", "rectangle"))
        if shape_type in {"ellipse", "roi_ellipse_exterior"}:
            values = (
                shape_type,
                shape.get("cx", 0),
                shape.get("cy", 0),
                shape.get("radius_x", shape.get("radius", 0)),
                shape.get("radius_y", shape.get("radius", 0)),
            )
        elif shape_type == "circle":
            values = ("circle", shape.get("cx", 0), shape.get("cy", 0), shape.get("radius", 0), "")
        else:
            values = ("rectangle", shape.get("x", 0), shape.get("y", 0), shape.get("width", 0), shape.get("height", 0))
        for column, value in enumerate(values):
            self.mask_shape_table.setItem(row, column, QTableWidgetItem(str(value)))

    def set_plugin_parameters(self, table: QTableWidget, definitions, values: Dict[str, Any]) -> None:
        table.setRowCount(0)
        for definition in definitions:
            row = table.rowCount()
            table.insertRow(row)
            key = str(definition["key"])
            spec = values.get(key, {}) if isinstance(values, dict) else {}
            distribution = QComboBox()
            distribution.addItems(("uniform", "log_uniform"))
            distribution.setCurrentText(str(spec.get("distribution", "uniform")))
            distribution.currentTextChanged.connect(self.configuration_edited)
            table.setItem(row, 0, QTableWidgetItem(key))
            table.item(row, 0).setFlags(table.item(row, 0).flags() & ~Qt.ItemIsEditable)
            table.setCellWidget(row, 1, distribution)
            table.setItem(row, 2, QTableWidgetItem(str(spec.get("minimum", definition.get("minimum", 0.0)))))
            table.setItem(row, 3, QTableWidgetItem(str(spec.get("maximum", definition.get("maximum", 1.0)))))
            meaning = f"{definition.get('label', key)}"
            if definition.get("unit"):
                meaning += f" [{definition['unit']}]"
            table.setItem(row, 4, QTableWidgetItem(meaning))
            table.item(row, 4).setFlags(table.item(row, 4).flags() & ~Qt.ItemIsEditable)

    @staticmethod
    def plugin_parameters(table: QTableWidget) -> Dict[str, Dict[str, Any]]:
        parameters: Dict[str, Dict[str, Any]] = {}
        for row in range(table.rowCount()):
            name = table.item(row, 0).text().strip()
            distribution = table.cellWidget(row, 1)
            parameters[name] = {
                "distribution": distribution.currentText() if isinstance(distribution, QComboBox) else "uniform",
                "minimum": float(table.item(row, 2).text()),
                "maximum": float(table.item(row, 3).text()),
            }
        return parameters

    def _add_layer_row(self) -> None:
        row = self.layer_table.rowCount()
        self.layer_table.insertRow(row)
        for column, value in enumerate(("1", "Silicon", "10.0", "10.0", "0.0", "0.0")):
            self.layer_table.setItem(row, column, QTableWidgetItem(value))

    def add_model_layer(self, spec: Dict[str, Any], row: Optional[int] = None) -> None:
        row = self.model_layer_table.rowCount() if row is None else row
        self.model_layer_table.insertRow(row)
        kind = QComboBox()
        kind.addItems(("conv2d", "maxpool2d", "batch_normalization", "dropout", "global_average_pooling2d", "flatten", "dense"))
        kind.setCurrentText(str(spec.get("type", "conv2d")))
        kind.currentTextChanged.connect(self.configuration_edited)
        activation = QComboBox()
        activation.addItems(("relu", "gelu", "tanh", "sigmoid", "linear"))
        activation.setCurrentText(str(spec.get("activation", "relu")))
        activation.currentTextChanged.connect(self.configuration_edited)
        self.model_layer_table.setCellWidget(row, 0, kind)
        self.model_layer_table.setItem(row, 1, QTableWidgetItem(str(spec.get("units", ""))))
        self.model_layer_table.setItem(row, 2, QTableWidgetItem(str(spec.get("kernel", spec.get("pool", "")))))
        self.model_layer_table.setCellWidget(row, 3, activation)
        self.model_layer_table.setItem(row, 4, QTableWidgetItem(str(spec.get("rate", ""))))

    def set_model_layers(self, layers) -> None:
        self.model_layer_table.setRowCount(0)
        for layer in layers:
            self.add_model_layer(layer)

    def model_layers(self):
        layers = []
        for row in range(self.model_layer_table.rowCount()):
            kind_widget = self.model_layer_table.cellWidget(row, 0)
            activation_widget = self.model_layer_table.cellWidget(row, 3)
            kind = kind_widget.currentText() if isinstance(kind_widget, QComboBox) else "conv2d"
            units_text = self.model_layer_table.item(row, 1).text().strip() if self.model_layer_table.item(row, 1) else ""
            size_text = self.model_layer_table.item(row, 2).text().strip() if self.model_layer_table.item(row, 2) else ""
            rate_text = self.model_layer_table.item(row, 4).text().strip() if self.model_layer_table.item(row, 4) else ""
            spec: Dict[str, Any] = {"type": kind}
            if kind in {"conv2d", "dense"}:
                spec["units"] = int(float(units_text or 32))
                spec["activation"] = activation_widget.currentText() if isinstance(activation_widget, QComboBox) else "relu"
            if kind == "conv2d":
                spec["kernel"] = int(float(size_text or 3))
            elif kind == "maxpool2d":
                spec["pool"] = int(float(size_text or 2))
            elif kind == "dropout":
                spec["rate"] = float(rate_text or 0.3)
            layers.append(spec)
        return layers

    def _move_model_layer(self, offset: int) -> None:
        row = self.model_layer_table.currentRow()
        target = row + offset
        if row < 0 or target < 0 or target >= self.model_layer_table.rowCount():
            return
        layers = self.model_layers()
        layers[row], layers[target] = layers[target], layers[row]
        self.set_model_layers(layers)
        self.model_layer_table.selectRow(target)

    @staticmethod
    def _remove_selected_rows(table: QTableWidget) -> None:
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    def remove_selected_mask_shapes(self) -> bool:
        before = self.mask_shape_table.rowCount()
        self._remove_selected_rows(self.mask_shape_table)
        return self.mask_shape_table.rowCount() != before

    def remove_mask_shapes_by_type(self, *shape_types: str) -> None:
        wanted = set(shape_types)
        for row in range(self.mask_shape_table.rowCount() - 1, -1, -1):
            item = self.mask_shape_table.item(row, 0)
            if item is not None and item.text() in wanted:
                self.mask_shape_table.removeRow(row)

    def mask_shapes(self):
        shapes = []
        for row in range(self.mask_shape_table.rowCount()):
            values = [self.mask_shape_table.item(row, column).text() if self.mask_shape_table.item(row, column) else "0" for column in range(5)]
            if values[0] == "circle":
                shapes.append({"type": "circle", "cx": int(float(values[1])), "cy": int(float(values[2])), "radius": int(float(values[3]))})
            elif values[0] in {"ellipse", "roi_ellipse_exterior"}:
                shapes.append({
                    "type": values[0],
                    "cx": float(values[1]),
                    "cy": float(values[2]),
                    "radius_x": max(1e-6, float(values[3])),
                    "radius_y": max(1e-6, float(values[4])),
                })
            else:
                shapes.append({"type": "rectangle", "x": int(float(values[1])), "y": int(float(values[2])), "width": int(float(values[3])), "height": int(float(values[4]))})
        return shapes

    def set_simulation_preview(
        self,
        comparison_images: Dict[str, np.ndarray],
        comparison_labels: Dict[str, str],
        stages,
        stats: Dict[str, Any],
        spectrum_x: np.ndarray,
        spectrum_y: np.ndarray,
    ) -> None:
        for key, image in comparison_images.items():
            for canvas in self.impact_canvases.get(key, []):
                canvas.set_data(image)
            for label in self.impact_value_labels.get(key, []):
                label.setText(comparison_labels.get(key, key.title()))
        self.preview_tabs.clear()
        self.preview_canvases.clear()
        for stage in stages:
            name = str(stage["name"])
            key = name.lower()
            canvas = ArrayCanvas(f"{name} simulated stage")
            canvas.set_data(stage["image"], stage.get("mask"))
            self.preview_canvases[key] = canvas
            self.preview_tabs.addTab(canvas, name)
        if not stages:
            empty = QLabel("No enabled preprocessing stages were returned.")
            empty.setAlignment(Qt.AlignCenter)
            self.preview_tabs.addTab(empty, "No stages")
        self.preview_views.setTabText(1, f"Pipeline stages ({len(stages)})")
        self._apply_display_settings("preview")
        self.histogram.set_data(spectrum_x, spectrum_y)
        self.preview_stats.setText("\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in stats.items()))

    def set_preview_busy(self, busy: bool, progress: int = 0, message: str = "") -> None:
        for button in (self.generate_preview_button, self.force_simulation_button, self.new_realization_button):
            button.setEnabled(not busy)
        self.preview_progress.setVisible(busy)
        self.preview_activity.setVisible(busy or bool(message))
        self.preview_progress.setValue(max(0, min(100, int(progress))))
        self.preview_activity.setText(message)

    def set_preview_progress(self, progress: int, message: str) -> None:
        self.preview_progress.setValue(max(0, min(100, int(progress))))
        self.preview_activity.setText(message)

    def set_comparison_details(
        self,
        details: Dict[str, Any],
        parameter_specs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._comparison_details = details
        self._comparison_parameter_specs = parameter_specs or {}
        self.preview_parameters_button.setEnabled(bool(details))

    @staticmethod
    def _append_parameter_tree(parent: QTreeWidgetItem, value: Any) -> None:
        if isinstance(value, dict):
            for key, child_value in value.items():
                child = QTreeWidgetItem((str(key).replace("_", " "), "" if isinstance(child_value, (dict, list)) else str(child_value)))
                parent.addChild(child)
                if isinstance(child_value, (dict, list)):
                    TrainsetBuildPage._append_parameter_tree(child, child_value)
        elif isinstance(value, list):
            for index, child_value in enumerate(value):
                child = QTreeWidgetItem((f"item {index + 1}", "" if isinstance(child_value, (dict, list)) else str(child_value)))
                parent.addChild(child)
                if isinstance(child_value, (dict, list)):
                    TrainsetBuildPage._append_parameter_tree(child, child_value)

    def show_comparison_parameters(self) -> None:
        if not self._comparison_details:
            return
        if self._parameter_dialog is not None and self._parameter_dialog.isVisible():
            self._parameter_dialog.raise_()
            self._parameter_dialog.activateWindow()
            return
        dialog = QDialog(self)
        self._parameter_dialog = dialog
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.setWindowTitle("Parameters used and editable What-if simulation")
        dialog.resize(900, 700)
        dialog_layout = QVBoxLayout(dialog)
        note = QLabel(
            "Minimum, midpoint and maximum are immutable audit snapshots. The What-if tab copies one snapshot, "
            "lets you edit its physics values, and renders a fourth simulation without replacing the saved three."
        )
        note.setWordWrap(True)
        dialog_layout.addWidget(note)
        tabs = QTabWidget()
        for key, heading in (("minimum", "Minimum"), ("midpoint", "Midpoint"), ("maximum", "Maximum")):
            tree = QTreeWidget()
            tree.setHeaderLabels(("Parameter", "Value"))
            tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
            root = QTreeWidgetItem((heading, ""))
            tree.addTopLevelItem(root)
            self._append_parameter_tree(root, self._comparison_details.get(key, {}))
            root.setExpanded(True)
            tabs.addTab(tree, heading)

        what_if_page = QWidget()
        what_if_layout = QVBoxLayout(what_if_page)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Start from"))
        source_combo = QComboBox()
        source_combo.addItem("Minimum snapshot", "minimum")
        source_combo.addItem("Midpoint snapshot", "midpoint")
        source_combo.addItem("Maximum snapshot", "maximum")
        source_combo.setCurrentIndex(1)
        source_row.addWidget(source_combo)
        source_row.addStretch(1)
        what_if_layout.addLayout(source_row)
        what_if_help = QLabel(
            "Edits are debounced: BornAgain runs after you pause, while unchanged physics reuses the preview cache. "
            "Noise and mask use the same realization so visual differences remain attributable to the edited values."
        )
        what_if_help.setWordWrap(True)
        what_if_help.setProperty("infoPanel", True)
        what_if_layout.addWidget(what_if_help)
        editor_and_image = QSplitter(Qt.Horizontal)
        editor = QWidget()
        editor_form = QFormLayout(editor)
        self._what_if_controls: Dict[str, QDoubleSpinBox] = {}
        midpoint_values = self._comparison_details.get("midpoint", {}).get(
            "editable physics",
            self._comparison_details.get("midpoint", {}).get("physics values", {}),
        )
        scalar_values = {
            name: value
            for name, value in midpoint_values.items()
            if not str(name).startswith("__") and isinstance(value, (int, float, np.number))
        }
        ordered_names = list(self._comparison_parameter_specs) or list(scalar_values)
        for name in ordered_names:
            if name not in scalar_values:
                continue
            spec = self._comparison_parameter_specs.get(name, {})
            control = QDoubleSpinBox()
            control.setDecimals(6)
            control.setRange(-1e12, 1e12)
            control.setValue(float(scalar_values[name]))
            control.setKeyboardTracking(False)
            low = spec.get("minimum")
            high = spec.get("maximum")
            control.setToolTip(
                f"Configured training range: {low} to {high}. What-if values may go outside it for diagnosis."
                if low is not None and high is not None
                else "Editable physics value for the fourth diagnostic simulation."
            )
            editor_form.addRow(ParameterCoverageWidget._axis_label(name), control)
            self._what_if_controls[name] = control
        editor_and_image.addWidget(editor)
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        self._what_if_canvas = ArrayCanvas("Edit a value or press Simulate now")
        self._what_if_canvas.setMinimumSize(330, 290)
        self._what_if_status = QLabel("What-if is independent of the three saved comparison snapshots.")
        self._what_if_status.setWordWrap(True)
        self._what_if_progress = QProgressBar()
        self._what_if_progress.setRange(0, 0)
        self._what_if_progress.setVisible(False)
        result_layout.addWidget(self._what_if_canvas, 1)
        result_layout.addWidget(self._what_if_progress)
        result_layout.addWidget(self._what_if_status)
        editor_and_image.addWidget(result_panel)
        editor_and_image.setStretchFactor(1, 1)
        what_if_layout.addWidget(editor_and_image, 1)
        what_if_actions = QHBoxLayout()
        auto_simulate = QCheckBox("Auto-simulate after edits")
        auto_simulate.setChecked(True)
        simulate_now = QPushButton("Simulate now")
        simulate_now.setObjectName("primaryAction")
        what_if_actions.addWidget(auto_simulate)
        what_if_actions.addStretch(1)
        what_if_actions.addWidget(simulate_now)
        what_if_layout.addLayout(what_if_actions)

        update_timer = QTimer(dialog)
        update_timer.setSingleShot(True)
        update_timer.setInterval(700)

        def request_what_if() -> None:
            values = {name: control.value() for name, control in self._what_if_controls.items()}
            if values:
                self.what_if_requested.emit(values)

        def schedule_what_if(*_args) -> None:
            if auto_simulate.isChecked():
                update_timer.start()

        def load_snapshot(*_args) -> None:
            key = str(source_combo.currentData())
            values = self._comparison_details.get(key, {}).get(
                "editable physics",
                self._comparison_details.get(key, {}).get("physics values", {}),
            )
            for name, control in self._what_if_controls.items():
                if name in values and isinstance(values[name], (int, float, np.number)):
                    control.blockSignals(True)
                    control.setValue(float(values[name]))
                    control.blockSignals(False)
            schedule_what_if()

        update_timer.timeout.connect(request_what_if)
        source_combo.currentIndexChanged.connect(load_snapshot)
        simulate_now.clicked.connect(request_what_if)
        for control in self._what_if_controls.values():
            control.valueChanged.connect(schedule_what_if)
        tabs.addTab(what_if_page, "What-if (editable)")
        dialog_layout.addWidget(tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.close)
        dialog_layout.addWidget(buttons)
        dialog.destroyed.connect(
            lambda *_args: setattr(self, "_parameter_dialog", None)
        )
        dialog.show()
        load_snapshot()

    def set_what_if_busy(self, busy: bool, message: str) -> None:
        if not hasattr(self, "_what_if_status"):
            return
        self._what_if_progress.setVisible(busy)
        self._what_if_status.setText(message)

    def set_what_if_result(self, image: np.ndarray, details: str) -> None:
        if self._parameter_dialog is None:
            return
        self._what_if_canvas.set_data(image)
        self._what_if_progress.setVisible(False)
        self._what_if_status.setText(details)

    def set_preview_stages(self, _reference, stages, stats: Dict[str, Any], spectrum_x: np.ndarray, spectrum_y: np.ndarray) -> None:
        """Compatibility adapter for older callers; all images are simulated."""
        final = np.asarray(stages[-1]["image"]) if stages else np.zeros((1, 1), dtype=np.float32)
        self.set_simulation_preview(
            {"minimum": final, "midpoint": final, "maximum": final},
            {"minimum": "Minimum", "midpoint": "Midpoint", "maximum": "Maximum"},
            stages,
            stats,
            spectrum_x,
            spectrum_y,
        )

    def set_parameter_samples(self, samples, parameter_names=None, parameter_specs=None) -> None:
        self.parameter_coverage.set_samples(samples, parameter_names, parameter_specs)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        content_width = self.stack.width() if hasattr(self, "stack") else self.width()
        if hasattr(self, "impact_responsive_stack"):
            self.impact_responsive_stack.setCurrentIndex(0 if content_width >= 1040 else 1)
        if hasattr(self, "dataset_splitter"):
            desired_orientation = Qt.Horizontal if content_width >= 1080 else Qt.Vertical
            if desired_orientation != getattr(self, "_dataset_splitter_orientation", None):
                self.dataset_splitter.setOrientation(desired_orientation)
                self._dataset_splitter_orientation = desired_orientation
            self.dataset_splitter.setStretchFactor(0, 1)
            self.dataset_splitter.setStretchFactor(1, 1)
            QTimer.singleShot(0, self._balance_dataset_splitter)
        if hasattr(self, "monitor_splitter"):
            self.monitor_splitter.setOrientation(Qt.Horizontal if content_width >= 900 else Qt.Vertical)
        if hasattr(self, "step_list"):
            self.step_list.setMaximumWidth(218 if self.width() >= 1180 else 190)

    def _balance_dataset_splitter(self) -> None:
        """Give both design panes usable space after Qt finishes the resize pass."""
        if not hasattr(self, "dataset_splitter"):
            return
        first = self.dataset_splitter.widget(0)
        second = self.dataset_splitter.widget(1)
        if self.dataset_splitter.orientation() == Qt.Vertical:
            first.setMinimumSize(0, 220)
            second.setMinimumSize(0, 220)
            available = max(440, self.dataset_splitter.height())
            self.dataset_splitter.setSizes(
                [max(220, int(available * 0.50)), max(220, int(available * 0.50))]
            )
        else:
            first.setMinimumSize(480, 0)
            second.setMinimumSize(340, 0)
            available = max(820, self.dataset_splitter.width())
            self.dataset_splitter.setSizes(
                [max(480, int(available * 0.60)), max(340, int(available * 0.40))]
            )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            #freshTrainsetBuildPage {
                background: #eef2f6;
                color: #1f2937;
                font-size: 13px;
            }
            #freshTrainsetBuildPage QLabel,
            #freshTrainsetBuildPage QCheckBox,
            #freshTrainsetBuildPage QRadioButton { color: #334155; }
            #pageTitle { color: #0f172a; font-size: 22px; font-weight: 700; }
            #pageSubtitle { color: #64748b; font-size: 13px; }
            #validationBadge, #jobState {
                background: #eff6ff;
                color: #1d4ed8;
                border: 1px solid #bfdbfe;
                border-radius: 12px;
                padding: 5px 11px;
                font-weight: 600;
            }
            #trainsetStepList {
                background: #ffffff;
                color: #475569;
                border: 1px solid #d7dee8;
                border-radius: 10px;
                padding: 7px;
                outline: 0;
            }
            #trainsetStepList::item {
                color: #475569;
                padding: 13px 10px;
                margin: 2px;
                border-radius: 7px;
            }
            #trainsetStepList::item:hover { background: #f1f5f9; }
            #trainsetStepList::item:selected {
                background: #eaf3ff;
                color: #1d4ed8;
                border: 1px solid #bfdbfe;
                font-weight: 600;
            }
            #designPreviewCard {
                background: #ffffff;
                border: 1px solid #d7dee8;
                border-radius: 10px;
            }
            #freshTrainsetBuildPage QLabel[sectionTitle="true"] {
                color: #0f172a;
                font-size: 16px;
                font-weight: 700;
            }
            #freshTrainsetBuildPage QLabel[cardBody="true"] { color: #64748b; }
            #freshTrainsetBuildPage QWidget[displayBar="true"] {
                background: #f8fafc;
                border: 1px solid #d7dee8;
                border-radius: 7px;
            }
            #freshTrainsetBuildPage QLabel[infoPanel="true"] {
                background: #eff6ff;
                color: #1e40af;
                border: 1px solid #bfdbfe;
                border-radius: 7px;
                padding: 9px;
            }
            #freshTrainsetBuildPage QGroupBox {
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #d7dee8;
                border-radius: 9px;
                margin-top: 13px;
                padding: 12px 9px 9px 9px;
                font-weight: 600;
            }
            #freshTrainsetBuildPage QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
                background: #ffffff;
                color: #334155;
            }
            #freshTrainsetBuildPage QLineEdit,
            #freshTrainsetBuildPage QSpinBox,
            #freshTrainsetBuildPage QDoubleSpinBox,
            #freshTrainsetBuildPage QComboBox,
            #freshTrainsetBuildPage QTextEdit,
            #freshTrainsetBuildPage QTableWidget {
                background: #ffffff;
                color: #111827;
                border: 1px solid #c6cfdb;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #dbeafe;
                selection-color: #1e3a8a;
            }
            #freshTrainsetBuildPage QLineEdit:focus,
            #freshTrainsetBuildPage QSpinBox:focus,
            #freshTrainsetBuildPage QDoubleSpinBox:focus,
            #freshTrainsetBuildPage QComboBox:focus,
            #freshTrainsetBuildPage QTextEdit:focus,
            #freshTrainsetBuildPage QTableWidget:focus { border: 1px solid #60a5fa; }
            #freshTrainsetBuildPage QComboBox QAbstractItemView {
                background: #ffffff;
                color: #111827;
                border: 1px solid #c6cfdb;
                selection-background-color: #dbeafe;
            }
            #freshTrainsetBuildPage QPushButton {
                background: #ffffff;
                color: #334155;
                border: 1px solid #c6cfdb;
                border-radius: 6px;
                padding: 7px 12px;
                font-weight: 500;
            }
            #freshTrainsetBuildPage QPushButton:hover {
                background: #f8fafc;
                border-color: #94a3b8;
            }
            #freshTrainsetBuildPage QPushButton:pressed { background: #eef2f7; }
            #freshTrainsetBuildPage QPushButton:disabled {
                color: #94a3b8;
                background: #f1f5f9;
                border-color: #e2e8f0;
            }
            #freshTrainsetBuildPage QPushButton#primaryAction {
                background: #2563eb;
                color: #ffffff;
                border-color: #2563eb;
                font-weight: 600;
            }
            #freshTrainsetBuildPage QPushButton#primaryAction:hover { background: #1d4ed8; }
            #freshTrainsetBuildPage QHeaderView::section {
                background: #f8fafc;
                color: #475569;
                border: 0;
                border-bottom: 1px solid #d7dee8;
                padding: 7px;
                font-weight: 600;
            }
            #freshTrainsetBuildPage QTabWidget::pane {
                background: #ffffff;
                border: 1px solid #d7dee8;
                border-radius: 7px;
                top: -1px;
            }
            #freshTrainsetBuildPage QTabBar::tab {
                background: #edf1f5;
                color: #64748b;
                border: 1px solid #d7dee8;
                border-bottom: 0;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            #freshTrainsetBuildPage QTabBar::tab:hover { background: #e2e8f0; }
            #freshTrainsetBuildPage QTabBar::tab:selected {
                background: #ffffff;
                color: #1d4ed8;
                font-weight: 600;
            }
            #designStageTabs QTabBar::tab { padding: 8px 7px; }
            #freshTrainsetBuildPage QScrollArea,
            #freshTrainsetBuildPage QStackedWidget { background: transparent; border: 0; }
            """
        )
