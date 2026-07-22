from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from matplotlib import colormaps
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

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
        painter.drawText(12, 20, "Intensity distribution")
        if not self.y.size or self.y.max() <= 0:
            return
        plot = self.rect().adjusted(12, 30, -12, -14)
        painter.setPen(QPen(QColor(37, 99, 235), 2))
        points = []
        for index, value in enumerate(self.y):
            x = plot.left() + int(index * plot.width() / max(len(self.y) - 1, 1))
            y = plot.bottom() - int(float(value) / float(self.y.max()) * plot.height())
            points.append(QPoint(x, y))
        for first, second in zip(points, points[1:]):
            painter.drawLine(first, second)


class ParameterCoverageWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.x = np.array([])
        self.y = np.array([])
        self.x_name = "Parameter 1"
        self.y_name = "Parameter 2"
        self.setMinimumHeight(150)

    def set_samples(self, samples) -> None:
        if not samples:
            self.x = self.y = np.array([])
            self.update()
            return
        names = list(samples[0])
        self.x_name = names[0] if names else "Parameter 1"
        self.y_name = names[1] if len(names) > 1 else self.x_name
        self.x = np.asarray([row[self.x_name] for row in samples], dtype=float)
        self.y = np.asarray([row[self.y_name] for row in samples], dtype=float)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        painter.setPen(QColor(71, 85, 105))
        painter.drawText(12, 20, f"{self.x_name} / {self.y_name} sampling coverage")
        if not self.x.size:
            return
        plot = self.rect().adjusted(18, 32, -14, -18)
        x_min, x_max = float(self.x.min()), float(self.x.max())
        y_min, y_max = float(self.y.min()), float(self.y.max())
        painter.setPen(QPen(QColor(37, 99, 235), 4))
        for x_value, y_value in zip(self.x, self.y):
            x = plot.left() + int((float(x_value) - x_min) / max(x_max - x_min, 1e-12) * plot.width())
            y = plot.bottom() - int((float(y_value) - y_min) / max(y_max - y_min, 1e-12) * plot.height())
            painter.drawPoint(x, y)


class TrainsetBuildPage(QWidget):
    step_changed = pyqtSignal(int)
    mask_region_created = pyqtSignal(str, dict)

    STEPS = ("Dataset Design", "Local Preview", "Model Design", "Local Run", "Monitor & Results")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fields: Dict[str, QWidget] = {}
        self.preview_canvases: Dict[str, ArrayCanvas] = {}
        self._display_controls: Dict[str, Dict[str, QWidget]] = {}
        self._step_states = ["Not started"] * len(self.STEPS)
        self._design_stage_ready = [False, False, False, False]
        self.setObjectName("freshTrainsetBuildPage")
        self._build()
        self._apply_style()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("2D Trainset & Model Workspace")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Design locally · validate with real scattering data · run a lightweight test · export for Maxwell later")
        subtitle.setObjectName("pageSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        self.project_name = QLineEdit("gisaxs_2d_project")
        self.project_name.setPlaceholderText("Project name")
        self.project_name.setMaximumWidth(260)
        header.addWidget(QLabel("Project"))
        header.addWidget(self.project_name)
        self.validation_badge = QLabel("Not validated")
        self.validation_badge.setObjectName("validationBadge")
        header.addWidget(self.validation_badge)
        root.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.step_list = QListWidget()
        self.step_list.setObjectName("trainsetStepList")
        self.step_list.setFixedWidth(238)
        for number, text in enumerate(self.STEPS, start=1):
            item = QListWidgetItem(f"{number}.  {text}")
            item.setData(Qt.UserRole, number - 1)
            self.step_list.addItem(item)
        self.step_list.currentRowChanged.connect(self._step_selected)
        splitter.addWidget(self.step_list)
        self.stack = QStackedWidget()
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
        self.load_button = QPushButton("Load Project")
        self.save_button = QPushButton("Save Project")
        self.preview_button = QPushButton("Preview")
        self.prepare_button = QPushButton("Prepare portable job")
        self.submit_button = QPushButton("Maxwell (reserved)")
        self.submit_button.setEnabled(False)
        self.submit_button.setToolTip("SSH submission is intentionally disabled in the local demo. The portable Slurm package interface is retained.")
        actions.addWidget(self.back_button)
        actions.addStretch(1)
        for button in (self.validate_button, self.load_button, self.save_button, self.preview_button, self.prepare_button, self.submit_button):
            actions.addWidget(button)
        root.addLayout(actions)
        self.back_button.clicked.connect(lambda: self.step_list.setCurrentRow(max(0, self.step_list.currentRow() - 1)))
        self.step_list.setCurrentRow(0)

    def set_step_state(self, index: int, state: str) -> None:
        if not 0 <= index < len(self.STEPS):
            return
        self._step_states[index] = state
        item = self.step_list.item(index)
        item.setText(f"{index + 1}.  {self.STEPS[index]}  ·  {state}")
        item.setToolTip(state)

    def set_design_stage_ready(self, index: int, ready: bool = True) -> None:
        if not 0 <= index < len(self._design_stage_ready):
            return
        self._design_stage_ready[index] = ready
        labels = ("Full detector", "ROI", "Masked image", "Mask only")
        self.design_tabs.setTabText(index, f"{'✓  ' if ready else ''}{labels[index]}")

    def _scroll(self, content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(content)
        return area

    def _spin(self, path: str, value: int, minimum: int = 0, maximum: int = 100000000) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        self.fields[path] = widget
        return widget

    def _double(self, path: str, value: float, minimum: float = -1e12, maximum: float = 1e12, decimals: int = 6) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        self.fields[path] = widget
        return widget

    def _line(self, path: str, value: str = "") -> QLineEdit:
        widget = QLineEdit(value)
        self.fields[path] = widget
        return widget

    def _combo(self, path: str, values, current: Optional[str] = None) -> QComboBox:
        widget = QComboBox()
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

        layout.addWidget(QLabel("Colormap"), 0, 0)
        layout.addWidget(colormap, 0, 1)
        layout.addWidget(log_scale, 0, 2)
        layout.addWidget(auto_scale, 0, 3)
        layout.addWidget(QLabel("Vmin"), 1, 0)
        layout.addWidget(vmin, 1, 1)
        layout.addWidget(QLabel("Vmax"), 1, 2)
        layout.addWidget(vmax, 1, 3)
        layout.setColumnStretch(4, 1)

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
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
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
        reference_help = QLabel("Use the real detector image to define ROI, thresholds, fixed masks and preprocessing.")
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
        mask_grid.addWidget(self._check("mask.threshold.enabled", True, "Threshold mask"), 1, 0, 1, 2)
        mask_grid.addWidget(QLabel("Minimum"), 2, 0)
        mask_grid.addWidget(self._double("mask.threshold.minimum", 0.0), 2, 1)
        mask_grid.addWidget(QLabel("Maximum"), 3, 0)
        mask_grid.addWidget(self._double("mask.threshold.maximum", 1e12), 3, 1)
        self.draw_rectangle_button = QPushButton("Add rectangle")
        self.draw_circle_button = QPushButton("Add elliptical mask")
        self.remove_mask_button = QPushButton("Remove selected")
        self.clear_masks_button = QPushButton("Clear all shapes")
        mask_actions = QGridLayout()
        mask_actions.addWidget(self.draw_rectangle_button, 0, 0)
        mask_actions.addWidget(self.draw_circle_button, 0, 1)
        mask_actions.addWidget(self.remove_mask_button, 1, 0)
        mask_actions.addWidget(self.clear_masks_button, 1, 1)
        mask_grid.addLayout(mask_actions, 4, 0, 1, 2)
        self.mask_shape_table = QTableWidget(0, 5)
        self.mask_shape_table.setHorizontalHeaderLabels(("Type", "X / CX", "Y / CY", "Width / Radius X", "Height / Radius Y"))
        self.mask_shape_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mask_shape_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mask_shape_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mask_shape_table.setMaximumHeight(135)
        mask_grid.addWidget(self.mask_shape_table, 5, 0, 1, 2)
        random_grid = QGridLayout()
        for row, (label, path, value) in enumerate((
            ("Vertical bars", "mask.random.vertical_bars", 2),
            ("Horizontal bars", "mask.random.horizontal_bars", 1),
            ("Circles", "mask.random.circles", 1),
        )):
            random_grid.addWidget(QLabel(label), row, 0)
            random_grid.addWidget(self._spin(path, value, 0, 100), row, 1)
        random_grid.addWidget(self._check("mask.random.beamstop", True, "Beamstop"), 3, 0, 1, 2)
        mask_grid.addLayout(random_grid, 6, 0, 1, 2)
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
        sample_grid.addWidget(QLabel("Width fraction minimum"), 6, 0)
        sample_grid.addWidget(self._double("sample.mixture.sigma_fraction_min", 0.01, 0.0, 1.0, 3), 6, 1)
        sample_grid.addWidget(QLabel("Width fraction maximum"), 7, 0)
        sample_grid.addWidget(self._double("sample.mixture.sigma_fraction_max", 0.30, 0.0, 1.0, 3), 7, 1)
        sample_grid.addWidget(self._check("sample.mixture.random_weights", True, "Random component weights"), 8, 0, 1, 2)
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
        self.sigma_constraint_check = self._check("sample.constraints.paracrystal_sigma_le_0_2d", True, "Enforce σ ≤ 0.2D for the paracrystal")
        structure_grid.addWidget(self.spacing_constraint_check, 3, 0, 1, 2)
        structure_grid.addWidget(self.sigma_constraint_check, 4, 0, 1, 2)
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

        layout.addWidget(self._scroll(content), 7)
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
        layout.addWidget(side, 3)
        return page

    def _preview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Validation level"))
        self.preview_mode = QComboBox()
        self.preview_mode.addItems(("Preview", "Dry run"))
        controls.addWidget(self.preview_mode)
        controls.addWidget(QLabel("Samples"))
        self.preview_count = QSpinBox()
        self.preview_count.setRange(1, 1000)
        self.preview_count.setValue(16)
        controls.addWidget(self.preview_count)
        self.generate_preview_button = QPushButton("Generate preview")
        self.generate_preview_button.setObjectName("primaryAction")
        controls.addWidget(self.generate_preview_button)
        controls.addStretch(1)
        self.preview_capability = QLabel("Reference preprocessing preview ready")
        controls.addWidget(self.preview_capability)
        layout.addLayout(controls)

        preprocessing = QGroupBox("Ordered preprocessing / augmentation chain")
        chain = QGridLayout(preprocessing)
        chain.addWidget(QLabel("Enable"), 0, 0)
        chain.addWidget(QLabel("Stage and effect"), 0, 1)
        chain.addWidget(QLabel("Controls"), 0, 2, 1, 3)
        chain.addWidget(self._check("pre.background.enabled", False), 1, 0)
        chain.addWidget(QLabel("1. Physical background — adds a specular ridge and Yoneda-like band"), 1, 1)
        chain.addWidget(QLabel("relative min / max"), 1, 2)
        chain.addWidget(self._double("pre.background.min", 0.05, 0.0, 10.0, 3), 1, 3)
        chain.addWidget(self._double("pre.background.max", 0.30, 0.0, 10.0, 3), 1, 4)
        chain.addWidget(self._check("pre.noise.enabled", True), 2, 0)
        chain.addWidget(QLabel("2. Detector noise — varies signal-to-noise ratio"), 2, 1)
        chain.addWidget(QLabel("SNR min / max (dB)"), 2, 2)
        chain.addWidget(self._double("pre.noise.min", 80.0), 2, 3)
        chain.addWidget(self._double("pre.noise.max", 110.0), 2, 4)
        chain.addWidget(self._check("pre.mask.enabled", True), 3, 0)
        chain.addWidget(QLabel("3. Apply mask — writes the configured mask value"), 3, 1, 1, 4)
        chain.addWidget(self._check("pre.log.enabled", True), 4, 0)
        chain.addWidget(QLabel("4. Log transform — compresses scattering dynamic range"), 4, 1, 1, 4)
        chain.addWidget(self._check("pre.normalize.enabled", True), 5, 0)
        chain.addWidget(QLabel("5. Normalize — maps valid pixels to a stable scale"), 5, 1)
        chain.addWidget(self._combo("pre.normalize.mode", ("range", "upper", "lower"), "range"), 5, 2)
        chain.addWidget(self._double("pre.normalize.lower", 0.0), 5, 3)
        chain.addWidget(self._double("pre.normalize.upper", 1.0), 5, 4)
        chain.addWidget(self._check("pre.edge.enabled", False), 6, 0)
        chain.addWidget(QLabel("6. Random edge crop — augmentation; crop then resize back"), 6, 1)
        chain.addWidget(QLabel("maximum px"), 6, 2)
        chain.addWidget(self._spin("pre.edge.maximum", 4, 0, 128), 6, 3)
        chain.addWidget(QLabel("Every enabled effect receives its own preview tab below."), 7, 0, 1, 5)
        layout.addWidget(preprocessing)

        main = QSplitter(Qt.Horizontal)
        self.preview_tabs = QTabWidget()
        for name in ("Reference", "ROI", "Masked image", "Mask only", "Physical Background", "Noise", "Log", "Normalize", "Random Edge Crop", "Final"):
            canvas = ArrayCanvas(f"{name} stage will appear after preview")
            self.preview_canvases[name.lower()] = canvas
            self.preview_tabs.addTab(canvas, name)
        layout.addWidget(self._make_display_bar("preview"))
        main.addWidget(self.preview_tabs)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview_stats = QLabel("Generate a preview to inspect tensor shape, mask fraction and dynamic range.")
        self.preview_stats.setWordWrap(True)
        self.preview_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.diagnostic_tabs = QTabWidget()
        self.histogram = HistogramWidget()
        self.parameter_coverage = ParameterCoverageWidget()
        self.diagnostic_tabs.addTab(self.histogram, "Intensity")
        self.diagnostic_tabs.addTab(self.parameter_coverage, "Parameter coverage")
        self.preview_gate_table = QTableWidget(4, 2)
        self.preview_gate_table.setHorizontalHeaderLabels(("Local readiness check", "State"))
        for row, gate in enumerate(("Configuration valid", "Local samples generated", "Tensor shapes compatible", "Storage estimate accepted")):
            self.preview_gate_table.setItem(row, 0, QTableWidgetItem(gate))
            self.preview_gate_table.setItem(row, 1, QTableWidgetItem("Pending"))
        self.preview_gate_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_gate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.storage_accept_check = QCheckBox("I reviewed the estimated local storage for this run")
        right_layout.addWidget(self.preview_stats)
        right_layout.addWidget(self.diagnostic_tabs, 1)
        right_layout.addWidget(self.storage_accept_check)
        right_layout.addWidget(self.preview_gate_table)
        main.addWidget(right)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 1)
        layout.addWidget(main, 1)
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
        return page

    def _hpc_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        tabs = QTabWidget()
        local = QWidget()
        local_form = QFormLayout(local)
        local_form.addRow("Local project/output folder", self._line("project.workspace", ""))
        local_form.addRow("Local Python executable", self._line("training.local_python", ""))
        self.local_python_button = QPushButton("Choose Python executable…")
        local_form.addRow(self.local_python_button)
        self.local_folder_button = QPushButton("Choose folder…")
        local_form.addRow(self.local_folder_button)
        self.local_prepare_button = QPushButton("Prepare local job package")
        self.local_generate_button = QPushButton("Generate physical dataset locally")
        self.local_train_button = QPushButton("Train locally")
        local_form.addRow("Smoke-test samples", self._spin("training.smoke_samples", 64, 8, 10000))
        local_form.addRow("Smoke-test epochs", self._spin("training.smoke_epochs", 2, 1, 20))
        self.local_smoke_button = QPushButton("Run reference-based lightweight smoke test")
        self.local_smoke_button.setObjectName("primaryAction")
        smoke_help = QLabel("Creates a small non-physical demo dataset from the reference image, then tests the configured model for a few epochs.")
        smoke_help.setWordWrap(True)
        local_form.addRow(self.local_prepare_button)
        local_form.addRow(self.local_generate_button)
        local_form.addRow(self.local_train_button)
        local_form.addRow(self.local_smoke_button)
        local_form.addRow(smoke_help)
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
        return page

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
        status.addStretch(1)
        status.addWidget(self.refresh_job_button)
        status.addWidget(self.sync_results_button)
        layout.addLayout(status)
        splitter = QSplitter(Qt.Horizontal)
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
        activation = QComboBox()
        activation.addItems(("relu", "gelu", "tanh", "sigmoid", "linear"))
        activation.setCurrentText(str(spec.get("activation", "relu")))
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

    def set_preview_stages(self, reference: np.ndarray, stages, stats: Dict[str, Any], spectrum_x: np.ndarray, spectrum_y: np.ndarray) -> None:
        self.preview_canvases["reference"].set_data(reference)
        for stage in stages:
            key = str(stage["name"]).lower()
            if key == "roi":
                self.preview_canvases["roi"].set_data(stage["image"], stage.get("mask"))
                continue
            if key == "mask":
                mask = stage.get("mask")
                self.preview_canvases["masked image"].set_data(stage["image"], mask)
                if mask is not None:
                    self.preview_canvases["mask only"].set_data(np.asarray(mask, dtype=np.float32), binary=True)
                continue
            target = "final" if key not in self.preview_canvases else key
            self.preview_canvases[target].set_data(stage["image"], stage.get("mask"))
        if stages:
            self.preview_canvases["final"].set_data(stages[-1]["image"], stages[-1].get("mask"))
        self.histogram.set_data(spectrum_x, spectrum_y)
        self.preview_stats.setText("\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in stats.items()))

    def set_parameter_samples(self, samples) -> None:
        self.parameter_coverage.set_samples(samples)

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
