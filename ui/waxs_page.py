"""Embedded in-situ scattering data processing page for the main GUI.

This module intentionally does not instantiate the legacy ``WAXS.WAXS.MainWindow``.
It reuses the old loader and keeps the page embeddable in the existing
``Ui_MainWindow`` stacked widget.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import colormaps
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.widgets import RectangleSelector

from utils.path_utils import normalize_path
from calibration.image_loader import detect_nxs_frame_count, load_detector_image


SUPPORTED_EXTENSIONS = {".nxs", ".tif", ".tiff"}
SCATTERING_FILTER = "Scattering Data (*.nxs *.tif *.tiff)"


@dataclass
class ImageLoadResult:
    file_path: str
    frame_index: int
    frame_count: int
    image: np.ndarray


@dataclass
class BatchSettings:
    folder: str
    pattern: str
    output_folder: str
    export_images: bool
    export_curves: bool
    export_background_subtracted: bool
    log_scale: bool
    colormap: str
    auto_scale: bool
    vmin: float
    vmax: float
    mask_min: float
    mask_max: float
    geometry: dict
    integration: dict


class ImageLoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, frame_index: int):
        super().__init__()
        self.file_path = file_path
        self.frame_index = int(frame_index)

    def run(self) -> None:
        try:
            frame_count = detect_nxs_frame_count(self.file_path)
            frame_index = max(0, min(self.frame_index, max(0, frame_count - 1)))
            image = load_image_matrix(self.file_path, frame_idx=frame_index)
            self.finished.emit(
                ImageLoadResult(
                    file_path=self.file_path,
                    frame_index=frame_index,
                    frame_count=frame_count,
                    image=np.asarray(image, dtype=np.float32),
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class BatchWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, settings: BatchSettings):
        super().__init__()
        self.settings = settings
        self._stop_requested = False
        self._pause_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def set_paused(self, paused: bool) -> None:
        self._pause_requested = bool(paused)

    def run(self) -> None:
        try:
            settings = self.settings
            files = sorted(glob.glob(os.path.join(settings.folder, settings.pattern)))
            files = [path for path in files if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS]
            if not files:
                raise RuntimeError("No matching .nxs, .tif, or .tiff files found.")

            os.makedirs(settings.output_folder, exist_ok=True)
            image_dir = os.path.join(settings.output_folder, "images")
            curve_dir = os.path.join(settings.output_folder, "1D")
            if settings.export_images:
                os.makedirs(image_dir, exist_ok=True)
            if settings.export_curves or settings.export_background_subtracted:
                os.makedirs(curve_dir, exist_ok=True)

            curve_columns: list[np.ndarray] = []
            curve_names: list[str] = []
            x_axis: Optional[np.ndarray] = None
            background_y: Optional[np.ndarray] = None
            bg_columns: list[np.ndarray] = []

            work_items: list[tuple[str, int, int]] = []
            for file_path in files:
                frame_count = detect_nxs_frame_count(file_path)
                for frame_index in range(frame_count):
                    work_items.append((file_path, frame_index, frame_count))

            total = max(1, len(work_items))
            for idx, (file_path, frame_index, frame_count) in enumerate(work_items):
                if self._stop_requested:
                    self.finished.emit("Batch stopped by user.")
                    return
                while self._pause_requested and not self._stop_requested:
                    QThread.msleep(100)
                if self._stop_requested:
                    self.finished.emit("Batch stopped by user.")
                    return

                image = np.asarray(load_image_matrix(file_path, frame_idx=frame_index), dtype=np.float32)
                stem = Path(file_path).stem
                suffix = f"_f{frame_index + 1:04d}" if frame_count > 1 else ""
                name = f"{stem}{suffix}"

                if settings.export_images:
                    export_image_png(
                        image,
                        os.path.join(image_dir, f"{name}.png"),
                        log_scale=settings.log_scale,
                        colormap=settings.colormap,
                        auto_scale=settings.auto_scale,
                        vmin=settings.vmin,
                        vmax=settings.vmax,
                        mask_min=settings.mask_min,
                        mask_max=settings.mask_max,
                    )

                if settings.export_curves or settings.export_background_subtracted:
                    x, y = integrate_image(
                        image,
                        settings.geometry,
                        settings.integration,
                        settings.mask_min,
                        settings.mask_max,
                    )
                    if x_axis is None:
                        x_axis = x
                        curve_columns.append(x)
                    curve_columns.append(y)
                    curve_names.append(name)

                    if settings.export_curves:
                        export_curve_csv(os.path.join(curve_dir, f"{name}.csv"), x, y)

                    if settings.export_background_subtracted:
                        if background_y is None:
                            background_y = y
                        corrected = y - background_y
                        bg_columns.append(corrected)
                        export_curve_csv(os.path.join(curve_dir, f"{name}_subbg.csv"), x, corrected)

                pct = int(round((idx + 1) * 100 / total))
                self.progress.emit(pct, f"Processed {name}")

            if curve_columns:
                write_matrix_csv(
                    os.path.join(curve_dir, "output.csv"),
                    curve_columns,
                    ["x"] + curve_names,
                )
            if bg_columns and x_axis is not None:
                write_matrix_csv(
                    os.path.join(curve_dir, "output_subbg.csv"),
                    [x_axis] + bg_columns,
                    ["x"] + curve_names[: len(bg_columns)],
                )

            self.finished.emit("Batch processing completed.")
        except Exception as exc:
            self.failed.emit(str(exc))


class ScatteringImageViewer(QWidget):
    fileDropped = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.figure = Figure(figsize=(6, 5), constrained_layout=False)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = None
        self.cax = None
        self.colorbar = None
        self._preview_cache_key = None
        self._preview_cache_array: Optional[np.ndarray] = None
        self._preview_cache_extent: tuple[float, float, float, float] | None = None
        self._reset_image_axes()
        self._placeholder()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toolbar, 0)
        layout.addWidget(self.canvas, 1)

    def _placeholder(self) -> None:
        self._reset_image_axes()
        self.ax.clear()
        self.cax.clear()
        self.cax.set_axis_off()
        self.ax.text(
            0.5,
            0.5,
            "Open a .nxs, .tif, or .tiff file",
            ha="center",
            va="center",
            color="#64748b",
            transform=self.ax.transAxes,
        )
        self.ax.set_axis_off()

    def show_image(
        self,
        image: np.ndarray,
        *,
        log_scale: bool,
        colormap: str,
        auto_scale: bool,
        vmin: float,
        vmax: float,
        mask_min: float,
        mask_max: float,
        flip_vertical: bool,
        title: str,
        extent: tuple[float, float, float, float] | None = None,
        xlabel: str = "X (pixel)",
        ylabel: str = "Y (pixel)",
    ) -> None:
        render_start = time.perf_counter()
        raw = np.asarray(image)
        preview_source, preview_extent = self._preview_image(raw, extent)
        preview = prepare_display_array(
            preview_source,
            log_scale=log_scale,
            mask_min=mask_min,
            mask_max=mask_max,
            flip_vertical=flip_vertical,
        )
        preview = np.ascontiguousarray(preview)
        if auto_scale:
            limits_start = time.perf_counter()
            limits = estimate_display_limits(
                raw,
                log_scale=log_scale,
                mask_min=mask_min,
                mask_max=mask_max,
            )
            limits_time = time.perf_counter() - limits_start
            if limits is not None:
                vmin, vmax = limits
        else:
            limits_time = 0.0

        self._ensure_image_axes()
        self.ax.clear()
        self.cax.clear()
        cmap = colormaps.get_cmap(colormap).copy()
        cmap.set_bad(cmap(0.0))
        artist = self.ax.imshow(
            preview,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            extent=preview_extent,
        )
        self.ax.set_aspect("equal", adjustable="box", anchor="C")
        self.ax.set_anchor("C")
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.colorbar = self.figure.colorbar(artist, cax=self.cax)
        render_time = time.perf_counter() - render_start
        self._log_display_debug(raw, preview, limits_time, render_time)

    def _reset_image_axes(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_axes([0.07, 0.08, 0.78, 0.86])
        self.cax = self.figure.add_axes([0.88, 0.08, 0.025, 0.86])
        self.colorbar = None

    def _ensure_image_axes(self) -> None:
        if self.ax is None or self.cax is None:
            self._reset_image_axes()

    def _preview_image(
        self,
        image: np.ndarray,
        extent: tuple[float, float, float, float] | None,
    ) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
        height, width = image.shape[:2]
        canvas_w = max(64, int(self.canvas.width()))
        canvas_h = max(64, int(self.canvas.height()))
        max_preview_pixels = 1_000_000
        stride = max(
            1,
            int(np.ceil(width / max(1, canvas_w))),
            int(np.ceil(height / max(1, canvas_h))),
            int(np.ceil(np.sqrt(max(1, image.size) / max_preview_pixels))),
        )
        preview = image[::stride, ::stride]
        if extent is None:
            preview_extent = (0.0, float(width), float(height), 0.0)
        else:
            preview_extent = extent
        self._preview_cache_key = (id(image), image.shape, str(image.dtype), stride, extent)
        self._preview_cache_array = preview
        self._preview_cache_extent = preview_extent
        return preview, preview_extent

    @staticmethod
    def _array_mb(arr: np.ndarray) -> float:
        return float(np.asarray(arr).nbytes) / (1024.0 * 1024.0)

    def _log_display_debug(
        self,
        raw: np.ndarray,
        preview: np.ndarray,
        limits_time: float,
        render_time: float,
    ) -> None:
        print(
            "[WAXS display] "
            f"raw shape={raw.shape} dtype={raw.dtype} MB={self._array_mb(raw):.2f}; "
            f"preview shape={preview.shape} dtype={preview.dtype} MB={self._array_mb(preview):.2f}; "
            f"display_limits={limits_time:.3f}s; render_preview={render_time:.3f}s"
        )

    def display_limits(
        self,
        image: np.ndarray,
        *,
        log_scale: bool,
        mask_min: float,
        mask_max: float,
        flip_vertical: bool,
    ) -> tuple[float, float] | None:
        del flip_vertical
        limits_start = time.perf_counter()
        limits = estimate_display_limits(
            image,
            log_scale=log_scale,
            mask_min=mask_min,
            mask_max=mask_max,
        )
        print(f"[WAXS display] display_limits={time.perf_counter() - limits_start:.3f}s")
        return limits

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS:
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event) -> None:
        path = event.mimeData().urls()[0].toLocalFile()
        if path:
            self.fileDropped.emit(normalize_path(path))


class InSituProcessingWidget(QWidget):
    """Modern embedded replacement for the legacy in-situ data window."""

    statusChanged = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("waxsEmbeddedPage")
        self.current_file: Optional[str] = None
        self.current_image: Optional[np.ndarray] = None
        self.current_frame_count = 1
        self._loader_thread: Optional[QThread] = None
        self._loader_worker: Optional[ImageLoadWorker] = None
        self._batch_thread: Optional[QThread] = None
        self._batch_worker: Optional[BatchWorker] = None
        self._roi_selector: Optional[RectangleSelector] = None
        self._circle_pick_cid: Optional[int] = None
        self._center_pick_cid: Optional[int] = None
        self._circle_pick_points: list[tuple[float, float]] = []
        self._cut_extent: tuple[float, float, float, float] | None = None
        self._current_view_is_cut = False
        self._active_view = "2d"

        self._build_ui()
        self._connect_signals()
        self._set_frame_controls_enabled(False)
        self._set_status("Ready")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        root.addWidget(self._build_toolbar(), 0)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setObjectName("waxsContentSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        center = QWidget(splitter)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        self.view_tabs = QTabBar(center)
        self.view_tabs.setObjectName("waxsViewTabs")
        self.view_tabs.setExpanding(False)
        self.view_tabs.addTab("2D Image")
        self.view_tabs.addTab("1D Curve")

        self.viewer = ScatteringImageViewer(center)
        self.meta_label = QLabel("No file loaded", center)
        self.meta_label.setObjectName("waxsMetadataLabel")
        self.meta_label.setWordWrap(True)
        center_layout.addWidget(self.view_tabs, 0)
        center_layout.addWidget(self.viewer, 1)
        center_layout.addWidget(self.meta_label, 0)

        self.tabs = QTabWidget(splitter)
        self.tabs.setObjectName("waxsControlTabs")
        self.tabs.setMinimumWidth(360)
        self.tabs.setMaximumWidth(520)
        self.tabs.addTab(self._display_tab(), "Display")
        self.tabs.addTab(self._mask_tab(), "Mask")
        self.tabs.addTab(self._geometry_tab(), "Geometry")
        self.tabs.addTab(self._roi_tab(), "ROI / Cut")
        self.tabs.addTab(self._integration_tab(), "1D Integration")
        self.tabs.addTab(self._batch_tab(), "Batch / In-situ")

        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self.status_label = QLabel("Ready", self)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(260)
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.progress, 0)
        root.addLayout(bottom)

    def _build_toolbar(self) -> QWidget:
        bar = QFrame(self)
        bar.setObjectName("waxsTopToolbar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.open_button = QPushButton("Open File", bar)
        self.reload_button = QPushButton("Reload", bar)
        self.export_button = QPushButton("Export Image", bar)
        self.frame_label = QLabel("Frame:", bar)
        self.frame_spin = QSpinBox(bar)
        self.frame_spin.setMinimum(1)
        self.frame_spin.setMaximum(1)
        self.frame_spin.setToolTip("Frame selector for .nxs files")

        self.toolbar_auto_scale = QCheckBox("Auto Scale", bar)
        self.toolbar_auto_scale.setChecked(True)
        self.toolbar_log_scale = QCheckBox("Log Scale", bar)
        self.toolbar_cmap = QComboBox(bar)
        self.toolbar_cmap.addItems(["turbo", "jet", "viridis", "plasma", "inferno", "magma", "gray"])

        layout.addWidget(self.open_button)
        layout.addWidget(self.reload_button)
        layout.addSpacing(10)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.frame_spin)
        layout.addSpacing(10)
        layout.addWidget(self.toolbar_auto_scale)
        layout.addWidget(self.toolbar_log_scale)
        layout.addWidget(QLabel("Colormap:", bar))
        layout.addWidget(self.toolbar_cmap)
        layout.addStretch(1)
        layout.addWidget(self.export_button)
        return bar

    def _display_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.vmin_spin = make_double_spin(-1e12, 1e12, 0.0)
        self.vmax_spin = make_double_spin(-1e12, 1e12, 800.0)
        self.display_auto_scale = QCheckBox("Auto color scale")
        self.display_auto_scale.setChecked(True)
        self.display_log = QCheckBox("Log intensity")
        self.display_cmap = QComboBox()
        self.display_cmap.addItems(["turbo", "jet", "viridis", "plasma", "inferno", "magma", "gray"])
        self.display_flip = QCheckBox("Flip vertical")

        layout.addRow("Colorbar Min:", self.vmin_spin)
        layout.addRow("Colorbar Max:", self.vmax_spin)
        layout.addRow("", self.display_auto_scale)
        layout.addRow("", self.display_log)
        layout.addRow("Colormap:", self.display_cmap)
        layout.addRow("", self.display_flip)
        return tab

    def _mask_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.mask_min_spin = make_double_spin(-1e12, 1e12, -1e12)
        self.mask_max_spin = make_double_spin(-1e12, 1e12, 1e12)
        self.bad_pixel_spin = make_double_spin(-1e12, 1e12, -1.0)
        self.apply_mask_check = QCheckBox("Apply mask")
        self.apply_mask_check.setChecked(True)
        self.reset_mask_button = QPushButton("Reset Mask")

        self.mask_min_spin.setToolTip("Values below this threshold are hidden.")
        self.mask_max_spin.setToolTip("Values above this threshold are hidden.")
        self.bad_pixel_spin.setToolTip("Reserved bad-pixel marker for P03-style gap handling.")

        layout.addRow("Mask Min:", self.mask_min_spin)
        layout.addRow("Mask Max:", self.mask_max_spin)
        layout.addRow("Bad Pixel Threshold:", self.bad_pixel_spin)
        layout.addRow("", self.apply_mask_check)
        layout.addRow("", self.reset_mask_button)
        return tab

    def _geometry_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.incidence_spin = make_double_spin(-90, 90, 0.5)
        self.center_x_spin = make_double_spin(-1e6, 1e6, 1000.0)
        self.center_y_spin = make_double_spin(-1e6, 1e6, 1000.0)
        self.distance_spin = make_double_spin(0.000001, 1e9, 2000.0)
        self.pixel_x_spin = make_double_spin(0.000001, 1e9, 75.0)
        self.pixel_y_spin = make_double_spin(0.000001, 1e9, 75.0)
        self.wavelength_spin = make_double_spin(0.000001, 1e9, 1.0332)

        self.incidence_spin.setToolTip("Incidence angle in degrees.")
        self.distance_spin.setToolTip("Detector distance in mm.")
        self.pixel_x_spin.setToolTip("Detector pixel size X in micrometers.")
        self.pixel_y_spin.setToolTip("Detector pixel size Y in micrometers.")
        self.wavelength_spin.setToolTip("Beam wavelength in Angstrom.")

        layout.addRow("Incidence Angle:", self.incidence_spin)
        layout.addRow("Beam Center X:", self.center_x_spin)
        layout.addRow("Beam Center Y:", self.center_y_spin)
        layout.addRow("Detector Distance:", self.distance_spin)
        layout.addRow("Pixel Size X:", self.pixel_x_spin)
        layout.addRow("Pixel Size Y:", self.pixel_y_spin)
        layout.addRow("Wavelength:", self.wavelength_spin)
        return tab

    def _roi_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.cut_type_combo = QComboBox()
        self.cut_type_combo.addItems(["Q Range", "Line Cut", "Circle Cut"])
        self.show_cut_region_check = QCheckBox("Show Cut Region")
        self.show_cut_region_check.setChecked(True)
        self.show_center_check = QCheckBox("Show Center")
        self.show_center_check.setChecked(True)
        self.pick_center_button = QPushButton("Pick Center")

        self.qr_min_spin = make_double_spin(-1e6, 1e6, -121.0)
        self.qr_max_spin = make_double_spin(-1e6, 1e6, -121.0)
        self.qz_min_spin = make_double_spin(-1e6, 1e6, -121.0)
        self.qz_max_spin = make_double_spin(-1e6, 1e6, -121.0)
        self.select_roi_button = QPushButton("Select Q Range from Image")
        self.clear_roi_button = QPushButton("Clear ROI")
        self.apply_cut_button = QPushButton("Apply Cut")

        self.line_center_x_spin = make_double_spin(-1e6, 1e6, 1000.0)
        self.line_center_y_spin = make_double_spin(-1e6, 1e6, 1000.0)
        self.line_width_spin = make_double_spin(0.0, 1e6, 100.0)
        self.line_height_spin = make_double_spin(0.0, 1e6, 20.0)
        self.select_line_button = QPushButton("Select Line Cut")

        self.circle_center_x_spin = make_double_spin(-1e6, 1e6, 1000.0)
        self.circle_center_y_spin = make_double_spin(-1e6, 1e6, 1000.0)
        self.circle_inner_spin = make_double_spin(0.0, 1e6, 50.0)
        self.circle_outer_spin = make_double_spin(0.0, 1e6, 200.0)
        self.circle_start_spin = make_double_spin(-360.0, 360.0, -180.0)
        self.circle_end_spin = make_double_spin(-360.0, 360.0, 180.0)
        self.select_circle_button = QPushButton("Select Circle Cut")

        hint = QLabel("Use -121 for no Q-range limit.", tab)
        hint.setWordWrap(True)

        layout.addRow("Cut Type:", self.cut_type_combo)
        layout.addRow("", self.show_cut_region_check)
        layout.addRow("", self.show_center_check)
        layout.addRow("", self.pick_center_button)
        self.q_range_header = QLabel("<b>Q Range</b>", tab)
        self.line_cut_header = QLabel("<b>Line Cut</b>", tab)
        self.circle_cut_header = QLabel("<b>Circle Cut</b>", tab)

        layout.addRow(self.q_range_header)
        layout.addRow("Qr Min:", self.qr_min_spin)
        layout.addRow("Qr Max:", self.qr_max_spin)
        layout.addRow("Qz Min:", self.qz_min_spin)
        layout.addRow("Qz Max:", self.qz_max_spin)
        layout.addRow("", hint)
        layout.addRow("", self.select_roi_button)
        layout.addRow(self.line_cut_header)
        layout.addRow("Center X:", self.line_center_x_spin)
        layout.addRow("Center Y:", self.line_center_y_spin)
        layout.addRow("Width:", self.line_width_spin)
        layout.addRow("Height:", self.line_height_spin)
        layout.addRow("", self.select_line_button)
        layout.addRow(self.circle_cut_header)
        layout.addRow("Center X:", self.circle_center_x_spin)
        layout.addRow("Center Y:", self.circle_center_y_spin)
        layout.addRow("Inner Radius:", self.circle_inner_spin)
        layout.addRow("Outer Radius:", self.circle_outer_spin)
        layout.addRow("Start Angle:", self.circle_start_spin)
        layout.addRow("End Angle:", self.circle_end_spin)
        layout.addRow("", self.select_circle_button)
        layout.addRow("", self.clear_roi_button)
        layout.addRow("", self.apply_cut_button)

        self._q_range_controls = (
            self.q_range_header,
            self.qr_min_spin,
            self.qr_max_spin,
            self.qz_min_spin,
            self.qz_max_spin,
            hint,
            self.select_roi_button,
        )
        self._line_cut_controls = (
            self.line_cut_header,
            self.line_center_x_spin,
            self.line_center_y_spin,
            self.line_width_spin,
            self.line_height_spin,
            self.select_line_button,
        )
        self._circle_cut_controls = (
            self.circle_cut_header,
            self.circle_center_x_spin,
            self.circle_center_y_spin,
            self.circle_inner_spin,
            self.circle_outer_spin,
            self.circle_start_spin,
            self.circle_end_spin,
            self.select_circle_button,
        )
        self._roi_layout = layout
        self._update_cut_tool_visibility()
        return tab

    def _integration_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.integration_mode = QComboBox()
        self.integration_mode.addItems(["Radial", "Azimuthal"])
        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(10, 20000)
        self.bin_spin.setValue(500)
        self.smooth_curve_check = QCheckBox("Smooth curve")
        self.x_axis_mode = QComboBox()
        self.x_axis_mode.addItems(["q", "pixel", "2theta"])
        self.integrate_button = QPushButton("Integrate")
        self.export_1d_button = QPushButton("Export 1D")
        self.integration_status = QLabel("No curve calculated.", tab)
        self.integration_status.setWordWrap(True)
        self._last_curve: tuple[np.ndarray, np.ndarray] | None = None

        layout.addRow("Integration Mode:", self.integration_mode)
        layout.addRow("Number of Bins:", self.bin_spin)
        layout.addRow("", self.smooth_curve_check)
        layout.addRow("X Axis Mode:", self.x_axis_mode)
        layout.addRow("", self.integrate_button)
        layout.addRow("", self.export_1d_button)
        layout.addRow("", self.integration_status)
        return tab

    def _batch_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        folder_group = QGroupBox("Input", tab)
        folder_layout = QGridLayout(folder_group)
        self.batch_folder_edit = QLineEdit(folder_group)
        self.batch_browse_button = QPushButton("Browse", folder_group)
        self.batch_pattern_edit = QLineEdit("*.tif", folder_group)
        self.batch_pattern_edit.setToolTip("Examples: *.tif, *.tiff, *.nxs, *_m*.nxs")
        self.batch_output_edit = QLineEdit(os.getcwd(), folder_group)
        self.batch_output_browse_button = QPushButton("Output", folder_group)
        folder_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        folder_layout.addWidget(self.batch_folder_edit, 0, 1)
        folder_layout.addWidget(self.batch_browse_button, 0, 2)
        folder_layout.addWidget(QLabel("File Pattern:"), 1, 0)
        folder_layout.addWidget(self.batch_pattern_edit, 1, 1, 1, 2)
        folder_layout.addWidget(QLabel("Output Folder:"), 2, 0)
        folder_layout.addWidget(self.batch_output_edit, 2, 1)
        folder_layout.addWidget(self.batch_output_browse_button, 2, 2)

        export_group = QGroupBox("Export", tab)
        export_layout = QVBoxLayout(export_group)
        self.batch_export_images = QCheckBox("Export 2D images", export_group)
        self.batch_export_curves = QCheckBox("Export 1D curves", export_group)
        self.batch_export_subbg = QCheckBox("Export background-subtracted results", export_group)
        export_layout.addWidget(self.batch_export_images)
        export_layout.addWidget(self.batch_export_curves)
        export_layout.addWidget(self.batch_export_subbg)

        buttons = QHBoxLayout()
        self.batch_start_button = QPushButton("Start", tab)
        self.batch_pause_button = QPushButton("Pause", tab)
        self.batch_stop_button = QPushButton("Stop", tab)
        self.batch_pause_button.setEnabled(False)
        self.batch_stop_button.setEnabled(False)
        buttons.addWidget(self.batch_start_button)
        buttons.addWidget(self.batch_pause_button)
        buttons.addWidget(self.batch_stop_button)

        outer.addWidget(folder_group)
        outer.addWidget(export_group)
        outer.addLayout(buttons)
        outer.addStretch(1)
        return tab

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self.open_file_dialog)
        self.reload_button.clicked.connect(self.reload_current_file)
        self.export_button.clicked.connect(self.export_current_image)
        self.viewer.fileDropped.connect(self.load_file)
        self.view_tabs.currentChanged.connect(self._on_view_tab_changed)
        self.frame_spin.valueChanged.connect(self._on_frame_changed)

        self.toolbar_auto_scale.toggled.connect(self.display_auto_scale.setChecked)
        self.display_auto_scale.toggled.connect(self.toolbar_auto_scale.setChecked)
        self.toolbar_log_scale.toggled.connect(self.display_log.setChecked)
        self.display_log.toggled.connect(self.toolbar_log_scale.setChecked)
        self.display_log.toggled.connect(self._on_log_intensity_toggled)
        self.toolbar_cmap.currentTextChanged.connect(self.display_cmap.setCurrentText)
        self.display_cmap.currentTextChanged.connect(self.toolbar_cmap.setCurrentText)
        self.cut_type_combo.currentTextChanged.connect(self._on_cut_type_changed)

        for widget in (
            self.vmin_spin,
            self.vmax_spin,
            self.mask_min_spin,
            self.mask_max_spin,
            self.display_auto_scale,
            self.display_cmap,
            self.display_flip,
            self.apply_mask_check,
            self.show_cut_region_check,
            self.show_center_check,
            self.qr_min_spin,
            self.qr_max_spin,
            self.qz_min_spin,
            self.qz_max_spin,
            self.line_center_x_spin,
            self.line_center_y_spin,
            self.line_width_spin,
            self.line_height_spin,
            self.circle_center_x_spin,
            self.circle_center_y_spin,
            self.circle_inner_spin,
            self.circle_outer_spin,
            self.circle_start_spin,
            self.circle_end_spin,
        ):
            signal = getattr(widget, "valueChanged", None) or getattr(widget, "toggled", None) or getattr(widget, "currentTextChanged", None)
            if signal is not None:
                signal.connect(self.refresh_view)

        self.reset_mask_button.clicked.connect(self.reset_mask)
        self.apply_cut_button.clicked.connect(self.apply_cut)
        self.clear_roi_button.clicked.connect(self.clear_cut)
        self.select_roi_button.clicked.connect(self._select_roi_hint)
        self.select_line_button.clicked.connect(self.start_line_cut_selection)
        self.select_circle_button.clicked.connect(self.start_circle_cut_selection)
        self.pick_center_button.clicked.connect(self.start_center_pick)
        self.integrate_button.clicked.connect(self.integrate_current_image)
        self.export_1d_button.clicked.connect(self.export_current_curve)
        self.batch_browse_button.clicked.connect(self.select_batch_folder)
        self.batch_output_browse_button.clicked.connect(self.select_batch_output_folder)
        self.batch_start_button.clicked.connect(self.start_batch)
        self.batch_pause_button.clicked.connect(self.toggle_batch_pause)
        self.batch_stop_button.clicked.connect(self.stop_batch)

    def open_file_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Scattering File", "", SCATTERING_FILTER)
        if file_path:
            self.load_file(normalize_path(file_path))

    def load_file(self, file_path: str, frame_index: int = 0) -> None:
        suffix = Path(file_path).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            QMessageBox.warning(self, "Unsupported File Type", "Unsupported file type. Please select .nxs, .tif, or .tiff.")
            return
        self.current_file = normalize_path(file_path)
        self._start_loader(self.current_file, frame_index)

    def reload_current_file(self) -> None:
        if not self.current_file:
            QMessageBox.information(self, "Reload", "No image loaded.")
            return
        self._start_loader(self.current_file, self.frame_spin.value() - 1)

    def _start_loader(self, file_path: str, frame_index: int) -> None:
        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._set_status("A file is already loading...")
            return

        self.progress.setRange(0, 0)
        self._set_status(f"Loading {Path(file_path).name}...")
        self._loader_thread = QThread(self)
        self._loader_worker = ImageLoadWorker(file_path, frame_index)
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.finished.connect(self._on_image_loaded)
        self._loader_worker.failed.connect(self._on_image_load_failed)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.failed.connect(self._loader_thread.quit)
        self._loader_thread.finished.connect(self._cleanup_loader)
        self._loader_thread.start()

    def _on_image_loaded(self, result: ImageLoadResult) -> None:
        self.current_file = result.file_path
        self.current_image = result.image
        self.current_frame_count = max(1, result.frame_count)
        self._current_view_is_cut = False
        self._cut_extent = None
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

        self.frame_spin.blockSignals(True)
        self.frame_spin.setMaximum(self.current_frame_count)
        self.frame_spin.setValue(result.frame_index + 1)
        self.frame_spin.blockSignals(False)
        self._set_frame_controls_enabled(Path(result.file_path).suffix.lower() == ".nxs")

        self._sync_selection_defaults_to_image()
        self._update_auto_colorbar_limits()
        self._show_2d_view()
        self.refresh_view()
        self._set_status(f"Loaded {Path(result.file_path).name}")

    def _on_image_load_failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_status("Failed to load file")
        QMessageBox.warning(self, "Failed to Load File", f"Failed to load file:\n{message}")

    def _cleanup_loader(self) -> None:
        self._loader_worker = None
        if self._loader_thread is not None:
            self._loader_thread.deleteLater()
        self._loader_thread = None

    def _on_frame_changed(self, value: int) -> None:
        if self.current_file and Path(self.current_file).suffix.lower() == ".nxs":
            self._start_loader(self.current_file, value - 1)

    def refresh_view(self) -> None:
        if self.current_image is None:
            return
        if self._active_view != "2d":
            return
        image = self.current_image
        extent = None
        xlabel = "X (pixel)"
        ylabel = "Y (pixel)"
        title = Path(self.current_file).name if self.current_file else "Detector Image"
        if self._current_view_is_cut:
            image, extent = self._cut_image_by_q_range(image)
            xlabel = "Qr (Å⁻¹)"
            ylabel = "Qz (Å⁻¹)"
            title = f"{title} - Cut"
        mask_min, mask_max = self._display_mask_limits()
        self.viewer.show_image(
            image,
            log_scale=self.display_log.isChecked(),
            colormap=self.display_cmap.currentText(),
            auto_scale=self.display_auto_scale.isChecked(),
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            mask_min=mask_min,
            mask_max=mask_max,
            flip_vertical=self.display_flip.isChecked(),
            title=title,
            extent=extent,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        self._draw_overlays()
        self._update_metadata(image)

    def _on_view_tab_changed(self, index: int) -> None:
        if index == 1:
            self._active_view = "1d"
            if self._last_curve is not None:
                self._plot_curve(*self._last_curve)
            else:
                self.viewer.figure.clear()
                self.viewer.ax = self.viewer.figure.add_subplot(111)
                self.viewer.cax = None
                self.viewer.colorbar = None
                self.viewer.ax.text(0.5, 0.5, "No 1D curve calculated", ha="center", va="center", transform=self.viewer.ax.transAxes)
                self.viewer.ax.set_axis_off()
                self.viewer.canvas.draw_idle()
            return
        self._active_view = "2d"
        self.refresh_view()

    def _show_2d_view(self) -> None:
        self._active_view = "2d"
        self.view_tabs.blockSignals(True)
        self.view_tabs.setCurrentIndex(0)
        self.view_tabs.blockSignals(False)

    def _show_1d_view(self) -> None:
        self._active_view = "1d"
        self.view_tabs.blockSignals(True)
        self.view_tabs.setCurrentIndex(1)
        self.view_tabs.blockSignals(False)

    def _on_log_intensity_toggled(self, checked: bool) -> None:
        self.vmin_spin.setToolTip(
            "Colorbar minimum in log10(intensity) units." if checked else "Colorbar minimum in linear intensity units."
        )
        self.vmax_spin.setToolTip(
            "Colorbar maximum in log10(intensity) units." if checked else "Colorbar maximum in linear intensity units."
        )
        self.mask_min_spin.setEnabled(not checked)
        self.mask_max_spin.setEnabled(not checked)
        self.apply_mask_check.setEnabled(not checked)
        self._update_auto_colorbar_limits()
        self.refresh_view()

    def _sync_selection_defaults_to_image(self) -> None:
        if self.current_image is None:
            return
        height, width = self.current_image.shape[:2]
        center_x = width / 2.0
        center_y = height / 2.0
        default_width = max(10.0, width * 0.25)
        default_height = max(4.0, height * 0.03)
        default_outer = max(10.0, min(width, height) * 0.25)
        default_inner = max(0.0, default_outer * 0.5)

        for spin, value in (
            (self.center_x_spin, center_x),
            (self.center_y_spin, center_y),
            (self.line_center_x_spin, center_x),
            (self.line_center_y_spin, center_y),
            (self.line_width_spin, default_width),
            (self.line_height_spin, default_height),
            (self.circle_center_x_spin, center_x),
            (self.circle_center_y_spin, center_y),
            (self.circle_inner_spin, default_inner),
            (self.circle_outer_spin, default_outer),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _set_values_without_refresh(self, updates: tuple[tuple[object, object], ...]) -> None:
        for widget, value in updates:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _set_combo_text_without_refresh(self, combo: QComboBox, text: str) -> None:
        combo.blockSignals(True)
        combo.setCurrentText(text)
        combo.blockSignals(False)

    def _on_cut_type_changed(self) -> None:
        self._update_cut_tool_visibility()
        self.refresh_view()

    def _update_cut_tool_visibility(self) -> None:
        if not hasattr(self, "_roi_layout"):
            return
        active = self.cut_type_combo.currentText() if hasattr(self, "cut_type_combo") else "Q Range"
        for controls, visible in (
            (getattr(self, "_q_range_controls", ()), active == "Q Range"),
            (getattr(self, "_line_cut_controls", ()), active == "Line Cut"),
            (getattr(self, "_circle_cut_controls", ()), active == "Circle Cut"),
        ):
            for widget in controls:
                widget.setVisible(visible)
                label = self._roi_layout.labelForField(widget)
                if label is not None:
                    label.setVisible(visible)

    def _draw_overlays(self) -> None:
        if self.current_image is None:
            return
        ax = self.viewer.ax
        if self.show_center_check.isChecked() and not self._current_view_is_cut:
            center_x = self.center_x_spin.value()
            center_y = self.center_y_spin.value()
            ax.plot(center_x, center_y, marker="+", color="#22d3ee", markersize=14, markeredgewidth=2.0)

        if self.show_cut_region_check.isChecked():
            cut_type = self.cut_type_combo.currentText()
            if cut_type == "Line Cut" and not self._current_view_is_cut:
                x0, y0, width, height = self._line_region()
                ax.add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="#f97316", linewidth=1.8))
                ax.plot(self.line_center_x_spin.value(), self.line_center_y_spin.value(), marker="x", color="#f97316", markersize=9)
            elif cut_type == "Circle Cut" and not self._current_view_is_cut:
                cx = self.circle_center_x_spin.value()
                cy = self.circle_center_y_spin.value()
                inner = self.circle_inner_spin.value()
                outer = self.circle_outer_spin.value()
                start = self.circle_start_spin.value()
                end = self.circle_end_spin.value()
                if end < start:
                    end += 360.0
                ax.add_patch(Wedge((cx, cy), outer, start, end, width=max(outer - inner, 1e-6), fill=False, edgecolor="#a855f7", linewidth=1.8))
                ax.add_patch(Circle((cx, cy), 3, fill=True, color="#a855f7"))
            elif cut_type == "Q Range" and self._current_view_is_cut:
                x0 = None if self.qr_min_spin.value() == -121.0 else self.qr_min_spin.value()
                x1 = None if self.qr_max_spin.value() == -121.0 else self.qr_max_spin.value()
                y0 = None if self.qz_min_spin.value() == -121.0 else self.qz_min_spin.value()
                y1 = None if self.qz_max_spin.value() == -121.0 else self.qz_max_spin.value()
                if None not in (x0, x1, y0, y1):
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#f97316", linewidth=1.8))

        self.viewer.canvas.draw_idle()

    def reset_mask(self) -> None:
        self._set_values_without_refresh(
            (
                (self.mask_min_spin, -1e12),
                (self.mask_max_spin, 1e12),
            )
        )
        self.apply_mask_check.blockSignals(True)
        self.apply_mask_check.setChecked(True)
        self.apply_mask_check.blockSignals(False)
        self.refresh_view()

    def apply_cut(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Apply Cut", "No image loaded.")
            return
        self._current_view_is_cut = True
        self._show_2d_view()
        self.refresh_view()

    def clear_cut(self) -> None:
        self._current_view_is_cut = False
        self._set_values_without_refresh(
            (
                (self.qr_min_spin, -121.0),
                (self.qr_max_spin, -121.0),
                (self.qz_min_spin, -121.0),
                (self.qz_max_spin, -121.0),
            )
        )
        self._show_2d_view()
        self.refresh_view()

    def _select_roi_hint(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "ROI Selection", "No image loaded.")
            return
        self._cancel_interactive_tools()
        self._current_view_is_cut = False
        self._show_2d_view()
        self.refresh_view()
        if self._roi_selector is not None:
            self._roi_selector.set_active(False)
            self._roi_selector = None
        self._roi_selector = RectangleSelector(
            self.viewer.ax,
            self._on_roi_selected,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=True,
        )
        self._set_status("Drag a rectangle on the detector image to select a Q-range ROI.")
        self.viewer.canvas.draw_idle()

    def start_line_cut_selection(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Line Cut", "No image loaded.")
            return
        self._cancel_interactive_tools()
        self._current_view_is_cut = False
        self._set_combo_text_without_refresh(self.cut_type_combo, "Line Cut")
        self._update_cut_tool_visibility()
        self._show_2d_view()
        self.refresh_view()
        self._roi_selector = RectangleSelector(
            self.viewer.ax,
            self._on_line_cut_selected,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=True,
        )
        self._set_status("Drag any rectangle on the image to define the line cut region.")
        self.viewer.canvas.draw_idle()

    def start_circle_cut_selection(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Circle Cut", "No image loaded.")
            return
        self._cancel_interactive_tools()
        self._current_view_is_cut = False
        self._set_combo_text_without_refresh(self.cut_type_combo, "Circle Cut")
        self._update_cut_tool_visibility()
        self._show_2d_view()
        self.refresh_view()
        self._circle_pick_points = [(self.circle_center_x_spin.value(), self.circle_center_y_spin.value())]
        self._circle_pick_cid = self.viewer.canvas.mpl_connect("button_press_event", self._on_circle_pick)
        self._set_status("Circle Cut: click inner/start point, then outer/end point.")

    def start_center_pick(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Pick Center", "No image loaded.")
            return
        self._cancel_interactive_tools()
        self._current_view_is_cut = False
        self._show_2d_view()
        self.refresh_view()
        self._center_pick_cid = self.viewer.canvas.mpl_connect("button_press_event", self._on_center_pick)
        self._set_status("Pick Center: click the detector image to set the center.")

    def _cancel_interactive_tools(self) -> None:
        if self._roi_selector is not None:
            self._roi_selector.set_active(False)
            self._roi_selector = None
        if self._circle_pick_cid is not None:
            self.viewer.canvas.mpl_disconnect(self._circle_pick_cid)
            self._circle_pick_cid = None
        if self._center_pick_cid is not None:
            self.viewer.canvas.mpl_disconnect(self._center_pick_cid)
            self._center_pick_cid = None

    def _on_line_cut_selected(self, press_event, release_event) -> None:
        if press_event.xdata is None or press_event.ydata is None or release_event.xdata is None or release_event.ydata is None:
            return
        x0, x1 = sorted([float(press_event.xdata), float(release_event.xdata)])
        y0, y1 = sorted([float(press_event.ydata), float(release_event.ydata)])
        self._set_values_without_refresh(
            (
                (self.line_center_x_spin, (x0 + x1) / 2.0),
                (self.line_center_y_spin, (y0 + y1) / 2.0),
                (self.line_width_spin, max(1.0, x1 - x0)),
                (self.line_height_spin, max(1.0, y1 - y0)),
            )
        )
        self._cancel_interactive_tools()
        self.refresh_view()
        self._set_status("Line cut region selected.")

    def _on_center_pick(self, event) -> None:
        if event.inaxes != self.viewer.ax or event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        self._set_values_without_refresh(
            (
                (self.center_x_spin, x),
                (self.line_center_x_spin, x),
                (self.circle_center_x_spin, x),
                (self.center_y_spin, y),
                (self.line_center_y_spin, y),
                (self.circle_center_y_spin, y),
            )
        )
        self._cancel_interactive_tools()
        self.refresh_view()
        self._set_status(f"Center picked: X={x:.2f}, Y={y:.2f}")

    def _on_circle_pick(self, event) -> None:
        if event.inaxes != self.viewer.ax or event.xdata is None or event.ydata is None:
            return
        self._circle_pick_points.append((float(event.xdata), float(event.ydata)))
        if len(self._circle_pick_points) == 1:
            self._set_values_without_refresh(
                (
                    (self.circle_center_x_spin, self._circle_pick_points[0][0]),
                    (self.circle_center_y_spin, self._circle_pick_points[0][1]),
                )
            )
            self._set_status("Circle Cut: click inner/start point.")
            self.refresh_view()
            return
        if len(self._circle_pick_points) == 2:
            cx, cy = self._circle_pick_points[0]
            x, y = self._circle_pick_points[1]
            self._set_values_without_refresh(
                (
                    (self.circle_inner_spin, max(0.0, float(np.hypot(x - cx, y - cy)))),
                    (self.circle_start_spin, self._angle_from_center(cx, cy, x, y)),
                )
            )
            self._set_status("Circle Cut: click outer/end point.")
            self.refresh_view()
            return

        cx, cy = self._circle_pick_points[0]
        x, y = self._circle_pick_points[2]
        outer = max(self.circle_inner_spin.value() + 1.0, float(np.hypot(x - cx, y - cy)))
        self._set_values_without_refresh(
            (
                (self.circle_outer_spin, outer),
                (self.circle_end_spin, self._angle_from_center(cx, cy, x, y)),
            )
        )
        self._cancel_interactive_tools()
        self.refresh_view()
        self._set_status("Circle cut region selected.")

    @staticmethod
    def _angle_from_center(cx: float, cy: float, x: float, y: float) -> float:
        return float(np.degrees(np.arctan2(y - cy, x - cx)))

    def _line_region(self) -> tuple[float, float, float, float]:
        width = max(1.0, self.line_width_spin.value())
        height = max(1.0, self.line_height_spin.value())
        x0 = self.line_center_x_spin.value() - width / 2.0
        y0 = self.line_center_y_spin.value() - height / 2.0
        return x0, y0, width, height

    def _on_roi_selected(self, press_event, release_event) -> None:
        if self.current_image is None:
            return
        if press_event.xdata is None or press_event.ydata is None or release_event.xdata is None or release_event.ydata is None:
            return

        x0, x1 = sorted([float(press_event.xdata), float(release_event.xdata)])
        y0, y1 = sorted([float(press_event.ydata), float(release_event.ydata)])

        if self._current_view_is_cut:
            self.qr_min_spin.setValue(x0)
            self.qr_max_spin.setValue(x1)
            self.qz_min_spin.setValue(y0)
            self.qz_max_spin.setValue(y1)
        else:
            height, width = self.current_image.shape[:2]
            col0 = max(0, min(width - 1, int(np.floor(x0))))
            col1 = max(0, min(width - 1, int(np.ceil(x1))))
            row0 = max(0, min(height - 1, int(np.floor(y0))))
            row1 = max(0, min(height - 1, int(np.ceil(y1))))
            if row1 < row0:
                row0, row1 = row1, row0
            if col1 < col0:
                col0, col1 = col1, col0
            qr, qz = compute_q_maps(self.current_image.shape, self._geometry_settings())
            roi_qr = qr[row0 : row1 + 1, col0 : col1 + 1]
            roi_qz = qz[row0 : row1 + 1, col0 : col1 + 1]
            if np.isfinite(roi_qr).any() and np.isfinite(roi_qz).any():
                self.qr_min_spin.setValue(float(np.nanmin(roi_qr)))
                self.qr_max_spin.setValue(float(np.nanmax(roi_qr)))
                self.qz_min_spin.setValue(float(np.nanmin(roi_qz)))
                self.qz_max_spin.setValue(float(np.nanmax(roi_qz)))

        if self._roi_selector is not None:
            self._roi_selector.set_active(False)
            self._roi_selector = None
        self._current_view_is_cut = True
        self.refresh_view()
        self._set_status("ROI selected and Q-range cut applied.")

    def integrate_current_image(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Integrate", "No image loaded.")
            return
        try:
            if self.cut_type_combo.currentText() == "Line Cut":
                x, y = line_cut_profile(
                    self.current_image,
                    self.line_center_x_spin.value(),
                    self.line_center_y_spin.value(),
                    self.line_width_spin.value(),
                    self.line_height_spin.value(),
                    *self._mask_limits(),
                )
            elif self.cut_type_combo.currentText() == "Circle Cut":
                x, y = circle_cut_profile(
                    self.current_image,
                    self.circle_center_x_spin.value(),
                    self.circle_center_y_spin.value(),
                    self.circle_inner_spin.value(),
                    self.circle_outer_spin.value(),
                    self.circle_start_spin.value(),
                    self.circle_end_spin.value(),
                    self.bin_spin.value(),
                    mode=self.integration_mode.currentText().lower(),
                    mask_min=self._mask_limits()[0],
                    mask_max=self._mask_limits()[1],
                )
            else:
                x, y = integrate_image(
                    self.current_image,
                    self._geometry_settings(),
                    self._integration_settings(),
                    *self._mask_limits(),
                )
            if self.smooth_curve_check.isChecked():
                y = smooth_curve(y)
            self._last_curve = (x, y)
            self._show_1d_view()
            self._plot_curve(x, y)
            self.integration_status.setText(f"Curve calculated: {len(x)} points.")
            self._set_status("1D integration completed")
        except Exception as exc:
            QMessageBox.warning(self, "Integration Failed", f"Failed to integrate:\n{exc}")

    def _plot_curve(self, x: np.ndarray, y: np.ndarray) -> None:
        self.viewer.figure.clear()
        self.viewer.colorbar = None
        self.viewer.cax = None
        self.viewer._preview_cache_key = None
        self.viewer._preview_cache_array = None
        self.viewer._preview_cache_extent = None
        ax = self.viewer.figure.add_subplot(111)
        self.viewer.ax = ax
        ax.plot(x, y)
        ax.set_xlabel(self.x_axis_mode.currentText())
        ax.set_ylabel("Intensity")
        ax.set_title("1D Integration")
        ax.grid(True, alpha=0.25)
        self.viewer.canvas.draw_idle()

    def export_current_curve(self) -> None:
        if self._last_curve is None:
            QMessageBox.information(self, "Export 1D", "No curve calculated.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export 1D Curve", "curve.csv", "CSV Files (*.csv)")
        if not path:
            return
        export_curve_csv(normalize_path(path), *self._last_curve)
        self._set_status("1D export completed")

    def export_current_image(self) -> None:
        if self.current_image is None:
            QMessageBox.information(self, "Export Image", "No image loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Image", "detector.png", "PNG Image (*.png)")
        if not path:
            return
        image = self.current_image
        if self._current_view_is_cut:
            image, _extent = self._cut_image_by_q_range(image)
        mask_min, mask_max = self._display_mask_limits()
        export_image_png(
            image,
            normalize_path(path),
            log_scale=self.display_log.isChecked(),
            colormap=self.display_cmap.currentText(),
            auto_scale=self.display_auto_scale.isChecked(),
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            mask_min=mask_min,
            mask_max=mask_max,
        )
        self._set_status("Export completed")

    def select_batch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.batch_folder_edit.setText(normalize_path(folder))

    def select_batch_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.batch_output_edit.setText(normalize_path(folder))

    def start_batch(self) -> None:
        if self._batch_thread is not None and self._batch_thread.isRunning():
            return
        folder = self.batch_folder_edit.text().strip()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Batch Processing", "Please select a valid input folder.")
            return
        output_folder = self.batch_output_edit.text().strip() or os.getcwd()
        settings = BatchSettings(
            folder=folder,
            pattern=self.batch_pattern_edit.text().strip() or "*.tif",
            output_folder=output_folder,
            export_images=self.batch_export_images.isChecked(),
            export_curves=self.batch_export_curves.isChecked(),
            export_background_subtracted=self.batch_export_subbg.isChecked(),
            log_scale=self.display_log.isChecked(),
            colormap=self.display_cmap.currentText(),
            auto_scale=self.display_auto_scale.isChecked(),
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            mask_min=self._display_mask_limits()[0],
            mask_max=self._display_mask_limits()[1],
            geometry=self._geometry_settings(),
            integration=self._integration_settings(),
        )
        if not (settings.export_images or settings.export_curves or settings.export_background_subtracted):
            QMessageBox.information(self, "Batch Processing", "Select at least one export option.")
            return

        self.progress.setValue(0)
        self._set_status("Batch processing started...")
        self.batch_start_button.setEnabled(False)
        self.batch_pause_button.setEnabled(True)
        self.batch_pause_button.setText("Pause")
        self.batch_stop_button.setEnabled(True)
        self._batch_thread = QThread(self)
        self._batch_worker = BatchWorker(settings)
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._cleanup_batch)
        self._batch_thread.start()

    def stop_batch(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.stop()
            self._set_status("Stopping batch processing...")

    def toggle_batch_pause(self) -> None:
        if self._batch_worker is None:
            return
        paused = self.batch_pause_button.text() == "Pause"
        self._batch_worker.set_paused(paused)
        self.batch_pause_button.setText("Resume" if paused else "Pause")
        self._set_status("Batch processing paused." if paused else "Batch processing resumed.")

    def _on_batch_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self._set_status(message)

    def _on_batch_finished(self, message: str) -> None:
        self.batch_start_button.setEnabled(True)
        self.batch_pause_button.setEnabled(False)
        self.batch_pause_button.setText("Pause")
        self.batch_stop_button.setEnabled(False)
        self.progress.setValue(100 if "completed" in message.lower() else 0)
        self._set_status(message)
        QMessageBox.information(self, "Batch Processing", message)

    def _on_batch_failed(self, message: str) -> None:
        self.batch_start_button.setEnabled(True)
        self.batch_pause_button.setEnabled(False)
        self.batch_pause_button.setText("Pause")
        self.batch_stop_button.setEnabled(False)
        self.progress.setValue(0)
        self._set_status("Batch processing failed")
        QMessageBox.warning(self, "Batch Processing Failed", message)

    def _cleanup_batch(self) -> None:
        self._batch_worker = None
        if self._batch_thread is not None:
            self._batch_thread.deleteLater()
        self._batch_thread = None

    def _set_frame_controls_enabled(self, enabled: bool) -> None:
        self.frame_label.setVisible(enabled)
        self.frame_spin.setVisible(enabled)
        self.frame_spin.setEnabled(enabled)

    def _mask_limits(self) -> tuple[float, float]:
        if not self.apply_mask_check.isChecked():
            return -1e12, 1e12
        return self.mask_min_spin.value(), self.mask_max_spin.value()

    def _display_mask_limits(self) -> tuple[float, float]:
        """Mask thresholds are defined in linear intensity space only."""
        if self.display_log.isChecked():
            return -1e12, 1e12
        return self._mask_limits()

    def _geometry_settings(self) -> dict:
        return {
            "incidence": self.incidence_spin.value(),
            "center_x": self.center_x_spin.value(),
            "center_y": self.center_y_spin.value(),
            "distance": self.distance_spin.value(),
            "pixel_x": self.pixel_x_spin.value(),
            "pixel_y": self.pixel_y_spin.value(),
            "wavelength": self.wavelength_spin.value(),
            "qr_min": self.qr_min_spin.value(),
            "qr_max": self.qr_max_spin.value(),
            "qz_min": self.qz_min_spin.value(),
            "qz_max": self.qz_max_spin.value(),
        }

    def _integration_settings(self) -> dict:
        return {
            "mode": self.integration_mode.currentText().lower(),
            "bins": self.bin_spin.value(),
            "x_axis": self.x_axis_mode.currentText().lower(),
        }

    def _cut_image_by_q_range(self, image: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
        qr, qz = compute_q_maps(image.shape, self._geometry_settings())
        mask = np.ones(image.shape, dtype=bool)
        for value, op, grid in (
            (self.qr_min_spin.value(), np.greater_equal, qr),
            (self.qr_max_spin.value(), np.less_equal, qr),
            (self.qz_min_spin.value(), np.greater_equal, qz),
            (self.qz_max_spin.value(), np.less_equal, qz),
        ):
            if value != -121.0:
                mask &= op(grid, value)
        cut = np.where(mask, image, np.nan)
        finite_qr = qr[np.isfinite(qr)]
        finite_qz = qz[np.isfinite(qz)]
        if finite_qr.size and finite_qz.size:
            extent = (float(np.nanmin(finite_qr)), float(np.nanmax(finite_qr)), float(np.nanmin(finite_qz)), float(np.nanmax(finite_qz)))
        else:
            extent = None
        return cut, extent

    def _update_auto_colorbar_limits(self) -> None:
        if self.current_image is None:
            return
        limits = self.viewer.display_limits(
            self.current_image,
            log_scale=self.display_log.isChecked(),
            mask_min=self._display_mask_limits()[0],
            mask_max=self._display_mask_limits()[1],
            flip_vertical=False,
        )
        if limits is None:
            return
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setValue(limits[0])
        self.vmax_spin.setValue(limits[1])
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)

    def _update_metadata(self, image: np.ndarray) -> None:
        arr = np.asarray(image, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            minmax = f"min/max: {np.nanmin(finite):.6g} / {np.nanmax(finite):.6g}"
        else:
            minmax = "min/max: n/a"
        name = Path(self.current_file).name if self.current_file else "No file"
        self.meta_label.setText(
            f"File: {name} | size: {arr.shape[1]} × {arr.shape[0]} | "
            f"frame: {self.frame_spin.value()} / {self.current_frame_count} | {minmax}"
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(message)


def detect_nxs_frame_count(file_path: str) -> int:
    if Path(file_path).suffix.lower() != ".nxs":
        return 1
    try:
        with h5py.File(file_path, "r") as handle:
            dataset = handle["/entry/instrument/detector/data"]
            if dataset.ndim == 3:
                return int(dataset.shape[0])
    except Exception:
        return 1
    return 1


def load_image_matrix(
    file_path: str,
    frame_idx: int = 0,
    dataset_path: str = "/entry/instrument/detector/data",
    dist_path: str = "/entry/instrument/detector/translation/distance",
    mask_path: str = "/entry/instrument/detector/pixel_mask",
) -> np.ndarray:
    """Load supported detector data without requiring the legacy top-level GUI.

    The logic mirrors the supported parts of ``WAXS.WAXS.load_image_matrix``:
    TIFF files are loaded as 2D matrices, and NXS files support P03-style
    multi-module stitching plus frame selection.
    """
    del dist_path, mask_path
    return load_detector_image(file_path, frame_idx=frame_idx, dataset_path=dataset_path).data


def load_tiff_matrix(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as image:
            arr = np.asarray(image)
    except Exception:
        import matplotlib.pyplot as plt

        arr = plt.imread(str(path))
    if arr.ndim == 3:
        arr = np.mean(arr[..., :3], axis=2)
    return np.asarray(arr, dtype=np.float32)


def make_double_spin(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(9)
    spin.setSingleStep(0.1)
    spin.setValue(value)
    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return spin


def prepare_display_array(
    image: np.ndarray,
    *,
    log_scale: bool,
    mask_min: float,
    mask_max: float,
    flip_vertical: bool,
) -> np.ndarray:
    source = np.asarray(image)
    arr = np.asarray(source, dtype=np.float32).copy()
    if log_scale:
        valid = np.isfinite(arr) & (arr >= mask_min) & (arr <= mask_max) & (arr > 0)
        arr[~valid] = np.nan
        np.log10(arr, out=arr, where=valid)
    else:
        invalid = ~np.isfinite(arr) | (arr < mask_min) | (arr > mask_max)
        arr[invalid] = np.nan
    if flip_vertical:
        arr = np.flipud(arr)
    return arr


def estimate_display_limits(
    image: np.ndarray,
    *,
    log_scale: bool,
    mask_min: float,
    mask_max: float,
    max_samples: int = 200_000,
    stride_hint: int = 20,
) -> tuple[float, float] | None:
    arr = np.asarray(image)
    flat = arr.ravel()
    if flat.size == 0:
        return None
    stride = max(int(stride_hint), int(np.ceil(flat.size / max_samples)))
    sample = np.asarray(flat[::stride], dtype=np.float32)
    if sample.size == 0:
        return None
    valid = np.isfinite(sample) & (sample >= mask_min) & (sample <= mask_max)
    if log_scale:
        valid &= sample > 0
    sample = sample[valid]
    if sample.size == 0:
        return None
    if log_scale:
        np.log10(sample, out=sample)
    return percentile_limits(sample)


def percentile_limits(arr: np.ndarray) -> tuple[float, float] | None:
    vals = np.asarray(arr, dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        return None
    if finite.all():
        vals = vals.ravel()
    else:
        vals = vals[finite]
    lo, hi = np.nanpercentile(vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if lo == hi:
        hi = lo + 1e-9
    return float(lo), float(hi)


def compute_q_maps(shape: tuple[int, int], geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape[:2]
    x_center = float(geometry["center_x"])
    y_center = float(geometry["center_y"])
    distance = float(geometry["distance"])
    pixel_x = float(geometry["pixel_x"])
    pixel_y = float(geometry["pixel_y"])
    wavelength = float(geometry["wavelength"])
    incidence = float(geometry["incidence"]) * np.pi / 180.0

    yy, xx = np.indices((height, width), dtype=float)
    qr_pix = (xx + 1.0) - x_center
    y_c = height - y_center
    qz_pix = (height - y_c) - (yy + 1.0)
    qr_m = qr_pix * pixel_x * 1e-6
    qz_m = qz_pix * pixel_y * 1e-6
    theta_f = np.arctan(qr_m / (distance * 1e-3)) / 2.0
    alpha_f = np.arctan(qz_m / np.sqrt((distance * 1e-3) ** 2 + qr_m**2))
    qx = 2 * np.pi / wavelength * (np.cos(2 * theta_f) * np.cos(alpha_f) - np.cos(incidence))
    qy = 2 * np.pi / wavelength * (np.sin(2 * theta_f) * np.cos(alpha_f))
    qz = 2 * np.pi / wavelength * (np.sin(alpha_f) + np.sin(incidence))
    qr = np.sign(qy) * np.sqrt(qx**2 + qy**2)
    return qr, qz


def integrate_image(
    image: np.ndarray,
    geometry: dict,
    integration: dict,
    mask_min: float,
    mask_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=float)
    valid = np.isfinite(arr) & (arr >= mask_min) & (arr <= mask_max)

    qr, qz = compute_q_maps(arr.shape, geometry)
    for key, op, grid in (
        ("qr_min", np.greater_equal, qr),
        ("qr_max", np.less_equal, qr),
        ("qz_min", np.greater_equal, qz),
        ("qz_max", np.less_equal, qz),
    ):
        value = float(geometry.get(key, -121.0))
        if value != -121.0:
            valid &= op(grid, value)

    mode = integration.get("mode", "radial")
    bins = int(integration.get("bins", 500))
    axis_mode = integration.get("x_axis", "q")

    if mode == "azimuthal":
        yy, xx = np.indices(arr.shape, dtype=float)
        x_values = np.degrees(np.arctan2(yy - geometry["center_y"], xx - geometry["center_x"]))
    elif axis_mode == "pixel":
        yy, xx = np.indices(arr.shape, dtype=float)
        x_values = np.sqrt((xx - geometry["center_x"]) ** 2 + (yy - geometry["center_y"]) ** 2)
    elif axis_mode == "2theta":
        yy, xx = np.indices(arr.shape, dtype=float)
        radius_m = np.sqrt(((xx - geometry["center_x"]) * geometry["pixel_x"] * 1e-6) ** 2 + ((yy - geometry["center_y"]) * geometry["pixel_y"] * 1e-6) ** 2)
        x_values = np.degrees(np.arctan(radius_m / (geometry["distance"] * 1e-3)))
    else:
        x_values = np.sqrt(qr**2 + qz**2)

    x = x_values[valid]
    y = arr[valid]
    if x.size == 0:
        raise RuntimeError("No valid pixels in the selected integration region.")

    edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), bins + 1)
    if edges[0] == edges[-1]:
        edges[-1] = edges[0] + 1e-9
    indices = np.digitize(x, edges) - 1
    valid_bins = (indices >= 0) & (indices < bins)
    indices = indices[valid_bins]
    y = y[valid_bins]
    sums = np.bincount(indices, weights=y, minlength=bins)
    counts = np.bincount(indices, minlength=bins)
    means = np.divide(sums, counts, out=np.full(bins, np.nan, dtype=float), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    finite = np.isfinite(means)
    return centers[finite], means[finite]


def line_cut_profile(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    mask_min: float,
    mask_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=float)
    h, w = arr.shape[:2]
    x0 = max(0, int(np.floor(center_x - width / 2.0)))
    x1 = min(w, int(np.ceil(center_x + width / 2.0)))
    y0 = max(0, int(np.floor(center_y - height / 2.0)))
    y1 = min(h, int(np.ceil(center_y + height / 2.0)))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Line cut region is empty.")
    region = arr[y0:y1, x0:x1].copy()
    region[(region < mask_min) | (region > mask_max)] = np.nan
    if width >= height:
        y = np.nanmean(region, axis=0)
        x = np.arange(x0, x1, dtype=float)
    else:
        y = np.nanmean(region, axis=1)
        x = np.arange(y0, y1, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        raise RuntimeError("No valid pixels in the selected line cut.")
    return x[finite], y[finite]


def circle_cut_profile(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
    start_angle: float,
    end_angle: float,
    bins: int,
    *,
    mode: str,
    mask_min: float,
    mask_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=float)
    start = normalize_angle_deg(float(start_angle))
    end = normalize_angle_deg(float(end_angle))
    inner = min(float(inner_radius), float(outer_radius))
    outer = max(float(inner_radius), float(outer_radius))
    h, w = arr.shape[:2]
    cx = float(center_x)
    cy = float(center_y)
    x0 = max(0, int(np.floor(cx - outer)))
    x1 = min(w, int(np.ceil(cx + outer)) + 1)
    y0 = max(0, int(np.floor(cy - outer)))
    y1 = min(h, int(np.ceil(cy + outer)) + 1)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Circle cut region is outside the image.")

    region = arr[y0:y1, x0:x1]
    yy, xx = np.indices(region.shape, dtype=np.float32)
    xx += float(x0)
    yy += float(y0)
    dx = xx - cx
    dy = yy - cy
    radius = np.hypot(dx, dy)
    angle = normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
    sector = angle_between(angle, start, end)
    valid = (
        np.isfinite(region)
        & (region >= mask_min)
        & (region <= mask_max)
        & (radius >= inner)
        & (radius <= outer)
        & sector
    )
    if not np.any(valid):
        raise RuntimeError("No valid pixels in the selected circle cut.")

    x_values = angle if mode == "azimuthal" else radius
    x = x_values[valid]
    y = region[valid]
    bins = int(bins)
    edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), bins + 1)
    if edges[0] == edges[-1]:
        edges[-1] = edges[0] + 1e-9
    indices = np.digitize(x, edges) - 1
    valid_bins = (indices >= 0) & (indices < bins)
    indices = indices[valid_bins]
    y = y[valid_bins]
    sums = np.bincount(indices, weights=y, minlength=bins)
    counts = np.bincount(indices, minlength=bins)
    means = np.divide(sums, counts, out=np.full(bins, np.nan, dtype=float), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    finite = np.isfinite(means)
    return centers[finite], means[finite]


def normalize_angle_deg(angle):
    return (np.asarray(angle) + 360.0) % 360.0


def angle_between(angle: np.ndarray, start: float, end: float) -> np.ndarray:
    if start <= end:
        return (angle >= start) & (angle <= end)
    return (angle >= start) | (angle <= end)


def smooth_curve(y: np.ndarray, window: int = 7) -> np.ndarray:
    if y.size < window:
        return y
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def export_curve_csv(path: str, x: np.ndarray, y: np.ndarray) -> None:
    arr = np.column_stack([x, y])
    np.savetxt(path, arr, delimiter=",", header="x,intensity", comments="", fmt="%.9g")


def write_matrix_csv(path: str, columns: list[np.ndarray], headers: list[str]) -> None:
    max_len = max((len(col) for col in columns), default=0)
    padded = [
        np.pad(np.asarray(col, dtype=float).ravel(), (0, max_len - len(col)), constant_values=np.nan)
        for col in columns
    ]
    matrix = np.column_stack(padded) if padded else np.empty((0, 0))
    np.savetxt(path, matrix, delimiter=",", header=",".join(headers), comments="", fmt="%.9g")


def export_image_png(
    image: np.ndarray,
    path: str,
    *,
    log_scale: bool,
    colormap: str,
    auto_scale: bool,
    vmin: float,
    vmax: float,
    mask_min: float,
    mask_max: float,
) -> None:
    import matplotlib.pyplot as plt

    if auto_scale:
        limits = estimate_display_limits(
            image,
            log_scale=log_scale,
            mask_min=mask_min,
            mask_max=mask_max,
        )
        if limits is not None:
            vmin, vmax = limits
    arr = prepare_display_array(
        image,
        log_scale=log_scale,
        mask_min=mask_min,
        mask_max=mask_max,
        flip_vertical=False,
    )
    fig, ax = plt.subplots()
    cmap = colormaps.get_cmap(colormap).copy()
    cmap.set_bad(cmap(0.0))
    artist = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    fig.colorbar(artist, ax=ax)
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
