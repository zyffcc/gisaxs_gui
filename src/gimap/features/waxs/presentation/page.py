"""Embedded in-situ scattering data processing page for the main GUI.

This module intentionally does not instantiate the legacy ``WAXS.WAXS.MainWindow``.
It reuses the old loader and keeps the page embeddable in the existing
``Ui_MainWindow`` stacked widget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import colormaps
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QMessageBox,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.widgets import RectangleSelector

from src.gimap.app.presentation import apply_design_system
from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)
from src.gimap.features.waxs.application import (
    IntegrateWaxsImageRequest,
    WaxsBatchRequest,
)

from .views import (
    WaxsAdvancedPanelView,
    WaxsBatchPanelView,
    WaxsConfigurePanelView,
    WaxsIntegrationPanelView,
    WaxsPageView,
    WaxsPreviewPanelView,
    WaxsRoiPanelView,
    WaxsToolbarView,
)


SUPPORTED_EXTENSIONS = {".nxs", ".tif", ".tiff"}
SCATTERING_FILTER = "Scattering Data (*.nxs *.tif *.tiff)"


@dataclass
class ImageLoadResult:
    file_path: str
    frame_index: int
    frame_count: int
    image: np.ndarray


class ImageLoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, frame_index: int, view_model):
        super().__init__()
        self.file_path = file_path
        self.frame_index = int(frame_index)
        self.view_model = view_model

    def run(self) -> None:
        try:
            loaded = self.view_model.load_image(Path(self.file_path), self.frame_index)
            if loaded is None:
                raise RuntimeError(
                    self.view_model.state.error_message or "Failed to load image."
                )
            self.finished.emit(
                ImageLoadResult(
                    file_path=str(loaded.path),
                    frame_index=loaded.frame_index,
                    frame_count=loaded.frame_count,
                    image=loaded.image,
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class BatchWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, request: WaxsBatchRequest, view_model):
        super().__init__()
        self.request = request
        self.view_model = view_model

    def stop(self) -> None:
        self.view_model.cancel_batch()

    def set_paused(self, paused: bool) -> None:
        self.view_model.set_batch_paused(paused)

    def run(self) -> None:
        try:
            def report(value) -> None:
                total = max(1, int(value.total))
                percent = int(round(int(value.completed) * 100 / total))
                self.progress.emit(percent, f"Processed {value.name}")

            result = self.view_model.run_batch(self.request, on_progress=report)
            if result is None:
                raise RuntimeError(
                    self.view_model.state.error_message
                    or "Batch processing failed."
                )
            if result.cancelled:
                self.finished.emit("Batch stopped by user.")
            elif result.failed_count:
                failures = [
                    item.error_message or item.name
                    for item in result.items
                    if item.status == "failed"
                ]
                self.failed.emit("; ".join(failures))
            else:
                self.finished.emit("Batch processing completed.")
        except Exception as exc:
            self.failed.emit(str(exc))


class ScatteringImageViewer(QWidget):
    fileDropped = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *, view_model=None):
        super().__init__(parent)
        if view_model is None:
            raise ValueError("ScatteringImageViewer requires WaxsViewModel")
        self.view_model = view_model
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
        preview = self.view_model.prepare_display(
            preview_source,
            log_scale=log_scale,
            mask_min=mask_min,
            mask_max=mask_max,
            flip_vertical=flip_vertical,
        )
        preview = np.ascontiguousarray(preview)
        if auto_scale:
            limits_start = time.perf_counter()
            limits = self.view_model.estimate_display_limits(
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
        limits = self.view_model.estimate_display_limits(
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
            self.fileDropped.emit(self.view_model.normalize_path(path))


class InSituProcessingWidget(QWidget, WaxsPageView):
    """Modern embedded replacement for the legacy in-situ data window."""

    statusChanged = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *, view_model=None):
        super().__init__(parent)
        if view_model is None:
            raise ValueError("InSituProcessingWidget requires WaxsViewModel")
        self.view_model = view_model
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

        self.setupUi(self)
        self._bind_form()
        self._connect_signals()
        self._set_frame_controls_enabled(False)
        self._set_status("Ready")

    def _bind_form(self) -> None:
        """Bind Python Views to the preserved WAXS behavior."""
        self.waxs_input_section = self.waxsInputSection
        self.waxs_preview_panel = self.waxsPreviewPanel
        self.waxs_configure_section = self.waxsConfigureSection
        self.waxs_advanced_section = self.waxsAdvancedSection
        self.waxs_run_section = self.waxsRunSection
        self.waxs_results_section = self.waxsResultsSection
        self.waxs_export_section = self.waxsExportSection
        self.controls_scroll = self.waxsControlsScrollArea

        for section, title, description, content, layout in (
            (
                self.waxs_input_section,
                self.waxsInputTitle,
                self.waxsInputDescription,
                self.waxsInputContent,
                self.waxsInputContentLayout,
            ),
            (
                self.waxs_preview_panel,
                self.waxsPreviewTitle,
                self.waxsPreviewDescription,
                self.waxsPreviewContent,
                self.waxsPreviewContentLayout,
            ),
            (
                self.waxs_configure_section,
                self.waxsConfigureTitle,
                self.waxsConfigureDescription,
                self.waxsConfigureContent,
                self.waxsConfigureContentLayout,
            ),
            (
                self.waxs_run_section,
                self.waxsRunTitle,
                self.waxsRunDescription,
                self.waxsRunContent,
                self.waxsRunContentLayout,
            ),
            (
                self.waxs_results_section,
                self.waxsResultsTitle,
                self.waxsResultsDescription,
                self.waxsResultsContent,
                self.waxsResultsContentLayout,
            ),
            (
                self.waxs_export_section,
                self.waxsExportTitle,
                self.waxsExportDescription,
                self.waxsExportContent,
                self.waxsExportContentLayout,
            ),
        ):
            bind_parameter_section(section, title, description, content, layout)
        bind_advanced_section(
            self.waxs_advanced_section,
            self.waxsAdvancedToggle,
            self.waxsAdvancedDescription,
            self.waxsAdvancedContent,
            self.waxsAdvancedContentLayout,
        )

        toolbar = QFrame(self.waxsInputContent)
        toolbar_ui = WaxsToolbarView()
        toolbar_ui.setupUi(toolbar)
        self._toolbar_ui = toolbar_ui
        self._expose_form(
            toolbar_ui,
            (
                "open_button",
                "reload_button",
                "frame_label",
                "frame_spin",
                "toolbar_auto_scale",
                "toolbar_log_scale",
                "toolbar_cmap",
            ),
        )
        self.waxsInputContentLayout.addWidget(toolbar)

        preview = QWidget(self.waxsPreviewContent)
        preview_ui = WaxsPreviewPanelView()
        preview_ui.setupUi(preview)
        self._preview_ui = preview_ui
        self.view_tabs = preview_ui.waxsViewTabs
        self.view_tabs.addTab("2D Image")
        self.view_tabs.addTab("1D Curve")
        self.viewer = ScatteringImageViewer(preview, view_model=self.view_model)
        preview_ui.viewerHostLayout.addWidget(self.viewer)
        self.meta_label = preview_ui.waxsMetadataLabel
        self.waxsPreviewContentLayout.addWidget(preview, 1)

        self.tabs = QTabWidget(self.waxsConfigureContent)
        configure_ui = WaxsConfigurePanelView()
        configure_ui.setupUi(self.tabs)
        self._configure_ui = configure_ui
        self.waxsConfigureContentLayout.addWidget(self.tabs)

        roi_panel = QWidget(self.tabs)
        roi_ui = WaxsRoiPanelView()
        roi_ui.setupUi(roi_panel)
        self._roi_ui = roi_ui
        self._expose_form(
            roi_ui,
            (
                "cut_type_combo",
                "show_cut_region_check",
                "show_center_check",
                "pick_center_button",
                "q_range_header",
                "qr_min_spin",
                "qr_max_spin",
                "qz_min_spin",
                "qz_max_spin",
                "qRangeHint",
                "select_roi_button",
                "line_cut_header",
                "line_center_x_spin",
                "line_center_y_spin",
                "line_width_spin",
                "line_height_spin",
                "select_line_button",
                "circle_cut_header",
                "circle_center_x_spin",
                "circle_center_y_spin",
                "circle_inner_spin",
                "circle_outer_spin",
                "circle_start_spin",
                "circle_end_spin",
                "select_circle_button",
                "clear_roi_button",
                "apply_cut_button",
            ),
        )
        configure_ui.roiTabLayout.addWidget(roi_panel)

        integration_panel = QWidget(self.tabs)
        integration_ui = WaxsIntegrationPanelView()
        integration_ui.setupUi(integration_panel)
        self._integration_ui = integration_ui
        self._expose_form(
            integration_ui,
            (
                "integration_mode",
                "bin_spin",
                "smooth_curve_check",
                "x_axis_mode",
                "integrate_button",
            ),
        )
        configure_ui.integrationTabLayout.addWidget(integration_panel)

        self.advanced_tabs = QTabWidget(self.waxsAdvancedContent)
        advanced_ui = WaxsAdvancedPanelView()
        advanced_ui.setupUi(self.advanced_tabs)
        self._advanced_ui = advanced_ui
        self._expose_form(
            advanced_ui,
            (
                "vmin_spin",
                "vmax_spin",
                "display_auto_scale",
                "display_log",
                "display_cmap",
                "display_flip",
                "mask_min_spin",
                "mask_max_spin",
                "bad_pixel_spin",
                "apply_mask_check",
                "reset_mask_button",
                "incidence_spin",
                "center_x_spin",
                "center_y_spin",
                "distance_spin",
                "pixel_x_spin",
                "pixel_y_spin",
                "wavelength_spin",
            ),
        )
        self.waxsAdvancedContentLayout.addWidget(self.advanced_tabs)

        batch_panel = QWidget(self.waxsRunContent)
        batch_ui = WaxsBatchPanelView()
        batch_ui.setupUi(batch_panel)
        self._batch_ui = batch_ui
        self._expose_form(
            batch_ui,
            (
                "batch_folder_edit",
                "batch_browse_button",
                "batch_pattern_edit",
                "batch_output_edit",
                "batch_output_browse_button",
                "batch_export_images",
                "batch_export_curves",
                "batch_export_subbg",
                "batch_start_button",
                "batch_pause_button",
                "batch_stop_button",
            ),
        )
        self.batch_output_edit.setText(self.view_model.working_directory())
        self.waxsRunContentLayout.insertWidget(0, batch_panel)

        self.status_label = self.waxs_job_status.message_label
        self.progress = self.waxs_job_status.progress_bar
        self.waxs_job_status.set_actions_visible(
            pause=False,
            cancel=False,
            details=False,
        )
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._last_curve: tuple[np.ndarray, np.ndarray] | None = None

        self._roi_layout = roi_ui.roi_layout
        self._q_range_controls = (
            self.q_range_header,
            self.qr_min_spin,
            self.qr_max_spin,
            self.qz_min_spin,
            self.qz_max_spin,
            self.qRangeHint,
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
        self._update_cut_tool_visibility()
        self.waxsContentSplitter.setStretchFactor(0, 5)
        self.waxsContentSplitter.setStretchFactor(1, 0)
        apply_design_system(self)

    def _expose_form(self, form, names: tuple[str, ...]) -> None:
        """Expose stable widget attributes from one generated subform."""
        for name in names:
            setattr(self, name, getattr(form, name))

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
            self.load_file(self.view_model.normalize_path(file_path))

    def load_file(self, file_path: str, frame_index: int = 0) -> None:
        suffix = Path(file_path).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            QMessageBox.warning(self, "Unsupported File Type", "Unsupported file type. Please select .nxs, .tif, or .tiff.")
            return
        self.current_file = self.view_model.normalize_path(file_path)
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

        self.set_job_state(
            "running",
            f"Loading {Path(file_path).name}...",
        )
        self._loader_thread = QThread(self)
        self._loader_worker = ImageLoadWorker(
            file_path,
            frame_index,
            self.view_model,
        )
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
        self.frame_spin.blockSignals(True)
        self.frame_spin.setMaximum(self.current_frame_count)
        self.frame_spin.setValue(result.frame_index + 1)
        self.frame_spin.blockSignals(False)
        self._set_frame_controls_enabled(Path(result.file_path).suffix.lower() == ".nxs")

        self._sync_selection_defaults_to_image()
        self._update_auto_colorbar_limits()
        self._show_2d_view()
        self.refresh_view()
        self.set_job_state(
            "succeeded",
            f"Loaded {Path(result.file_path).name}",
            progress=100,
        )

    def _on_image_load_failed(self, message: str) -> None:
        self.set_job_state("failed", "Failed to load file", progress=0)
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
            qr, qz = self.view_model.compute_q_maps(
                self.current_image.shape,
                self._geometry_settings(),
            )
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
            cut_kind = "full"
            selection = None
            if self.cut_type_combo.currentText() == "Line Cut":
                cut_kind = "line"
                selection = {
                    "center_x": self.line_center_x_spin.value(),
                    "center_y": self.line_center_y_spin.value(),
                    "width": self.line_width_spin.value(),
                    "height": self.line_height_spin.value(),
                }
            elif self.cut_type_combo.currentText() == "Circle Cut":
                cut_kind = "circle"
                selection = {
                    "center_x": self.circle_center_x_spin.value(),
                    "center_y": self.circle_center_y_spin.value(),
                    "inner_radius": self.circle_inner_spin.value(),
                    "outer_radius": self.circle_outer_spin.value(),
                    "start_angle": self.circle_start_spin.value(),
                    "end_angle": self.circle_end_spin.value(),
                }
            integration = self._integration_settings()
            integration["smooth"] = self.smooth_curve_check.isChecked()
            curve = self.view_model.integrate(
                IntegrateWaxsImageRequest(
                    image=self.current_image,
                    geometry=self._geometry_settings(),
                    integration=integration,
                    mask_min=self._mask_limits()[0],
                    mask_max=self._mask_limits()[1],
                    cut_kind=cut_kind,
                    selection=selection,
                )
            )
            if curve is None:
                raise RuntimeError(
                    self.view_model.state.error_message or "Integration failed."
                )
            x, y = curve.x, curve.intensity
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
        exported = self.view_model.export_curve(
            Path(self.view_model.normalize_path(path))
        )
        if exported is None:
            QMessageBox.warning(
                self,
                "Export Failed",
                self.view_model.state.error_message or "Failed to export curve.",
            )
            return
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
        exported = self.view_model.export_image(
            Path(self.view_model.normalize_path(path)),
            image,
            {
                "log_scale": self.display_log.isChecked(),
                "colormap": self.display_cmap.currentText(),
                "auto_scale": self.display_auto_scale.isChecked(),
                "vmin": self.vmin_spin.value(),
                "vmax": self.vmax_spin.value(),
                "mask_min": mask_min,
                "mask_max": mask_max,
            },
        )
        if exported is None:
            QMessageBox.warning(
                self,
                "Export Failed",
                self.view_model.state.error_message or "Failed to export image.",
            )
            return
        self._set_status("Export completed")

    def select_batch_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.batch_folder_edit.setText(self.view_model.normalize_path(folder))

    def select_batch_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.batch_output_edit.setText(self.view_model.normalize_path(folder))

    def start_batch(self) -> None:
        if self._batch_thread is not None and self._batch_thread.isRunning():
            return
        folder = self.batch_folder_edit.text().strip()
        if not self.view_model.is_directory(folder):
            QMessageBox.warning(self, "Batch Processing", "Please select a valid input folder.")
            return
        output_folder = (
            self.batch_output_edit.text().strip()
            or self.view_model.working_directory()
        )
        request = WaxsBatchRequest(
            folder=Path(folder),
            pattern=self.batch_pattern_edit.text().strip() or "*.tif",
            output_folder=Path(output_folder),
            export_images=self.batch_export_images.isChecked(),
            export_curves=self.batch_export_curves.isChecked(),
            export_background_subtracted=self.batch_export_subbg.isChecked(),
            display={
                "log_scale": self.display_log.isChecked(),
                "colormap": self.display_cmap.currentText(),
                "auto_scale": self.display_auto_scale.isChecked(),
                "vmin": self.vmin_spin.value(),
                "vmax": self.vmax_spin.value(),
                "mask_min": self._display_mask_limits()[0],
                "mask_max": self._display_mask_limits()[1],
            },
            mask_min=self._display_mask_limits()[0],
            mask_max=self._display_mask_limits()[1],
            geometry=self._geometry_settings(),
            integration=self._integration_settings(),
            continue_on_error=False,
        )
        if not (
            request.export_images
            or request.export_curves
            or request.export_background_subtracted
        ):
            QMessageBox.information(self, "Batch Processing", "Select at least one export option.")
            return

        self.set_job_state(
            "running",
            "Batch processing started...",
            progress=0,
        )
        self.batch_start_button.setEnabled(False)
        self.batch_pause_button.setEnabled(True)
        self.batch_pause_button.setText("Pause")
        self.batch_stop_button.setEnabled(True)
        self._batch_thread = QThread(self)
        self._batch_worker = BatchWorker(request, self.view_model)
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
            self.set_job_state(
                "running",
                "Stopping batch processing...",
                progress=self.progress.value(),
            )

    def toggle_batch_pause(self) -> None:
        if self._batch_worker is None:
            return
        paused = self.batch_pause_button.text() == "Pause"
        self._batch_worker.set_paused(paused)
        self.batch_pause_button.setText("Resume" if paused else "Pause")
        self.set_job_state(
            "paused" if paused else "running",
            "Batch processing paused." if paused else "Batch processing resumed.",
            progress=self.progress.value(),
        )

    def _on_batch_progress(self, value: int, message: str) -> None:
        self.set_job_state("running", message, progress=value)

    def _on_batch_finished(self, message: str) -> None:
        self.batch_start_button.setEnabled(True)
        self.batch_pause_button.setEnabled(False)
        self.batch_pause_button.setText("Pause")
        self.batch_stop_button.setEnabled(False)
        completed = "completed" in message.lower()
        self.set_job_state(
            "succeeded" if completed else "cancelled",
            message,
            progress=100 if completed else 0,
        )
        QMessageBox.information(self, "Batch Processing", message)

    def _on_batch_failed(self, message: str) -> None:
        self.batch_start_button.setEnabled(True)
        self.batch_pause_button.setEnabled(False)
        self.batch_pause_button.setText("Pause")
        self.batch_stop_button.setEnabled(False)
        self.set_job_state("failed", "Batch processing failed", progress=0)
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
        result = self.view_model.cut_image(image, self._geometry_settings())
        return result.image, result.extent

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

    def set_job_state(
        self,
        state: str,
        message: str,
        *,
        progress: int | None = None,
    ) -> None:
        """Update shared status presentation while retaining 0–100 aliases。"""

        normalized_progress = None if progress is None else progress / 100.0
        self.waxs_job_status.set_state(
            state,
            message,
            progress=normalized_progress,
        )
        if progress is not None:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(progress))))
        self.statusChanged.emit(message)


def make_double_spin(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(9)
    spin.setSingleStep(0.1)
    spin.setValue(value)
    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return spin


__all__ = [
    "BatchWorker",
    "ImageLoadResult",
    "ImageLoadWorker",
    "InSituProcessingWidget",
    "SCATTERING_FILTER",
    "SUPPORTED_EXTENSIONS",
    "ScatteringImageViewer",
    "make_double_spin",
]
