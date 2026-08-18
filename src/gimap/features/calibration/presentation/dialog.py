"""Feature-owned PyQt presentation for geometry calibration."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from PyQt5.QtCore import QObject, QSignalBlocker, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QTableWidgetItem,
)

from ..application import (
    AmbiguousImageDatasetError,
    CalibrationCancelledError,
)
from src.gimap.app.bootstrap import create_standalone_legacy_context
from src.gimap.app.presentation import apply_design_system
from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)
from ..domain import (
    CalibrationCandidate,
    CalibrationResult,
    DetectorImage,
)
from src.gimap.app.presentation.assets import app_icon

from .views import GeometryCalibrationDialogView


LOGGER = logging.getLogger(__name__)

CENTER_COLOR = "#ff4d8d"
DETECTED_RING_COLOR = "#00d9ff"
MATCHED_RING_COLOR = "#ffd54a"
UNMATCHED_RING_COLOR = "#ff8a3d"


class ImageLoaderWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, path: str, view_model, dataset_path: Optional[str] = None):
        super().__init__()
        self.path = path
        self.view_model = view_model
        self.dataset_path = dataset_path

    def run(self) -> None:
        try:
            self.finished.emit(self.view_model.load_image(self.path, self.dataset_path))
        except Exception as exc:
            LOGGER.exception("Failed to load calibration image")
            self.failed.emit(exc)


class CalibrationWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, view_model, options: dict):
        super().__init__()
        self.view_model = view_model
        self.options = options
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        try:
            self.finished.emit(
                self.view_model.run_calibration(
                    self.options,
                    progress=lambda value, stage: self.progress.emit(value, stage),
                    cancelled=lambda: self.cancel_requested,
                )
            )
        except Exception as exc:
            if not isinstance(exc, CalibrationCancelledError):
                LOGGER.exception("Geometry calibration failed")
            self.failed.emit(exc)


class GeometryCalibrationDialog(QDialog, GeometryCalibrationDialogView):
    calibrationApplied = pyqtSignal(object)

    @property
    def image(self) -> Optional[DetectorImage]:
        return self.view_model.image

    @image.setter
    def image(self, value: Optional[DetectorImage]) -> None:
        self.view_model.image = value

    @property
    def result(self) -> Optional[CalibrationResult]:
        return self.view_model.result

    @result.setter
    def result(self, value: Optional[CalibrationResult]) -> None:
        self.view_model.result = value

    def __init__(self, main_window=None, app_context=None, view_model=None):
        super().__init__(main_window)
        self.setupUi(self)
        self.main_window = main_window
        self.app_context = (
            app_context
            or getattr(main_window, "app_context", None)
            or getattr(view_model, "app_context", None)
            or create_standalone_legacy_context()
        )
        if view_model is None:
            from ..bootstrap import create_calibration_view_model

            view_model = create_calibration_view_model(self.app_context)
        self.view_model = view_model
        self.image: Optional[DetectorImage] = None
        self.result: Optional[CalibrationResult] = None
        self._load_thread: Optional[QThread] = None
        self._load_worker: Optional[ImageLoaderWorker] = None
        self._cal_thread: Optional[QThread] = None
        self._cal_worker: Optional[CalibrationWorker] = None
        self._dragging_center = False
        self._close_when_idle = False
        self._reset_preview_view = True
        self._preview_cache: dict[tuple[int, bool], tuple] = {}
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(80)
        self._overlay_timer.timeout.connect(self.redraw_preview)
        self.detector_models = self.view_model.detector_models
        self.setWindowIcon(app_icon())
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint
        )
        self._bind_form()
        self._apply_dialog_style()
        self._connect_signals()
        self._set_running(False)

    def _apply_dialog_style(self) -> None:
        self.setStyleSheet("""
            QPushButton#primaryCalibrationButton {
                background: #2563eb; color: white; border: 1px solid #1d4ed8;
                border-radius: 6px; padding: 7px 14px; font-weight: 600;
            }
            QPushButton#primaryCalibrationButton:hover { background: #1d4ed8; }
            QPushButton#primaryCalibrationButton:disabled {
                background: #cbd5e1; color: #64748b; border-color: #cbd5e1;
            }
            QPushButton#previewActionButton, QPushButton#manualRefineButton {
                border: 1px solid #cbd5e1; border-radius: 6px;
                background: #f8fafc; padding: 6px 11px;
            }
            QPushButton#previewActionButton:hover, QPushButton#manualRefineButton:hover {
                background: #eef2ff; border-color: #94a3b8;
            }
            QPushButton#manualRefineButton:checked {
                background: #e0e7ff; border-color: #6366f1; color: #312e81;
            }
            QLabel#overlayLegend {
                background: #1f2937; color: white; border-radius: 6px;
                padding: 6px 10px;
            }
            QLabel#previewInfo { color: #475569; padding: 1px 2px; }
            QGroupBox#manualRefinementGroup {
                border: 1px solid #cbd5e1; border-radius: 8px;
                margin-top: 12px; padding-top: 10px; font-weight: 600;
            }
            QLabel#manualHint { color: #475569; font-weight: 400; }
        """)
        # Keep the key actions visually stable under the host application's
        # interchangeable light/dark themes, some of which install broad
        # QPushButton rules after child widgets have been constructed.
        self.calibrate_button.setStyleSheet("""
            QPushButton {
                background-color: #2563eb; color: white;
                border: 1px solid #1d4ed8; border-radius: 6px;
                padding: 7px 14px; font-weight: 600;
            }
            QPushButton:hover { background-color: #1d4ed8; }
            QPushButton:disabled {
                background-color: #cbd5e1; color: #64748b;
                border-color: #cbd5e1;
            }
        """)
        preview_style = """
            QPushButton {
                background-color: #f8fafc; color: #1f2937;
                border: 1px solid #cbd5e1; border-radius: 6px;
                padding: 6px 11px;
            }
            QPushButton:hover { background-color: #eef2ff; border-color: #94a3b8; }
            QPushButton:checked {
                background-color: #e0e7ff; color: #312e81;
                border-color: #6366f1;
            }
            QPushButton:disabled { color: #94a3b8; background-color: #f1f5f9; }
        """
        for button in (
            self.fit_image_button, self.clean_preview_button,
            self.expand_preview_button, self.manual_refine_button,
        ):
            button.setStyleSheet(preview_style)

    def _bind_form(self) -> None:
        """Attach behavior and dynamic content to the Designer-owned form."""
        bind_parameter_section(
            self.calibration_input_section,
            self.calibrationInputTitle,
            self.calibrationInputDescription,
            self.calibrationInputContent,
            self.calibrationInputContentLayout,
        )
        bind_parameter_section(
            self.calibration_run_section,
            self.calibrationRunTitle,
            self.calibrationRunDescription,
            self.calibrationRunContent,
            self.calibrationRunContentLayout,
        )
        bind_parameter_section(
            self.calibration_preview_panel,
            self.calibrationPreviewTitle,
            self.calibrationPreviewDescription,
            self.calibrationPreviewContent,
            self.calibrationPreviewContentLayout,
        )
        bind_parameter_section(
            self.calibration_results_section,
            self.calibrationResultsTitle,
            self.calibrationResultsDescription,
            self.calibrationResultsContent,
            self.calibrationResultsContentLayout,
        )
        bind_parameter_section(
            self.calibration_export_section,
            self.calibrationExportTitle,
            self.calibrationExportDescription,
            self.calibrationExportContent,
            self.calibrationExportContentLayout,
        )
        bind_advanced_section(
            self.calibration_advanced_section,
            self.calibrationAdvancedToggle,
            self.calibrationAdvancedDescription,
            self.calibrationAdvancedContent,
            self.calibrationAdvancedContentLayout,
        )
        bind_advanced_section(
            self.calibration_manual_section,
            self.calibrationManualToggle,
            self.calibrationManualDescription,
            self.calibrationManualContent,
            self.calibrationManualContentLayout,
        )
        for section in (
            self.calibration_input_section,
            self.calibration_advanced_section,
            self.calibration_run_section,
            self.calibration_preview_panel,
            self.calibration_results_section,
            self.calibration_manual_section,
            self.calibration_export_section,
        ):
            apply_design_system(section)

        self.standard_combo.addItem("Auto Detect", "auto")
        for standard in self.view_model.standard_options():
            self.standard_combo.addItem(standard.display_name, standard.key)
        self.detector_combo.addItem("Auto detected", None)
        for detector_name in self.detector_models:
            self.detector_combo.addItem(detector_name, detector_name)
        self.detector_combo.addItem("Custom pixel size", "custom")

        self.calibrate_button.setObjectName("primaryCalibrationButton")
        for button in (
            self.fit_image_button,
            self.clean_preview_button,
            self.expand_preview_button,
        ):
            button.setObjectName("previewActionButton")
        self.manual_refine_button.setObjectName("manualRefineButton")
        self.manual_group.setObjectName("manualRefinementGroup")
        self.manual_hint.setObjectName("manualHint")
        self.preview_info_label.setObjectName("previewInfo")
        self.overlay_legend.setObjectName("overlayLegend")

        self.job_status.set_actions_visible(
            pause=False,
            cancel=False,
            details=False,
        )
        self.progress = self.job_status.progress_bar
        self.stage_label = self.job_status.message_label

        self.figure = Figure(figsize=(7, 5), constrained_layout=False)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.96)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.calibrationToolbarHostLayout.addWidget(self.toolbar)
        self.calibrationFigureHostLayout.addWidget(self.canvas)

        self.overlay_legend.setText(
            f'<span style="color:{CENTER_COLOR}">━━</span> '
            '<span style="color:#f8fafc">Center</span> &nbsp;&nbsp; '
            f'<span style="color:{DETECTED_RING_COLOR}">┄┄┄</span> '
            '<span style="color:#f8fafc">Detected</span> &nbsp;&nbsp; '
            f'<span style="color:{MATCHED_RING_COLOR}">━━</span> '
            '<span style="color:#f8fafc">Matched</span> &nbsp;&nbsp; '
            f'<span style="color:{UNMATCHED_RING_COLOR}">╌╌╌</span> '
            '<span style="color:#f8fafc">Other theoretical</span>'
        )
        self.overlay_legend.setAttribute(Qt.WA_StyledBackground, True)
        self.overlay_legend.setStyleSheet(
            "background-color: #1f2937; color: #f8fafc; "
            "border-radius: 6px; padding: 6px 10px;"
        )
        self.overlay_legend.setVisible(False)

        self.result_labels = {
            "Beam center X": self.result_center_x,
            "Beam center Y": self.result_center_y,
            "Distance": self.result_distance,
            "Detector rotation": self.result_rotation,
            "Matched rings": self.result_rings,
            "RMS residual": self.result_rms,
            "Confidence": self.result_confidence,
            "Warning": self.result_warning,
        }
        self.candidate_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.results_splitter.setStretchFactor(0, 1)
        self.results_splitter.setStretchFactor(1, 2)
        self.right_splitter.setStretchFactor(0, 4)
        self.right_splitter.setStretchFactor(1, 2)
        self.right_splitter.setSizes([430, 150])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(self.open_image_dialog)
        self.path_edit.returnPressed.connect(self._load_path_edit)
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.cancel_button.clicked.connect(self.cancel_calibration)
        self.close_button.clicked.connect(self.close)
        self.apply_button.clicked.connect(self.apply_result)
        self.export_button.clicked.connect(self.export_result)
        self.import_button.clicked.connect(self.import_result)
        self.candidate_table.itemSelectionChanged.connect(self._candidate_selected)
        self.log_check.toggled.connect(self.redraw_preview)
        self.mask_check.toggled.connect(self.redraw_preview)
        self.rings_check.toggled.connect(self.redraw_preview)
        self.fit_image_button.clicked.connect(self.fit_preview_to_image)
        self.clean_preview_button.toggled.connect(self._clean_preview_toggled)
        self.expand_preview_button.clicked.connect(self._toggle_preview_expanded)
        self.manual_refine_button.toggled.connect(self.manual_group.setChecked)
        self.calibrationManualToggle.toggled.connect(
            self.manual_group.setChecked
        )
        self.manual_group.toggled.connect(self._manual_group_toggled)
        self.standard_combo.currentIndexChanged.connect(self._populate_theory_rings)
        self.detector_combo.currentIndexChanged.connect(self._detector_model_changed)
        for widget in (self.manual_x, self.manual_y, self.manual_distance):
            widget.valueChanged.connect(lambda _value: self._overlay_timer.start())
        self.refine_ring_button.clicked.connect(self.fit_selected_ring)
        self.canvas.mpl_connect("button_press_event", self._preview_press)
        self.canvas.mpl_connect("motion_notify_event", self._preview_move)
        self.canvas.mpl_connect("button_release_event", self._preview_release)

    def _set_running(self, running: bool) -> None:
        self.open_button.setEnabled(not running)
        for widget in (
            self.path_edit, self.energy_spin, self.standard_combo,
            self.estimated_distance_spin, self.range_combo, self.detector_combo,
            self.pixel_x_spin, self.pixel_y_spin,
            self.custom_min_spin, self.custom_max_spin, self.background_check,
        ):
            widget.setEnabled(not running)
        self.calibrate_button.setEnabled(not running and self.image is not None)
        self.cancel_button.setEnabled(running)
        self.apply_button.setEnabled(not running and self.result is not None)
        self.export_button.setEnabled(not running and self.result is not None)
        self.clean_preview_button.setEnabled(not running and self.result is not None)
        self.manual_refine_button.setEnabled(not running and self.result is not None)
        self.manual_group.setEnabled(not running and self.result is not None)

    def _manual_group_toggled(self, checked: bool) -> None:
        self.calibration_manual_section.set_expanded(checked)
        self.manual_panel.setVisible(checked)
        self.manual_group.setMaximumHeight(16777215 if checked else 40)
        blocker = QSignalBlocker(self.manual_refine_button)
        self.manual_refine_button.setChecked(checked)
        self.manual_refine_button.setText("Finish manual" if checked else "Manual refine")
        del blocker

    def fit_preview_to_image(self) -> None:
        self._reset_preview_view = True
        self.redraw_preview()

    def _clean_preview_toggled(self, checked: bool) -> None:
        self.clean_preview_button.setText("Show overlays" if checked else "Clean image")
        self.redraw_preview()

    def _toggle_preview_expanded(self) -> None:
        expanded = self.results_splitter.isVisible()
        self.results_splitter.setVisible(not expanded)
        self.expand_preview_button.setText("Show results" if expanded else "Focus image")
        self._reset_preview_view = True
        QTimer.singleShot(0, self.redraw_preview)

    def open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Calibration Image", "", "Detector Images (*.nxs *.cbf);;NXS Files (*.nxs);;CBF Files (*.cbf)"
        )
        if path:
            self.load_image(self.view_model.normalize_path(path))

    def _load_path_edit(self) -> None:
        path = self.path_edit.text().strip().strip('"')
        if path:
            self.load_image(self.view_model.normalize_path(path))

    def load_image(self, path: str, dataset_path: Optional[str] = None) -> None:
        if self._load_thread is not None and self._load_thread.isRunning():
            return
        self.path_edit.setText(path)
        self.job_status.set_state("running", "Reading image...", progress=None)
        self._set_running(True)
        self._load_thread = QThread(self)
        self._load_worker = ImageLoaderWorker(path, self.view_model, dataset_path)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._image_loaded)
        self._load_worker.failed.connect(lambda exc: self._image_failed(path, exc))
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._cleanup_loader)
        self._load_thread.start()

    def _image_loaded(self, image: DetectorImage) -> None:
        self.image = image
        self.result = None
        self._preview_cache.clear()
        self._reset_preview_view = True
        self.clean_preview_button.setChecked(False)
        self.manual_group.setChecked(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        if image.energy_kev:
            self.energy_spin.setValue(image.energy_kev)
        if image.pixel_size_x_m:
            self.pixel_x_spin.setValue(image.pixel_size_x_m * 1e6)
        if image.pixel_size_y_m:
            self.pixel_y_spin.setValue(image.pixel_size_y_m * 1e6)
        if image.distance_m:
            self.estimated_distance_spin.setValue(image.distance_m * 1000.0)
        else:
            current_distance = self.view_model.current_geometry(
                {
                    "distance": 0.0,
                    "beam_center_x": 0.0,
                    "beam_center_y": 0.0,
                }
            )["distance"]
            if current_distance and float(current_distance) > 0:
                self.estimated_distance_spin.setValue(float(current_distance))
        if image.pixel_size_x_m and image.pixel_size_y_m:
            self.pixel_label.setText(f"{image.pixel_size_x_m * 1e6:.3f} × {image.pixel_size_y_m * 1e6:.3f} µm (metadata)")
        else:
            self.pixel_label.setText("Not detected — enter in Advanced Settings")
            self.calibration_advanced_section.set_expanded(True)
        self.detector_label.setText(image.detector_name or "Not identified")
        detector_index = 0
        if image.detector_name:
            normalized = " ".join(image.detector_name.lower().split())
            for index in range(1, self.detector_combo.count()):
                model_name = self.detector_combo.itemData(index)
                if model_name and model_name != "custom" and model_name.lower() in normalized:
                    detector_index = index
                    break
        self.detector_combo.setCurrentIndex(detector_index)
        if not image.detector_name:
            self.detector_label.setText("Not identified — choose a detector model")
        detected_standards = self.view_model.detected_standard_keys(image.source_path)
        if len(detected_standards) == 1:
            standard_index = self.standard_combo.findData(detected_standards[0])
            if standard_index >= 0:
                self.standard_combo.setCurrentIndex(standard_index)
        elif len(detected_standards) > 1:
            auto_index = self.standard_combo.findData("auto")
            if auto_index >= 0:
                self.standard_combo.setCurrentIndex(auto_index)
        energy_note = ""
        if image.metadata.get("energy_source"):
            energy_note = " | energy from companion NXS"
        standard_note = ""
        if len(detected_standards) > 1:
            standard_note = " | multiple standard names found; comparing patterns automatically"
        self.stage_label.setText(
            f"Loaded {self.view_model.source_name(image.source_path)} — "
            f"{image.data.shape[1]} × {image.data.shape[0]} pixels"
            f"{energy_note}{standard_note}. "
            "Click Auto Calibration."
        )
        self.job_status.set_state(
            "succeeded",
            self.stage_label.text(),
            progress=1.0,
        )
        self.preview_info_label.setText(
            f"{self.view_model.source_name(image.source_path)}  ·  "
            f"{image.data.shape[1]} × {image.data.shape[0]} px"
        )
        self.candidate_table.setRowCount(0)
        self._clear_result_labels()
        self.redraw_preview()
        self._set_running(False)

    def _detector_model_changed(self) -> None:
        model_name = self.detector_combo.currentData()
        if not model_name or model_name == "custom":
            if model_name == "custom":
                self.calibration_advanced_section.set_expanded(True)
            return
        model = self.detector_models.get(model_name, {})
        pixel_x = model.get("pixel_size_x")
        pixel_y = model.get("pixel_size_y", pixel_x)
        if pixel_x:
            self.pixel_x_spin.setValue(float(pixel_x))
        if pixel_y:
            self.pixel_y_spin.setValue(float(pixel_y))
        self.pixel_label.setText(f"{float(pixel_x):.3f} × {float(pixel_y):.3f} µm ({model_name})")
        self.detector_label.setText(model_name)

    def _image_failed(self, path: str, exc: Exception) -> None:
        self.job_status.set_state("failed", "Failed to load image.", progress=0.0)
        if isinstance(exc, AmbiguousImageDatasetError):
            from PyQt5.QtWidgets import QInputDialog
            selected, ok = QInputDialog.getItem(self, "Select NXS Dataset", "Detector image dataset:", exc.paths, 0, False)
            if ok and selected:
                QTimer.singleShot(150, lambda: self.load_image(path, selected))
        else:
            QMessageBox.warning(self, "Calibration Image", str(exc))
        self._set_running(False)

    def _cleanup_loader(self) -> None:
        self._load_worker = None
        if self._load_thread is not None:
            self._load_thread.deleteLater()
        self._load_thread = None
        if self._close_when_idle and self._cal_thread is None:
            QTimer.singleShot(0, self.close)

    def _distance_range(self) -> tuple[float, float]:
        index = self.range_combo.currentIndex()
        if index == 1:
            return 500.0, 10000.0
        if index == 2:
            return 30.0, 1500.0
        if index == 3:
            low, high = self.custom_min_spin.value(), self.custom_max_spin.value()
            if low >= high:
                raise ValueError("Custom distance minimum must be smaller than the maximum.")
            return low, high
        return 30.0, 10000.0

    def start_calibration(self) -> None:
        if self.image is None or (self._cal_thread is not None and self._cal_thread.isRunning()):
            return
        try:
            options = {
                "energy_kev": self.energy_spin.value(),
                "standard_key": self.standard_combo.currentData(),
                "estimated_distance_mm": self.estimated_distance_spin.value() or None,
                "distance_range_mm": self._distance_range(),
                "pixel_size_x_m": self.pixel_x_spin.value() * 1e-6,
                "pixel_size_y_m": self.pixel_y_spin.value() * 1e-6,
                "subtract_background": self.background_check.isChecked(),
            }
        except ValueError as exc:
            QMessageBox.warning(self, "Calibration Input", str(exc))
            return
        self.job_status.set_state(
            "running",
            "Starting calibration...",
            progress=0.0,
        )
        self._set_running(True)
        self._cal_thread = QThread(self)
        self._cal_worker = CalibrationWorker(self.view_model, options)
        self._cal_worker.moveToThread(self._cal_thread)
        self._cal_thread.started.connect(self._cal_worker.run)
        self._cal_worker.progress.connect(self._calibration_progress)
        self._cal_worker.finished.connect(self._calibration_finished)
        self._cal_worker.failed.connect(self._calibration_failed)
        self._cal_worker.finished.connect(self._cal_thread.quit)
        self._cal_worker.failed.connect(self._cal_thread.quit)
        self._cal_thread.finished.connect(self._cleanup_calibration)
        self._cal_thread.start()

    def cancel_calibration(self) -> None:
        if self._cal_worker is not None:
            self._cal_worker.cancel()
            self.job_status.set_state(
                "running",
                "Cancelling after the current numerical step...",
                progress=self.progress.value() / max(1, self.progress.maximum()),
            )
            self.cancel_button.setEnabled(False)

    def _calibration_progress(self, value: int, stage: str) -> None:
        self.job_status.set_state("running", stage, progress=value / 100.0)

    def _calibration_finished(self, result: CalibrationResult) -> None:
        self.result = result
        self.job_status.set_state(
            "succeeded",
            "Calibration complete. Review the selected candidate, then Apply.",
            progress=1.0,
        )
        candidate_blocker = QSignalBlocker(self.candidate_table)
        self._populate_candidates()
        self.candidate_table.selectRow(0)
        del candidate_blocker
        self._show_candidate(result.selected_candidate)
        self._set_running(False)
        self.manual_group.setChecked(True)

    def _calibration_failed(self, exc: Exception) -> None:
        if isinstance(exc, CalibrationCancelledError):
            self.job_status.set_state(
                "cancelled", "Calibration cancelled.", progress=0.0
            )
        else:
            self.job_status.set_state(
                "failed",
                "Calibration failed. Adjust the inputs and try again.",
                progress=0.0,
            )
            QMessageBox.warning(self, "Geometry Calibration", str(exc))
        self._set_running(False)

    def _cleanup_calibration(self) -> None:
        self._cal_worker = None
        if self._cal_thread is not None:
            self._cal_thread.deleteLater()
        self._cal_thread = None
        if self._close_when_idle and self._load_thread is None:
            QTimer.singleShot(0, self.close)

    def _populate_candidates(self) -> None:
        if self.result is None:
            return
        self.candidate_table.setRowCount(len(self.result.candidates))
        for row, candidate in enumerate(self.result.candidates):
            values = (
                self.view_model.standard_display_name(candidate.standard_key),
                f"{candidate.distance_mm:.2f} mm",
                f"{candidate.center_x_px:.1f}, {candidate.center_y_px:.1f}",
                str(candidate.matched_ring_count),
                f"{candidate.rms_residual_px:.2f} px",
                candidate.confidence,
            )
            for column, value in enumerate(values):
                self.candidate_table.setItem(row, column, QTableWidgetItem(value))

    def _candidate_selected(self) -> None:
        if self.result is None:
            return
        rows = self.candidate_table.selectionModel().selectedRows()
        if not rows:
            return
        candidate = self.view_model.select_candidate(rows[0].row())
        self._show_candidate(candidate)

    def _show_candidate(self, candidate: CalibrationCandidate) -> None:
        self.result_labels["Beam center X"].setText(f"{candidate.center_x_px:.3f} px")
        self.result_labels["Beam center Y"].setText(f"{candidate.center_y_px:.3f} px")
        self.result_labels["Distance"].setText(f"{candidate.distance_mm:.3f} mm")
        self.result_labels["Detector rotation"].setText(f"{candidate.detector_rotation_deg:.3f}°")
        self.result_labels["Matched rings"].setText(str(candidate.matched_ring_count))
        self.result_labels["RMS residual"].setText(f"{candidate.rms_residual_px:.3f} px")
        self.result_labels["Confidence"].setText(candidate.confidence)
        self.result_labels["Warning"].setText(" ".join(candidate.warnings) or "None")
        blockers = [QSignalBlocker(widget) for widget in (
            self.manual_x, self.manual_y, self.manual_distance,
        )]
        self.manual_x.setValue(candidate.center_x_px)
        self.manual_y.setValue(candidate.center_y_px)
        self.manual_distance.setValue(candidate.distance_mm)
        del blockers
        standard_name = self.view_model.standard_display_name(candidate.standard_key)
        self.preview_info_label.setText(
            f"{self.view_model.source_name(self.image.source_path) if self.image else ''}  ·  "
            f"{standard_name}  ·  {candidate.distance_mm:.2f} mm  ·  "
            f"{candidate.matched_ring_count} matched rings  ·  {candidate.confidence} confidence"
        )
        self._populate_manual_rings(candidate)
        self.redraw_preview()

    def _clear_result_labels(self) -> None:
        for label in self.result_labels.values():
            label.setText("—")

    def _display_candidate(self) -> Optional[CalibrationCandidate]:
        return self.view_model.display_candidate(
            manual_enabled=self.manual_group.isChecked(),
            center_x_px=self.manual_x.value(),
            center_y_px=self.manual_y.value(),
            distance_mm=self.manual_distance.value(),
        )

    def _prepared_preview(self) -> tuple:
        """Return a cached, resolution-adaptive detector preview."""
        if self.image is None:
            raise ValueError("No calibration image is loaded.")
        log_scale = self.log_check.isChecked()
        key = (id(self.image.data), log_scale)
        cached = self._preview_cache.get(key)
        if cached is not None:
            return cached
        data = np.asarray(self.image.data, dtype=np.float32)
        height, width = data.shape
        max_preview_pixels = 1_400_000
        stride = max(1, int(np.ceil(np.sqrt(data.size / max_preview_pixels))))
        sampled = data[::stride, ::stride]
        invalid = ~np.isfinite(sampled)
        if self.image.mask is not None:
            invalid |= np.asarray(self.image.mask, dtype=bool)[::stride, ::stride]
        valid = ~invalid
        display = np.zeros(sampled.shape, dtype=np.float32)
        if log_scale:
            display[valid] = np.log1p(np.maximum(sampled[valid], 0.0))
        else:
            display[valid] = sampled[valid]
        values = display[valid]
        if values.size:
            percentile_sample = values[::max(1, values.size // 250_000)]
            vmin, vmax = np.percentile(percentile_sample, (1.0, 99.7))
        else:
            vmin, vmax = 0.0, 1.0
        result = (
            display,
            invalid,
            (-0.5, width - 0.5, height - 0.5, -0.5),
            float(vmin),
            float(max(vmax, vmin + 1e-6)),
            height,
            width,
        )
        self._preview_cache[key] = result
        return result

    @staticmethod
    def _ellipse_intersects_image(
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
        width: int,
        height: int,
    ) -> bool:
        if radius_x <= 0 or radius_y <= 0:
            return False
        nearest_x = float(np.clip(center_x, 0.0, width - 1.0))
        nearest_y = float(np.clip(center_y, 0.0, height - 1.0))
        minimum = np.hypot(
            (nearest_x - center_x) / radius_x,
            (nearest_y - center_y) / radius_y,
        )
        maximum = max(
            np.hypot((x - center_x) / radius_x, (y - center_y) / radius_y)
            for x, y in (
                (0.0, 0.0), (width - 1.0, 0.0),
                (0.0, height - 1.0), (width - 1.0, height - 1.0),
            )
        )
        return minimum <= 1.02 and maximum >= 0.98

    def redraw_preview(self) -> None:
        old_xlim, old_ylim = self.axes.get_xlim(), self.axes.get_ylim()
        had_image = bool(self.axes.images)
        self.axes.clear()
        if self.image is None:
            self.axes.text(0.5, 0.5, "Open a .nxs or .cbf calibration image", ha="center", va="center", transform=self.axes.transAxes)
            self.overlay_legend.setVisible(False)
            self.canvas.draw_idle()
            return
        display, invalid, extent, vmin, vmax, height, width = self._prepared_preview()
        self.axes.imshow(
            display, cmap="viridis", origin="upper", extent=extent,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        clean_preview = self.clean_preview_button.isChecked()
        if not clean_preview and self.mask_check.isChecked() and invalid.any():
            overlay = np.ma.masked_where(~invalid, invalid.astype(float))
            self.axes.imshow(
                overlay, cmap="Reds", alpha=0.30, origin="upper",
                extent=extent, vmin=0, vmax=1, interpolation="nearest",
            )
        candidate = self._display_candidate()
        if candidate is not None and not clean_preview:
            cx, cy = candidate.center_x_px, candidate.center_y_px
            self.axes.axvline(cx, color=CENTER_COLOR, linestyle="-.", linewidth=1.2, alpha=0.95)
            self.axes.axhline(cy, color=CENTER_COLOR, linestyle="-.", linewidth=1.2, alpha=0.95)
            if self.rings_check.isChecked() and self.result is not None:
                for radius in candidate.detected_peak_radii_px:
                    if not self._ellipse_intersects_image(cx, cy, radius, radius, width, height):
                        continue
                    self.axes.add_patch(Ellipse(
                        (cx, cy), 2 * radius, 2 * radius, fill=False,
                        edgecolor=DETECTED_RING_COLOR, linestyle=":",
                        linewidth=0.8, alpha=0.62,
                    ))
                for ring in self.view_model.theoretical_ring_overlays(candidate):
                    ellipse_width = ring.width_px
                    ellipse_height = ring.height_px
                    if not self._ellipse_intersects_image(
                        cx, cy, 0.5 * ellipse_width, 0.5 * ellipse_height,
                        self.image.data.shape[1], self.image.data.shape[0],
                    ):
                        continue
                    self.axes.add_patch(Ellipse(
                        (cx, cy), ellipse_width, ellipse_height, fill=False,
                        edgecolor=MATCHED_RING_COLOR if ring.matched else UNMATCHED_RING_COLOR,
                        linestyle="-" if ring.matched else "--",
                        linewidth=1.5 if ring.matched else 0.8,
                        alpha=0.95 if ring.matched else 0.65,
                    ))
        self.overlay_legend.setVisible(candidate is not None and not clean_preview)
        self.axes.set_xlabel("Detector X (pixel)")
        self.axes.set_ylabel("Detector Y (pixel)")
        self.axes.set_aspect("equal", adjustable="box")
        if had_image and not self._reset_preview_view:
            self.axes.set_xlim(old_xlim)
            self.axes.set_ylim(old_ylim)
        else:
            self.axes.set_xlim(-0.5, width - 0.5)
            self.axes.set_ylim(height - 0.5, -0.5)
            self._reset_preview_view = False
        self.canvas.draw_idle()

    def _populate_manual_rings(self, candidate: CalibrationCandidate) -> None:
        self.experimental_ring_combo.clear()
        for radius in candidate.detected_peak_radii_px:
            self.experimental_ring_combo.addItem(f"{radius:.2f} px", radius)
        self._populate_theory_rings()

    def _populate_theory_rings(self) -> None:
        self.theory_ring_combo.clear()
        key = self.result.selected_candidate.standard_key if self.result else self.standard_combo.currentData()
        for index, q in enumerate(self.view_model.standard_q_values(key)):
            self.theory_ring_combo.addItem(f"{index + 1}: q={q:.5f} Å⁻¹", q)

    def fit_selected_ring(self) -> None:
        if self.result is None or self.experimental_ring_combo.currentData() is None or self.theory_ring_combo.currentData() is None:
            return
        try:
            distance = self.view_model.manual_ring_distance(
                float(self.experimental_ring_combo.currentData()),
                float(self.theory_ring_combo.currentData()),
            )
            self.manual_distance.setValue(distance)
            self.stage_label.setText("Manual distance updated from the selected experimental/theoretical ring pair.")
        except ValueError as exc:
            QMessageBox.warning(self, "Manual Refinement", str(exc))

    def _preview_press(self, event) -> None:
        if self.manual_group.isChecked() and event.inaxes is self.axes and event.xdata is not None and event.ydata is not None:
            self._dragging_center = True
            self.manual_x.setValue(event.xdata)
            self.manual_y.setValue(event.ydata)

    def _preview_move(self, event) -> None:
        if self._dragging_center and event.inaxes is self.axes and event.xdata is not None and event.ydata is not None:
            self.manual_x.setValue(event.xdata)
            self.manual_y.setValue(event.ydata)

    def _preview_release(self, _event) -> None:
        self._dragging_center = False

    def _commit_manual_values(self) -> None:
        if self.result is not None:
            self.view_model.commit_manual_refinement(
                manual_enabled=self.manual_group.isChecked(),
                center_x_px=self.manual_x.value(),
                center_y_px=self.manual_y.value(),
                distance_mm=self.manual_distance.value(),
            )

    def _sync_main_window_geometry(self) -> None:
        """把已保存的 calibration 结果反映到现有 PyQt controls。"""
        if self.result is None or self.main_window is None:
            return
        candidate = self.result.selected_candidate
        page = getattr(getattr(self.main_window, "components", None), "waxs_page", None)
        if page is not None:
            controls = {
                "center_x_spin": candidate.center_x_px,
                "center_y_spin": candidate.center_y_px,
                "distance_spin": candidate.distance_mm,
                "pixel_x_spin": self.result.pixel_size_x_m * 1e6,
                "pixel_y_spin": self.result.pixel_size_y_m * 1e6,
                "wavelength_spin": self.result.wavelength_angstrom,
            }
            for name, value in controls.items():
                widget = getattr(page, name, None)
                if widget is not None:
                    widget.setValue(float(value))
            if hasattr(page, "refresh_view"):
                page.refresh_view()
        if hasattr(self.main_window, "statusbar"):
            self.main_window.statusbar.showMessage(
                "Geometry calibration applied: center "
                f"({candidate.center_x_px:.2f}, {candidate.center_y_px:.2f}), "
                f"distance {candidate.distance_mm:.2f} mm"
            )

    def apply_result(self) -> None:
        if self.result is None:
            return
        self._commit_manual_values()
        if self.view_model.result_differs_significantly():
            answer = QMessageBox.question(
                self, "Apply Geometry",
                "This calibration differs significantly from the current manually configured geometry. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.view_model.apply_result()
        self._sync_main_window_geometry()
        self.calibrationApplied.emit(self.result)
        QMessageBox.information(self, "Geometry Calibration", "The calibrated geometry was applied to SAXS, GISAXS, and GIWAXS state.")

    def export_result(self) -> None:
        if self.result is None:
            return
        self._commit_manual_values()
        default = self.view_model.default_export_path(self.result.source_image)
        path, _ = QFileDialog.getSaveFileName(self, "Export Calibration", default, "JSON Files (*.json)")
        if path:
            try:
                self.view_model.export_result(path)
                self.stage_label.setText(f"Calibration exported to {path}")
            except Exception as exc:
                LOGGER.exception("Failed to export calibration")
                QMessageBox.warning(self, "Export Calibration", str(exc))

    def import_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Calibration", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            previous_image = self.image
            self.result = self.view_model.import_result(path)
            self.path_edit.setText(self.result.source_image)
            if self.image is not previous_image:
                self._preview_cache.clear()
            self.energy_spin.setValue(self.result.energy_kev)
            self.pixel_x_spin.setValue(self.result.pixel_size_x_m * 1e6)
            self.pixel_y_spin.setValue(self.result.pixel_size_y_m * 1e6)
            candidate_blocker = QSignalBlocker(self.candidate_table)
            self._populate_candidates()
            self.candidate_table.selectRow(0)
            del candidate_blocker
            self._show_candidate(self.result.selected_candidate)
            self.stage_label.setText(
                f"Imported calibration from {self.view_model.source_name(path)}"
            )
            self._set_running(False)
            self.manual_group.setChecked(True)
        except Exception as exc:
            LOGGER.exception("Failed to import calibration")
            QMessageBox.warning(self, "Import Calibration", str(exc))

    def closeEvent(self, event) -> None:
        if self._cal_thread is not None and self._cal_thread.isRunning():
            answer = QMessageBox.question(self, "Calibration Running", "Cancel calibration and close?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.cancel_calibration()
            self._close_when_idle = True
            self.hide()
            event.ignore()
            return
        if self._load_thread is not None and self._load_thread.isRunning():
            self._close_when_idle = True
            self.hide()
            event.ignore()
            return
        event.accept()
