from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from PyQt5.QtCore import QObject, QSignalBlocker, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.application import apply_calibration_result
from calibration.engine import CalibrationCancelled, CalibrationEngine
from calibration.geometry_model import distance_from_ring_radius, q_to_ring_radius_m
from calibration.image_loader import AmbiguousDatasetError, load_detector_image
from calibration.models import CalibrationCandidate, CalibrationResult, DetectorImage
from calibration.serialization import load_calibration, save_calibration
from calibration.standards import STANDARDS, available_standards
from core.global_params import global_params
from ui.app_assets import app_icon
from utils.path_utils import normalize_path


LOGGER = logging.getLogger(__name__)

CENTER_COLOR = "#ff4d8d"
DETECTED_RING_COLOR = "#00d9ff"
MATCHED_RING_COLOR = "#ffd54a"
UNMATCHED_RING_COLOR = "#ff8a3d"


class ImageLoaderWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, path: str, dataset_path: Optional[str] = None):
        super().__init__()
        self.path = path
        self.dataset_path = dataset_path

    def run(self) -> None:
        try:
            self.finished.emit(load_detector_image(self.path, dataset_path=self.dataset_path))
        except Exception as exc:
            LOGGER.exception("Failed to load calibration image")
            self.failed.emit(exc)


class CalibrationWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, image: DetectorImage, options: dict):
        super().__init__()
        self.image = image
        self.options = options
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        try:
            engine = CalibrationEngine(
                progress=lambda value, stage: self.progress.emit(value, stage),
                cancelled=lambda: self.cancel_requested,
            )
            self.finished.emit(engine.calibrate(self.image, **self.options))
        except Exception as exc:
            if not isinstance(exc, CalibrationCancelled):
                LOGGER.exception("Geometry calibration failed")
            self.failed.emit(exc)


def _spin(minimum: float, maximum: float, value: float, decimals: int = 4) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setKeyboardTracking(False)
    return widget


class GeometryCalibrationDialog(QDialog):
    calibrationApplied = pyqtSignal(object)

    def __init__(self, main_window=None):
        super().__init__(main_window)
        self.main_window = main_window
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
        try:
            detector_path = Path(__file__).resolve().parents[1] / "config" / "detectors.json"
            self.detector_models = json.loads(detector_path.read_text(encoding="utf-8"))
        except Exception:
            self.detector_models = {}
        self.setWindowTitle("Geometry Calibration")
        self.setWindowIcon(app_icon())
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint
        )
        self.setMinimumSize(900, 560)
        self.resize(1180, 760)
        self.setModal(False)
        self._build_ui()
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

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        controls = QWidget()
        controls.setMinimumWidth(280)
        controls.setMaximumWidth(410)
        left = QVBoxLayout(controls)
        left.setContentsMargins(0, 0, 6, 0)

        file_group = QGroupBox("Calibration image")
        file_layout = QHBoxLayout(file_group)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Paste a .cbf/.nxs path or use Open...")
        self.open_button = QPushButton("Open...")
        file_layout.addWidget(self.path_edit, 1)
        file_layout.addWidget(self.open_button)
        left.addWidget(file_group)

        input_group = QGroupBox("Input")
        input_form = QFormLayout(input_group)
        self.energy_spin = _spin(0.1, 200.0, 12.0, 5)
        self.energy_spin.setSuffix(" keV")
        self.standard_combo = QComboBox()
        self.standard_combo.addItem("Auto Detect", "auto")
        for standard in available_standards():
            self.standard_combo.addItem(standard.display_name, standard.key)
        self.estimated_distance_spin = _spin(0.0, 100000.0, 0.0, 2)
        self.estimated_distance_spin.setSuffix(" mm")
        self.estimated_distance_spin.setSpecialValueText("Optional")
        self.range_combo = QComboBox()
        self.range_combo.addItems(["Auto (30-10000 mm)", "SAXS (500-10000 mm)", "WAXS (30-1500 mm)", "Custom"])
        self.pixel_label = QLabel("Open an image")
        self.pixel_label.setWordWrap(True)
        self.detector_label = QLabel("Open an image")
        self.detector_combo = QComboBox()
        self.detector_combo.addItem("Auto detected", None)
        for detector_name in self.detector_models:
            self.detector_combo.addItem(detector_name, detector_name)
        self.detector_combo.addItem("Custom pixel size", "custom")
        input_form.addRow("Energy:", self.energy_spin)
        input_form.addRow("Standard:", self.standard_combo)
        input_form.addRow("Estimated distance:", self.estimated_distance_spin)
        input_form.addRow("Distance range:", self.range_combo)
        input_form.addRow("Pixel size:", self.pixel_label)
        input_form.addRow("Detector:", self.detector_label)
        input_form.addRow("Detector model:", self.detector_combo)
        left.addWidget(input_group)

        self.advanced_group = QGroupBox("Advanced Settings")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_form = QFormLayout(self.advanced_group)
        self.pixel_x_spin = _spin(1.0, 1000.0, 172.0, 4)
        self.pixel_y_spin = _spin(1.0, 1000.0, 172.0, 4)
        self.custom_min_spin = _spin(1.0, 100000.0, 30.0, 1)
        self.custom_max_spin = _spin(1.0, 100000.0, 10000.0, 1)
        self.background_check = QCheckBox("Subtract slowly varying background")
        self.background_check.setChecked(True)
        self.mask_check = QCheckBox("Show invalid-pixel mask")
        self.log_check = QCheckBox("Log intensity")
        self.log_check.setChecked(True)
        self.rings_check = QCheckBox("Show ring overlays")
        self.rings_check.setChecked(True)
        advanced_form.addRow("Pixel X:", self.pixel_x_spin)
        self.pixel_x_spin.setSuffix(" µm")
        advanced_form.addRow("Pixel Y:", self.pixel_y_spin)
        self.pixel_y_spin.setSuffix(" µm")
        advanced_form.addRow("Custom minimum:", self.custom_min_spin)
        self.custom_min_spin.setSuffix(" mm")
        advanced_form.addRow("Custom maximum:", self.custom_max_spin)
        self.custom_max_spin.setSuffix(" mm")
        advanced_form.addRow(self.background_check)
        advanced_form.addRow(self.log_check)
        advanced_form.addRow(self.mask_check)
        advanced_form.addRow(self.rings_check)
        left.addWidget(self.advanced_group)

        run_row = QHBoxLayout()
        self.calibrate_button = QPushButton("Auto Calibration")
        self.calibrate_button.setObjectName("primaryCalibrationButton")
        self.calibrate_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        run_row.addWidget(self.calibrate_button, 1)
        run_row.addWidget(self.cancel_button)
        left.addLayout(run_row)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.stage_label = QLabel("Open a calibration image to begin.")
        self.stage_label.setWordWrap(True)
        left.addWidget(self.progress)
        left.addWidget(self.stage_label)
        left.addStretch(1)
        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("calibrationControlsScroll")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(280)
        controls_scroll.setMaximumWidth(430)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        self.right_splitter = QSplitter(Qt.Vertical)
        figure_group = QGroupBox("Image Preview")
        figure_layout = QVBoxLayout(figure_group)
        self.figure = Figure(figsize=(7, 5), constrained_layout=False)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.96)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        figure_layout.addWidget(self.toolbar)
        preview_actions = QHBoxLayout()
        self.fit_image_button = QPushButton("Reset view")
        self.clean_preview_button = QPushButton("Clean image")
        self.expand_preview_button = QPushButton("Focus image")
        self.manual_refine_button = QPushButton("Manual refine")
        for button in (
            self.fit_image_button, self.clean_preview_button,
            self.expand_preview_button, self.manual_refine_button,
        ):
            button.setObjectName("manualRefineButton" if button is self.manual_refine_button else "previewActionButton")
        self.clean_preview_button.setCheckable(True)
        self.clean_preview_button.setEnabled(False)
        self.manual_refine_button.setCheckable(True)
        self.manual_refine_button.setEnabled(False)
        preview_actions.addWidget(self.fit_image_button)
        preview_actions.addWidget(self.clean_preview_button)
        preview_actions.addWidget(self.expand_preview_button)
        preview_actions.addWidget(self.manual_refine_button)
        preview_actions.addStretch(1)
        figure_layout.addLayout(preview_actions)
        self.preview_info_label = QLabel("Open a calibration image to begin")
        self.preview_info_label.setObjectName("previewInfo")
        self.preview_info_label.setWordWrap(True)
        figure_layout.addWidget(self.preview_info_label)
        self.overlay_legend = QLabel(
            f'<span style="color:{CENTER_COLOR}">━━</span> '
            '<span style="color:#f8fafc">Center</span> &nbsp;&nbsp; '
            f'<span style="color:{DETECTED_RING_COLOR}">┄┄┄</span> '
            '<span style="color:#f8fafc">Detected</span> &nbsp;&nbsp; '
            f'<span style="color:{MATCHED_RING_COLOR}">━━</span> '
            '<span style="color:#f8fafc">Matched</span> &nbsp;&nbsp; '
            f'<span style="color:{UNMATCHED_RING_COLOR}">╌╌╌</span> '
            '<span style="color:#f8fafc">Other theoretical</span>'
        )
        self.overlay_legend.setObjectName("overlayLegend")
        self.overlay_legend.setTextFormat(Qt.RichText)
        # Force a painted background even when the host application theme
        # supplies broad QLabel rules after this dialog is constructed.
        self.overlay_legend.setAttribute(Qt.WA_StyledBackground, True)
        self.overlay_legend.setStyleSheet(
            "background-color: #1f2937; color: #f8fafc; "
            "border-radius: 6px; padding: 6px 10px;"
        )
        self.overlay_legend.setVisible(False)
        figure_layout.addWidget(self.overlay_legend)
        figure_layout.addWidget(self.canvas, 1)
        self.right_splitter.addWidget(figure_group)

        lower = QSplitter(Qt.Horizontal)
        result_group = QGroupBox("Results")
        result_form = QFormLayout(result_group)
        self.result_labels = {name: QLabel("—") for name in (
            "Beam center X", "Beam center Y", "Distance", "Detector rotation",
            "Matched rings", "RMS residual", "Confidence", "Warning",
        )}
        self.result_labels["Warning"].setWordWrap(True)
        for name, label in self.result_labels.items():
            result_form.addRow(name + ":", label)
        lower.addWidget(result_group)

        candidates_group = QGroupBox("Candidate solutions")
        candidates_layout = QVBoxLayout(candidates_group)
        self.candidate_table = QTableWidget(0, 6)
        self.candidate_table.setHorizontalHeaderLabels(["Standard", "Distance", "Center", "Rings", "RMS", "Confidence"])
        self.candidate_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.candidate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        candidates_layout.addWidget(self.candidate_table)
        lower.addWidget(candidates_group)
        lower.setStretchFactor(0, 1)
        lower.setStretchFactor(1, 2)
        self.results_splitter = lower
        self.right_splitter.addWidget(lower)
        self.right_splitter.setStretchFactor(0, 4)
        self.right_splitter.setStretchFactor(1, 2)
        self.right_splitter.setSizes([430, 150])
        right_layout.addWidget(self.right_splitter, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.manual_group = QGroupBox("Manual refinement · drag the center marker or edit values")
        self.manual_group.setObjectName("manualRefinementGroup")
        self.manual_group.setCheckable(True)
        self.manual_group.setChecked(False)
        self.manual_group.setEnabled(False)
        manual_group_layout = QVBoxLayout(self.manual_group)
        self.manual_panel = QWidget()
        manual = QGridLayout(self.manual_panel)
        manual.setContentsMargins(4, 2, 4, 4)
        self.manual_hint = QLabel(
            "Fine-tune only when the overlay needs correction. Changes are previewed immediately."
        )
        self.manual_hint.setObjectName("manualHint")
        self.manual_x = _spin(-100000.0, 100000.0, 0.0, 3)
        self.manual_y = _spin(-100000.0, 100000.0, 0.0, 3)
        self.manual_distance = _spin(0.01, 100000.0, 1000.0, 3)
        self.experimental_ring_combo = QComboBox()
        self.theory_ring_combo = QComboBox()
        self.refine_ring_button = QPushButton("Fit selected ring")
        manual.addWidget(self.manual_hint, 0, 0, 1, 6)
        manual.addWidget(QLabel("Center X:"), 1, 0)
        manual.addWidget(self.manual_x, 1, 1)
        manual.addWidget(QLabel("Center Y:"), 1, 2)
        manual.addWidget(self.manual_y, 1, 3)
        manual.addWidget(QLabel("Distance (mm):"), 1, 4)
        manual.addWidget(self.manual_distance, 1, 5)
        manual.addWidget(QLabel("Detected ring:"), 2, 0)
        manual.addWidget(self.experimental_ring_combo, 2, 1, 1, 2)
        manual.addWidget(QLabel("Theoretical peak:"), 2, 3)
        manual.addWidget(self.theory_ring_combo, 2, 4)
        manual.addWidget(self.refine_ring_button, 2, 5)
        manual_group_layout.addWidget(self.manual_panel)
        self.manual_panel.setVisible(False)
        self.manual_group.setMaximumHeight(40)
        root.addWidget(self.manual_group)

        buttons = QHBoxLayout()
        self.import_button = QPushButton("Import Calibration...")
        self.export_button = QPushButton("Export Calibration...")
        self.apply_button = QPushButton("Apply")
        self.close_button = QPushButton("Close")
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.export_button)
        buttons.addStretch(1)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.close_button)
        root.addLayout(buttons)

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
            self.load_image(normalize_path(path))

    def _load_path_edit(self) -> None:
        path = self.path_edit.text().strip().strip('"')
        if path:
            self.load_image(normalize_path(path))

    def load_image(self, path: str, dataset_path: Optional[str] = None) -> None:
        if self._load_thread is not None and self._load_thread.isRunning():
            return
        self.path_edit.setText(path)
        self.progress.setRange(0, 0)
        self.stage_label.setText("Reading image...")
        self._set_running(True)
        self._load_thread = QThread(self)
        self._load_worker = ImageLoaderWorker(path, dataset_path)
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
            current_distance = global_params.get_parameter("fitting", "detector.distance", 0.0)
            if current_distance and float(current_distance) > 0:
                self.estimated_distance_spin.setValue(float(current_distance))
        if image.pixel_size_x_m and image.pixel_size_y_m:
            self.pixel_label.setText(f"{image.pixel_size_x_m * 1e6:.3f} × {image.pixel_size_y_m * 1e6:.3f} µm (metadata)")
        else:
            self.pixel_label.setText("Not detected — enter in Advanced Settings")
            self.advanced_group.setChecked(True)
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
        source_name = str(image.source_path).lower()
        aliases = {
            "agbh": ("agbh", "ag_behenate", "silver_behenate"),
            "lab6": ("lab6", "lanthanum_hexaboride"),
            "ceo2": ("ceo2", "cerium_dioxide"),
        }
        detected_standards = [
            key for key, names in aliases.items() if any(name in source_name for name in names)
        ]
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
            f"Loaded {Path(image.source_path).name} — {image.data.shape[1]} × {image.data.shape[0]} pixels"
            f"{energy_note}{standard_note}. "
            "Click Auto Calibration."
        )
        self.preview_info_label.setText(
            f"{Path(image.source_path).name}  ·  {image.data.shape[1]} × {image.data.shape[0]} px"
        )
        self.candidate_table.setRowCount(0)
        self._clear_result_labels()
        self.redraw_preview()
        self._set_running(False)

    def _detector_model_changed(self) -> None:
        model_name = self.detector_combo.currentData()
        if not model_name or model_name == "custom":
            if model_name == "custom":
                self.advanced_group.setChecked(True)
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
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.stage_label.setText("Failed to load image.")
        if isinstance(exc, AmbiguousDatasetError):
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
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.stage_label.setText("Starting calibration...")
        self._set_running(True)
        self._cal_thread = QThread(self)
        self._cal_worker = CalibrationWorker(self.image, options)
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
            self.stage_label.setText("Cancelling after the current numerical step...")
            self.cancel_button.setEnabled(False)

    def _calibration_progress(self, value: int, stage: str) -> None:
        self.progress.setValue(value)
        self.stage_label.setText(stage)

    def _calibration_finished(self, result: CalibrationResult) -> None:
        self.result = result
        self.progress.setValue(100)
        self.stage_label.setText("Calibration complete. Review the selected candidate, then Apply.")
        candidate_blocker = QSignalBlocker(self.candidate_table)
        self._populate_candidates()
        self.candidate_table.selectRow(0)
        del candidate_blocker
        self._show_candidate(result.selected_candidate)
        self._set_running(False)
        self.manual_group.setChecked(True)

    def _calibration_failed(self, exc: Exception) -> None:
        self.progress.setValue(0)
        if isinstance(exc, CalibrationCancelled):
            self.stage_label.setText("Calibration cancelled.")
        else:
            self.stage_label.setText("Calibration failed. Adjust the inputs and try again.")
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
            standard = STANDARDS.get(candidate.standard_key)
            values = (
                standard.display_name if standard else candidate.standard_key,
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
        candidate = self.result.candidates[rows[0].row()]
        self.result.selected_candidate = candidate
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
        standard = STANDARDS.get(candidate.standard_key)
        standard_name = standard.display_name if standard else candidate.standard_key
        self.preview_info_label.setText(
            f"{Path(self.image.source_path).name if self.image else ''}  ·  "
            f"{standard_name}  ·  {candidate.distance_mm:.2f} mm  ·  "
            f"{candidate.matched_ring_count} matched rings  ·  {candidate.confidence} confidence"
        )
        self._populate_manual_rings(candidate)
        self.redraw_preview()

    def _clear_result_labels(self) -> None:
        for label in self.result_labels.values():
            label.setText("—")

    def _display_candidate(self) -> Optional[CalibrationCandidate]:
        if self.result is None:
            return None
        candidate = copy.deepcopy(self.result.selected_candidate)
        if self.manual_group.isChecked():
            candidate.center_x_px = self.manual_x.value()
            candidate.center_y_px = self.manual_y.value()
            candidate.distance_mm = self.manual_distance.value()
        return candidate

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
                standard = STANDARDS.get(candidate.standard_key)
                if standard:
                    radii_m = q_to_ring_radius_m(standard.q_values_inv_angstrom, self.result.wavelength_angstrom, candidate.distance_mm)
                    used = {match.theoretical_index for match in candidate.matched_rings}
                    for index, radius_m in enumerate(radii_m):
                        ellipse_width = 2.0 * radius_m / self.result.pixel_size_x_m
                        ellipse_height = 2.0 * radius_m / self.result.pixel_size_y_m
                        if not self._ellipse_intersects_image(
                            cx, cy, 0.5 * ellipse_width, 0.5 * ellipse_height,
                            self.image.data.shape[1], self.image.data.shape[0],
                        ):
                            continue
                        matched = index in used
                        self.axes.add_patch(Ellipse(
                            (cx, cy), ellipse_width, ellipse_height, fill=False,
                            edgecolor=MATCHED_RING_COLOR if matched else UNMATCHED_RING_COLOR,
                            linestyle="-" if matched else "--", linewidth=1.5 if matched else 0.8,
                            alpha=0.95 if matched else 0.65,
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
        if key == "auto":
            key = "agbh"
        standard = STANDARDS.get(key)
        if standard:
            for index, q in enumerate(standard.q_values_inv_angstrom):
                self.theory_ring_combo.addItem(f"{index + 1}: q={q:.5f} Å⁻¹", q)

    def fit_selected_ring(self) -> None:
        if self.result is None or self.experimental_ring_combo.currentData() is None or self.theory_ring_combo.currentData() is None:
            return
        try:
            distance = distance_from_ring_radius(
                float(self.experimental_ring_combo.currentData()), float(self.theory_ring_combo.currentData()),
                self.result.wavelength_angstrom,
                0.5 * (self.result.pixel_size_x_m + self.result.pixel_size_y_m),
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
        if self.result is not None and self.manual_group.isChecked():
            candidate = self.result.selected_candidate
            candidate.center_x_px = self.manual_x.value()
            candidate.center_y_px = self.manual_y.value()
            candidate.distance_mm = self.manual_distance.value()
            if "Geometry was manually adjusted after automatic calibration." not in candidate.warnings:
                candidate.warnings.append("Geometry was manually adjusted after automatic calibration.")

    def apply_result(self) -> None:
        if self.result is None:
            return
        self._commit_manual_values()
        current_distance = float(global_params.get_parameter("fitting", "detector.distance", self.result.selected_candidate.distance_mm))
        current_x = float(global_params.get_parameter("fitting", "detector.beam_center_x", self.result.selected_candidate.center_x_px))
        current_y = float(global_params.get_parameter("fitting", "detector.beam_center_y", self.result.selected_candidate.center_y_px))
        candidate = self.result.selected_candidate
        significant = (
            abs(current_distance - candidate.distance_mm) / max(abs(current_distance), 1.0) > 0.05
            or abs(current_x - candidate.center_x_px) > 10.0
            or abs(current_y - candidate.center_y_px) > 10.0
        )
        if significant:
            answer = QMessageBox.question(
                self, "Apply Geometry",
                "This calibration differs significantly from the current manually configured geometry. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        apply_calibration_result(self.result, self.main_window)
        global_params.save_user_parameters()
        self.calibrationApplied.emit(self.result)
        QMessageBox.information(self, "Geometry Calibration", "The calibrated geometry was applied to SAXS, GISAXS, and GIWAXS state.")

    def export_result(self) -> None:
        if self.result is None:
            return
        self._commit_manual_values()
        default = str(Path(self.result.source_image).with_suffix(".gimap-calibration.json"))
        path, _ = QFileDialog.getSaveFileName(self, "Export Calibration", default, "JSON Files (*.json)")
        if path:
            try:
                save_calibration(self.result, normalize_path(path))
                self.stage_label.setText(f"Calibration exported to {path}")
            except Exception as exc:
                LOGGER.exception("Failed to export calibration")
                QMessageBox.warning(self, "Export Calibration", str(exc))

    def import_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Calibration", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            self.result = load_calibration(normalize_path(path))
            self.path_edit.setText(self.result.source_image)
            if Path(self.result.source_image).exists():
                self.image = load_detector_image(self.result.source_image)
                self._preview_cache.clear()
            self.energy_spin.setValue(self.result.energy_kev)
            self.pixel_x_spin.setValue(self.result.pixel_size_x_m * 1e6)
            self.pixel_y_spin.setValue(self.result.pixel_size_y_m * 1e6)
            candidate_blocker = QSignalBlocker(self.candidate_table)
            self._populate_candidates()
            self.candidate_table.selectRow(0)
            del candidate_blocker
            self._show_candidate(self.result.selected_candidate)
            self.stage_label.setText(f"Imported calibration from {Path(path).name}")
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
