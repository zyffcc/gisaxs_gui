"""Form Setup behavior for Calibration."""

from __future__ import annotations

import logging


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


from PyQt5.QtCore import QSignalBlocker, QTimer, Qt

from PyQt5.QtWidgets import (
    QHeaderView,
)


from src.gimap.app.presentation import apply_design_system

from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)


from ..preview_style import (
    CENTER_COLOR,
    DETECTED_RING_COLOR,
    MATCHED_RING_COLOR,
    UNMATCHED_RING_COLOR,
)

LOGGER = logging.getLogger(__name__)


class FormSetupMixin:
    """Own form setup presentation behavior."""

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
            self.fit_image_button,
            self.clean_preview_button,
            self.expand_preview_button,
            self.manual_refine_button,
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
            "background-color: #1f2937; color: #f8fafc; border-radius: 6px; padding: 6px 10px;"
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
        self.candidate_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
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
        self.calibrationManualToggle.toggled.connect(self.manual_group.setChecked)
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
            self.path_edit,
            self.energy_spin,
            self.standard_combo,
            self.estimated_distance_spin,
            self.range_combo,
            self.detector_combo,
            self.pixel_x_spin,
            self.pixel_y_spin,
            self.custom_min_spin,
            self.custom_max_spin,
            self.background_check,
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
