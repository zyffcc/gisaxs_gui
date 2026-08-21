"""Form Setup coordination for WAXS."""

from __future__ import annotations


import numpy as np


from PyQt5.QtWidgets import (
    QFrame,
    QTabWidget,
    QWidget,
)


from src.gimap.app.presentation import apply_design_system

from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)


from ..views import (
    WaxsAdvancedPanelView,
    WaxsBatchPanelView,
    WaxsConfigurePanelView,
    WaxsIntegrationPanelView,
    WaxsPreviewPanelView,
    WaxsRoiPanelView,
    WaxsToolbarView,
)

from ..image_viewer import ScatteringImageViewer
from ..theme import waxs_stylesheet
from ..workflow_layout import install_waxs_workflow


class FormSetupMixin:
    """Own form setup presentation behavior."""

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
        install_waxs_workflow(self)
        self.setStyleSheet(self.styleSheet() + "\n" + waxs_stylesheet())

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
            signal = (
                getattr(widget, "valueChanged", None)
                or getattr(widget, "toggled", None)
                or getattr(widget, "currentTextChanged", None)
            )
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
