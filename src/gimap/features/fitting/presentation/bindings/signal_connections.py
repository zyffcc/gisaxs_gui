"""Signal Connections for fitting presentation."""

from __future__ import annotations


from PyQt5.QtCore import Qt


class SignalConnectionsMixin:
    """Own signal connections behavior."""

    def _setup_connections(self):
        """No description."""
        if hasattr(self.ui, "gisaxsInputImportButton"):
            self.ui.gisaxsInputImportButton.clicked.connect(self._import_gisaxs_file)

        if self._previous_image_button is not None:
            self._previous_image_button.clicked.connect(self._show_previous_folder_image)
        if self._next_image_button is not None:
            self._next_image_button.clicked.connect(self._show_next_folder_image)

        if hasattr(self.ui, "gisaxsInputImportButtonValue"):
            self.ui.gisaxsInputImportButtonValue.returnPressed.connect(
                self._on_import_value_changed
            )

        if hasattr(self.ui, "gisaxsInputStackValue"):
            self.ui.gisaxsInputStackValue.returnPressed.connect(self._on_stack_value_changed)

        if hasattr(self.ui, "gisaxsInputShowButton"):
            self.ui.gisaxsInputShowButton.clicked.connect(self._show_image)

        if hasattr(self.ui, "fittingPickCenterButton"):
            self.ui.fittingPickCenterButton.toggled.connect(
                self._toggle_main_center_tool
            )
        if hasattr(self.ui, "fittingSelectRegionButton"):
            self.ui.fittingSelectRegionButton.toggled.connect(
                self._toggle_main_region_tool
            )
        if hasattr(self.ui, "fittingResetDetectorViewButton"):
            self.ui.fittingResetDetectorViewButton.clicked.connect(
                self._reset_main_detector_view
            )
        if hasattr(self.ui, "fittingOpenDetectorWindowButton"):
            self.ui.fittingOpenDetectorWindowButton.clicked.connect(
                self._show_independent_window
            )

        if hasattr(self.ui, "gisaxsInputAutoShowCheckBox"):
            self.ui.gisaxsInputAutoShowCheckBox.toggled.connect(self._on_auto_show_changed)

        if hasattr(self.ui, "gisaxsInputModelCombox"):
            try:
                self.ui.gisaxsInputModelCombox.currentTextChanged.connect(
                    self._on_load_mode_changed
                )
            except Exception:
                pass

        if hasattr(self.ui, "gisaxsInputIntLogCheckBox"):
            self.ui.gisaxsInputIntLogCheckBox.toggled.connect(self._on_log_changed)

        if hasattr(self.ui, "gisaxsInputAutoScaleCheckBox"):
            self.ui.gisaxsInputAutoScaleCheckBox.toggled.connect(self._on_auto_scale_changed)

        if hasattr(self.ui, "gisaxsInputDisplayModeQ"):
            self.ui.gisaxsInputDisplayModeQ.toggled.connect(self._on_q_mode_changed)
        if hasattr(self.ui, "gisaxsInputDisplayModePixel"):
            self.ui.gisaxsInputDisplayModePixel.toggled.connect(self._on_q_mode_changed)

        if hasattr(self.ui, "gisaxsInputCenterAutoFindingButton"):
            self.ui.gisaxsInputCenterAutoFindingButton.clicked.connect(self._auto_find_center)

        if hasattr(self.ui, "gisaxsInputCutButton"):
            self.ui.gisaxsInputCutButton.clicked.connect(lambda _checked=False: self._perform_cut())

        detector_panel = getattr(self.ui, "fittingDetectorSetupPanel", None)
        if detector_panel is not None:
            detector_panel.settings_applied.connect(self._on_detector_parameters_changed)
        elif hasattr(self.ui, "gisaxsInputDetectorParaButton"):
            self.ui.gisaxsInputDetectorParaButton.clicked.connect(self._show_detector_parameters)

        if hasattr(self.ui, "gisaxsInputGraphicsView"):
            preview_view = self.ui.gisaxsInputGraphicsView
            preview_view.setToolTip(
                "Drop a CBF, NXS, or TIFF file here to load it. Double-click to open a larger window."
            )
            preview_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            preview_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            preview_view.setAlignment(Qt.AlignCenter)
            preview_view.mouseDoubleClickEvent = self._on_graphics_view_double_click
            preview_view.setAcceptDrops(True)
            preview_view.installEventFilter(self)
            preview_view.viewport().setAcceptDrops(True)
            preview_view.viewport().installEventFilter(self)

        if hasattr(self.ui, "fitGraphicsView"):
            self.ui.fitGraphicsView.setToolTip(
                "Double-click to open a larger independent fit window."
            )
            self.ui.fitGraphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.ui.fitGraphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.ui.fitGraphicsView.setAlignment(Qt.AlignCenter)
            self.ui.fitGraphicsView.mouseDoubleClickEvent = self._on_fit_graphics_view_double_click
            self.ui.fitGraphicsView.installEventFilter(self)
        if hasattr(self.ui, "fittingOpenResultWindowButton"):
            self.ui.fittingOpenResultWindowButton.clicked.connect(
                lambda _checked=False: self._on_fit_graphics_view_double_click(None)
            )

        if hasattr(self.ui, "fitStartButton"):
            self.ui.fitStartButton.clicked.connect(self._start_fitting)

        if hasattr(self.ui, "FittingClearFittingButton_2"):
            self.ui.FittingClearFittingButton_2.clicked.connect(self._clear_fitting_data)

        if hasattr(self.ui, "fitLogXCheckBox"):
            self.ui.fitLogXCheckBox.toggled.connect(self._on_fit_log_changed)
        if hasattr(self.ui, "fitLogYCheckBox"):
            self.ui.fitLogYCheckBox.toggled.connect(self._on_fit_log_changed)
        if hasattr(self.ui, "fitQViewModeComboBox"):
            self.ui.fitQViewModeComboBox.currentIndexChanged.connect(
                self._on_q_preparation_changed
            )
        if hasattr(self.ui, "fitCurveViewModeComboBox"):
            self.ui.fitCurveViewModeComboBox.currentIndexChanged.connect(
                self._on_curve_view_mode_changed
            )

        for _name in ["fitBGShowCheckBox", "fitResShowCheckBox"]:
            if hasattr(self.ui, _name):
                try:
                    getattr(self.ui, _name).toggled.connect(self._on_component_checkbox_changed)
                except Exception:
                    pass

        if hasattr(self.ui, "OthersNormalizeCheckBox"):
            self.ui.OthersNormalizeCheckBox.toggled.connect(self._on_normalize_changed)
        if hasattr(self.ui, "fitNormCheckBox"):
            self.ui.fitNormCheckBox.toggled.connect(self._on_normalize_changed)

        if hasattr(self.ui, "PositiveOnlyCheckBox"):
            self.ui.PositiveOnlyCheckBox.toggled.connect(self._on_positive_only_changed)
        if hasattr(self.ui, "fitRegionPositiveOnlyCheckBox"):
            self.ui.fitRegionPositiveOnlyCheckBox.toggled.connect(self._on_positive_only_changed)
        if hasattr(self.ui, "fitRegionNegativeOnlyCheckBox"):
            self.ui.fitRegionNegativeOnlyCheckBox.toggled.connect(self._on_positive_only_changed)

        if hasattr(self.ui, "fitResetButton"):
            self.ui.fitResetButton.clicked.connect(self._reset_fitting)

        if hasattr(self.ui, "fitImport1dFileButton"):
            self.ui.fitImport1dFileButton.clicked.connect(self._import_1d_file)

        if hasattr(self.ui, "fitImport1dFileValue"):
            self.ui.fitImport1dFileValue.returnPressed.connect(self._on_1d_file_value_changed)

        if hasattr(self.ui, "FittingExportButton"):
            self.ui.FittingExportButton.clicked.connect(self._export_fitting_data)

        if hasattr(self.ui, "FittingManualFittingButton"):
            self.ui.FittingManualFittingButton.clicked.connect(
                lambda _checked=False: self._perform_manual_fitting(reveal_result=True)
            )

        if hasattr(self.ui, "FittingAutoRefineButton"):
            self.ui.FittingAutoRefineButton.clicked.connect(self._show_manual_auto_refine_dialog)

        if hasattr(self.ui, "FittingAutoFittingButton"):
            self.ui.FittingAutoFittingButton.clicked.connect(self.open_ai_fitting_workspace)
        if hasattr(self.ui, "aiFittingRefreshButton"):
            self.ui.aiFittingRefreshButton.clicked.connect(self._refresh_ai_fitting_models)
        if hasattr(self.ui, "aiFittingOpenWorkspaceButton"):
            self.ui.aiFittingOpenWorkspaceButton.clicked.connect(self.open_ai_fitting_workspace)
        if hasattr(self.ui, "aiFittingExportOutputButton"):
            self.ui.aiFittingExportOutputButton.clicked.connect(self._export_ai_prediction_output)
        if hasattr(self.ui, "aiFittingModelComboBox"):
            self.ui.aiFittingModelComboBox.currentIndexChanged.connect(self._on_ai_model_selected)
        if hasattr(self.ui, "aiFittingConstraintComboBox"):
            self.ui.aiFittingConstraintComboBox.currentTextChanged.connect(
                self._on_ai_constraint_mode_changed
            )
        if hasattr(self.ui, "aiFittingFixedKComboBox"):
            self.ui.aiFittingFixedKComboBox.currentTextChanged.connect(
                lambda text: self._on_ai_fixed_k_changed(text)
            )
        if hasattr(self.ui, "aiFittingCombinationButton"):
            self.ui.aiFittingCombinationButton.clicked.connect(
                self._show_ai_fixed_combination_dialog
            )
        if hasattr(self.ui, "aiFittingFastPredictButton"):
            self.ui.aiFittingFastPredictButton.clicked.connect(
                lambda: self._start_ai_prediction("fast")
            )
        if hasattr(self.ui, "aiFittingFullAutoFitButton"):
            self.ui.aiFittingFullAutoFitButton.clicked.connect(
                lambda: self._start_ai_prediction("full")
            )
        if hasattr(self.ui, "aiFittingStopButton"):
            self.ui.aiFittingStopButton.clicked.connect(self._stop_ai_fitting_process)
        if hasattr(self.ui, "aiFittingAdvancedConstraintsButton"):
            self.ui.aiFittingAdvancedConstraintsButton.clicked.connect(
                self._show_advanced_constraints_dialog
            )
        self._connect_ai_fitting_settings_widgets()

        if hasattr(self.ui, "FittingAutoKButton"):
            self.ui.FittingAutoKButton.clicked.connect(self._on_auto_k_button_clicked)

        if hasattr(self.ui, "fitCurrentDataCheckBox"):
            self.ui.fitCurrentDataCheckBox.toggled.connect(self._on_current_data_checkbox_changed)

        self._connect_cutline_parameter_signals(
            mode=self._default_signal_mode,
            overrides=self._signal_mode_overrides,
        )

        self._connect_parameter_widgets()

        self._setup_fitting_text_browser()
        self._setup_fitting_parameters_context_menu()
        self._refresh_ai_fitting_models()
        self._restore_main_ai_settings()

    def _connect_cutline_parameter_signals(self, mode: str = "changed", overrides: dict = None):
        """Register Cut Line, center, and color-scale widgets with global parameter persistence.

        Args:
            mode: Signal mode. Use ``changed`` for live value updates or ``finished`` for commit-only updates.
            overrides: Optional per-widget signal mode overrides.
        """
        mapping = [
            ("gisaxsInputCenterVerticalValue", "center_vertical"),
            ("gisaxsInputCenterParallelValue", "center_parallel"),
            ("gisaxsInputCutLineVerticalValue", "cutline_vertical"),
            ("gisaxsInputCutLineParallelValue", "cutline_parallel"),
            ("gisaxsInputVminValue", "vmin"),
            ("gisaxsInputVmaxValue", "vmax"),
        ]
        overrides = overrides or {}
        for widget_name, param_key in mapping:
            if not hasattr(self.ui, widget_name):
                continue
            w = getattr(self.ui, widget_name)

            # 函数说明：实现 after commit 相关逻辑。
            def _after_commit(info, value, p=param_key):
                try:
                    if p in ("vmin", "vmax"):
                        self._on_color_scale_value_committed()
                        self._add_fitting_message(f"Meta commit GISAXS {p} = {value}", "INFO")
                        self._refresh_insitu_workflow_step_styles()
                        return
                    self._on_parameter_display_changed()
                    self._add_fitting_message(f"Meta commit GISAXS {p} = {value}", "INFO")
                    self._refresh_insitu_workflow_step_styles()
                except Exception:
                    pass

            widget_mode = overrides.get(widget_name, mode)
            meta = {
                "persist": "settings",
                "key_path": ("fitting", f"gisaxs_input.{param_key}"),
                "trigger_fit": False,
                "debounce_ms": self._param_debounce_ms,
                "epsilon_abs": self._param_abs_eps,
                "epsilon_rel": self._param_rel_eps,
                "after_commit": _after_commit,
                "connect_mode": widget_mode,
            }
            self.param_trigger_manager.register_parameter_widget(
                widget=w,
                widget_id=f"meta_gisaxs_{param_key}",
                category="gisaxs_input",
                immediate_handler=lambda v: None,
                delayed_handler=None,
                connect_signals=True,
                meta=meta,
            )

    def _restore_gisaxs_input_parameters(self):
        """Restore persisted GISAXS input and Cut Line values after widget setup."""
        mapping = [
            ("gisaxsInputCenterVerticalValue", "center_vertical", 0.0),
            ("gisaxsInputCenterParallelValue", "center_parallel", 0.0),
            ("gisaxsInputCutLineVerticalValue", "cutline_vertical", 10.0),
            ("gisaxsInputCutLineParallelValue", "cutline_parallel", 10.0),
            ("gisaxsInputVminValue", "vmin", None),
            ("gisaxsInputVmaxValue", "vmax", None),
        ]
        for widget_name, param_key, default_value in mapping:
            if not hasattr(self.ui, widget_name):
                continue
            try:
                value = self.fitting_view_model.get_setting(
                    "fitting", f"gisaxs_input.{param_key}", default_value
                )
                if value is None:
                    continue
                widget = getattr(self.ui, widget_name)
                old_block_state = widget.blockSignals(True)
                try:
                    widget.setValue(float(value))
                finally:
                    widget.blockSignals(old_block_state)
            except Exception:
                continue
