"""Signal wiring of the curve-first Fitting workspace."""

from __future__ import annotations

from PyQt5.QtCore import Qt

from src.gimap.app.presentation.theme import theme_manager


class SignalConnectionsMixin:
    """Connect the curve, model and fitting controls to their handlers."""

    def _setup_connections(self):
        if hasattr(self.ui, "fitGraphicsView"):
            view = self.ui.fitGraphicsView
            view.setToolTip("Double-click to open a larger independent fit window.")
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setAlignment(Qt.AlignCenter)
            view.mouseDoubleClickEvent = self._on_fit_graphics_view_double_click
            view.installEventFilter(self)
        if hasattr(self.ui, "fittingOpenResultWindowButton"):
            self.ui.fittingOpenResultWindowButton.clicked.connect(
                lambda _checked=False: self._on_fit_graphics_view_double_click(None)
            )

        if hasattr(self.ui, "fitResetButton"):
            self.ui.fitResetButton.clicked.connect(self._reset_fitting)
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

        for name in ("fitBGShowCheckBox", "fitResShowCheckBox"):
            if hasattr(self.ui, name):
                getattr(self.ui, name).toggled.connect(self._on_component_checkbox_changed)

        if hasattr(self.ui, "OthersNormalizeCheckBox"):
            self.ui.OthersNormalizeCheckBox.toggled.connect(self._on_normalize_changed)
        if hasattr(self.ui, "fitNormCheckBox"):
            self.ui.fitNormCheckBox.toggled.connect(self._on_normalize_changed)

        for name in ("PositiveOnlyCheckBox", "fitRegionPositiveOnlyCheckBox", "fitRegionNegativeOnlyCheckBox"):
            if hasattr(self.ui, name):
                getattr(self.ui, name).toggled.connect(self._on_positive_only_changed)

        if hasattr(self.ui, "fitImport1dFileButton"):
            self.ui.fitImport1dFileButton.clicked.connect(self._import_1d_file)
        if hasattr(self.ui, "fitImport1dFileValue"):
            self.ui.fitImport1dFileValue.returnPressed.connect(self._on_1d_file_value_changed)

        if hasattr(self.ui, "FittingExportButton"):
            self.ui.FittingExportButton.clicked.connect(self._export_fitting_data)
        if hasattr(self.ui, "fitExportPlotButton"):
            self.ui.fitExportPlotButton.clicked.connect(self._export_plot)
        theme_manager().changed.connect(self._restyle_curve_plots)
        if hasattr(self.ui, "FittingManualFittingButton"):
            self.ui.FittingManualFittingButton.clicked.connect(
                lambda _checked=False: self._perform_manual_fitting(reveal_result=True)
            )
        if hasattr(self.ui, "FittingAutoRefineButton"):
            self.ui.FittingAutoRefineButton.clicked.connect(
                lambda _checked=False: self._show_manual_auto_refine_dialog("local")
            )
        if hasattr(self.ui, "FittingGlobalSearchButton"):
            self.ui.FittingGlobalSearchButton.clicked.connect(
                lambda _checked=False: self._show_manual_auto_refine_dialog("global")
            )

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
        if hasattr(self.ui, "aiFittingExperimentalButton"):
            self.ui.aiFittingExperimentalButton.clicked.connect(
                lambda: self._start_ai_prediction("experimental")
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

        self._setup_fitting_text_browser()
        self._setup_fitting_parameters_context_menu()
        self._refresh_ai_fitting_models()
        self._restore_main_ai_settings()


__all__ = ["SignalConnectionsMixin"]
