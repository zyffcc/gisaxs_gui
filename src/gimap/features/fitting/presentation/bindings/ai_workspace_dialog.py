"""Ai Workspace Dialog for fitting presentation."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextBrowser,
    QDialog,
    QComboBox,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QGroupBox,
)


from ..binding_primitives import (
    _ai_catalog,
)


class AiWorkspaceDialogMixin:
    """Own ai workspace dialog behavior."""

    def open_ai_fitting_workspace(self) -> None:
        if getattr(self, "_ai_fitting_dialog", None) is not None:
            self._refresh_ai_fitting_models()
            self._ai_fitting_dialog.show()
            self._ai_fitting_dialog.raise_()
            self._ai_fitting_dialog.activateWindow()
            return

        dialog = QDialog(self.main_window or self.ui)
        dialog.setWindowTitle("AI Auto Fitting Workspace")
        dialog.resize(980, 760)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        input_group = QGroupBox("A. Input / Data", dialog)
        input_layout = QHBoxLayout(input_group)
        arrays = self._current_ai_curve_arrays()
        point_count = len(arrays[0]) if arrays is not None else 0
        self._ai_input_summary_label = QLabel(
            f"Current fitting curve: {point_count} valid points | ROI and preprocessing are applied before inference.",
            input_group,
        )
        self._ai_input_summary_label.setWordWrap(True)
        input_layout.addWidget(self._ai_input_summary_label, 1)
        inspect_input_btn = QPushButton("Inspect Input...", input_group)
        inspect_input_btn.clicked.connect(self._show_ai_input_data_dialog)
        input_layout.addWidget(inspect_input_btn)
        layout.addWidget(input_group)

        model_group = QGroupBox("B. Model", dialog)
        model_group_layout = QVBoxLayout(model_group)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("AI Model:", dialog))
        self._ai_model_combo = QComboBox(dialog)
        self._ai_model_combo.setMinimumWidth(360)
        model_row.addWidget(self._ai_model_combo, 1)
        refresh_btn = QPushButton("Refresh", dialog)
        browse_btn = QPushButton("Browse...", dialog)
        model_row.addWidget(refresh_btn)
        model_row.addWidget(browse_btn)
        model_group_layout.addLayout(model_row)
        self._ai_model_status_label = QLabel("Checkpoint status: not inspected", model_group)
        self._ai_model_status_label.setWordWrap(True)
        model_group_layout.addWidget(self._ai_model_status_label)
        layout.addWidget(model_group)

        strategy_group = QGroupBox("C. Fitting Strategy", dialog)
        strategy_layout = QVBoxLayout(strategy_group)
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile:", strategy_group))
        self._ai_profile_combo = QComboBox(strategy_group)
        self._ai_profile_combo.addItems(list(_ai_catalog(self).profile_names()))
        selected_profile = str(
            self._ai_run_settings().get("profile", _ai_catalog(self).default_profile_name)
        )
        self._ai_profile_combo.setCurrentText(
            selected_profile
            if _ai_catalog(self).has_profile(selected_profile)
            else _ai_catalog(self).default_profile_name
        )
        profile_row.addWidget(self._ai_profile_combo)
        profile_row.addWidget(QLabel("State:", strategy_group))
        self._ai_profile_state_label = QLabel(
            "Custom"
            if self._ai_run_settings().get("profile_overrides")
            else self._ai_profile_combo.currentText(),
            strategy_group,
        )
        profile_row.addWidget(self._ai_profile_state_label)
        profile_row.addWidget(QLabel("Random seed:", strategy_group))
        self._ai_random_seed_spin = QSpinBox(strategy_group)
        self._ai_random_seed_spin.setRange(0, 2_147_483_647)
        self._ai_random_seed_spin.setValue(int(self._ai_run_settings().get("random_seed", 123)))
        profile_row.addWidget(self._ai_random_seed_spin)
        profile_row.addWidget(QLabel("Time budget (s, 0=none):", strategy_group))
        self._ai_time_budget_spin = QDoubleSpinBox(strategy_group)
        self._ai_time_budget_spin.setRange(0.0, 86400.0)
        self._ai_time_budget_spin.setDecimals(1)
        budget = self._ai_run_settings().get("time_budget_seconds")
        self._ai_time_budget_spin.setValue(float(budget or 0.0))
        profile_row.addWidget(self._ai_time_budget_spin)
        profile_row.addStretch(1)
        strategy_layout.addLayout(profile_row)

        constraint_row = QHBoxLayout()
        constraint_row.addWidget(QLabel("Constraint Mode:", dialog))
        self._ai_constraint_combo = QComboBox(dialog)
        self._ai_constraint_combo.addItems(
            ["Free", "Fixed K", "Fixed Combination", "Current Manual Model"]
        )
        constraint_row.addWidget(self._ai_constraint_combo)
        self._ai_constraint_k_combo = QComboBox(dialog)
        self._ai_constraint_k_combo.addItems(["1", "2", "3", "4"])
        constraint_row.addWidget(QLabel("K:", dialog))
        constraint_row.addWidget(self._ai_constraint_k_combo)
        self._ai_constraint_combination_button = QPushButton("Choose Combination...", dialog)
        self._ai_constraint_combination_button.setVisible(False)
        constraint_row.addWidget(self._ai_constraint_combination_button)
        constraint_row.addStretch(1)
        strategy_layout.addLayout(constraint_row)

        self._ai_status_label = QLabel("Status: Ready", dialog)
        self._ai_progress = QProgressBar(dialog)
        self._ai_progress.setRange(0, 100)
        self._ai_progress.setValue(0)
        layout.addWidget(self._ai_status_label)
        layout.addWidget(self._ai_progress)

        settings_grid = QGridLayout()
        settings_grid.addWidget(QLabel("Full samples:", dialog), 0, 0)
        self._ai_full_samples_spin = QSpinBox(dialog)
        self._ai_full_samples_spin.setRange(1, 1_000_000)
        settings_grid.addWidget(self._ai_full_samples_spin, 0, 1)
        settings_grid.addWidget(QLabel("Refine top:", dialog), 0, 2)
        self._ai_refine_top_n_spin = QSpinBox(dialog)
        self._ai_refine_top_n_spin.setRange(0, 100)
        settings_grid.addWidget(self._ai_refine_top_n_spin, 0, 3)
        settings_grid.addWidget(QLabel("Max eval:", dialog), 1, 0)
        self._ai_refine_max_nfev_spin = QSpinBox(dialog)
        self._ai_refine_max_nfev_spin.setRange(1, 100000)
        settings_grid.addWidget(self._ai_refine_max_nfev_spin, 1, 1)
        settings_grid.addWidget(QLabel("Progress every:", dialog), 1, 2)
        self._ai_progress_every_spin = QSpinBox(dialog)
        self._ai_progress_every_spin.setRange(0, 10000)
        settings_grid.addWidget(self._ai_progress_every_spin, 1, 3)
        settings_grid.addWidget(QLabel("Sample std:", dialog), 2, 0)
        self._ai_sampling_std_spin = QDoubleSpinBox(dialog)
        self._ai_sampling_std_spin.setDecimals(5)
        self._ai_sampling_std_spin.setRange(0.00001, 10.0)
        self._ai_sampling_std_spin.setSingleStep(0.001)
        settings_grid.addWidget(self._ai_sampling_std_spin, 2, 1)
        settings_grid.addWidget(QLabel("Target logRMSE:", dialog), 2, 2)
        self._ai_target_logrmse_spin = QDoubleSpinBox(dialog)
        self._ai_target_logrmse_spin.setDecimals(8)
        self._ai_target_logrmse_spin.setRange(0.0, 10.0)
        self._ai_target_logrmse_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_target_logrmse_spin, 2, 3)
        settings_grid.addWidget(QLabel("ftol:", dialog), 3, 0)
        self._ai_refine_ftol_spin = QDoubleSpinBox(dialog)
        self._ai_refine_ftol_spin.setDecimals(10)
        self._ai_refine_ftol_spin.setRange(0.0, 1.0)
        self._ai_refine_ftol_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_refine_ftol_spin, 3, 1)
        settings_grid.addWidget(QLabel("xtol:", dialog), 3, 2)
        self._ai_refine_xtol_spin = QDoubleSpinBox(dialog)
        self._ai_refine_xtol_spin.setDecimals(10)
        self._ai_refine_xtol_spin.setRange(0.0, 1.0)
        self._ai_refine_xtol_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_refine_xtol_spin, 3, 3)
        settings_grid.addWidget(QLabel("gtol:", dialog), 4, 0)
        self._ai_refine_gtol_spin = QDoubleSpinBox(dialog)
        self._ai_refine_gtol_spin.setDecimals(10)
        self._ai_refine_gtol_spin.setRange(0.0, 1.0)
        self._ai_refine_gtol_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_refine_gtol_spin, 4, 1)
        settings_grid.addWidget(QLabel("Stall patience:", dialog), 4, 2)
        self._ai_stall_patience_spin = QSpinBox(dialog)
        self._ai_stall_patience_spin.setRange(0, 100000)
        self._ai_stall_patience_spin.setToolTip("0 disables stall early stop.")
        settings_grid.addWidget(self._ai_stall_patience_spin, 4, 3)
        settings_grid.addWidget(QLabel("Stall tol:", dialog), 5, 0)
        self._ai_stall_tol_spin = QDoubleSpinBox(dialog)
        self._ai_stall_tol_spin.setDecimals(10)
        self._ai_stall_tol_spin.setRange(0.0, 1.0)
        self._ai_stall_tol_spin.setSingleStep(0.00000001)
        settings_grid.addWidget(self._ai_stall_tol_spin, 5, 1)
        self._ai_advanced_settings_toggle = QPushButton("Advanced settings...", strategy_group)
        self._ai_advanced_settings_toggle.setCheckable(True)
        self._ai_advanced_settings_container = QWidget(strategy_group)
        self._ai_advanced_settings_container.setLayout(settings_grid)
        self._ai_advanced_settings_container.setVisible(False)
        self._ai_advanced_settings_toggle.toggled.connect(
            self._ai_advanced_settings_container.setVisible
        )
        strategy_layout.addWidget(self._ai_advanced_settings_toggle)
        strategy_layout.addWidget(self._ai_advanced_settings_container)
        layout.addWidget(strategy_group)

        physical_group = QGroupBox("D. Physical Constraints", dialog)
        physical_layout = QHBoxLayout(physical_group)
        self._ai_constraint_summary_label = QLabel(
            "Geometry-aware defaults: positivity, size-distribution bounds, and optional-D hard-core spacing.",
            physical_group,
        )
        self._ai_constraint_summary_label.setWordWrap(True)
        physical_layout.addWidget(self._ai_constraint_summary_label, 1)
        physical_btn = QPushButton("Configure...", physical_group)
        physical_btn.clicked.connect(self._show_advanced_constraints_dialog)
        physical_layout.addWidget(physical_btn)
        layout.addWidget(physical_group)

        action_row = QHBoxLayout()
        self._ai_action_buttons = []
        for text in (
            "Run AI Auto Fit",
            "Show Input Data",
            "Show Results",
            "Refine Selected Candidate",
            "Reset",
        ):
            btn = QPushButton(text, dialog)
            btn.setMinimumHeight(28)
            if text == "Run AI Auto Fit":
                btn.clicked.connect(lambda _checked=False: self._start_ai_prediction("profile"))
            elif text == "Show Input Data":
                btn.clicked.connect(lambda _checked=False: self._show_ai_input_data_dialog())
            elif text == "Show Results":
                btn.clicked.connect(lambda _checked=False: self._show_ai_candidate_table())
            elif text == "Refine Selected Candidate":
                btn.clicked.connect(lambda _checked=False: self._show_ai_candidate_table())
            else:
                btn.clicked.connect(lambda _checked=False: self._reset_ai_workspace_defaults())
            action_row.addWidget(btn)
            self._ai_action_buttons.append(btn)
        self._ai_stop_button = QPushButton("Stop", dialog)
        self._ai_stop_button.setEnabled(False)
        self._ai_stop_button.clicked.connect(self._stop_ai_fitting_process)
        action_row.addWidget(self._ai_stop_button)
        layout.addLayout(action_row)

        self._ai_log_browser = QTextBrowser(dialog)
        self._ai_log_browser.setMinimumHeight(180)
        self._ai_log_browser.setPlaceholderText("AI fitting log")
        if getattr(self, "_ai_log_lines", None):
            self._ai_log_browser.setPlainText("\n".join(self._ai_log_lines))
        layout.addWidget(self._ai_log_browser, 1)

        close_row = QHBoxLayout()
        self._ai_open_output_button = QPushButton("Open Output Folder", dialog)
        self._ai_open_output_button.setEnabled(bool(getattr(self, "_ai_output_dir", None)))
        self._ai_open_output_button.clicked.connect(self._open_ai_output_folder)
        close_row.addWidget(self._ai_open_output_button)
        self._ai_export_output_button = QPushButton("Export Output...", dialog)
        self._ai_export_output_button.setEnabled(bool(getattr(self, "_ai_output_dir", None)))
        self._ai_export_output_button.clicked.connect(self._export_ai_prediction_output)
        close_row.addWidget(self._ai_export_output_button)
        close_row.addStretch(1)
        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        refresh_btn.clicked.connect(self._refresh_ai_fitting_models)
        browse_btn.clicked.connect(self._browse_ai_fitting_model)
        self._ai_model_combo.currentIndexChanged.connect(self._on_ai_model_selected)
        self._ai_constraint_combo.currentTextChanged.connect(self._on_ai_constraint_mode_changed)
        self._ai_constraint_k_combo.currentTextChanged.connect(
            lambda text: self._on_ai_fixed_k_changed(text)
        )
        self._ai_constraint_combination_button.clicked.connect(
            self._show_ai_fixed_combination_dialog
        )
        self._ai_profile_combo.currentTextChanged.connect(self._set_ai_profile)
        self._ai_random_seed_spin.valueChanged.connect(
            lambda value: (
                self._save_ai_fitting_settings(random_seed=int(value)),
                self._mark_ai_profile_custom(random_seed=int(value)),
            )
        )
        self._ai_time_budget_spin.valueChanged.connect(
            lambda value: (
                self._save_ai_fitting_settings(
                    time_budget_seconds=(float(value) if value > 0 else None)
                ),
                self._mark_ai_profile_custom(
                    time_budget_seconds=(float(value) if value > 0 else None)
                ),
            )
        )
        self._sync_workspace_ai_run_widgets()
        workspace_setting_map = {
            self._ai_full_samples_spin: "candidate_count",
            self._ai_refine_top_n_spin: "refinement_count",
            self._ai_refine_max_nfev_spin: "max_evaluations",
            self._ai_progress_every_spin: "progress_interval",
            self._ai_refine_ftol_spin: "tolerance",
            self._ai_refine_xtol_spin: "tolerance",
            self._ai_refine_gtol_spin: "tolerance",
            self._ai_stall_patience_spin: "stall_patience",
            self._ai_stall_tol_spin: "stall_tolerance",
            self._ai_sampling_std_spin: "sampling_std",
            self._ai_target_logrmse_spin: "target_log_rmse",
        }
        for widget, key in workspace_setting_map.items():
            widget.valueChanged.connect(
                lambda value, setting_key=key: self._mark_ai_profile_custom(**{setting_key: value})
            )
        dialog.finished.connect(lambda _result: setattr(self, "_ai_fitting_dialog", None))
        self._ai_fitting_dialog = dialog
        self._refresh_ai_fitting_models()
        self._restore_ai_workspace_settings()
        dialog.show()
