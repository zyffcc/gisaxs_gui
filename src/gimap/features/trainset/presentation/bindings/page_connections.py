"""Page Connections coordination for Trainset."""

from __future__ import annotations


from pathlib import Path


from PyQt5.QtWidgets import (
    QCheckBox,
)


class PageConnectionsMixin:
    """Own page connections presentation behavior."""

    def _connect_page(self) -> None:
        page = self.page
        page.reference_button.clicked.connect(self._select_reference)
        page.pick_beam_center_button.clicked.connect(self._begin_beam_center)
        page.draw_roi_button.clicked.connect(lambda: self._begin_roi("roi"))
        page.draw_rectangle_button.clicked.connect(lambda: self._begin_mask("rectangle"))
        page.draw_circle_button.clicked.connect(lambda: self._begin_mask("ellipse"))
        page.remove_mask_button.clicked.connect(self._remove_selected_masks)
        page.clear_masks_button.clicked.connect(self._clear_masks)
        page.random_mask_preview_button.clicked.connect(self._new_random_mask_example)
        page.mask_region_created.connect(self._region_created)
        page.generate_preview_button.clicked.connect(self._generate_preview)
        page.force_simulation_button.clicked.connect(self._force_generate_preview)
        page.new_realization_button.clicked.connect(self._new_preview_realization)
        page.what_if_requested.connect(self._start_what_if)
        page.preview_button.clicked.connect(lambda: page.step_list.setCurrentRow(1))
        page.validate_button.clicked.connect(self._validate_and_report)
        page.load_button.clicked.connect(self._load_project_dialog)
        page.save_button.clicked.connect(self._save_project_dialog)
        page.prepare_button.clicked.connect(self._prepare_local_job)
        page.submit_button.clicked.connect(self._submit_maxwell)
        page.model_validate_button.clicked.connect(self._validate_model_contract)
        page.local_folder_button.clicked.connect(self._choose_workspace)
        page.local_dataset_folder_button.clicked.connect(self._choose_dataset_folder)
        page.local_results_folder_button.clicked.connect(self._choose_results_folder)
        page.local_cache_folder_button.clicked.connect(self._choose_cache_folder)
        page.local_python_button.clicked.connect(self._choose_local_python)
        page.local_prepare_button.clicked.connect(self._prepare_local_job)
        page.local_generate_test_button.clicked.connect(self._run_local_physical_test)
        page.local_generate_button.clicked.connect(self._run_local_generation)
        page.local_train_button.clicked.connect(self._run_local_training)
        page.local_smoke_button.clicked.connect(self._run_local_smoke_test)
        page.local_pause_button.clicked.connect(self._toggle_local_pause)
        page.local_stop_button.clicked.connect(self._stop_local_process)
        page.connection_button.clicked.connect(self._test_connection)
        page.hpc_prepare_button.clicked.connect(self._prepare_hpc_job)
        page.hpc_submit_button.clicked.connect(self._submit_maxwell)
        page.refresh_job_button.clicked.connect(self._refresh_job)
        page.sync_results_button.clicked.connect(self._sync_results)
        page.register_model_button.clicked.connect(self._register_best_model)
        page.storage_accept_check.toggled.connect(self._storage_acceptance_changed)
        page.auto_remember_check.toggled.connect(self._auto_remember_toggled)
        page.reset_defaults_button.clicked.connect(self.reset_to_defaults)
        page.configuration_edited.connect(self._schedule_autosave_from_page)
        page.project_name.textChanged.connect(self._schedule_autosave_from_page)
        page.reference_path.editingFinished.connect(self._load_reference_from_field)
        page.fields["detector.preset"].currentTextChanged.connect(self._apply_detector_preset)
        page.particle_combo.currentTextChanged.connect(self._particle_plugin_changed)
        page.interference_combo.currentTextChanged.connect(self._interference_plugin_changed)
        for path, widget in page.fields.items():
            if path.startswith("roi."):
                signal = getattr(widget, "valueChanged", None) or getattr(
                    widget, "currentTextChanged", None
                )
                if signal is not None:
                    signal.connect(self._roi_config_changed)
            elif path.startswith("detector.") or path.startswith("beam."):
                signal = getattr(widget, "valueChanged", None) or getattr(
                    widget, "currentTextChanged", None
                )
                if signal is not None:
                    signal.connect(self._geometry_changed)
            if path.startswith("mask."):
                signal = (
                    getattr(widget, "valueChanged", None)
                    or getattr(widget, "currentTextChanged", None)
                    or getattr(widget, "toggled", None)
                )
                if signal is not None:
                    signal.connect(self._mask_config_changed)
            edit_signal = (
                getattr(widget, "valueChanged", None)
                or getattr(widget, "currentTextChanged", None)
                or getattr(widget, "toggled", None)
                or getattr(widget, "textChanged", None)
            )
            if edit_signal is not None:
                edit_signal.connect(self._schedule_autosave_from_page)
            if path in {
                "pre.background.enabled",
                "pre.gaussian.enabled",
                "pre.poisson.enabled",
            } and isinstance(widget, QCheckBox):
                widget.toggled.connect(
                    lambda _checked: self._refresh_impact_options(self._collect_config())
                )
        page.mask_shape_table.itemChanged.connect(self._mask_config_changed)
        for table in (
            page.mask_shape_table,
            page.particle_parameter_table,
            page.interference_parameter_table,
            page.layer_table,
            page.model_layer_table,
        ):
            table.itemChanged.connect(self._schedule_autosave_from_page)

    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        remembered = self.trainset_view_model.load_settings()
        if not isinstance(remembered, dict) or int(remembered.get("schema_version", 0)) < 2:
            remembered = self.trainset_view_model.load_settings(reload=True)
        if isinstance(remembered, dict) and int(remembered.get("schema_version", 0)) >= 2:
            self.config = self.trainset_view_model.merge_config_with_defaults(remembered)
        self._apply_config_to_page(self.config)
        self._update_capabilities()
        self._update_geometry_label()
        reference = str(self.config.get("project", {}).get("reference_file", "")).strip()
        if reference and Path(reference).exists():
            self._load_reference(reference)
        elif reference:
            self.page.design_info.setText(
                "Remembered reference is unavailable. Choose a new file to restore "
                "the ROI and threshold-mask preview."
            )
        if self.page.auto_remember_check.isChecked():
            self.status_updated.emit(
                "Trainset workspace ready · remembered settings restored automatically"
            )
        else:
            self.status_updated.emit("Trainset workspace ready · automatic memory is off")
