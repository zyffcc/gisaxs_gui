"""Fitting Session State for fitting presentation."""

from __future__ import annotations

import os

import copy


from src.gimap.shared.file_paths import normalize_path


class FittingSessionStateMixin:
    """Own fitting session state behavior."""

    def _set_default_parameters(self):
        """No description."""
        self.current_parameters = {
            "imported_gisaxs_file": "",
            "stack_count": 1,
            "cut_region": {"x": 0, "y": 0, "width": 100, "height": 100},
            "fitting_params": {},
        }

    def get_parameters(self):
        """No description."""
        return self.current_parameters.copy()

    def set_parameters(self, parameters):
        """No description."""
        self.current_parameters.update(parameters)
        self.parameters_changed.emit(self.current_parameters)

    def get_imported_file(self):
        """ISAXS"""
        return self.current_parameters.get("imported_gisaxs_file", "")

    def get_session_data(self):
        """Return the lightweight fitting session data used by the app runtime."""
        session_data = {}

        gisaxs_file = self.current_parameters.get("imported_gisaxs_file", "")
        if gisaxs_file:
            gisaxs_file = normalize_path(gisaxs_file)
            session_data["last_opened_file"] = gisaxs_file
            session_data["imported_gisaxs_file"] = gisaxs_file
            session_data["last_directory"] = os.path.dirname(gisaxs_file)

        one_d_file = getattr(self, "current_1d_file_path", None)
        if one_d_file:
            one_d_file = normalize_path(one_d_file)
            session_data["last_1d_file"] = one_d_file
            session_data["last_1d_directory"] = os.path.dirname(one_d_file)

        session_data["load_mode"] = getattr(self, "load_mode", "Single")
        session_data["display_mode"] = getattr(self, "display_mode", "normal")
        session_data["stack_value"] = self._get_stack_value_text()
        session_data["stack_count"] = self.current_parameters.get("stack_count", 1)
        session_data["nxs_frame_index"] = self.current_parameters.get("nxs_frame_index", 0)
        session_data["insitu_range"] = self.current_parameters.get("insitu_range", "")
        session_data["fit_current_data"] = self._get_checkbox_state("fitCurrentDataCheckBox", False)
        session_data["fit_log_x"] = self._get_checkbox_state("fitLogXCheckBox", False)
        session_data["fit_log_y"] = self._get_checkbox_state("fitLogYCheckBox", False)
        session_data["fit_norm"] = self._get_checkbox_state("fitNormCheckBox", False)
        session_data["auto_show"] = self._is_auto_show_enabled()
        session_data["load_mode"] = getattr(self, "load_mode", "Single")
        session_data["ai_fitting"] = copy.deepcopy(self._ai_run_settings())
        session_data["insitu_workflow"] = self.fitting_view_model.insitu.snapshot_insitu_workflow()
        return session_data

    def restore_session(self, session_data):
        """Restore the last opened fitting session with the current UI pathways."""
        if not isinstance(session_data, dict):
            return

        self._restore_ai_session_settings(session_data.get("ai_fitting"))
        insitu_snapshot = session_data.get("insitu_workflow")
        if isinstance(insitu_snapshot, dict):
            try:
                self.fitting_view_model.insitu.restore_insitu_workflow(insitu_snapshot)
            except (KeyError, TypeError, ValueError):
                pass

        last_file = session_data.get("last_opened_file") or session_data.get("imported_gisaxs_file")
        if last_file:
            last_file = normalize_path(last_file)

        if hasattr(self.ui, "gisaxsInputAutoShowCheckBox"):
            try:
                self.ui.gisaxsInputAutoShowCheckBox.blockSignals(True)
                self.ui.gisaxsInputAutoShowCheckBox.setChecked(
                    bool(session_data.get("auto_show", self._is_auto_show_enabled()))
                )
                self.ui.gisaxsInputAutoShowCheckBox.blockSignals(False)
            except Exception:
                pass

        load_mode = str(session_data.get("load_mode", "")).strip()
        if load_mode:
            for combo_name in ("gisaxsInputModelCombox", "gisaxsInputModeValue"):
                if not hasattr(self.ui, combo_name):
                    continue
                try:
                    combo = getattr(self.ui, combo_name)
                    index = combo.findText(load_mode)
                    if index >= 0:
                        combo.setCurrentIndex(index)
                    break
                except Exception:
                    pass

        stack_value = str(
            session_data.get("stack_value", "") or session_data.get("insitu_range", "")
        ).strip()
        if stack_value:
            self._set_stack_value_text(stack_value)

        try:
            self._sync_ui_to_parameters()
        except Exception:
            pass

        self._restore_fit_checkboxes(session_data)

        if last_file and os.path.exists(last_file):
            self.current_parameters["imported_gisaxs_file"] = last_file
            self._set_nxs_frame_state(last_file, session_data.get("nxs_frame_index", 0))
            if hasattr(self.ui, "gisaxsInputImportButtonValue"):
                self.ui.gisaxsInputImportButtonValue.setText(os.path.basename(last_file))

            try:
                self._scan_folder_images_for_file(last_file)
            except Exception:
                pass

            try:
                self._validate_imported_file(last_file)
            except Exception:
                pass

            try:
                self._update_stack_display()
            except Exception:
                pass

            try:
                self._refresh_vmin_vmax_display()
            except Exception:
                pass

            try:
                if (
                    hasattr(self.ui, "gisaxsInputAutoShowCheckBox")
                    and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
                ):
                    self._show_image()
            except Exception:
                pass

            self.parameters_changed.emit(self.current_parameters)
            self.status_updated.emit(f"Session restored: {os.path.basename(last_file)}")

        if session_data.get("display_mode") == "normal":
            try:
                self._switch_to_normal_display_mode()
            except Exception:
                pass

        one_d_file = session_data.get("last_1d_file")
        if one_d_file:
            try:
                one_d_file = normalize_path(one_d_file)
                self.current_1d_file_path = one_d_file
                if hasattr(self.ui, "fitImport1dFileValue"):
                    self.ui.fitImport1dFileValue.setText(one_d_file)
            except Exception:
                pass
