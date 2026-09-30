"""Session state of the Fitting workspace: the curve, display controls and the series."""

from __future__ import annotations

import copy
import os

from src.gimap.shared.file_paths import normalize_path


class FittingSessionStateMixin:
    """Save and restore what the user was fitting."""

    def _set_default_parameters(self):
        self.current_parameters = {"fitting_params": {}}

    def get_parameters(self):
        return self.current_parameters.copy()

    def set_parameters(self, parameters):
        self.current_parameters.update(parameters)
        self.parameters_changed.emit(self.current_parameters)

    def get_imported_file(self):
        """The curve file being fitted (``""`` when none)."""
        return str(getattr(self, "current_1d_file_path", "") or "")

    def get_session_data(self):
        """Return the lightweight fitting session data used by the app runtime."""
        session_data = {}
        one_d_file = getattr(self, "current_1d_file_path", None)
        if one_d_file:
            one_d_file = normalize_path(one_d_file)
            session_data["last_1d_file"] = one_d_file
            session_data["last_1d_directory"] = os.path.dirname(one_d_file)
        session_data["display_mode"] = getattr(self, "display_mode", "normal")
        session_data["fit_log_x"] = self._is_fit_log_x_enabled()
        session_data["fit_x_scale"] = self._get_x_axis_scale()
        session_data["fit_q_branch"] = self._get_q_branch()
        session_data["fit_q_combination"] = self._get_q_combination_mode()
        session_data["fit_q_view_mode"] = self._get_q_view_mode()
        session_data["fit_log_y"] = self._get_checkbox_state("fitLogYCheckBox", True)
        session_data["fit_norm"] = self._get_checkbox_state("fitNormCheckBox", False)
        session_data["ai_fitting"] = copy.deepcopy(self._ai_run_settings())
        session_data["insitu_workflow"] = self.fitting_view_model.insitu.snapshot_insitu_workflow()
        session_data["insitu_recipe"] = self.fitting_view_model.insitu.snapshot_recipe()
        return session_data

    def restore_session(self, session_data):
        """Restore display controls, the in-situ series and the last curve path."""
        if not isinstance(session_data, dict):
            return
        self._restore_ai_session_settings(session_data.get("ai_fitting"))
        insitu_snapshot = session_data.get("insitu_workflow")
        if isinstance(insitu_snapshot, dict):
            try:
                self.fitting_view_model.insitu.restore_insitu_workflow(insitu_snapshot)
            except (KeyError, TypeError, ValueError):
                pass
        recipe_snapshot = session_data.get("insitu_recipe")
        if isinstance(recipe_snapshot, dict):
            try:
                self.fitting_view_model.insitu.restore_recipe(recipe_snapshot)
            except (KeyError, TypeError, ValueError):
                pass

        self._restore_fit_checkboxes(session_data)

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


__all__ = ["FittingSessionStateMixin"]
