"""Global Parameter Controls for fitting presentation."""

from __future__ import annotations


import numpy as np

from PyQt5.QtWidgets import (
    QPushButton,
)


from ..binding_primitives import (
    _scientific_commands,
)


class GlobalParameterControlsMixin:
    """Own global parameter controls behavior."""

    def get_global_parameter(self, param: str) -> float:
        """No description."""
        return self.model_params_manager.get_global_parameter("fitting", param, 0.0)

    def set_global_parameter(self, param: str, value: float) -> bool:
        """No description."""
        success = self.model_params_manager.set_global_parameter("fitting", param, value)
        if success:
            if param == "background" and hasattr(self.ui, "fitBGValue"):
                self.ui.fitBGValue.blockSignals(True)
                self.ui.fitBGValue.setValue(value)
                self.ui.fitBGValue.blockSignals(False)
            elif param == "sigma_res" and hasattr(self.ui, "fitSigmaResValue"):
                self.ui.fitSigmaResValue.blockSignals(True)
                self.ui.fitSigmaResValue.setValue(value)
                self.ui.fitSigmaResValue.blockSignals(False)
            elif param == "k_value" and hasattr(self.ui, "fitKValue"):
                self.ui.fitKValue.blockSignals(True)
                self.ui.fitKValue.setValue(value)
                self.ui.fitKValue.blockSignals(False)

            self.model_params_manager.save_parameters()
        return success

    def get_all_global_parameters(self) -> dict:
        """No description."""
        return self.model_params_manager.get_all_global_parameters("fitting")

    def reset_global_parameters(self):
        """No description."""
        self.set_global_parameter("background", 0.0)
        self.set_global_parameter("sigma_res", 0.1)
        self.set_global_parameter("nu_res", 5.0)
        self.set_global_parameter("int_res", 0.0)
        self.set_global_parameter("k_value", 1.0)
        self._add_fitting_success("Global parameters reset to default values")

    def _save_auto_k_enabled(self):
        """auto-K"""
        try:
            self.preferences.set("_auto_k_enabled", self._auto_k_enabled)
            self.preferences.save()
        except Exception as e:
            print(f"Failed to save auto-K setting: {e}")

    def _load_auto_k_enabled(self):
        """No description."""
        try:
            self._auto_k_enabled = self.preferences.get("_auto_k_enabled", False)
            self._update_auto_k_button_style()
        except Exception as e:
            print(f"Failed to load auto-K setting: {e}")
            self._auto_k_enabled = False

    def _update_auto_k_button_style(self):
        """No description."""
        if hasattr(self.ui, "FittingAutoKButton"):
            if self._auto_k_enabled:
                self.ui.FittingAutoKButton.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
                )
                self.ui.FittingAutoKButton.setText("Auto-K: ON")
            else:
                self.ui.FittingAutoKButton.setStyleSheet("")
                self.ui.FittingAutoKButton.setText("Auto-K: OFF")
            self._sync_global_secondary_button_widths()

    def _sync_global_secondary_button_widths(self):
        """Keep Auto-K and step-reset buttons visually aligned after runtime style changes."""
        auto_k_button = getattr(self.ui, "FittingAutoKButton", None)
        if auto_k_button is None:
            return

        parent = auto_k_button.parentWidget()
        if parent is None:
            return

        buttons = [auto_k_button]
        for button in parent.findChildren(QPushButton):
            name = button.objectName() or ""
            if name.endswith("ResetButton") and button not in buttons:
                buttons.append(button)

        if len(buttons) <= 1:
            return

        for button in buttons:
            button.ensurePolished()

        target_width = max(button.sizeHint().width() for button in buttons)
        for button in buttons:
            button.setMinimumWidth(target_width)
            button.setMaximumWidth(target_width)
            button.updateGeometry()

    def _on_auto_k_button_clicked(self):
        """auto-K"""
        self._auto_k_enabled = not self._auto_k_enabled

        self._save_auto_k_enabled()

        self._update_auto_k_button_style()

        status = "enabled" if self._auto_k_enabled else "disabled"
        self._add_fitting_message(f"Auto K-value optimization {status}")

        if self._auto_k_enabled and hasattr(self, "I") and hasattr(self, "I_fitting"):
            if self.I is not None and self.I_fitting is not None:
                self._optimize_k_value()

    def _optimize_k_value(self):
        """Optimize the legacy multiplicative K value through the fitting domain."""
        current_k = None
        try:
            if (
                not hasattr(self, "I")
                or not hasattr(self, "I_fitting")
                or self.I is None
                or self.I_fitting is None
            ):
                self._add_fitting_error("No fitting data available for K-value optimization")
                return
            observed_full = np.asarray(self.I, dtype=float)
            fitted_full = np.asarray(self.I_fitting, dtype=float)
            if observed_full.size == 0 or fitted_full.size == 0:
                self._add_fitting_error("Empty data arrays for K-value optimization")
                return
            if observed_full.shape != fitted_full.shape:
                self._add_fitting_error(
                    f"Data shape mismatch: I{observed_full.shape} vs I_fitting{fitted_full.shape}"
                )
                return
            current_k = float(self.get_global_parameter("k_value"))
            self._add_fitting_message(f"Starting K-value optimization from {current_k:.6f}...")

            observed_used = observed_full
            fitted_used = fitted_full
            q_values = None
            try:
                if getattr(self, "q", None) is not None:
                    q_values = np.asarray(self.q)
            except Exception:
                q_values = None
            roi_min = getattr(self, "_roi_min", None)
            roi_max = getattr(self, "_roi_max", None)
            if (
                q_values is not None
                and q_values.size == observed_full.size == fitted_full.size
                and roi_min is not None
                and roi_max is not None
                and np.isfinite(roi_min)
                and np.isfinite(roi_max)
                and roi_min < roi_max
            ):
                mask = (
                    np.isfinite(q_values)
                    & np.isfinite(observed_full)
                    & np.isfinite(fitted_full)
                    & (q_values >= float(roi_min))
                    & (q_values <= float(roi_max))
                )
                if np.any(mask):
                    observed_used = observed_full[mask]
                    fitted_used = fitted_full[mask]

            optimized = _scientific_commands(self).ai.optimize_scale(
                observed_used, fitted_used, current_k
            )
            k_opt = optimized.scale
            safe_k = max(abs(current_k), 1e-12)
            self.I_fitting = k_opt * (fitted_full / safe_k)

            if not self.set_global_parameter("k_value", k_opt):
                self._add_fitting_error("Failed to set optimized K-value")
                return
            if hasattr(self.ui, "fitKValue"):
                self.ui.fitKValue.blockSignals(True)
                self.ui.fitKValue.setValue(k_opt)
                self.ui.fitKValue.blockSignals(False)
                self._add_fitting_message(f"UI K-value updated to {k_opt:.6f}")
            if isinstance(getattr(self, "fitting", None), dict):
                meta = self.fitting.get("meta") or {}
                parameters = meta.get("params") or {}
                parameters["k"] = float(k_opt)
                meta["params"] = parameters
                self.fitting["meta"] = meta

            self._update_GUI_image("fitting")
            self._update_outside_window("fitting")
            improvement = (
                (optimized.residual_before - optimized.residual_after)
                / max(optimized.residual_before, 1e-12)
            ) * 100
            self._add_fitting_success(
                f"K-value optimized ({optimized.method}): {current_k:.6f} -> {k_opt:.6f}"
            )
            self._add_fitting_success(
                f"Residual improvement: {improvement:.2f}% "
                f"({optimized.residual_before:.6e} -> "
                f"{optimized.residual_after:.6e})"
            )
            self._add_fitting_message(
                f"Data range - I_exp: [{np.min(optimized.observed):.3e}, "
                f"{np.max(optimized.observed):.3e}], I_base: "
                f"[{np.min(optimized.base):.3e}, {np.max(optimized.base):.3e}]"
            )
        except Exception as exc:
            self._add_fitting_error(f"Error during K-value optimization: {exc}")
            if current_k is not None:
                self.set_global_parameter("k_value", current_k)

    def _on_parameter_editing_finished(self, widget_id: int, shape: str, param: str):
        """No description."""
        try:
            if self._loading_parameters or self._initializing:
                return

            param_mapping = self._get_parameter_widget_mapping(widget_id, shape)
            widget_name = param_mapping.get(param)

            if widget_name and hasattr(self.ui, widget_name):
                widget = getattr(self.ui, widget_name)
                current_value = widget.value()

                particle_id = f"particle_{widget_id}"
                success = self.model_params_manager.set_particle_parameter(
                    "fitting", particle_id, shape.lower(), param, current_value
                )

                if success:
                    self.model_params_manager.save_parameters()
                    self._add_particle_message(
                        f"Saved to JSON: {particle_id}.{shape.lower()}.{param} = {current_value}"
                    )
                else:
                    self._add_fitting_error(
                        f"Failed to save parameter: {particle_id}.{shape.lower()}.{param} = {current_value}"
                    )

            is_fitting_mode = self._is_in_fitting_mode()

            if is_fitting_mode:
                self._add_particle_message(
                    f"Fitting mode: auto-updating after {shape}.{param} edit finished"
                )
                self._perform_manual_fitting()
            else:
                self._add_particle_message(
                    f"Normal mode: parameter {shape}.{param} edit finished (saved only)"
                )

        except Exception as e:
            self._add_fitting_error(f"Failed to handle parameter editing finished: {e}")

    def _on_global_parameter_editing_finished(self, param_name: str):
        """No description."""
        try:
            if self._loading_parameters or self._initializing:
                return

            current_value = None
            if param_name == "background" and hasattr(self.ui, "fitBGValue"):
                current_value = self.ui.fitBGValue.value()
            elif param_name == "sigma_res" and hasattr(self.ui, "fitSigmaResValue"):
                current_value = self.ui.fitSigmaResValue.value()
            elif param_name == "nu_res" and hasattr(self.ui, "fitNuResValue"):
                current_value = self.ui.fitNuResValue.value()
            elif param_name == "int_res" and hasattr(self.ui, "fitIntResValue"):
                current_value = self.ui.fitIntResValue.value()
            elif param_name == "k_value" and hasattr(self.ui, "fitKValue"):
                current_value = self.ui.fitKValue.value()

            if current_value is not None:
                success = self.model_params_manager.set_global_parameter(
                    "fitting", param_name, current_value
                )

                if success:
                    self.model_params_manager.save_parameters()
                    self._add_particle_message(
                        f"Saved global parameter to JSON: {param_name} = {current_value}"
                    )
                else:
                    self._add_fitting_error(
                        f"Failed to save global parameter: {param_name} = {current_value}"
                    )

            is_fitting_mode = self._is_in_fitting_mode()

            if is_fitting_mode:
                self._add_particle_message(
                    f"Fitting mode: auto-updating after global {param_name} edit finished"
                )
                self._perform_manual_fitting()
            else:
                self._add_particle_message(
                    f"Normal mode: global parameter {param_name} edit finished (saved only)"
                )

        except Exception as e:
            self._add_fitting_error(f"Failed to handle global parameter editing finished: {e}")

    def _is_in_fitting_mode(self) -> bool:
        """No description."""
        return hasattr(self, "_fitting_mode_active") and self._fitting_mode_active
