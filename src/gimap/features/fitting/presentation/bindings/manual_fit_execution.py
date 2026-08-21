"""Manual Fit Execution for fitting presentation."""

from __future__ import annotations

import time

import numpy as np


from src.gimap.features.fitting.application import (
    ManualFitRequest,
)


from ..binding_primitives import (
    COMPONENT_PARAMETER_SCHEMAS,
)


class ManualFitExecutionMixin:
    """Own manual fit execution behavior."""

    def _perform_manual_fitting(self, *, reveal_result: bool = False):
        """Bridge legacy parameter widgets to the manual fitting ViewModel command."""
        try:
            active_shapes, shape_configs = self._collect_active_particles()
            if not active_shapes:
                self._add_fitting_error("No active particle shapes selected for fitting")
                return
            self._add_fitting_success(f"Active shapes: {active_shapes}")
            self._last_active_particle_ids = shape_configs.copy()

            if (
                hasattr(self.ui, "fitCurrentDataCheckBox")
                and self.ui.fitCurrentDataCheckBox.isChecked()
            ):
                if getattr(self, "current_cut_data", None) is None:
                    self._add_fitting_error("No Cut data available for fitting")
                    return
                q_data = np.asarray(self.current_cut_data["x_coords"], dtype=float)
                intensity_data = np.asarray(
                    self.current_cut_data["y_intensity"], dtype=float
                )
                q_source_kind = "cut"
            else:
                if getattr(self, "current_1d_data", None) is None:
                    self._add_fitting_error("No 1D file data available for fitting")
                    return
                q_data = np.asarray(self.current_1d_data["q"], dtype=float)
                intensity_data = np.asarray(self.current_1d_data["I"], dtype=float)
                q_source_kind = "1d"

            prepared_curve = self._prepare_signed_q_data(q_data, intensity_data)
            q_data = prepared_curve.q

            parameter_aliases = {
                "intensity": "Int",
                "radius": "R",
                "sigma_radius": "sigma_R",
                "height": "h",
                "sigma_height": "sigma_h",
                "diameter": "D",
                "sigma_diameter": "sigma_D",
            }
            parameters = []
            for index, shape in enumerate(active_shapes, 1):
                widget_id = shape_configs[index - 1]
                shape_display = self._shape_display_name(shape)
                shape_values = {}
                for (
                    parameter_key,
                    _suffix,
                    _label,
                    default,
                    _decimals,
                    _step,
                ) in COMPONENT_PARAMETER_SCHEMAS.get(shape_display, []):
                    alias = parameter_aliases[parameter_key]
                    value = self._get_particle_parameter(widget_id, alias, default)
                    parameters.append(value)
                    shape_values[alias] = value
                diameter = shape_values.get("D", 0.0)
                diameter_sigma = shape_values.get("sigma_D", 0.0)
                state = "disabled" if diameter == 0 or diameter_sigma == 0 else "enabled"
                self._add_fitting_success(
                    f"Shape {index} ({shape_display}): Structure factor {state} "
                    f"(D={diameter}, sigma_D={diameter_sigma})"
                )

            global_inputs = (
                ("fitBGValue", "background", 0.0),
                ("fitSigmaResValue", "sigma_res", 0.1),
                ("fitNuResValue", "nu_res", 5.0),
                ("fitIntResValue", "int_res", 0.0),
                ("fitKValue", "k_value", 1.0),
            )
            global_values = []
            for widget_name, global_key, default in global_inputs:
                if hasattr(self.ui, widget_name):
                    value = float(getattr(self.ui, widget_name).value())
                elif hasattr(self, "get_global_parameter"):
                    value = float(self.get_global_parameter(global_key))
                else:
                    value = float(default)
                global_values.append(value)
            parameters.extend(global_values)

            sigma_res, nu_res, int_res = global_values[1:4]
            resolution_state = "disabled" if sigma_res == 0 or int_res == 0 else "active"
            self._add_fitting_success(
                f"Lorentzian resolution {resolution_state}: "
                f"sigma_res={sigma_res}, nu_res={nu_res}, int_res={int_res}"
            )
            request = ManualFitRequest(
                q=q_data,
                q_source_unit=self._get_q_source_unit(q_source_kind),
                shapes=tuple(active_shapes),
                parameters=tuple(parameters),
            )
            result = self.fitting_view_model.run_manual_fit(request)
            self._sync_fitting_workflow()
            if result is None:
                message = (
                    self.fitting_view_model.state.error_message
                    or "Current model calculation failed"
                )
                self._add_fitting_error(message)
                return

            param_dict = dict(zip(result.parameter_names, result.parameters))
            self._add_fitting_success(
                "q converted to internal model unit nm^-1 "
                f"(source={request.q_source_unit}, display={self._get_q_display_unit()})"
            )
            self._add_fitting_success(
                f"Created model with parameters: {list(result.parameter_names)}"
            )
            self._add_fitting_success(f"Using parameters: {param_dict}")
            self._validate_parameter_retrieval(active_shapes, shape_configs)

            fitting_result = result.intensity
            stats = {
                "min": float(np.min(fitting_result)),
                "max": float(np.max(fitting_result)),
                "mean": float(np.mean(fitting_result)),
                "sum": float(np.sum(fitting_result)),
            }
            self._add_fitting_success("Fitting calculation completed successfully")
            self._add_fitting_success(f"Result stats: {stats}")
            self.I_fitting = fitting_result
            self.has_fitting_data = True
            self._has_fitting_data = True
            self.fitting = {
                "q": np.array(result.q, copy=True),
                "I": np.array(fitting_result, copy=True),
                "meta": {
                    "shapes": list(result.shapes),
                    "params": param_dict,
                    "timestamp": time.time(),
                    "source": "fitting",
                    "data_source": q_source_kind,
                    "q_source_unit": request.q_source_unit,
                    "q_model_unit": "nm",
                    "q_branch": prepared_curve.branch,
                    "q_combination": prepared_curve.combination,
                },
            }
            self.display_mode = "fitting"
            self._fitting_mode_active = True
            if reveal_result and hasattr(self, "_set_curve_view_mode"):
                self._set_curve_view_mode("compare", refresh=False)
            if not getattr(self, "_suppress_workflow_plot_updates", False):
                self._update_GUI_image("fitting")
                self._update_outside_window("fitting")
                tabs = getattr(self.ui, "fittingPreviewTabs", None)
                if reveal_result and tabs is not None and np.asarray(fitting_result).size:
                    tabs.setCurrentIndex(1)
            if self._auto_k_enabled and getattr(self, "I", None) is not None:
                try:
                    self._optimize_k_value()
                except Exception as exc:
                    self._add_fitting_error(f"Auto K-value optimization failed: {exc}")
        except Exception as exc:
            self._fail_fitting_step("fit", str(exc))
            self._add_fitting_error(f"Current model calculation failed: {exc}")

    def _store_fitting_data(self, q_data, intensity_data, active_shapes):
        """No description."""
        try:
            self._fitting_q_data = np.array(q_data)
            self._fitting_intensity_data = np.array(intensity_data)
            self._fitting_shapes = active_shapes.copy() if active_shapes else []
            self._has_fitting_data = True

            self._update_gui_fitting_display()

        except Exception as e:
            pass

    def _switch_to_fitting_display_mode(self):
        """No description."""
        try:
            self._display_mode = "fitting"
            self.display_mode = "fitting"
            self._fitting_mode_active = True

            self._refresh_all_displays_for_fitting_mode()

        except Exception as e:
            pass

    def _switch_to_normal_display_mode(self):
        """No description."""
        try:
            self._display_mode = "normal"
            self.display_mode = "normal"
            self._fitting_mode_active = False

            self._fitting_q_data = None
            self._fitting_intensity_data = None
            self._fitting_shapes = []
            self._has_fitting_data = False
            try:
                self.has_fitting_data = False
                self.I_fitting = None
            except Exception:
                pass

        except Exception as e:
            pass

    def _update_gui_fitting_display(self):
        """Render the current fitting plot in fitGraphicsView."""
        try:
            if not hasattr(self, "_fitting_q_data") or self._fitting_q_data is None:
                return
            try:
                self.display_mode = "fitting"
                self._display_mode = "fitting"
                self._fitting_mode_active = True
            except Exception:
                pass

            self._plot_fitting_result(
                self._fitting_q_data, self._fitting_intensity_data, self._fitting_shapes
            )

        except Exception as e:
            pass

    def _refresh_all_displays_for_fitting_mode(self):
        """No description."""
        try:
            if not self._has_fitting_data:
                return

            self._update_gui_fitting_display()

            if (
                hasattr(self, "independent_fit_window")
                and self.independent_fit_window is not None
                and self.independent_fit_window.isVisible()
            ):
                self._refresh_external_window_fitting_display()

        except Exception as e:
            pass

    def _refresh_external_window_fitting_display(self):
        """No description."""
        try:
            if not self._has_fitting_data:
                return

            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            original_x_data = None
            original_y_data = None
            data_label = ""

            if (
                hasattr(self.ui, "fitCurrentDataCheckBox")
                and self.ui.fitCurrentDataCheckBox.isChecked()
            ):
                if hasattr(self, "current_cut_data") and self.current_cut_data is not None:
                    original_x_data = np.array(self.current_cut_data["x_coords"])
                    original_y_data = np.array(self.current_cut_data["y_intensity"])
                    data_label = "Cut Data"
            else:
                if hasattr(self, "current_1d_data") and self.current_1d_data is not None:
                    original_x_data = np.array(self.current_1d_data["q"])
                    original_y_data = np.array(self.current_1d_data["I"])
                    data_label = "1D File Data"

            x_label = self._build_q_axis_label()
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f"Current Model Result - {', '.join(self._fitting_shapes)}"

            self._update_independent_window_with_fitting(
                original_x_data,
                original_y_data,
                data_label,
                self._fitting_q_data,
                self._fitting_intensity_data,
                self._fitting_shapes,
                x_label,
                y_label,
                title,
                log_x,
                log_y,
                normalize,
            )

        except Exception as e:
            pass

    def _get_particle_parameter(self, shape_idx, param_name, default_value):
        """No description."""
        try:
            current_shape = self.get_particle_shape(shape_idx)
            if current_shape == "None":
                return default_value

            control_name = self._get_ui_control_name(shape_idx, current_shape, param_name)
            if control_name and hasattr(self.ui, control_name):
                control = getattr(self.ui, control_name)
                if hasattr(control, "value"):
                    value = control.value()
                    return value
                elif hasattr(control, "text"):
                    try:
                        value = float(control.text())
                        return value
                    except ValueError:
                        pass

            if hasattr(self, "model_params_manager"):
                particle_id = f"particle_{shape_idx}"
                shape_key = self._shape_key(current_shape)
                param_key = self._parameter_key_from_alias(current_shape, param_name)
                value = self.model_params_manager.get_particle_parameter(
                    "fitting", particle_id, shape_key, param_key
                )
                if value is not None:
                    return value

            return default_value

        except Exception as e:
            return default_value

    def _get_ui_control_name(self, shape_idx, shape_name, param_name):
        """No description."""
        try:
            shape_display = self._shape_display_name(shape_name)
            token = self._shape_object_token(shape_display)
            param_key = self._parameter_key_from_alias(shape_display, param_name)
            suffix = None
            for schema_shape, schema in COMPONENT_PARAMETER_SCHEMAS.items():
                if self._shape_key(schema_shape) != self._shape_key(shape_display):
                    continue
                for candidate_key, candidate_suffix, _label, _default, _decimals, _step in schema:
                    if candidate_key == param_key:
                        suffix = candidate_suffix
                        break
                if suffix:
                    break
            if not suffix:
                return None

            candidate = f"fitParticle{token}{suffix}Value_{shape_idx}"
            return candidate if hasattr(self.ui, candidate) else None

        except Exception:
            return None
