"""Manual Refine Setup for fitting presentation."""

from __future__ import annotations

import re


import numpy as np


from ..binding_primitives import (
    COMPONENT_PARAMETER_SCHEMAS,
    _scientific_commands,
)


class ManualRefineSetupMixin:
    """Own manual refine setup behavior."""

    def _build_manual_refine_setup(self):
        try:
            active_shapes, shape_configs = self._collect_active_particles()
            if not active_shapes:
                self._add_fitting_error("No active particle shapes selected for Auto Refine")
                return None

            q_data = None
            y_data = None
            q_source_kind = None
            if (
                hasattr(self.ui, "fitCurrentDataCheckBox")
                and self.ui.fitCurrentDataCheckBox.isChecked()
            ):
                if getattr(self, "current_cut_data", None) is not None:
                    q_data = np.asarray(self.current_cut_data.get("x_coords"), dtype=float)
                    y_data = np.asarray(self.current_cut_data.get("y_intensity"), dtype=float)
                    q_source_kind = "cut"
            else:
                if getattr(self, "current_1d_data", None) is not None:
                    q_data = np.asarray(self.current_1d_data.get("q"), dtype=float)
                    y_data = np.asarray(self.current_1d_data.get("I"), dtype=float)
                    q_source_kind = "1d"
            if q_data is None or y_data is None:
                self._add_fitting_error("No input curve available for Auto Refine")
                return None

            n = min(q_data.size, y_data.size)
            q_data, y_data = q_data[:n], y_data[:n]
            q_data, y_data = self._filter_ai_excluded_points_for_display(q_data, y_data)
            mask = np.isfinite(q_data) & np.isfinite(y_data) & (y_data > 0)
            if self._roi_active():
                lo = min(float(self._roi_min), float(self._roi_max))
                hi = max(float(self._roi_min), float(self._roi_max))
                mask &= (q_data >= lo) & (q_data <= hi)
            q_data, y_data = q_data[mask], y_data[mask]
            if q_data.size < 8:
                self._add_fitting_error(
                    "Auto Refine needs at least 8 valid positive-intensity points"
                )
                return None

            q_model = self._convert_q_values_for_model(q_data, source=q_source_kind)
            model_func = _scientific_commands(self).model.build_function(active_shapes)
            param_names = _scientific_commands(self).model.parameter_names(active_shapes)
            params = self._get_current_manual_param_values(active_shapes, shape_configs)
            if not params or len(params) != len(param_names):
                self._add_fitting_error("Could not read current manual fitting parameters")
                return None

            descriptors = self._build_manual_refine_param_descriptors(
                active_shapes, shape_configs, param_names, params
            )
            self._last_active_particle_ids = shape_configs.copy()
            return {
                "shapes": active_shapes,
                "shape_configs": shape_configs,
                "q_raw": q_data,
                "q_model": q_model,
                "y": y_data,
                "q_source_kind": q_source_kind,
                "model_func": model_func,
                "param_names": param_names,
                "params": descriptors,
            }
        except Exception as exc:
            self._add_fitting_error(f"Auto Refine setup failed: {exc}")
            return None

    def _get_current_manual_param_values(self, active_shapes, shape_configs):
        param_aliases = {
            "intensity": "Int",
            "radius": "R",
            "sigma_radius": "sigma_R",
            "height": "h",
            "sigma_height": "sigma_h",
            "diameter": "D",
            "sigma_diameter": "sigma_D",
        }
        params = []
        for shape, widget_id in zip(active_shapes, shape_configs):
            shape_display = self._shape_display_name(shape)
            schema = COMPONENT_PARAMETER_SCHEMAS.get(shape_display, [])
            for param_key, _suffix, _label, default_value, _decimals, _step in schema:
                alias = param_aliases[param_key]
                params.append(float(self._get_particle_parameter(widget_id, alias, default_value)))

        global_defaults = [
            ("fitBGValue", "background", 0.0),
            ("fitSigmaResValue", "sigma_res", 0.1),
            ("fitNuResValue", "nu_res", 5.0),
            ("fitIntResValue", "int_res", 0.0),
            ("fitKValue", "k_value", 1.0),
        ]
        for widget_name, global_key, default in global_defaults:
            if hasattr(self.ui, widget_name):
                params.append(float(getattr(self.ui, widget_name).value()))
            elif hasattr(self, "get_global_parameter"):
                params.append(float(self.get_global_parameter(global_key)))
            else:
                params.append(float(default))
        return params

    def _build_manual_refine_param_descriptors(
        self, active_shapes, shape_configs, param_names, params
    ):
        descriptors = []
        global_map = {
            "BG": ("fitBGValue", "background", "Global BG"),
            "sigma_Res": ("fitSigmaResValue", "sigma_res", "Global sigma_Res"),
            "nu_Res": ("fitNuResValue", "nu_res", "Global nu_Res"),
            "int_Res": ("fitIntResValue", "int_res", "Global int_Res"),
            "k": ("fitKValue", "k_value", "Global k"),
        }
        for idx, (name, value) in enumerate(zip(param_names, params)):
            match = re.match(r"^(.*?)(\d+)$", str(name))
            desc = {
                "index": idx,
                "name": str(name),
                "value": float(value),
                "scope": "global",
                "label": str(name),
                "widget_name": None,
                "global_key": None,
                "widget_id": None,
                "shape": None,
                "alias": None,
            }
            if match:
                alias = match.group(1)
                seq_index = int(match.group(2))
                widget_id = (
                    shape_configs[seq_index - 1] if 1 <= seq_index <= len(shape_configs) else None
                )
                shape = (
                    active_shapes[seq_index - 1] if 1 <= seq_index <= len(active_shapes) else None
                )
                widget_name = (
                    self._get_ui_control_name(widget_id, shape, alias)
                    if widget_id and shape
                    else None
                )
                desc.update(
                    {
                        "scope": "particle",
                        "label": f"Particle {seq_index} ({self._shape_display_name(shape)} {widget_id}) {alias}",
                        "widget_name": widget_name,
                        "widget_id": widget_id,
                        "shape": shape,
                        "alias": alias,
                    }
                )
            else:
                widget_name, global_key, label = global_map.get(str(name), (None, None, str(name)))
                desc.update(
                    {
                        "scope": "global",
                        "label": label,
                        "widget_name": widget_name,
                        "global_key": global_key,
                    }
                )
            descriptors.append(desc)
        return descriptors

    def _manual_refine_default_selected(self, name: str) -> bool:
        return _scientific_commands(self).refinement.default_selected(name)

    def _manual_refine_dialog_state(self) -> dict:
        try:
            state = self.preferences.get("manual_auto_refine", {})
            return state if isinstance(state, dict) else {}
        except Exception:
            return (
                getattr(self, "_manual_auto_refine_state", {})
                if isinstance(getattr(self, "_manual_auto_refine_state", None), dict)
                else {}
            )

    def _save_manual_refine_dialog_state(self, rows: dict) -> None:
        rows = rows if isinstance(rows, dict) else {}
        self._manual_auto_refine_state = rows
        try:
            self.preferences.set("manual_auto_refine", rows)
            self.preferences.save()
        except Exception:
            pass

    def _default_manual_refine_bounds(self, name: str, value: float):
        return _scientific_commands(self).refinement.default_bounds(name, value)

    def _run_manual_auto_refine(
        self, setup, selected, options, progress_callback=None, stop_callback=None
    ):
        return _scientific_commands(self).refinement.execute(
            setup,
            selected,
            options,
            progress_callback=progress_callback,
            stop_callback=stop_callback,
        )

    def _apply_manual_refine_result(self, setup, refined_params, apply_indices=None):
        old_loading = getattr(self, "_loading_parameters", False)
        self._loading_parameters = True
        try:
            if apply_indices is not None:
                apply_indices = {int(idx) for idx in apply_indices}
            for idx, (desc, value) in enumerate(zip(setup["params"], refined_params)):
                if apply_indices is not None and idx not in apply_indices:
                    continue
                value = float(value)
                widget_name = desc.get("widget_name")
                if widget_name and hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    if hasattr(widget, "blockSignals"):
                        widget.blockSignals(True)
                    try:
                        widget.setValue(value)
                    finally:
                        if hasattr(widget, "blockSignals"):
                            widget.blockSignals(False)

                if desc.get("scope") == "particle":
                    widget_id = desc.get("widget_id")
                    shape = desc.get("shape")
                    alias = desc.get("alias")
                    if widget_id and shape and alias and hasattr(self, "model_params_manager"):
                        particle_id = f"particle_{widget_id}"
                        param_key = self._parameter_key_from_alias(shape, alias)
                        self.model_params_manager.set_particle_parameter(
                            "fitting",
                            particle_id,
                            self._shape_key(shape),
                            param_key,
                            value,
                        )
                elif desc.get("global_key") and hasattr(self, "model_params_manager"):
                    self.model_params_manager.set_global_parameter(
                        "fitting", desc["global_key"], value
                    )
            try:
                self.model_params_manager.save_parameters()
            except Exception:
                pass
        finally:
            self._loading_parameters = old_loading

    def _preview_manual_refine_curve(self, setup, params):
        if params is None:
            return
        try:
            params = np.asarray(params, dtype=float)
            q_raw = None
            if getattr(self, "q", None) is not None:
                q_raw = np.asarray(self.q, dtype=float)
                q_raw = q_raw[np.isfinite(q_raw)]
            if q_raw is None or q_raw.size == 0:
                q_raw = np.asarray(setup.get("q_raw", setup["q_model"]), dtype=float)
            q_model = self._convert_q_values_for_model(q_raw, source=setup.get("q_source_kind"))
            y_fit = np.asarray(setup["model_func"](q_model, *params), dtype=float)
            if y_fit.size == 0:
                return
            param_dict = {
                str(name): float(value) for name, value in zip(setup.get("param_names", []), params)
            }
            self.I_fitting = y_fit
            self.has_fitting_data = True
            self._has_fitting_data = True
            self.fitting = {
                "q": np.array(q_raw[: y_fit.size], copy=True),
                "I": np.array(y_fit, copy=True),
                "meta": {
                    "shapes": list(setup.get("shapes", [])),
                    "params": param_dict,
                    "source": "auto_refine_preview",
                    "data_source": setup.get("q_source_kind"),
                    "q_source_unit": self._get_q_source_unit(setup.get("q_source_kind")),
                    "q_model_unit": "nm",
                    "preview": True,
                },
            }
            self.display_mode = "fitting"
            self._display_mode = "fitting"
            self._fitting_mode_active = True
            self._update_GUI_image("fitting")
            self._update_outside_window("fitting")
        except Exception as exc:
            self._add_fitting_error(f"Auto Refine preview update failed: {exc}")
