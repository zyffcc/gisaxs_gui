"""Particle Connections for fitting presentation."""

from __future__ import annotations


from ..binding_primitives import (
    COMPONENT_FORMULA_TOOLTIPS,
    COMPONENT_PARAMETER_SCHEMAS,
)


class ParticleConnectionsMixin:
    """Own particle connections behavior."""

    def _setup_particle_connections(self, widget_ids=None):
        """widget"""
        widget_ids = widget_ids or self._iter_particle_widget_ids()
        for widget_id in widget_ids:
            config = self.particle_shape_configs[widget_id]
            if hasattr(self.ui, config["combobox"]):
                combobox = getattr(self.ui, config["combobox"])

                combobox.currentIndexChanged.connect(
                    lambda index, wid=widget_id: self._on_particle_shape_changed(wid, index)
                )
                combobox.currentTextChanged.connect(
                    lambda text, combo=combobox: combo.setToolTip(
                        COMPONENT_FORMULA_TOOLTIPS.get(text, COMPONENT_FORMULA_TOOLTIPS["None"])
                    )
                )

                self._add_fitting_message(
                    f"Connected Particle Widget {widget_id}: {config['combobox']} -> {config['stack_widget']}",
                    "INFO",
                )

    def _setup_parameter_ranges(self, widget_ids=None):
        """No description."""
        min_value = -1e10
        max_value = 1e10
        decimals = 2

        widgets_set = 0

        # Configure component parameter controls from the dynamic schema.
        widget_ids = widget_ids or self._iter_particle_widget_ids()
        for widget_id in widget_ids:
            for shape_name, schema in COMPONENT_PARAMETER_SCHEMAS.items():
                mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
                decimals_by_param = {
                    param_key: param_decimals
                    for param_key, _suffix, _label, _default, param_decimals, _step in schema
                }
                step_by_param = {
                    param_key: step
                    for param_key, _suffix, _label, _default, _decimals, step in schema
                }
                for param_key, widget_name in mapping.items():
                    if hasattr(self.ui, widget_name):
                        widget = getattr(self.ui, widget_name)
                        widget.setRange(min_value, max_value)
                        widget.setDecimals(decimals_by_param.get(param_key, decimals))
                        widget.setSingleStep(step_by_param.get(param_key, 0.1))
                        widgets_set += 1

        if hasattr(self.ui, "fitBGValue"):
            self.ui.fitBGValue.setRange(min_value, max_value)
            self.ui.fitBGValue.setDecimals(6)
            self.ui.fitBGValue.setSingleStep(0.1)
            widgets_set += 1

        if hasattr(self.ui, "fitSigmaResValue"):
            self.ui.fitSigmaResValue.setRange(min_value, max_value)
            self.ui.fitSigmaResValue.setDecimals(6)
            self.ui.fitSigmaResValue.setSingleStep(0.1)
            widgets_set += 1

        if hasattr(self.ui, "fitNuResValue"):
            self.ui.fitNuResValue.setRange(min_value, max_value)
            self.ui.fitNuResValue.setDecimals(4)
            self.ui.fitNuResValue.setSingleStep(0.1)
            widgets_set += 1

        if hasattr(self.ui, "fitIntResValue"):
            self.ui.fitIntResValue.setRange(min_value, max_value)
            self.ui.fitIntResValue.setDecimals(6)
            self.ui.fitIntResValue.setSingleStep(0.01)
            widgets_set += 1

        if hasattr(self.ui, "fitKValue"):
            self.ui.fitKValue.setRange(min_value, max_value)
            self.ui.fitKValue.setDecimals(4)
            self.ui.fitKValue.setSingleStep(0.1)
            widgets_set += 1

        self._add_fitting_success(
            f"Set ranges for {widgets_set} parameter widgets: [{min_value}, {max_value}] with {decimals} decimals"
        )

    def _setup_particle_parameter_connections(self, widget_ids=None):
        """No description."""
        widget_ids = widget_ids or self._iter_particle_widget_ids()
        for widget_id in widget_ids:
            for shape_name in COMPONENT_PARAMETER_SCHEMAS:
                mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
                shape_lower = self._shape_key(shape_name)
                for param_key, widget_name in mapping.items():
                    if not hasattr(self.ui, widget_name):
                        continue
                    w = getattr(self.ui, widget_name)

                    # 函数说明：实现 after commit 相关逻辑。
                    def _after_commit(info, value, wid=widget_id, shp=shape_lower, p=param_key):
                        try:
                            self._add_particle_message(f"Meta commit {wid}.{shp}.{p} = {value}")
                            has_data = (
                                hasattr(self, "current_cut_data")
                                and self.current_cut_data is not None
                            ) or (
                                hasattr(self, "current_1d_data")
                                and self.current_1d_data is not None
                            )
                            if has_data:
                                try:
                                    self.display_mode = "fitting"
                                    self._display_mode = "fitting"
                                    self._fitting_mode_active = True
                                except Exception:
                                    pass
                                self._perform_manual_fitting()
                        except Exception:
                            pass

                    widget_mode = self._signal_mode_overrides.get(
                        widget_name, self._default_signal_mode
                    )
                    meta = {
                        "persist": "model_particle",
                        "particle_id": f"particle_{widget_id}",
                        "shape": shape_lower,
                        "param": param_key,
                        "trigger_fit": True,
                        "debounce_ms": self._param_debounce_ms,
                        "epsilon_abs": self._param_abs_eps,
                        "epsilon_rel": self._param_rel_eps,
                        "after_commit": _after_commit,
                        "connect_mode": widget_mode,
                    }
                    meta_id = f"meta_particle_{widget_id}_{shape_lower}_{param_key}"
                    self.param_trigger_manager.register_parameter_widget(
                        widget=w,
                        widget_id=meta_id,
                        category="fitting_particles",
                        immediate_handler=lambda v: None,
                        delayed_handler=None,
                        connect_signals=True,
                        meta=meta,
                    )
                    self._particle_parameter_meta_ids[widget_id].append(meta_id)

    def _setup_global_parameter_connections(self):
        """No description."""
        mapping = [
            ("fitBGValue", "background"),
            ("fitSigmaResValue", "sigma_res"),
            ("fitNuResValue", "nu_res"),
            ("fitIntResValue", "int_res"),
            ("fitKValue", "k_value"),
        ]
        for widget_name, param_key in mapping:
            if not hasattr(self.ui, widget_name):
                continue
            w = getattr(self.ui, widget_name)

            # 函数说明：实现 after commit 相关逻辑。
            def _after_commit(info, value, p=param_key):
                try:
                    self._add_particle_message(f"Meta commit global {p} = {value}")
                    has_data = (
                        hasattr(self, "current_cut_data") and self.current_cut_data is not None
                    ) or (hasattr(self, "current_1d_data") and self.current_1d_data is not None)
                    if has_data:
                        try:
                            self.display_mode = "fitting"
                            self._display_mode = "fitting"
                            self._fitting_mode_active = True
                        except Exception:
                            pass
                        self._perform_manual_fitting()
                except Exception:
                    pass

            widget_mode = self._signal_mode_overrides.get(widget_name, self._default_signal_mode)
            meta = {
                "persist": "model_global",
                "param": param_key,
                "trigger_fit": True,
                "debounce_ms": self._param_debounce_ms,
                "epsilon_abs": self._param_abs_eps,
                "epsilon_rel": self._param_rel_eps,
                "after_commit": _after_commit,
                "connect_mode": widget_mode,
            }
            self.param_trigger_manager.register_parameter_widget(
                widget=w,
                widget_id=f"meta_global_{param_key}",
                category="fitting_global",
                immediate_handler=lambda v: None,
                delayed_handler=None,
                connect_signals=True,
                meta=meta,
            )
            self._add_fitting_message(
                f"Connected (meta, mode={self._signal_mode_overrides.get(widget_name, self._default_signal_mode)}) {widget_name}",
                "INFO",
            )
