"""Particle Model State for fitting presentation."""

from __future__ import annotations


from pathlib import Path


from src.gimap.features.fitting.presentation.layout_primitives import (
    CurrentPageHeightStackedWidget,
)


from ..binding_primitives import (
    COMPONENT_PARAMETER_SCHEMAS,
)


class ParticleModelStateMixin:
    """Own particle model state behavior."""

    def _initialize_global_parameters(self):
        """No description."""
        try:
            if hasattr(self.ui, "fitBGValue"):
                saved_value = self.model_params_manager.get_global_parameter(
                    "fitting", "background", 0.0
                )
                self.ui.fitBGValue.blockSignals(True)
                self.ui.fitBGValue.setValue(saved_value)
                self.ui.fitBGValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitBGValue to {saved_value}", "INFO")

            if hasattr(self.ui, "fitSigmaResValue"):
                saved_value = self.model_params_manager.get_global_parameter(
                    "fitting", "sigma_res", 0.1
                )
                self.ui.fitSigmaResValue.blockSignals(True)
                self.ui.fitSigmaResValue.setValue(saved_value)
                self.ui.fitSigmaResValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitSigmaResValue to {saved_value}", "INFO")

            if hasattr(self.ui, "fitNuResValue"):
                saved_value = self.model_params_manager.get_global_parameter(
                    "fitting", "nu_res", 5.0
                )
                self.ui.fitNuResValue.blockSignals(True)
                self.ui.fitNuResValue.setValue(saved_value)
                self.ui.fitNuResValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitNuResValue to {saved_value}", "INFO")

            if hasattr(self.ui, "fitIntResValue"):
                saved_value = self.model_params_manager.get_global_parameter(
                    "fitting", "int_res", 0.0
                )
                self.ui.fitIntResValue.blockSignals(True)
                self.ui.fitIntResValue.setValue(saved_value)
                self.ui.fitIntResValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitIntResValue to {saved_value}", "INFO")

            if hasattr(self.ui, "fitKValue"):
                saved_value = self.model_params_manager.get_global_parameter(
                    "fitting", "k_value", 1.0
                )
                self.ui.fitKValue.blockSignals(True)
                self.ui.fitKValue.setValue(saved_value)
                self.ui.fitKValue.blockSignals(False)
                self._add_fitting_message(f"Initialized fitKValue to {saved_value}", "INFO")

        except Exception as e:
            self._add_fitting_error(f"Failed to initialize global parameters: {e}")

    def _initialize_particle_states(self, widget_ids=None):
        """SON"""
        try:
            self._initializing = True

            target_ids = widget_ids or self._iter_particle_widget_ids()
            for widget_id in target_ids:
                particle_id = f"particle_{widget_id}"

                saved_shape = self.model_params_manager.get_particle_shape("fitting", particle_id)
                is_enabled = self.model_params_manager.get_particle_enabled("fitting", particle_id)

                self._add_fitting_message(
                    f"Initializing {particle_id}: shape={saved_shape}, enabled={is_enabled}", "INFO"
                )

                config = self.particle_shape_configs[widget_id]
                if hasattr(self.ui, config["combobox"]):
                    combobox = getattr(self.ui, config["combobox"])

                    combo_index = None
                    for index, page_config in config["pages"].items():
                        if page_config["name"] == saved_shape:
                            combo_index = index
                            break

                    if combo_index is not None:
                        combobox.blockSignals(True)
                        combobox.setCurrentIndex(combo_index)
                        combobox.blockSignals(False)

                        page_config = config["pages"][combo_index]
                        self._switch_particle_page(widget_id, page_config, saved_shape)

                        if not is_enabled or saved_shape == "None":
                            self._freeze_particle_controls(widget_id)
                            self._add_fitting_message(
                                f"{particle_id} controls frozen (disabled/None)", "INFO"
                            )
                        else:
                            self._unfreeze_particle_controls(widget_id, saved_shape)
                            self._load_particle_parameters_from_json(widget_id, saved_shape)
                            self._add_fitting_message(
                                f"{particle_id} controls active with {saved_shape} parameters",
                                "INFO",
                            )

        except Exception as e:
            self._add_fitting_error(f"Failed to initialize particle states: {e}")
            import traceback

            self._add_fitting_error(f"Traceback: {traceback.format_exc()}")
        finally:
            self._initializing = False
            self._schedule_model_parameters_region_refresh()

    def _set_particle_page_and_state(self, widget_id: int, combo_index: int, shape_name: str):
        """No description."""
        config = self.particle_shape_configs[widget_id]
        page_config = config["pages"][combo_index]

        if hasattr(self.ui, config["stack_widget"]):
            stack_widget = getattr(self.ui, config["stack_widget"])

            stack_widget.setCurrentIndex(page_config["page_index"])

            if shape_name == "None":
                self._set_particle_none_state(widget_id)
            else:
                self._set_particle_active_state(widget_id, shape_name)

    def _load_particle_parameters(self, widget_id: int, shape_name: str):
        """UI"""
        if shape_name == "None":
            return

        try:
            particle_id = f"particle_{widget_id}"
            shape_key = self._shape_key(shape_name)

            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)

            for param_key, widget_name in param_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)

                    value = self.model_params_manager.get_particle_parameter(
                        "fitting", particle_id, shape_key, param_key
                    )

                    if value is not None:
                        widget.blockSignals(True)
                        widget.setValue(value)
                        widget.blockSignals(False)

                        self._add_particle_message(
                            f"Loaded {param_key}={value} for particle {widget_id} ({shape_name})"
                        )
                    else:
                        self._add_particle_message(
                            f"No value found for {param_key} in particle {widget_id} ({shape_name})"
                        )

        except Exception as e:
            self._add_fitting_error(f"Failed to load parameters for particle {widget_id}: {e}")

    def _get_parameter_widget_mapping(self, widget_id: int, shape_name: str) -> dict:
        """Return parameter key to spinbox object-name mapping for a component."""
        for schema_shape, schema in COMPONENT_PARAMETER_SCHEMAS.items():
            if self._shape_key(schema_shape) == self._shape_key(shape_name):
                token = self._shape_object_token(schema_shape)
                return {
                    param_key: f"fitParticle{token}{suffix}Value_{widget_id}"
                    for param_key, suffix, _label, _default, _decimals, _step in schema
                }
        return {}

    def _on_particle_shape_changed(self, widget_id: int, combo_index: int):
        """Handle a particle component selection change and persist it to JSON."""
        config = self.particle_shape_configs[widget_id]
        page_config = config["pages"][combo_index]

        if hasattr(self.ui, config["stack_widget"]):
            stack_widget = getattr(self.ui, config["stack_widget"])
            shape_name = page_config["name"]

            particle_id = f"particle_{widget_id}"
            current_shape = self.model_params_manager.get_particle_shape("fitting", particle_id)

            if current_shape == shape_name:
                self._add_particle_message(
                    f"Particle {widget_id} already in {shape_name} state; skipping"
                )
                return

            self._add_particle_message(
                f"Particle {widget_id} shape changing: {current_shape} -> {shape_name}"
            )

            if shape_name == "None":
                self.model_params_manager.set_particle_shape("fitting", particle_id, "None")
                self.model_params_manager.set_particle_enabled("fitting", particle_id, False)
                self._add_particle_message(f"Saved {particle_id} as None (disabled)")
            else:
                self.model_params_manager.set_particle_shape("fitting", particle_id, shape_name)
                self.model_params_manager.set_particle_enabled("fitting", particle_id, True)
                self._add_particle_message(f"Saved {particle_id} as {shape_name} (enabled)")

            self.model_params_manager.save_parameters()

            self._switch_particle_page(widget_id, page_config, shape_name)

            if shape_name == "None":
                self._freeze_particle_controls(widget_id)
                self._add_particle_message(f"Particle {widget_id} controls frozen (None state)")
            else:
                self._unfreeze_particle_controls(widget_id, shape_name)
                self._load_particle_parameters_from_json(widget_id, shape_name)
                self._add_particle_message(
                    f"Particle {widget_id} controls unfrozen ({shape_name} state)"
                )

            self._schedule_model_parameters_region_refresh()

    def _switch_particle_page(self, widget_id: int, page_config: dict, shape_name: str):
        """No description."""
        config = self.particle_shape_configs[widget_id]
        if hasattr(self.ui, config["stack_widget"]):
            stack_widget = getattr(self.ui, config["stack_widget"])
            target_page_index = page_config["page_index"]
            current_page_index = stack_widget.currentIndex()

            self._add_particle_message(
                f"Switching page: {current_page_index} -> {target_page_index} for {shape_name}"
            )

            if target_page_index == current_page_index:
                temp_page_index = 1 if target_page_index == 0 else 0
                stack_widget.setCurrentIndex(temp_page_index)
                from PyQt5.QtWidgets import QApplication

                QApplication.processEvents()
                stack_widget.setCurrentIndex(target_page_index)
                self._add_particle_message(
                    f"Forced refresh: temp({temp_page_index}) -> {target_page_index}"
                )
            else:
                stack_widget.setCurrentIndex(target_page_index)

            if isinstance(stack_widget, CurrentPageHeightStackedWidget):
                stack_widget.sync_current_height()
            stack_widget.updateGeometry()
            parent_widget = stack_widget.parentWidget()
            if parent_widget is not None:
                self._sync_particle_widget_height(parent_widget)
            self._schedule_model_parameters_region_refresh()

            final_page_index = stack_widget.currentIndex()
            if final_page_index == target_page_index:
                self._add_particle_message(f"Page switch confirmed: {final_page_index}")
            else:
                self._add_particle_message(
                    f"Page switch failed: expected {target_page_index}, got {final_page_index}"
                )

    def _freeze_particle_controls(self, widget_id: int):
        """None"""
        for shape_name in COMPONENT_PARAMETER_SCHEMAS:
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            for param_key, widget_name in param_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    widget.setEnabled(False)

    def _unfreeze_particle_controls(self, widget_id: int, active_shape: str):
        """No description."""
        for shape_name in COMPONENT_PARAMETER_SCHEMAS:
            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            is_active = self._shape_key(shape_name) == self._shape_key(active_shape)

            for param_key, widget_name in param_mapping.items():
                if hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    widget.setEnabled(is_active)

    def _load_particle_parameters_from_json(self, widget_id: int, shape_name: str):
        """Load particle parameters from JSON into the UI."""
        try:
            self._loading_parameters = True

            particle_id = f"particle_{widget_id}"
            shape_key = self._shape_key(shape_name)
            shape_params = self.model_params_manager.get_particle_parameter(
                "fitting", particle_id, shape_key
            )

            if not shape_params:
                self._add_particle_message(
                    f"No parameters found in JSON for {particle_id}.{shape_name}"
                )
                return

            param_mapping = self._get_parameter_widget_mapping(widget_id, shape_name)
            loaded_count = 0

            for param_key, widget_name in param_mapping.items():
                if param_key in shape_params and hasattr(self.ui, widget_name):
                    widget = getattr(self.ui, widget_name)
                    value = shape_params[param_key]
                    widget.setValue(value)
                    loaded_count += 1
                    self._add_particle_message(
                        f"Loaded {param_key}={value} for {particle_id}.{shape_name}"
                    )

            self._add_particle_message(
                f"Loaded {loaded_count} parameters from JSON for {particle_id}.{shape_name}"
            )

        except Exception as e:
            self._add_fitting_error(f"Failed to load parameters from JSON: {e}")
        finally:
            self._loading_parameters = False

    def set_particle_shape(self, widget_id: int, shape_name: str):
        """Set the active shape for a particle widget.

        Args:
            widget_id: Particle widget index, such as 1, 2, or 3.
            shape_name: Shape name, such as ``Sphere``, ``Cylinder``, or ``None``.
        """
        config = self.particle_shape_configs.get(widget_id)
        if not config:
            self._add_fitting_warning(f"Particle Widget {widget_id} not found")
            return False

        combo_index = None
        for index, page_config in config["pages"].items():
            if page_config["name"] == shape_name:
                combo_index = index
                break

        if combo_index is None:
            self._add_fitting_warning(
                f"Shape {shape_name} not found for particle widget {widget_id}"
            )
            return False

        if hasattr(self.ui, config["combobox"]):
            combobox = getattr(self.ui, config["combobox"])
            combobox.setCurrentIndex(combo_index)
            return True

        return False

    def get_particle_shape(self, widget_id: int) -> str:
        """Return the active shape for a particle widget.

        Args:
            widget_id: Particle widget index, such as 1, 2, or 3.

        Returns:
            Active shape name, or ``None`` when unavailable.
        """
        config = self.particle_shape_configs.get(widget_id)
        if not config:
            return "None"

        if hasattr(self.ui, config["combobox"]):
            combobox = getattr(self.ui, config["combobox"])
            current_index = combobox.currentIndex()

            page_config = config["pages"].get(current_index)
            if page_config:
                return page_config["name"]

        return "None"

    def get_particles_status(self) -> dict:
        """No description."""
        status = {}
        for widget_id in self._iter_particle_widget_ids():
            status[widget_id] = self.get_particle_shape(widget_id)
        return status

    def reset_all_particles(self):
        """No description."""
        for widget_id in self._iter_particle_widget_ids():
            self.set_particle_shape(widget_id, "None")
        self._add_fitting_success("All particles reset to None state")

    def add_new_particle_shape(self, shape_name: str, control_types: list):
        """Add a new particle shape definition to the particle widget pages.

        Args:
            shape_name: Shape name, for example ``Ellipsoid``.
            control_types: Control identifiers, for example ``['Int', 'Ra', 'Rb', 'Rc', 'D', 'BG']``.
        """
        self.particle_control_types[shape_name] = control_types

        for widget_id in self._iter_particle_widget_ids():
            pages = self.particle_shape_configs[widget_id]["pages"]
            new_index = len(pages) - 1

            none_config = pages.pop(len(pages) - 1)

            pages[new_index] = {
                "name": shape_name,
                "page_index": new_index,
            }

            pages[len(pages)] = none_config

        self._add_fitting_success(
            f"Added new particle shape: {shape_name} with {len(control_types)} controls"
        )
        self._add_fitting_warning(
            "Note: You need to add corresponding UI pages and ComboBox items manually"
        )

    def get_all_particle_parameters(self) -> dict:
        """No description."""
        return self.model_params_manager.get_all_particles("fitting")

    def save_particle_parameters(self) -> bool:
        """No description."""
        return self.model_params_manager.save_parameters()

    def reload_particle_parameters(self) -> bool:
        """I"""
        success = self.model_params_manager.load_parameters()
        if success:
            self._initialize_particle_states()
            self._initialize_global_parameters()
            self._add_fitting_success("Particle and global parameters reloaded from file")
        else:
            self._add_fitting_error("Failed to reload particle parameters")
        return success

    def export_particle_parameters(self, filepath: str) -> bool:
        """No description."""
        try:
            self.fitting_view_model.storage.export_model_parameters(
                Path(self.model_params_manager.config_file),
                Path(filepath),
            )
            self._add_fitting_success(f"Particle parameters exported to: {filepath}")
            return True
        except Exception as e:
            self._add_fitting_error(f"Failed to export parameters: {e}")
            return False

    def import_particle_parameters(self, filepath: str) -> bool:
        """No description."""
        try:
            self.fitting_view_model.storage.import_model_parameters(
                Path(filepath),
                Path(self.model_params_manager.config_file),
            )
            success = self.reload_particle_parameters()
            if success:
                self._add_fitting_success(f"Particle parameters imported from: {filepath}")
            return success
        except Exception as e:
            self._add_fitting_error(f"Failed to import parameters: {e}")
            return False
