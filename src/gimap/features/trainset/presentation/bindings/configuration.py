"""Configuration coordination for Trainset."""

from __future__ import annotations

import copy


import sys


from typing import Any, Dict


from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
)


from ..config_fields import _deep_get, _deep_set


class ConfigurationMixin:
    """Own configuration presentation behavior."""

    def _schedule_autosave_from_page(self, *_args) -> None:
        if self._applying_config or not self.page.auto_remember_check.isChecked():
            return
        self._autosave_timer.start()

    def _auto_remember_toggled(self, checked: bool) -> None:
        if self._applying_config:
            return
        if checked:
            self._autosave_timer.start(100)
            self.status_updated.emit("Automatic TrainSet memory enabled")
            return
        self._autosave_timer.stop()
        # Persist the opt-out itself once; later edits are deliberately ignored.
        config = self._collect_config()
        self.trainset_view_model.save_settings(copy.deepcopy(config))
        self.status_updated.emit(
            "Automatic TrainSet memory disabled · later edits will not replace the remembered settings"
        )

    def _persist_current_config(self) -> None:
        if self._applying_config or not self.page.auto_remember_check.isChecked():
            return
        try:
            config = self._collect_config()
            self.trainset_view_model.save_settings(copy.deepcopy(config))
            self.status_updated.emit("TrainSet settings remembered automatically")
        except Exception as exc:
            self.status_updated.emit(f"Could not remember TrainSet settings: {exc}")

    def _widget_value(self, widget):
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        return None

    def _set_widget_value(self, widget, value) -> None:
        widget.blockSignals(True)
        try:
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                if widget.findText(str(value)) < 0:
                    widget.addItem(str(value))
                widget.setCurrentText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
        finally:
            widget.blockSignals(False)

    def _collect_config(self) -> Dict[str, Any]:
        config = copy.deepcopy(self.config)
        config["project"]["name"] = self.page.project_name.text().strip()
        config.setdefault("runtime", {})["auto_remember"] = (
            self.page.auto_remember_check.isChecked()
        )
        for path, widget in self.page.fields.items():
            if path.startswith("pre.") or path in {
                "sample.particle_label",
                "sample.particle_material",
                "sample.interference_label",
            }:
                continue
            _deep_set(config, path, self._widget_value(widget))

        particle_label = self.page.fields["sample.particle_label"].currentText()
        particle = next(
            (spec for spec in self.catalog.plugins("particle") if spec.label == particle_label),
            None,
        )
        config["sample"]["particles"] = [
            {
                "plugin": particle.key if particle else "spherical_segment",
                "material": self.page.fields["sample.particle_material"].currentText(),
                "enabled": True,
                "parameters": self.page.plugin_parameters(self.page.particle_parameter_table),
            }
        ]
        interference_label = self.page.fields["sample.interference_label"].currentText()
        interference = next(
            (
                spec
                for spec in self.catalog.plugins("interference")
                if spec.label == interference_label
            ),
            None,
        )
        config["sample"]["interference"] = {
            "plugin": interference.key if interference else "none",
            "enabled": bool(interference and interference.key != "none"),
            "parameters": self.page.plugin_parameters(self.page.interference_parameter_table),
        }
        layers = []
        for row in range(self.page.layer_table.rowCount()):
            values = [
                self.page.layer_table.item(row, column).text()
                if self.page.layer_table.item(row, column)
                else ""
                for column in range(6)
            ]
            if values[1]:
                layers.append(
                    {
                        "enabled": values[0].strip().lower() not in {"0", "false", "no"},
                        "material": values[1],
                        "thickness_nm": {
                            "minimum": float(values[2] or 0),
                            "maximum": float(values[3] or values[2] or 0),
                        },
                        "roughness_nm": {
                            "minimum": float(values[4] or 0),
                            "maximum": float(values[5] or values[4] or 0),
                        },
                    }
                )
        config["sample"]["layers"] = layers
        config["mask"]["fixed_shapes"] = self.page.mask_shapes()

        background_step: Dict[str, Any] = {
            "plugin": "physical_background",
            "enabled": self.page.fields["pre.background.enabled"].isChecked(),
        }
        for definition in self.catalog.background_parameters():
            key = str(definition["key"])
            background_step[f"{key}_min"] = self.page.fields[f"pre.background.{key}.min"].value()
            background_step[f"{key}_max"] = self.page.fields[f"pre.background.{key}.max"].value()
        config["preprocessing"]["steps"] = [
            background_step,
            {
                "plugin": "gaussian_noise",
                "enabled": self.page.fields["pre.gaussian.enabled"].isChecked(),
                "snr_min_db": self.page.fields["pre.gaussian.min"].value(),
                "snr_max_db": self.page.fields["pre.gaussian.max"].value(),
            },
            {
                "plugin": "poisson_noise",
                "enabled": self.page.fields["pre.poisson.enabled"].isChecked(),
                "count_scale_min": self.page.fields["pre.poisson.min"].value(),
                "count_scale_max": self.page.fields["pre.poisson.max"].value(),
            },
            {"plugin": "mask", "enabled": self.page.fields["pre.mask.enabled"].isChecked()},
            {
                "plugin": "log",
                "enabled": self.page.fields["pre.log.enabled"].isChecked(),
                "epsilon": 1e-6,
            },
            {
                "plugin": "normalize",
                "enabled": self.page.fields["pre.normalize.enabled"].isChecked(),
                "mode": self.page.fields["pre.normalize.mode"].currentText(),
                "lower": self.page.fields["pre.normalize.lower"].value(),
                "upper": self.page.fields["pre.normalize.upper"].value(),
            },
            {
                "plugin": "random_edge_crop",
                "enabled": self.page.fields["pre.edge.enabled"].isChecked(),
                "maximum_px": self.page.fields["pre.edge.maximum"].value(),
            },
        ]
        config["model"]["layers"] = self.page.model_layers()
        config = self.trainset_view_model.synchronize_config(config)
        self.config = config
        self.page.update_cache_grid_summary(config)
        self.parameters_changed.emit("Trainset parameters", copy.deepcopy(config))
        return config

    def _apply_config_to_page(self, config: Dict[str, Any]) -> None:
        self._applying_config = True
        config = self.trainset_view_model.synchronize_config(config)
        self.config = config
        self.page.project_name.setText(str(config.get("project", {}).get("name", "")))
        self.page.auto_remember_check.blockSignals(True)
        self.page.auto_remember_check.setChecked(
            bool(config.get("runtime", {}).get("auto_remember", True))
        )
        self.page.auto_remember_check.blockSignals(False)
        special = {
            "sample.particle_label": next(
                (
                    spec.label
                    for spec in self.catalog.plugins("particle")
                    if spec.key == config["sample"]["particles"][0]["plugin"]
                ),
                "Spherical segment",
            ),
            "sample.particle_material": config["sample"]["particles"][0].get("material", "Copper"),
            "sample.interference_label": next(
                (
                    spec.label
                    for spec in self.catalog.plugins("interference")
                    if spec.key == config["sample"]["interference"].get("plugin")
                ),
                "None",
            ),
        }
        for path, widget in self.page.fields.items():
            if path in special:
                self._set_widget_value(widget, special[path])
            elif not path.startswith("pre."):
                value = _deep_get(config, path)
                if value is not None:
                    self._set_widget_value(widget, value)
        if not self.page.fields["project.workspace"].text().strip():
            self.page.fields["project.workspace"].setText(str(self.project_root / "trainset_jobs"))
        if not self.page.fields["training.local_python"].text().strip():
            self.page.fields["training.local_python"].setText(sys.executable)
        steps = {step["plugin"]: step for step in config.get("preprocessing", {}).get("steps", [])}
        background = steps.get("physical_background", {})
        gaussian = steps.get("gaussian_noise", steps.get("noise", {}))
        poisson = steps.get("poisson_noise", {})
        pre_map = {
            "pre.background.enabled": background.get("enabled", False),
            "pre.gaussian.enabled": gaussian.get("enabled", True),
            "pre.gaussian.min": gaussian.get("snr_min_db", 80.0),
            "pre.gaussian.max": gaussian.get("snr_max_db", 110.0),
            "pre.poisson.enabled": poisson.get("enabled", False),
            "pre.poisson.min": poisson.get("count_scale_min", 1.0),
            "pre.poisson.max": poisson.get("count_scale_max", 20.0),
            "pre.mask.enabled": steps.get("mask", {}).get("enabled", True),
            "pre.log.enabled": steps.get("log", {}).get("enabled", True),
            "pre.normalize.enabled": steps.get("normalize", {}).get("enabled", True),
            "pre.normalize.mode": steps.get("normalize", {}).get("mode", "range"),
            "pre.normalize.lower": steps.get("normalize", {}).get("lower", 0.0),
            "pre.normalize.upper": steps.get("normalize", {}).get("upper", 1.0),
            "pre.edge.enabled": steps.get("random_edge_crop", {}).get("enabled", False),
            "pre.edge.maximum": steps.get("random_edge_crop", {}).get("maximum_px", 4),
        }
        for definition in self.catalog.background_parameters():
            key = str(definition["key"])
            legacy_min = (
                background.get("fraction_min", definition["minimum"])
                if key == "target_fraction"
                else definition["minimum"]
            )
            legacy_max = (
                background.get("fraction_max", definition["maximum"])
                if key == "target_fraction"
                else definition["maximum"]
            )
            pre_map[f"pre.background.{key}.min"] = background.get(f"{key}_min", legacy_min)
            pre_map[f"pre.background.{key}.max"] = background.get(f"{key}_max", legacy_max)
        for path, value in pre_map.items():
            self._set_widget_value(self.page.fields[path], value)
        self._update_threshold_controls()

        self.page.mask_shape_table.setRowCount(0)
        for shape in config.get("mask", {}).get("fixed_shapes", []):
            self.page.add_mask_shape(shape)
        particle_config = config["sample"]["particles"][0]
        particle_plugin = self.catalog.plugin("particle", particle_config["plugin"])
        self.page.particle_help.setText(particle_plugin.description)
        self.page.set_plugin_parameters(
            self.page.particle_parameter_table,
            particle_plugin.parameters,
            particle_config.get("parameters", {}),
        )
        interference_config = config["sample"]["interference"]
        interference_plugin = self.catalog.plugin(
            "interference", interference_config.get("plugin", "none")
        )
        self.page.interference_help.setText(interference_plugin.description)
        self.page.set_plugin_parameters(
            self.page.interference_parameter_table,
            interference_plugin.parameters,
            interference_config.get("parameters", {}),
        )
        self.page.segment_constraint_check.setVisible(
            particle_config.get("plugin") == "spherical_segment"
        )
        is_paracrystal = interference_config.get("plugin") == "paracrystal"
        self.page.spacing_constraint_check.setEnabled(
            is_paracrystal and "radius_nm" in particle_config.get("parameters", {})
        )
        self.page.spacing_constraint_check.setVisible(
            is_paracrystal and "radius_nm" in particle_config.get("parameters", {})
        )
        self.page.random_mask_panel.setVisible(config.get("mask", {}).get("mode") == "random")
        self.page.layer_table.setRowCount(0)
        for row, layer in enumerate(config.get("sample", {}).get("layers", [])):
            self.page.layer_table.insertRow(row)
            from PyQt5.QtWidgets import QTableWidgetItem

            thickness = layer.get("thickness_nm", {})
            roughness = layer.get("roughness_nm", {})
            values = (
                "1" if layer.get("enabled", True) else "0",
                layer.get("material", ""),
                thickness.get("minimum", 0) if isinstance(thickness, dict) else thickness,
                thickness.get("maximum", 0) if isinstance(thickness, dict) else thickness,
                roughness.get("minimum", 0) if isinstance(roughness, dict) else roughness,
                roughness.get("maximum", 0) if isinstance(roughness, dict) else roughness,
            )
            for column, value in enumerate(values):
                self.page.layer_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.page.set_model_layers(self.catalog.normalized_layers(config.get("model", {})))
        self.page.update_cache_grid_summary(config)
        self._refresh_impact_options(config)
        self._applying_config = False
