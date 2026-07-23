from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from PyQt5.QtCore import QObject, QProcess, QRunnable, QThreadPool, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QSpinBox,
)

from core.global_params import global_params
from trainset.backends import SlurmBackend, read_metrics
from trainset.config import (
    PHYSICAL_BACKGROUND_PARAMETERS,
    default_project_config,
    load_project_config,
    merge_config,
    save_project_config,
    synchronize_parameter_specs,
    trainable_parameter_names,
    validate_project_config,
)
from trainset.generator import (
    DatasetGenerator,
    apply_preprocessing,
    build_fixed_mask,
    build_random_mask,
    build_roi_shape_mask,
    crop_roi,
    load_scattering_image,
    merge_threshold_mask,
)
from trainset.geometry import roi_to_spherical_ranges
from trainset.job_package import prepare_job_package
from trainset.modeling import build_keras_model, normalized_layers, static_contract
from trainset.plugins import REGISTRY
from trainset.simulation import simulate_pattern
from ui.trainset_build_page import TrainsetBuildPage


class _WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)


class _FunctionWorker(QRunnable):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.with_progress = bool(kwargs.pop("_with_progress", False))
        self.kwargs = kwargs
        self.signals = _WorkerSignals()

    def run(self):
        try:
            if self.with_progress:
                result = self.function(self.signals.progress.emit, *self.args, **self.kwargs)
            else:
                result = self.function(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))


def _deep_get(mapping: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _deep_set(mapping: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target = mapping
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


class TrainsetController(QObject):
    """Configuration-driven Trainset Build controller.

    PyQt owns editing and visualization only. Dataset generation, job packaging,
    local execution and Slurm operations live in the trainset package.
    """

    parameters_changed = pyqtSignal(str, dict)
    generation_started = pyqtSignal()
    generation_finished = pyqtSignal()
    generation_error = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self.parent_controller = parent
        self.window = getattr(parent, "parent", None)
        self.project_root = Path(__file__).resolve().parents[1]
        self.config: Dict[str, Any] = default_project_config()
        self.reference_image: Optional[np.ndarray] = None
        self.package_dir: Optional[Path] = None
        self.local_process: Optional[QProcess] = None
        self._pending_local_arguments = None
        self.thread_pool = QThreadPool.globalInstance()
        self._initialized = False
        self._remote_refresh_running = False
        self._result_sync_started = False
        self._legacy_page_widgets = []
        self._bornagain_preview_cache: Dict[str, np.ndarray] = {}
        self._preview_realization = 0
        self._preview_busy = False
        self._preview_worker: Optional[_FunctionWorker] = None
        self._random_mask_example: Optional[np.ndarray] = None
        self._applying_config = False
        self._what_if_busy = False
        self._what_if_worker: Optional[_FunctionWorker] = None
        self._pending_what_if_values: Optional[Dict[str, float]] = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(900)
        self._autosave_timer.timeout.connect(self._persist_current_config)
        self.monitor_timer = QTimer(self)
        self.monitor_timer.setInterval(15000)
        self.monitor_timer.timeout.connect(self._refresh_job)
        self.page = TrainsetBuildPage()
        self._replace_generated_page()
        self._connect_page()

    def _replace_generated_page(self) -> None:
        host = self.ui.trainsetBuildPage
        layout = host.layout()
        if layout is None:
            from PyQt5.QtWidgets import QVBoxLayout

            layout = QVBoxLayout(host)
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(host)
                self._legacy_page_widgets.append(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.page)
        self.ui.trainsetWorkspace = self.page

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
        page.local_python_button.clicked.connect(self._choose_local_python)
        page.local_prepare_button.clicked.connect(self._prepare_local_job)
        page.local_generate_test_button.clicked.connect(self._run_local_physical_test)
        page.local_generate_button.clicked.connect(self._run_local_generation)
        page.local_train_button.clicked.connect(self._run_local_training)
        page.local_smoke_button.clicked.connect(self._run_local_smoke_test)
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
                signal = getattr(widget, "valueChanged", None) or getattr(widget, "currentTextChanged", None)
                if signal is not None:
                    signal.connect(self._roi_config_changed)
            elif path.startswith("detector.") or path.startswith("beam."):
                signal = getattr(widget, "valueChanged", None) or getattr(widget, "currentTextChanged", None)
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
                widget.toggled.connect(lambda _checked: self._refresh_impact_options(self._collect_config()))
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
        remembered = global_params.get_module_parameters("trainset")
        if not isinstance(remembered, dict) or int(remembered.get("schema_version", 0)) < 2:
            # Some legacy startup paths briefly expose the built-in Trainset
            # dictionary before the shared manager finishes merging its JSON.
            # Read the same user-parameter file as a deterministic fallback.
            remembered_path = Path(str(global_params.user_params_file))
            if not remembered_path.is_absolute():
                remembered_path = self.project_root / remembered_path
            try:
                file_values = json.loads(remembered_path.read_text(encoding="utf-8"))
                candidate = file_values.get("trainset", {})
                if isinstance(candidate, dict) and int(candidate.get("schema_version", 0)) >= 2:
                    remembered = candidate
            except (OSError, ValueError, TypeError):
                pass
        if isinstance(remembered, dict) and int(remembered.get("schema_version", 0)) >= 2:
            self.config = synchronize_parameter_specs(
                merge_config(default_project_config(), remembered)
            )
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
            self.status_updated.emit("Trainset workspace ready · remembered settings restored automatically")
        else:
            self.status_updated.emit("Trainset workspace ready · automatic memory is off")

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
        global_params.set_module_parameters("trainset", copy.deepcopy(config))
        global_params.save_user_parameters()
        self.status_updated.emit("Automatic TrainSet memory disabled · later edits will not replace the remembered settings")

    def _persist_current_config(self) -> None:
        if self._applying_config or not self.page.auto_remember_check.isChecked():
            return
        try:
            config = self._collect_config()
            global_params.set_module_parameters("trainset", copy.deepcopy(config))
            global_params.save_user_parameters()
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
        config.setdefault("runtime", {})["auto_remember"] = self.page.auto_remember_check.isChecked()
        for path, widget in self.page.fields.items():
            if path.startswith("pre.") or path in {"sample.particle_label", "sample.particle_material", "sample.interference_label"}:
                continue
            _deep_set(config, path, self._widget_value(widget))

        particle_label = self.page.fields["sample.particle_label"].currentText()
        particle = next((spec for spec in REGISTRY.list("particle") if spec.label == particle_label), None)
        config["sample"]["particles"] = [{
            "plugin": particle.key if particle else "spherical_segment",
            "material": self.page.fields["sample.particle_material"].currentText(),
            "enabled": True,
            "parameters": self.page.plugin_parameters(self.page.particle_parameter_table),
        }]
        interference_label = self.page.fields["sample.interference_label"].currentText()
        interference = next((spec for spec in REGISTRY.list("interference") if spec.label == interference_label), None)
        config["sample"]["interference"] = {
            "plugin": interference.key if interference else "none",
            "enabled": bool(interference and interference.key != "none"),
            "parameters": self.page.plugin_parameters(self.page.interference_parameter_table),
        }
        layers = []
        for row in range(self.page.layer_table.rowCount()):
            values = [self.page.layer_table.item(row, column).text() if self.page.layer_table.item(row, column) else "" for column in range(6)]
            if values[1]:
                layers.append({
                    "enabled": values[0].strip().lower() not in {"0", "false", "no"},
                    "material": values[1],
                    "thickness_nm": {"minimum": float(values[2] or 0), "maximum": float(values[3] or values[2] or 0)},
                    "roughness_nm": {"minimum": float(values[4] or 0), "maximum": float(values[5] or values[4] or 0)},
                })
        config["sample"]["layers"] = layers
        config["mask"]["fixed_shapes"] = self.page.mask_shapes()

        background_step: Dict[str, Any] = {
            "plugin": "physical_background",
            "enabled": self.page.fields["pre.background.enabled"].isChecked(),
        }
        for definition in PHYSICAL_BACKGROUND_PARAMETERS:
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
            {"plugin": "log", "enabled": self.page.fields["pre.log.enabled"].isChecked(), "epsilon": 1e-6},
            {"plugin": "normalize", "enabled": self.page.fields["pre.normalize.enabled"].isChecked(), "mode": self.page.fields["pre.normalize.mode"].currentText(), "lower": self.page.fields["pre.normalize.lower"].value(), "upper": self.page.fields["pre.normalize.upper"].value()},
            {"plugin": "random_edge_crop", "enabled": self.page.fields["pre.edge.enabled"].isChecked(), "maximum_px": self.page.fields["pre.edge.maximum"].value()},
        ]
        config["model"]["layers"] = self.page.model_layers()
        config = synchronize_parameter_specs(config)
        self.config = config
        self.parameters_changed.emit("Trainset parameters", copy.deepcopy(config))
        return config

    def _apply_config_to_page(self, config: Dict[str, Any]) -> None:
        self._applying_config = True
        config = synchronize_parameter_specs(config)
        self.config = config
        self.page.project_name.setText(str(config.get("project", {}).get("name", "")))
        self.page.auto_remember_check.blockSignals(True)
        self.page.auto_remember_check.setChecked(
            bool(config.get("runtime", {}).get("auto_remember", True))
        )
        self.page.auto_remember_check.blockSignals(False)
        special = {
            "sample.particle_label": next((spec.label for spec in REGISTRY.list("particle") if spec.key == config["sample"]["particles"][0]["plugin"]), "Spherical segment"),
            "sample.particle_material": config["sample"]["particles"][0].get("material", "Copper"),
            "sample.interference_label": next((spec.label for spec in REGISTRY.list("interference") if spec.key == config["sample"]["interference"].get("plugin")), "None"),
        }
        for path, widget in self.page.fields.items():
            if path in special:
                self._set_widget_value(widget, special[path])
            elif not path.startswith("pre."):
                value = _deep_get(config, path)
                if value is not None:
                    self._set_widget_value(widget, value)
        if not self.page.fields["project.workspace"].text().strip():
            self.page.fields["project.workspace"].setText(
                str(self.project_root / "trainset_jobs")
            )
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
        for definition in PHYSICAL_BACKGROUND_PARAMETERS:
            key = str(definition["key"])
            legacy_min = background.get("fraction_min", definition["minimum"]) if key == "target_fraction" else definition["minimum"]
            legacy_max = background.get("fraction_max", definition["maximum"]) if key == "target_fraction" else definition["maximum"]
            pre_map[f"pre.background.{key}.min"] = background.get(f"{key}_min", legacy_min)
            pre_map[f"pre.background.{key}.max"] = background.get(f"{key}_max", legacy_max)
        for path, value in pre_map.items():
            self._set_widget_value(self.page.fields[path], value)
        self._update_threshold_controls()

        self.page.mask_shape_table.setRowCount(0)
        for shape in config.get("mask", {}).get("fixed_shapes", []):
            self.page.add_mask_shape(shape)
        particle_config = config["sample"]["particles"][0]
        particle_plugin = REGISTRY.get("particle", particle_config["plugin"])
        self.page.particle_help.setText(particle_plugin.description)
        self.page.set_plugin_parameters(self.page.particle_parameter_table, particle_plugin.parameters, particle_config.get("parameters", {}))
        interference_config = config["sample"]["interference"]
        interference_plugin = REGISTRY.get("interference", interference_config.get("plugin", "none"))
        self.page.interference_help.setText(interference_plugin.description)
        self.page.set_plugin_parameters(self.page.interference_parameter_table, interference_plugin.parameters, interference_config.get("parameters", {}))
        self.page.segment_constraint_check.setVisible(particle_config.get("plugin") == "spherical_segment")
        is_paracrystal = interference_config.get("plugin") == "paracrystal"
        self.page.spacing_constraint_check.setEnabled(is_paracrystal and "radius_nm" in particle_config.get("parameters", {}))
        self.page.spacing_constraint_check.setVisible(is_paracrystal and "radius_nm" in particle_config.get("parameters", {}))
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
        self.page.set_model_layers(normalized_layers(config.get("model", {})))
        self._refresh_impact_options(config)
        self._applying_config = False

    def _update_threshold_controls(self) -> None:
        enabled = self.page.fields["mask.threshold.enabled"].isChecked()
        automatic = self.page.fields["mask.threshold.auto_reference_upper"].isChecked()
        self.page.fields["mask.threshold.minimum"].setEnabled(enabled)
        self.page.fields["mask.threshold.maximum"].setEnabled(enabled and not automatic)

    def _update_reference_threshold_suggestion(self) -> None:
        self._update_threshold_controls()
        if self.reference_image is None:
            self.page.threshold_summary.setText(
                "Load a reference to calculate detector-gap and hot-pixel locations."
            )
            return
        try:
            roi_image = crop_roi(self.reference_image, self._current_roi())
            if not roi_image.size:
                raise ValueError("The selected ROI is empty.")
            threshold = self.config.get("mask", {}).get("threshold", {})
            if self.page.fields["mask.threshold.auto_reference_upper"].isChecked():
                quantile = float(threshold.get("upper_quantile", 99.999))
                finite = roi_image[np.isfinite(roi_image)]
                if not finite.size:
                    raise ValueError("The ROI has no finite intensity values.")
                upper = float(np.percentile(finite, quantile))
                self._set_widget_value(
                    self.page.fields["mask.threshold.maximum"],
                    upper,
                )
            low = float(self.page.fields["mask.threshold.minimum"].value())
            high = float(self.page.fields["mask.threshold.maximum"].value())
            invalid = ~np.isfinite(roi_image)
            below = np.isfinite(roi_image) & (roi_image < low)
            above = np.isfinite(roi_image) & (roi_image > high)
            total = int(roi_image.size)
            masked = int(np.count_nonzero(invalid | below | above))
            self.page.threshold_summary.setText(
                f"Reference threshold locations: {masked:,}/{total:,} masked "
                f"({masked / max(total, 1):.2%}) · below {low:.5g}: {np.count_nonzero(below):,} · "
                f"above {high:.5g}: {np.count_nonzero(above):,} · non-finite: {np.count_nonzero(invalid):,}"
            )
        except Exception as exc:
            self.page.threshold_summary.setText(f"Reference threshold unavailable: {exc}")

    def _particle_plugin_changed(self, label: str) -> None:
        plugin = next((spec for spec in REGISTRY.list("particle") if spec.label == label), None)
        if plugin is None:
            return
        existing = self.page.plugin_parameters(self.page.particle_parameter_table) if self.page.particle_parameter_table.rowCount() else {}
        self.page.set_plugin_parameters(self.page.particle_parameter_table, plugin.parameters, existing)
        self.page.particle_help.setText(plugin.description)
        is_segment = plugin.key == "spherical_segment"
        self.page.segment_constraint_check.setVisible(is_segment)
        show_spacing = self.page.interference_combo.currentText() == "Paracrystal" and any(
            item["key"] == "radius_nm" for item in plugin.parameters
        )
        self.page.spacing_constraint_check.setEnabled(show_spacing)
        self.page.spacing_constraint_check.setVisible(show_spacing)

    def _interference_plugin_changed(self, label: str) -> None:
        plugin = next((spec for spec in REGISTRY.list("interference") if spec.label == label), None)
        if plugin is None:
            return
        existing = self.page.plugin_parameters(self.page.interference_parameter_table) if self.page.interference_parameter_table.rowCount() else {}
        self.page.set_plugin_parameters(self.page.interference_parameter_table, plugin.parameters, existing)
        self.page.interference_help.setText(plugin.description)
        is_paracrystal = plugin.key == "paracrystal"
        show_spacing = is_paracrystal and self.page.particle_combo.currentText() != "Box"
        self.page.spacing_constraint_check.setEnabled(show_spacing)
        self.page.spacing_constraint_check.setVisible(show_spacing)

    def _select_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Load real 2D scattering file",
            str(Path.home()),
            "Scattering files (*.cbf *.edf *.tif *.tiff *.png *.jpg *.npy *.npz *.h5 *.hdf5 *.nxs);;All files (*)",
        )
        if path:
            self.page.reference_path.setText(path)
            self._load_reference(path)

    def _load_reference_from_field(self) -> None:
        path = self.page.reference_path.text().strip()
        if path:
            self._load_reference(path)

    def _load_reference(self, path: str) -> None:
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[name-defined]
            image = load_scattering_image(path)
            self.reference_image = image
            self.config["project"]["reference_file"] = path
            self.page.reference_path.setText(path)
            if self.page.fields["detector.preset"].currentText() == "Custom":
                self.page.fields["detector.pixels_x"].setValue(int(image.shape[1]))
                self.page.fields["detector.pixels_y"].setValue(int(image.shape[0]))
            self._update_reference_threshold_suggestion()
            for index in range(4):
                self.page.set_design_stage_ready(index, index == 0)
            self._refresh_design_overlay()
            self.page.design_tabs.setCurrentIndex(0)
            self.page.set_step_state(0, "Reference loaded")
            self.page.design_info.setText(f"{Path(path).name}\nShape: {image.shape[1]} × {image.shape[0]} · dtype: {image.dtype}")
            self.status_updated.emit(f"Loaded reference scattering file: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self.window, "Reference load failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _geometry_changed(self, *_args) -> None:
        self._update_geometry_label()
        if self.reference_image is not None:
            self._refresh_design_overlay()

    def _roi_config_changed(self, *_args) -> None:
        self._update_geometry_label()
        if self.reference_image is None:
            return
        self._update_reference_threshold_suggestion()
        self._refresh_design_overlay()
        self.page.set_design_stage_ready(1, True)
        self.page.design_tabs.setCurrentIndex(1)
        self.page.set_step_state(0, "ROI ready")

    def _mask_config_changed(self, *_args) -> None:
        mode = self.page.fields["mask.mode"].currentText()
        self.page.random_mask_panel.setVisible(mode == "random")
        self._random_mask_example = None
        self._update_reference_threshold_suggestion()
        if self.reference_image is None:
            return
        self._refresh_design_overlay()
        self.page.set_design_stage_ready(2, True)
        self.page.set_design_stage_ready(3, True)
        self.page.design_tabs.setCurrentIndex(2)
        self.page.set_step_state(0, "Mask ready")

    def _apply_detector_preset(self, name: str) -> None:
        presets = {
            "PILATUS3 X 2M": (1475, 1679, 0.172, 0.172),
            "EIGER2 X 4M": (2068, 2162, 0.075, 0.075),
        }
        values = presets.get(name)
        if values is None:
            return
        for path, value in zip(
            ("detector.pixels_x", "detector.pixels_y", "detector.pixel_size_x_mm", "detector.pixel_size_y_mm"),
            values,
        ):
            self._set_widget_value(self.page.fields[path], value)
        self._geometry_changed()

    def _update_geometry_label(self) -> None:
        try:
            config = self._collect_config()
            ranges = roi_to_spherical_ranges(config)
            self.page.roi_range_label.setText(
                f"BornAgain detector: φ {ranges['phi_min_deg']:.4f}° … {ranges['phi_max_deg']:.4f}° · "
                f"α {ranges['alpha_min_deg']:.4f}° … {ranges['alpha_max_deg']:.4f}°"
            )
        except Exception as exc:
            self.page.roi_range_label.setText(f"Geometry incomplete: {exc}")

    def _begin_roi(self, mode: str = "roi") -> None:
        if self.reference_image is None:
            QMessageBox.information(self.window, "ROI", "Load a real scattering file first.")
            return
        beam_center = (
            self.page.fields["detector.beam_center_x_px"].value(),
            self.page.fields["detector.beam_center_y_px"].value(),
        )
        self.page.full_detector_canvas.set_data(self.reference_image, roi=self._current_roi(), beam_center=beam_center)
        self.page.full_detector_canvas.set_draw_mode(mode)
        self.page.design_tabs.setCurrentIndex(0)
        self.status_updated.emit("Draw the rectangular ROI on the detector image")

    def _begin_beam_center(self) -> None:
        if self.reference_image is None:
            QMessageBox.information(self.window, "Beam center", "Load a real scattering file first.")
            return
        self.page.full_detector_canvas.set_draw_mode("beam_center")
        self.page.design_tabs.setCurrentIndex(0)
        self.status_updated.emit("Click the direct-beam position on the full detector")

    def _begin_mask(self, mode: str) -> None:
        if self.reference_image is None:
            QMessageBox.information(self.window, "Mask", "Load a real scattering file first.")
            return
        try:
            roi_image = crop_roi(self.reference_image, self._current_roi())
            self.page.roi_design_canvas.set_data(roi_image)
            self.page.roi_design_canvas.set_draw_mode(mode)
            self.page.design_tabs.setCurrentIndex(1)
            self.status_updated.emit(f"Draw a {mode} fixed mask in ROI coordinates")
        except Exception as exc:
            QMessageBox.warning(self.window, "Mask", str(exc))

    def _region_created(self, mode: str, payload: Dict[str, Any]) -> None:
        if mode == "beam_center":
            self._set_widget_value(self.page.fields["detector.beam_center_x_px"], payload["x"])
            self._set_widget_value(self.page.fields["detector.beam_center_y_px"], payload["y"])
            self._update_geometry_label()
            self.page.set_step_state(0, "Beam center selected")
            self.status_updated.emit(f"Beam center selected at x={payload['x']:.1f}, y={payload['y']:.1f} px")
        elif mode == "roi":
            for key in ("x", "y", "width", "height"):
                self._set_widget_value(self.page.fields[f"roi.{key}"], int(payload[key]))
            self.page.full_detector_canvas.set_draw_mode("")
            self._update_geometry_label()
            self.page.set_design_stage_ready(1, True)
            self.page.design_tabs.setCurrentIndex(1)
            self.page.set_step_state(0, "ROI ready")
        else:
            self.page.add_mask_shape(payload)
            self.page.roi_design_canvas.set_draw_mode("")
            self.page.set_design_stage_ready(2, True)
            self.page.set_design_stage_ready(3, True)
            self.page.design_tabs.setCurrentIndex(2)
            self.page.set_step_state(0, "Mask ready")
        self._refresh_design_overlay()

    def _clear_masks(self) -> None:
        self.page.mask_shape_table.setRowCount(0)
        self._refresh_design_overlay()
        self.page.set_design_stage_ready(2, True)
        self.page.set_design_stage_ready(3, True)
        self.page.design_tabs.setCurrentIndex(2)
        self.page.set_step_state(0, "Mask updated")

    def _remove_selected_masks(self) -> None:
        if not self.page.remove_selected_mask_shapes():
            self.status_updated.emit("Select one or more mask rows to remove")
            return
        self._refresh_design_overlay()
        self.page.set_design_stage_ready(2, True)
        self.page.set_design_stage_ready(3, True)
        self.page.design_tabs.setCurrentIndex(2)
        self.page.set_step_state(0, "Mask updated")
        self.status_updated.emit("Removed selected mask regions")

    def _current_roi(self) -> Dict[str, int]:
        return {key: int(self.page.fields[f"roi.{key}"].value()) for key in ("x", "y", "width", "height")}

    def _refresh_design_overlay(self, *_args) -> None:
        if self.reference_image is None:
            return
        try:
            config = self._collect_config()
            roi = self._current_roi()
            roi_image = crop_roi(self.reference_image, roi)
            if config.get("mask", {}).get("mode") == "random":
                if self._random_mask_example is None or self._random_mask_example.shape != roi_image.shape:
                    self._random_mask_example = build_random_mask(roi_image.shape, config, np.random.default_rng())
                mask = merge_threshold_mask(roi_image, self._random_mask_example, config)
                mask_label = "random example + threshold (preview only)"
            else:
                mask = build_fixed_mask(roi_image, config)
                mask_label = "fixed"
            roi_shape_mask = build_roi_shape_mask(roi_image.shape, config)
            self.page.full_detector_canvas.set_data(
                self.reference_image,
                roi=roi,
                beam_center=(
                    self.page.fields["detector.beam_center_x_px"].value(),
                    self.page.fields["detector.beam_center_y_px"].value(),
                ),
            )
            self.page.roi_design_canvas.set_data(roi_image, mask=roi_shape_mask if roi_shape_mask.any() else None)
            self.page.masked_design_canvas.set_data(roi_image, mask=mask)
            self.page.mask_only_canvas.set_data(mask.astype(np.float32), binary=True)
            self.page.design_info.setText(
                f"Reference: {self.reference_image.shape[1]} × {self.reference_image.shape[0]}\n"
                f"ROI tensor: {roi_image.shape[1]} × {roi_image.shape[0]} · {mask_label} masked: {mask.mean():.2%}\n"
                "Use Draw ROI for detector coordinates; mask shapes are edited in ROI coordinates."
            )
            self.page.design_info.setToolTip(f"Mask mode: {mask_label}; masked fraction: {mask.mean():.2%}")
        except Exception as exc:
            self.page.design_info.setText(str(exc))

    def _new_random_mask_example(self) -> None:
        try:
            config = self._collect_config()
            roi = self._current_roi()
            shape = (int(roi["height"]), int(roi["width"]))
            self._random_mask_example = build_random_mask(shape, config, np.random.default_rng())
            self.page.random_mask_panel.setVisible(True)
            if self.reference_image is not None:
                self._refresh_design_overlay()
            else:
                self.page.mask_only_canvas.set_data(self._random_mask_example.astype(np.float32), binary=True)
                self.page.design_tabs.setCurrentIndex(3)
                self.page.design_info.setText(
                    f"Random mask example: {shape[1]} × {shape[0]} · masked {self._random_mask_example.mean():.2%}. "
                    "Load an experimental image only if you want to overlay it."
                )
            self.status_updated.emit("Generated a fresh unseeded random-mask example")
        except Exception as exc:
            QMessageBox.warning(self.window, "Random mask", str(exc))

    def _refresh_impact_options(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or self.config
        previous = self.page.impact_parameter_combo.currentData()
        self.page.impact_parameter_combo.blockSignals(True)
        self.page.impact_parameter_combo.clear()
        for key, spec in config.get("parameters", {}).items():
            if float(spec.get("maximum", 0.0)) > float(spec.get("minimum", 0.0)):
                self.page.impact_parameter_combo.addItem(f"Physics · {key}", f"physics|{key}")
        steps = {str(step.get("plugin")): step for step in config.get("preprocessing", {}).get("steps", [])}
        if steps.get("physical_background", {}).get("enabled", False):
            for definition in PHYSICAL_BACKGROUND_PARAMETERS:
                self.page.impact_parameter_combo.addItem(
                    f"Background · {definition['label']}",
                    f"physical_background|{definition['key']}",
                )
        if steps.get("gaussian_noise", steps.get("noise", {})).get("enabled", False):
            self.page.impact_parameter_combo.addItem("Gaussian noise · SNR (dB)", "gaussian_noise|snr_db")
        if steps.get("poisson_noise", {}).get("enabled", False):
            self.page.impact_parameter_combo.addItem("Poisson noise · photon-count scale", "poisson_noise|count_scale")
        if previous is not None:
            index = self.page.impact_parameter_combo.findData(previous)
            if index >= 0:
                self.page.impact_parameter_combo.setCurrentIndex(index)
        self.page.impact_parameter_combo.blockSignals(False)

    def _impact_range(self, config: Dict[str, Any]) -> tuple[str, str, float, float]:
        data = str(self.page.impact_parameter_combo.currentData() or "")
        plugin, _, key = data.partition("|")
        if plugin == "physics":
            spec = config.get("parameters", {}).get(key, {})
            return plugin, key, float(spec.get("minimum", 0.0)), float(spec.get("maximum", 0.0))
        steps = {str(step.get("plugin")): step for step in config.get("preprocessing", {}).get("steps", [])}
        if plugin == "physical_background":
            step = steps.get(plugin, {})
            definition = next(item for item in PHYSICAL_BACKGROUND_PARAMETERS if item["key"] == key)
            return plugin, key, float(step.get(f"{key}_min", definition["minimum"])), float(
                step.get(f"{key}_max", definition["maximum"])
            )
        if plugin == "gaussian_noise":
            step = steps.get(plugin, steps.get("noise", {}))
            return plugin, key, float(step.get("snr_min_db", 80.0)), float(step.get("snr_max_db", 110.0))
        step = steps.get("poisson_noise", {})
        return "poisson_noise", "count_scale", float(step.get("count_scale_min", 1.0)), float(
            step.get("count_scale_max", 20.0)
        )

    def _cached_simulation(self, config: Dict[str, Any], sampled: Dict[str, Any]) -> tuple[np.ndarray, bool]:
        payload = {
            "beam": config.get("beam"),
            "detector": config.get("detector"),
            "roi": config.get("roi"),
            "simulation": config.get("simulation"),
            "sample": config.get("sample"),
            "sampled": sampled,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if key in self._bornagain_preview_cache:
            return self._bornagain_preview_cache[key].copy(), True
        image = simulate_pattern(config, sampled)
        if len(self._bornagain_preview_cache) >= 24:
            self._bornagain_preview_cache.pop(next(iter(self._bornagain_preview_cache)))
        self._bornagain_preview_cache[key] = np.asarray(image, dtype=np.float32).copy()
        return np.asarray(image, dtype=np.float32), False

    def _force_generate_preview(self) -> None:
        self._start_preview(force=True)

    def _new_preview_realization(self) -> None:
        if self._preview_busy:
            self.status_updated.emit("A simulated comparison is already running")
            return
        self._preview_realization += 1
        self._start_preview(force=False)

    def _generate_preview(self) -> None:
        self._start_preview(force=False)

    def _start_preview(self, force: bool = False) -> None:
        if self._preview_busy:
            self.status_updated.emit("A simulated comparison is already running")
            return
        config = self._collect_config()
        self._refresh_impact_options(config)
        valid, errors, warnings = validate_project_config(config, require_reference=False)
        if not valid:
            QMessageBox.warning(self.window, "Preview blocked", "\n".join(errors))
            return
        generator = DatasetGenerator(config)
        if not generator.bornagain_available:
            QMessageBox.warning(
                self.window,
                "Preview failed",
                "BornAgain is required because Local Preview displays simulated training images, not the experimental reference.",
            )
            return
        plugin, key, minimum, maximum = self._impact_range(config)
        compared_text = self.page.impact_parameter_combo.currentText()
        self._preview_busy = True
        self.page.set_preview_busy(True, 2, "Preparing the simulated comparison…")
        self.progress_updated.emit(2)
        worker = _FunctionWorker(
            self._compute_preview,
            copy.deepcopy(config),
            plugin,
            key,
            minimum,
            maximum,
            compared_text,
            self.page.preview_count.value(),
            self._preview_realization,
            warnings,
            force,
            _with_progress=True,
        )
        worker.signals.progress.connect(self._preview_progressed)
        worker.signals.finished.connect(self._preview_finished)
        worker.signals.error.connect(self._preview_failed)
        self._preview_worker = worker
        self.thread_pool.start(worker)

    def _compute_preview(
        self,
        progress,
        config: Dict[str, Any],
        plugin: str,
        key: str,
        minimum: float,
        maximum: float,
        compared_text: str,
        preview_count: int,
        realization: int,
        warnings,
        force: bool,
    ) -> Dict[str, Any]:
        if force:
            self._bornagain_preview_cache.clear()
        started = time.perf_counter()
        generator = DatasetGenerator(config)
        midpoint = 0.5 * (minimum + maximum)
        comparison_values = (("minimum", minimum), ("midpoint", midpoint), ("maximum", maximum))
        base_sample = {
            name: 0.5 * (float(spec.get("minimum", 0.0)) + float(spec.get("maximum", 0.0)))
            for name, spec in config.get("parameters", {}).items()
        }
        simulation_seed = int(config.get("project", {}).get("seed", 42))
        realization_seed = simulation_seed + 1009 * realization
        comparison_images: Dict[str, np.ndarray] = {}
        comparison_labels: Dict[str, str] = {}
        comparison_details: Dict[str, Any] = {}
        midpoint_stages = []
        cache_hits = 0
        cache_misses = 0
        ranges = roi_to_spherical_ranges(config)
        for index, (position, value) in enumerate(comparison_values):
            progress(10 + index * 23, f"BornAgain simulation {index + 1}/3: {position} {key} = {value:.5g}")
            sampled = dict(base_sample)
            overrides: Dict[str, float] = {}
            if plugin == "physics":
                sampled[key] = value
            else:
                overrides[f"{plugin}.{key}"] = value
            mixture_generator = DatasetGenerator(config)
            mixture_generator.rng = np.random.default_rng(simulation_seed)
            simulation_values = mixture_generator._mixture_values(sampled)
            raw, cache_hit = self._cached_simulation(config, simulation_values)
            cache_hits += int(cache_hit)
            cache_misses += int(not cache_hit)
            progress(20 + index * 23, f"Applying enabled preprocessing to {position} image…")
            realization_rng = np.random.default_rng(realization_seed + 17)
            if config.get("mask", {}).get("mode") == "random":
                mask = build_random_mask(raw.shape, config, realization_rng)
                mask = merge_threshold_mask(raw, mask, config)
            else:
                mask = build_fixed_mask(raw, config)
            preprocessing_trace: Dict[str, Any] = {}
            stages = apply_preprocessing(
                raw,
                config,
                mask,
                realization_rng,
                overrides=overrides,
                trace=preprocessing_trace,
            )
            final_image = np.asarray(stages[-1]["image"], dtype=np.float32)
            comparison_images[position] = final_image
            comparison_labels[position] = f"{position.title()} · {value:.5g}"
            comparison_details[position] = {
                "comparison": {"parameter": compared_text, "value": float(value)},
                "editable physics": copy.deepcopy(sampled),
                "physics values": simulation_values,
                "preprocessing realization": preprocessing_trace or "none enabled",
                "beam": config.get("beam", {}),
                "detector": config.get("detector", {}),
                "roi": config.get("roi", {}),
                "angular range": {
                    "phi min deg": ranges["phi_min_deg"],
                    "phi max deg": ranges["phi_max_deg"],
                    "alpha top deg": ranges["alpha_top_deg"],
                    "alpha bottom deg": ranges["alpha_bottom_deg"],
                },
            }
            if position == "midpoint":
                midpoint_stages = stages
        progress(82, "Sampling label coverage and calculating diagnostics…")
        parameter_samples = generator.sample_parameters(preview_count)
        total_samples = int(config["dataset"]["number_of_samples"])
        final_image = comparison_images["midpoint"]
        bytes_per_sample = final_image.nbytes + final_image.size + 4 * len(config.get("parameters", {}))
        valid_values = final_image[np.isfinite(final_image)]
        histogram, edges = (
            np.histogram(valid_values, bins=64)
            if valid_values.size
            else (np.zeros(64, dtype=float), np.arange(65, dtype=float))
        )
        elapsed = time.perf_counter() - started
        stats = {
            "source": "BornAgain simulation (experimental reference is geometry guidance only)",
            "orientation": "x right, y down, qz/exit angle higher at image top",
            "compared_parameter": compared_text,
            "range": f"{minimum:.6g} / {midpoint:.6g} / {maximum:.6g}",
            "tensor_shape": [1, int(final_image.shape[0]), int(final_image.shape[1]), 1],
            "enabled_pipeline": " → ".join(str(stage["name"]) for stage in midpoint_stages),
            "bornagain_cache": f"{cache_hits} hit(s), {cache_misses} recomputed",
            "stochastic_realization": realization + 1,
            "estimated_dataset_gib": round(total_samples * bytes_per_sample / (1024**3), 3),
            "preview_elapsed_s": round(elapsed, 3),
        }
        if warnings:
            stats["warning"] = " · ".join(warnings)
        progress(96, "Rendering the comparison in the GUI…")
        return {
            "comparison_images": comparison_images,
            "comparison_labels": comparison_labels,
            "comparison_details": comparison_details,
            "stages": midpoint_stages,
            "stats": stats,
            "spectrum_x": (edges[:-1] + edges[1:]) / 2.0,
            "spectrum_y": histogram,
            "parameter_samples": parameter_samples,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        }

    def _preview_progressed(self, progress: int, message: str) -> None:
        self.page.set_preview_progress(progress, message)
        self.progress_updated.emit(progress)
        self.status_updated.emit(message)

    def _preview_finished(self, result: Dict[str, Any]) -> None:
        self._preview_busy = False
        self._preview_worker = None
        self.page.set_simulation_preview(
            result["comparison_images"],
            result["comparison_labels"],
            result["stages"],
            result["stats"],
            result["spectrum_x"],
            result["spectrum_y"],
        )
        self.page.set_comparison_details(
            result["comparison_details"],
            self.config.get("parameters", {}),
        )
        cache_hits = int(result["cache_hits"])
        cache_misses = int(result["cache_misses"])
        self.page.preview_cache_status.setText(
            f"BornAgain cache: {len(self._bornagain_preview_cache)} image(s) · last update {cache_hits} hit / {cache_misses} rerun"
        )
        particle = next(iter(self.config.get("sample", {}).get("particles", [])), {})
        form_factor_names = list(particle.get("parameters", {}))
        self.page.set_parameter_samples(
            result["parameter_samples"],
            form_factor_names,
            self.config.get("parameters", {}),
        )
        self.page.preview_gate_table.item(0, 1).setText("Ready")
        self.page.preview_gate_table.item(1, 1).setText("Ready")
        self.page.preview_gate_table.item(2, 1).setText("Ready")
        self._storage_acceptance_changed(self.page.storage_accept_check.isChecked())
        self.page.validation_badge.setText("Preview ready")
        self.page.set_step_state(1, "Preview ready")
        self.page.set_preview_busy(False, 100, "Preview ready. The GUI remained responsive during simulation.")
        self.progress_updated.emit(100)
        self.status_updated.emit("BornAgain simulation impact preview generated")

    def _preview_failed(self, message: str) -> None:
        self._preview_busy = False
        self._preview_worker = None
        self.page.set_preview_busy(False, 0, f"Preview failed: {message}")
        QMessageBox.warning(self.window, "Preview failed", message)
        self.generation_error.emit(message)

    def _start_what_if(self, values: Dict[str, float]) -> None:
        numeric = {str(key): float(value) for key, value in values.items()}
        if self._what_if_busy:
            self._pending_what_if_values = numeric
            self.page.set_what_if_busy(
                True,
                "Current simulation is finishing · the latest edit is queued.",
            )
            return
        config = self._collect_config()
        self._what_if_busy = True
        self._pending_what_if_values = None
        self.page.set_what_if_busy(True, "Running editable What-if simulation in the background…")
        worker = _FunctionWorker(
            self._compute_what_if,
            copy.deepcopy(config),
            numeric,
            self._preview_realization,
        )
        worker.signals.finished.connect(self._what_if_finished)
        worker.signals.error.connect(self._what_if_failed)
        self._what_if_worker = worker
        self.thread_pool.start(worker)

    def _compute_what_if(
        self,
        config: Dict[str, Any],
        sampled: Dict[str, float],
        realization: int,
    ) -> Dict[str, Any]:
        constraints = config.get("sample", {}).get("constraints", {})
        if (
            constraints.get("segment_height_le_2r", False)
            and "height_nm" in sampled
            and "radius_nm" in sampled
            and sampled["height_nm"] > 2.0 * sampled["radius_nm"]
        ):
            raise ValueError("What-if violates h ≤ 2R. Adjust height/radius or disable the physical constraint.")
        if (
            constraints.get("interparticle_spacing_gt_2r", False)
            and "D_nm" in sampled
            and "radius_nm" in sampled
            and sampled["D_nm"] <= 2.0 * sampled["radius_nm"]
        ):
            raise ValueError("What-if violates D > 2R. Adjust spacing/radius or disable the physical constraint.")

        seed = int(config.get("project", {}).get("seed", 42))
        generator = DatasetGenerator(config)
        generator.rng = np.random.default_rng(seed)
        simulation_values = generator._mixture_values(sampled)
        raw, cache_hit = self._cached_simulation(config, simulation_values)
        realization_rng = np.random.default_rng(seed + 1009 * realization + 17)
        if config.get("mask", {}).get("mode") == "random":
            mask = build_random_mask(raw.shape, config, realization_rng)
            mask = merge_threshold_mask(raw, mask, config)
        else:
            mask = build_fixed_mask(raw, config)
        trace: Dict[str, Any] = {}
        stages = apply_preprocessing(
            raw,
            config,
            mask,
            realization_rng,
            trace=trace,
        )
        image = np.asarray(stages[-1]["image"], dtype=np.float32)
        return {
            "image": image,
            "cache_hit": cache_hit,
            "values": sampled,
            "pipeline": " → ".join(str(stage["name"]) for stage in stages),
        }

    def _what_if_finished(self, result: Dict[str, Any]) -> None:
        self._what_if_busy = False
        self._what_if_worker = None
        values = ", ".join(f"{key}={value:.5g}" for key, value in result["values"].items())
        cache_text = "BornAgain cache reused" if result["cache_hit"] else "BornAgain recomputed"
        self.page.set_what_if_result(
            result["image"],
            f"{cache_text} · {values}\nPipeline: {result['pipeline']}",
        )
        self._run_pending_what_if()

    def _what_if_failed(self, message: str) -> None:
        self._what_if_busy = False
        self._what_if_worker = None
        self.page.set_what_if_busy(False, f"What-if not simulated: {message}")
        self._run_pending_what_if()

    def _run_pending_what_if(self) -> None:
        pending = self._pending_what_if_values
        self._pending_what_if_values = None
        if pending is not None:
            QTimer.singleShot(0, lambda values=pending: self._start_what_if(values))

    def _validate_and_report(self) -> bool:
        config = self._collect_config()
        valid, errors, warnings = validate_project_config(config)
        if valid:
            self.page.validation_badge.setText("Configuration valid")
            self.page.preview_gate_table.item(0, 1).setText("Ready")
            text = "Configuration is valid."
            if warnings:
                text += "\n\nWarnings:\n" + "\n".join(f"• {item}" for item in warnings)
            QMessageBox.information(self.window, "Validation", text)
        else:
            self.page.validation_badge.setText("Validation failed")
            QMessageBox.warning(self.window, "Validation", "\n".join(f"• {item}" for item in errors))
        return valid

    def _validate_model_contract(self) -> None:
        try:
            config = self._collect_config()
            height, width = int(config["roi"]["height"]), int(config["roi"]["width"])
            outputs = len(trainable_parameter_names(config))
            if outputs < 1:
                raise ValueError("At least one physics parameter needs a non-zero range.")
            layers = normalized_layers(config.get("model", {}))
            static_summary = static_contract((height, width, 1), outputs, layers)
            try:
                import tensorflow as tf

                model = build_keras_model(tf, (height, width, 1), outputs, config["model"])
                result = model(np.zeros((1, height, width, 1), dtype=np.float32), training=False)
                summary = f"Forward pass OK\n\n{static_summary}\n\nBatch output: {tuple(result.shape)}\nTrainable weights: {model.count_params():,}"
            except Exception as exc:
                summary = f"Static tensor contract\n\n{static_summary}\n\nTensorFlow forward pass unavailable: {exc}"
            self.page.model_summary.setPlainText(summary)
            self.page.preview_gate_table.item(2, 1).setText("Ready")
            self.page.set_step_state(2, "Contract ready")
        except Exception as exc:
            QMessageBox.warning(self.window, "Model validation", str(exc))

    def _save_project_dialog(self) -> None:
        config = self._collect_config()
        default = self.project_root / f"{config['project']['name']}.yaml"
        path, _ = QFileDialog.getSaveFileName(self.window, "Save trainset project", str(default), "YAML (*.yaml *.yml);;JSON (*.json)")
        if path:
            save_project_config(config, path)
            self.status_updated.emit(f"Saved trainset project: {path}")

    def _load_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Load trainset project",
            str(self.project_root),
            "Project configuration (*.yaml *.yml *.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.set_parameters(load_project_config(path))
            self.status_updated.emit(f"Loaded trainset project: {path}")
        except Exception as exc:
            QMessageBox.critical(self.window, "Project load failed", str(exc))

    def _choose_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(self.window, "Choose local trainset workspace", str(self.project_root))
        if path:
            self.page.fields["project.workspace"].setText(path)

    def _choose_local_python(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self.window,
            "Choose local Python executable",
            str(Path(sys.executable).parent),
            "Python executable (python.exe python);;All files (*)",
        )
        if selected:
            self.page.fields["training.local_python"].setText(selected)

    def _workspace(self) -> Path:
        configured = self.page.fields["project.workspace"].text().strip()
        return Path(configured) if configured else self.project_root / "trainset_jobs"

    def _prepare_local_job(self) -> None:
        self._prepare_job(local=True)

    def _prepare_hpc_job(self) -> None:
        self._prepare_job(local=False)

    def _prepare_job(self, local: bool) -> None:
        config = self._collect_config()
        config["training"]["backend"] = "local" if local else "slurm"
        valid, errors, _warnings = validate_project_config(config)
        if not valid:
            QMessageBox.warning(self.window, "Job package blocked", "\n".join(errors))
            return
        try:
            self.package_dir = prepare_job_package(config, self._workspace(), self.project_root)
            config["runtime"]["last_project_dir"] = str(self.package_dir)
            self.page.package_tree.setPlainText(
                f"Prepared: {self.package_dir}\n\n"
                "config.yaml\nmanifest.json\ngenerate_dataset.py\ntrain.py\nvalidate_config.py\n"
                "environment.yml\nslurm_generate.sh\nslurm_train.sh\nsrc/trainset/\nsrc/calibration/\ndataset/\nresults/\nlogs/"
            )
            self.page.set_step_state(3, "Package ready")
            self.status_updated.emit("Reproducible local/HPC job package prepared")
        except Exception as exc:
            QMessageBox.critical(self.window, "Prepare job failed", str(exc))

    def _ensure_package(self) -> bool:
        if self.package_dir and self.package_dir.exists():
            return True
        self._prepare_local_job()
        return bool(self.package_dir and self.package_dir.exists())

    def _start_local_process(self, arguments, follow_up=None) -> None:
        if not self._ensure_package():
            return
        if self.local_process and self.local_process.state() != QProcess.NotRunning:
            QMessageBox.information(self.window, "Local backend", "A local generation/training process is already running.")
            return
        process = QProcess(self)
        process.setWorkingDirectory(str(self.package_dir))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self.page.job_log.append(bytes(process.readAllStandardOutput()).decode(errors="replace").rstrip()))
        process.started.connect(
            lambda: (
                self.generation_started.emit(),
                self.page.job_state.setText("RUNNING"),
                self.page.set_step_state(4, "RUNNING"),
            )
        )
        process.finished.connect(self._local_process_finished)
        process.errorOccurred.connect(lambda _error: self.generation_error.emit(process.errorString()))
        self.local_process = process
        self._pending_local_arguments = follow_up
        python_executable = self.page.fields["training.local_python"].text().strip() or sys.executable
        process.start(python_executable, arguments)
        self.page.step_list.setCurrentRow(4)

    def _run_local_physical_test(self) -> None:
        """Generate a small, genuinely physical BornAgain dataset."""
        samples = int(self._collect_config().get("training", {}).get("smoke_samples", 64))
        self._start_local_process(["generate_dataset.py", "--samples", str(samples), "--mode", "full"])

    def _run_local_generation(self) -> None:
        count = int(self._collect_config()["dataset"]["number_of_samples"])
        self._start_local_process(["generate_dataset.py", "--samples", str(count), "--mode", "full"])

    def _run_local_training(self) -> None:
        self._start_local_process(["train.py"])

    def _run_local_smoke_test(self) -> None:
        config = self._collect_config()
        reference = str(config.get("project", {}).get("reference_file", ""))
        if not reference or not Path(reference).exists():
            QMessageBox.information(self.window, "Local smoke test", "Load a real reference image first. It is used only to test the local data/model pipeline.")
            return
        self._prepare_local_job()
        if not self.package_dir:
            return
        samples = int(config.get("training", {}).get("smoke_samples", 64))
        epochs = int(config.get("training", {}).get("smoke_epochs", 2))
        self.page.job_log.clear()
        self.page.job_log.append("LIGHTWEIGHT DEMO: reference-derived images test I/O and training only; they are not a physical BornAgain dataset.")
        self._start_local_process(
            ["generate_dataset.py", "--samples", str(samples), "--mode", "demo"],
            follow_up=["train.py", "--smoke", "--epochs", str(epochs)],
        )

    def _local_process_finished(self, exit_code: int, _status) -> None:
        state = "COMPLETED" if exit_code == 0 else "FAILED"
        self.page.job_state.setText(state)
        self.page.set_step_state(4, state)
        if exit_code == 0:
            self.generation_finished.emit()
            pending = self._pending_local_arguments
            self._pending_local_arguments = None
            if pending:
                self.page.job_log.append("Starting lightweight training…")
                QTimer.singleShot(0, lambda: self._start_local_process(pending))
                return
        else:
            self._pending_local_arguments = None
            self.generation_error.emit(f"Local process exited with code {exit_code}")
        self._load_local_metrics()

    def _load_local_metrics(self) -> None:
        if not self.package_dir:
            return
        records = read_metrics(self.package_dir / "results" / "metrics.jsonl")
        self.page.metrics_table.setRowCount(0)
        from PyQt5.QtWidgets import QTableWidgetItem

        for row, record in enumerate(records):
            self.page.metrics_table.insertRow(row)
            values = (record.get("epoch", ""), record.get("loss", record.get("train_loss", "")), record.get("val_loss", ""), record.get("lr", ""))
            for column, value in enumerate(values):
                self.page.metrics_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _slurm_backend(self) -> SlurmBackend:
        config = self._collect_config()
        if not config["hpc"]["user"] or not config["hpc"]["remote_path"]:
            raise ValueError("Configure Maxwell user and remote project path first.")
        return SlurmBackend(config)

    def _run_worker(self, function, success, title: str, on_error=None) -> None:
        worker = _FunctionWorker(function)
        worker.signals.finished.connect(success)
        def report_error(message: str) -> None:
            if on_error is not None:
                on_error()
            QMessageBox.critical(self.window, title, message)
        worker.signals.error.connect(report_error)
        self.thread_pool.start(worker)

    def _storage_acceptance_changed(self, accepted: bool) -> None:
        self.page.preview_gate_table.item(3, 1).setText("Ready" if accepted else "Pending")

    def _missing_submission_gates(self):
        missing = []
        for row in range(self.page.preview_gate_table.rowCount()):
            gate = self.page.preview_gate_table.item(row, 0).text()
            state = self.page.preview_gate_table.item(row, 1).text()
            if state != "Ready":
                missing.append(gate)
        return missing

    def _test_connection(self) -> None:
        try:
            backend = self._slurm_backend()
            self.page.connection_button.setEnabled(False)
            self._run_worker(
                backend.connection_check,
                lambda result: (self.page.connection_button.setEnabled(True), self.page.preview_gate_table.item(3, 1).setText("Ready"), QMessageBox.information(self.window, "Maxwell", f"Connection successful: {result}")),
                "Maxwell connection failed",
                on_error=lambda: self.page.connection_button.setEnabled(True),
            )
        except Exception as exc:
            QMessageBox.warning(self.window, "Maxwell", str(exc))

    def _submit_maxwell(self) -> None:
        missing = self._missing_submission_gates()
        if missing:
            self.page.step_list.setCurrentRow(1)
            QMessageBox.warning(
                self.window,
                "Submission checks incomplete",
                "Complete these checks before submitting:\n\n" + "\n".join(f"• {item}" for item in missing),
            )
            return
        if not self.package_dir or not self.package_dir.exists():
            self._prepare_hpc_job()
        if not self.package_dir:
            return
        reply = QMessageBox.question(self.window, "Submit to Maxwell", f"Upload and submit this job package?\n\n{self.package_dir}", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            backend = self._slurm_backend()

            def upload_submit():
                backend.upload(self.package_dir)
                return backend.submit()

            self.page.hpc_submit_button.setEnabled(False)
            self._run_worker(
                upload_submit,
                self._submission_finished,
                "Maxwell submission failed",
                on_error=lambda: self.page.hpc_submit_button.setEnabled(True),
            )
        except Exception as exc:
            QMessageBox.warning(self.window, "Maxwell", str(exc))

    def _submission_finished(self, jobs: Dict[str, str]) -> None:
        self.page.hpc_submit_button.setEnabled(True)
        job_id = jobs["train_job_id"]
        self.config["runtime"]["last_job_id"] = job_id
        self.page.job_id_label.setText(f"Generate: {jobs['generate_job_id']} · Train: {job_id}")
        self.page.job_state.setText("SUBMITTED")
        self.page.set_step_state(3, "Submitted")
        self.page.set_step_state(4, f"Job {job_id}")
        self.page.step_list.setCurrentRow(4)
        self._result_sync_started = False
        self.monitor_timer.start()
        self.status_updated.emit(f"Submitted Maxwell jobs: {jobs}")

    def _refresh_job(self) -> None:
        if self._remote_refresh_running:
            return
        job_id = str(self.config.get("runtime", {}).get("last_job_id", ""))
        if not job_id:
            self._load_local_metrics()
            return
        try:
            backend = self._slurm_backend()
            self._remote_refresh_running = True

            def query():
                return backend.query(job_id), backend.tail(job_id)

            self._run_worker(
                query,
                self._job_refreshed,
                "Job refresh failed",
                on_error=lambda: setattr(self, "_remote_refresh_running", False),
            )
        except Exception as exc:
            self._remote_refresh_running = False
            QMessageBox.warning(self.window, "Job refresh", str(exc))

    def _job_refreshed(self, payload) -> None:
        self._remote_refresh_running = False
        status, log = payload
        self.page.job_state.setText(status.state)
        self.page.set_step_state(4, status.state)
        self.page.job_id_label.setText(f"Job ID: {status.job_id} · Elapsed: {status.elapsed} · MaxRSS: {status.max_rss}")
        self.page.job_log.setPlainText(log or status.raw)
        normalized_state = status.state.upper().split("+", 1)[0].split()[0] if status.state else "UNKNOWN"
        terminal = normalized_state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}
        if terminal:
            self.monitor_timer.stop()
        if normalized_state == "COMPLETED" and not self._result_sync_started:
            self._result_sync_started = True
            self._sync_results()

    def _sync_results(self) -> None:
        try:
            backend = self._slurm_backend()
            destination = self._workspace() / self.config["project"]["name"] / "results"
            self._run_worker(lambda: backend.download_results(destination), lambda _result: (self._load_local_metrics(), QMessageBox.information(self.window, "Results", f"Results synchronized to:\n{destination}")), "Result synchronization failed")
        except Exception as exc:
            QMessageBox.warning(self.window, "Results", str(exc))

    def _register_best_model(self) -> None:
        config = self._collect_config()
        roots = []
        if self.package_dir:
            roots.append(self.package_dir / "results")
        last_project = str(config.get("runtime", {}).get("last_project_dir", "")).strip()
        if last_project:
            roots.append(Path(last_project) / "results")
        roots.append(self._workspace() / config["project"]["name"] / "results")

        model_path = None
        for root in roots:
            for name in ("best_model.keras", "best_model.h5", "best_model.pt", "best_model.pth"):
                candidate = root / name
                if candidate.exists():
                    model_path = candidate
                    break
            if model_path is not None:
                break
        if model_path is None:
            selected, _ = QFileDialog.getOpenFileName(
                self.window,
                "Select trained model",
                str(roots[0] if roots else self.project_root),
                "Models (*.keras *.h5 *.pt *.pth);;All files (*)",
            )
            if not selected:
                return
            model_path = Path(selected)

        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(config["project"]["name"])).strip("_") or "trainset_model"
        module_dir = self.project_root / "modules" / f"generated_{slug}"
        module_dir.mkdir(parents=True, exist_ok=True)
        parameter_names = trainable_parameter_names(config)
        target_min = [float(config["parameters"][name]["minimum"]) for name in parameter_names]
        target_max = [float(config["parameters"][name]["maximum"]) for name in parameter_names]
        roi = config["roi"]
        inference_config = copy.deepcopy(config)
        for step in inference_config.get("preprocessing", {}).get("steps", []):
            if step.get("plugin") in {"noise", "gaussian_noise", "poisson_noise", "physical_background", "random_edge_crop"}:
                step["enabled"] = False
        module = {
            "id": f"generated_{slug}",
            "name": f"{config['project']['name']} (trained)",
            "model": {
                "format": "pytorch" if model_path.suffix.lower() in {".pt", ".pth"} else "tensorflow_keras",
                "model_path": str(model_path.resolve()),
            },
            "preprocess": {
                "entry": "preprocessing:preprocess",
                "steps": [step["plugin"] for step in inference_config["preprocessing"]["steps"] if step.get("enabled")],
                "params": {"trainset_config": inference_config},
            },
            "io": {"input_shape": [1, int(roi["height"]), int(roi["width"]), 1]},
            "outputs": {
                "type": "parameters",
                "parameter_names": parameter_names,
                "target_min": target_min,
                "target_max": target_max,
            },
        }
        (module_dir / "module.yaml").write_text(
            yaml.safe_dump(module, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (module_dir / "preprocessing.py").write_text(
            """from __future__ import annotations
import copy
import numpy as np
from trainset.generator import apply_preprocessing, build_fixed_mask, crop_roi

def preprocess(image, preprocess_config, module_folder=None, return_steps=False):
    cfg = copy.deepcopy(preprocess_config["params"]["trainset_config"])
    roi_image = crop_roi(np.asarray(image, dtype=np.float32), cfg["roi"])
    mask = build_fixed_mask(roi_image, cfg)
    stages = apply_preprocessing(
        roi_image,
        cfg,
        mask,
        np.random.default_rng(int(cfg["project"]["seed"])),
    )
    result = np.asarray(stages[-1]["image"], dtype=np.float32)
    snapshots = [{"label": stage["name"], "image": stage["image"]} for stage in stages]
    return (result, snapshots) if return_steps else result
""",
            encoding="utf-8",
        )

        predictor = getattr(self.parent_controller, "gisaxs_predict_controller", None)
        if predictor is not None:
            predictor._refresh_modules()
            combo = getattr(self.ui, "gisaxsPredictModuleSelectCombox", None)
            module_name = module["name"]
            if combo is not None and combo.findText(module_name) >= 0:
                combo.setCurrentText(module_name)
            predictor._on_module_selected(module_name)
            self.parent_controller._switch_to_gisaxs_predict()
        QMessageBox.information(self.window, "Model registered", f"Registered prediction module:\n{module_dir}")

    def _update_capabilities(self) -> None:
        available = DatasetGenerator(self.config).bornagain_available
        self.page.preview_capability.setText("BornAgain local simulation available" if available else "BornAgain not installed locally · reference preview only")

    def get_parameters(self) -> Dict[str, Any]:
        return self._collect_config()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        if not isinstance(parameters, dict):
            return
        self.config = synchronize_parameter_specs(merge_config(default_project_config(), parameters))
        self._apply_config_to_page(self.config)
        reference = self.config.get("project", {}).get("reference_file")
        if reference and Path(reference).exists():
            self._load_reference(str(reference))
        self._update_capabilities()
        self._update_geometry_label()
        runtime = self.config.get("runtime", {})
        hpc = self.config.get("hpc", {})
        if runtime.get("last_job_id") and hpc.get("user") and hpc.get("remote_path"):
            self.monitor_timer.start()

    def validate_parameters(self):
        valid, errors, warnings = validate_project_config(self._collect_config())
        return valid, "\n".join(errors or warnings)

    def reset_to_defaults(self) -> None:
        remember = self.page.auto_remember_check.isChecked()
        self.monitor_timer.stop()
        self.config = default_project_config()
        self.config.setdefault("runtime", {})["auto_remember"] = remember
        self.reference_image = None
        self._apply_config_to_page(self.config)
        for canvas in (
            self.page.full_detector_canvas,
            self.page.roi_design_canvas,
            self.page.masked_design_canvas,
            self.page.mask_only_canvas,
        ):
            canvas.set_draw_mode("")
            canvas.set_data(None)
        for index in range(4):
            self.page.set_design_stage_ready(index, False)
        for index in range(len(self.page.STEPS)):
            self.page.set_step_state(index, "Not started")
        self.page.design_tabs.setCurrentIndex(0)
        self.page.validation_badge.setText("Not validated")
        self.page.threshold_summary.setText(
            "Load a reference to calculate detector-gap and hot-pixel locations."
        )
        self._update_capabilities()
        self._update_geometry_label()
        if remember:
            self._autosave_timer.start(100)
        self.status_updated.emit("TrainSet settings reset to built-in defaults")
