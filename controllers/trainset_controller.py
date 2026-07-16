from __future__ import annotations

import copy
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

from trainset.backends import SlurmBackend, read_metrics
from trainset.config import default_project_config, load_project_config, merge_config, save_project_config, validate_project_config
from trainset.generator import DatasetGenerator, build_fixed_mask, build_random_mask, build_roi_shape_mask, crop_roi, load_scattering_image
from trainset.geometry import roi_to_spherical_ranges
from trainset.job_package import prepare_job_package
from trainset.plugins import REGISTRY
from ui.trainset_build_page import TrainsetBuildPage


class _WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class _FunctionWorker(QRunnable):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = _WorkerSignals()

    def run(self):
        try:
            self.signals.finished.emit(self.function(*self.args, **self.kwargs))
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
        self.thread_pool = QThreadPool.globalInstance()
        self._initialized = False
        self._remote_refresh_running = False
        self._result_sync_started = False
        self._legacy_page_widgets = []
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
        page.draw_roi_button.clicked.connect(lambda: self._begin_roi("roi"))
        page.draw_ellipse_roi_button.clicked.connect(lambda: self._begin_roi("roi_ellipse"))
        page.draw_rectangle_button.clicked.connect(lambda: self._begin_mask("rectangle"))
        page.draw_circle_button.clicked.connect(lambda: self._begin_mask("ellipse"))
        page.remove_mask_button.clicked.connect(self._remove_selected_masks)
        page.clear_masks_button.clicked.connect(self._clear_masks)
        page.mask_region_created.connect(self._region_created)
        page.generate_preview_button.clicked.connect(self._generate_preview)
        page.preview_button.clicked.connect(lambda: page.step_list.setCurrentRow(1))
        page.validate_button.clicked.connect(self._validate_and_report)
        page.load_button.clicked.connect(self._load_project_dialog)
        page.save_button.clicked.connect(self._save_project_dialog)
        page.prepare_button.clicked.connect(self._prepare_hpc_job)
        page.submit_button.clicked.connect(self._submit_maxwell)
        page.model_validate_button.clicked.connect(self._validate_model_contract)
        page.local_folder_button.clicked.connect(self._choose_workspace)
        page.local_python_button.clicked.connect(self._choose_local_python)
        page.local_prepare_button.clicked.connect(self._prepare_local_job)
        page.local_generate_button.clicked.connect(self._run_local_generation)
        page.local_train_button.clicked.connect(self._run_local_training)
        page.connection_button.clicked.connect(self._test_connection)
        page.hpc_prepare_button.clicked.connect(self._prepare_hpc_job)
        page.hpc_submit_button.clicked.connect(self._submit_maxwell)
        page.refresh_job_button.clicked.connect(self._refresh_job)
        page.sync_results_button.clicked.connect(self._sync_results)
        page.register_model_button.clicked.connect(self._register_best_model)
        page.storage_accept_check.toggled.connect(self._storage_acceptance_changed)
        page.reference_path.editingFinished.connect(self._load_reference_from_field)
        page.fields["detector.preset"].currentTextChanged.connect(self._apply_detector_preset)
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
        page.mask_shape_table.itemChanged.connect(self._mask_config_changed)

    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._apply_config_to_page(self.config)
        self._update_capabilities()
        self._update_geometry_label()
        self.status_updated.emit("Trainset workspace ready")

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
        for path, widget in self.page.fields.items():
            if path.startswith("pre.") or path in {"sample.particle_label", "sample.particle_material", "sample.interference_label", "model.channels"}:
                continue
            _deep_set(config, path, self._widget_value(widget))

        parameters: Dict[str, Dict[str, Any]] = {}
        for row in range(self.page.parameter_table.rowCount()):
            cells = [self.page.parameter_table.item(row, column) for column in range(4)]
            if not cells[0] or not cells[0].text().strip():
                continue
            parameters[cells[0].text().strip()] = {
                "distribution": cells[1].text().strip() if cells[1] else "uniform",
                "minimum": float(cells[2].text()) if cells[2] else 0.0,
                "maximum": float(cells[3].text()) if cells[3] else 1.0,
            }
        config["parameters"] = parameters

        particle_label = self.page.fields["sample.particle_label"].currentText()
        particle = next((spec for spec in REGISTRY.list("particle") if spec.label == particle_label), None)
        config["sample"]["particles"] = [{
            "plugin": particle.key if particle else "spherical_segment",
            "material": self.page.fields["sample.particle_material"].currentText(),
            "enabled": True,
        }]
        interference_label = self.page.fields["sample.interference_label"].currentText()
        interference = next((spec for spec in REGISTRY.list("interference") if spec.label == interference_label), None)
        config["sample"]["interference"]["plugin"] = interference.key if interference else "none"
        config["sample"]["interference"]["enabled"] = bool(interference and interference.key != "none")
        layers = []
        for row in range(self.page.layer_table.rowCount()):
            values = [self.page.layer_table.item(row, column).text() if self.page.layer_table.item(row, column) else "" for column in range(4)]
            if values[1]:
                layers.append({"enabled": values[0].strip().lower() not in {"0", "false", "no"}, "material": values[1], "thickness_nm": float(values[2] or 0), "roughness_nm": float(values[3] or 0)})
        config["sample"]["layers"] = layers
        config["mask"]["fixed_shapes"] = self.page.mask_shapes()

        config["preprocessing"]["steps"] = [
            {"plugin": "noise", "enabled": self.page.fields["pre.noise.enabled"].isChecked(), "snr_min_db": self.page.fields["pre.noise.min"].value(), "snr_max_db": self.page.fields["pre.noise.max"].value()},
            {"plugin": "mask", "enabled": self.page.fields["pre.mask.enabled"].isChecked()},
            {"plugin": "log", "enabled": self.page.fields["pre.log.enabled"].isChecked(), "epsilon": 1e-6},
            {"plugin": "normalize", "enabled": self.page.fields["pre.normalize.enabled"].isChecked(), "mode": self.page.fields["pre.normalize.mode"].currentText(), "lower": self.page.fields["pre.normalize.lower"].value(), "upper": self.page.fields["pre.normalize.upper"].value()},
            {"plugin": "random_edge_crop", "enabled": self.page.fields["pre.edge.enabled"].isChecked(), "maximum_px": self.page.fields["pre.edge.maximum"].value()},
        ]
        channels_text = self.page.fields["model.channels"].text()
        config["model"]["channels"] = [int(item.strip()) for item in channels_text.split(",") if item.strip()]
        self.config = config
        self.parameters_changed.emit("Trainset parameters", copy.deepcopy(config))
        return config

    def _apply_config_to_page(self, config: Dict[str, Any]) -> None:
        self.page.project_name.setText(str(config.get("project", {}).get("name", "")))
        special = {
            "sample.particle_label": next((spec.label for spec in REGISTRY.list("particle") if spec.key == config["sample"]["particles"][0]["plugin"]), "Spherical segment"),
            "sample.particle_material": config["sample"]["particles"][0].get("material", "Copper"),
            "sample.interference_label": next((spec.label for spec in REGISTRY.list("interference") if spec.key == config["sample"]["interference"].get("plugin")), "None"),
            "model.channels": ", ".join(str(value) for value in config["model"]["channels"]),
        }
        for path, widget in self.page.fields.items():
            if path in special:
                self._set_widget_value(widget, special[path])
            elif not path.startswith("pre."):
                value = _deep_get(config, path)
                if value is not None:
                    self._set_widget_value(widget, value)
        steps = {step["plugin"]: step for step in config.get("preprocessing", {}).get("steps", [])}
        pre_map = {
            "pre.noise.enabled": steps.get("noise", {}).get("enabled", True),
            "pre.noise.min": steps.get("noise", {}).get("snr_min_db", 80.0),
            "pre.noise.max": steps.get("noise", {}).get("snr_max_db", 110.0),
            "pre.mask.enabled": steps.get("mask", {}).get("enabled", True),
            "pre.log.enabled": steps.get("log", {}).get("enabled", True),
            "pre.normalize.enabled": steps.get("normalize", {}).get("enabled", True),
            "pre.normalize.mode": steps.get("normalize", {}).get("mode", "range"),
            "pre.normalize.lower": steps.get("normalize", {}).get("lower", 0.0),
            "pre.normalize.upper": steps.get("normalize", {}).get("upper", 1.0),
            "pre.edge.enabled": steps.get("random_edge_crop", {}).get("enabled", False),
            "pre.edge.maximum": steps.get("random_edge_crop", {}).get("maximum_px", 4),
        }
        for path, value in pre_map.items():
            self._set_widget_value(self.page.fields[path], value)

        self.page.mask_shape_table.setRowCount(0)
        for shape in config.get("mask", {}).get("fixed_shapes", []):
            self.page.add_mask_shape(shape)
        self.page.parameter_table.setRowCount(0)
        for row, (name, spec) in enumerate(config.get("parameters", {}).items()):
            self.page.parameter_table.insertRow(row)
            from PyQt5.QtWidgets import QTableWidgetItem

            for column, value in enumerate((name, spec.get("distribution", "uniform"), spec.get("minimum", 0), spec.get("maximum", 1))):
                self.page.parameter_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.page.layer_table.setRowCount(0)
        for row, layer in enumerate(config.get("sample", {}).get("layers", [])):
            self.page.layer_table.insertRow(row)
            from PyQt5.QtWidgets import QTableWidgetItem

            values = ("1" if layer.get("enabled", True) else "0", layer.get("material", ""), layer.get("thickness_nm", 0), layer.get("roughness_nm", 0))
            for column, value in enumerate(values):
                self.page.layer_table.setItem(row, column, QTableWidgetItem(str(value)))

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
        self._refresh_design_overlay()
        self.page.set_design_stage_ready(1, True)
        self.page.design_tabs.setCurrentIndex(1)
        self.page.set_step_state(0, "ROI ready")

    def _mask_config_changed(self, *_args) -> None:
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
        self.page.full_detector_canvas.set_data(self.reference_image, roi=self._current_roi())
        self.page.full_detector_canvas.set_draw_mode(mode)
        self.page.design_tabs.setCurrentIndex(0)
        shape = "ellipse" if mode == "roi_ellipse" else "rectangle"
        self.status_updated.emit(f"Draw the ROI {shape} on the detector image")

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
        if mode in {"roi", "roi_ellipse"}:
            for key in ("x", "y", "width", "height"):
                self._set_widget_value(self.page.fields[f"roi.{key}"], int(payload[key]))
            table = self.page.mask_shape_table
            table.blockSignals(True)
            try:
                self.page.remove_mask_shapes_by_type("roi_ellipse_exterior")
                if mode == "roi_ellipse":
                    width, height = int(payload["width"]), int(payload["height"])
                    self.page.add_mask_shape({
                        "type": "roi_ellipse_exterior",
                        "cx": width / 2.0,
                        "cy": height / 2.0,
                        "radius_x": max(1.0, width / 2.0),
                        "radius_y": max(1.0, height / 2.0),
                    })
            finally:
                table.blockSignals(False)
            self.page.full_detector_canvas.set_draw_mode("")
            self._update_geometry_label()
            self.page.set_design_stage_ready(1, True)
            self.page.design_tabs.setCurrentIndex(1)
            self.page.set_step_state(0, "Ellipse ROI ready" if mode == "roi_ellipse" else "ROI ready")
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
                seed = int(config.get("project", {}).get("seed", 42))
                mask = build_random_mask(roi_image.shape, config, np.random.default_rng(seed))
                mask_label = "random example"
            else:
                mask = build_fixed_mask(roi_image, config)
                mask_label = "fixed"
            roi_shape_mask = build_roi_shape_mask(roi_image.shape, config)
            self.page.full_detector_canvas.set_data(self.reference_image, roi=roi)
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

    def _generate_preview(self) -> None:
        config = self._collect_config()
        valid, errors, warnings = validate_project_config(config, require_reference=True)
        if not valid:
            QMessageBox.warning(self.window, "Preview blocked", "\n".join(errors))
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)  # type: ignore[name-defined]
            started = time.perf_counter()
            generator = DatasetGenerator(config)
            if self.page.preview_mode.currentText() == "Dry run":
                batch = generator.generate(self.page.preview_count.value(), mode="dry")
                image = batch["images"][0]
                result = generator.generate(1, mode="preview")
                result.stages.append({"name": "Final", "image": image, "mask": batch["masks"][0]})
            else:
                result = generator.generate(self.page.preview_count.value(), mode="preview")
            parameter_samples = generator.sample_parameters(self.page.preview_count.value())
            total_samples = int(config["dataset"]["number_of_samples"])
            final_image = np.asarray(result.stages[-1]["image"], dtype=np.float32)
            bytes_per_sample = final_image.nbytes + final_image.size + 4 * len(config.get("parameters", {}))
            result.stats["estimated_dataset_gib"] = round(total_samples * bytes_per_sample / (1024**3), 3)
            elapsed = time.perf_counter() - started
            result.stats["preview_elapsed_s"] = round(elapsed, 3)
            if self.page.preview_mode.currentText() == "Dry run":
                result.stats["estimated_generation_hours"] = round(
                    elapsed / max(1, self.page.preview_count.value()) * total_samples / 3600.0,
                    2,
                )
            else:
                result.stats["generation_time_estimate"] = "Run Dry run with BornAgain to estimate"
            self.page.set_preview_stages(self.reference_image, result.stages, result.stats, result.spectrum_x, result.spectrum_y)
            self.page.set_parameter_samples(parameter_samples)
            self.page.preview_gate_table.item(0, 1).setText("Ready")
            self.page.preview_gate_table.item(1, 1).setText("Ready")
            self.page.preview_gate_table.item(2, 1).setText("Ready")
            self._storage_acceptance_changed(self.page.storage_accept_check.isChecked())
            self.page.validation_badge.setText("Preview ready")
            self.page.set_step_state(1, "Preview ready")
            self.progress_updated.emit(100)
            self.status_updated.emit("Local reference pipeline preview generated")
        except Exception as exc:
            QMessageBox.warning(self.window, "Preview failed", str(exc))
            self.generation_error.emit(str(exc))
        finally:
            QApplication.restoreOverrideCursor()

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
            channels = config["model"]["channels"]
            out_h, out_w = height, width
            for _ in channels:
                out_h, out_w = out_h // 2, out_w // 2
            if min(out_h, out_w) < 1:
                raise ValueError("Too many pooling stages for the configured ROI size.")
            outputs = len(config["parameters"])
            try:
                import tensorflow as tf

                inputs = tf.keras.Input(shape=(height, width, 1))
                x = inputs
                for channel in channels:
                    x = tf.keras.layers.Conv2D(int(channel), int(config["model"]["kernel_size"]), padding="same", activation="relu")(x)
                    x = tf.keras.layers.MaxPool2D()(x)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                prediction = tf.keras.layers.Dense(outputs)(x)
                model = tf.keras.Model(inputs, prediction)
                result = model(np.zeros((1, height, width, 1), dtype=np.float32), training=False)
                summary = f"Forward pass OK\nInput: (1, {height}, {width}, 1)\nOutput: {tuple(result.shape)}\nParameters: {model.count_params():,}"
            except Exception as exc:
                summary = f"Static tensor contract OK\nInput: (1, {height}, {width}, 1)\nOutput: (1, {outputs})\nTensorFlow forward pass unavailable: {exc}"
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
                "environment.yml\nslurm_generate.sh\nslurm_train.sh\nsrc/trainset/\ndataset/\nresults/\nlogs/"
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

    def _start_local_process(self, arguments) -> None:
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
        python_executable = self.page.fields["training.local_python"].text().strip() or sys.executable
        process.start(python_executable, arguments)
        self.page.step_list.setCurrentRow(4)

    def _run_local_generation(self) -> None:
        count = self.page.preview_count.value() if self.page.preview_mode.currentText() == "Dry run" else int(self._collect_config()["dataset"]["number_of_samples"])
        self._start_local_process(["generate_dataset.py", "--samples", str(count), "--mode", "full"])

    def _run_local_training(self) -> None:
        self._start_local_process(["train.py"])

    def _local_process_finished(self, exit_code: int, _status) -> None:
        state = "COMPLETED" if exit_code == 0 else "FAILED"
        self.page.job_state.setText(state)
        self.page.set_step_state(4, state)
        if exit_code == 0:
            self.generation_finished.emit()
        else:
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
        self.page.preview_gate_table.item(4, 1).setText("Ready" if accepted else "Pending")

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
        parameter_names = list(config.get("parameters", {}))
        target_min = [float(config["parameters"][name]["minimum"]) for name in parameter_names]
        target_max = [float(config["parameters"][name]["maximum"]) for name in parameter_names]
        roi = config["roi"]
        inference_config = copy.deepcopy(config)
        for step in inference_config.get("preprocessing", {}).get("steps", []):
            if step.get("plugin") in {"noise", "random_edge_crop"}:
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
        self.config = merge_config(default_project_config(), parameters)
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
        self.monitor_timer.stop()
        self.config = default_project_config()
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
        self._update_capabilities()
        self._update_geometry_label()
