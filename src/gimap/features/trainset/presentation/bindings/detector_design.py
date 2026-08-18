"""Detector Design coordination for Trainset."""

from __future__ import annotations


from pathlib import Path

from typing import Any, Dict, Optional

import numpy as np

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
)


class DetectorDesignMixin:
    """Own detector design presentation behavior."""

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
            threshold = self.config.get("mask", {}).get("threshold", {})
            automatic = self.page.fields["mask.threshold.auto_reference_upper"].isChecked()
            low = float(self.page.fields["mask.threshold.minimum"].value())
            high = float(self.page.fields["mask.threshold.maximum"].value())
            summary = self.trainset_view_model.threshold_summary(
                self.reference_image,
                self._current_roi(),
                threshold,
                automatic_upper=automatic,
                lower=low,
                upper=high,
            )
            if summary is None:
                raise ValueError(
                    self.trainset_view_model.state.error_message
                    or "Reference threshold unavailable"
                )
            if automatic:
                high = float(summary["upper"])
                self._set_widget_value(
                    self.page.fields["mask.threshold.maximum"],
                    high,
                )
            total = int(summary["total"])
            masked = int(summary["masked"])
            self.page.threshold_summary.setText(
                f"Reference threshold locations: {masked:,}/{total:,} masked "
                f"({masked / max(total, 1):.2%}) · below {low:.5g}: {int(summary['below']):,} · "
                f"above {high:.5g}: {int(summary['above']):,} · non-finite: {int(summary['invalid']):,}"
            )
        except Exception as exc:
            self.page.threshold_summary.setText(f"Reference threshold unavailable: {exc}")

    def _particle_plugin_changed(self, label: str) -> None:
        plugin = next(
            (spec for spec in self.catalog.plugins("particle") if spec.label == label), None
        )
        if plugin is None:
            return
        existing = (
            self.page.plugin_parameters(self.page.particle_parameter_table)
            if self.page.particle_parameter_table.rowCount()
            else {}
        )
        self.page.set_plugin_parameters(
            self.page.particle_parameter_table, plugin.parameters, existing
        )
        self.page.particle_help.setText(plugin.description)
        is_segment = plugin.key == "spherical_segment"
        self.page.segment_constraint_check.setVisible(is_segment)
        show_spacing = self.page.interference_combo.currentText() == "Paracrystal" and any(
            item["key"] == "radius_nm" for item in plugin.parameters
        )
        self.page.spacing_constraint_check.setEnabled(show_spacing)
        self.page.spacing_constraint_check.setVisible(show_spacing)

    def _interference_plugin_changed(self, label: str) -> None:
        plugin = next(
            (spec for spec in self.catalog.plugins("interference") if spec.label == label), None
        )
        if plugin is None:
            return
        existing = (
            self.page.plugin_parameters(self.page.interference_parameter_table)
            if self.page.interference_parameter_table.rowCount()
            else {}
        )
        self.page.set_plugin_parameters(
            self.page.interference_parameter_table, plugin.parameters, existing
        )
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
            image = self.trainset_view_model.load_reference(Path(path))
            if image is None:
                raise RuntimeError(
                    self.trainset_view_model.state.error_message or "Reference load failed"
                )
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
            self.page.design_info.setText(
                f"{Path(path).name}\nShape: {image.shape[1]} × {image.shape[0]} · dtype: {image.dtype}"
            )
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
            (
                "detector.pixels_x",
                "detector.pixels_y",
                "detector.pixel_size_x_mm",
                "detector.pixel_size_y_mm",
            ),
            values,
        ):
            self._set_widget_value(self.page.fields[path], value)
        self._geometry_changed()

    def _update_geometry_label(self) -> None:
        try:
            config = self._collect_config()
            ranges = self.trainset_view_model.geometry_ranges(config)
            if ranges is None:
                raise RuntimeError(
                    self.trainset_view_model.state.error_message or "Geometry calculation failed"
                )
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
        self.page.full_detector_canvas.set_data(
            self.reference_image, roi=self._current_roi(), beam_center=beam_center
        )
        self.page.full_detector_canvas.set_draw_mode(mode)
        self.page.design_tabs.setCurrentIndex(0)
        self.status_updated.emit("Draw the rectangular ROI on the detector image")

    def _begin_beam_center(self) -> None:
        if self.reference_image is None:
            QMessageBox.information(
                self.window, "Beam center", "Load a real scattering file first."
            )
            return
        self.page.full_detector_canvas.set_draw_mode("beam_center")
        self.page.design_tabs.setCurrentIndex(0)
        self.status_updated.emit("Click the direct-beam position on the full detector")

    def _begin_mask(self, mode: str) -> None:
        if self.reference_image is None:
            QMessageBox.information(self.window, "Mask", "Load a real scattering file first.")
            return
        try:
            roi_image = self.trainset_view_model.crop_reference(
                self.reference_image, self._current_roi()
            )
            if roi_image is None:
                raise RuntimeError(
                    self.trainset_view_model.state.error_message or "ROI crop failed"
                )
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
            self.status_updated.emit(
                f"Beam center selected at x={payload['x']:.1f}, y={payload['y']:.1f} px"
            )
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
        return {
            key: int(self.page.fields[f"roi.{key}"].value())
            for key in ("x", "y", "width", "height")
        }

    def _refresh_design_overlay(self, *_args) -> None:
        if self.reference_image is None:
            return
        try:
            config = self._collect_config()
            roi = self._current_roi()
            overlay = self.trainset_view_model.design_overlay(
                self.reference_image,
                roi,
                config,
                self._random_mask_example,
            )
            if overlay is None:
                raise RuntimeError(
                    self.trainset_view_model.state.error_message or "Design overlay failed"
                )
            roi_image = overlay["roi_image"]
            mask = overlay["mask"]
            mask_label = overlay["mask_label"]
            roi_shape_mask = overlay["roi_shape_mask"]
            self._random_mask_example = overlay["random_mask"]
            self.page.full_detector_canvas.set_data(
                self.reference_image,
                roi=roi,
                beam_center=(
                    self.page.fields["detector.beam_center_x_px"].value(),
                    self.page.fields["detector.beam_center_y_px"].value(),
                ),
            )
            self.page.roi_design_canvas.set_data(
                roi_image, mask=roi_shape_mask if roi_shape_mask.any() else None
            )
            self.page.masked_design_canvas.set_data(roi_image, mask=mask)
            self.page.mask_only_canvas.set_data(mask.astype(np.float32), binary=True)
            self.page.design_info.setText(
                f"Reference: {self.reference_image.shape[1]} × {self.reference_image.shape[0]}\n"
                f"ROI tensor: {roi_image.shape[1]} × {roi_image.shape[0]} · {mask_label} masked: {mask.mean():.2%}\n"
                "Use Draw ROI for detector coordinates; mask shapes are edited in ROI coordinates."
            )
            self.page.design_info.setToolTip(
                f"Mask mode: {mask_label}; masked fraction: {mask.mean():.2%}"
            )
        except Exception as exc:
            self.page.design_info.setText(str(exc))

    def _new_random_mask_example(self) -> None:
        try:
            config = self._collect_config()
            roi = self._current_roi()
            shape = (int(roi["height"]), int(roi["width"]))
            self._random_mask_example = self.trainset_view_model.generate_random_mask(shape, config)
            if self._random_mask_example is None:
                raise RuntimeError(
                    self.trainset_view_model.state.error_message or "Random mask generation failed"
                )
            self.page.random_mask_panel.setVisible(True)
            if self.reference_image is not None:
                self._refresh_design_overlay()
            else:
                self.page.mask_only_canvas.set_data(
                    self._random_mask_example.astype(np.float32), binary=True
                )
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
        steps = {
            str(step.get("plugin")): step
            for step in config.get("preprocessing", {}).get("steps", [])
        }
        if steps.get("physical_background", {}).get("enabled", False):
            for definition in self.catalog.background_parameters():
                self.page.impact_parameter_combo.addItem(
                    f"Background · {definition['label']}",
                    f"physical_background|{definition['key']}",
                )
        if steps.get("gaussian_noise", steps.get("noise", {})).get("enabled", False):
            self.page.impact_parameter_combo.addItem(
                "Gaussian noise · SNR (dB)", "gaussian_noise|snr_db"
            )
        if steps.get("poisson_noise", {}).get("enabled", False):
            self.page.impact_parameter_combo.addItem(
                "Poisson noise · photon-count scale", "poisson_noise|count_scale"
            )
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
        steps = {
            str(step.get("plugin")): step
            for step in config.get("preprocessing", {}).get("steps", [])
        }
        if plugin == "physical_background":
            step = steps.get(plugin, {})
            definition = next(
                item for item in self.catalog.background_parameters() if item["key"] == key
            )
            return (
                plugin,
                key,
                float(step.get(f"{key}_min", definition["minimum"])),
                float(step.get(f"{key}_max", definition["maximum"])),
            )
        if plugin == "gaussian_noise":
            step = steps.get(plugin, steps.get("noise", {}))
            return (
                plugin,
                key,
                float(step.get("snr_min_db", 80.0)),
                float(step.get("snr_max_db", 110.0)),
            )
        step = steps.get("poisson_noise", {})
        return (
            "poisson_noise",
            "count_scale",
            float(step.get("count_scale_min", 1.0)),
            float(step.get("count_scale_max", 20.0)),
        )

    def _force_generate_preview(self) -> None:
        self._start_preview(force=True)
