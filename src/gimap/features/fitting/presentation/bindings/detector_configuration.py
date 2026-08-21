"""Detector Configuration for fitting presentation."""

from __future__ import annotations


import numpy as np


from PyQt5.QtWidgets import (
    QMessageBox,
)

from src.gimap.features.fitting.presentation.detector_parameters_dialog import (
    DetectorParametersDialog,
)


from ..binding_primitives import (
    GISAXS_IMAGE_COLORMAPS,
    _scientific_commands,
)
from ..detector_data_access import analysis_image_for
from ..state import DetectorDisplayState
from ...application import normalize_horizontal_q_axis


class DetectorConfigurationMixin:
    """Own detector configuration behavior."""

    def _auto_find_center(self):
        """GISAXS"""
        self._sync_ui_to_parameters()

        analysis_image = analysis_image_for(self)
        if analysis_image is None:
            QMessageBox.warning(self.main_window, "Warning", "Please import an image first.")
            return

        if not self._has_displayed_image:
            QMessageBox.warning(
                self.main_window,
                "Warning",
                "Please display the image first by clicking the Show button.",
            )
            return

        try:
            had_cut = self._has_existing_cut_result()
            self.status_updated.emit("Searching for the center point automatically...")

            vertical_profile, horizontal_profile = _scientific_commands(self).image.center_profiles(
                analysis_image
            )
            raw_center_y = np.argmax(vertical_profile)
            height = analysis_image.shape[0]
            pixel_center_y = float(height - 1 - raw_center_y)

            profile_total = float(np.sum(horizontal_profile))
            if not np.isfinite(profile_total) or profile_total <= 0:
                raise ValueError("No positive detector intensity remains for center finding")
            pixel_center_x = float(
                np.sum(np.arange(len(horizontal_profile)) * horizontal_profile) / profile_total
            )

            pixel_cutline_height = self._auto_horizontal_cut_thickness_pixels()

            # 4. Estimate cut-line width from the horizontal intensity profile.
            pixel_cutline_width = float(self._calculate_95_percent_width(horizontal_profile))
            if self._should_show_q_axis():
                column_min = int(
                    np.clip(
                        np.floor(pixel_center_x - pixel_cutline_width / 2.0),
                        0,
                        analysis_image.shape[1] - 1,
                    )
                )
                column_max = int(
                    np.clip(
                        np.ceil(pixel_center_x + pixel_cutline_width / 2.0),
                        0,
                        analysis_image.shape[1] - 1,
                    )
                )
                display_y_min = pixel_center_y - pixel_cutline_height / 2.0
                display_y_max = pixel_center_y + pixel_cutline_height / 2.0
                row_min = int(
                    np.clip(
                        np.floor((height - 1) - display_y_max),
                        0,
                        height - 1,
                    )
                )
                row_max = int(
                    np.clip(
                        np.ceil((height - 1) - display_y_min),
                        0,
                        height - 1,
                    )
                )
                self._apply_pixel_region_to_active_coordinates(
                    (row_min, row_max, column_min, column_max),
                    q_mode=True,
                    horizontal_axis=self._horizontal_q_axis(),
                )
                selection_info = dict(self.current_parameter_selection)
                draft_values = (
                    selection_info["center_x"],
                    selection_info["center_y"],
                    selection_info["width"],
                    selection_info["height"],
                )
            else:
                self._set_numeric_control_silently(
                    "gisaxsInputCenterVerticalValue", pixel_center_y
                )
                self._set_numeric_control_silently(
                    "gisaxsInputCenterParallelValue", pixel_center_x
                )
                self._set_numeric_control_silently(
                    "gisaxsInputCutLineVerticalValue", pixel_cutline_height
                )
                self._set_numeric_control_silently(
                    "gisaxsInputCutLineParallelValue", pixel_cutline_width
                )
                self._persist_cut_region_parameters(
                    pixel_center_x,
                    pixel_center_y,
                    pixel_cutline_width,
                    pixel_cutline_height,
                )
                selection_info = self._create_selection_from_parameters(
                    pixel_center_x,
                    pixel_center_y,
                    pixel_cutline_width,
                    pixel_cutline_height,
                )
                draft_values = (
                    pixel_center_x,
                    pixel_center_y,
                    pixel_cutline_width,
                    pixel_cutline_height,
                )
            self._update_parameter_selection_display(selection_info)

            if self._should_show_q_axis():
                self.status_updated.emit(
                    f"Auto center found and snapped to {self._horizontal_q_axis()}/qz grid: "
                    f"center=({draft_values[0]:.6f}, {draft_values[1]:.6f}) nm^-1, "
                    f"span=({draft_values[2]:.6f}, {draft_values[3]:.6f}) nm^-1"
                )
            else:
                self.status_updated.emit(
                    f"Auto center found: center=({pixel_center_x:.1f}, {pixel_center_y:.1f}), "
                    f"cut size=({pixel_cutline_width:.1f}, {pixel_cutline_height:.1f})"
                )

            self.data_source = "cut"
            try:
                if hasattr(self, "_switch_to_normal_display_mode") and callable(
                    self._switch_to_normal_display_mode
                ):
                    self._switch_to_normal_display_mode()
                else:
                    self.display_mode = "normal"
                    if hasattr(self, "_display_mode"):
                        self._display_mode = "normal"
                    if hasattr(self, "_fitting_mode_active"):
                        self._fitting_mode_active = False
                self.display_mode = "normal"
            except Exception:
                self.display_mode = "normal"
            if hasattr(self.ui, "fitCurrentDataCheckBox"):
                try:
                    self.ui.fitCurrentDataCheckBox.blockSignals(True)
                    self.ui.fitCurrentDataCheckBox.setChecked(True)
                finally:
                    try:
                        self.ui.fitCurrentDataCheckBox.blockSignals(False)
                    except Exception:
                        pass

            try:
                self._initialize_roi_from_current_q(force_full=True)
            except Exception:
                pass
            self._apply_roi_to_data_and_refresh()
            self._update_GUI_image("normal")
            self._update_outside_window("normal")
            self._record_cut_geometry_draft(*draft_values)
            if had_cut:
                self._refresh_existing_cut_preserving_view()

        except Exception as e:
            self._fail_fitting_step("center", str(e))
            self.status_updated.emit(f"Cut operation failed: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Cut operation failed:\n{str(e)}")

    def _auto_horizontal_cut_thickness_pixels(self) -> float:
        """Return the detector-row thickness of the horizontal Yoneda cut band."""
        thickness_control = getattr(
            self.ui, "gisaxsAutoYonedaCutThicknessSpinBox", None
        )
        return float(thickness_control.value() if thickness_control is not None else 5)

    def _calculate_95_percent_width(self, profile):
        """Return the profile width containing roughly 95 percent of the intensity."""
        profile = np.asarray(profile, dtype=float)
        if profile.size == 0:
            return 50.0

        total_intensity = float(np.sum(profile))
        if total_intensity <= 0:
            return 50.0

        center_idx = int(np.argmax(profile))
        target_intensity = total_intensity * 0.95
        left_idx = center_idx
        right_idx = center_idx
        current_intensity = float(profile[center_idx])

        while current_intensity < target_intensity and (
            left_idx > 0 or right_idx < profile.size - 1
        ):
            left_val = profile[left_idx - 1] if left_idx > 0 else 0
            right_val = profile[right_idx + 1] if right_idx < profile.size - 1 else 0

            if left_val >= right_val and left_idx > 0:
                left_idx -= 1
                current_intensity += float(profile[left_idx])
            elif right_idx < profile.size - 1:
                right_idx += 1
                current_intensity += float(profile[right_idx])
            else:
                break

        width = right_idx - left_idx + 1
        min_width = 20.0
        max_width = profile.size * 0.8
        return float(max(min_width, min(width, max_width)))

    def _show_detector_parameters(self):
        """Show the detector parameters dialog."""
        try:
            if getattr(self, "detector_params_dialog", None) is not None:
                if self.detector_params_dialog.isVisible():
                    self.detector_params_dialog.raise_()
                    self.detector_params_dialog.activateWindow()
                    return

            self.detector_params_dialog = DetectorParametersDialog(
                self.main_window,
                view_model=self.fitting_view_model,
            )
            self.detector_params_dialog.parameters_changed.connect(
                self._on_detector_parameters_changed
            )
            self.detector_params_dialog.finished.connect(self._on_detector_dialog_finished)
            self.detector_params_dialog.show()
            self.detector_params_dialog.raise_()
            self.detector_params_dialog.activateWindow()

            self.status_updated.emit("Detector Parameters dialog opened")

        except Exception as e:
            self.status_updated.emit(f"Failed to display Detector Parameters dialog: {str(e)}")
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Detector Parameters dialog cannot be displayed:\n{str(e)}",
            )

    def _on_detector_dialog_finished(self):
        """Clear detector dialog reference after close."""
        try:
            self.detector_params_dialog = None
            self.status_updated.emit("Detector Parameters dialog closed")
        except Exception as e:
            self.status_updated.emit(f"Failed to clear detector dialog: {str(e)}")

    def _on_detector_parameters_changed(self, parameters=None):
        """Handle detector parameter changes from the dialog."""
        try:
            had_cut = self._has_existing_cut_result()
            previous_q_mode = bool(getattr(self, "_last_q_mode", self._should_show_q_axis()))
            previous_horizontal = getattr(
                self, "_last_horizontal_q_axis", self._horizontal_q_axis()
            )
            pixel_region = self._current_selection_pixel_region(
                q_mode=previous_q_mode,
                horizontal_axis=previous_horizontal,
            )
            self._update_cutline_labels_units()
            self._update_cutline_step_sizes()

            try:
                self._compute_q_meshgrids_and_store()
            except Exception:
                pass
            self._seed_independent_q_cache()

            current_q_mode = self._should_show_q_axis()
            current_horizontal = self._horizontal_q_axis()
            if pixel_region is not None:
                self._apply_pixel_region_to_active_coordinates(
                    pixel_region,
                    q_mode=current_q_mode,
                    horizontal_axis=current_horizontal,
                )
            self._last_q_mode = current_q_mode
            self._last_horizontal_q_axis = current_horizontal

            if analysis_image_for(self) is not None:
                if getattr(self, "_mirror_fill_detector_gaps", False):
                    self._reapply_input_image_options(refresh=False)
                self._refresh_image_display()
            self._complete_fitting_step("setup", "Detector settings saved")
            if had_cut:
                self._refresh_existing_cut_preserving_view()
                self.status_updated.emit(
                    "Detector parameters saved; the existing cut curve was refreshed"
                )
            else:
                self.status_updated.emit("Detector parameters updated and saved")

        except Exception as e:
            self._fail_fitting_step("setup", str(e))
            self.status_updated.emit(f"Failed to process detector parameter change: {str(e)}")

    def _current_selection_pixel_region(self, *, q_mode: bool, horizontal_axis: str):
        """Resolve the active overlay to analysis-array bounds before geometry changes."""

        info = getattr(self, "current_parameter_selection", None)
        image = analysis_image_for(self)
        if not info or image is None:
            return None
        height, width = image.shape
        bounds = info.get("bounds", {})
        try:
            if q_mode or info.get("is_q_space", False):
                raw_bounds = (
                    info.get("pixel_row_min"),
                    info.get("pixel_row_max"),
                    info.get("pixel_column_min"),
                    info.get("pixel_column_max"),
                )
                if all(value is not None for value in raw_bounds):
                    return tuple(int(value) for value in raw_bounds)
                grid = self._detector_q_grid()
                if grid is None:
                    return None
                region = grid.snap_region(
                    bounds.get("x_min", info.get("center_x", 0.0)),
                    bounds.get("x_max", info.get("center_x", 0.0)),
                    bounds.get("y_min", info.get("center_y", 0.0)),
                    bounds.get("y_max", info.get("center_y", 0.0)),
                    horizontal_axis,
                )
                return (
                    region.row_min,
                    region.row_max,
                    region.column_min,
                    region.column_max,
                )

            x_min = float(bounds.get("x_min", info.get("pixel_center_x", 0.0)))
            x_max = float(bounds.get("x_max", info.get("pixel_center_x", 0.0)))
            y_min = float(bounds.get("y_min", info.get("pixel_center_y", 0.0)))
            y_max = float(bounds.get("y_max", info.get("pixel_center_y", 0.0)))
            return (
                int(np.clip(round((height - 1) - max(y_min, y_max)), 0, height - 1)),
                int(np.clip(round((height - 1) - min(y_min, y_max)), 0, height - 1)),
                int(np.clip(round(min(x_min, x_max)), 0, width - 1)),
                int(np.clip(round(max(x_min, x_max)), 0, width - 1)),
            )
        except (TypeError, ValueError):
            return None

    def _apply_pixel_region_to_active_coordinates(
        self,
        pixel_region,
        *,
        q_mode: bool,
        horizontal_axis: str,
    ) -> None:
        """Project one detector-cell region into the currently visible controls."""

        image = analysis_image_for(self)
        if image is None:
            return
        row_min, row_max, column_min, column_max = pixel_region
        image_height = image.shape[0]
        if q_mode:
            grid = self._detector_q_grid()
            if grid is None:
                return
            region = grid.region_from_pixels(
                row_min,
                row_max,
                column_min,
                column_max,
                horizontal_axis,
            )
            center_x = region.center_horizontal
            center_y = region.center_qz
            selection_width = region.width
            selection_height = region.height
            bounds = {
                "x_min": region.horizontal_min,
                "x_max": region.horizontal_max,
                "y_min": region.qz_min,
                "y_max": region.qz_max,
            }
        else:
            center_x = (column_min + column_max) / 2.0
            display_y_min = (image_height - 1) - row_max
            display_y_max = (image_height - 1) - row_min
            center_y = (display_y_min + display_y_max) / 2.0
            selection_width = float(column_max - column_min + 1)
            selection_height = float(display_y_max - display_y_min + 1)
            bounds = {
                "x_min": float(column_min),
                "x_max": float(column_max),
                "y_min": float(display_y_min),
                "y_max": float(display_y_max),
            }

        values = {
            "gisaxsInputCenterParallelValue": center_x,
            "gisaxsInputCenterVerticalValue": center_y,
            "gisaxsInputCutLineParallelValue": selection_width,
            "gisaxsInputCutLineVerticalValue": selection_height,
        }
        for name, value in values.items():
            if hasattr(self.ui, name):
                self._set_numeric_control_silently(name, value)
        self.current_parameter_selection = {
            "center_x": center_x,
            "center_y": center_y,
            "width": selection_width,
            "height": selection_height,
            "pixel_center_x": (column_min + column_max) / 2.0,
            "pixel_center_y": (image_height - 1) - ((row_min + row_max) / 2.0),
            "pixel_width": column_max - column_min + 1,
            "pixel_height": row_max - row_min + 1,
            "pixel_row_min": row_min,
            "pixel_row_max": row_max,
            "pixel_column_min": column_min,
            "pixel_column_max": column_max,
            "bounds": bounds,
            "is_q_space": bool(q_mode),
            "horizontal_q_axis": horizontal_axis if q_mode else None,
        }
        self._persist_cut_region_parameters(
            center_x,
            center_y,
            selection_width,
            selection_height,
        )

    def _current_detector_display_state(self) -> DetectorDisplayState:
        """Capture the display contract shared by both detector projections."""
        state = DetectorDisplayState(
            log_intensity=self._is_log_mode_enabled(),
            auto_scale=self._is_auto_scale_enabled(),
            vmin=getattr(self, "_current_vmin", None),
            vmax=getattr(self, "_current_vmax", None),
            colormap=getattr(self, "_image_colormap", "viridis"),
            show_cut_region=bool(getattr(self, "_show_cut_region", True)),
            show_center=bool(getattr(self, "_show_center", True)),
            show_q_axis=bool(self._should_show_q_axis()),
            horizontal_q_axis=self._horizontal_q_axis(),
        )
        self.fitting_view_model.update_detector_display(state)
        return state

    def _apply_detector_display_state(self, state: DetectorDisplayState) -> None:
        """Apply an independent-window edit to the shared detector state."""
        if not isinstance(state, DetectorDisplayState):
            return
        previous_q_mode = self._should_show_q_axis()
        previous_horizontal = self._horizontal_q_axis()
        pixel_region = self._current_selection_pixel_region(
            q_mode=previous_q_mode,
            horizontal_axis=previous_horizontal,
        )
        self._show_cut_region = state.show_cut_region
        self._show_center = state.show_center
        self._image_colormap = (
            state.colormap if state.colormap in GISAXS_IMAGE_COLORMAPS else "viridis"
        )
        self._current_vmin = state.vmin
        self._current_vmax = state.vmax
        for name, checked in (
            ("gisaxsInputIntLogCheckBox", state.log_intensity),
            ("gisaxsInputAutoScaleCheckBox", state.auto_scale),
            ("gisaxsInputShowCutRegionCheckBox", state.show_cut_region),
            ("gisaxsInputShowCenterCheckBox", state.show_center),
        ):
            widget = getattr(self.ui, name, None)
            if widget is None:
                continue
            old_block = widget.blockSignals(True)
            widget.setChecked(bool(checked))
            widget.blockSignals(old_block)
        for name, value in (
            ("gisaxsInputVminValue", state.vmin),
            ("gisaxsInputVmaxValue", state.vmax),
        ):
            widget = getattr(self.ui, name, None)
            if widget is None or value is None:
                continue
            old_block = widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(old_block)
        self.fitting_view_model.set_setting(
            "fitting", "detector.show_q_axis", bool(state.show_q_axis)
        )
        horizontal_q_axis = normalize_horizontal_q_axis(state.horizontal_q_axis)
        self.fitting_view_model.set_setting(
            "fitting", "detector.horizontal_q_axis", horizontal_q_axis
        )
        detector_panel = getattr(self.ui, "fittingDetectorSetupPanel", None)
        if detector_panel is not None:
            detector_panel.show_q_axis_checkbox.blockSignals(True)
            detector_panel.show_q_axis_checkbox.setChecked(bool(state.show_q_axis))
            detector_panel.show_q_axis_checkbox.blockSignals(False)
            detector_panel.horizontal_q_combo.blockSignals(True)
            index = detector_panel.horizontal_q_combo.findData(horizontal_q_axis)
            detector_panel.horizontal_q_combo.setCurrentIndex(index if index >= 0 else 0)
            detector_panel.horizontal_q_combo.setEnabled(bool(state.show_q_axis))
            detector_panel.horizontal_q_combo.blockSignals(False)
        if state.vmin is not None:
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.vmin", float(state.vmin)
            )
        if state.vmax is not None:
            self.fitting_view_model.set_setting(
                "fitting", "gisaxs_input.vmax", float(state.vmax)
            )
        self.fitting_view_model.update_detector_display(state)
        self.fitting_view_model.save_settings()
        coordinate_changed = (
            previous_q_mode != bool(state.show_q_axis)
            or previous_horizontal != horizontal_q_axis
        )
        if coordinate_changed:
            self._update_cutline_labels_units()
            self._update_cutline_step_sizes()
            if pixel_region is not None:
                self._apply_pixel_region_to_active_coordinates(
                    pixel_region,
                    q_mode=bool(state.show_q_axis),
                    horizontal_axis=horizontal_q_axis,
                )
            self._last_q_mode = bool(state.show_q_axis)
            self._last_horizontal_q_axis = horizontal_q_axis
        self._apply_image_display_options(sync_window=False)
        if coordinate_changed and self._has_existing_cut_result():
            self._refresh_existing_cut_preserving_view()

    def _on_independent_detector_display_state_changed(
        self, state: DetectorDisplayState
    ) -> None:
        self._apply_detector_display_state(state)
