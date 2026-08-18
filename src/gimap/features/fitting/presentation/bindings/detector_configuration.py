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
    _scientific_commands,
)


class DetectorConfigurationMixin:
    """Own detector configuration behavior."""

    def _auto_find_center(self):
        """GISAXS"""
        self._sync_ui_to_parameters()

        if self.current_stack_data is None:
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
            self.status_updated.emit("Searching for the center point automatically...")

            vertical_profile, horizontal_profile = _scientific_commands(self).image.center_profiles(
                self.current_stack_data
            )
            raw_center_y = np.argmax(vertical_profile)
            height = self.current_stack_data.shape[0]
            pixel_center_y = float(height - 1 - raw_center_y)

            profile_total = float(np.sum(horizontal_profile))
            if not np.isfinite(profile_total) or profile_total <= 0:
                raise ValueError("No positive detector intensity remains for center finding")
            pixel_center_x = float(
                np.sum(np.arange(len(horizontal_profile)) * horizontal_profile) / profile_total
            )

            pixel_cutline_height = 20.0

            # 4. Estimate cut-line width from the horizontal intensity profile.
            pixel_cutline_width = float(self._calculate_95_percent_width(horizontal_profile))

            self.ui.gisaxsInputCenterVerticalValue.setValue(pixel_center_y)
            self.ui.gisaxsInputCenterParallelValue.setValue(pixel_center_x)
            self.ui.gisaxsInputCutLineVerticalValue.setValue(pixel_cutline_height)
            self.ui.gisaxsInputCutLineParallelValue.setValue(pixel_cutline_width)

            selection_info = self._create_selection_from_parameters(
                pixel_center_x,
                pixel_center_y,
                pixel_cutline_width,
                pixel_cutline_height,
            )
            self._update_parameter_selection_display(selection_info)

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

        except Exception as e:
            self.status_updated.emit(f"Cut operation failed: {str(e)}")
            QMessageBox.critical(self.main_window, "Error", f"Cut operation failed:\n{str(e)}")

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
            self._update_cutline_labels_units()
            self._update_cutline_step_sizes()

            if hasattr(self, "_update_parameter_values_for_q_axis"):
                self._update_parameter_values_for_q_axis()

            try:
                self._compute_q_meshgrids_and_store()
            except Exception:
                pass

            if (
                self.current_cut_data is not None
                and getattr(self, "current_stack_data", None) is not None
            ):
                self._perform_cut()
                self.status_updated.emit("Detector parameters updated; Cut results recalculated")
            else:
                if getattr(self, "current_stack_data", None) is not None:
                    self._refresh_image_display()
                self.status_updated.emit("Detector parameters updated and saved")

        except Exception as e:
            self.status_updated.emit(f"Failed to process detector parameter change: {str(e)}")
