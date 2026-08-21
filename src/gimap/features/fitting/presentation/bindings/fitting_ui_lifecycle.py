"""Fitting Ui Lifecycle for fitting presentation."""

from __future__ import annotations


from PyQt5.QtCore import QTimer


from PyQt5.QtWidgets import QShortcut

from PyQt5.QtGui import QKeySequence

from ..binding_primitives import (
    _scientific_commands,
    is_matplotlib_available,
)
from ..detector_data_access import analysis_image_for
from ...application import DetectorQGrid, normalize_horizontal_q_axis


class FittingUiLifecycleMixin:
    """Own fitting ui lifecycle behavior."""

    def _setup_meta_debug_shortcut(self):
        """No description."""
        try:
            sc = QShortcut(QKeySequence("Ctrl+Alt+M"), self.ui)

            # 函数说明：实现 dump 相关逻辑。
            def _dump():
                snap = self.param_trigger_manager.debug_dump_meta(verbose=False)
                self._add_fitting_message("==== META SNAPSHOT ====", "INFO")
                for wid, data in snap.items():
                    self._add_fitting_message(f"{wid}: {data}", "INFO")

            sc.activated.connect(_dump)
            self._add_fitting_message("Meta debug shortcut Ctrl+Alt+M registered", "INFO")
        except Exception as e:
            print(f"Failed to register meta debug shortcut: {e}")

    def _initialize_ui(self):
        """No description."""
        if hasattr(self.ui, "gisaxsInputImportButtonValue"):
            self.ui.gisaxsInputImportButtonValue.clear()

        if hasattr(self.ui, "gisaxsInputStackValue"):
            self.ui.gisaxsInputStackValue.setText("1")

        if hasattr(self.ui, "gisaxsInputStackDisplayLabel"):
            self.ui.gisaxsInputStackDisplayLabel.setText("")

        if hasattr(self.ui, "gisaxsInputModelCombox"):
            try:
                last_mode = self.fitting_view_model.get_setting(
                    "fitting", "gisaxs_input.load_mode", "Single"
                )
                idx = self.ui.gisaxsInputModelCombox.findText(last_mode)
                self.ui.gisaxsInputModelCombox.setCurrentIndex(idx if idx >= 0 else 0)
                self.load_mode = self.ui.gisaxsInputModelCombox.currentText()
            except Exception:
                self.load_mode = "Single"
            self._update_stack_controls_visibility()

        if hasattr(self.ui, "gisaxsInputIntLogCheckBox"):
            self.ui.gisaxsInputIntLogCheckBox.setChecked(True)

        if hasattr(self.ui, "gisaxsInputAutoScaleCheckBox"):
            self.ui.gisaxsInputAutoScaleCheckBox.setChecked(True)

        # Manual Single/Stack imports always render immediately.  Keep the
        # legacy checkbox checked for session/JSON compatibility and for the
        # separate in-situ refresh policy.
        if hasattr(self.ui, "gisaxsInputAutoShowCheckBox"):
            self.ui.gisaxsInputAutoShowCheckBox.blockSignals(True)
            self.ui.gisaxsInputAutoShowCheckBox.setChecked(True)
            self.ui.gisaxsInputAutoShowCheckBox.blockSignals(False)

        self._initialize_fit_checkboxes()

        if hasattr(self.ui, "gisaxsInputVminValue"):
            self.ui.gisaxsInputVminValue.setValue(0.0)
            self.ui.gisaxsInputVminValue.setDecimals(6)
            self.ui.gisaxsInputVminValue.setRange(-99999.999999, 99999.999999)
            self.ui.gisaxsInputVminValue.setSingleStep(0.1)
            self.ui.gisaxsInputVminValue.setKeyboardTracking(True)
            self._setup_smart_display(self.ui.gisaxsInputVminValue)

        if hasattr(self.ui, "gisaxsInputVmaxValue"):
            self.ui.gisaxsInputVmaxValue.setValue(0.0)
            self.ui.gisaxsInputVmaxValue.setDecimals(6)
            self.ui.gisaxsInputVmaxValue.setRange(-99999.999999, 99999.999999)
            self.ui.gisaxsInputVmaxValue.setSingleStep(0.1)
            self.ui.gisaxsInputVmaxValue.setKeyboardTracking(True)
            self._setup_smart_display(self.ui.gisaxsInputVmaxValue)

        if hasattr(self.ui, "gisaxsInputCenterVerticalValue"):
            self.ui.gisaxsInputCenterVerticalValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCenterVerticalValue.setDecimals(2)
            self.ui.gisaxsInputCenterVerticalValue.setValue(0.0)
            self.ui.gisaxsInputCenterVerticalValue.setKeyboardTracking(True)

        if hasattr(self.ui, "gisaxsInputCenterParallelValue"):
            self.ui.gisaxsInputCenterParallelValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCenterParallelValue.setDecimals(2)
            self.ui.gisaxsInputCenterParallelValue.setValue(0.0)
            self.ui.gisaxsInputCenterParallelValue.setKeyboardTracking(True)

        if hasattr(self.ui, "gisaxsInputCutLineVerticalValue"):
            self.ui.gisaxsInputCutLineVerticalValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCutLineVerticalValue.setDecimals(2)
            self.ui.gisaxsInputCutLineVerticalValue.setValue(10.0)
            self.ui.gisaxsInputCutLineVerticalValue.setKeyboardTracking(True)

        if hasattr(self.ui, "gisaxsInputCutLineParallelValue"):
            self.ui.gisaxsInputCutLineParallelValue.setRange(-99999.99, 99999.99)
            self.ui.gisaxsInputCutLineParallelValue.setDecimals(2)
            self.ui.gisaxsInputCutLineParallelValue.setValue(10.0)
            self.ui.gisaxsInputCutLineParallelValue.setKeyboardTracking(True)

        self._restore_gisaxs_input_parameters()
        self._initialize_image_display_option_widgets()
        if analysis_image_for(self) is not None:
            QTimer.singleShot(0, self._refresh_image_display)

        self._update_cutline_step_sizes()

        if hasattr(self, "_on_q_mode_changed"):
            QTimer.singleShot(100, self._update_cutline_step_sizes)

        self._set_default_parameters()

        self._update_cutline_labels_units()

        self._initialize_q_mode_state()

        self._check_dependencies()

        self._enforce_insitu_visibility_once()

    def _initialize_fit_checkboxes(self):
        """No description."""
        try:
            if hasattr(self.ui, "fitCurrentDataCheckBox"):
                self.ui.fitCurrentDataCheckBox.blockSignals(True)
                self.ui.fitCurrentDataCheckBox.setChecked(False)
                self.ui.fitCurrentDataCheckBox.blockSignals(False)

            if hasattr(self.ui, "fitLogXCheckBox"):
                self.ui.fitLogXCheckBox.blockSignals(True)
                self.ui.fitLogXCheckBox.setChecked(False)
                self.ui.fitLogXCheckBox.blockSignals(False)
            combo = getattr(self.ui, "fitQViewModeComboBox", None)
            if combo is not None:
                combo.blockSignals(True)
                combo.setCurrentIndex(max(0, combo.findData("signed")))
                combo.blockSignals(False)

            if hasattr(self.ui, "fitLogYCheckBox"):
                self.ui.fitLogYCheckBox.blockSignals(True)
                self.ui.fitLogYCheckBox.setChecked(False)
                self.ui.fitLogYCheckBox.blockSignals(False)

            if hasattr(self.ui, "fitNormCheckBox"):
                self.ui.fitNormCheckBox.blockSignals(True)
                self.ui.fitNormCheckBox.setChecked(False)
                self.ui.fitNormCheckBox.blockSignals(False)

        except Exception as e:
            pass

    def _restore_fit_checkboxes(self, session_data):
        """No description."""
        try:
            if hasattr(self.ui, "fitCurrentDataCheckBox"):
                self.ui.fitCurrentDataCheckBox.blockSignals(True)
                self.ui.fitCurrentDataCheckBox.setChecked(
                    session_data.get("fit_current_data", False)
                )
                self.ui.fitCurrentDataCheckBox.blockSignals(False)

            if hasattr(self.ui, "fitLogXCheckBox"):
                self.ui.fitLogXCheckBox.blockSignals(True)
                self.ui.fitLogXCheckBox.setChecked(session_data.get("fit_log_x", False))
                self.ui.fitLogXCheckBox.blockSignals(False)
            combo = getattr(self.ui, "fitQViewModeComboBox", None)
            if combo is not None:
                q_view_mode = session_data.get("fit_q_view_mode")
                if not q_view_mode:
                    q_view_mode = self._q_view_mode_from_legacy(
                        session_data.get("fit_q_branch", "both"),
                        session_data.get("fit_q_combination", "separate"),
                    )
                combo.blockSignals(True)
                combo.setCurrentIndex(max(0, combo.findData(q_view_mode)))
                combo.blockSignals(False)
            self._update_q_view_hint()

            if hasattr(self.ui, "fitLogYCheckBox"):
                self.ui.fitLogYCheckBox.blockSignals(True)
                self.ui.fitLogYCheckBox.setChecked(session_data.get("fit_log_y", False))
                self.ui.fitLogYCheckBox.blockSignals(False)

            if hasattr(self.ui, "fitNormCheckBox"):
                self.ui.fitNormCheckBox.blockSignals(True)
                self.ui.fitNormCheckBox.setChecked(session_data.get("fit_norm", False))
                self.ui.fitNormCheckBox.blockSignals(False)

        except Exception as e:
            pass

    def _initialize_q_mode_state(self):
        """No description."""
        try:
            current_q_mode = self._should_show_q_axis()
            self._last_q_mode = current_q_mode
            self._last_horizontal_q_axis = self._horizontal_q_axis()
        except Exception as e:
            self._last_q_mode = False
            self._last_horizontal_q_axis = "qy"

    def _setup_smart_display(self, spinbox):
        """No description."""
        try:
            spinbox.valueChanged.connect(lambda value: self._update_spinbox_format(spinbox, value))
            spinbox.editingFinished.connect(
                lambda: self._update_spinbox_format(spinbox, spinbox.value())
            )
            self._update_spinbox_format(spinbox, spinbox.value())
        except Exception:
            spinbox.setDecimals(2)

    def _update_spinbox_format(self, spinbox, value):
        """No description."""
        try:
            is_log_mode = False
            if hasattr(self.ui, "gisaxsInputIntLogCheckBox"):
                is_log_mode = self.ui.gisaxsInputIntLogCheckBox.isChecked()

            if is_log_mode:
                spinbox.setDecimals(2)
            else:
                if abs(value - round(value)) < 1e-9:
                    spinbox.setDecimals(0)
                else:
                    value_str = f"{value:.6f}".rstrip("0").rstrip(".")
                    if "." in value_str:
                        decimal_places = len(value_str.split(".")[1])
                        decimal_places = min(decimal_places, 6)
                        decimal_places = max(decimal_places, 1)
                        spinbox.setDecimals(decimal_places)
                    else:
                        spinbox.setDecimals(0)
        except Exception:
            try:
                if (
                    hasattr(self.ui, "gisaxsInputIntLogCheckBox")
                    and self.ui.gisaxsInputIntLogCheckBox.isChecked()
                ):
                    spinbox.setDecimals(2)
                else:
                    spinbox.setDecimals(0)
            except:
                spinbox.setDecimals(2)

    def _refresh_vmin_vmax_display(self):
        """No description."""
        try:
            if hasattr(self.ui, "gisaxsInputVminValue"):
                self._update_spinbox_format(
                    self.ui.gisaxsInputVminValue, self.ui.gisaxsInputVminValue.value()
                )
            if hasattr(self.ui, "gisaxsInputVmaxValue"):
                self._update_spinbox_format(
                    self.ui.gisaxsInputVmaxValue, self.ui.gisaxsInputVmaxValue.value()
                )
        except Exception:
            pass

    def _check_dependencies(self):
        """No description."""
        if not self.fitting_view_model.storage.dependency_available("fabio"):
            self.status_updated.emit(
                "Warning: fabio library not available. CBF processing will be disabled."
            )
        if not is_matplotlib_available():
            self.status_updated.emit(
                "Warning: matplotlib not available. Image display will be disabled."
            )

    def _is_q_space_mode(self):
        """Q-space"""
        try:
            return self._should_show_q_axis()
        except Exception:
            return False

    def _delayed_cut_update(self):
        """No description."""
        try:
            if hasattr(self, "_cut_data") and self._cut_data is not None:
                self._execute_cut()
        except Exception as e:
            pass

    def _on_parameter_display_changed(self):
        """No description."""
        try:
            if getattr(self, "_initializing", False):
                return
            had_cut = self._has_existing_cut_result()
            if hasattr(self, "_update_stack_display"):
                self._update_stack_display()

            center_x = 0
            center_y = 0
            width = 0
            height = 0

            if hasattr(self.ui, "gisaxsInputCenterParallelValue"):
                center_x = self.ui.gisaxsInputCenterParallelValue.value()
            if hasattr(self.ui, "gisaxsInputCenterVerticalValue"):
                center_y = self.ui.gisaxsInputCenterVerticalValue.value()
            if hasattr(self.ui, "gisaxsInputCutLineParallelValue"):
                width = self.ui.gisaxsInputCutLineParallelValue.value()
            if hasattr(self.ui, "gisaxsInputCutLineVerticalValue"):
                height = self.ui.gisaxsInputCutLineVerticalValue.value()

            if width > 0 and height > 0:
                selection_info = self._create_selection_from_parameters(
                    center_x, center_y, width, height
                )
                self._update_parameter_selection_display(selection_info)

            self._delayed_cutline_update()
            if had_cut:
                self._refresh_existing_cut_preserving_view()

        except Exception as e:
            pass

    def _trigger_delayed_cut_update(self):
        """ut"""
        try:
            if not hasattr(self, "_cut_update_timer"):
                from PyQt5.QtCore import QTimer

                self._cut_update_timer = QTimer()
                self._cut_update_timer.setSingleShot(True)
                self._cut_update_timer.timeout.connect(self._delayed_cut_image_update)

            self._cut_update_timer.stop()
            self._cut_update_timer.start(300)

        except Exception as e:
            pass

    def _delayed_cut_image_update(self):
        """Update the detector overlay without implicitly recalculating the cut."""
        try:
            self._delayed_cutline_update()

        except Exception as e:
            pass

    def _on_cutline_parameters_immediate_update(self):
        """No description."""
        try:
            if hasattr(self, "_cut_update_timer"):
                self._cut_update_timer.stop()

            self._delayed_cut_image_update()

        except Exception as e:
            pass

    def _update_cutline_step_sizes(self):
        """No description."""
        try:
            is_q_mode = self._is_q_space_mode()

            if is_q_mode:
                step_size = 0.0001
                decimals = 6
            else:
                step_size = 1.0
                decimals = 2

            cutline_controls = [
                "gisaxsInputCenterVerticalValue",
                "gisaxsInputCenterParallelValue",
                "gisaxsInputCutLineVerticalValue",
                "gisaxsInputCutLineParallelValue",
            ]

            for control_name in cutline_controls:
                if hasattr(self.ui, control_name):
                    control = getattr(self.ui, control_name)
                    if hasattr(control, "setDecimals"):
                        control.setDecimals(decimals)
                    control.setSingleStep(step_size)

            cutline_step_controls = [
                "gisaxsInputCutLineVerticalStep",
                "gisaxsInputCutLineParallelStep",
                "gisaxsInputCenterVerticalStep",
                "gisaxsInputCenterParallelStep",
            ]
            for control_name in cutline_step_controls:
                if hasattr(self.ui, control_name):
                    control = getattr(self.ui, control_name)
                    control.setProperty("defaultStepValue", step_size)
                    control.blockSignals(True)
                    if hasattr(control, "setDecimals"):
                        control.setDecimals(decimals)
                    control.setSingleStep(step_size)
                    control.setValue(step_size)
                    control.blockSignals(False)

            self.status_updated.emit(
                f"Cut Line step size updated to {step_size} ({'Q-space' if is_q_mode else 'Pixel'} mode)"
            )

        except Exception as e:
            self.status_updated.emit(f"Error updating cut line step sizes: {str(e)}")

    def _update_cutline_labels_units(self):
        """No description."""
        try:
            show_q_axis = self._should_show_q_axis()

            if show_q_axis:
                horizontal_name = self._horizontal_q_axis()
                vertical_label = "Center qz (nm⁻¹)"
                horizontal_label = f"Center {horizontal_name} (nm⁻¹)"
                vertical_size_label = "qz span (nm⁻¹)"
                horizontal_size_label = f"{horizontal_name} span (nm⁻¹)"
            else:
                vertical_label = "Center Vertical (px)"
                horizontal_label = "Center Parallel (px)"
                vertical_size_label = "Vertical (px)"
                horizontal_size_label = "Parallel (px)"

            if hasattr(self.ui, "gisaxsInputCenterVerticalLabel"):
                self.ui.gisaxsInputCenterVerticalLabel.setText(vertical_label)

            if hasattr(self.ui, "gisaxsInputCenterParallelLabel"):
                self.ui.gisaxsInputCenterParallelLabel.setText(horizontal_label)

            if hasattr(self.ui, "gisaxsInputCutLineVerticalLabel"):
                self.ui.gisaxsInputCutLineVerticalLabel.setText(vertical_size_label)

            if hasattr(self.ui, "gisaxsInputCutLineParallelLabel"):
                self.ui.gisaxsInputCutLineParallelLabel.setText(horizontal_size_label)

        except Exception as e:
            pass

    def _should_show_q_axis(self):
        """No description."""
        try:
            return self.fitting_view_model.get_setting("fitting", "detector.show_q_axis", False)
        except Exception:
            return False

    def _horizontal_q_axis(self):
        """Return the selected detector horizontal coordinate (qy or signed qr)."""
        try:
            return normalize_horizontal_q_axis(
                self.fitting_view_model.get_setting(
                    "fitting", "detector.horizontal_q_axis", "qy"
                )
            )
        except Exception:
            return "qy"

    def _horizontal_q_label(self):
        return r"$q_r$ (nm$^{-1}$)" if self._horizontal_q_axis() == "qr" else r"$q_y$ (nm$^{-1}$)"

    def _detector_q_grid(self):
        """Return the q grid aligned with the canonical analysis image."""
        if self.qy_matrix is None or self.qz_matrix is None or self.qr_matrix is None:
            self._compute_q_meshgrids_and_store()
        if self.qy_matrix is None or self.qz_matrix is None or self.qr_matrix is None:
            return None
        try:
            return DetectorQGrid(self.qy_matrix, self.qz_matrix, self.qr_matrix)
        except ValueError:
            return None

    def _get_cached_q_meshgrids(self):
        """Return active horizontal-q and qz grids in analysis-array order."""
        try:
            grid = self._detector_q_grid()
            return grid.meshes(self._horizontal_q_axis()) if grid is not None else (None, None)
        except Exception:
            return None, None

    def _get_display_q_meshgrids(self):
        """Return q grids in the same row order as the detector preview image."""
        grid = self._detector_q_grid()
        return (
            grid.display_meshes(self._horizontal_q_axis())
            if grid is not None
            else (None, None)
        )

    def _snap_q_point(self, horizontal_q, qz):
        grid = self._detector_q_grid()
        if grid is None:
            return None
        return grid.nearest_point(horizontal_q, qz, self._horizontal_q_axis())

    def _snap_q_region(self, horizontal_min, horizontal_max, qz_min, qz_max):
        grid = self._detector_q_grid()
        if grid is None:
            return None
        return grid.snap_region(
            horizontal_min,
            horizontal_max,
            qz_min,
            qz_max,
            self._horizontal_q_axis(),
        )

    def _convert_q_to_pixel_coordinates(
        self,
        center_horizontal,
        center_qz,
        width_q,
        height_q,
    ):
        """Map q selection values to the nearest detector cells."""

        grid = self._detector_q_grid()
        if grid is None:
            return {"center_x": 0, "center_y": 0, "width": 1, "height": 1}
        point = grid.nearest_point(
            center_horizontal,
            center_qz,
            self._horizontal_q_axis(),
        )
        region = grid.snap_region(
            center_horizontal - width_q / 2.0,
            center_horizontal + width_q / 2.0,
            center_qz - height_q / 2.0,
            center_qz + height_q / 2.0,
            self._horizontal_q_axis(),
        )
        return {
            "center_x": point.column,
            "center_y": grid.qz.shape[0] - 1 - point.row,
            "width": region.column_max - region.column_min + 1,
            "height": region.row_max - region.row_min + 1,
            "row_min": region.row_min,
            "row_max": region.row_max,
            "column_min": region.column_min,
            "column_max": region.column_max,
        }
