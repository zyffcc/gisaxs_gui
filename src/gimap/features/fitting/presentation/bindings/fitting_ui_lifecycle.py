"""Fitting Ui Lifecycle for fitting presentation."""

from __future__ import annotations


from PyQt5.QtCore import QTimer


from PyQt5.QtWidgets import QShortcut

from PyQt5.QtGui import QKeySequence

from ..binding_primitives import (
    _scientific_commands,
    is_matplotlib_available,
)


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
            try:
                if self.load_mode == "In-situ" and hasattr(self.ui, "gisaxsInputStackValue"):
                    self.ui.gisaxsInputStackValue.setVisible(False)
            except Exception:
                pass
            self._update_stack_controls_visibility()

        if hasattr(self.ui, "gisaxsInputIntLogCheckBox"):
            self.ui.gisaxsInputIntLogCheckBox.setChecked(True)

        if hasattr(self.ui, "gisaxsInputAutoScaleCheckBox"):
            self.ui.gisaxsInputAutoScaleCheckBox.setChecked(True)

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
        if getattr(self, "current_stack_data", None) is not None:
            QTimer.singleShot(0, self._refresh_image_display)

        self._update_cutline_step_sizes()

        if hasattr(self, "_on_q_mode_changed"):
            QTimer.singleShot(100, self._update_cutline_step_sizes)

        self._set_default_parameters()

        self._update_cutline_labels_units()

        self._initialize_q_mode_state()

        self._check_dependencies()

        try:
            if getattr(self, "load_mode", "Single") == "In-situ" and self._is_auto_show_enabled():
                self._start_insitu_timer()
            self._enforce_insitu_visibility_once()
        except Exception:
            pass

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
        except Exception as e:
            self._last_q_mode = False

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

            self._trigger_delayed_cut_update()

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
        """Cut"""
        try:
            if (
                self.current_cut_data is not None
                and hasattr(self, "current_stack_data")
                and self.current_stack_data is not None
            ):
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

                self._perform_cut()
                self.status_updated.emit(
                    f"Auto-updated cut with new parameters: Center({center_x}, {center_y}), Size({width} x {height})"
                )

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
                step_size = 0.01
            else:
                step_size = 1.0

            cutline_controls = [
                "gisaxsInputCenterVerticalValue",
                "gisaxsInputCenterParallelValue",
                "gisaxsInputCutLineVerticalValue",
                "gisaxsInputCutLineParallelValue",
            ]

            for control_name in cutline_controls:
                if hasattr(self.ui, control_name):
                    control = getattr(self.ui, control_name)
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
                unit_suffix = " (q)"
            else:
                unit_suffix = " (px)"

            if hasattr(self.ui, "gisaxsInputCenterVerticalLabel"):
                self.ui.gisaxsInputCenterVerticalLabel.setText(f"Center Vertical{unit_suffix}")

            if hasattr(self.ui, "gisaxsInputCenterParallelLabel"):
                self.ui.gisaxsInputCenterParallelLabel.setText(f"Center Parallel{unit_suffix}")

            if hasattr(self.ui, "gisaxsInputCutLineVerticalLabel"):
                self.ui.gisaxsInputCutLineVerticalLabel.setText(f"Vertical{unit_suffix}")

            if hasattr(self.ui, "gisaxsInputCutLineParallelLabel"):
                self.ui.gisaxsInputCutLineParallelLabel.setText(f"Parallel{unit_suffix}")

        except Exception as e:
            pass

    def _should_show_q_axis(self):
        """No description."""
        try:
            return self.fitting_view_model.get_setting("fitting", "detector.show_q_axis", False)
        except Exception:
            return False

    def _get_cached_q_meshgrids(self):
        """Return the active fitting-view state."""
        try:
            if (
                self.independent_window is not None
                and hasattr(self.independent_window, "_qy_mesh")
                and self.independent_window._qy_mesh is not None
            ):
                return self.independent_window._qy_mesh, self.independent_window._qz_mesh

            if hasattr(self, "current_stack_data") and self.current_stack_data is not None:
                height, width = self.current_stack_data.shape

                pixel_size_x = self.fitting_view_model.get_setting(
                    "fitting", "detector.pixel_size_x", 172.0
                )
                pixel_size_y = self.fitting_view_model.get_setting(
                    "fitting", "detector.pixel_size_y", 172.0
                )
                beam_center_x = self.fitting_view_model.get_setting(
                    "fitting", "detector.beam_center_x", width / 2.0
                )
                beam_center_y = self.fitting_view_model.get_setting(
                    "fitting", "detector.beam_center_y", height / 2.0
                )
                distance = self.fitting_view_model.get_setting(
                    "fitting", "detector.distance", 2565.0
                )
                theta_in_deg = self.fitting_view_model.get_setting("beam", "grazing_angle", 0.4)
                wavelength = self.fitting_view_model.get_setting("beam", "wavelength", 0.1045)

                detector = _scientific_commands(self).q_space.create_detector(
                    image_shape=(height, width),
                    pixel_size_x=pixel_size_x,
                    pixel_size_y=pixel_size_y,
                    beam_center_x=beam_center_x,
                    beam_center_y=beam_center_y,
                    distance=distance,
                    theta_in_deg=theta_in_deg,
                    wavelength=wavelength,
                    crop_params=None,
                )

                return detector.get_qy_qz_meshgrids()

            return None, None

        except Exception as e:
            return None, None
