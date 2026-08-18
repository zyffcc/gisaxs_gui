"""Color Display Controls for fitting presentation."""

from __future__ import annotations

import time

import numpy as np


class ColorDisplayControlsMixin:
    """Own color display controls behavior."""

    def _calculate_vmin_vmax(self, image_data, use_log=True):
        """Calculate the display color range from robust percentiles."""
        try:
            t0 = time.perf_counter()
            if use_log:
                safe_data = np.where(
                    np.isfinite(image_data),
                    np.maximum(image_data, 0.001),
                    np.nan,
                )
                log_data = np.log(safe_data)
                finite_values = log_data[np.isfinite(log_data)]
            else:
                finite_values = np.asarray(image_data)[np.isfinite(image_data)]
            if finite_values.size == 0:
                return None, None
            vmin = np.percentile(finite_values, 1)
            vmax = np.percentile(finite_values, 99)

            print(f"[Timing] autoscale calculation: {(time.perf_counter() - t0) * 1000:.2f} ms")
            return vmin, vmax
        except Exception:
            return None, None

    def _update_vmin_vmax_ui(self, vmin, vmax):
        """No description."""
        try:
            if vmin is not None and vmax is not None:
                self._updating_color_scale_ui = True
                try:
                    if hasattr(self.ui, "gisaxsInputVminValue"):
                        self.ui.gisaxsInputVminValue.setValue(float(vmin))
                    if hasattr(self.ui, "gisaxsInputVmaxValue"):
                        self.ui.gisaxsInputVmaxValue.setValue(float(vmax))
                finally:
                    self._updating_color_scale_ui = False

                self._current_vmin = float(vmin)
                self._current_vmax = float(vmax)
                self._refresh_vmin_vmax_display()
        except Exception:
            pass

    def _get_vmin_vmax_from_ui(self):
        """No description."""
        try:
            vmin = None
            vmax = None

            if hasattr(self.ui, "gisaxsInputVminValue"):
                vmin = self.ui.gisaxsInputVminValue.value()
            if hasattr(self.ui, "gisaxsInputVmaxValue"):
                vmax = self.ui.gisaxsInputVmaxValue.value()

            return vmin, vmax
        except Exception:
            return None, None

    def _handle_color_scale(self, image_data):
        """No description."""
        try:
            is_auto_scale = self._is_auto_scale_enabled()
            is_first_image = not self._has_displayed_image
            is_log = self._is_log_mode_enabled()

            if is_first_image:
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
                self._has_displayed_image = True

            elif is_auto_scale:
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)

            else:
                vmin, vmax = self._get_vmin_vmax_from_ui()
                self._current_vmin = vmin
                self._current_vmax = vmax

        except Exception:
            try:
                is_log = self._is_log_mode_enabled()
                vmin, vmax = self._calculate_vmin_vmax(image_data, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
            except Exception:
                pass

    def _is_auto_scale_enabled(self):
        """No description."""
        return self._get_checkbox_state("gisaxsInputAutoScaleCheckBox", True)

    def _is_log_mode_enabled(self):
        """og"""
        return self._get_checkbox_state("gisaxsInputIntLogCheckBox", True)

    def _on_color_scale_value_committed(self, *args):
        """Apply manually edited vmin/vmax values to all image views."""
        try:
            if self._updating_color_scale_ui or getattr(self, "_initializing", False):
                return

            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmin is None or vmax is None:
                return

            vmin = float(vmin)
            vmax = float(vmax)
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                self.status_updated.emit("Invalid color scale values")
                return
            if vmax <= vmin:
                self.status_updated.emit("Invalid color scale: vmax must be greater than vmin")
                return

            if (
                hasattr(self.ui, "gisaxsInputAutoScaleCheckBox")
                and self.ui.gisaxsInputAutoScaleCheckBox.isChecked()
            ):
                self.ui.gisaxsInputAutoScaleCheckBox.blockSignals(True)
                self.ui.gisaxsInputAutoScaleCheckBox.setChecked(False)
                self.ui.gisaxsInputAutoScaleCheckBox.blockSignals(False)

            self._current_vmin = vmin
            self._current_vmax = vmax
            self._refresh_vmin_vmax_display()
            if self.current_stack_data is not None:
                self._refresh_image_display()
            self.status_updated.emit(f"Color scale updated: Vmin={vmin:.3f}, Vmax={vmax:.3f}")
        except Exception as e:
            self.status_updated.emit(f"Color scale update error: {str(e)}")

    def _on_auto_scale_changed(self):
        """No description."""
        try:
            is_enabled = self._is_auto_scale_enabled()
            self.status_updated.emit(f"AutoScale {'enabled' if is_enabled else 'disabled'}")

            if is_enabled and self.current_stack_data is not None:
                is_log = self._is_log_mode_enabled()
                display_image = self._get_current_display_image()
                if display_image is None:
                    display_image = self.current_stack_data
                vmin, vmax = self._calculate_vmin_vmax(display_image, use_log=is_log)
                if vmin is not None and vmax is not None:
                    self._update_vmin_vmax_ui(vmin, vmax)
                    self._refresh_image_display()
        except Exception as e:
            self.status_updated.emit(f"AutoScale change error: {str(e)}")

    def _on_vmin_value_changed(self):
        """No description."""
        try:
            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmin is not None:
                self._current_vmin = vmin
                if self.current_stack_data is not None:
                    self._refresh_image_display()
        except Exception:
            pass

    def _on_vmax_value_changed(self):
        """No description."""
        try:
            vmin, vmax = self._get_vmin_vmax_from_ui()
            if vmax is not None:
                self._current_vmax = vmax
                if self.current_stack_data is not None:
                    self._refresh_image_display()
        except Exception:
            pass

    def _on_auto_show_changed(self):
        """AutoShow"""
        auto_show = (
            hasattr(self.ui, "gisaxsInputAutoShowCheckBox")
            and self.ui.gisaxsInputAutoShowCheckBox.isChecked()
        )
        self.status_updated.emit(f"AutoShow {'enabled' if auto_show else 'disabled'}")
        try:
            if getattr(self, "load_mode", "Single") == "In-situ":
                if auto_show:
                    self._start_insitu_timer()
                else:
                    self._stop_insitu_timer()
        except Exception:
            pass

    def _on_log_changed(self):
        """Handle log-scale display changes and refresh image views."""
        try:
            is_log = self._is_log_mode_enabled()

            self._refresh_vmin_vmax_display()

            if self.current_stack_data is not None:
                if self._is_auto_scale_enabled():
                    display_image = self._get_current_display_image()
                    if display_image is None:
                        display_image = self.current_stack_data
                    vmin, vmax = self._calculate_vmin_vmax(display_image, use_log=is_log)
                    if vmin is not None and vmax is not None:
                        self._update_vmin_vmax_ui(vmin, vmax)

                self._refresh_image_display()

            self.status_updated.emit(
                f"*** DISPLAY MODE CHANGED TO: {'LOG' if is_log else 'LINEAR'} ***"
            )

        except Exception as e:
            self.status_updated.emit(f"Log mode change error: {str(e)}")

    def _on_fit_display_option_changed(self):
        """No description."""
        try:
            if getattr(self, "_initializing", True):
                return

            if (
                hasattr(self.ui, "fitCurrentDataCheckBox")
                and self.ui.fitCurrentDataCheckBox.isChecked()
            ):
                self._perform_cut()
                self.status_updated.emit("Fit display options changed - Cut results updated")
            else:
                if self.current_1d_data is not None and hasattr(self, "q") and self.q is not None:
                    mode = self.display_mode if hasattr(self, "display_mode") else "normal"
                    self._update_GUI_image(mode)
                    self._update_outside_window(mode)
                    self.status_updated.emit(
                        "Fit display options changed - 1D data display updated"
                    )
                else:
                    self.status_updated.emit("Fit display options changed - no data to update")

        except Exception as e:
            self.status_updated.emit(f"Fit display option change error: {str(e)}")

    def _on_current_data_checkbox_changed(self, checked):
        """No description."""
        try:
            if getattr(self, "_initializing", True):
                return

            if checked:
                if self.current_stack_data is not None:
                    self._perform_cut()
                    self.status_updated.emit("Current Data enabled - Cut operation performed")
                else:
                    self.status_updated.emit(
                        "Current Data enabled - No GISAXS data available for cut operation"
                    )
            else:
                if self.current_1d_data is not None:
                    self.q = self.current_1d_data["q"]
                    self.I = self.current_1d_data["I"]
                    self.data_source = "1d"
                    self.display_mode = "normal"

                    self._update_GUI_image("normal")
                    self._update_outside_window("normal")
                    self.status_updated.emit("Current Data disabled - 1D data restored")
                else:
                    self._clear_fit_graphics_view()
                    if (
                        hasattr(self, "independent_fit_window")
                        and self.independent_fit_window is not None
                        and self.independent_fit_window.isVisible()
                    ):
                        self.independent_fit_window.ax.clear()
                        self.independent_fit_window.canvas.draw()
                    self.status_updated.emit("Current Data disabled - No 1D data available")

        except Exception as e:
            self.status_updated.emit(f"Current Data checkbox change error: {str(e)}")
