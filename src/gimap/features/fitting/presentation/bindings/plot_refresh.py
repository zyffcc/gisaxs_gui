"""Plot Refresh coordination for the fitting workspace."""

from __future__ import annotations


import numpy as np

from PyQt5.QtCore import QTimer


from ..binding_primitives import (
    COMPONENT_PARAMETER_SCHEMAS,
)


class PlotRefreshMixin:
    """Own plot refresh presentation behavior."""

    def _validate_parameter_retrieval(self, active_shapes, shape_configs):
        """No description."""
        try:
            self._add_fitting_success("=== Parameter Retrieval Validation ===")

            for i, shape in enumerate(active_shapes, 1):
                shape_idx = shape_configs[i - 1]
                current_shape = self.get_particle_shape(shape_idx)

                self._add_fitting_success(
                    f"Shape {i}: {shape} (widget {shape_idx}, actual: {current_shape})"
                )

                shape_display = self._shape_display_name(shape)
                if self._shape_key(shape_display) == "none":
                    continue
                schema = COMPONENT_PARAMETER_SCHEMAS.get(shape_display, [])
                token = self._shape_object_token(shape_display)

                for param_key, suffix, _label, _default, _decimals, _step in schema:
                    control_name = f"fitParticle{token}{suffix}Value_{shape_idx}"

                    if hasattr(self.ui, control_name):
                        control = getattr(self.ui, control_name)
                        if hasattr(control, "value"):
                            value = control.value()
                            self._add_fitting_success(f"  {param_key}: {control_name} = {value}")
                        else:
                            self._add_fitting_error(
                                f"  {param_key}: {control_name} has no 'value' method"
                            )
                    else:
                        self._add_fitting_error(f"  {param_key}: {control_name} not found in UI")

            self._add_fitting_success("Global Parameters:")
            if hasattr(self.ui, "fitBGValue"):
                bg_value = self.ui.fitBGValue.value()
                self._add_fitting_success(f"  background: fitBGValue = {bg_value}")
            else:
                self._add_fitting_error("  fitBGValue not found")

            if hasattr(self.ui, "fitSigmaResValue"):
                sigma_res = self.ui.fitSigmaResValue.value()
                self._add_fitting_success(f"  sigma_res: fitSigmaResValue = {sigma_res}")
            else:
                self._add_fitting_error("  fitSigmaResValue not found")

            if hasattr(self.ui, "fitNuResValue"):
                nu_res = self.ui.fitNuResValue.value()
                self._add_fitting_success(f"  nu_res: fitNuResValue = {nu_res}")
            else:
                self._add_fitting_error("  fitNuResValue not found")

            if hasattr(self.ui, "fitIntResValue"):
                int_res = self.ui.fitIntResValue.value()
                self._add_fitting_success(f"  int_res: fitIntResValue = {int_res}")
            else:
                self._add_fitting_error("  fitIntResValue not found")

            if hasattr(self.ui, "fitKValue"):
                k_value = self.ui.fitKValue.value()
                self._add_fitting_success(f"  k_value: fitKValue = {k_value}")
            else:
                self._add_fitting_error("  fitKValue not found")

            self._add_fitting_success("=== Validation Complete ===")

        except Exception as e:
            self._add_fitting_error(f"Parameter validation failed: {str(e)}")

    def _clear_fitting_data(self):
        """fitting"""
        try:
            if not hasattr(self, "I_fitting") or self.I_fitting is None:
                self.status_updated.emit("No fitting data to clear")
                return

            self.I_fitting = None
            self.has_fitting_data = False

            self.display_mode = "normal"
            self._fitting_mode_active = False

            self._update_GUI_image("normal")
            self._update_outside_window("normal")

            self.status_updated.emit("Fitting data cleared")

        except Exception as e:
            self.status_updated.emit(f"Error clearing fitting data: {str(e)}")

    def _force_update_gui_points_only(self):
        """GUI"""
        try:
            if not hasattr(self.ui, "fitGraphicsView"):
                return

            if not hasattr(self, "_current_fit_figure") or self._current_fit_figure is None:
                return

            if not hasattr(self, "_current_fit_canvas") or self._current_fit_canvas is None:
                return

            x_data, y_data, data_label = self._get_current_data_for_display()
            if x_data is None or y_data is None:
                return

            self._current_fit_figure.clear()
            ax = self._current_fit_figure.add_subplot(111)

            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            plot_y = y_data.copy()
            if normalize:
                max_val = np.max(y_data)
                if max_val > 0:
                    plot_y = y_data / max_val

            x_plot = self._convert_q_values_for_display(x_data)
            ax.scatter(x_plot, plot_y, s=30, alpha=0.7, color="blue", label=data_label, zorder=2)

            x_label = self._build_q_axis_label()
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f"Data Points Only - {data_label}"

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")

            self._current_fit_canvas.draw()

        except Exception as e:
            pass

    def _update_fitting_plot_points_only(self):
        """No description."""
        try:
            if not hasattr(self, "current_cut_data") or self.current_cut_data is None:
                return

            if (
                hasattr(self.ui, "fitGraphicsView")
                and hasattr(self, "_current_fit_figure")
                and self._current_fit_figure is not None
            ):
                self._current_fit_figure.clear()
                ax = self._current_fit_figure.add_subplot(111)

                log_x = self._get_checkbox_state("fitLogXCheckBox", False)
                log_y = self._get_checkbox_state("fitLogYCheckBox", False)

                cut_data = self.current_cut_data
                x_data = None
                y_data = None
                if "x_coords" in cut_data and "y_intensity" in cut_data:
                    x_data = cut_data["x_coords"]
                    y_data = cut_data["y_intensity"]
                elif "x" in cut_data and "y" in cut_data:
                    x_data = cut_data["x"]
                    y_data = cut_data["y"]

                if x_data is not None and y_data is not None:
                    x_plot = self._convert_q_values_for_display(x_data)
                    ax.scatter(x_plot, y_data, c="blue", s=20, alpha=0.7, label="Data")

                    if log_x:
                        ax.set_xscale("log")
                    if log_y:
                        ax.set_yscale("log")

                    ax.set_xlabel(self._build_q_axis_label())
                    ax.set_ylabel("Intensity")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                if hasattr(self, "_current_fit_canvas") and self._current_fit_canvas is not None:
                    self._current_fit_canvas.draw()

        except Exception as e:
            pass

    def _on_fit_log_changed(self):
        """Log-x/Log-y"""
        try:
            mode = self.display_mode if hasattr(self, "display_mode") else "normal"
            try:
                self._sync_roi_controls_to_current_display(reset_to_domain=True)
                self._apply_roi_to_data_and_refresh()
            except Exception:
                pass
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
            self.status_updated.emit("Display log scale updated")
            try:
                QTimer.singleShot(0, self._adjust_roi_bounds_for_log_x)
            except Exception:
                self._adjust_roi_bounds_for_log_x()
        except Exception as e:
            self.status_updated.emit(f"Error updating log scale: {str(e)}")

    def _on_normalize_changed(self):
        """Normalize"""
        try:
            mode = self.display_mode if hasattr(self, "display_mode") else "normal"
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
            self.status_updated.emit("Normalize setting updated")
        except Exception as e:
            self.status_updated.emit(f"Error updating normalize setting: {str(e)}")

    def _on_positive_only_changed(self):
        """No description."""
        try:
            if getattr(self, "_syncing_axis_filter", False):
                return

            previous_mode = getattr(self, "_last_axis_filter_mode", "all")
            self._sync_axis_filter_controls()
            current_filter_mode = self._get_independent_axis_filter_mode()
            self._last_axis_filter_mode = current_filter_mode
            if (
                getattr(self, "data_source", None) == "cut"
                and getattr(self, "current_stack_data", None) is not None
            ):
                self._perform_cut(points_override=self._resolve_cut_points())
                self.status_updated.emit("Cut recalculated for the selected q-axis range")
                return
            try:
                self._sync_roi_controls_to_current_display(
                    reset_to_domain=(previous_mode != current_filter_mode)
                )
                self._apply_roi_to_data_and_refresh()
            except Exception:
                pass
            mode = self.display_mode if hasattr(self, "display_mode") else "normal"
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
            self.status_updated.emit("Display settings synced across main and independent views")
        except Exception as e:
            self.status_updated.emit(f"Error updating display sync: {str(e)}")

    def _update_fitting_plot(self):
        """No description."""
        try:
            if not hasattr(self, "fitting_data") or self.fitting_data is None:
                return

            if (
                hasattr(self.ui, "fitGraphicsView")
                and hasattr(self, "_current_fit_figure")
                and self._current_fit_figure is not None
            ):
                self._current_fit_figure.clear()
                ax = self._current_fit_figure.add_subplot(111)

                log_x = self._get_checkbox_state("fitLogXCheckBox", False)
                log_y = self._get_checkbox_state("fitLogYCheckBox", False)

                if hasattr(self, "current_cut_data") and self.current_cut_data is not None:
                    cut_data = self.current_cut_data
                    if "x_coords" in cut_data and "y_intensity" in cut_data:
                        ax.scatter(
                            self._convert_q_values_for_display(cut_data["x_coords"]),
                            cut_data["y_intensity"],
                            c="blue",
                            s=20,
                            alpha=0.7,
                            label="Data",
                        )
                    elif "x" in cut_data and "y" in cut_data:
                        ax.scatter(
                            self._convert_q_values_for_display(cut_data["x"]),
                            cut_data["y"],
                            c="blue",
                            s=20,
                            alpha=0.7,
                            label="Data",
                        )

                fitting_data = self.fitting_data
                if isinstance(fitting_data, dict) and "x" in fitting_data and "y" in fitting_data:
                    ax.plot(
                        self._convert_q_values_for_display(fitting_data["x"]),
                        fitting_data["y"],
                        "r-",
                        linewidth=2,
                        label="Fit",
                    )

                if log_x:
                    ax.set_xscale("log")
                if log_y:
                    ax.set_yscale("log")

                ax.set_xlabel(self._build_q_axis_label())
                ax.set_ylabel("Intensity")
                ax.legend()
                ax.grid(True, alpha=0.3)

                if hasattr(self, "_current_fit_canvas") and self._current_fit_canvas is not None:
                    self._current_fit_canvas.draw()

        except Exception:
            pass

    def _update_fitting_mode_displays_without_line(self):
        """No description."""
        try:
            self._update_gui_points_only()

            if (
                hasattr(self, "independent_fit_window")
                and self.independent_fit_window is not None
                and self.independent_fit_window.isVisible()
            ):
                self._update_external_window_points_only()

        except Exception as e:
            pass

    def _update_gui_points_only(self):
        """No description."""
        try:
            if not hasattr(self.ui, "fitGraphicsView"):
                return

            x_data, y_data, data_label = self._get_current_data_for_display()
            if x_data is None or y_data is None:
                return

            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            self._plot_data_points_only(x_data, y_data, data_label, log_x, log_y, normalize)

        except Exception as e:
            pass

    def _update_external_window_points_only(self):
        """No description."""
        try:
            if not hasattr(self.independent_fit_window, "ax"):
                return

            x_data, y_data, data_label = self._get_current_data_for_display()
            if x_data is None or y_data is None:
                return

            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            ax = self.independent_fit_window.ax
            ax.clear()

            plot_y = y_data.copy()
            if normalize:
                max_val = np.max(y_data)
                if max_val > 0:
                    plot_y = y_data / max_val

            x_raw, x_plot, plot_y, filter_mode = self._filter_q_data_for_independent_display(
                x_data, plot_y
            )
            x_raw, x_plot, plot_y = self._filter_ai_excluded_points_for_display(
                x_raw, x_plot, plot_y
            )
            x_plot = self._convert_q_values_for_display(x_plot)
            if x_plot.size == 0 or plot_y is None or plot_y.size == 0:
                return

            ax.scatter(x_plot, plot_y, s=30, alpha=0.7, color="blue", label=data_label, zorder=2)

            x_label = self._build_q_axis_label(filter_mode=filter_mode)
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f"Fitting Display Mode - {data_label}"

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(1.8)

            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")

            if hasattr(self.independent_fit_window, "canvas"):
                try:
                    if hasattr(self.independent_fit_window, "set_deletable_points"):
                        self.independent_fit_window.set_deletable_points(x_raw, x_plot, plot_y)
                except Exception:
                    pass
                self.independent_fit_window.canvas.draw()

        except Exception as e:
            pass

    def _get_current_data_for_display(self):
        """No description."""
        try:
            if (
                hasattr(self.ui, "fitCurrentDataCheckBox")
                and self.ui.fitCurrentDataCheckBox.isChecked()
            ):
                if hasattr(self, "current_cut_data") and self.current_cut_data is not None:
                    return (
                        np.array(self.current_cut_data["x_coords"]),
                        np.array(self.current_cut_data["y_intensity"]),
                        "Cut Data",
                    )
            else:
                if hasattr(self, "current_1d_data") and self.current_1d_data is not None:
                    return (
                        np.array(self.current_1d_data["q"]),
                        np.array(self.current_1d_data["I"]),
                        "1D File Data",
                    )

            return None, None, ""

        except Exception as e:
            return None, None, ""

    def _plot_data_points_only(self, x_data, y_data, data_label, log_x, log_y, normalize):
        """UI"""
        try:
            if not hasattr(self.ui, "fitGraphicsView"):
                return

            # Use the existing fitting GUI figure and canvas
            if hasattr(self, "_current_fit_figure") and self._current_fit_figure is not None:
                self._current_fit_figure.clear()
                ax = self._current_fit_figure.add_subplot(111)

                # Processing data
                plot_y = y_data.copy()
                if normalize:
                    max_val = np.max(y_data)
                    if max_val > 0:
                        plot_y = y_data / max_val

                # Plotting data points
                x_plot = self._convert_q_values_for_display(x_data)
                ax.scatter(
                    x_plot, plot_y, s=30, alpha=0.7, color="blue", label=data_label, zorder=2
                )

                # Setting up labels and styles
                x_label = self._build_q_axis_label()
                y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
                title = f"Fitting Display Mode - {data_label}"

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Setting logarithmic coordinates
                if log_x:
                    ax.set_xscale("log")
                if log_y:
                    ax.set_yscale("log")

                # Refresh Canvas
                if hasattr(self, "_current_fit_canvas") and self._current_fit_canvas is not None:
                    self._current_fit_canvas.draw()

        except Exception as e:
            pass
