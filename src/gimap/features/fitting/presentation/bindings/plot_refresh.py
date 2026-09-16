"""Plot Refresh coordination for the fitting workspace."""

from __future__ import annotations

from ..detector_data_access import analysis_image_for
from ..curve_rendering import CurvePlotSpec, CurveSeries, render_curve_plot


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
        x, y, label = self._get_current_data_for_display()
        if x is not None and y is not None:
            self._plot_data_points_only(x, y, label, self._is_fit_log_x_enabled(),
                                        self._is_fit_log_y_enabled(), self._is_fit_norm_enabled())
            figure = getattr(self, "_current_fit_figure", None)
            if figure is not None and figure.axes:
                figure.axes[0].set_title(f"Data Points Only - {label}")

    def _update_fitting_plot_points_only(self):
        cut = getattr(self, "current_cut_data", None)
        if cut is None:
            return
        x, y = self._legacy_cut_arrays(cut)
        if x is not None and y is not None:
            self._render_legacy_curve_series((CurveSeries(
                self._convert_q_values_for_display(x), y, "Data", "blue",
                marker_size=20, alpha=0.7,
            ),), y_label="Intensity", title="", log_y=self._get_checkbox_state("fitLogYCheckBox", False))

    def _on_fit_log_changed(self):
        """Log-x/Log-y"""
        try:
            self._current_curve_view_state()
            self._update_q_view_hint()
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

    def _on_q_preparation_changed(self, _index=None):
        """Refresh every curve consumer from the same signed-q preparation state."""
        if getattr(self, "_syncing_q_preparation", False):
            return
        self._syncing_q_preparation = True
        try:
            self._current_curve_view_state()
            self._update_q_view_hint()
            self._sync_axis_filter_controls()
            if (
                getattr(self, "data_source", None) == "cut"
                and analysis_image_for(self) is not None
            ):
                self._mark_cut_stale(
                    "q display mode changed; update the cut before fitting"
                )
            self._sync_roi_controls_to_current_display(reset_to_domain=True)
            self._apply_roi_to_data_and_refresh()
            mode = getattr(self, "display_mode", "normal")
            self._update_GUI_image(mode)
            self._update_outside_window(mode)
            self.status_updated.emit(
                "q preparation updated for preview, fitting region and export"
            )
        except Exception as exc:
            self._add_fitting_error(f"Unable to apply q preparation: {exc}")
        finally:
            self._syncing_q_preparation = False

    def _update_q_view_hint(self) -> None:
        label = getattr(self.ui, "fitQViewHintLabel", None)
        if label is None:
            return
        mode_text = {
            "signed": "Signed q",
            "positive": "Positive q",
            "negative": "Negative q",
            "negative_abs": "Negative branch folded to |q|",
            "fold": "±q overlaid as |q|",
            "average": "±q averaged as |q|",
        }.get(self._get_q_view_mode(), "Signed q")
        scale_text = {
            "linear": "linear axis",
            "log": "log axis",
            "symlog": "symmetric-log axis",
        }[self._get_x_axis_scale()]
        label.setText(f"{mode_text} · {scale_text}")

    def _on_normalize_changed(self):
        """Normalize"""
        try:
            self._current_curve_view_state()
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
                and analysis_image_for(self) is not None
            ):
                self._mark_cut_stale(
                    "q-branch selection changed; review the curve and update the cut if needed"
                )
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
        fitting = getattr(self, "fitting_data", None)
        if fitting is None:
            return
        series = []
        cut = getattr(self, "current_cut_data", None)
        if cut is not None:
            x, y = self._legacy_cut_arrays(cut)
            if x is not None and y is not None:
                series.append(CurveSeries(self._convert_q_values_for_display(x), y,
                                          "Data", "blue", marker_size=20, alpha=0.7))
        if isinstance(fitting, dict) and "x" in fitting and "y" in fitting:
            series.append(CurveSeries(self._convert_q_values_for_display(fitting["x"]),
                                      fitting["y"], "Fit", "red", style="line", linewidth=2, alpha=1))
        self._render_legacy_curve_series(tuple(series), y_label="Intensity", title="",
                                         log_y=self._get_checkbox_state("fitLogYCheckBox", False))

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
        window = getattr(self, "independent_fit_window", None)
        if window is None or window.ax is None:
            return
        x, y, label = self._get_current_data_for_display()
        if x is None or y is None:
            return
        values = y.copy()
        normalize = self._is_fit_norm_enabled()
        if normalize and np.max(y) > 0:
            values = y / np.max(y)
        raw_x, plot_x, values, filter_mode = self._filter_q_data_for_independent_display(x, values)
        raw_x, plot_x, values = self._filter_ai_excluded_points_for_display(raw_x, plot_x, values)
        plot_x = self._convert_q_values_for_display(plot_x)
        series = (CurveSeries(plot_x, values, label, "blue", marker_size=30, alpha=0.7),)
        render_curve_plot(window.ax, CurvePlotSpec(
            series, self._build_q_axis_label(filter_mode=filter_mode),
            "Normalized Intensity" if normalize else "Intensity (a.u.)",
            f"Fitting Display Mode - {label}", x_scale=self._get_x_axis_scale(),
            log_y=self._is_fit_log_y_enabled(),
        ))
        window.set_deletable_points(raw_x, plot_x, values)
        window.canvas.draw_idle()

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
        """Refresh the legacy points-only projection without rebuilding its canvas."""
        figure = getattr(self, "_current_fit_figure", None)
        if figure is None:
            return
        plot_y = np.asarray(y_data).copy()
        if normalize:
            maximum = np.max(plot_y)
            if maximum > 0:
                plot_y = plot_y / maximum
        spec = CurvePlotSpec(
            series=(CurveSeries(self._convert_q_values_for_display(x_data), plot_y,
                                data_label, "blue", marker_size=30, alpha=0.7),),
            x_label=self._build_q_axis_label(),
            y_label="Normalized Intensity" if normalize else "Intensity (a.u.)",
            title=f"Fitting Display Mode - {data_label}",
            x_scale=self._get_x_axis_scale(), log_y=log_y,
        )
        axes = figure.axes[0] if figure.axes else figure.add_subplot(111)
        render_curve_plot(axes, spec)
        self._current_fit_canvas.draw_idle()

    def _render_legacy_curve_series(self, series, *, y_label, title, log_y):
        figure = getattr(self, "_current_fit_figure", None)
        if figure is None:
            return
        axes = figure.axes[0] if figure.axes else figure.add_subplot(111)
        render_curve_plot(axes, CurvePlotSpec(series, self._build_q_axis_label(), y_label,
                                             title, x_scale=self._get_x_axis_scale(), log_y=log_y))
        self._current_fit_canvas.draw_idle()

    @staticmethod
    def _legacy_cut_arrays(cut):
        if "x_coords" in cut and "y_intensity" in cut:
            return cut["x_coords"], cut["y_intensity"]
        if "x" in cut and "y" in cut:
            return cut["x"], cut["y"]
        return None, None
