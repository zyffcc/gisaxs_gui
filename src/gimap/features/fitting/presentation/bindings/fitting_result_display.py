"""Fitting Result Display for fitting presentation."""

from __future__ import annotations


import numpy as np

from PyQt5.QtCore import Qt


from src.gimap.app.presentation.responsive_layout import (
    move_window_to_cursor_screen,
)

from ..binding_primitives import (
    IndependentFitWindow,
    _qobject_is_alive,
    _scientific_commands,
    is_matplotlib_available,
)


class FittingResultDisplayMixin:
    """Own fitting result display behavior."""

    def _plot_fitting_result(self, q_data, intensity_data, active_shapes):
        """No description."""
        try:
            if not hasattr(self.ui, "fitGraphicsView"):
                return

            if not is_matplotlib_available():
                self._add_fitting_error("Matplotlib not available for plotting")
                return

            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            original_x_data = None
            original_y_data = None
            data_label = ""

            if (
                hasattr(self.ui, "fitCurrentDataCheckBox")
                and self.ui.fitCurrentDataCheckBox.isChecked()
            ):
                if hasattr(self, "current_cut_data") and self.current_cut_data is not None:
                    original_x_data = np.array(self.current_cut_data["x_coords"])
                    original_y_data = np.array(self.current_cut_data["y_intensity"])
                    data_label = "Cut Data"
            else:
                if hasattr(self, "current_1d_data") and self.current_1d_data is not None:
                    original_x_data = np.array(self.current_1d_data["q"])
                    original_y_data = np.array(self.current_1d_data["I"])
                    data_label = "1D File Data"

            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return

            fig = Figure(figsize=(9.6, 7.2), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            fitting_y_data = np.array(intensity_data)
            plot_original_y = original_y_data.copy() if original_y_data is not None else None
            norm_divisor = None

            if normalize and original_y_data is not None:
                max_original = np.max(original_y_data)
                if max_original > 0:
                    norm_divisor = max_original
                    plot_original_y = original_y_data / max_original
                    fitting_y_data = fitting_y_data / max_original

            original_x_plot = (
                self._convert_q_values_for_display(original_x_data)
                if original_x_data is not None
                else None
            )
            fitting_x_plot = self._convert_q_values_for_display(q_data)

            if original_x_plot is not None and plot_original_y is not None:
                ax.scatter(
                    original_x_plot,
                    plot_original_y,
                    s=20,
                    alpha=0.7,
                    color="blue",
                    label=data_label,
                    zorder=2,
                )

            ax.plot(
                fitting_x_plot,
                fitting_y_data,
                color="red",
                linewidth=2,
                label=f"Fitting ({', '.join(active_shapes)})",
                zorder=3,
            )

            x_label = (
                self._build_q_axis_label()
                if "q" in str(original_x_data).lower() or len(q_data) > 0
                else "Position"
            )
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f"Manual Fitting Result - {', '.join(active_shapes)}"

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(1.8)
            ax.tick_params(axis="both", which="both", width=1.6, labelsize=12)

            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            self._apply_fit_y_axis_limits(
                ax,
                experimental_y=plot_original_y,
                fitting_y=fitting_y_data,
                log_y=log_y,
            )

            self._draw_roi_guides_if_active(ax)

            fig.tight_layout()

            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)

            self._current_fit_figure = fig
            self._current_fit_canvas = canvas

            if not hasattr(self, "current_fitting_data"):
                self.current_fitting_data = {}

            self.current_fitting_data = {
                "q": q_data.copy(),
                "I_fitted": intensity_data.copy(),
                "shapes": active_shapes.copy(),
                "title": title,
                "original_x": original_x_data.copy() if original_x_data is not None else None,
                "original_y": original_y_data.copy() if original_y_data is not None else None,
                "data_label": data_label,
            }

            self._add_fitting_success(f"Fitting result plotted for shapes: {active_shapes}")

        except Exception as e:
            self._add_fitting_error(f"Failed to plot fitting result: {str(e)}")

    def _get_last_fitting_spec_and_params(self, fallback_shapes=None):
        """Return the last fitting shapes and ordered parameter list.

        Prefer metadata stored in ``self.fitting``; fall back to active UI widgets when metadata is unavailable.
        """
        try:
            import re

            shapes = None
            param_dict = None
            if isinstance(getattr(self, "fitting", None), dict):
                meta = self.fitting.get("meta", {})
                shapes = meta.get("shapes")
                param_dict = meta.get("params")
            if shapes and param_dict:
                tmpl = _scientific_commands(self).model.parameter_names(shapes)
                params_list = []
                ok = True
                for name in tmpl:
                    if name in param_dict:
                        params_list.append(float(param_dict[name]))
                    else:
                        ok = False
                        break
                if ok:
                    return shapes, params_list

            act_shapes, act_idx = self._collect_active_particles()
            if not act_shapes:
                return (fallback_shapes, None) if fallback_shapes else (None, None)

            self._last_active_particle_ids = act_idx.copy()
            tmpl = _scientific_commands(self).model.parameter_names(act_shapes)
            params_list = []

            default_map = {
                "Int": 1.0,
                "R": 10.0,
                "sigma_R": 0.1,
                "D": 100.0,
                "sigma_D": 0.1,
                "h": 20.0,
                "sigma_h": 0.1,
                "BG": 0.0,
                "sigma_Res": 0.1,
                "nu_Res": 5.0,
                "int_Res": 0.0,
                "k": 1.0,
            }
            global_widget_map = {
                "BG": ("fitBGValue", "background"),
                "sigma_Res": ("fitSigmaResValue", "sigma_res"),
                "nu_Res": ("fitNuResValue", "nu_res"),
                "int_Res": ("fitIntResValue", "int_res"),
                "k": ("fitKValue", "k_value"),
            }

            for template_name in tmpl:
                match = re.match(r"^(.*?)(\d+)$", str(template_name))
                if match:
                    base_name = match.group(1)
                    seq_index = int(match.group(2))
                    widget_id = act_idx[seq_index - 1] if 1 <= seq_index <= len(act_idx) else None
                    default_value = default_map.get(base_name, 0.0)
                    if widget_id is None:
                        params_list.append(float(default_value))
                    else:
                        params_list.append(
                            float(self._get_particle_parameter(widget_id, base_name, default_value))
                        )
                else:
                    widget_name, global_key = global_widget_map.get(
                        str(template_name), (None, None)
                    )
                    default_value = default_map.get(str(template_name), 0.0)
                    if widget_name and hasattr(self.ui, widget_name):
                        params_list.append(float(getattr(self.ui, widget_name).value()))
                    elif global_key and hasattr(self, "get_global_parameter"):
                        params_list.append(float(self.get_global_parameter(global_key)))
                    else:
                        params_list.append(float(default_value))

            return act_shapes, [float(x) for x in params_list]
        except Exception:
            return (fallback_shapes, None) if fallback_shapes else (None, None)

    def _on_component_checkbox_changed(self, *_):
        """Refresh fitting displays when component visibility changes in fitting mode."""
        try:
            if not self._is_in_fitting_mode():
                return
            self._update_GUI_image("fitting")
            self._update_outside_window("fitting")
        except Exception:
            pass

    def _show_fitting_in_external_window(self, q_data, intensity_data, active_shapes):
        """No description."""
        try:
            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            original_x_data = None
            original_y_data = None
            data_label = ""

            if (
                hasattr(self.ui, "fitCurrentDataCheckBox")
                and self.ui.fitCurrentDataCheckBox.isChecked()
            ):
                if hasattr(self, "current_cut_data") and self.current_cut_data is not None:
                    original_x_data = np.array(self.current_cut_data["x_coords"])
                    original_y_data = np.array(self.current_cut_data["y_intensity"])
                    data_label = "Cut Data"
            else:
                if hasattr(self, "current_1d_data") and self.current_1d_data is not None:
                    original_x_data = np.array(self.current_1d_data["q"])
                    original_y_data = np.array(self.current_1d_data["I"])
                    data_label = "1D File Data"

            if not _qobject_is_alive(self.independent_fit_window):
                self.independent_fit_window = None

            if self.independent_fit_window is None or not self.independent_fit_window.isVisible():
                self.independent_fit_window = IndependentFitWindow(self.main_window)
                self.independent_fit_window.setAttribute(Qt.WA_DeleteOnClose, True)
                self.independent_fit_window.destroyed.connect(
                    lambda _obj=None: setattr(self, "independent_fit_window", None)
                )

                self.independent_fit_window.status_updated.connect(self.status_updated.emit)
                self.independent_fit_window.show_positive_cb.toggled.connect(
                    self._on_positive_only_changed
                )
                if hasattr(self.independent_fit_window, "show_negative_cb"):
                    self.independent_fit_window.show_negative_cb.toggled.connect(
                        self._on_positive_only_changed
                    )
                if hasattr(self.independent_fit_window, "q_unit_combo"):
                    self.independent_fit_window.q_unit_combo.currentTextChanged.connect(
                        self._on_positive_only_changed
                    )
                if hasattr(self.independent_fit_window, "y_range_combo"):
                    self.independent_fit_window.y_range_combo.currentTextChanged.connect(
                        self._on_positive_only_changed
                    )
                if hasattr(self.independent_fit_window, "input_point_delete_requested"):
                    self.independent_fit_window.input_point_delete_requested.connect(
                        self._exclude_ai_input_point_from_plot
                    )
                try:
                    self._sync_axis_filter_controls()
                except Exception:
                    pass

            x_label = (
                self._build_q_axis_label()
                if "q" in str(original_x_data).lower() or len(q_data) > 0
                else "Position"
            )
            y_label = "Normalized Intensity" if normalize else "Intensity (a.u.)"
            title = f"Manual Fitting Result - {', '.join(active_shapes)}"

            self._update_independent_window_with_fitting(
                original_x_data,
                original_y_data,
                data_label,
                q_data,
                intensity_data,
                active_shapes,
                x_label,
                y_label,
                title,
                log_x,
                log_y,
                normalize,
            )

            if not self.independent_fit_window.isVisible():
                move_window_to_cursor_screen(self.independent_fit_window)
            self.independent_fit_window.show()
            self.independent_fit_window.raise_()
            self.independent_fit_window.activateWindow()

            if hasattr(self.independent_fit_window, "canvas"):
                self.independent_fit_window.canvas.setFocus()

            self._add_fitting_success(f"Fitting result displayed in external window")
            return True

        except Exception as e:
            self._add_fitting_error(f"Failed to show fitting in external window: {str(e)}")
            return False

    def _update_independent_window_with_fitting(
        self,
        original_x,
        original_y,
        data_label,
        fitting_x,
        fitting_y,
        shapes,
        x_label,
        y_label,
        title,
        log_x,
        log_y,
        normalize,
    ):
        """No description."""
        try:
            if not _qobject_is_alive(self.independent_fit_window) or not hasattr(
                self.independent_fit_window, "ax"
            ):
                self.independent_fit_window = None
                return

            ax = self.independent_fit_window.ax
            if ax is None:
                return
            ax.clear()

            fitting_x = (
                np.asarray(fitting_x, dtype=float).reshape(-1)
                if fitting_x is not None
                else np.array([], dtype=float)
            )
            plot_fitting_y = (
                np.asarray(fitting_y, dtype=float).reshape(-1)
                if fitting_y is not None
                else np.array([], dtype=float)
            )
            nf = min(fitting_x.size, plot_fitting_y.size)
            fitting_x, plot_fitting_y = fitting_x[:nf], plot_fitting_y[:nf]
            if nf > 0:
                mask = np.isfinite(fitting_x) & np.isfinite(plot_fitting_y)
                fitting_x, plot_fitting_y = fitting_x[mask], plot_fitting_y[mask]

            if original_x is not None and original_y is not None:
                original_x = np.asarray(original_x, dtype=float).reshape(-1)
                plot_original_y = np.asarray(original_y, dtype=float).reshape(-1)
                no = min(original_x.size, plot_original_y.size)
                original_x, plot_original_y = original_x[:no], plot_original_y[:no]
                if no > 0:
                    mask = np.isfinite(original_x) & np.isfinite(plot_original_y)
                    original_x, plot_original_y = original_x[mask], plot_original_y[mask]
            else:
                original_x = None
                plot_original_y = None

            if (
                original_x is None or plot_original_y is None or original_x.size == 0
            ) and fitting_x.size == 0:
                self.status_updated.emit("No plottable fitting data for independent window.")
                return

            if normalize and original_y is not None:
                max_original = (
                    np.nanmax(plot_original_y)
                    if plot_original_y is not None and plot_original_y.size
                    else 0.0
                )
                if max_original > 0:
                    plot_original_y = plot_original_y / max_original
                    plot_fitting_y = plot_fitting_y / max_original

            filter_mode = self._get_independent_axis_filter_mode()
            original_x_plot = original_x
            fitting_x_plot = fitting_x
            original_x_raw_for_delete = None
            if original_x is not None and plot_original_y is not None:
                original_x, plot_original_y = self._filter_ai_excluded_points_for_display(
                    original_x, plot_original_y
                )
                original_x_raw_for_delete, original_x_plot, plot_original_y, filter_mode = (
                    self._filter_q_data_for_independent_display(original_x, plot_original_y)
                )
            if fitting_x is not None and plot_fitting_y is not None:
                _, fitting_x_plot, plot_fitting_y, _ = self._filter_q_data_for_independent_display(
                    fitting_x, plot_fitting_y
                )

            original_x_plot = self._convert_q_values_for_display(original_x_plot)
            fitting_x_plot = self._convert_q_values_for_display(fitting_x_plot)

            if log_x:
                if original_x_plot is not None and plot_original_y is not None:
                    mask = np.asarray(original_x_plot) > 0
                    original_x_plot = np.asarray(original_x_plot)[mask]
                    plot_original_y = np.asarray(plot_original_y)[mask]
                    if original_x_raw_for_delete is not None:
                        original_x_raw_for_delete = np.asarray(original_x_raw_for_delete)[mask]
                if fitting_x_plot is not None and plot_fitting_y is not None:
                    mask = np.asarray(fitting_x_plot) > 0
                    fitting_x_plot = np.asarray(fitting_x_plot)[mask]
                    plot_fitting_y = np.asarray(plot_fitting_y)[mask]

            if log_y:
                if original_x_plot is not None and plot_original_y is not None:
                    mask = np.asarray(plot_original_y) > 0
                    original_x_plot = np.asarray(original_x_plot)[mask]
                    plot_original_y = np.asarray(plot_original_y)[mask]
                    if original_x_raw_for_delete is not None:
                        original_x_raw_for_delete = np.asarray(original_x_raw_for_delete)[mask]
                if fitting_x_plot is not None and plot_fitting_y is not None:
                    mask = np.asarray(plot_fitting_y) > 0
                    fitting_x_plot = np.asarray(fitting_x_plot)[mask]
                    plot_fitting_y = np.asarray(plot_fitting_y)[mask]

            if original_x is not None and plot_original_y is not None and len(original_x_plot) > 0:
                ax.scatter(
                    original_x_plot,
                    plot_original_y,
                    s=30,
                    alpha=0.7,
                    color="blue",
                    label=data_label,
                    zorder=2,
                )

            if fitting_x is not None and plot_fitting_y is not None and len(fitting_x_plot) > 0:
                ax.plot(
                    fitting_x_plot,
                    plot_fitting_y,
                    color="red",
                    linewidth=2.5,
                    label=f"Fitting ({', '.join(shapes)})",
                    zorder=3,
                )

            plot_x_label = x_label
            if isinstance(x_label, str) and "q" in x_label.lower():
                plot_x_label = self._build_q_axis_label(filter_mode=filter_mode)
            ax.set_xlabel(plot_x_label)
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
            self._apply_fit_y_axis_limits(
                ax,
                experimental_y=plot_original_y,
                fitting_y=plot_fitting_y,
                log_y=log_y,
            )

            try:
                if (
                    original_x_raw_for_delete is not None
                    and plot_original_y is not None
                    and hasattr(self.independent_fit_window, "set_deletable_points")
                ):
                    self.independent_fit_window.set_deletable_points(
                        original_x_raw_for_delete,
                        original_x_plot,
                        plot_original_y,
                    )
            except Exception:
                pass

            if hasattr(self.independent_fit_window, "canvas"):
                self.independent_fit_window.canvas.draw()

        except Exception as e:
            self._add_fitting_error(f"Failed to update independent window with fitting: {str(e)}")
