"""Detector Display for fitting presentation."""

from __future__ import annotations


import numpy as np


from ..binding_primitives import (
    _scientific_commands,
)


class DetectorDisplayMixin:
    """Own detector display behavior."""

    def _update_GUI_image(self, mode):
        """UI"""
        try:
            if not self._has_valid_data():
                return

            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()
            filter_mode = self._get_independent_axis_filter_mode()
            positive_only = filter_mode == "positive"
            negative_only = filter_mode == "negative"

            q_data, I_data = self._get_roi_active_arrays()
            if q_data is None or I_data is None:
                return
            q_data, I_data = self._filter_ai_excluded_points_for_display(q_data, I_data)

            q_data, q_plot, I_data, _ = self._filter_q_data_for_independent_display(q_data, I_data)
            if q_data.size == 0 or I_data is None or I_data.size == 0:
                return
            q_data_display = self._convert_q_values_for_display(q_data)
            q_plot = self._convert_q_values_for_display(q_plot)

            norm_factor = 1.0
            if normalize:
                max_I = np.max(I_data) if I_data.size > 0 else 0.0
                if max_I > 0:
                    norm_factor = float(max_I)
                    I_data = I_data / norm_factor

            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return

            fig = Figure(figsize=(9.6, 7.2), dpi=80)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            fitting_y_for_limits = None
            extra_y_for_limits = []

            if mode == "normal" and log_x and not positive_only and not negative_only:
                positive_mask = q_data > 0
                negative_mask = q_data < 0
                zero_mask = q_data == 0

                if np.any(positive_mask):
                    ax.plot(
                        q_data_display[positive_mask],
                        I_data[positive_mask],
                        "o-",
                        color="blue",
                        markersize=4,
                        linewidth=1,
                        alpha=0.8,
                        label=f"{self.data_source.upper()} Data (q>0)"
                        if self.data_source
                        else "Data (q>0)",
                        zorder=2,
                    )

                if np.any(negative_mask):
                    ax.plot(
                        np.abs(q_data_display[negative_mask]),
                        I_data[negative_mask],
                        "o-",
                        color="red",
                        markersize=4,
                        linewidth=1,
                        alpha=0.8,
                        label=f"{self.data_source.upper()} Data (q<0, |q|)"
                        if self.data_source
                        else "Data (q<0, |q|)",
                        zorder=2,
                    )

                if np.any(zero_mask):
                    ax.plot(
                        q_data_display[zero_mask],
                        I_data[zero_mask],
                        "o",
                        color="green",
                        markersize=6,
                        alpha=0.8,
                        label=f"{self.data_source.upper()} Data (q=0)"
                        if self.data_source
                        else "Data (q=0)",
                        zorder=3,
                    )
            else:
                ax.scatter(
                    q_plot,
                    I_data,
                    s=30,
                    alpha=0.7,
                    color="blue",
                    label=f"{self.data_source.upper()} Data" if self.data_source else "Data",
                    zorder=2,
                )
            self._draw_roi_guides_if_active(ax)

            try:
                show_bg = self._get_checkbox_state("fitBGShowCheckBox", False)
                show_res = self._get_checkbox_state("fitResShowCheckBox", False)
                particle_flags = self._get_particle_sequence_flags()
                show_any = show_bg or show_res or any(particle_flags.values())
            except Exception:
                particle_flags = {}
                show_any = False

            norm_divisor = norm_factor if normalize and norm_factor > 0 else None
            if mode == "fitting" and show_any:
                shapes, params_list = self._get_last_fitting_spec_and_params()
                if shapes and params_list:
                    try:
                        q_model = self._convert_q_values_for_model(q_data, source=self.data_source)
                        comp = _scientific_commands(self).model.components(
                            shapes, q_model, params_list
                        )
                        # BG
                        if show_bg and comp.get("BG_total") is not None:
                            y_bg = (
                                comp["BG_total"] / norm_divisor
                                if norm_divisor
                                else comp["BG_total"]
                            )
                            ax.plot(
                                q_plot,
                                y_bg,
                                linestyle="--",
                                color="#666666",
                                linewidth=1.5,
                                label="bg",
                                zorder=2,
                            )
                            extra_y_for_limits.append(y_bg)
                        # Resolution function
                        if show_res and comp.get("resolution") is not None:
                            y_res = (
                                comp["resolution"] / norm_divisor
                                if norm_divisor
                                else comp["resolution"]
                            )
                            ax.plot(
                                q_plot,
                                y_res,
                                linestyle="--",
                                color="#8E44AD",
                                linewidth=1.5,
                                label="Res.",
                                zorder=2,
                            )
                            extra_y_for_limits.append(y_res)
                        # Particles
                        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
                        for item in comp.get("particles", []):
                            idx = int(item.get("index", 0))
                            if particle_flags.get(idx, False):
                                yv = item.get("I")
                                if yv is not None:
                                    shape_name = str(item.get("shape", "Particle")).capitalize()
                                    widget_id = self._sequence_index_to_widget_id(idx)
                                    color_key = widget_id if widget_id is not None else idx
                                    color = (
                                        colors[(color_key - 1) % len(colors)]
                                        if color_key
                                        else colors[(idx - 1) % len(colors)]
                                    )
                                    yv_plot = yv / norm_divisor if norm_divisor else yv
                                    label_id = (
                                        f"{shape_name} {widget_id}"
                                        if widget_id is not None
                                        else f"{shape_name} {idx}"
                                    )
                                    ax.plot(
                                        q_plot,
                                        yv_plot,
                                        linestyle="--",
                                        color=color,
                                        linewidth=1.5,
                                        label=label_id,
                                        zorder=2,
                                    )
                                    extra_y_for_limits.append(yv_plot)
                        ax.legend()
                    except Exception:
                        pass

            if mode == "fitting" and self.has_fitting_data and self.I_fitting is not None:
                I_fitting_arr = np.asarray(self.I_fitting)
                q_full = np.asarray(self.q)

                mask_full = np.isfinite(q_full)
                if self._roi_active():
                    mask_full &= (q_full >= self._roi_min) & (q_full <= self._roi_max)
                if positive_only:
                    mask_full &= q_full > 0
                elif negative_only:
                    mask_full &= q_full < 0

                q_fit_raw = q_full[mask_full]
                I_fitting_data = I_fitting_arr[mask_full]
                _, q_fit_plot, I_fitting_data, _ = self._filter_q_data_for_independent_display(
                    q_fit_raw, I_fitting_data
                )
                q_fit_plot = self._convert_q_values_for_display(q_fit_plot)

                if normalize and norm_factor > 0 and I_fitting_data.size > 0:
                    I_fitting_data = I_fitting_data / norm_factor

                plot_len = min(len(q_fit_plot), len(I_fitting_data))
                if plot_len > 0:
                    fitting_y_for_limits = I_fitting_data[:plot_len]
                    ax.plot(
                        q_fit_plot[:plot_len],
                        I_fitting_data[:plot_len],
                        color="red",
                        linewidth=2,
                        label="Fitting",
                        zorder=3,
                    )

            x_label = self._build_q_axis_label(filter_mode=filter_mode)
            if (
                mode == "normal"
                and log_x
                and not positive_only
                and not negative_only
                and np.any(q_data < 0)
            ):
                x_label = self._build_q_axis_label(filter_mode="all", absolute=True)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Normalized Intensity" if normalize else "Intensity (a.u.)")
            ax.set_title(
                f"{mode.capitalize()} Mode - {self.data_source.upper() if self.data_source else 'Data'}"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

            self._apply_log_scales(ax, log_x, log_y)
            self._apply_fit_y_axis_limits(
                ax,
                experimental_y=I_data,
                fitting_y=fitting_y_for_limits,
                extra_y_values=extra_y_for_limits,
                log_y=log_y,
            )

            fig.tight_layout()

            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(self.ui.fitGraphicsView, proxy_widget, keep_aspect=True)

            self._current_fit_figure = fig
            self._current_fit_canvas = canvas

        except Exception:
            pass

    def _update_outside_window(self, mode):
        """No description."""
        try:
            if (
                not hasattr(self, "independent_fit_window")
                or self.independent_fit_window is None
                or not self.independent_fit_window.isVisible()
            ):
                return

            log_x = self._is_fit_log_x_enabled()
            log_y = self._is_fit_log_y_enabled()
            normalize = self._is_fit_norm_enabled()

            filter_mode = self._get_independent_axis_filter_mode()
            positive_only = filter_mode == "positive"
            negative_only = filter_mode == "negative"

            q_data, I_data = self._get_roi_active_arrays()

            if q_data is None or I_data is None or len(q_data) == 0 or len(I_data) == 0:
                return
            q_data, I_data = self._filter_ai_excluded_points_for_display(q_data, I_data)

            q_data, q_plot, I_data, _ = self._filter_q_data_for_independent_display(q_data, I_data)
            if q_data.size == 0 or I_data is None or I_data.size == 0:
                return
            q_data_display = self._convert_q_values_for_display(q_data)
            q_plot = self._convert_q_values_for_display(q_plot)

            norm_factor = 1.0
            if normalize:
                max_I = np.max(I_data) if I_data.size > 0 else 0.0
                if max_I > 0:
                    norm_factor = float(max_I)
                    I_data = I_data / norm_factor

            ax = self.independent_fit_window.ax
            ax.clear()
            fitting_y_for_limits = None
            extra_y_for_limits = []

            if mode == "normal" and log_x and not positive_only and not negative_only:
                positive_mask = q_data > 0
                negative_mask = q_data < 0
                zero_mask = q_data == 0

                if np.any(positive_mask):
                    ax.plot(
                        q_data_display[positive_mask],
                        I_data[positive_mask],
                        "o-",
                        color="blue",
                        markersize=4,
                        linewidth=1,
                        alpha=0.8,
                        label=f"{self.data_source.upper()} Data (q>0)"
                        if self.data_source
                        else "Data (q>0)",
                        zorder=2,
                    )

                if np.any(negative_mask):
                    ax.plot(
                        np.abs(q_data_display[negative_mask]),
                        I_data[negative_mask],
                        "o-",
                        color="red",
                        markersize=4,
                        linewidth=1,
                        alpha=0.8,
                        label=f"{self.data_source.upper()} Data (q<0, |q|)"
                        if self.data_source
                        else "Data (q<0, |q|)",
                        zorder=2,
                    )

                if np.any(zero_mask):
                    ax.plot(
                        q_data_display[zero_mask],
                        I_data[zero_mask],
                        "o",
                        color="green",
                        markersize=6,
                        alpha=0.8,
                        label=f"{self.data_source.upper()} Data (q=0)"
                        if self.data_source
                        else "Data (q=0)",
                        zorder=3,
                    )
            else:
                ax.scatter(
                    q_plot,
                    I_data,
                    s=30,
                    alpha=0.7,
                    color="blue",
                    label=f"{self.data_source.upper()} Data" if self.data_source else "Data",
                    zorder=2,
                )
            self._draw_roi_guides_if_active(ax)

            try:
                show_bg = self._get_checkbox_state("fitBGShowCheckBox", False)
                show_res = self._get_checkbox_state("fitResShowCheckBox", False)
                particle_flags = self._get_particle_sequence_flags()
                show_any = show_bg or show_res or any(particle_flags.values())
            except Exception:
                particle_flags = {}
                show_any = False

            if mode == "fitting" and show_any:
                shapes, params_list = self._get_last_fitting_spec_and_params()
                if shapes and params_list:
                    try:
                        q_model = self._convert_q_values_for_model(q_data, source=self.data_source)
                        comp = _scientific_commands(self).model.components(
                            shapes, q_model, params_list
                        )
                        norm_divisor = norm_factor if normalize and norm_factor > 0 else None
                        # BG
                        if show_bg and comp.get("BG_total") is not None:
                            y_bg = (
                                comp["BG_total"] / norm_divisor
                                if norm_divisor
                                else comp["BG_total"]
                            )
                            ax.plot(
                                q_plot,
                                y_bg,
                                linestyle="--",
                                color="#666666",
                                linewidth=1.5,
                                label="bg",
                                zorder=2,
                            )
                            extra_y_for_limits.append(y_bg)
                        # Resolution function
                        if show_res and comp.get("resolution") is not None:
                            y_res = (
                                comp["resolution"] / norm_divisor
                                if norm_divisor
                                else comp["resolution"]
                            )
                            ax.plot(
                                q_plot,
                                y_res,
                                linestyle="--",
                                color="#8E44AD",
                                linewidth=1.5,
                                label="Res.",
                                zorder=2,
                            )
                            extra_y_for_limits.append(y_res)
                        # Particles
                        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
                        for item in comp.get("particles", []):
                            idx = int(item.get("index", 0))
                            if particle_flags.get(idx, False):
                                yv = item.get("I")
                                if yv is not None:
                                    shape_name = str(item.get("shape", "Particle")).capitalize()
                                    widget_id = self._sequence_index_to_widget_id(idx)
                                    color_key = widget_id if widget_id is not None else idx
                                    color = (
                                        colors[(color_key - 1) % len(colors)]
                                        if color_key
                                        else colors[(idx - 1) % len(colors)]
                                    )
                                    yv_plot = yv / norm_divisor if norm_divisor else yv
                                    label_id = (
                                        f"{shape_name} {widget_id}"
                                        if widget_id is not None
                                        else f"{shape_name} {idx}"
                                    )
                                    ax.plot(
                                        q_plot,
                                        yv_plot,
                                        linestyle="--",
                                        color=color,
                                        linewidth=1.5,
                                        label=label_id,
                                        zorder=2,
                                    )
                                    extra_y_for_limits.append(yv_plot)
                        ax.legend()
                    except Exception:
                        pass

            if mode == "fitting" and self.has_fitting_data and self.I_fitting is not None:
                I_fitting_arr = np.asarray(self.I_fitting)
                q_full = np.asarray(self.q)
                # Build mask to align with displayed q_data (ROI + axis filter)
                mask_full = np.isfinite(q_full)
                if self._roi_active():
                    mask_full &= (q_full >= self._roi_min) & (q_full <= self._roi_max)
                if positive_only:
                    mask_full &= q_full > 0
                elif negative_only:
                    mask_full &= q_full < 0

                q_fit_raw = q_full[mask_full]
                I_fitting_data = I_fitting_arr[mask_full]
                q_fit_raw, q_fit_plot, I_fitting_data, _ = (
                    self._filter_q_data_for_independent_display(q_fit_raw, I_fitting_data)
                )
                q_fit_plot = self._convert_q_values_for_display(q_fit_plot)

                if normalize and norm_factor > 0 and I_fitting_data.size > 0:
                    I_fitting_data = I_fitting_data / norm_factor

                # Trim/pad safety: align length with plotted q values
                plot_len = min(len(q_fit_plot), len(I_fitting_data))
                if plot_len > 0:
                    fitting_y_for_limits = I_fitting_data[:plot_len]
                    ax.plot(
                        q_fit_plot[:plot_len],
                        I_fitting_data[:plot_len],
                        color="red",
                        linewidth=2.5,
                        label="Fitting",
                        zorder=3,
                    )

            x_label = self._build_q_axis_label(filter_mode=filter_mode)
            if (
                mode == "normal"
                and log_x
                and not positive_only
                and not negative_only
                and np.any(np.array(self.q) < 0)
            ):
                x_label = self._build_q_axis_label(filter_mode="all", absolute=True)

            ax.set_xlabel(x_label)
            ax.set_ylabel("Normalized Intensity" if normalize else "Intensity (a.u.)")
            ax.set_title(
                f"{mode.capitalize()} Mode - {self.data_source.upper() if self.data_source else 'Data'}"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()

            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(1.8)

            self._apply_log_scales(ax, log_x, log_y)
            self._apply_fit_y_axis_limits(
                ax,
                experimental_y=I_data,
                fitting_y=fitting_y_for_limits,
                extra_y_values=extra_y_for_limits,
                log_y=log_y,
            )

            try:
                if hasattr(self.independent_fit_window, "set_deletable_points"):
                    self.independent_fit_window.set_deletable_points(q_data, q_plot, I_data)
            except Exception:
                pass

            if hasattr(self.independent_fit_window, "canvas"):
                self.independent_fit_window.canvas.draw_idle()

        except Exception:
            pass

    def _get_cut_center_coordinates(self):
        """No description."""
        center_x = 0.0
        center_y = 0.0

        if hasattr(self.ui, "gisaxsInputCenterParallelValue"):
            center_x = self.ui.gisaxsInputCenterParallelValue.value()
        if hasattr(self.ui, "gisaxsInputCenterVerticalValue"):
            center_y = self.ui.gisaxsInputCenterVerticalValue.value()

        return center_x, center_y
