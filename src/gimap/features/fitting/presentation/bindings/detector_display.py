"""Shared curve display orchestration for the fitting workspace."""

from __future__ import annotations

import numpy as np

from ..binding_primitives import _scientific_commands
from ..curve_rendering import (
    MODEL_COLOR,
    CurvePlotSpec,
    CurveSeries,
    experimental_curve_series,
    render_curve_plot,
)


_COMPONENT_COLORS = (
    "#1F77B4",
    "#2CA02C",
    "#FF7F0E",
    "#D62728",
    "#9467BD",
    "#8C564B",
)


class DetectorDisplayMixin:
    """Build one plot specification and project it into every curve view."""

    def _build_curve_plot_spec(self, mode: str) -> CurvePlotSpec | None:
        state = self._current_curve_view_state(sync_window=False)
        q_data, intensity, source_sign = self._get_roi_active_prepared_arrays()
        if q_data is None or intensity is None or source_sign is None:
            return None
        q_data, intensity, source_sign = self._filter_ai_excluded_points_for_display(
            q_data, intensity, source_sign
        )
        if q_data.size == 0 or intensity.size == 0:
            return None

        normalizer = 1.0
        if state.normalize:
            maximum = float(np.max(intensity)) if intensity.size else 0.0
            if maximum > 0:
                normalizer = maximum
                intensity = intensity / normalizer

        q_plot = self._convert_q_values_for_display(q_data)
        data_label = (
            f"{self.data_source.upper()} Data"
            if getattr(self, "data_source", None)
            else "Data"
        )
        series: list[CurveSeries] = []
        show_data = state.layer_mode in {"data", "compare"}
        show_model = mode == "fitting" and state.layer_mode in {"compare", "model"}
        if show_data:
            series.extend(
                experimental_curve_series(
                    q_plot,
                    intensity,
                    source_sign=source_sign,
                    q_mode=state.q_mode,
                    label=data_label,
                )
            )

        extra_y: list[np.ndarray] = []
        fitting_y = None
        if show_model:
            fitting_y = self._append_component_series(
                series,
                extra_y,
                q_data=q_data,
                q_plot=q_plot,
                normalizer=normalizer,
            )

        if fitting_y is not None and fitting_y.size:
            if state.normalize and normalizer > 0:
                fitting_y = fitting_y / normalizer
            series.append(
                CurveSeries(
                    q_plot,
                    fitting_y,
                    "Fitting",
                    MODEL_COLOR,
                    role="model",
                    style="line",
                    linewidth=2.3,
                    alpha=1.0,
                    zorder=3,
                )
            )

        roi_bounds = None
        if self._roi_active():
            converted = self._convert_q_values_for_display(
                np.asarray([self._roi_min, self._roi_max], dtype=float)
            )
            if converted.size == 2 and np.all(np.isfinite(converted)):
                roi_bounds = tuple(sorted((float(converted[0]), float(converted[1]))))

        raw_delete_q = np.asarray(q_data, dtype=float)
        if state.q_mode == "fold":
            raw_delete_q = np.abs(raw_delete_q) * np.asarray(source_sign, dtype=float)

        return CurvePlotSpec(
            series=tuple(series),
            x_label=self._build_q_axis_label(
                filter_mode=self._get_independent_axis_filter_mode()
            ),
            y_label="Normalized Intensity" if state.normalize else "Intensity (a.u.)",
            title={
                "data": "Experimental Curve",
                "model": "Current Model",
                "compare": "Data / Model Comparison",
            }.get(state.layer_mode, "Curve"),
            x_scale=self._get_x_axis_scale(),
            log_y=state.log_y,
            roi_bounds=roi_bounds,
            experimental_y=intensity if show_data else None,
            fitting_y=fitting_y,
            extra_y=tuple(extra_y),
            deletable_raw_q=raw_delete_q if show_data else None,
            deletable_plot_x=np.asarray(q_plot) if show_data else None,
            deletable_y=np.asarray(intensity) if show_data else None,
        )

    def _append_component_series(
        self,
        series: list[CurveSeries],
        extra_y: list[np.ndarray],
        *,
        q_data: np.ndarray,
        q_plot: np.ndarray,
        normalizer: float,
    ) -> np.ndarray | None:
        show_background = self._get_checkbox_state("fitBGShowCheckBox", False)
        show_resolution = self._get_checkbox_state("fitResShowCheckBox", False)
        particle_flags = self._get_particle_sequence_flags()
        shapes, parameters = self._get_last_fitting_spec_and_params()
        if not shapes or not parameters:
            return None
        try:
            components = _scientific_commands(self).model.components(
                shapes,
                self._convert_q_values_for_model(q_data, source=self.data_source),
                parameters,
            )
        except Exception:
            return None

        divisor = normalizer if normalizer > 0 else 1.0
        if show_background and components.get("BG_total") is not None:
            values = np.asarray(components["BG_total"], dtype=float) / divisor
            series.append(
                CurveSeries(
                    q_plot,
                    values,
                    "Background",
                    "#64748B",
                    role="component",
                    style="line",
                    linestyle="--",
                )
            )
            extra_y.append(values)
        if show_resolution and components.get("resolution") is not None:
            values = np.asarray(components["resolution"], dtype=float) / divisor
            series.append(
                CurveSeries(
                    q_plot,
                    values,
                    "Resolution",
                    "#8E44AD",
                    role="component",
                    style="line",
                    linestyle="--",
                )
            )
            extra_y.append(values)

        for item in components.get("particles", []):
            index = int(item.get("index", 0))
            values = item.get("I")
            if not particle_flags.get(index, False) or values is None:
                continue
            widget_id = self._sequence_index_to_widget_id(index)
            color_key = widget_id if widget_id is not None else index
            color = _COMPONENT_COLORS[(max(1, color_key) - 1) % len(_COMPONENT_COLORS)]
            shape = str(item.get("shape", "Particle")).capitalize()
            label = f"{shape} {widget_id if widget_id is not None else index}"
            plotted = np.asarray(values, dtype=float) / divisor
            series.append(
                CurveSeries(
                    q_plot,
                    plotted,
                    label,
                    color,
                    role="component",
                    style="line",
                    linestyle="--",
                )
            )
            extra_y.append(plotted)
        total = components.get("total")
        if total is None:
            return None
        return np.asarray(total, dtype=float)

    def _render_curve_plot_spec(self, axes, spec: CurvePlotSpec) -> None:
        render_curve_plot(axes, spec)
        self._apply_fit_y_axis_limits(
            axes,
            experimental_y=spec.experimental_y,
            fitting_y=spec.fitting_y,
            extra_y_values=spec.extra_y,
            log_y=spec.log_y,
        )

    def _update_GUI_image(self, mode):
        """Render the current curve state in the embedded canvas."""
        try:
            if not self._has_valid_data():
                return
            spec = self._build_curve_plot_spec(mode)
            if spec is None:
                return
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            scene = self._setup_fit_graphics_scene()
            if scene is None:
                return
            figure = Figure(figsize=(9.6, 7.2), dpi=80)
            canvas = FigureCanvasQTAgg(figure)
            axes = figure.add_subplot(111)
            self._render_curve_plot_spec(axes, spec)
            figure.tight_layout()
            proxy_widget = scene.addWidget(canvas)
            self._fit_view_to_item(
                self._active_curve_graphics_view(), proxy_widget, keep_aspect=True
            )
            self._current_fit_figure = figure
            self._current_fit_canvas = canvas
            self._current_curve_plot_spec = spec
        except Exception as exc:
            emitter = getattr(getattr(self, "status_updated", None), "emit", None)
            if callable(emitter):
                emitter(f"Unable to render fitting curve: {exc}")

    def _update_outside_window(self, mode):
        """Render the same plot specification in the independent canvas."""
        window = getattr(self, "independent_fit_window", None)
        if window is None or not window.isVisible() or window.ax is None:
            return
        try:
            spec = self._build_curve_plot_spec(mode)
            if spec is None:
                return
            self._render_curve_plot_spec(window.ax, spec)
            window.set_deletable_points(
                spec.deletable_raw_q,
                spec.deletable_plot_x,
                spec.deletable_y,
            )
            window.canvas.draw_idle()
            window._current_curve_plot_spec = spec
        except Exception as exc:
            emitter = getattr(getattr(self, "status_updated", None), "emit", None)
            if callable(emitter):
                emitter(f"Unable to render independent fitting curve: {exc}")

    def _get_cut_center_coordinates(self):
        center_x = 0.0
        center_y = 0.0
        if hasattr(self.ui, "gisaxsInputCenterParallelValue"):
            center_x = self.ui.gisaxsInputCenterParallelValue.value()
        if hasattr(self.ui, "gisaxsInputCenterVerticalValue"):
            center_y = self.ui.gisaxsInputCenterVerticalValue.value()
        return center_x, center_y
