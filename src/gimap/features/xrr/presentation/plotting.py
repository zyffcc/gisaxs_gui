"""Matplotlib rendering for XRR detector and curve projections."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle


class XrrPlotPresenter:
    def __init__(self, live_panel, curve_panel, *, on_detector_click):
        self.live_figure = Figure(figsize=(7, 5), tight_layout=True)
        self.live_canvas = FigureCanvasQTAgg(self.live_figure)
        self.live_axis = self.live_figure.add_subplot(111)
        live_panel.set_plot_widget(self.live_canvas)
        live_panel.show_empty()
        self.curve_figure = Figure(figsize=(7, 4), tight_layout=True)
        self.curve_canvas = FigureCanvasQTAgg(self.curve_figure)
        self.curve_axis = self.curve_figure.add_subplot(111)
        curve_panel.set_plot_widget(self.curve_canvas)
        curve_panel.show_empty()
        self._live_panel = live_panel
        self._curve_panel = curve_panel
        self.live_canvas.mpl_connect("button_press_event", on_detector_click)

    def render_detector(
        self,
        image,
        full_shape,
        *,
        roi_center: tuple[float, float] | None,
        radius_px: int,
        direct_center: tuple[float, float],
        title: str,
    ) -> None:
        values = np.asarray(image, dtype=float)
        height, width = (int(full_shape[0]), int(full_shape[1]))
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = np.nanpercentile(finite, (1.0, 99.5))
            if not np.isfinite(high) or high <= low:
                low, high = float(np.nanmin(finite)), float(np.nanmax(finite) + 1.0)
        else:
            low, high = 0.0, 1.0
        display = np.log1p(np.clip(values - min(0.0, low), 0.0, None))
        display_low = np.log1p(max(0.0, low - min(0.0, low)))
        display_high = np.log1p(max(1.0, high - min(0.0, low)))

        axis = self.live_axis
        axis.clear()
        axis.imshow(
            display,
            cmap="viridis",
            origin="upper",
            extent=(0.0, float(width), float(height), 0.0),
            interpolation="nearest",
            vmin=display_low,
            vmax=display_high,
            aspect="equal",
        )
        axis.plot(
            [direct_center[0]],
            [direct_center[1]],
            marker="+",
            markersize=13,
            markeredgewidth=2,
            color="#00d9ff",
            label="Direct beam",
        )
        if roi_center is not None:
            circle = Circle(
                roi_center,
                max(0.55, float(radius_px)),
                fill=False,
                linewidth=2.0,
                edgecolor="#ff4d8d",
                label="Specular ROI",
            )
            axis.add_patch(circle)
            axis.plot(roi_center[0], roi_center[1], ".", color="#ff4d8d", markersize=4)
        axis.set_xlim(0.0, float(width))
        axis.set_ylim(float(height), 0.0)
        axis.set_xlabel("Detector x (pixel)")
        axis.set_ylabel("Detector y (pixel)")
        axis.set_title(title)
        axis.legend(loc="upper right")
        self._live_panel.show_plot()
        self.live_canvas.draw_idle()

    def render_curve(self, result, *, log_y: bool) -> None:
        axis = self.curve_axis
        axis.clear()
        if result is None or not result.points:
            self._curve_panel.show_empty()
            self.curve_canvas.draw_idle()
            return
        qz = result.qz
        intensity = result.intensity
        axis.plot(qz, intensity, "o-", color="#2f7ed8", markersize=4, linewidth=1.2)
        axis.set_xlabel("qz (Å⁻¹)")
        axis.set_ylabel("Extracted intensity")
        axis.grid(True, which="both", alpha=0.22)
        if log_y and np.any(np.isfinite(intensity) & (intensity > 0)):
            axis.set_yscale("log")
        self._curve_panel.show_plot()
        self.curve_canvas.draw_idle()


__all__ = ["XrrPlotPresenter"]
