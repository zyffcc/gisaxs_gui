"""Auditable image outlines and radius distributions for the SEM tool."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy import ndimage

from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.theme.figures import restyle_figure
from src.gimap.app.presentation.theme import theme_manager
from src.gimap.shared.figures import cjk_fallback


class SemPlotPresenter:
    def __init__(self, image_panel, distribution_panel):
        self.image_panel, self.distribution_panel = image_panel, distribution_panel
        self.image_figure = Figure(figsize=(7, 5), layout="constrained")
        self.image_canvas = FigureCanvasQTAgg(self.image_figure)
        self.image_axis = self.image_figure.add_subplot(111)
        self.distribution_figure = Figure(figsize=(7, 5), layout="constrained")
        self.distribution_canvas = FigureCanvasQTAgg(self.distribution_figure)
        self.distribution_axis = self.distribution_figure.add_subplot(111)
        image_panel.set_plot_widget(self.image_canvas)
        distribution_panel.set_plot_widget(self.distribution_canvas)
        theme_manager().changed.connect(self._restyle)
        self.render_image(None)
        self.render_distribution(None)

    def render_image(self, image, measurements=None):
        axis = self.image_axis
        axis.clear()
        if image is None:
            self.image_panel.show_empty()
        else:
            values = np.asarray(image, dtype=float)
            finite = values[np.isfinite(values)]
            low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
            if high <= low:
                high = low + 1
            height, width = values.shape
            extent = (0, width, height, 0)
            axis.imshow(values, cmap="gray", origin="upper", vmin=low, vmax=high, extent=extent)
            if measurements is not None and measurements.labels is not None:
                labels = np.asarray(measurements.labels)
                accepted = [p.label for p in measurements.particles if p.accepted]
                retained = np.isin(labels, accepted) & (labels != 0)
                edge = (labels != 0) & (ndimage.maximum_filter(labels, size=3) != ndimage.minimum_filter(labels, size=3))
                overlay = np.zeros((*values.shape, 4), dtype=float)
                overlay[edge & retained] = (0.05, 0.75, 0.9, 0.95)
                overlay[edge & ~retained] = (1.0, 0.55, 0.1, 0.95)
                axis.imshow(overlay, origin="upper", extent=extent)
                roi = measurements.settings.get("roi_xyxy")
                if roi is not None:
                    left, top, right, bottom = roi
                    axis.add_patch(Rectangle((left, top), right - left, bottom - top,
                                             fill=False, edgecolor="#ef9b20", linewidth=1.5))
                axis.set_title(tr("Cyan: accepted   Orange: rejected   Yellow: ROI"))
            else:
                axis.set_title(tr("SEM image — set calibration and exclude annotations using the ROI"))
            axis.set_xlabel(tr("Image x (pixel)"))
            axis.set_ylabel(tr("Image y (pixel)"))
            self.image_panel.show_plot()
        cjk_fallback(self.image_figure)
        restyle_figure(self.image_figure)
        self.image_canvas.draw_idle()

    def render_distribution(self, comparison):
        axis = self.distribution_axis
        axis.clear()
        if comparison is None:
            self.distribution_panel.show_empty()
        else:
            edges = np.asarray(comparison.display_sem_edges_nm)
            values = np.asarray(comparison.display_sem_values)
            axis.bar(edges[:-1], values, width=np.diff(edges), align="edge",
                     color="#4b9bd4", alpha=0.65, label="SEM", linewidth=0)
            if comparison.display_prediction_values is not None:
                axis.plot(comparison.display_prediction_radius_nm,
                          comparison.display_prediction_values, "o-", color="#e06632",
                          linewidth=1.5, markersize=3, label=tr("Prediction"))
            axis.set_xlabel(tr("Radius R (nm)"))
            axis.set_xlim(float(edges[0]), float(edges[-1]))
            paper = comparison.metadata.get("display_mode") == "paper"
            axis.set_ylabel(tr("Display amplitude (independently scaled)") if paper else tr("Probability per bin"))
            axis.set_ylim(bottom=0)
            axis.legend(loc="best")
            self.distribution_panel.show_plot()
        cjk_fallback(self.distribution_figure)
        restyle_figure(self.distribution_figure)
        self.distribution_canvas.draw_idle()

    def dispose(self):
        """Break Qt-canvas/Figure cycles before Qt destroys the embedded widgets."""
        try:
            theme_manager().changed.disconnect(self._restyle)
        except (TypeError, RuntimeError):
            pass
        for canvas, figure in ((self.image_canvas, self.image_figure),
                               (self.distribution_canvas, self.distribution_figure)):
            canvas._draw_pending = False
            canvas.close()
            figure.clear()
            figure.set_canvas(None)
            canvas.figure = None

    def _restyle(self, *_args):
        for figure, canvas in ((self.image_figure, self.image_canvas),
                               (self.distribution_figure, self.distribution_canvas)):
            restyle_figure(figure)
            canvas.draw_idle()


__all__ = ["SemPlotPresenter"]
