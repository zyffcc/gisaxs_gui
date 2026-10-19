"""Detached light-color scientific figures; common data with the GUI preview."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.gimap.shared.figures import publication_figure, save_figure, style_publication_axes


def distribution_figure(comparison):
    figure = publication_figure(width_cm=15, height_cm=10)
    axes = figure.subplots()
    edges = comparison.display_sem_edges_nm
    values = comparison.display_sem_values
    axes.bar((edges[:-1] + edges[1:]) / 2, values, width=np.diff(edges),
             color="#79AFC8", edgecolor="white", linewidth=0.35, label="SEM", align="center")
    if comparison.display_prediction_values is not None:
        axes.plot(comparison.display_prediction_radius_nm, comparison.display_prediction_values,
                  color="#B74C41", linewidth=1.4, label="Prediction", zorder=3)
    axes.set_xlabel("Radius R (nm)")
    paper = comparison.metadata["display_mode"] == "paper"
    axes.set_ylabel("Probability / scaled prediction" if paper else "Probability per bin")
    axes.set_xlim(float(edges[0]), float(edges[-1]))
    maximum = float(np.max(values))
    if comparison.display_prediction_values is not None:
        maximum = max(maximum, float(np.max(comparison.display_prediction_values)))
    axes.set_ylim(0, max(maximum * 1.25, 0.01))
    axes.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=7, frameon=False,
                borderaxespad=0)
    style_publication_axes(axes)
    axes.set_facecolor("white")
    figure.set_facecolor("white")
    return figure


def overlay_rgba(measurements):
    labels = measurements.labels
    if labels is None:
        return None
    from scipy import ndimage

    # Different labels on touching objects retain visible boundaries.
    boundary = labels != ndimage.minimum_filter(labels, size=3)
    boundary |= labels != ndimage.maximum_filter(labels, size=3)
    boundary &= labels > 0
    accepted_labels = [particle.label for particle in measurements.particles if particle.accepted]
    accepted = np.isin(labels, accepted_labels)
    overlay = np.zeros((*labels.shape, 4), float)
    overlay[boundary & accepted] = [0.05, 0.75, 0.9, 0.95]
    overlay[boundary & ~accepted] = [1.0, 0.55, 0.1, 0.95]
    return overlay


def segmentation_figure(measurements):
    figure = publication_figure(width_cm=15, height_cm=12, fixed_aspect=True)
    axes = figure.subplots()
    height, width = measurements.image.shape
    extent = (0, width, height, 0)
    finite = measurements.image[np.isfinite(measurements.image)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    if high <= low:
        high = low + 1
    axes.imshow(measurements.image, cmap="gray", origin="upper", extent=extent,
                interpolation="nearest", vmin=low, vmax=high)
    axes.imshow(overlay_rgba(measurements), origin="upper", extent=extent, interpolation="nearest")
    roi = measurements.settings["roi_xyxy"]
    from matplotlib.patches import Rectangle

    axes.add_patch(Rectangle((roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1],
                             edgecolor="#E1BA48", facecolor="none", linewidth=0.8))
    axes.set_xlabel("x (pixel)")
    axes.set_ylabel("y (pixel)")
    axes.set_title("Cyan: accepted   Orange: rejected   Yellow: ROI", fontsize=8)
    style_publication_axes(axes)
    figure.set_facecolor("white")
    return figure


def export_distribution_figures(comparison, out_dir: Path):
    import matplotlib as mpl

    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8,
                         "savefig.facecolor": "white", "text.color": "black",
                         "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"}):
        figure = distribution_figure(comparison)
        paths = (
            save_figure(figure, out_dir / "sem_distribution.png"),
            save_figure(figure, out_dir / "sem_distribution.pdf"),
            save_figure(figure, out_dir / "sem_distribution.svg"),
        )
        figure.clear()
    return paths


def export_overlay(measurements, out_dir: Path):
    import matplotlib as mpl

    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "font.size": 8, "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"}):
        figure = segmentation_figure(measurements)
        paths = (
            save_figure(figure, out_dir / "sem_overlay.png"),
            save_figure(figure, out_dir / "sem_overlay.pdf"),
            save_figure(figure, out_dir / "sem_overlay.svg"),
        )
        figure.clear()
    return paths
