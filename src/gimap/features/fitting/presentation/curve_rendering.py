"""Shared plot specification and Matplotlib renderer for all Fitting curve views."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


POSITIVE_Q_COLOR = "#2563EB"
NEGATIVE_Q_COLOR = "#E11D48"
MODEL_COLOR = "#DC2626"


@dataclass(frozen=True)
class CurveSeries:
    x: np.ndarray
    y: np.ndarray
    label: str
    color: str
    role: str = "data"
    style: str = "scatter"
    linestyle: str = "-"
    linewidth: float = 1.8
    marker_size: float = 28.0
    alpha: float = 0.8
    zorder: int = 2

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float).reshape(-1)
        y = np.asarray(self.y, dtype=float).reshape(-1)
        count = min(x.size, y.size)
        finite = np.isfinite(x[:count]) & np.isfinite(y[:count])
        object.__setattr__(self, "x", x[:count][finite])
        object.__setattr__(self, "y", y[:count][finite])


@dataclass(frozen=True)
class CurvePlotSpec:
    series: tuple[CurveSeries, ...]
    x_label: str
    y_label: str
    title: str
    x_scale: str = "linear"
    log_y: bool = False
    roi_bounds: tuple[float, float] | None = None
    experimental_y: np.ndarray | None = None
    fitting_y: np.ndarray | None = None
    extra_y: tuple[np.ndarray, ...] = field(default_factory=tuple)
    deletable_raw_q: np.ndarray | None = None
    deletable_plot_x: np.ndarray | None = None
    deletable_y: np.ndarray | None = None


def experimental_curve_series(
    plot_q,
    intensity,
    *,
    source_sign=None,
    q_mode: str,
    label: str,
) -> tuple[CurveSeries, ...]:
    """Build measured-data layers, preserving branch identity in overlay mode."""
    q = np.asarray(plot_q, dtype=float).reshape(-1)
    values = np.asarray(intensity, dtype=float).reshape(-1)
    count = min(q.size, values.size)
    q, values = q[:count], values[:count]
    signs = (
        np.asarray(source_sign, dtype=np.int8).reshape(-1)[:count]
        if source_sign is not None
        else np.sign(q).astype(np.int8, copy=False)
    )

    if q_mode != "fold" or signs.size != count:
        return (
            CurveSeries(
                q,
                values,
                label,
                POSITIVE_Q_COLOR,
                style="scatter",
            ),
        )

    layers = []
    for sign, suffix, color in (
        (1, "+q", POSITIVE_Q_COLOR),
        (-1, "−q mirrored", NEGATIVE_Q_COLOR),
    ):
        mask = signs == sign
        if not np.any(mask):
            continue
        order = np.argsort(q[mask], kind="mergesort")
        layers.append(
            CurveSeries(
                q[mask][order],
                values[mask][order],
                f"{label} · {suffix}",
                color,
                style="scatter",
            )
        )
    return tuple(layers)


def render_curve_plot(axes, spec: CurvePlotSpec) -> None:
    """Render one immutable specification into an embedded or independent axes."""
    if getattr(axes, "_gimap_legacy_render", False):
        axes.clear()
        axes._gimap_legacy_render = False
    # Axes can also be cleared by legacy callers; discard detached cache entries.
    cached = getattr(axes, "_gimap_curve_artists", {})
    active = {}
    for index, series in enumerate(spec.series):
        if series.x.size == 0:
            continue
        key = (index, series.role, series.style, series.label)
        artist = cached.pop(key, None)
        if artist is not None and artist not in axes.get_children():
            artist = None
        if series.style == "scatter":
            if artist is None:
                artist = axes.scatter([], [])
            artist.set_offsets(np.column_stack((series.x, series.y)))
            artist.set_sizes([series.marker_size])
            artist.set_color(series.color)
        else:
            if artist is None:
                artist, = axes.plot([], [])
            artist.set_data(series.x, series.y)
            artist.set_color(series.color)
            artist.set_linestyle(series.linestyle)
            artist.set_linewidth(series.linewidth)
        artist.set_alpha(series.alpha)
        artist.set_label(series.label)
        artist.set_zorder(series.zorder)
        active[key] = artist
    for artist in cached.values():
        if artist in axes.get_children():
            artist.remove()
    axes._gimap_curve_artists = active

    roi_artists = getattr(axes, "_gimap_roi_artists", [])
    roi_artists = [artist for artist in roi_artists if artist in axes.get_children()]
    if spec.roi_bounds is None:
        for artist in roi_artists:
            artist.remove()
        roi_artists = []
    else:
        if not roi_artists:
            roi_artists = [axes.axvline(value, color="#F97316", linestyle="--",
                                       linewidth=1.2, alpha=0.8)
                           for value in spec.roi_bounds]
        for artist, value in zip(roi_artists, spec.roi_bounds):
            artist.set_xdata([float(value), float(value)])
    axes._gimap_roi_artists = roi_artists

    axes.set_xlabel(spec.x_label)
    axes.set_ylabel(spec.y_label)
    axes.set_title(spec.title)
    axes.grid(True, alpha=0.3)
    if spec.x_scale == "symlog":
        x_values = [series.x for series in spec.series if series.x.size]
        merged = np.concatenate(x_values) if x_values else np.array([], dtype=float)
        nonzero = np.abs(merged[np.isfinite(merged) & (merged != 0)])
        threshold = float(np.min(nonzero) * 0.5) if nonzero.size else 1e-6
        axes.set_xscale("symlog", linthresh=max(threshold, 1e-12))
    else:
        axes.set_xscale(spec.x_scale if spec.x_scale in {"linear", "log"} else "linear")
    axes.set_yscale("log" if spec.log_y else "linear")
    for axis in ("top", "bottom", "left", "right"):
        axes.spines[axis].set_linewidth(1.8)
    axes.tick_params(axis="both", which="both", width=1.6, labelsize=12)
    handles, labels = axes.get_legend_handles_labels()
    legend_key = (
        tuple((id(handle), label) for handle, label in zip(handles, labels)),
        tuple((series.color, series.alpha, series.linestyle, series.linewidth,
               series.marker_size) for series in spec.series if series.x.size),
    )
    if legend_key != getattr(axes, "_gimap_legend_key", None):
        if axes.get_legend() is not None:
            axes.get_legend().remove()
        if handles:
            axes.legend(handles, labels)
        axes._gimap_legend_key = legend_key
    # relim() ignores scatter collections; include their current coordinates explicitly.
    axes.relim()
    for series in spec.series:
        if series.x.size and series.style == "scatter":
            axes.update_datalim(np.column_stack((series.x, series.y)))
    axes.autoscale(enable=True)


__all__ = [
    "CurvePlotSpec",
    "CurveSeries",
    "MODEL_COLOR",
    "NEGATIVE_Q_COLOR",
    "POSITIVE_Q_COLOR",
    "experimental_curve_series",
    "render_curve_plot",
]
