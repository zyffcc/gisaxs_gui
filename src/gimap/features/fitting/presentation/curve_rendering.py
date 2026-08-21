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
    axes.clear()
    for series in spec.series:
        if series.x.size == 0:
            continue
        if series.style == "scatter":
            axes.scatter(
                series.x,
                series.y,
                s=series.marker_size,
                alpha=series.alpha,
                color=series.color,
                label=series.label,
                zorder=series.zorder,
            )
        else:
            axes.plot(
                series.x,
                series.y,
                color=series.color,
                linestyle=series.linestyle,
                linewidth=series.linewidth,
                alpha=series.alpha,
                label=series.label,
                zorder=series.zorder,
            )

    if spec.roi_bounds is not None:
        for value in spec.roi_bounds:
            axes.axvline(
                float(value),
                color="#F97316",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

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
    if handles:
        axes.legend(handles, labels)


__all__ = [
    "CurvePlotSpec",
    "CurveSeries",
    "MODEL_COLOR",
    "NEGATIVE_Q_COLOR",
    "POSITIVE_Q_COLOR",
    "experimental_curve_series",
    "render_curve_plot",
]
