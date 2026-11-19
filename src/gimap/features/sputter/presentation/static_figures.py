"""The figure of a series' two parts for the report of After Deposition (``static_export.py``).

One panel for each part the series has, on the same axis of nominal thickness so that where the film connects is seen:
**Separate particles** — radius R, height H, spacing D and the gap D − 2R (nm), the coverage on the right axis;
**Connected film** — the film's thickness (its continuous layer), its height, the relief's height and spacing (nm), the
film's density on the right axis. Drawn without a window (Agg), in the words of the interface language; Chinese
labels get a Chinese font (``shared/figures.py``).
"""

from __future__ import annotations

from io import BytesIO
from typing import Callable

from src.gimap.shared.figures import cjk_fallback

from ..application import PART_FILM, PART_PARTICLES, ReportSeries, gap

AXIS_NOMINAL = "Nominal thickness (nm)"
AXIS_SIZE = "Size (nm)"
AXIS_COVERAGE = "Coverage"
AXIS_DENSITY = "Film density"
PARTICLE_CURVES = (("R", "Radius R", "#1f77b4"), ("height", "Height H", "#2a9d5c"), ("d_hex", "Spacing D", "#9467bd"),
                   ("gap", "Gap", "#e08a1e"))
FILM_CURVES = (("t_b", "Film thickness", "#8c564b"), ("height", "Film height", "#2a9d5c"),
               ("T", "Relief height", "#17becf"), ("d_hex", "Relief spacing", "#9467bd"))
FIGURE_TEMPLATES = (AXIS_NOMINAL, AXIS_SIZE, AXIS_COVERAGE, AXIS_DENSITY,
                    *(label for _key, label, _colour in (*PARTICLE_CURVES, *FILM_CURVES)))
"""The words of the figure that need Chinese."""


def _value(row: dict, key: str):
    if key == "gap":
        return gap(row)
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def parts_figure(series: ReportSeries, say: Callable[..., str]) -> bytes:
    """PNG of the series' parts (b"" when it has no sample)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    panels = [(PART_PARTICLES, rows, PARTICLE_CURVES, ("eta", AXIS_COVERAGE)) for rows in (series.particles(),) if rows]
    panels += [(PART_FILM, rows, FILM_CURVES, ("rb", AXIS_DENSITY)) for rows in (series.film(),) if rows]
    if not panels:
        return b""
    nominal = [_value(row, "nominal_nm") for row in series.rows]
    nominal = [value for value in nominal if value is not None]
    span = (min(nominal), max(nominal)) if nominal else (0.0, 1.0)
    margin = max(0.5, 0.06 * (span[1] - span[0]))
    figure = Figure(figsize=(5.2 * len(panels), 3.5), dpi=150)
    FigureCanvasAgg(figure)
    for index, (title, rows, curves, (share, share_label)) in enumerate(panels, 1):
        axes = figure.add_subplot(1, len(panels), index)
        x = [_value(row, "nominal_nm") for row in rows]
        for key, label, colour in curves:
            pairs = [(a, b) for a, b in zip(x, (_value(row, key) for row in rows)) if a is not None and b is not None]
            if pairs:
                axes.plot(*zip(*pairs), "o-", color=colour, ms=4, lw=1.3, label=say(label))
        axes.set_xlim(span[0] - margin, span[1] + margin)
        axes.set_ylim(0, None)
        axes.set_title(say(title), fontsize=10)
        axes.set_xlabel(say(AXIS_NOMINAL), fontsize=9)
        axes.set_ylabel(say(AXIS_SIZE), fontsize=9)
        axes.grid(alpha=0.25)
        right = axes.twinx()
        pairs = [(a, b) for a, b in zip(x, (_value(row, share) for row in rows)) if a is not None and b is not None]
        if pairs:
            right.plot(*zip(*pairs), "s--", color="#64748b", ms=3.5, lw=1.0, label=say(share_label))
        right.set_ylim(0, 1.1)
        right.set_ylabel(say(share_label), fontsize=9, color="#64748b")
        lines = axes.get_legend_handles_labels()
        extra = right.get_legend_handles_labels()
        axes.legend(lines[0] + extra[0], lines[1] + extra[1], fontsize=7, loc="best", framealpha=0.85)
        for each in (axes, right):
            each.tick_params(labelsize=8)
    figure.tight_layout()
    cjk_fallback(figure)
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()


__all__ = ["FIGURE_TEMPLATES", "parts_figure"]
