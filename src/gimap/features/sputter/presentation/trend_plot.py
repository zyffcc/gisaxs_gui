"""Size Trend tab: R, D (and the TS length L) over time, and the metal thickness.

The physical path (default) draws every window's network value as a faint point and the smoothed trend as a
line, so a single window's jump is visible but does not pull the eye. The other explanations draw each value
with its simulation-based error bar, coloured by grade. The Metal thickness panel shows the island-volume
equivalent thickness t_eq as a band (conditional: hemispheres h = R on a hexagonal arrangement), the nominal
thickness rate·(t − t0) when a deposition rate is given, and shades the film windows: there the thickness is
not determined. The film onset of the physical path is marked with its interval on every panel. When the end of the
deposition t_end is given (the shutter is closed by hand; a scan can go on after it) the nominal thickness stays at
rate·(t_end − t0) after it (a plateau) and t_end is marked on the thickness panel — when it falls within the
windows; an end after the last window (or before the first) is only named in the legend, and the time axis keeps to
the windows.

The literature explanation draws R and D of the literature method the same way (grade Lit: R from the nominal
thickness δ and 2π/q* by mass balance); its thickness panel shows δ only (the islands hold it by construction), the
windows after its percolation are shaded (D = 2π/q* goes on, R is not given) and the percolation is marked with its
interval (the scatter of the windows and the errors of the rate and t0). A percolation that is not determined is
shaded from the earliest time it may come at to the end of the peak phase it was looked for in. With a deposition
rate the top axis also gives δ = rate × (t − t0), with ticks only up to the end of the deposition (δ stays there;
no δ axis when the deposition ended before the first window: δ is the same in every window).
"""

from __future__ import annotations

import math

from PyQt5.QtWidgets import QLabel, QSizePolicy

from src.gimap.app.presentation.i18n import tr

from ..application import (FILM_NOT_DETERMINED, GRADES, LITERATURE_GRADE, StructureResult, deposition_end, onset_of,
                           percolation_errors, percolation_last_time)
from .texts import text

COLORS = {"A": "#1f5fa8", "B": "#7fa3d6", "C": "#d07a10", "S": "#2a8a3e", "T": "#c0392b", "L": "#6a3d9a",
          "P": "#159a9c", LITERATURE_GRADE: "#b0651c"}
"""Grade colours: readable on the light and the dark plot background."""
TREND_COLORS = {"R": "#0b5d8a", "D": "#8e3b8e", "teq": "#b8860b", "nominal": "#7f7f7f"}
FILM_SHADE = "#8c8c8c"
ONSET_COLOR = "#d62728"
AFTER_PERCOLATION = "after percolation: D = 2π/q* goes on, R is not given (film)"
INTERVAL_LABEL = "68% interval of the percolation (scatter, rate ±{rate_error}%, t0 ±{t0_error} s)"
UNDETERMINED_LABEL = "percolation not determined: from {time} s or later"
BY_CONSTRUCTION = "literature method: the islands hold δ by construction"
DEPOSITION_END = "deposition ended {time} s"
ENDED_AFTER = "deposition ended {time} s, after the last window"
ENDED_BEFORE = "deposition ended {time} s, before the first window"
SMOOTHED = ("physical", "literature")


def thickness_axis(result):
    """(rate, t0, t_end) of the nominal thickness δ = rate·(min(t, t_end) − t0) of a result (t_end None: the
    deposition lasts to the end of the scan), or None without a rate."""
    thickness = result.summary.get("thickness") if result is not None else None
    if not isinstance(thickness, dict) or result.variant not in SMOOTHED:
        return None
    try:
        rate = float(thickness.get("rate_nm_s"))
        t0 = float(thickness.get("t0_s") or 0.0)
    except (TypeError, ValueError):
        return None
    return (rate, t0, deposition_end(result.summary)) if rate > 0 else None


class TrendPlot:
    def __init__(self, layout):
        self.canvas = None
        self.figure = None
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as exc:  # matplotlib missing: the table and the report stay
            layout.addWidget(QLabel(text("Plotting is unavailable: {error}", error=exc)))
            return
        self.figure = Figure(figsize=(7, 6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)
        # The page redraws on a theme switch (``SputterPage._theme_changed``): a connection of its own, which Qt
        # removes with the page (``theme_figure``'s connection to the canvas crashed when the page was deleted).

    def draw(self, result, *, log_scale: bool) -> None:
        if self.canvas is None:
            return
        from src.gimap.app.presentation.theme.figures import restyle_figure
        from src.gimap.shared.figures import cjk_fallback

        fig = self.figure
        fig.clear()
        if result is not None and result.rows:
            self._panels(fig, result, log_scale)
        restyle_figure(fig)
        cjk_fallback(fig)  # Chinese labels in a font that has them
        self.canvas.draw_idle()

    # -- panels -----------------------------------------------------------------------------------------
    def _panels(self, fig, result: StructureResult, log_scale: bool) -> None:
        physical = result.variant in SMOOTHED
        panels = [("R", "R (nm)", tr("R, equivalent hemisphere radius")),
                  ("D", "D (nm)", tr("D, centre-to-centre spacing (peak windows)"))]
        if result.ts_length().time:
            panels.append(("L", "L (nm)", tr("TS length L (film windows)")))
        if physical or result.has("t_eq_nm") or result.has("nominal_thickness_nm"):
            panels.append(("thickness", tr("Metal thickness (nm)"), ""))
        axes = fig.subplots(len(panels), 1, sharex=True)
        axes = list(axes) if len(panels) > 1 else [axes]
        spans = _padded(result.film_spans(), [r.get("time_s") for r in result.rows])
        onset, interval, kind = onset_of(result)
        onset_label = "percolation {time} s (literature method)" if kind == "percolation" else "film onset {time} s"
        times = [r.get("time_s") for r in result.rows if r.get("time_s") is not None]
        ended = deposition_end(result.summary)
        for ax, (name, label, title) in zip(axes, panels):
            for low, high in spans:
                ax.axvspan(low, high, color=FILM_SHADE, alpha=.13, lw=0, zorder=0)
            if name == "L":
                ts = result.ts_length()
                ax.plot(ts.time, ts.value, "o", ms=2.5, color="#5a8a2a", label=tr("TS length L (network)"))
            elif name == "thickness" and result.variant == "literature":
                self._delta_only(ax, result, spans)
            elif name == "thickness":
                self._thickness(ax, result, spans)
            elif physical:
                self._physical(ax, result, name)
            else:
                self._graded(ax, result, name)
            if name == "thickness" and ended is not None:  # the nominal thickness stays constant after it
                _deposition_end(ax, ended, times)
            first = name == panels[0][0]
            if interval is not None and times:
                end = (percolation_last_time(result.summary) if kind == "percolation" else None) or max(times)
                self._interval(ax, result, interval, onset, end, kind, first)
            if onset is not None:
                ax.axvline(onset, color=ONSET_COLOR, lw=1, ls=":",
                           label=text(onset_label, time=f"{onset:.1f}") if first else None)
            if log_scale and _positive(ax):
                ax.set_yscale("log")
                _plain_ticks(ax)
            ax.set_ylabel(label)
            if title:
                ax.set_title(title, fontsize=8, loc="left")
            ax.grid(alpha=.25)
            legend = ax.legend(fontsize=7, loc="best")
            if legend is not None:
                legend.get_frame().set_alpha(.7)
        axes[-1].set_xlabel(tr("time (s)"))
        axis = thickness_axis(result)
        if axis is not None and not (axis[2] is not None and times and axis[2] < min(times)):
            rate, t0, t_end = axis
            top = axes[0].secondary_xaxis("top", functions=(lambda t: rate * (t - t0), lambda d: d / rate + t0))
            if t_end is not None:  # δ stays at rate × (t_end − t0) after the deposition ended
                _ticks_up_to(top, rate * (t_end - t0))
            top.set_xlabel(tr("δ, nominal thickness (nm)"), fontsize=8)
            top.tick_params(labelsize=7)

    @staticmethod
    def _interval(ax, result: StructureResult, interval: tuple, onset, end: float, kind: str, first: bool) -> None:
        """The onset's interval; an open end runs to ``end`` (a percolation that is not determined has no onset,
        only the earliest time it may come at: it is hatched to the end of the peak phase)."""
        low, high = interval
        if kind != "percolation":
            ax.axvspan(low, high if high is not None else end, color=ONSET_COLOR, alpha=.08, lw=0, zorder=0)
            return
        if onset is None:
            label = text(UNDETERMINED_LABEL, time=f"{low:.1f}")
        else:
            label = text(INTERVAL_LABEL, **percolation_errors(result.summary))
        ax.axvspan(low, high if high is not None else end, color=ONSET_COLOR, alpha=.08, lw=0, zorder=0,
                   hatch="//" if high is None else None, label=label if first else None)

    @staticmethod
    def _physical(ax, result: StructureResult, name: str) -> None:
        points = result.column(f"{name}_nm")
        grade = LITERATURE_GRADE if result.variant == "literature" else "P"
        if points.time:
            ax.plot(points.time, points.value, "o", ms=2.6, alpha=.3, color=COLORS[grade], lw=0,
                    label=text("{grade}: {meaning}", grade=grade, meaning=tr(GRADES[grade])))
        trend = result.column(f"{name}_trend_nm")
        if trend.time:
            ax.plot(trend.time, trend.value, "-", lw=1.7, color=TREND_COLORS[name], label=tr("trend (smoothed)"))
        if name == "D":
            _direct(ax, result)

    @staticmethod
    def _delta_only(ax, result: StructureResult, spans: list) -> None:
        """The literature explanation's thickness panel: only δ (the islands hold it by construction)."""
        nominal = result.column("nominal_thickness_nm")
        if nominal.time:
            ax.plot(nominal.time, nominal.value, "--", lw=1.3, color=TREND_COLORS["nominal"],
                    label=tr("nominal: rate × (t − t0)"))
        ax.text(.02, .92, tr(BY_CONSTRUCTION), transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                color="#555555", bbox=dict(facecolor="white", alpha=.65, lw=0))
        if spans:
            low_t, high_t = spans[0]
            ax.text((low_t + high_t) / 2, .5, tr(AFTER_PERCOLATION), transform=ax.get_xaxis_transform(),
                    ha="center", va="center", fontsize=7.5, color="#555555", bbox=dict(facecolor="white", alpha=.65, lw=0))

    @staticmethod
    def _graded(ax, result: StructureResult, name: str) -> None:
        for grade in ("C", "B", "A", "S", "T", "L", "P", LITERATURE_GRADE):
            s = result.series(name, grade)
            if not s.time:
                continue
            low = [v - lo for v, lo in zip(s.value, s.low)]
            high = [hi - v for v, hi in zip(s.value, s.high)]
            ax.errorbar(s.time, s.value, yerr=[low, high], fmt="o", ms=2.6, lw=.5, color=COLORS[grade], alpha=.85,
                        label=text("{grade}: {meaning}", grade=grade, meaning=tr(GRADES[grade])))
        if name == "D":
            _direct(ax, result)

    @staticmethod
    def _thickness(ax, result: StructureResult, spans: list) -> None:
        times = [r.get("time_s") for r in result.rows]
        nan = float("nan")

        def values(key):
            return [nan if r.get(key) is None else r[key] for r in result.rows]

        teq, low, high = values("t_eq_nm"), values("t_eq_lo_nm"), values("t_eq_hi_nm")
        if any(not math.isnan(v) for v in teq):
            band_low = [lo if not math.isnan(lo) else v for lo, v in zip(low, teq)]
            band_high = [hi if not math.isnan(hi) else v for hi, v in zip(high, teq)]
            ax.fill_between(times, band_low, band_high, color=TREND_COLORS["teq"], alpha=.22, lw=0)
            ax.plot(times, teq, "-", lw=1.3, color=TREND_COLORS["teq"],
                    label=tr("t_eq: island-volume equivalent (conditional)"))
        nominal = result.column("nominal_thickness_nm")
        if nominal.time:
            ax.plot(nominal.time, nominal.value, "--", lw=1.3, color=TREND_COLORS["nominal"],
                    label=tr("nominal: rate × (t − t0)"))
        if spans:
            low_t, high_t = spans[0]
            ax.text((low_t + high_t) / 2, .5, tr(FILM_NOT_DETERMINED), transform=ax.get_xaxis_transform(),
                    ha="center", va="center", fontsize=7.5, color="#555555",
                    bbox=dict(facecolor="white", alpha=.65, lw=0))
        elif not nominal.time and all(math.isnan(v) for v in teq):
            ax.text(.5, .5, tr("No thickness value in this result."), transform=ax.transAxes, ha="center",
                    va="center", fontsize=8, color="#555555")


def _direct(ax, result: StructureResult) -> None:
    bad = result.inconsistent_spacing()
    if bad.time:
        ax.plot(bad.time, bad.value, "o", ms=7, mfc="none", mec="#d62728", mew=.9,
                label=tr("D vs 2π/q* outside simulated 90% range"))
    direct = result.direct_spacing()
    if direct.time:
        ax.plot(direct.time, direct.value, "x", ms=3, color="0.5", label=tr("2π/q* (direct)"))


def _deposition_end(ax, ended: float, times: list) -> None:
    """The end of the deposition: a line where it falls within the windows; outside them only a legend entry (a line
    there would stretch the shared time axis, e.g. to an end long after the scan, and squeeze the windows)."""
    style = dict(color=TREND_COLORS["nominal"], lw=1, ls="-.")
    time = f"{ended:.1f}"
    if times and min(times) <= ended <= max(times):
        ax.axvline(ended, label=text(DEPOSITION_END, time=time), **style)
    else:
        ax.plot([], [], label=text(ENDED_BEFORE if times and ended < min(times) else ENDED_AFTER, time=time), **style)


def _ticks_up_to(axis, limit: float) -> None:
    """Ticks of the top δ axis only up to ``limit``, the thickness at the end of the deposition: the axis goes on
    linearly in time, but no more metal was deposited after it."""
    import numpy as np
    from matplotlib.ticker import AutoLocator

    class Deposited(AutoLocator):
        def tick_values(self, vmin, vmax):
            values = np.asarray(super().tick_values(vmin, vmax))
            return values[values <= limit + 1e-9 * abs(limit)]

    axis.xaxis.set_major_locator(Deposited())


def _padded(spans: list, times: list) -> list:
    """Film runs widened by half a window on each side (a run of one window is still seen)."""
    times = sorted(t for t in times if t is not None)
    steps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    half = steps[len(steps) // 2] / 2 if steps else 0.25
    return [(low - half, high + half) for low, high in spans]


def _plain_ticks(ax) -> None:
    """Plain numbers on a logarithmic axis (“2.5”, not “2.5 × 10⁰”); the minor ticks are labelled only when the
    axis spans less than a decade."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    plain = FuncFormatter(lambda value, _position: f"{value:g}")
    ax.yaxis.set_major_formatter(plain)
    low, high = ax.get_ylim()
    ax.yaxis.set_minor_formatter(plain if low > 0 and high / low < 10 else NullFormatter())


def _positive(ax) -> bool:
    """Only an axis with positive data can be logarithmic."""
    for line in ax.get_lines():
        if any(y is not None and not (isinstance(y, float) and math.isnan(y)) and y > 0 for y in line.get_ydata()):
            return True
    return bool(ax.collections)


TEMPLATES = (AFTER_PERCOLATION, BY_CONSTRUCTION, DEPOSITION_END, ENDED_AFTER, ENDED_BEFORE, INTERVAL_LABEL,
             UNDETERMINED_LABEL)
"""Texts the trend says from a name (the interface translates them)."""

__all__ = ["AFTER_PERCOLATION", "BY_CONSTRUCTION", "COLORS", "DEPOSITION_END", "ENDED_AFTER", "ENDED_BEFORE",
           "TEMPLATES", "TrendPlot", "thickness_axis"]
