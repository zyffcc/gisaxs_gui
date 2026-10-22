"""Size Trend tab: R, D (and the TS length L) over time, and the metal thickness.

The physical path (default) draws every window's network value as a faint point and the smoothed trend as a
line, so a single window's jump is visible but does not pull the eye. The other explanations draw each value
with its simulation-based error bar, coloured by grade. The Metal thickness panel shows the island-volume
equivalent thickness t_eq as a band (conditional: hemispheres h = R on a hexagonal arrangement), the nominal
thickness rate·(t − t0) when a deposition rate is given, and shades the film windows: there the thickness is
not determined. The film onset of the physical path is marked with its interval on every panel.
"""

from __future__ import annotations

import math

from PyQt5.QtWidgets import QLabel, QSizePolicy

from src.gimap.app.presentation.i18n import tr

from ..application import FILM_NOT_DETERMINED, GRADES, StructureResult, film_onset
from .texts import text

COLORS = {"A": "#1f5fa8", "B": "#7fa3d6", "C": "#d07a10", "S": "#2a8a3e", "T": "#c0392b", "L": "#6a3d9a",
          "P": "#159a9c"}
"""Grade colours: readable on the light and the dark plot background."""
TREND_COLORS = {"R": "#0b5d8a", "D": "#8e3b8e", "teq": "#b8860b", "nominal": "#7f7f7f"}
FILM_SHADE = "#8c8c8c"
ONSET_COLOR = "#d62728"


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
        physical = result.variant == "physical"
        panels = [("R", "R (nm)", tr("R, equivalent hemisphere radius")),
                  ("D", "D (nm)", tr("D, centre-to-centre spacing (peak windows)"))]
        if result.ts_length().time:
            panels.append(("L", "L (nm)", tr("TS length L (film windows)")))
        if physical or result.has("t_eq_nm") or result.has("nominal_thickness_nm"):
            panels.append(("thickness", tr("Metal thickness (nm)"), ""))
        axes = fig.subplots(len(panels), 1, sharex=True)
        axes = list(axes) if len(panels) > 1 else [axes]
        spans = _padded(result.film_spans(), [r.get("time_s") for r in result.rows])
        onset, interval = film_onset(result.summary)
        for ax, (name, label, title) in zip(axes, panels):
            for low, high in spans:
                ax.axvspan(low, high, color=FILM_SHADE, alpha=.13, lw=0, zorder=0)
            if name == "L":
                ts = result.ts_length()
                ax.plot(ts.time, ts.value, "o", ms=2.5, color="#5a8a2a", label=tr("TS length L (network)"))
            elif name == "thickness":
                self._thickness(ax, result, spans)
            elif physical:
                self._physical(ax, result, name)
            else:
                self._graded(ax, result, name)
            if onset is not None:
                if interval is not None:
                    ax.axvspan(interval[0], interval[1], color=ONSET_COLOR, alpha=.08, lw=0, zorder=0)
                ax.axvline(onset, color=ONSET_COLOR, lw=1, ls=":",
                           label=text("film onset {time} s", time=f"{onset:.1f}") if name == panels[0][0] else None)
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

    @staticmethod
    def _physical(ax, result: StructureResult, name: str) -> None:
        points = result.column(f"{name}_nm")
        if points.time:
            ax.plot(points.time, points.value, "o", ms=2.6, alpha=.3, color=COLORS["P"], lw=0,
                    label=text("{grade}: {meaning}", grade="P", meaning=tr(GRADES["P"])))
        trend = result.column(f"{name}_trend_nm")
        if trend.time:
            ax.plot(trend.time, trend.value, "-", lw=1.7, color=TREND_COLORS[name], label=tr("trend (smoothed)"))
        if name == "D":
            _direct(ax, result)

    @staticmethod
    def _graded(ax, result: StructureResult, name: str) -> None:
        for grade in ("C", "B", "A", "S", "T", "L", "P"):
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


__all__ = ["COLORS", "TrendPlot"]
