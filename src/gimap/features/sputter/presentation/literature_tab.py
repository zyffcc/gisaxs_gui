"""Literature tab: the series against published growth, over the nominal thickness δ (nm).

Four panels over δ = rate × (t − t0): (1) D — the literature method's D = 2π/q* per window (faint; crosses after
its percolation) and its trend, peaks it does not use (host, outliers and jumps, late peaks) as open circles, the
network's D and the published D(δ) curves of the metal, labelled with their citation keys; (2) 2R/D of both
methods with the published range and the percolation line 2R = D; (3) the coverage of both with the published
values; (4) the network's mass balance η = t_eq/δ. The published percolation band of the metal and host and the
literature-method percolation with its interval (the scatter of the windows and the errors of the rate and t0) are
marked on every panel; a percolation that is not determined is hatched from the earliest δ it may come at to the
end of the peak phase. The TS onset of the network's path is marked too. After the end of the deposition t_end (when
it is given) δ stays at rate × (t_end − t0): later windows are drawn at that δ, and the header says so. The method is a
check of the network's physical path (the page's default explanation). Below: the checks with their verdicts
and references, the caveats and the references with their DOI. Without a deposition rate the tab says how to
enter one. Texts are in the interface language (``refresh_language``); the page redraws on a theme switch.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSizePolicy, QSplitter, QTextBrowser, QVBoxLayout, QWidget

from src.gimap.app.presentation.i18n import tr

from ..application import StructureResult, checks_html, film_onset, percolation_errors, percolation_sentence
from .texts import note_text, text

LIT_COLOUR = "#b0651c"
NET_COLOUR = "#159a9c"
NOT_USED = "#8c8c8c"
BAND_COLOUR = "#2a8a3e"
PERC_COLOUR = "#d62728"
ONSET_COLOUR = "#6a3d9a"
CURVE_COLOURS = ("#1f5fa8", "#7f7f7f", "#c49a00", "#8e3b8e", "#3a7d44", "#5a6f8c", "#a05050")
EMPTY = "Predict a scan or open a results folder: its comparison with published growth appears here."
OTHER_VARIANT = ("The comparison with the literature uses the physical files: choose Physical path or Literature "
                 "method under More options ▸ Explanation.")
OLD_RESULT = ("This result was made before the literature method existed: enter the deposition rate again under More "
              "options (or press Predict) to add it.")
NEEDS_RATE = ("The literature method needs the deposition rate (for example from a quartz balance): enter it under More "
              "options ▸ Deposition rate (nm/s), and the start of the deposition t0 if you know it. The files are "
              "rewritten without reading the frames again. Until then only the monotony of D is tested.")
HEADER = ("<b>{metal} on {polymer}</b> · published percolation band of {host}: {low}–{high} nm · deposition rate "
          "{rate} nm/s, t0 = {t0} s")
INTERVAL = "68% interval (scatter, rate ±{rate_error}%, t0 ±{t0_error} s)"
UNDETERMINED = "percolation not determined: from δ = {low} nm or later"
T0_ASSUMED = " (assumed: the first frame)"
T_END = " · deposition ended at t_end = {t_end} s: later windows stay at δ = {delta} nm"
UNKNOWN_POLYMER = "a polymer of unknown type"
TEMPLATES = (EMPTY, OTHER_VARIANT, OLD_RESULT, NEEDS_RATE, HEADER, INTERVAL, UNDETERMINED, T0_ASSUMED, T_END,
             UNKNOWN_POLYMER)
"""Texts the tab says from a name (the interface translates them)."""


def _num(value) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fmt(value) -> str:
    x = _num(value)
    return "–" if x is None else f"{x:.3g}"


def _points(rows: list, key: str, *, when=None) -> tuple[list, list]:
    """(δ, value) of the windows with δ > 0 and a value (and ``when(row)``)."""
    xs, ys = [], []
    for r in rows:
        x, y = _num(r.get("delta_nm")), _num(r.get(key))
        if x is not None and x > 0 and y is not None and y > 0 and (when is None or when(r)):
            xs.append(x); ys.append(y)
    return xs, ys


def _plain(ax) -> None:
    """Plain numbers on the logarithmic axes (“10”, not “10¹”)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    plain = FuncFormatter(lambda value, _position: f"{value:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(plain)
        low, high = ax.get_ylim()
        ax.yaxis.set_minor_formatter(plain if low > 0 and high / low < 10 else NullFormatter())


class LiteratureTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.result: Optional[StructureResult] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.message = QLabel(EMPTY, self)
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.RichText)
        layout.addWidget(self.message)
        self.splitter = QSplitter(Qt.Vertical, self)
        self.canvas = self.figure = None
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception as exc:  # matplotlib missing: the checks stay
            self.splitter.addWidget(QLabel(text("Plotting is unavailable: {error}", error=exc)))
        else:
            self.figure = Figure(figsize=(7, 6), dpi=100, constrained_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.splitter.addWidget(self.canvas)
        self.report = QTextBrowser(self)
        self.report.setOpenExternalLinks(True)
        self.splitter.addWidget(self.report)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        layout.addWidget(self.splitter, 1)

    # -- state -------------------------------------------------------------------------------------------------
    def set_result(self, result: Optional[StructureResult]) -> None:
        self.result = result
        self.redraw()

    def refresh_language(self) -> None:
        self.redraw()

    def literature(self) -> dict:
        if self.result is None:
            return {}
        block = self.result.summary.get("literature")
        return block if isinstance(block, dict) else {}

    def redraw(self) -> None:
        result, lit = self.result, self.literature()
        if result is None:
            words = tr(EMPTY)
        elif result.variant not in ("physical", "literature"):
            words = tr(OTHER_VARIANT)
        elif not lit:
            words = tr(OLD_RESULT)
        elif lit.get("status") == "needs_rate":
            words = tr(NEEDS_RATE)
        elif lit.get("status") != "ok":
            words = note_text(lit.get("message") or "")
        else:
            words = self._header(lit)
        self.message.setText(words)
        self.report.setHtml(checks_html(result.summary, text, note_text) if lit and result is not None else "")
        self._draw(result if lit.get("status") == "ok" else None, lit)

    def _header(self, lit: dict) -> str:
        band = lit.get("band") or {}
        polymer = lit.get("polymer")
        words = text(HEADER, metal=lit.get("metal") or "?",
                     polymer=polymer if polymer and polymer != "polymer" else tr(UNKNOWN_POLYMER),
                     host=band.get("host_class") or lit.get("host_class") or "?", low=_fmt(band.get("low_nm")),
                     high=_fmt(band.get("high_nm")), rate=_fmt(lit.get("rate_nm_s")), t0=_fmt(lit.get("t0_s")))
        if lit.get("t0_assumed"):
            words += tr(T0_ASSUMED)
        rate, t0, t_end = _num(lit.get("rate_nm_s")), _num(lit.get("t0_s")) or 0.0, _num(lit.get("t_end_s"))
        if rate is not None and t_end is not None:
            words += text(T_END, t_end=_fmt(t_end), delta=_fmt(rate * (t_end - t0)))
        sentence = percolation_sentence(self.result.summary, text)
        return words + ("<br>" + sentence if sentence else "")

    # -- plots -------------------------------------------------------------------------------------------------
    def _draw(self, result: Optional[StructureResult], lit: dict) -> None:
        if self.canvas is None:
            return
        from src.gimap.app.presentation.theme.figures import restyle_figure
        from src.gimap.shared.figures import cjk_fallback

        fig = self.figure
        with warnings.catch_warnings():  # clearing shared logarithmic axes resets them to 0..1 first
            warnings.filterwarnings("ignore", message="Attempt to set non-positive xlim")
            fig.clear()
        if result is not None:
            axes = fig.subplots(2, 2, sharex=True)
            panels = (self._spacing, self._ratio, self._coverage, self._mass)
            for ax, panel in zip(axes.ravel(), panels):
                ax.set_xscale("log")
                panel(ax, result.rows, lit)
                self._marks(ax, result, lit, first=panel is self._spacing)
                ax.grid(alpha=.25)
                legend = ax.legend(fontsize=6.5, loc="best")
                if legend is not None:
                    legend.get_frame().set_alpha(.7)
            for ax in axes.ravel():
                _plain(ax)
            for ax in axes[1]:
                ax.set_xlabel(tr("δ, nominal thickness (nm)"))
        restyle_figure(fig)
        cjk_fallback(fig)
        self.canvas.draw_idle()

    @staticmethod
    def _spacing(ax, rows: list, lit: dict) -> None:
        after = lambda r: r.get("lit_status") == "after_percolation"  # noqa: E731
        x, y = _points(rows, "D_lit_nm", when=lambda r: not after(r))
        ax.plot(x, y, "o", ms=2.6, alpha=.35, color=LIT_COLOUR, lw=0, label=tr("D = 2π/q* (literature method)"))
        x, y = _points(rows, "D_lit_nm", when=after)
        if x:
            ax.plot(x, y, "x", ms=3, color=LIT_COLOUR, alpha=.6, lw=0, label=tr("2π/q* after the percolation"))
        x, y = _points(rows, "D_lit_trend_nm")
        ax.plot(x, y, "-", lw=1.6, color=LIT_COLOUR, label=tr("trend of D (non-decreasing)"))
        x, y = _points(rows, "D_corr_nm", when=lambda r: r.get("lit_status") in ("host_peak", "outlier", "late_peak"))
        if x:
            ax.plot(x, y, "o", ms=4, mfc="none", mec=NOT_USED, mew=.7, lw=0, label=tr("peaks not used (host, outlier, late)"))
        x, y = _points(rows, "network_D_nm")
        if x:
            ax.plot(x, y, ".", ms=2.5, alpha=.45, color=NET_COLOUR, lw=0, label=tr("D of the network"))
        for curve, colour in zip(lit.get("reference_curves") or [], CURVE_COLOURS):
            points = [(p[0], p[1]) for p in curve.get("points") or [] if len(p) == 2]
            if points:
                cx, cy = zip(*points)
                ax.plot(cx, cy, "s--", ms=3, lw=1, color=colour,
                        label=f"{curve.get('ref', '')} {curve.get('label', '')}".strip())
        ax.set_yscale("log")
        ax.set_ylabel("D (nm)")
        ax.set_title(tr("Spacing D against δ"), fontsize=8, loc="left")

    @staticmethod
    def _ratio(ax, rows: list, lit: dict) -> None:
        for band in lit.get("two_R_over_D_bands") or []:
            low, high = (band.get("range") or [None, None])[:2]
            if band.get("kind") == "mass balance" and low is not None and high is not None and high > low \
                    and "coalescence" in str(band.get("regime")):
                ax.axhspan(low, high, color=BAND_COLOUR, alpha=.1, lw=0,
                           label=text("published 2R/D, islands and coalescence ({refs})",
                                      refs=", ".join(band.get("refs") or [])))
        ax.axhline(1.0, color=PERC_COLOUR, lw=1, ls="--", label=tr("percolation: 2R = D"))
        x, y = _points(rows, "two_R_over_D_lit")
        ax.plot(x, y, "o", ms=2.6, alpha=.35, color=LIT_COLOUR, lw=0, label=tr("literature method"))
        x, y = _points(rows, "two_R_over_D_lit_trend")
        ax.plot(x, y, "-", lw=1.6, color=LIT_COLOUR, label=tr("literature method, trend"))
        x, y = _points(rows, "two_R_over_D_net")
        ax.plot(x, y, "o", ms=2.4, alpha=.4, color=NET_COLOUR, lw=0, label=tr("network"))
        ax.set_ylabel("2R/D")
        ax.set_title(tr("2R/D against δ"), fontsize=8, loc="left")

    @staticmethod
    def _coverage(ax, rows: list, lit: dict) -> None:
        at = (lit.get("coverage_at_percolation") or {}).get("range") or []
        if len(at) == 2:
            ax.axhspan(at[0], at[1], color=BAND_COLOUR, alpha=.1, lw=0,
                       label=text("published coverage at percolation {low}–{high}", low=_fmt(at[0]), high=_fmt(at[1])))
        first = True
        for band in lit.get("coverage_bands") or []:
            delta, value = band.get("delta_nm") or [], band.get("range") or []
            if len(delta) == 2 and len(value) == 2 and max(delta) > 0:
                where = max(.5 * (delta[0] + delta[1]), .05)
                ax.plot([where, where], value, "-", lw=2.2, color=BAND_COLOUR, alpha=.8,
                        label=tr("published coverage at its δ") if first else None)
                ax.plot([where], [.5 * (value[0] + value[1])], "D", ms=3, color=BAND_COLOUR)
                first = False
        x, y = _points(rows, "coverage_lit")
        ax.plot(x, y, "o", ms=2.6, alpha=.35, color=LIT_COLOUR, lw=0, label=tr("literature method"))
        x, y = _points(rows, "coverage_net")
        ax.plot(x, y, "o", ms=2.4, alpha=.4, color=NET_COLOUR, lw=0, label=tr("network"))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(tr("coverage"))
        ax.set_title(tr("Coverage against δ"), fontsize=8, loc="left")

    @staticmethod
    def _mass(ax, rows: list, lit: dict) -> None:
        ax.axhline(1.0, color=LIT_COLOUR, lw=1.2, label=tr("literature method: 1 by construction"))
        threshold = _num(lit.get("eta_warning"))
        if threshold is not None:
            ax.axhline(threshold, color=PERC_COLOUR, lw=1, ls=":",
                       label=text("{value}: flagged below (engineering choice)", value=_fmt(threshold)))
        x, y = _points(rows, "eta_net")
        ax.plot(x, y, "o", ms=2.4, alpha=.4, color=NET_COLOUR, lw=0, label=tr("network: η = t_eq/δ"))
        ax.set_yscale("log")
        ax.set_ylabel("η")
        ax.set_title(tr("Share of the nominal metal in the network's islands"), fontsize=8, loc="left")

    @staticmethod
    def _marks(ax, result: StructureResult, lit: dict, *, first: bool) -> None:
        band = lit.get("band") or {}
        low, high = _num(band.get("low_nm")), _num(band.get("high_nm"))
        if low is not None and high is not None:
            ax.axvspan(low, high, color=BAND_COLOUR, alpha=.07, lw=0, zorder=0,
                       label=text("published percolation, {host}", host=band.get("host_class") or "") if first else None)
        perc = lit.get("percolation") or {}
        at = _num(perc.get("delta_nm"))
        low, high = (list(perc.get("interval_delta_nm") or []) + [None, None])[:2]
        low, high, last = _num(low), _num(high), _num(perc.get("last_delta_nm"))
        if low is not None and (high is not None or last is not None):
            if at is None:  # not determined: from low on, or not at all in the peak phase
                label = text(UNDETERMINED, low=_fmt(low))
            else:
                label = text(INTERVAL, **percolation_errors(result.summary))
            ax.axvspan(max(low, 1e-3), high if high is not None else max(last, low), color=PERC_COLOUR, alpha=.1,
                       lw=0, zorder=0, hatch="//" if high is None else None, label=label if first else None)
        if at is not None:
            ax.axvline(at, color=PERC_COLOUR, lw=1, ls="--",
                       label=text("percolation (literature method) δ = {delta} nm", delta=_fmt(at)) if first else None)
        rate, t0, t_end = _num(lit.get("rate_nm_s")), _num(lit.get("t0_s")) or 0.0, _num(lit.get("t_end_s"))
        onset, _interval = film_onset(result.summary)
        if rate and onset is not None and onset > t0:
            delta = rate * ((onset if t_end is None else min(onset, t_end)) - t0)  # δ stays after the deposition
            ax.axvline(delta, color=ONSET_COLOUR, lw=1, ls=":",
                       label=text("TS onset of the network path δ = {delta} nm", delta=_fmt(delta)) if first else None)


__all__ = ["LiteratureTab", "TEMPLATES"]
