"""The automatic analysis as one web page to keep or send: the report and its pictures.

Everything is in the file (pictures as data URIs), so it opens in any browser without GIMaP.
The text is the same Markdown report the page shows, converted by Qt. The pictures follow the
procedure: GIWAXS gets the q map with the analysed rings and I(q) with its peaks; GISAXS gets the
q map and the fit of the horizontal cut (the data and the best fit on |qy|, log–log, as in the
Results tab), or the horizontal cut itself when it was not fitted; Find Geometry (no technique
analysed) gets a “Geometry report” with the q map only.
"""

from __future__ import annotations

import base64
import html
import io
import time
from typing import Callable, Optional

import numpy as np
from PyQt5.QtGui import QTextDocument

from ..application import fit_on, peak_markers, pipeline_markdown, ring_overlays, series_markdown

STYLE = """
body { font-family: system-ui, "Segoe UI", sans-serif; max-width: 980px; margin: 24px auto; padding: 0 16px;
       color: #1d1d1f; background: #ffffff; line-height: 1.45; }
h1 { font-size: 1.5em; margin-bottom: 0.2em; }
.muted { color: #666666; font-size: 0.9em; }
figure { margin: 18px 0; }
figure img { max-width: 100%; border: 1px solid #dddddd; }
figcaption { color: #666666; font-size: 0.85em; }
table { border-collapse: collapse; margin: 8px 0; }
td, th { border: 1px solid #cccccc; padding: 3px 8px; text-align: left; }
"""
SUBTITLE = "Automatic analysis, no AI"
CAPTIONS = {
    "giwaxs": (
        "The q map of the analysed frames with the analysed rings: white = measured, orange = in a shadow, "
        "red dashed = not measured.",
        "I(q) of the whole detector with the peaks: red dashed = reliable, grey dotted = flagged (spike, halo, weak).",
    ),
    "gisaxs": (
        "The q map (q∥–qz, log intensity) of the analysed frames.",
        "The fitted horizontal cut I(|qy|) on log–log axes: the data (points) and the best fit (line).",
    ),
    "geometry": ("The q map of the frame with the geometry found.", ""),
}
TITLES = {"gisaxs": "GISAXS report", "giwaxs": "GIWAXS report", "geometry": "Geometry report"}
_GIWAXS_HEADING = "# GIWAXS — "
_GIWAXS_CLOSING = "Sizes are Scherrer lower bounds"
GISAXS_CUT_CAPTION = "The horizontal cut I(qy) at the Yoneda band (not fitted here)."
NO_RINGS_CAPTION = "The q map of the analysed frames."
FIT_COLOURS = ("#1f4e79", "#c62828")


def _markdown_html(text: str) -> str:
    """The body of Qt's HTML for ``text`` (Markdown)."""
    document = QTextDocument()
    document.setMarkdown(text)
    page = document.toHtml()
    start = page.find(">", page.find("<body")) + 1
    end = page.rfind("</body>")
    return page[start:end] if start > 0 and end > start else html.escape(text)


def _figure(png: Optional[bytes], caption: str) -> str:
    if not png:
        return ""
    data = base64.b64encode(png).decode("ascii")
    return (
        f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def procedure_of(report: dict) -> str:
    """``gisaxs``, ``giwaxs`` or ``geometry`` (Find Geometry: no technique analysed, on any frame)."""
    procedure = report.get("procedure")
    return procedure if procedure in ("gisaxs", "geometry") else "giwaxs"


def report_html(
    report: dict,
    markdown: str,
    *,
    q_map: Optional[bytes] = None,
    curve: Optional[bytes] = None,
    procedure: Optional[str] = None,
    title: Optional[str] = None,
    captions: Optional[tuple[str, str]] = None,
) -> str:
    """One self-contained page: header, the two pictures, then the report text.

    ``procedure`` (``gisaxs`` | ``giwaxs`` | ``geometry``, default from the report) chooses the title and the captions.
    """
    procedure = procedure or procedure_of(report)
    title = title or TITLES.get(procedure, f"{procedure.upper()} report")
    q_caption, curve_caption = captions or CAPTIONS.get(procedure, CAPTIONS["giwaxs"])
    frame = str(report.get("frame") or "")
    made = time.strftime("%Y-%m-%d %H:%M")
    return "\n".join((
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="muted">{html.escape(frame)} · {SUBTITLE} · GIMaP, {made}</p>',
        _figure(q_map, q_caption),
        _figure(curve, curve_caption),
        _markdown_html(markdown),
        "</body></html>",
    ))


def report_markdown(report: dict, start_report: Optional[dict] = None) -> str:
    """The report text, with the start-versus-end comparison when the start of the series was analysed."""
    text = pipeline_markdown(report)
    if report.get("procedure") == "geometry":
        text = _geometry_markdown(text)
    if start_report is not None:
        text += "\n" + series_markdown(start_report, report)
    return text


def _geometry_markdown(text: str) -> str:
    """Find Geometry's report: titled “Geometry”, without the closing note about peak sizes and phases.

    ``pipeline_markdown`` (also the command line's report, left as it is) titles every report that is not
    GISAXS “GIWAXS”, though a geometry-only run analysed neither technique.
    """
    lines = text.rstrip("\n").split("\n")
    if lines and lines[0].startswith(_GIWAXS_HEADING):
        lines[0] = "# Geometry — " + lines[0][len(_GIWAXS_HEADING):]
    if lines and lines[-1].startswith(_GIWAXS_CLOSING):
        lines.pop()
    return "\n".join(lines).rstrip("\n") + "\n"


def fit_png(report: dict, width: int = 1000) -> Optional[bytes]:
    """The GISAXS fit as in the Results tab: the fitted data and the best fit on |qy| (Å⁻¹), log–log."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fit = (report.get("gisaxs") or {}).get("fit") or {}
    solutions = fit.get("solutions") or []
    data = fit.get("data") or {}
    q = np.abs(np.asarray(data.get("q_inv_angstrom") or (), dtype=float))
    intensity = np.asarray(data.get("intensity") or (), dtype=float)
    if not solutions or q.size < 2 or q.size != intensity.size:
        return None
    order = np.argsort(q)
    q, intensity = q[order], intensity[order]
    best = (fit.get("curves") or [None])[0] or fit.get("best_curve")
    model = fit_on(q, best)
    shown = np.isfinite(q) & np.isfinite(intensity) & (q > 0) & (intensity > 0)
    if not shown.any():
        return None
    inches = width / 100.0
    figure = Figure(figsize=(inches, inches * 0.5), dpi=100, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    axes.loglog(q[shown], intensity[shown], ".", color=FIT_COLOURS[0], markersize=2.5, label="data")
    fitted = shown & np.isfinite(model) & (model > 0)
    if fitted.any():
        top = solutions[0]
        component = (top.get("components") or [{}])[0]
        radius = component.get("R")
        name = str(top.get("model") or "").replace("_", " ")
        detail = f" R {float(radius):.3g} nm" if radius is not None else ""
        axes.loglog(q[fitted], model[fitted], "-", color=FIT_COLOURS[1], linewidth=1.4,
                    label=f"best fit: {name}{detail} (χ² {float(top.get('chi2') or 0):.3g})")
    axes.set_xlabel("|qy| (Å⁻¹)")
    axes.set_ylabel("Intensity")
    axes.set_title(str(fit.get("curve") or "horizontal cut"), fontsize=9)
    axes.legend(fontsize=8, loc="best")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()


def report_page(report: dict, start_report: Optional[dict], automation: Callable[[], object]) -> str:
    """The web page, with the pictures drawn on the GUI thread (``AnalyzeAutomation`` for the frame's own)."""
    procedure = procedure_of(report)
    rings = ring_overlays(report)
    pictures: dict = {}
    captions = list(CAPTIONS[procedure])
    if procedure == "gisaxs":
        drawers = (
            ("q_map", lambda page: page.preview_png(1000, rings=())),
            ("curve", lambda page: fit_png(report, 1000)),
        )
    elif procedure == "geometry":  # Find Geometry: the frame in q, no curve analysed
        drawers = (("q_map", lambda page: page.preview_png(1000, rings=())),)
    else:
        drawers = (
            ("q_map", lambda page: page.preview_png(1000, rings=rings)),
            ("curve", lambda page: page.curve_png("radial", peak_markers(report), 1000)),
        )
        if not rings:
            captions[0] = NO_RINGS_CAPTION
    for name, draw in drawers:
        try:
            pictures[name] = draw(automation())
        except Exception:  # a page without a picture is still a report
            pictures[name] = None
    if procedure == "gisaxs" and pictures.get("curve") is None:
        try:
            pictures["curve"] = automation().curve_png("horizontal", (), 1000)
        except Exception:
            pictures["curve"] = None
        captions[1] = GISAXS_CUT_CAPTION
    return report_html(report, report_markdown(report, start_report), procedure=procedure,
                       captions=(captions[0], captions[1]), **pictures)


__all__ = ["fit_png", "procedure_of", "report_html", "report_markdown", "report_page"]
