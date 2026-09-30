"""The guided analysis as one web page to keep or send: the report, the q map and I(q) as pictures.

Everything is in the file (pictures as data URIs), so it opens in any browser without GIMaP.
The text is the same Markdown report the page shows, converted by Qt.
"""

from __future__ import annotations

import base64
import html
import time
from typing import Callable, Optional

from PyQt5.QtGui import QTextDocument

from ..application import peak_markers, pipeline_markdown, ring_overlays, series_markdown

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


def report_html(
    report: dict,
    markdown: str,
    *,
    q_map: Optional[bytes] = None,
    curve: Optional[bytes] = None,
    title: str = "GIWAXS report",
) -> str:
    """One self-contained page: header, the two pictures, then the report text."""
    frame = str(report.get("frame") or "")
    made = time.strftime("%Y-%m-%d %H:%M")
    return "\n".join((
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="muted">{html.escape(frame)} · made by GIMaP (Guided GIWAXS, no AI) on {made}</p>',
        _figure(q_map, "The q map of the analysed frames with the analysed rings: white = measured, "
                       "orange = in a shadow, red dashed = not measured."),
        _figure(curve, "I(q) of the whole detector with the peaks: red dashed = reliable, grey dotted = "
                       "flagged (spike, halo, weak)."),
        _markdown_html(markdown),
        "</body></html>",
    ))


def report_markdown(report: dict, start_report: Optional[dict] = None) -> str:
    """The report text, with the start-versus-end comparison when the start of the series was analysed."""
    text = pipeline_markdown(report)
    if start_report is not None:
        text += "\n" + series_markdown(start_report, report)
    return text


def report_page(report: dict, start_report: Optional[dict], automation: Callable[[], object]) -> str:
    """The web page, with the pictures drawn by Analyze (``AnalyzeAutomation``) on the GUI thread."""
    pictures = {}
    for name, draw in (
        ("q_map", lambda page: page.preview_png(1000, rings=ring_overlays(report))),
        ("curve", lambda page: page.curve_png("radial", peak_markers(report), 1000)),
    ):
        try:
            pictures[name] = draw(automation())
        except Exception:  # a page without a picture is still a report
            pictures[name] = None
    return report_html(report, report_markdown(report, start_report), **pictures)


__all__ = ["report_html", "report_markdown", "report_page"]
