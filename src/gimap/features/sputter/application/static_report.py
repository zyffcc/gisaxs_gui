"""The report of samples measured after their deposition: one HTML file for every series of the list, and one CSV.

**What it reads.** Each series folder the analysis package wrote: ``report.csv`` (one row per sample — the median over
its frames of every quantity, with the fit deviation) and its two figures, ``report.png`` (each quantity against the
nominal thickness) and ``patterns.png`` (each sample's measured and model image, their ratio and two cuts). A series
without ``report.csv`` (nothing analysed into it yet) is left out.

**The stage** of a sample is named from its fitted structure, by one rule written in the report: without a continuous
layer under the grains (thinner than 0.3 nm) the grains are *separate islands* below a coverage of 0.8 and *grains in
contact* above; with a layer thinner than the grains stand above it, *grains on a thin layer*; with a thicker one, a
*continuous film* when its density is at least 0.95 of the bulk metal's, else a *film with a density below 1*.

**The words** are given by the caller (``say(template, **values)``: the page says them in the interface language);
numbers, units and file names stay as they are. The HTML holds its figures (PNG, base64), so it is one file to send.
"""

from __future__ import annotations

import base64
import csv
import html
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

NO_LAYER_NM = 0.3
IN_CONTACT_COVERAGE = 0.8
DENSE = 0.95
ISLANDS, IN_CONTACT, ON_LAYER, FILM, LOOSE_FILM = ("separate islands", "grains in contact", "grains on a thin layer",
                                                   "continuous film", "film with a density below 1")
STAGES = (ISLANDS, IN_CONTACT, ON_LAYER, FILM, LOOSE_FILM)
REPORT_TITLE = "Au films after their deposition: structure from GISAXS"
REPORT_FACTS = "{samples} samples in {series} series · {date} · GIMaP {version} · analysis package {package}"
REPORT_GEOMETRY = "Geometry: {geometry} · mode {mode}"
READING = ("Each value is the median over the usable frames of a sample. The fit deviation is the deviance per "
           "detector cell with a 5 % allowance for the model: up to 4 is a good fit, 4 to 10 a fair one, above 10 the "
           "model does not describe the frames and the values are only an order of magnitude.")
AMOUNT_NOTE = ("The metal amount h_m comes from the shape of the pattern, not from an absolute intensity (the incident "
               "flux is not known): within about 30 % of the nominal thickness is normal.")
STAGE_NOTE = ("Stage: without a continuous layer under the grains (thinner than 0.3 nm), separate islands below a "
              "coverage of 0.8 and grains in contact above; a layer thinner than the grains stand above it: grains on "
              "a thin layer; a thicker one: a continuous film, or a film with a density below 1 when its density is "
              "below 0.95 of the bulk metal's.")
VALIDATION_NOTE = "No independent measurement (AFM, SEM, XRR) checked these values."
SERIES_TITLE = "Series {series}"
SERIES_BARE = "Bare substrate: {bare} · folder: {folder}"
BARE_ON_SI = "none (the film directly on Si)"
FIGURE_EVOLUTION = "Each quantity against the nominal thickness"
FIGURE_PATTERNS = "Measured and model images, their ratio and two cuts, sample by sample"
HEADERS = ("Sample", "Nominal (nm)", "Stage", "h_m (nm)", "Base layer (nm)", "Base density", "Grain height T (nm)",
           "Radius R (nm)", "Spacing D (nm)", "Coverage", "Fit deviation")
REPORT_TEMPLATES = (*STAGES, REPORT_TITLE, REPORT_FACTS, REPORT_GEOMETRY, READING, AMOUNT_NOTE, STAGE_NOTE,
                    VALIDATION_NOTE, SERIES_TITLE, SERIES_BARE, BARE_ON_SI, FIGURE_EVOLUTION, FIGURE_PATTERNS, *HEADERS)
"""The words of the report (the page says them in its language)."""
COLUMNS = (("t_b", 1), ("rb", 2), ("T", 1), ("R", 1), ("d_hex", 0), ("eta", 2))
CSV_COLUMNS = ("series", "sample", "nominal_nm", "stage", "h_m_nm", "base_nm", "base_density", "grain_height_nm",
               "radius_nm", "spacing_nm", "coverage", "fit_deviation", "frames", "bare_substrate", "series_folder")


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stage(t_b, height, coverage, density) -> str:
    """The stage of a sample (``STAGES``) from its base layer t_b, the grains' height above it, their coverage and
    the base layer's relative density ("" when a value is missing)."""
    t_b, height, coverage, density = (_number(value) for value in (t_b, height, coverage, density))
    if t_b is None:
        return ""
    if t_b < NO_LAYER_NM:
        return "" if coverage is None else ISLANDS if coverage < IN_CONTACT_COVERAGE else IN_CONTACT
    if height is not None and t_b < height:
        return ON_LAYER
    return "" if density is None else FILM if density >= DENSE else LOOSE_FILM


@dataclass
class ReportSeries:
    name: str
    folder: Path
    bare: str = ""
    """The bare substrate's folder name ("": the film on Si)."""
    rows: list = field(default_factory=list)
    """One dict per sample, as ``report.csv`` has it, by nominal thickness."""
    figures: dict = field(default_factory=dict)
    """``evolution`` / ``patterns`` → PNG bytes."""


def read_series(name: str, folder, bare: str = "") -> Optional[ReportSeries]:
    """The series in ``folder`` as its report needs it (None: no ``report.csv`` there)."""
    folder = Path(folder)
    table = folder / "report.csv"
    if not table.is_file():
        return None
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (_number(row.get("nominal_nm")) or 0.0, row.get("label") or ""))
    figures = {}
    for key, file in (("evolution", "report.png"), ("patterns", "patterns.png")):
        if (folder / file).is_file():
            figures[key] = (folder / file).read_bytes()
    return ReportSeries(name, folder, bare, rows, figures)


def _shown(value, digits: int) -> str:
    number = _number(value)
    return "–" if number is None else f"{number:.{digits}f}"


def sample_stage(row: dict) -> str:
    return stage(row.get("t_b"), row.get("T"), row.get("eta"), row.get("rb"))


def write_csv(series: list, path) -> Path:
    """Every sample of every series, one row each, with units in the column names."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for one in series:
            for row in one.rows:
                writer.writerow([one.name, row.get("label", ""), row.get("nominal_nm", ""), sample_stage(row),
                                 row.get("h_m_model", ""), row.get("t_b", ""), row.get("rb", ""), row.get("T", ""),
                                 row.get("R", ""), row.get("d_hex", ""), row.get("eta", ""), row.get("fit", ""),
                                 row.get("frames", ""), one.bare, str(one.folder)])
    return path


STYLE = """
body { font-family: "Segoe UI", system-ui, -apple-system, "Microsoft YaHei", sans-serif; color: #1f2937;
       max-width: 1180px; margin: 24px auto; padding: 0 20px; line-height: 1.5; background: #f8fafc; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 28px 0 4px; color: #1d4ed8; }
.meta { color: #64748b; font-size: 13px; margin: 2px 0; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin: 12px 0; }
.notes p { margin: 4px 0; font-size: 13px; color: #334155; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 5px 8px; border-bottom: 1px solid #e2e8f0; text-align: right; white-space: nowrap; }
th { background: #f1f5f9; color: #334155; font-weight: 600; }
td.text, th.text { text-align: left; }
td.good { color: #15803d; } td.fair { color: #b45309; } td.poor { color: #b91c1c; }
figure { margin: 12px 0 0; }
figure img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px; background: #fff; }
figcaption { color: #64748b; font-size: 12px; margin-top: 4px; }
"""


def _fit_class(value) -> str:
    number = _number(value)
    return "" if number is None else "good" if number <= 4 else "fair" if number <= 10 else "poor"


def _meta(words: str) -> str:
    return f'<p class="meta">{html.escape(words)}</p>'


def report_html(series: list, say: Callable[..., str], facts: dict) -> str:
    """The HTML of the report. ``facts``: date, version, package, geometry and mode (shown as they are)."""
    escape = html.escape
    samples = sum(len(one.rows) for one in series)
    parts = ["<!doctype html>", '<html><head><meta charset="utf-8">',
             f"<title>{escape(say(REPORT_TITLE))}</title><style>{STYLE}</style></head><body>",
             f"<h1>{escape(say(REPORT_TITLE))}</h1>",
             _meta(say(REPORT_FACTS, samples=samples, series=len(series), date=facts.get("date", ""),
                       version=facts.get("version", ""), package=facts.get("package", ""))),
             _meta(say(REPORT_GEOMETRY, geometry=facts.get("geometry", ""), mode=facts.get("mode", ""))),
             '<div class="card notes">', *(f"<p>{escape(say(note))}</p>" for note in
                                          (READING, AMOUNT_NOTE, STAGE_NOTE, VALIDATION_NOTE)), "</div>"]
    for one in series:
        parts.append(f"<h2>{escape(say(SERIES_TITLE, series=one.name))}</h2>")
        parts.append(_meta(say(SERIES_BARE, bare=one.bare or say(BARE_ON_SI), folder=str(one.folder))))
        parts.append('<div class="card"><table><tr>')
        parts += [f'<th class="{"text" if index in (0, 2) else ""}">{escape(say(header))}</th>'
                  for index, header in enumerate(HEADERS)]
        parts.append("</tr>")
        for row in one.rows:
            named = sample_stage(row)
            cells = [f'<td class="text">{escape(row.get("label", ""))}</td>',
                     f'<td>{_shown(row.get("nominal_nm"), 1)}</td>',
                     f'<td class="text">{escape(say(named)) if named else "–"}</td>',
                     f'<td>{_shown(row.get("h_m_model"), 1)}</td>']
            cells += [f"<td>{_shown(row.get(key), digits)}</td>" for key, digits in COLUMNS]
            cells.append(f'<td class="{_fit_class(row.get("fit"))}">{_shown(row.get("fit"), 2)}</td>')
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</table></div>")
        for key, caption in (("evolution", FIGURE_EVOLUTION), ("patterns", FIGURE_PATTERNS)):
            if key in one.figures:
                data = base64.b64encode(one.figures[key]).decode("ascii")
                parts.append(f'<figure class="card"><img alt="{escape(key)}" src="data:image/png;base64,{data}">'
                             f"<figcaption>{escape(say(caption))}</figcaption></figure>")
    parts.append("</body></html>")
    return "\n".join(parts)


__all__ = ["CSV_COLUMNS", "FILM", "HEADERS", "IN_CONTACT", "ISLANDS", "LOOSE_FILM", "ON_LAYER", "REPORT_TEMPLATES",
           "ReportSeries", "STAGES", "read_series", "report_html", "sample_stage", "stage", "write_csv"]
