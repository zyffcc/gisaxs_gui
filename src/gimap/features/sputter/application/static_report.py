"""The report of samples measured after their deposition: one HTML file for every series of the list, and one CSV.

**What it reads.** Each series folder the analysis package wrote: ``report.csv`` (one row per sample — the median over
its frames of every quantity, with the fit deviation) and its figure of the fits, ``patterns.png`` (each sample's
measured and model image, their ratio and two cuts). A series without ``report.csv`` (nothing analysed into it yet) is
left out. The figure of the evolution is drawn by the caller (``figures``), in the two parts below.

**Two parts.** A sample is a *connected film* by the rule of the main screen (``domain/real_space.py``, the Real Space
picture and the captions of the value tiles): a continuous metal layer under the grains, or grains joined into a
network across the picture. A connected film has no separate particles: its radius and spacing are those of the relief
of its surface, and the report gives it the film's own values — the thickness and density of the layer, its height
and the relief. The other samples are *separate particles*: radius, height, spacing, the gap between neighbours and
the coverage. Each series has a table for each part it has.

**The stage** of a sample is named with the same picture: *separate islands* (no two grains touch), *grains in
contact* (they touch, no cluster crosses the picture), *grains joined into a network* (a cluster crosses it); with a
continuous layer, *grains on a thin layer* while the layer is thinner than the grains stand above it, else a
*continuous film* — a *film with a density below 1* when its density is below 0.95 of the bulk metal's.

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

from ..domain import arrangement, grown, real_space, top_grains, visible

DENSE = 0.95
ISLANDS, IN_CONTACT, NETWORK = "separate islands", "grains in contact", "grains joined into a network"
ON_LAYER, FILM, LOOSE_FILM = "grains on a thin layer", "continuous film", "film with a density below 1"
STAGES = (ISLANDS, IN_CONTACT, NETWORK, ON_LAYER, FILM, LOOSE_FILM)
JOINED_STAGES = (NETWORK, ON_LAYER, FILM, LOOSE_FILM)
REPORT_TITLE = "Au films after their deposition: structure from GISAXS"
REPORT_FACTS = "{samples} samples in {series} series · {date} · GIMaP {version} · analysis package {package}"
REPORT_GEOMETRY = "Geometry: {geometry} · mode {mode}"
READING = ("Each value is the median over the usable frames of a sample. The fit deviation is the deviance per "
           "detector cell with a 5 % allowance for the model: up to 4 is a good fit, 4 to 10 a fair one, above 10 the "
           "model does not describe the frames and the values are only an order of magnitude.")
AMOUNT_NOTE = ("The metal amount h_m comes from the shape of the pattern, not from an absolute intensity (the incident "
               "flux is not known): within about 30 % of the nominal thickness is normal.")
PARTS_NOTE = ("Each series is in two parts. Separate particles: the grains stand apart or touch without joining "
              "across the picture — their radius, height, spacing and the gap between them. Connected film: a "
              "continuous layer under the grains, or grains joined into a network — there are no separate particles "
              "any more, so the film's thickness, density and height are given, and the radius and spacing are those "
              "of the relief of its surface.")
STAGE_NOTE = ("Stage, as the Real Space picture of the main screen draws the solution: separate islands, grains in "
              "contact (touching, no cluster across the picture) or grains joined into a network (a cluster across "
              "it); with a continuous layer, grains on a thin layer while the layer is thinner than the grains stand "
              "above it, else a continuous film — a film with a density below 1 when its density is below 0.95 of the "
              "bulk metal's.")
VALIDATION_NOTE = "No independent measurement (AFM, SEM, XRR) checked these values."
SERIES_TITLE = "Series {series}"
SERIES_BARE = "Bare substrate: {bare} · folder: {folder}"
BARE_ON_SI = "none (the film directly on Si)"
PART_PARTICLES = "Separate particles"
PART_FILM = "Connected film"
JOINS_AT = "The film is connected from {nominal} nm on."
FIGURE_EVOLUTION = "The two parts against the nominal thickness"
FIGURE_PATTERNS = "Measured and model images, their ratio and two cuts, sample by sample"
PARTICLE_HEADERS = ("Sample", "Nominal (nm)", "Stage", "h_m (nm)", "Radius R (nm)", "Height H (nm)", "Spacing D (nm)",
                    "Gap D − 2R (nm)", "Coverage", "Fit deviation")
FILM_HEADERS = ("Sample", "Nominal (nm)", "Stage", "h_m (nm)", "Film thickness (nm)", "Film density",
                "Film height (nm)", "Relief height (nm)", "Relief radius (nm)", "Relief spacing (nm)", "Fit deviation")
REPORT_TEMPLATES = (*STAGES, REPORT_TITLE, REPORT_FACTS, REPORT_GEOMETRY, READING, AMOUNT_NOTE, PARTS_NOTE, STAGE_NOTE,
                    VALIDATION_NOTE, SERIES_TITLE, SERIES_BARE, BARE_ON_SI, PART_PARTICLES, PART_FILM, JOINS_AT,
                    FIGURE_EVOLUTION, FIGURE_PATTERNS, *PARTICLE_HEADERS, *FILM_HEADERS)
"""The words of the report (the page says them in its language)."""
CSV_COLUMNS = ("series", "sample", "nominal_nm", "part", "stage", "h_m_nm", "radius_nm", "height_nm", "spacing_nm",
               "gap_nm", "coverage", "film_thickness_nm", "film_density", "relief_height_nm", "fit_deviation",
               "frames", "bare_substrate", "series_folder")


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stage(radius, spacing, height, base, relief=None, density=None) -> str:
    """The stage of a sample (``STAGES``) from its grain radius and spacing, the height of the metal (base layer and
    grains), the base layer, the grains' height above it and the layer's relative density ("" without a radius and a
    spacing)."""
    space = real_space({"R_nm": radius, "D_nm": spacing, "H_nm": height, "base_nm": base})
    if space is None:
        return ""
    if space.continuous:
        above = _number(relief)
        above = space.grain_height if above is None else above
        if space.base < above:
            return ON_LAYER
        density = _number(density)
        return LOOSE_FILM if density is not None and density < DENSE else FILM
    kind = arrangement(space, visible(space, top_grains(space)))
    return {"islands": ISLANDS, "coalescing": IN_CONTACT}.get(kind, NETWORK)


def sample_stage(row: dict) -> str:
    """The stage of a sample; within a series read by ``read_series`` the one of the growing film (``series_stages``)."""
    if row.get("stage_in_series"):
        return row["stage_in_series"]
    return stage(row.get("R"), row.get("d_hex"), row.get("height"), row.get("t_b"), row.get("T"), row.get("rb"))


def series_stages(rows: list) -> list:
    """The stage of every sample of a series (rows by nominal thickness) as the main screen's real-space picture draws
    them: the samples as one growing film (``grown``), so whether grains are separate, touch or form a network follows
    the same grains from sample to sample. A sample with a continuous layer keeps its own stage."""
    spaces = [real_space({"R_nm": row.get("R"), "D_nm": row.get("d_hex"), "H_nm": row.get("height"),
                          "base_nm": row.get("t_b")}) for row in rows]
    stages = []
    for index, row in enumerate(rows):
        own = stage(row.get("R"), row.get("d_hex"), row.get("height"), row.get("t_b"), row.get("T"), row.get("rb"))
        growing = grown(spaces, index) if own in (ISLANDS, IN_CONTACT, NETWORK) else None
        if growing is not None:
            own = {"islands": ISLANDS, "coalescing": IN_CONTACT}.get(arrangement(*growing), NETWORK)
        stages.append(own)
    return stages


def joined(row: dict) -> bool:
    """The sample is a connected film (no separate particles)."""
    return sample_stage(row) in JOINED_STAGES


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

    def particles(self) -> list:
        return [row for row in self.rows if not joined(row)]

    def film(self) -> list:
        return [row for row in self.rows if joined(row)]


def read_series(name: str, folder, bare: str = "") -> Optional[ReportSeries]:
    """The series in ``folder`` as its report needs it (None: no ``report.csv`` there)."""
    folder = Path(folder)
    table = folder / "report.csv"
    if not table.is_file():
        return None
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (_number(row.get("nominal_nm")) or 0.0, row.get("label") or ""))
    for row, named in zip(rows, series_stages(rows)):
        row["stage_in_series"] = named
    figures = {}
    if (folder / "patterns.png").is_file():
        figures["patterns"] = (folder / "patterns.png").read_bytes()
    return ReportSeries(name, folder, bare, rows, figures)


def _shown(value, digits: int) -> str:
    number = _number(value)
    return "–" if number is None else f"{number:.{digits}f}"


def gap(row: dict) -> Optional[float]:
    """The gap between neighbouring grains, D − 2R (never below 0), nm."""
    radius, spacing = _number(row.get("R")), _number(row.get("d_hex"))
    return None if radius is None or spacing is None else max(spacing - 2.0 * radius, 0.0)


def write_csv(series: list, path) -> Path:
    """Every sample of every series, one row each, with units in the column names: a connected film's radius and
    spacing are those of its relief."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for one in series:
            for row in one.rows:
                film = joined(row)
                writer.writerow([one.name, row.get("label", ""), row.get("nominal_nm", ""),
                                 "connected film" if film else "separate particles", sample_stage(row),
                                 row.get("h_m_model", ""), row.get("R", ""), row.get("height", ""),
                                 row.get("d_hex", ""), "" if film else gap(row), "" if film else row.get("eta", ""),
                                 row.get("t_b", "") if film else "", row.get("rb", "") if film else "",
                                 row.get("T", "") if film else "", row.get("fit", ""), row.get("frames", ""),
                                 one.bare, str(one.folder)])
    return path


STYLE = """
body { font-family: "Segoe UI", system-ui, -apple-system, "Microsoft YaHei", sans-serif; color: #1f2937;
       max-width: 1180px; margin: 24px auto; padding: 0 20px; line-height: 1.5; background: #f8fafc; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 28px 0 4px; color: #1d4ed8; }
h3 { font-size: 15px; margin: 14px 0 4px; color: #334155; }
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


def _table(rows: list, headers: tuple, cells, say) -> list:
    escape = html.escape
    parts = ['<div class="card"><table><tr>']
    parts += [f'<th class="{"text" if index in (0, 2) else ""}">{escape(say(header))}</th>'
              for index, header in enumerate(headers)]
    parts.append("</tr>")
    for row in rows:
        named = sample_stage(row)
        line = [f'<td class="text">{escape(row.get("label", ""))}</td>', f'<td>{_shown(row.get("nominal_nm"), 1)}</td>',
                f'<td class="text">{escape(say(named)) if named else "–"}</td>',
                f'<td>{_shown(row.get("h_m_model"), 1)}</td>']
        line += [f"<td>{shown}</td>" for shown in cells(row)]
        line.append(f'<td class="{_fit_class(row.get("fit"))}">{_shown(row.get("fit"), 2)}</td>')
        parts.append("<tr>" + "".join(line) + "</tr>")
    parts.append("</table></div>")
    return parts


def _particle_cells(row: dict) -> list:
    return [_shown(row.get("R"), 1), _shown(row.get("height"), 1), _shown(row.get("d_hex"), 0),
            _shown(gap(row), 1), _shown(row.get("eta"), 2)]


def _film_cells(row: dict) -> list:
    return [_shown(row.get("t_b"), 1), _shown(row.get("rb"), 2), _shown(row.get("height"), 1),
            _shown(row.get("T"), 1), _shown(row.get("R"), 1), _shown(row.get("d_hex"), 0)]


def _figure(png: bytes, caption: str, key: str) -> str:
    data = base64.b64encode(png).decode("ascii")
    return (f'<figure class="card"><img alt="{html.escape(key)}" src="data:image/png;base64,{data}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


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
                                          (READING, PARTS_NOTE, AMOUNT_NOTE, STAGE_NOTE, VALIDATION_NOTE)), "</div>"]
    for one in series:
        parts.append(f"<h2>{escape(say(SERIES_TITLE, series=one.name))}</h2>")
        parts.append(_meta(say(SERIES_BARE, bare=one.bare or say(BARE_ON_SI), folder=str(one.folder))))
        film = one.film()
        if film:
            parts.append(_meta(say(JOINS_AT, nominal=_shown(film[0].get("nominal_nm"), 1))))
        if "evolution" in one.figures:
            parts.append(_figure(one.figures["evolution"], say(FIGURE_EVOLUTION), "evolution"))
        for title, rows, headers, cells in ((PART_PARTICLES, one.particles(), PARTICLE_HEADERS, _particle_cells),
                                            (PART_FILM, film, FILM_HEADERS, _film_cells)):
            if rows:
                parts.append(f"<h3>{escape(say(title))}</h3>")
                parts += _table(rows, headers, cells, say)
        if "patterns" in one.figures:
            parts.append(_figure(one.figures["patterns"], say(FIGURE_PATTERNS), "patterns"))
    parts.append("</body></html>")
    return "\n".join(parts)


__all__ = ["CSV_COLUMNS", "FILM", "FILM_HEADERS", "IN_CONTACT", "ISLANDS", "JOINED_STAGES", "LOOSE_FILM", "NETWORK",
           "ON_LAYER", "PARTICLE_HEADERS", "REPORT_TEMPLATES", "ReportSeries", "STAGES", "gap", "joined",
           "read_series", "report_html", "sample_stage", "stage", "write_csv"]
