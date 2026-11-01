"""The GISAXS part of the report as Markdown: the cut, the halves, the length scale 2π/q* and the fit."""

from __future__ import annotations

import math
from pathlib import Path

from .gisaxs_choice import LENGTH_SCALE, POOR_CHI2, bounds_text, chosen


def pixel_span(band, limit=None, pixels=None) -> str:
    """The detector pixels a cut uses, both ends included, as Analyze's Cuts card counts them (``605–609``).

    ``band``: the band's continuous edges ``(low, high)`` (``horizontal_rows``); the cut reduces the whole
    pixels ``floor(low)`` up to ``ceil(high)`` (exclusive), within ``0…limit`` (the image's rows or columns).
    ``pixels``: ``[first, last]``, both included, when Analyze's status names them (``horizontal_pixel_rows``,
    ``vertical_pixel_columns``); they win.
    """
    try:
        if pixels is not None:
            start, stop = int(pixels[0]), int(pixels[1]) + 1
        else:
            low, high = float(band[0]), float(band[1])
            start, stop = max(0, math.floor(low)), math.ceil(high)
            if limit:
                stop = min(int(limit), stop)
    except (TypeError, ValueError, IndexError):
        return "?"
    return f"{start}–{max(start, stop - 1)}"


def image_shape(status) -> tuple:
    """(rows, columns) of the frame a status describes (``(None, None)`` when it does not say)."""
    shape = (status or {}).get("shape") or ()
    return (shape[0], shape[1]) if len(shape) == 2 else (None, None)


def cut_spans(cuts: dict, status) -> tuple[str, str]:
    """(rows of the horizontal cut, columns of the vertical cut), both ends included, as Analyze counts them.

    ``cuts``: the status's ``gisaxs`` part; the pixels it names (``horizontal_pixel_rows``,
    ``vertical_pixel_columns``: first and last, both included) win over the band edges, which are clipped
    to the frame of ``status``.
    """
    cuts = cuts or {}
    rows, columns = image_shape(status)
    return (
        pixel_span(cuts.get("horizontal_rows"), rows, cuts.get("horizontal_pixel_rows")),
        pixel_span(cuts.get("vertical_columns"), columns, cuts.get("vertical_pixel_columns")),
    )


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _pixel(value) -> str:
    """A detector position to 0.01 px, as the decisions say it (four significant digits would round a column
    such as 597.263 to 597.3, and 1234.56 to 1235)."""
    return "—" if value is None else f"{float(value):.2f}"


FIT_DECIMALS = {"R": 2, "h": 2, "D": 1, "sigma_R": 3, "sigma_h": 3, "sigma_D": 3, "chi2": 2, "distance_nm": 1}
"""The decimals of each fitted quantity in the tables, the same in every row (R 8.60 next to R 9.44, not R 8.6)."""


def _fixed(value, key: str) -> str:
    return "—" if value is None else f"{float(value):.{FIT_DECIMALS[key]}f}"


def _parameters(component: dict) -> str:
    names = (("R", "R"), ("sigma_R", "σR/R"), ("h", "h"), ("sigma_h", "σh/h"), ("D", "D"), ("sigma_D", "σD/D"))
    units = {"R": " nm", "h": " nm", "D": " nm"}
    return ", ".join(
        f"{label} {_fixed(component[key], key)}{units.get(key, '')}" for key, label in names if component.get(key) is not None
    )


def fit_rows(fit: dict) -> list[str]:
    """The solutions as a Markdown table (units in the header); the values at a search bound named in the note,
    the solution the run reports marked "(chosen)"."""
    solutions = fit.get("solutions") or []
    if not solutions:
        return ["No fit solution."]
    index, _solution = chosen(fit)
    lines = ["| rank | model | parameters | χ²/N | note |", "|---|---|---|---|---|"]
    for number, row in enumerate(solutions):
        component = (row.get("components") or [{}])[0]
        notes = [bounds_text(row)] + list(row.get("warnings") or ()) + ([] if row.get("converged", True) else ["not converged"])
        rank = f"{row['rank']} (chosen)" if number == index else row["rank"]
        lines.append(f"| {rank} | {row['model']} | {_parameters(component)} | {_fixed(row['chi2'], 'chi2')} | "
                     f"{'; '.join(note for note in notes if note)} |")
    return lines


def gisaxs_sections(report: dict) -> list[str]:
    gisaxs = report.get("gisaxs") or {}
    cuts = gisaxs.get("cuts") or {}
    lines = ["## 切线 / Cuts", ""]
    # The pixels each cut uses, both ends included (as Analyze's Cuts card and the Results tab say them).
    rows, columns = cut_spans(cuts, (report.get("tables") or {}).get("status"))
    source = {"yoneda": "at the Yoneda band", "horizon": "just above the horizon (no Yoneda band found)",
              "series": "at the cut of the series"}.get(
        cuts.get("horizontal_source"), "set by hand"
    )
    alpha = cuts.get("yoneda_alpha_f_deg")
    lines.append(
        f"- horizontal cut I(qy) {source}: rows {rows}"
        + (f", αf = {_number(alpha, 3)}°" if alpha is not None else "")
    )
    lines.append(f"- vertical cut I(qz): columns {columns}")
    symmetry = gisaxs.get("symmetry")
    if symmetry:
        lines.append(
            f"- symmetry axis x = {_pixel(symmetry.get('x_px'))} px (calibrated {_pixel(symmetry.get('initial_x_px'))} px; "
            f"asymmetry {_number(symmetry.get('loss_before'), 3)} → {_number(symmetry.get('loss_after'), 3)})"
        )
    halves = gisaxs.get("halves")
    if halves:
        lines.append(f"- halves: **{halves['side']}** — {halves['reason']}")
    spacing = gisaxs.get("spacing")
    lines += ["", "## 特征长度 / Length scale 2π/q*", ""]
    if spacing is None:
        lines.append("No side maximum or shoulder away from qy = 0: the cut gives no length scale 2π/q* in the q range.")
    else:
        where = ("at the peak of the cut" if spacing.get("kind") == "maximum"
                 else "at a shoulder of the cut (a change of slope, not a resolved peak)")
        lines.append(f"- 2π/q* = {_number(spacing['distance_nm'], 3)} nm {where}, |qy| = {_number(spacing['q'])} Å⁻¹: "
                     f"{LENGTH_SCALE}.")
    fit = gisaxs.get("fit")
    lines += ["", "## 拟合 / Fit of I(qy)", ""]
    if not fit:
        lines.append("Not fitted here: Send to Fitting in GIMaP fits the prepared curve with a chosen model.")
    else:
        lines += [
            f"Curve: {fit.get('curve')} ({fit.get('points')} points, |qy| {_number((fit.get('q_range_inv_angstrom') or [None])[0], 3)}–"
            f"{_number((fit.get('q_range_inv_angstrom') or [None, None])[1], 3)} Å⁻¹).",
            "",
            *fit_rows(fit),
            "",
            "Numerical fits of single particle families with size dispersity and a paracrystal distance D, ranked by "
            "the mean Huber loss (δ = 2) of the σ-normalised residuals r; χ²/N is the mean of r² over the points (no "
            "degrees-of-freedom correction). χ²/N close to each other means the curve alone does not decide the "
            "model: choose it from what is known of the sample. A non-linear value at a search bound (the note "
            "column) is not determined by the curve: among the solutions within 10 % of the first-ranked χ²/N the "
            "run reports (chosen) the best-ranked one that converged with no non-linear value at a bound. The "
            "particle amplitudes, the background and the beam-tail amplitude are solved non-negative and are not "
            f"counted; one at 0 is named in the Decisions (fit caveat). A reported solution with χ²/N above "
            f"{POOR_CHI2:g} is a poor fit (Needs attention). A screening fit: confirm the model in Fitting.",
        ]
    return lines + [""]


def gisaxs_batch_row(report: dict) -> tuple[str, str, str]:
    """(halves, 2π/q*, the fit the run reports) of one frame for the batch table."""
    gisaxs = report.get("gisaxs") or {}
    halves = (gisaxs.get("halves") or {}).get("side") or "—"
    spacing = gisaxs.get("spacing")
    spacing_text = "—" if not spacing else _fixed(spacing["distance_nm"], "distance_nm") + (
        "" if spacing.get("kind") == "maximum" else " (shoulder)")
    index, best = chosen(gisaxs.get("fit"))
    if best is None:
        return halves, spacing_text, "—"
    component = (best.get("components") or [{}])[0]
    text = (f"{best['model']} R {_fixed(component.get('R'), 'R')}, D {_fixed(component.get('D'), 'D')} nm, "
            f"χ²/N {_fixed(best['chi2'], 'chi2')}")
    return halves, spacing_text, text + (f" (rank {best['rank']})" if index else "") + (
        f" ⚠ {bounds_text(best)}" if bounds_text(best) else "")


def gisaxs_batch_table(reports: list[dict], state_of) -> list[str]:
    lines = [
        "| frame | status | halves | 2π/q* (nm) | chosen fit | open questions |",
        "|---|---|---|---|---|---|",
    ]
    for report in reports:
        halves, spacing, fit = gisaxs_batch_row(report)
        attention = report.get("needs_attention") or []
        lines.append(
            f"| {Path(report.get('frame') or '').name} | {state_of(report)} | {halves} | {spacing} | {fit} | "
            f"{', '.join(item['item'] for item in attention) or '—'} |"
        )
    return lines


__all__ = [
    "cut_spans", "fit_rows", "gisaxs_batch_row", "gisaxs_batch_table", "gisaxs_sections", "image_shape", "pixel_span",
]
