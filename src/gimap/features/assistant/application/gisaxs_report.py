"""The GISAXS part of the report as Markdown: the cut, the halves, the length scale 2π/q* and the fit."""

from __future__ import annotations

import math
from pathlib import Path

from src.gimap.shared import fit_names

from .gisaxs_choice import ASK_PROBABILITY, LENGTH_SCALE, POOR_CHI2, bounds_text, chance, chosen, probability, sizes
from .report_language import ENGLISH, heading


ASYMMETRY_DEFINITION = (
    "asymmetry: the mean difference between the two halves of the cut mirrored about the axis — of asinh(I / scale) "
    "at equal distances left and right, large differences counted linearly, weighted by 1 / distance; 0 when the "
    "halves are equal, before → after the axis was refined (a score, not a probability or an uncertainty)"
)


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


FIT_DECIMALS = {"R": 2, "h": 2, "D": 1, "sigma_R": 3, "sigma_h": 3, "sigma_D": 3, "sigma_nn": 3, "chi2": 2,
                "distance_nm": 1}
"""The decimals of each fitted quantity in the tables, the same in every row (R 8.60 next to R 9.44, not R 8.6)."""


def _fixed(value, key: str) -> str:
    return "—" if value is None else f"{float(value):.{FIT_DECIMALS[key]}f}"


def _parameters(component: dict) -> str:
    # The disorder of the distance under the name of Fitting's Model table, σnn/D (a row of an earlier version has
    # only σD/D = σnn/D / √(2π), and keeps that name).
    disorder = ("sigma_nn", "σnn/D") if component.get("sigma_nn") is not None else ("sigma_D", "σD/D")
    names = (("R", "R"), ("sigma_R", "σR/R"), ("h", "h"), ("sigma_h", "σh/h"), ("D", "D"), disorder)
    units = {"R": " nm", "h": " nm", "D": " nm"}
    return ", ".join(
        f"{label} {_fixed(component[key], key)}{units.get(key, '')}" for key, label in names if component.get(key) is not None
    )


def _all_parameters(solution: dict) -> str:
    """The parameters of every component (``1: R …; 2: R …`` when there are two populations)."""
    components = solution.get("components") or [{}]
    if len(components) == 1:
        return _parameters(components[0])
    return "; ".join(f"{number}: {_parameters(component)}" for number, component in enumerate(components, start=1))


def _probability(solution: dict) -> str:
    if solution.get("alone"):  # the one solution of a composition that was given: 1.00 would be no finding
        return fit_names.ONLY_ONE
    value = probability(solution)
    return "—" if math.isnan(value) else f"{value:.2f}"


def fit_rows(fit: dict) -> list[str]:
    """The solutions as a Markdown table (units in the header); the values at a bound of their range named in the
    note, the solution the run reports (the most probable) marked "(chosen)"."""
    solutions = fit.get("solutions") or []
    if not solutions:
        return ["No fit solution."]
    index, _solution = chosen(fit)
    kind = fit_names.BIC_WEIGHT if fit_names.weight_key(_solution or {}) == "bic_weight" else "probability"
    lines = [f"| rank | model | {kind} | parameters | χ²ᵣ | note |", "|---|---|---|---|---|---|"]
    for number, row in enumerate(solutions):
        more = len(row.get("more_warnings") or ())  # of background, beam tail and resolution: under the table
        notes = ([bounds_text(row)] + list(row.get("warnings") or ())
                 + ([] if row.get("converged", True) else ["not converged"])
                 + ([f"{more} more on background, beam tail and resolution (below)"] if more else []))
        rank = f"{row['rank']} (chosen)" if number == index else row["rank"]
        lines.append(f"| {rank} | {row['model']} | {_probability(row)} | {_all_parameters(row)} | "
                     f"{_fixed(row['chi2'], 'chi2')} | {'; '.join(note for note in notes if note)} |")
    return lines


def gisaxs_sections(report: dict, language: str = ENGLISH) -> list[str]:
    gisaxs = report.get("gisaxs") or {}
    cuts = gisaxs.get("cuts") or {}
    lines = [heading("Cuts", language), ""]
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
            + (" — **not applied**: too far from the calibrated centre (see Needs attention)"
               if symmetry.get("not_applied") else "")
        )
        lines.append(f"  - {ASYMMETRY_DEFINITION}")
    halves = gisaxs.get("halves")
    if halves:
        lines.append(f"- halves: **{halves['side']}** — {halves['reason']}")
    spacing = gisaxs.get("spacing")
    lines += ["", heading("Length scale 2π/q*", language), ""]
    if spacing is None:
        lines.append("No side maximum or shoulder away from qy = 0: the cut gives no length scale 2π/q* in the q range.")
    else:
        where = ("at the peak of the cut" if spacing.get("kind") == "maximum"
                 else "at a shoulder of the cut (a change of slope, not a resolved peak)")
        lines.append(f"- 2π/q* = {_number(spacing['distance_nm'], 3)} nm {where}, |qy| = {_number(spacing['q'])} Å⁻¹: "
                     f"{LENGTH_SCALE}.")
    fit = gisaxs.get("fit")
    lines += ["", heading("Fit of I(qy)", language), ""]
    if not fit:
        lines.append("Not fitted here: Send to Fitting in GIMaP fits the prepared curve with a chosen model.")
    else:
        _, selected = chosen(fit)
        by_bic = fit_names.weight_key(selected or {}) == "bic_weight"
        comparison = ("Evidence sampling failed. BIC weights compare the solutions by exp(−BIC/2), "
                      "normalised over the solutions compared." if by_bic else
                      f"Posterior probabilities compare the solutions by their evidences. {fit_names.RELATIVE}.")
        diagnostics = fit_names.diagnostic_lines((selected or {}).get("diagnostics") or {}, (selected or {}).get("ess"))
        medians = (selected or {}).get("posterior_summary") or {}
        lines += [
            f"Curve: {fit.get('curve')} ({fit.get('points')} points, |qy| {_number((fit.get('q_range_inv_angstrom') or [None])[0], 3)}–"
            f"{_number((fit.get('q_range_inv_angstrom') or [None, None])[1], 3)} Å⁻¹).",
            *[f"- {item['text']}" for item in fit.get("curve_notes") or () if item.get("text")],
            *_points_lines(fit),
            *_method_lines(fit),
            "",
            *fit_rows(fit),
            *_more_lines(fit),
            *(["", "| parameter | least-squares value | posterior median | posterior 68 % |",
                "|---|---|---|---|", *[f"| {name} | {_number(selected.get('least_squares_values', {}).get(name))} | "
                f"{_number(row['median'])} | {_number(row['low68'])}–{_number(row['high68'])} |"
                for name, row in medians.items()]] if medians else []),
            *[f"- {template.format(**values)}" for template, values in diagnostics],
            "",
            "Predict parameters: GIMaP's network proposes compositions — one or two populations of spheres and "
            "upright cylinders with size dispersity, \"(S)\" with a paracrystal distance D — and their values; each "
            f"solution is refined by least squares. {comparison} The run reports the highest-weight solution (chosen), "
            f"with its refined sizes and their 68 % intervals in the Decisions. Another solution with weight "
            f"at least {ASK_PROBABILITY:g} is a point to check. "
            f"{fit_names.CHI2_DEFINITION}; it is the number Fitting shows for the same solution. A "
            "non-linear value at a bound of its range (the note column) is not determined by the curve. The "
            "particle amplitudes (Scale), the Background and the beam tail's amplitude Tail A cannot go below 0; one "
            f"at 0 is named in the Decisions (fit caveat). A reported solution with χ²ᵣ above {POOR_CHI2:g} is a "
            "poor fit (Needs attention). Refine the model you expect in Fitting.",
            "",
            "Names and units: R, h, D in nm; σR/R, σh/h, σnn/D relative; Tail w and σ_q in nm⁻¹. With several "
            "populations a value carries its population's number, R (1), D (1), R (2): the populations as the "
            "parameters column numbers them. Keys of the CSV and JSON records — "
            f"{fit_names.legend(max(len(row.get('components') or ()) for row in fit.get('solutions') or [{}]))}.",
        ]
    return lines + [""]


def _points_lines(fit: dict) -> list[str]:
    """How many of the curve's points Predict parameters fitted and why the others were left out."""
    text = fit_names.points_text(fit.get("fitted_points"))
    return [f"- fitted: {text} (χ²ᵣ is over the fitted points)"] if text else []


def _more_lines(fit: dict) -> list[str]:
    """The notes of background, beam tail and resolution of each solution, in one paragraph under the table."""
    rows = [(row.get("rank"), row.get("more_warnings") or ()) for row in fit.get("solutions") or ()]
    said = [f"rank {rank}: {'; '.join(notes)}" for rank, notes in rows if notes]
    return ["", "Background, beam tail and resolution — " + " · ".join(said) + "."] if said else []


def _method_lines(fit: dict) -> list[str]:
    """Which network and seed made the solutions (``fit["method"]``; nothing for a record without it)."""
    method = fit.get("method") or {}
    network = method.get("network") or {}
    if not network.get("name"):
        return []
    digest = f" (weights SHA-256 {network['sha256']})" if network.get("sha256") else ""
    seed = "" if method.get("seed") is None else f", seed {method['seed']}"
    return ["", f"Network: {network['name']}{digest}{seed}."]


def gisaxs_batch_row(report: dict) -> tuple[str, str, str]:
    """(halves, 2π/q*, the fit the run reports) of one frame for the batch table."""
    gisaxs = report.get("gisaxs") or {}
    halves = (gisaxs.get("halves") or {}).get("side") or "—"
    spacing = gisaxs.get("spacing")
    spacing_text = "—" if not spacing else _fixed(spacing["distance_nm"], "distance_nm") + (
        "" if spacing.get("kind") == "maximum" else " (shoulder)")
    _index, best = chosen(gisaxs.get("fit"))
    if best is None:
        return halves, spacing_text, "—"
    said = chance(best, "p ")
    text = (f"{best['model']} {sizes(best, ' ', FIT_DECIMALS)}, χ²ᵣ {_fixed(best['chi2'], 'chi2')}"
            + (f", {said}" if said else ""))
    return halves, spacing_text, text + (f" ⚠ {bounds_text(best)}" if bounds_text(best) else "")


def gisaxs_batch_table(reports: list[dict], state_of, labels=None) -> list[str]:
    """The GISAXS frames of a batch; ``labels``: what each frame is called, by ``id(report)`` (default: its name)."""
    labels = labels or {}
    lines = [
        "| frame | status | halves | 2π/q* (nm) | chosen fit | open questions |",
        "|---|---|---|---|---|---|",
    ]
    for report in reports:
        halves, spacing, fit = gisaxs_batch_row(report)
        attention = report.get("needs_attention") or []
        lines.append(
            f"| {labels.get(id(report)) or Path(report.get('frame') or '').name} | {state_of(report)} | {halves} | "
            f"{spacing} | {fit} | "
            f"{', '.join(item['item'] for item in attention) or '—'} |"
        )
    return lines


__all__ = [
    "cut_spans", "fit_rows", "gisaxs_batch_row", "gisaxs_batch_table", "gisaxs_sections", "image_shape", "pixel_span",
]
