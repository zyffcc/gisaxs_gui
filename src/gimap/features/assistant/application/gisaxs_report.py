"""The GISAXS part of the report as Markdown: the cut, the halves, the spacing and the fit."""

from __future__ import annotations

from pathlib import Path


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _parameters(component: dict) -> str:
    names = (("R", "R"), ("sigma_R", "σR/R"), ("h", "h"), ("sigma_h", "σh/h"), ("D", "D"), ("sigma_D", "σD/D"))
    units = {"R": " nm", "h": " nm", "D": " nm"}
    return ", ".join(
        f"{label} {_number(component[key], 3)}{units.get(key, '')}" for key, label in names if component.get(key) is not None
    )


def fit_rows(fit: dict) -> list[str]:
    """The solutions as a Markdown table (units in the header)."""
    solutions = fit.get("solutions") or []
    if not solutions:
        return ["No fit solution."]
    lines = ["| rank | model | parameters | χ² | note |", "|---|---|---|---|---|"]
    for row in solutions:
        component = (row.get("components") or [{}])[0]
        note = "; ".join(row.get("warnings") or ()) or ("" if row.get("converged") else "not converged")
        lines.append(f"| {row['rank']} | {row['model']} | {_parameters(component)} | {_number(row['chi2'], 3)} | {note} |")
    return lines


def gisaxs_sections(report: dict) -> list[str]:
    gisaxs = report.get("gisaxs") or {}
    cuts = gisaxs.get("cuts") or {}
    lines = ["## 切线 / Cuts", ""]
    rows = cuts.get("horizontal_rows") or [None, None]
    source = {"yoneda": "at the Yoneda band", "horizon": "just above the horizon (no Yoneda band found)"}.get(
        cuts.get("horizontal_source"), "set by hand"
    )
    alpha = cuts.get("yoneda_alpha_f_deg")
    lines.append(
        f"- horizontal cut I(qy) {source}: rows {_number(rows[0])}–{_number(rows[1])}"
        + (f", αf = {_number(alpha, 3)}°" if alpha is not None else "")
    )
    columns = cuts.get("vertical_columns") or [None, None]
    lines.append(f"- vertical cut I(qz): columns {_number(columns[0])}–{_number(columns[1])}")
    symmetry = gisaxs.get("symmetry")
    if symmetry:
        lines.append(
            f"- symmetry axis x = {_number(symmetry.get('x_px'))} px (calibrated {_number(symmetry.get('initial_x_px'))} px; "
            f"asymmetry {_number(symmetry.get('loss_before'), 3)} → {_number(symmetry.get('loss_after'), 3)})"
        )
    halves = gisaxs.get("halves")
    if halves:
        lines.append(f"- halves: **{halves['side']}** — {halves['reason']}")
    spacing = gisaxs.get("spacing")
    lines += ["", "## 间距 / In-plane spacing", ""]
    if spacing is None:
        lines.append("No side maximum or shoulder away from qy = 0: no dominant in-plane distance in the q range.")
    else:
        kind = "side maximum" if spacing.get("kind") == "maximum" else "shoulder (a hint, not a resolved peak)"
        lines.append(f"- {kind} at |qy| = {_number(spacing['q'])} Å⁻¹ → D ≈ 2π/q = {_number(spacing['distance_nm'], 3)} nm")
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
            "Numerical fits of single particle families with size dispersity and a paracrystal distance D. χ² close to "
            "each other means the curve alone does not decide the model: choose it from what is known of the sample.",
        ]
    return lines + [""]


def gisaxs_batch_row(report: dict) -> tuple[str, str, str]:
    """(halves, spacing, best fit) of one frame for the batch table."""
    gisaxs = report.get("gisaxs") or {}
    halves = (gisaxs.get("halves") or {}).get("side") or "—"
    spacing = gisaxs.get("spacing")
    spacing_text = "—" if not spacing else f"{spacing['distance_nm']:.3g}" + ("" if spacing.get("kind") == "maximum" else " (shoulder)")
    solutions = (gisaxs.get("fit") or {}).get("solutions") or []
    if not solutions:
        return halves, spacing_text, "—"
    best = solutions[0]
    component = (best.get("components") or [{}])[0]
    return halves, spacing_text, f"{best['model']} R {_number(component.get('R'), 3)}, D {_number(component.get('D'), 3)} nm, χ² {_number(best['chi2'], 3)}"


def gisaxs_batch_table(reports: list[dict], state_of) -> list[str]:
    lines = [
        "| frame | status | halves | spacing D (nm) | best fit | open questions |",
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


__all__ = ["fit_rows", "gisaxs_batch_row", "gisaxs_batch_table", "gisaxs_sections"]
