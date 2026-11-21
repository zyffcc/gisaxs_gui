"""Tables of a GISAXS fit for files: the fitted curve and the solutions (units in the headers)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.gimap.shared import fit_names

from .gisaxs_choice import INTERVAL_KEYS, chosen, interval

A_TO_NM = 10.0
"""q in Å⁻¹ × 10 = q in nm⁻¹ (the fit works in nm⁻¹)."""


def fit_on(q_inv_angstrom, best_curve: Optional[dict]) -> np.ndarray:
    """The best fit at |q| of the data (NaN outside the fitted range)."""
    q = np.abs(np.asarray(q_inv_angstrom, dtype=float)) * A_TO_NM
    best_curve = best_curve or {}
    x = np.asarray(best_curve.get("q_inv_nm") or (), dtype=float)
    y = np.asarray(best_curve.get("intensity") or (), dtype=float)
    if x.size < 2 or x.size != y.size:
        return np.full(q.shape, np.nan)
    order = np.argsort(x)
    return np.interp(q, x[order], y[order], left=np.nan, right=np.nan)


def fit_curve_table(fit: dict) -> tuple[str, np.ndarray]:
    """``(header, columns)``: q (Å⁻¹), I, σ of the fitted curve and the fit the run reports (``best_curve``: the
    chosen solution, ``gisaxs_choice``) at the same q."""
    data = fit.get("data") or {}
    q = np.asarray(data.get("q_inv_angstrom") or (), dtype=float)
    columns = np.column_stack([
        q, np.asarray(data.get("intensity") or (), dtype=float), np.asarray(data.get("sigma") or (), dtype=float),
        fit_on(q, fit.get("best_curve")),
    ]) if q.size else np.empty((0, 4))
    return f"q (A^-1), I, sigma, I_fit  [{fit.get('curve')}]", columns


def _intervals(row: dict, number: int) -> list[str]:
    """``R_low68`` … ``D_high68`` of one component (from 0) of a solution, as text ("" where the fit gave none)."""
    cells = []
    for key in INTERVAL_KEYS:
        found = interval(row, key, number)
        cells += ["", ""] if found is None else [str(found[0]), str(found[1])]
    return cells


def fit_solutions_csv(fit: dict, provenance: str = "") -> str:
    """One line per component of each solution: rank, model, χ²ᵣ, probability, parameters (R, h, D in nm; the σ
    columns are relative widths σ/value, without a unit), the 68 % posterior intervals of R, h and D when the fit
    gave them, then ``at_bounds`` (the values that ended on a bound of their range, as ``name:upper`` or
    ``name:lower``, a component's number after the name when there are several) and ``chosen`` (the solution the
    run reports, the most probable). ``#`` lines come first: the lines of ``provenance`` when given (the release,
    the inputs by SHA-256 and the geometry, as in the curve tables), then what the table is (with the curve that was
    fitted) and the units; then the column names."""
    rows = fit.get("solutions") or []
    index, _solution = chosen(fit)
    keys = sorted({key for row in rows for component in row["components"] for key in component if key != "type"})
    ranges = any(row.get("intervals") for row in rows)
    limits = [f"{key}_{end}68" for key in INTERVAL_KEYS for end in ("low", "high")] if ranges else []
    lines = [f"# {line}" for line in provenance.splitlines()] + [
        f"# GISAXS fit solutions [{' '.join(str(fit.get('curve')).split())}]: one line per component of each solution",
        "# units: R, h, D in nm; sigma_R, sigma_h, sigma_D are relative widths (sigma/value, no unit); "
        "chi2 (the reduced chi^2 = sum((I - model)^2 / sigma^2) / (N - number of free values) over the fitted "
        "points, with the curve's sigma), probability (posterior probability of the solution, relative to the "
        "compositions that were tried), log_rmse, amplitude and weight have no unit; "
        + ("R_low68 ... D_high68: the 68 % posterior interval in nm; " if ranges else "")
        + "at_bounds: values that ended on a bound of their range (not determined by the curve); chosen: the solution "
        "reported",
        *([f"# fitted points: {fit_names.points_text(fit.get('fitted_points'))}"] if fit.get("fitted_points") else []),
        "rank,model,chi2,probability,log_rmse,converged,component," + ",".join(keys + limits) + ",at_bounds,chosen",
    ]
    for number, row in enumerate(rows):
        many = len(row["components"] or ()) > 1
        ended = " ".join(
            f"{item.get('parameter')}{'' if not many or item.get('component') is None else int(item['component']) + 1}"
            f":{item.get('bound')}" for item in row.get("at_bounds") or ())
        chance = "" if row.get("probability") is None else str(row["probability"])
        for place, component in enumerate(row["components"] or [{"type": ""}]):
            values = ",".join(["" if component.get(key) is None else str(component[key]) for key in keys]
                              + (_intervals(row, place) if ranges else []))
            lines.append(
                f"{row['rank']},{row['model']},{row['chi2']},{chance},{row['log_rmse']},{row.get('converged', True)},"
                f"{component['type']},{values},{ended},{number == index}"
            )
    return "\n".join(lines) + "\n"


__all__ = ["fit_curve_table", "fit_on", "fit_solutions_csv"]
