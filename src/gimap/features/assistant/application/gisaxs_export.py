"""Tables of a GISAXS fit for files: the fitted curve and the solutions (units in the headers)."""

from __future__ import annotations

from typing import Optional

import numpy as np

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
    """``(header, columns)``: q (Å⁻¹), I, σ of the fitted curve and the best fit at the same q."""
    data = fit.get("data") or {}
    q = np.asarray(data.get("q_inv_angstrom") or (), dtype=float)
    columns = np.column_stack([
        q, np.asarray(data.get("intensity") or (), dtype=float), np.asarray(data.get("sigma") or (), dtype=float),
        fit_on(q, fit.get("best_curve")),
    ]) if q.size else np.empty((0, 4))
    return f"q (A^-1), I, sigma, I_fit  [{fit.get('curve')}]", columns


def fit_solutions_csv(fit: dict) -> str:
    """One line per component of each solution: rank, model, χ², parameters (R, h, D, σ in nm)."""
    rows = fit.get("solutions") or []
    keys = sorted({key for row in rows for component in row["components"] for key in component if key != "type"})
    lines = ["rank,model,chi2,log_rmse,converged,component," + ",".join(keys)]
    for row in rows:
        for component in row["components"] or [{"type": ""}]:
            values = ",".join("" if component.get(key) is None else str(component[key]) for key in keys)
            lines.append(
                f"{row['rank']},{row['model']},{row['chi2']},{row['log_rmse']},{row['converged']},{component['type']},{values}"
            )
    return "\n".join(lines) + "\n"


__all__ = ["fit_curve_table", "fit_on", "fit_solutions_csv"]
