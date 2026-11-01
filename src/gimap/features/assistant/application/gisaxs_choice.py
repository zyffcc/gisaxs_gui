"""Which solution of the quick GISAXS fit the automatic analysis reports, and how a value at a search bound is said.

The fit ranks its solutions by a robust misfit (the mean Huber loss). A non-linear value that ended on a bound of
its search range (``at_bounds`` of a solution, e.g. D at 500 nm) is not determined by the curve. Among the solutions
the curve does not tell apart — χ²/N at most ``CLOSE_CHI2`` times that of the first-ranked solution — the run reports
the best-ranked one that converged with no non-linear value at a bound; when there is none, the first-ranked solution
stays and the report says that it sits at a bound. The fit record keeps the index of the reported solution
(``chosen``).

The linear terms — the particle amplitudes, the background BG and the beam-tail amplitude A — are solved by
non-negative least squares at every step and are not in ``at_bounds``: one that ends at 0 is held there by that
limit, which the report says for the reported solution (``zero_terms``) without counting it in the choice. A
converged reported solution with χ²/N above ``POOR_CHI2`` misses the curve by far more than its σ (``poor_fit``).
"""

from __future__ import annotations

import math
from typing import Optional

CLOSE_CHI2 = 1.1
"""Solutions whose χ²/N is within 10 % of the first solution's are not distinguished by the curve.

The fit ranks its solutions by a robust misfit, the mean Huber loss (δ = 2) of the σ-normalised residuals r, so
that a few outlying points do not decide; χ²/N is the mean of r² over the N points, with no degrees-of-freedom
correction (unlike Fitting's χ²ᵣ). All solutions are fitted to the same points, so their ratio is that of χ²."""
BOUND_NAMES = {"R": "R", "sigma_R": "σR/R", "h": "h", "sigma_h": "σh/h", "D": "D", "sigma_D": "σD/D",
               "sigma_Res": "σ_Res", "nu_Res": "ν_Res"}
"""How a value at a bound is named (as the report's table names the parameters; σ_Res, ν_Res: the beam tail)."""
AT_UPPER = "{name} at its upper bound"
AT_LOWER = "{name} at its lower bound"
"""How a value at a bound is said (one phrase per value; the interface translates them by these forms)."""
LENGTH_SCALE = "a length scale, not necessarily the particle spacing; check it with a model fit"
"""What 2π/q* of a peak or shoulder of the cut is: a length the curve shows, which the form factor or the beam tail
can make as well as the interparticle correlation (BornAgain simulations without interference show a shoulder too)."""
POOR_CHI2 = 10.0
"""A converged reported solution whose χ²/N is above this is a poor fit: it misses the curve by far more than the σ
of its points. The limit is Fitting's (``fitting/application/series_fit.POOR_CHI2``, for its χ²ᵣ, which differs from
χ²/N only by the degrees-of-freedom correction, a few parameters in hundreds of points); features import nothing of
each other, so a test keeps the two equal."""
LINEAR_TERMS = (("background", "background BG"), ("resolution_amplitude", "beam-tail amplitude A"))
"""The linear terms of a solution's ``globals`` that can end at their non-negativity limit, as the report names them
(the particle amplitudes too, which the fit warns of itself: "negligible fitted amplitude")."""


def poor_fit(solution: Optional[dict]) -> bool:
    """A converged solution with χ²/N above ``POOR_CHI2`` (a fit that did not converge is said as such)."""
    if not solution or not solution.get("converged", True):
        return False
    chi2 = _chi2(solution)
    return math.isfinite(chi2) and chi2 > POOR_CHI2


def zero_terms(solution: Optional[dict]) -> list[str]:
    """The names of the linear terms of ``globals`` that ended at 0 (``["background BG"]``)."""
    values = (solution or {}).get("globals") or {}
    found = []
    for key, name in LINEAR_TERMS:
        try:
            if values.get(key) is not None and float(values[key]) <= 0.0:
                found.append(name)
        except (TypeError, ValueError):
            continue
    return found


def bound_phrases(solution: dict) -> list[str]:
    """``["D at its upper bound", …]``: the values of a solution that ended on a search bound (empty when the fit
    did not say). With more than one component the component's number follows the name (``R 2``)."""
    phrases = []
    many = len(solution.get("components") or ()) > 1
    for item in solution.get("at_bounds") or ():
        name = BOUND_NAMES.get(str(item.get("parameter")), str(item.get("parameter")))
        if many and item.get("component") is not None:
            name = f"{name} {int(item['component']) + 1}"
        phrases.append((AT_UPPER if item.get("bound") == "upper" else AT_LOWER).format(name=name))
    return phrases


def bounds_text(solution: dict) -> str:
    """The values at a bound, in one line ("" when none)."""
    return ", ".join(bound_phrases(solution))


def flagged(solution: dict) -> bool:
    """A non-linear value at a search bound, or a fit that did not converge: what the run does not report when it can
    avoid it (a linear term at 0 is not counted: ``zero_terms``)."""
    return bool(solution.get("at_bounds")) or not solution.get("converged", True)


def _chi2(solution: dict) -> float:
    try:
        return float(solution.get("chi2"))
    except (TypeError, ValueError):
        return math.nan


def close_solutions(solutions: list[dict]) -> list[dict]:
    """The solutions whose χ²/N is at most ``CLOSE_CHI2`` times the first-ranked solution's (the first included)."""
    if not solutions:
        return []
    limit = CLOSE_CHI2 * _chi2(solutions[0])
    return [solutions[0]] + [item for item in solutions[1:] if _chi2(item) <= limit]


def chosen_index(solutions: list[dict]) -> int:
    """The solution the run reports: the best-ranked of ``close_solutions`` that is not ``flagged``, else the first."""
    if not solutions:
        return 0
    limit = CLOSE_CHI2 * _chi2(solutions[0])
    return next((index for index, item in enumerate(solutions)
                 if (index == 0 or _chi2(item) <= limit) and not flagged(item)), 0)


def chosen(fit: Optional[dict]) -> tuple[int, Optional[dict]]:
    """``(index, solution)`` the run reports (``fit["chosen"]``; the first solution in a record without it)."""
    solutions = (fit or {}).get("solutions") or []
    if not solutions:
        return 0, None
    try:
        index = int((fit or {}).get("chosen") or 0)
    except (TypeError, ValueError):
        index = 0
    index = index if 0 <= index < len(solutions) else 0
    return index, solutions[index]


__all__ = [
    "AT_LOWER", "AT_UPPER", "BOUND_NAMES", "CLOSE_CHI2", "LENGTH_SCALE", "LINEAR_TERMS", "POOR_CHI2", "bound_phrases",
    "bounds_text", "chosen", "chosen_index", "close_solutions", "flagged", "poor_fit", "zero_terms",
]
