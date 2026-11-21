"""Which solution of Predict parameters the automatic analysis reports, when it asks the person to choose, and how a
value at a bound of its range is said.

Predict parameters (Fitting's ``create_curve_fit``) gives the distinct solutions of a cut most probable first: GIMaP's
network proposes compositions and values, each solution is refined by least squares and weighted by its posterior
probability. The run reports the most probable solution (the first; the fit record keeps its index, ``chosen``).
When another solution is probable too — its probability at least ``ASK_PROBABILITY`` — the curve alone does not
decide the model and the run asks the person to choose (``rivals``). A non-linear value that ended on a bound of its
range (``at_bounds`` of a solution, e.g. D at its upper bound) is not determined by the curve: a point to check.

The linear terms — the particle amplitudes, the Background and the beam tail's amplitude Tail A — cannot go below 0 and
are not in ``at_bounds``: one that ends at 0 is held there by that limit, which the report says for the reported
solution (``zero_terms``). A converged reported solution with χ²ᵣ above ``POOR_CHI2`` misses the curve by far more
than its σ (``poor_fit``).
"""

from __future__ import annotations

import math
from typing import Optional

from src.gimap.shared import fit_names

ASK_PROBABILITY = 0.2
"""A solution after the first with at least this posterior probability makes the run ask the person to choose the
model: the curve does not tell the two apart well enough for the run to decide."""
BOUND_NAMES = {key: fit_names.NAMES[key] for key in ("R", "sigma_R", "h", "sigma_h", "D", "sigma_D", "sigma_nn",
                                                      "sigma_Res", "nu_Res")}
"""How a value at a bound is named: the names of Fitting's Model table (``shared/fit_names``; Tail w, Tail ν: the
beam tail), the same in the table, the flags and the decisions."""
AT_UPPER = "{name} at its upper bound"
AT_LOWER = "{name} at its lower bound"
"""How a value at a bound is said (one phrase per value; the interface translates them by these forms)."""
LENGTH_SCALE = "a length scale, not necessarily the particle spacing; check it with a model fit"
"""What 2π/q* of a peak or shoulder of the cut is: a length the curve shows, which the form factor or the beam tail
can make as well as the interparticle correlation (BornAgain simulations without interference show a shoulder too)."""
POOR_CHI2 = 10.0
"""A converged reported solution whose χ²ᵣ is above this is a poor fit: it misses the curve by far more than the σ
of its points. The limit is Fitting's (``fitting/application/series_fit.POOR_CHI2``, for the same χ²ᵣ); features
import nothing of each other, so a test keeps the two equal."""
LINEAR_TERMS = (("background", "Background"), ("resolution_amplitude", "Tail A"))
"""The linear terms of a solution's ``globals`` that can end at their non-negativity limit, as the report names them."""
INTERVAL_KEYS = ("R", "h", "D")
"""The values whose posterior intervals the report gives (per component: ``R_1``, ``h_2`` … of ``intervals``)."""


def probability(solution: Optional[dict]) -> float:
    """The posterior probability of a solution (NaN when the fit did not give one)."""
    try:
        value = float((solution or {}).get("probability"))
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def rivals(solutions: list[dict]) -> list[dict]:
    """The solutions after the first whose probability is at least ``ASK_PROBABILITY`` (empty: the run decides)."""
    return [item for item in solutions[1:] if probability(item) >= ASK_PROBABILITY]


def poor_fit(solution: Optional[dict]) -> bool:
    """A converged solution with χ²ᵣ above ``POOR_CHI2`` (a fit that did not converge is said as such)."""
    if not solution or not solution.get("converged", True):
        return False
    chi2 = _chi2(solution)
    return math.isfinite(chi2) and chi2 > POOR_CHI2


def zero_terms(solution: Optional[dict]) -> list[str]:
    """The names of the linear terms of ``globals`` that ended at 0 (``["Background"]``)."""
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
    """``["D at its upper bound", …]``: the values of a solution that ended on a bound of their range (empty when the
    fit did not say). With more than one component the component's number follows the name (``R (2)``)."""
    phrases = []
    many = len(solution.get("components") or ()) > 1
    for item in solution.get("at_bounds") or ():
        name = fit_names.display_name(str(item.get("parameter")), False)
        if many and item.get("component") is not None:
            name = f"{name} ({int(item['component']) + 1})"
        phrases.append((AT_UPPER if item.get("bound") == "upper" else AT_LOWER).format(name=name))
    return phrases


def bounds_text(solution: dict) -> str:
    """The values at a bound, in one line ("" when none)."""
    return ", ".join(bound_phrases(solution))


def sizes(solution: Optional[dict], equals: str = " = ", decimals: Optional[dict] = None) -> str:
    """R of every population and D of those with a paracrystal, in the order of the radii (``R = 3.06 / 6.02 nm,
    D = 49.4 / – nm``; one population: ``R = 3.06 nm, D = 49.4 nm``)."""
    return fit_names.sizes_text((solution or {}).get("components") or (), equals, decimals)


def chance(solution: Optional[dict], name: str = "probability ") -> str:
    """``probability 0.42``; ``only this composition was tried`` for the one solution of a composition that was
    given (its probability of 1 is no finding); "" when the fit gave none."""
    if (solution or {}).get("alone"):
        return fit_names.ONLY_ONE
    value = probability(solution)
    return "" if math.isnan(value) else f"{name}{value:.2f}"


def interval(solution: Optional[dict], key: str, component: int = 0, level: int = 68) -> Optional[tuple[float, float]]:
    """``(low, high)`` of the ``level`` % posterior interval of a component's value (``key``: R, h, D; ``component``
    from 0), or ``None`` when the fit gave none."""
    found = ((solution or {}).get("intervals") or {}).get(f"{key}_{component + 1}") or {}
    try:
        low, high = float(found[f"low{level}"]), float(found[f"high{level}"])
    except (KeyError, TypeError, ValueError):
        return None
    return (low, high) if math.isfinite(low) and math.isfinite(high) else None


def _chi2(solution: dict) -> float:
    try:
        return float(solution.get("chi2"))
    except (TypeError, ValueError):
        return math.nan


def chosen(fit: Optional[dict]) -> tuple[int, Optional[dict]]:
    """``(index, solution)`` the run reports (``fit["chosen"]``: the most probable, the first; a record without it:
    the first)."""
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
    "ASK_PROBABILITY", "AT_LOWER", "AT_UPPER", "BOUND_NAMES", "INTERVAL_KEYS", "LENGTH_SCALE", "LINEAR_TERMS", "POOR_CHI2",
    "bound_phrases", "bounds_text", "chance", "chosen", "interval", "poor_fit", "probability", "rivals", "sizes",
    "zero_terms",
]
