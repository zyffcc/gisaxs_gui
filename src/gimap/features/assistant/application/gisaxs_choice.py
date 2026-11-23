"""Which solution of Predict parameters the automatic analysis reports, when it asks the person to choose, and how a
value at a bound of its range is said.

Predict parameters (Fitting's ``create_curve_fit``) gives the distinct solutions of a cut most probable first: GIMaP's
network proposes compositions and values, each solution is refined by least squares and weighted by its posterior
probability. The run reports the most probable solution (the first; the fit record keeps its index, ``chosen``).
When another composition has weight at least ``ASK_PROBABILITY``, the run asks the person to choose the model
(``rivals``). For probable solutions of the same composition, it compares R, h and D with their 68 % intervals.
A non-linear value that ended on a bound of its
range (``at_bounds`` of a solution, e.g. D at its upper bound) is not determined by the curve: a point to check.

The linear terms — the particle amplitudes, the Background and the beam tail's amplitude Tail A — cannot go below 0 and
are not in ``at_bounds``: one that ends at 0 is held there by that limit, which the report says for the reported
solution (``zero_terms``). A converged reported solution with χ²ᵣ above ``POOR_CHI2`` misses the curve by far more
than its σ (``poor_fit``). A sphere or random cylinder fitted at q_z = 0 whose q_z·R at the cut is above
``QZ_R_LIMIT`` has a biased radius (``interface_caveat``, from the flag ``qzR``).
"""

from __future__ import annotations

import math
from typing import Optional

from src.gimap.shared import fit_names
from src.gimap.shared.numbers import format_number

ASK_PROBABILITY = 0.2
"""Minimum weight for a competing composition or an alternative set of values within the reported composition."""
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
QZ_R_LIMIT = 1.7
"""q_z·R of the cut above which the form factor at q_z = 0 biases a sphere's radius: Fitting's ``refine.QZR_LIMIT``,
the limit of its flag ``qzR`` (features import nothing of each other, so a test keeps the two equal)."""
QZ_R = (
    "The reported solution ({model}) was fitted with the form factor at q_z = 0 (the effective interface), but at the "
    "cut's q_z = {qz} nm⁻¹ q_z·R is {values} for {names}, above {limit}: there that treatment biases the radius, "
    "usually too small."
)
QZ_R_HINT = (
    "Send to Fitting, enter αi, αf and λ of the cut under Model ▸ Spheres in grazing incidence and run Predict "
    "parameters again: spheres are then fitted with the exit-reflected path. Random cylinders stay at q_z = 0."
)
"""A reported solution with the flag ``qzR`` (``interface_caveat``)."""
LINEAR_TERMS = (("background", "Background"), ("resolution_amplitude", "Tail A"))
"""The linear terms of a solution's ``globals`` that can end at their non-negativity limit, as the report names them."""
INTERVAL_KEYS = ("R", "h", "D")
"""The values whose posterior intervals the report gives (per component: ``R_1``, ``h_2`` … of ``intervals``)."""


def _finite(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def probability(solution: Optional[dict]) -> float:
    """The solution's own weight. Older BIC records stored it under 'probability'."""
    row = solution or {}
    return _finite(row.get(fit_names.weight_key(row), row.get("probability")))


def model_probability(solution: Optional[dict]) -> float:
    """The recorded weight of this composition, or the solution's own weight in older records."""
    row = solution or {}
    diagnostics = row.get("diagnostics") or {}
    weight = diagnostics.get("composition_" + fit_names.weight_key(row))
    return probability(row) if weight is None else _finite(weight)


def weight_text(weight: float) -> str:
    """Show the recorded weight to two decimals, or a dash if the record has none."""
    return f"{weight:.2f}" if math.isfinite(weight) else "—"


def composition_modes(solution: dict, solutions=()) -> int:
    """Include modes omitted from the solution list when the fit recorded their count."""
    recorded = (solution.get("diagnostics") or {}).get("composition_modes") or 1
    model = solution.get("model")
    listed = sum(row.get("model") == model for row in solutions) if model else 0
    return max(recorded, listed)


def rivals(solutions: list[dict], reported: Optional[dict] = None) -> list[dict]:
    """One representative per competing model, selected by its solution weight."""
    if not solutions:
        return []
    model = (reported or solutions[0]).get("model")
    if not model:
        return []
    representatives = {}
    for row in solutions:
        other = row.get("model")
        if not other or other == model or not model_probability(row) >= ASK_PROBABILITY:
            continue
        previous = representatives.get(other)
        if previous is None or math.isnan(probability(previous)) or probability(row) > probability(previous):
            representatives[other] = row
    return list(representatives.values())


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


def component_estimate(solution, key, index):
    precise = solution.get("least_squares_values") or {}
    name = f"{key}_{index + 1}"
    value = precise.get(name)
    components = solution.get("components") or ()
    if value is None and index < len(components):
        value = components[index].get(key)
    return value


def displayed_estimate(solution, key, index=None):
    """The value and reported 68 % interval in the units of a solution's display field."""
    intervals = solution.get("intervals") or {}
    precise = solution.get("least_squares_values") or {}
    if index is None:
        name = {"sigma_Res": "tail_width", "nu_Res": "tail_exponent",
                "resolution_amplitude": "res_amplitude"}.get(key, key)
        value = (solution.get("globals") or {}).get(key)
        if precise.get(name) is not None:
            value = precise[name]
        return value, _bounds(intervals.get(name) or intervals.get(key), 68)

    value = component_estimate(solution, key, index)
    bounds = interval(solution, key, index)
    if key == "sigma_D" and bounds is None:
        # Predict fits σnn/D. This display field is σD/D = (σnn/D) / √(2π).
        source = f"sigma_nn_{index + 1}"
        scale = math.sqrt(2 * math.pi)
        if precise.get(source) is not None:
            value = precise[source] / scale
        original = _bounds(intervals.get(source), 68)
        if original is not None:
            bounds = tuple(endpoint / scale for endpoint in original)
    return value, bounds


def component_number(solution, key, index):
    value, bounds = displayed_estimate(solution, key, index)
    return format_number(value) if bounds is None else fit_names.interval_values(value, *bounds)[0]


def population_values(solution, key):
    """Parameter values in radius order, using each population's original interval."""
    components = (solution or {}).get("components") or []
    ordered = sorted((i for i, c in enumerate(components) if c.get("R") is not None), key=lambda i: components[i]["R"])
    if not any(components[i].get(key) is not None for i in ordered):
        return "—"
    return " / ".join(component_number(solution, key, i) if components[i].get(key) is not None else "–" for i in ordered)


def sizes(solution: Optional[dict], equals: str = " = ") -> str:
    radii = population_values(solution, "R")
    spacing = population_values(solution, "D")
    return f"R{equals}{radii} nm" + ("" if spacing == "—" else f", D{equals}{spacing} nm")


def chance(solution: Optional[dict], solutions=(), *, render=str.format) -> str:
    """The composition's weight, with the solution's own share when the composition has several modes."""
    row = solution or {}
    if row.get("alone"):
        return render(fit_names.ONLY_ONE)
    weight = model_probability(row)
    if math.isnan(weight):
        return ""
    by_bic = fit_names.weight_key(row) == "bic_weight"
    if composition_modes(row, solutions) > 1:
        template = ("model " + fit_names.BIC_WEIGHT + " {weight} (this solution {share})" if by_bic else
                    "model probability {weight} (this solution {share})")
        return render(template, weight=weight_text(weight), share=weight_text(probability(row)))
    template = fit_names.BIC_WEIGHT + " {weight}" if by_bic else "probability {weight}"
    return render(template, weight=weight_text(weight))


def interval(solution: Optional[dict], key: str, component: int = 0, level: int = 68) -> Optional[tuple[float, float]]:
    """``(low, high)`` of the ``level`` % posterior interval of a component's value (``key``: R, h, D; ``component``
    from 0), or ``None`` when the fit gave none."""
    return _bounds(((solution or {}).get("intervals") or {}).get(f"{key}_{component + 1}"), level)


def _estimate_text(value, bounds):
    if bounds is None:
        return f"{format_number(value)} nm (68 % interval unavailable)"
    return fit_names.interval_text(value, *bounds, "nm")


def _value_differences(reported, other):
    count = max(len(reported.get("components") or ()), len(other.get("components") or ()), 1)
    for index in range(count):
        for key in INTERVAL_KEYS:
            first = component_estimate(reported, key, index)
            second = component_estimate(other, key, index)
            if key != "R" and first is None and second is None:
                continue  # h and D exist only for components that fit them
            first, second = _finite(first), _finite(second)
            a, b = interval(reported, key, index), interval(other, key, index)
            if a and b and a[0] <= second <= a[1] and b[0] <= first <= b[1]:
                continue
            name = fit_names.display_name(f"{key}_{index + 1}", count > 1)
            yield name, _estimate_text(first, a), _estimate_text(second, b)


def compare_same_model(solutions: list[dict], reported: Optional[dict] = None) -> tuple[int, list[tuple[str, str, str]]]:
    """Count solutions consistent with the reported one (including itself), and list differences for the others.

    A probable solution agrees only if every refined R, h and D is in the corresponding 68 % interval of the
    other solution, in both directions. The comparison uses full precision before formatting the differences.
    """
    if not solutions:
        return 0, []
    reported = reported or solutions[0]
    overlapping = 1
    differences = []
    for other in solutions:
        if other is reported or not reported.get("model") or other.get("model") != reported["model"]:
            continue
        if probability(other) >= ASK_PROBABILITY:
            distinct = list(_value_differences(reported, other))
            if distinct:
                differences.extend(distinct)
            else:
                overlapping += 1
    return overlapping, differences


def _bounds(found, level):
    if not found:
        return None
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


def interface_caveat(solution: Optional[dict], cut: Optional[dict]) -> str:
    """``QZ_R`` for a solution whose populations carry the flag ``qzR`` (fitted at q_z = 0, q_z·R of the cut above
    ``QZ_R_LIMIT``), with q_z·R of each; "" without such a flag or without the cut's q_z (``cut``: the fit record's
    ``cut_qz``). The flag numbers the populations in the order of their radii, as the solution's values."""
    qz = (cut or {}).get("q_z_inv_nm")
    if not solution or not qz:
        return ""
    ordered = sorted((c for c in solution.get("components") or () if c.get("R") is not None), key=lambda c: float(c["R"]))
    many = len(solution.get("components") or ()) > 1
    names, values = [], []
    for flag in solution.get("flags") or ():
        kind, flagged = fit_names.flag_names(flag)
        slot = fit_names.split_name(flagged[0])[1] if kind == "qzR" and flagged else None
        if slot is not None and 1 <= slot <= len(ordered):
            names.append(fit_names.display_name(flagged[0], many))
            values.append(f"{float(qz) * float(ordered[slot - 1]['R']):.2f}")
    if not names:
        return ""
    return QZ_R.format(model=solution.get("model"), qz=f"{float(qz):.3g}", values=", ".join(values),
                       names=", ".join(names), limit=f"{QZ_R_LIMIT:g}")


__all__ = [
    "ASK_PROBABILITY", "AT_LOWER", "AT_UPPER", "BOUND_NAMES", "INTERVAL_KEYS", "LENGTH_SCALE", "LINEAR_TERMS", "POOR_CHI2",
    "QZ_R", "QZ_R_HINT", "QZ_R_LIMIT", "bound_phrases", "bounds_text", "chance", "chosen", "compare_same_model",
    "composition_modes", "interface_caveat", "interval", "model_probability", "poor_fit", "probability", "rivals",
    "sizes", "weight_text", "zero_terms",
]
