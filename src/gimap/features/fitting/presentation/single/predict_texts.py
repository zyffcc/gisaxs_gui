"""What the single-curve page says of a solution of Predict parameters, in the interface language: its values with
their intervals, its probability, the points its χ²ᵣ is of, its flags and how two solutions of one composition
differ. The names and the flags' sentences are those of ``shared/fit_names`` (the Model table's names), the same
as in the 1D Predict rows and the automatic analysis's report.
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared import fit_names

SHOWN = ("R", "sigma_R", "h", "sigma_h", "D", "sigma_nn")
"""The populations' values listed in Fit details."""
VALUES = "Values in Model with their 68 % posterior intervals (refined value [interval]): {values}."
VALUES_BIC = "Values in Model with 68 % intervals from the least-squares errors (refined value [interval]): {values}."
ONLY_ONE = "Only this composition was tried: no probability."
AMONG_ONE = "Probability {p} among the solutions of the one composition that was tried."
RELATIVE = "Probability {p}, relative to the compositions that were tried."
POINTS = "χ²ᵣ is over {points}."
DIFFERS = "Differs from solution {other} in: {values}."
THERE = "{name} {value} (there {other})"
REFINE_POINTS = ("Refine fitted {refined} points; Predict parameters fitted {fitted} of {given} ({why}): the two χ²ᵣ "
                 "are not of the same points.")
P_TIP = "p: the posterior probability of the solution; probabilities are relative to the compositions that were tried."
SIZES_TIP = "Every population's value, in the order of the radii; –: a population without a paracrystal."
NO_ERROR = {
    "bound": "No error: the value is at a bound of its range.",
    "scale": "No error: the Scale of this population is 0, so its values change nothing.",
    "tail": "No error: Tail A is 0, so this value changes nothing.",
    "curve": "No error: the curve does not constrain this value.",
}
"""Why a free value has no error after Refine (the ± column shows “–”)."""


def count_solutions(count: int) -> str:
    """“1 solution”, “3 solutions”."""
    return tr("1 solution") if count == 1 else trf("{count} solutions", count=count)


def chi2_definition() -> str:
    return tr(fit_names.CHI2_DEFINITION)


def points_text(points) -> str:
    """“596 of 598 points; 2 below the network's q window” (“” when the solution does not say)."""
    if not points:
        return ""
    fitted, given, below, unusable = (tuple(points) + (0, 0, 0))[:4]
    parts = [trf(fit_names.POINTS, fitted=fitted, given=given)]
    if below:
        parts.append(trf(fit_names.BELOW_WINDOW, count=below))
    if unusable:
        parts.append(trf(fit_names.UNUSABLE, count=unusable))
    return "; ".join(parts)


def points_tuple(points) -> tuple:
    """``Predicted.points`` (a dict) as the session keeps it."""
    if not points or points.get("fitted") is None:
        return ()
    return (int(points["fitted"]), int(points.get("given") or points["fitted"]), int(points.get("below_window") or 0),
            int(points.get("unusable") or 0))


def sizes_cells(model) -> tuple[str, str]:
    """The R and D cells of the solutions table: every population, in the order of the radii (“3.06 / 6.02”,
    “49.4 / –”)."""
    ordered = sorted(model.components, key=lambda component: component.value("R"))
    if not ordered:
        return "—", "—"
    radii = " / ".join(f"{component.value('R'):.3g}" for component in ordered)
    if not any(component.structure for component in ordered):
        return radii, "—"
    return radii, " / ".join(f"{component.value('D'):.3g}" if component.structure else "–" for component in ordered)


def _shown(solution, many: bool):
    """(internal name, display name, unit, best, low68, high68) of the populations' values."""
    for name, values in solution.intervals:
        key, _slot = fit_names.split_name(name)
        if key not in SHOWN or values[0] is None or values[2] is None or values[3] is None:
            continue
        yield name, fit_names.display_name(name, many, tr), fit_names.unit_of(name), values[0], values[2], values[3]


def probability_line(solution, solutions) -> str:
    if solution.probability is None:
        return ""
    if solution.tried == 1:
        same = [s for s in solutions if s.tried == 1 and s.probability is not None]
        return tr(ONLY_ONE) if len(same) <= 1 else trf(AMONG_ONE, p=f"{solution.probability:.2f}")
    return trf(RELATIVE, p=f"{solution.probability:.2f}")


def solution_lines(solution, solutions, many: bool) -> list[str]:
    """Fit details of the solution in Model: the refined values with their intervals (never the median), the
    probability, the points of χ²ᵣ; then the flags of the populations and of the solution, one line each, and the
    flags of background, beam tail and resolution in one paragraph."""
    flags = tuple(solution.flags)
    by_errors = "bic" in flags
    shown = list(_shown(solution, many))
    lines = []
    if shown:
        parts = [f"{label} {fit_names.interval_text(best, low, high, unit)}" for _n, label, unit, best, low, high in shown]
        lines.append(trf(VALUES_BIC if by_errors else VALUES, values="; ".join(parts)))
    chance = probability_line(solution, solutions)
    if chance:
        lines.append(chance)
    if solution.points:
        lines.append(trf(POINTS, points=points_text(solution.points)))
    if not by_errors:
        bounded = {name for flag in flags if flag.startswith(("at_bound:", "unconstrained:"))
                   for name in fit_names.flag_names(flag)[1]}
        edged = [label for name, label, _u, best, low, high in shown
                 if name not in bounded and fit_names.at_edge(best, low, high)]
        if edged:
            lines.append("⚠ " + trf(fit_names.EDGE, names=", ".join(edged)))
    first, second = fit_names.split_flags(flags)
    lines += ["⚠ " + trf(template, **values) for template, values in fit_names.grouped(first, many, " ↔ ", tr)]
    rest = [trf(template, **values) for template, values in fit_names.grouped(second, many, " ↔ ", tr)]
    if rest:
        lines.append(trf(fit_names.MORE, notes="; ".join(rest)))
    return lines


def differing_tip(solution, solutions) -> str:
    """For a solution listed next to another of the same composition: the values in which the two differ by more
    than 1 % (σnn/D, σ_q, the tail …), which the table's columns do not show."""
    other = next((s for s in solutions if s is not solution and s.label == solution.label and s.intervals), None)
    if other is None or not solution.intervals:
        return ""
    many = len(solution.model.components) > 1
    there = dict(other.intervals)
    parts = []
    for name, values in solution.intervals:
        mine, theirs = values[0], (there.get(name) or (None,))[0]
        if mine is None or theirs is None:
            continue
        if abs(mine - theirs) > 0.01 * max(abs(mine), abs(theirs), 1e-300):
            unit = fit_names.unit_of(name)
            unit = f" {unit}" if unit else ""
            parts.append(trf(THERE, name=fit_names.display_name(name, many, tr), value=f"{mine:.3g}{unit}",
                             other=f"{theirs:.3g}{unit}"))
    if not parts:
        return ""
    return trf(DIFFERS, other=list(solutions).index(other) + 1, values="; ".join(parts))


def refine_points_line(result, solutions) -> str:
    """After Predict then Refine: “Refine fitted 598 points; Predict parameters fitted 596 of 598 (2 below the
    network's q window) …” when the two are not of the same points (“” otherwise)."""
    predicted = next((s for s in solutions if s.points), None)
    if predicted is None or result is None or predicted.points[0] == result.points:
        return ""
    fitted, given, below, unusable = predicted.points
    why = [trf(fit_names.BELOW_WINDOW, count=below)] if below else []
    if unusable:
        why.append(trf(fit_names.UNUSABLE, count=unusable))
    return trf(REFINE_POINTS, refined=result.points, fitted=fitted, given=given,
               why="; ".join(why) or tr("another selection of points"))


def no_error_reason(path, parameter, result, model) -> str:
    """Why a free value has no error after a fit (the tooltip of “–” in the ± column)."""
    owner, key = path
    if path in tuple(result.at_bounds):
        return tr(NO_ERROR["bound"])
    if owner != "globals" and owner < len(model.components) and key != "Int" and model.components[owner].value("Int") <= 0:
        return tr(NO_ERROR["scale"])
    if owner == "globals" and key in ("res_width", "res_exponent") and model.globals["res_amplitude"].value <= 0:
        return tr(NO_ERROR["tail"])
    lower, upper = parameter.lower, parameter.upper
    if parameter.value <= lower or parameter.value >= upper:
        return tr(NO_ERROR["bound"])
    return tr(NO_ERROR["curve"])


__all__ = [
    "ONLY_ONE", "P_TIP", "POINTS", "SIZES_TIP", "chi2_definition", "count_solutions", "differing_tip", "no_error_reason", "points_text",
    "points_tuple", "probability_line", "refine_points_line", "sizes_cells", "solution_lines",
]
