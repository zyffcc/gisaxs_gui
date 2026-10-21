"""fit1d parameters that stop at an edge, resolved automatically.

After a fit, each population parameter at an edge of its range (the kit's ``at_bound``, or within ``NEAR`` of
the edge of a range constraint) is one of:

* ``default`` — an edge of the kit's default box (tied to the curve's q window): widen that side;
* ``user`` — an edge of a range the user set in Model: widen that side as well (the user's fixed values are
  never touched);
* ``physical`` — σR/R or w/D at their lower limit, D at 2R (touching particles): nothing to widen;
* ``other`` — a low-q term: only said.

``resolve`` widens them together, fits again and keeps the change when χ²/point stays within ``ACCEPT`` of before
(a wider range can only fit better or equally); else it goes back to the fit before. At most ``ROUNDS`` rounds.

A value still at an edge after that (or at a physical one) means the curve is flat towards the edge: it does not
tell the value, only a bound. ``settle`` then walks it inward — fixed, everything else fitted again — while χ²
stays within the 1σ tolerance max(1, χ²/dof) of the best, and reports it at the least extreme value the curve
allows, as a one-sided bound (``≥`` / ``≤``; ``mark_bounds`` writes it into the result's table). When even the
first step inward is clearly worse and the edge is a limit of the model itself, the curve wants a value beyond the
model (``AT_LIMIT``). Each step is a ``Note`` (an English template, translated by the interface).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Optional

from .fit1d_bounds import D_PER_R, LABELS, LIMITS, WIDTHS, Constraint, Note, measurable, number, unit_text

ROUNDS = 2
ACCEPT = 1.02
"""A change is kept when χ²/point after it is at most this factor of χ²/point before."""
NEAR = 0.01
"""A value within this share of a range (log for R and D) of its edge is at that edge. The kit flags only values
within its own, much smaller tolerance; a value 1 % from the edge of a user's range is held there just the same."""
PHYSICAL = {("sigma_R_rel", "lower"): "a smaller size spread cannot be told apart from none on this curve",
            ("w_rel", "lower"): "a better order cannot be told apart on this curve",
            ("D_nm", "lower"): "D = 2R means touching particles"}

WIDENED_DEFAULT = ("{name} stopped at the {side} edge of the default range ({old}{unit}): widened to {new}{unit} "
                   "and fitted again (χ²/point {before} → {after}).")
WIDENED_USER = ("{name} stopped at the {side} edge of your range ({old}{unit}): widened to {new}{unit} and fitted "
                "again (χ²/point {before} → {after}).")
KEPT = ("Widening or fixing {names} made the fit worse (χ²/point {before} → {after}): the fit before is kept, "
        "with them at the edge.")
REMAINING = ("{name} = {value}{unit} is still at the {side} edge of its range: the curve does not determine it "
             "beyond there.")
AT_OTHER = "The low-q term {name} stops at the {side} edge of its range."
BOUNDED = ("{name}: every value {relation} {value}{unit} fits the curve as well (χ² within {tolerance} of the best), "
           "so the curve gives only this bound. Reported at {value}{unit}, the least extreme of them; the other "
           "values are fitted with it there.")
AT_LIMIT = ("{name} = {value}{unit} sits at the {side} limit of the model, and the curve wants a value beyond it: "
            "the model may be missing something (another population, the background, the resolution).")
STEPS = {"R_nm": 1.25, "D_nm": 1.25, "sigma_R_rel": 0.05, "w_rel": 0.05}
"""How far one step goes inward from an edge: a factor for R and D, an amount for the relative widths."""
STEP_COUNT = 8


@dataclass(frozen=True)
class Edge:
    target: str
    key: str
    side: str
    kind: str
    value: Optional[float]

    @property
    def name(self) -> str:
        return f"{self.target}.{self.key}" if self.key else self.target

    @property
    def label(self) -> str:
        return f"{self.target} {LABELS.get(self.key, self.key)}" if self.key else self.target

    def to_dict(self) -> dict:
        return {**asdict(self), "name": self.name}


def _value(populations, target: str, key: str) -> Optional[float]:
    for population in populations or ():
        if population.get("name") == target and isinstance(population.get(key), (int, float)):
            return float(population[key])
    return None


def near_edges(constraints: Iterable[Constraint], populations, tolerance: float = NEAR) -> list[str]:
    """``"pop1.R_nm@upper"`` for each active population parameter that sits at an edge of its range constraint."""
    out = []
    for item in constraints:
        if item.kind != "range" or item.key not in LIMITS or not item.high > item.low:
            continue
        population = next((p for p in populations or () if p.get("name") == item.target), None)
        value = _value(populations, item.target, item.key)
        if value is None or population is None or population.get("inactive"):
            continue
        if item.key in ("R_nm", "D_nm") and item.low > 0 and value > 0:
            share = math.log(value / item.low) / math.log(item.high / item.low)
        else:
            share = (value - item.low) / (item.high - item.low)
        if share <= tolerance:
            out.append(f"{item.name}@lower")
        elif share >= 1 - tolerance:
            out.append(f"{item.name}@upper")
    return out


def classify_edges(at_bound: Iterable[str], constraints: Iterable[Constraint], populations=()) -> list[Edge]:
    """``at_bound``: the kit's ``flags.at_bound`` (``"pop1.R_nm@upper"``), with ``near_edges`` added;
    ``populations``: the kit's result populations (``name``, ``R_nm``, …)."""
    constraints = list(constraints)
    by_name = {item.name: item for item in constraints}
    edges = []
    for entry in dict.fromkeys([*at_bound, *near_edges(constraints, populations)]):
        name, _, side = str(entry).partition("@")
        target, _, key = name.partition(".")
        given = by_name.get(name)
        if key not in LIMITS:
            kind = "other"
        elif given is not None and given.kind == "fixed":
            continue  # a fixed value is where it was put
        elif (key, side) in PHYSICAL:
            kind = "physical"
        elif given is not None and given.source == "model":
            kind = "user"
        else:
            kind = "default"
        edges.append(Edge(target, key, side, kind, _value(populations, target, key)))
    return edges


def default_range(target: str, key: str, populations, q_min: float, q_max: float) -> tuple[float, float]:
    """The kit's default box of a population parameter on this curve (D's upper edge is 40 R of the fitted R)."""
    window = measurable(q_min, q_max)
    if key == "R_nm":
        return window["R_nm"]
    if key == "D_nm":
        radius = _value(populations, target, "R_nm") or window["R_nm"][0]
        return D_PER_R[0] * window["R_nm"][0], D_PER_R[1] * radius
    return WIDTHS[key]


def resolution(edge: Edge, constraints: Iterable[Constraint], populations, q_min: float,
               q_max: float) -> Optional[Constraint]:
    """The constraint that answers ``edge``: its side widened (× 2 or ÷ 2 for R and D, to the kit's limit for the
    widths); None when nothing can be done."""
    current = next((item for item in constraints if item.name == edge.name), None)
    component = None if current is None else current.component
    if edge.kind not in ("default", "user"):  # a physical edge cannot be widened: ``settle`` gives its bound
        return None
    if current is not None and current.kind == "range":
        low, high = current.low, current.high
    else:
        low, high = default_range(edge.target, edge.key, populations, q_min, q_max)
    lowest, highest = LIMITS[edge.key]
    if edge.key in ("R_nm", "D_nm"):
        reach = edge.value if edge.value is not None else (high if edge.side == "upper" else low)
        new_low, new_high = (low, max(high, reach) * 2) if edge.side == "upper" else (min(low, reach) / 2, high)
    else:
        new_low, new_high = (low, highest) if edge.side == "upper" else (lowest, high)
    new_low, new_high = max(new_low, lowest), min(new_high, highest)
    if (new_low, new_high) == (low, high) or new_low >= new_high:
        return None
    return Constraint(edge.target, edge.key, "range", new_low, new_high, component, "widened")


def _old_edge(edge: Edge, before: Optional[Constraint], populations, q_min: float, q_max: float) -> float:
    if before is not None and before.kind == "range":
        return before.high if edge.side == "upper" else before.low
    low, high = default_range(edge.target, edge.key, populations, q_min, q_max)
    return high if edge.side == "upper" else low


def action_note(edge: Edge, before: Optional[Constraint], after: Constraint, chi2: tuple, populations,
                q_min: float, q_max: float) -> Note:
    values = dict(name=edge.label, side=edge.side, unit=unit_text(edge.key), before=f"{chi2[0]:.3g}",
                  after=f"{chi2[1]:.3g}")
    old = _old_edge(edge, before, populations, q_min, q_max)
    new = after.high if edge.side == "upper" else after.low
    template = WIDENED_USER if edge.kind == "user" else WIDENED_DEFAULT
    return Note(template, {**values, "old": number(old), "new": number(new)})


def remaining_note(edge: Edge) -> Note:
    if edge.kind == "other":
        return Note(AT_OTHER, dict(name=edge.name, side=edge.side))
    return Note(REMAINING, dict(name=edge.label, side=edge.side, unit=unit_text(edge.key),
                                value="?" if edge.value is None else number(edge.value)))


@dataclass
class Resolved:
    """The fit kept, the constraints it was made with, what was done (``actions``) and what is still at an edge."""

    result: dict
    raw: dict
    constraints: list
    actions: list
    edges: list
    refits: int


def _total(result: dict) -> tuple[float, float]:
    """(χ², degrees of freedom) of a kit result."""
    summary = result.get("summary") or {}
    chi2, dof = summary.get("chi2"), summary.get("dof") or summary.get("n_points") or 1
    if not isinstance(chi2, (int, float)):
        chi2 = (summary.get("chi2_per_point") or math.inf) * (summary.get("n_points") or 1)
    return float(chi2), float(max(dof, 1))


def _inward(edge: Edge, value: float, step: int) -> float:
    lowest, highest = LIMITS[edge.key]
    toward_lower = edge.side == "upper"
    if edge.key in ("R_nm", "D_nm"):
        moved = value / STEPS[edge.key] ** step if toward_lower else value * STEPS[edge.key] ** step
    else:
        moved = value - STEPS[edge.key] * step if toward_lower else value + STEPS[edge.key] * step
    return min(max(moved, lowest), highest)


def settle(fit: Callable[[list], tuple], resolved: "Resolved", *, cancelled: Callable[[], bool] = lambda: False,
           report: Callable[[str], None] = lambda _line: None, limit: int = 3) -> "Resolved":
    """The values still at an edge, each at the least extreme value the curve allows (see the module)."""
    result, raw, constraints = resolved.result, resolved.raw, list(resolved.constraints)
    actions, refits = list(resolved.actions), resolved.refits
    chi2_best, dof = _total(result)
    tolerance = max(1.0, chi2_best / dof)
    for edge in [edge for edge in resolved.edges if edge.kind != "other" and edge.value is not None][:limit]:
        if cancelled():
            break
        report("A value stays at the edge of its range: finding the least extreme value the curve allows")
        kept, failed = None, None
        others = [c for c in constraints if c.name != edge.name]
        for step in range(1, STEP_COUNT + 1):
            value = _inward(edge, edge.value, step)
            if kept is not None and value == kept[0] or value == edge.value:
                break
            trial = others + [Constraint(edge.target, edge.key, "fixed", value, value, None, f"bound_{edge.side}")]
            outcome = fit(trial)
            refits += 1
            if _total(outcome[0])[0] - chi2_best <= tolerance:
                kept, failed = (value, outcome, trial), None
            else:
                failed = value
                break
        if kept is None and failed is not None:  # even one step is too far: half of it
            value = math.sqrt(edge.value * failed) if edge.key in ("R_nm", "D_nm") else (edge.value + failed) / 2
            trial = others + [Constraint(edge.target, edge.key, "fixed", value, value, None, f"bound_{edge.side}")]
            outcome = fit(trial)
            refits += 1
            if _total(outcome[0])[0] - chi2_best <= tolerance:
                kept = (value, outcome, trial)
        values = dict(name=edge.label, side=edge.side, unit=unit_text(edge.key))
        if kept is None:
            lowest, highest = LIMITS[edge.key]
            if edge.value <= lowest * 1.0001 or edge.value >= highest * 0.9999:
                actions.append(Note(AT_LIMIT, {**values, "value": number(edge.value)}))
            continue
        value, (result, raw), constraints = kept
        actions.append(Note(BOUNDED, {**values, "value": number(value), "tolerance": f"{tolerance:.3g}",
                                      "relation": "≥" if edge.side == "upper" else "≤"}))
    best = (raw.get("results") or [{}])[0]
    edges = classify_edges((best.get("flags") or {}).get("at_bound") or (), constraints, result["populations"])
    edges = [edge for edge in edges if not any(c.name == edge.name and is_bound(c) for c in constraints)]
    return Resolved(result, raw, constraints, actions, edges, refits)


def is_bound(constraint) -> bool:
    return str(constraint.source).startswith("bound_")


def relation(constraint) -> str:
    """``≥`` for a value that went to the upper edge (every larger value fits as well), ``≤`` for the lower."""
    return "≥" if constraint.source == "bound_upper" else "≤"


def mark_bounds(result: dict, constraints) -> None:
    """The values given as a bound (``settle``): their rows of the result's table say ``≥ x`` / ``≤ x``."""
    bounds = {c.name: c for c in constraints if is_bound(c)}
    for row in result.get("table") or ():
        bound = bounds.get(row.get("key"))
        if bound is not None:
            row["status"] = "one_sided"
            row["text"] = f"{relation(bound)} {number(bound.low)}"
            row["error"] = None


def chi2_of(result: dict) -> float:
    value = (result.get("summary") or {}).get("chi2_per_point")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else math.inf


def resolve(fit: Callable[[list], tuple], constraints: list, result: dict, raw: dict, q_min: float, q_max: float,
            *, rounds: int = ROUNDS, cancelled: Callable[[], bool] = lambda: False,
            report: Callable[[str], None] = lambda _line: None) -> Resolved:
    """``fit(constraints) -> (result, raw)``: the kit's fit with these constraints; ``result``/``raw``: the fit made
    with ``constraints``. Rounds of ``resolution`` until nothing is at an edge that can be answered."""
    constraints, actions, refits = list(constraints), [], 0

    def edges_of(result, raw, constraints):
        best = (raw.get("results") or [{}])[0]
        return classify_edges((best.get("flags") or {}).get("at_bound") or (), constraints, result["populations"])

    edges = edges_of(result, raw, constraints)
    for _round in range(rounds):
        changes = []
        for edge in edges:
            answer = resolution(edge, constraints, result["populations"], q_min, q_max)
            if answer is not None:
                changes.append((edge, next((c for c in constraints if c.name == edge.name), None), answer))
        if not changes or cancelled():
            break
        report("A parameter stopped at the edge of its range: fitting again with it resolved")
        trial = [c for c in constraints if c.name not in {answer.name for _e, _b, answer in changes}]
        trial += [answer for _e, _b, answer in changes]
        new_result, new_raw = fit(trial)
        refits += 1
        before, after = chi2_of(result), chi2_of(new_result)
        if not after <= before * ACCEPT:
            actions.append(Note(KEPT, dict(names=", ".join(edge.label for edge, _b, _a in changes),
                                           before=f"{before:.3g}", after=f"{after:.3g}")))
            break
        actions.extend(action_note(edge, old, answer, (before, after), result["populations"], q_min, q_max)
                       for edge, old, answer in changes)
        result, raw, constraints = new_result, new_raw, trial
        edges = edges_of(result, raw, constraints)
    return Resolved(result, raw, constraints, actions, edges, refits)


__all__ = ["ACCEPT", "AT_LIMIT", "AT_OTHER", "BOUNDED", "Edge", "KEPT", "STEPS", "is_bound", "mark_bounds", "relation",
           "settle", "NEAR", "PHYSICAL", "REMAINING", "ROUNDS",
           "Resolved", "WIDENED_DEFAULT", "WIDENED_USER", "action_note", "chi2_of", "classify_edges",
           "default_range", "near_edges", "remaining_note", "resolution", "resolve"]
