"""Parameters of Fitting's own model (Refine, Search) that stop at an edge of their range, resolved automatically.

After a fit, a free parameter at an edge of its min–max range (``FitResult.at_bounds``) is answered as fit1d's
are (``fit1d_edges``): a size, distance or width of the resolution peak gets that side of its range widened
(× 2 upwards, ÷ 2 downwards), the model is fitted again from the values found, and the change is kept when χ²ᵣ
stays within ``ACCEPT`` of before; at most ``ROUNDS`` rounds. Scales at 0 (a term the curve does not need) and the
factor k are left alone.

A value still at an edge after that — or a relative spread at its limit, which is not widened — means the curve is
flat towards the edge: it gives only a bound. ``settle`` walks the value inward, fixed, the others fitted again,
while the total χ² stays within the 1σ tolerance max(1, χ²ᵣ) of the best, and keeps it at the least extreme value
the curve allows, fixed in Model (``BOUNDED``). Each step is a ``Note`` whose ``{name}`` the interface fills from
``values["path"]``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from .fit1d_bounds import Note
from .fit1d_edges import ACCEPT, ROUNDS
from .fit_model import LINEAR, FitModel

SPREADS = frozenset({"sigma_R", "sigma_h", "sigma_D"})
LEFT_ALONE = LINEAR | {"k"}
WIDENED = ("{name} stopped at the {side} edge of its range ({old}): widened to {new} and fitted again "
           "(χ²ᵣ {before} → {after}).")
BOUNDED = ("{name}: every value {relation} {value} fits the curve as well (χ² within {tolerance} of the best), so "
           "the curve gives only this bound. Reported at {value}, the least extreme of them, and fixed there in "
           "Model; the other values are fitted with it.")
SIZE_STEP, SPREAD_STEP, STEP_COUNT = 1.25, 0.05, 8
"""One step inward from an edge: a factor for sizes and widths of the peak, an amount for relative spreads."""
KEPT = ("Widening {count} parameters made the fit worse (χ²ᵣ {before} → {after}): the fit before is kept, "
        "with them at the edge.")
REMAINING = "{name} = {value} is still at the {side} edge of its range: the curve does not determine it beyond there."


def _side(parameter) -> str:
    low, high = parameter.lower, parameter.upper
    if not math.isfinite(high):
        return "lower"
    if low > 0 and parameter.value > 0:  # sizes span decades: compare on a log scale
        return "lower" if math.log(parameter.value / low) <= math.log(high / parameter.value) else "upper"
    return "lower" if parameter.value - low <= high - parameter.value else "upper"


def answer(model: FitModel, path: tuple) -> Optional[tuple[FitModel, dict]]:
    """The model with ``path``'s edge widened, and what was done (``side``, ``old``, ``new``); None when it is
    left alone (a scale, k, a spread, a range that cannot be widened)."""
    key = path[1]
    parameter = model.get(path)
    if key in LEFT_ALONE or key in SPREADS or not parameter.free:
        return None
    side = _side(parameter)
    if side == "upper":
        if not math.isfinite(parameter.upper):
            return None
        old, new = parameter.upper, parameter.upper * 2 if parameter.upper > 0 else parameter.upper + 1
        return model.with_parameter(path, upper=new), dict(side=side, old=old, new=new)
    if parameter.lower <= 0:
        return None
    old, new = parameter.lower, parameter.lower / 2
    return model.with_parameter(path, lower=new), dict(side=side, old=old, new=new)


def _total(result) -> float:
    return float(result.chi2_reduced) * max(int(result.points) - int(result.free), 1)


def _inward(parameter, side: str, key: str, step: int) -> float:
    if key in SPREADS:
        moved = parameter.value + (-1 if side == "upper" else 1) * SPREAD_STEP * step
    elif parameter.value > 0:
        moved = parameter.value / SIZE_STEP ** step if side == "upper" else parameter.value * SIZE_STEP ** step
    else:
        return parameter.value
    return min(max(moved, parameter.lower), parameter.upper)


def settle(fit: Callable[[FitModel], object], result, *, limit: int = 3,
           cancelled: Callable[[], bool] = lambda: False) -> tuple:
    """The values still at an edge, each fixed at the least extreme value the curve allows (see the module):
    (the result kept, the notes of what was done)."""
    if result.stopped or not math.isfinite(result.chi2_reduced):
        return result, []
    best, tolerance, actions = _total(result), max(1.0, float(result.chi2_reduced)), []
    edges = [path for path in result.at_bounds if path[1] not in LEFT_ALONE and result.model.get(path).free]
    for path in edges[:limit]:
        if cancelled():
            break
        parameter = result.model.get(path)
        side = _side(parameter)
        kept = None
        for step in range(1, STEP_COUNT + 1):
            value = _inward(parameter, side, path[1], step)
            if value == parameter.value or (kept is not None and value == kept[0]):
                break
            trial = fit(result.model.with_parameter(path, value=value, free=False))
            if trial.stopped or _total(trial) - best > tolerance:
                break
            kept = (value, trial)
        if kept is None:
            continue
        value, result = kept
        actions.append(Note(BOUNDED, dict(path=list(path), name=f"{path[0]}:{path[1]}", value=f"{value:.4g}",
                                          relation="≥" if side == "upper" else "≤", tolerance=f"{tolerance:.3g}")))
    return result, actions


def unused(result) -> list:
    """Scales at 0: terms the curve does not need (said, not answered)."""
    return [path for path in result.at_bounds if path[1] in LINEAR]


def resolve(fit: Callable[[FitModel], object], result, *, rounds: int = ROUNDS,
            cancelled: Callable[[], bool] = lambda: False) -> tuple:
    """``fit(model) -> FitResult``; ``result``: the fit made. Returns the result kept, the notes of what was done
    and those of what is still at an edge."""
    actions = []
    for _round in range(rounds):
        if result.stopped or cancelled():
            break
        model, done = result.model, []
        for path in result.at_bounds:
            change = answer(model, path)
            if change is not None:
                model = change[0]
                done.append((path, change[1]))
        if not done:
            break
        trial = fit(model)
        before, after = result.chi2_reduced, trial.chi2_reduced
        if trial.stopped or not after <= before * ACCEPT:
            actions.append(Note(KEPT, dict(count=len(done), before=f"{before:.3g}", after=f"{after:.3g}")))
            break
        for path, change in done:
            values = dict(path=list(path), name=f"{path[0]}:{path[1]}", side=change["side"], before=f"{before:.3g}",
                          after=f"{after:.3g}")
            actions.append(Note(WIDENED, {**values, "old": f"{change['old']:.4g}", "new": f"{change['new']:.4g}"}))
        result = trial
    if rounds:
        result, bounds = settle(fit, result, cancelled=cancelled)
        actions.extend(bounds)
    remaining = []
    for path in result.at_bounds:
        if path[1] in LEFT_ALONE or not result.model.get(path).free:
            continue
        parameter = result.model.get(path)
        remaining.append(Note(REMAINING, dict(path=list(path), name=f"{path[0]}:{path[1]}", side=_side(parameter),
                                              value=f"{parameter.value:.4g}")))
    return result, actions, remaining


__all__ = ["BOUNDED", "KEPT", "REMAINING", "SPREADS", "WIDENED", "answer", "resolve", "settle", "unused"]
