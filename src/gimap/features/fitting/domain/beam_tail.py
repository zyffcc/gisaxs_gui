"""The beam tail A / (1 + (|q|/w)^ν) in Refine and Search: ranges tied to the curve, a start from its first points,
and the verdicts on values the curve does not determine (Physics and Methods, section 9.6).

With the generic ranges (A in [0, ∞), w in [0.001, 0.1] nm⁻¹) a fit can trade a tail of huge A and a width far below
the first point for a power law A (w/q)^ν over the curve: A runs to 10¹⁷ and more with no defined error. While the
tail still has the generic ranges, a manual fit therefore uses ranges of its curve (``limits``):

* A ≤ ``AMPLITUDE_FACTOR`` × the largest intensity of the fitted points (in units of k);
* w between ``WIDTH_LOW`` × q_min and min(``WIDTH_HIGH`` × q_min, q_max) of the fitted points (the window of the
  proposal network's prior; the shoulder of the tail stays near the curve's first point);
* ν between 1 and ``EXPONENT_HIGH`` (steeper “tails” are steps, not the wings of a beam profile).

Predict parameters keeps its own prior's ranges: a solution of it does not have the generic ranges, so it is never
changed here. ``undetermined`` names the free values whose 1σ error is not finite, or (sizes, spreads, distances,
the tail's shape) above ``UNDETERMINED`` × the value; ``tail_flags`` says when the tail is in use (A above the lower
edge of its range) and A, w or ν is at an edge of its range or has no finite error. ``flagged``, what a series counts:
a tail flag, or a value without a finite error. ``tail_flags`` and ``undetermined`` treat the diffuse term
(``fit_model.DIFFUSE``) as they treat the tail.
"""

from __future__ import annotations

import math

import numpy as np

from .fit_model import INFO, LINEAR, FitModel

AMPLITUDE_FACTOR = 10.0
WIDTH_LOW, WIDTH_HIGH = 0.3, 30.0
EXPONENT_HIGH = 6.0
UNDETERMINED = 10.0
"""A free value whose 1σ error exceeds this many times its size is reported as not determined by the curve."""
TAIL = ("res_amplitude", "res_width", "res_exponent")
DIFFUSE = ("diffuse_amplitude", "diffuse_width", "diffuse_hurst")
RELATIVE = frozenset({"R", "sigma_R", "h", "sigma_h", "D", "sigma_nn", "t_shell", "R_hs", "phi", "res_width",
                      "res_exponent", "diffuse_width"})
"""Values that are sizes, spreads, distances or the tail's shape: their error is judged against the value."""
LOW_POINTS = 6
"""Points at the start of the curve whose slope in log–log gives the tail's starting exponent."""


def is_generic(model: FitModel) -> bool:
    """Whether the tail still has the ranges of a new model (``fit_model.INFO``), i.e. nobody chose them."""
    return all((model.globals[key].lower, model.globals[key].upper) == (INFO[key].lower, INFO[key].upper)
               for key in TAIL)


def limits(model: FitModel, data) -> dict:
    """The tail's ranges for the fitted points ``data`` (``FitData``): ``{key: (lower, upper)}``."""
    q, intensity = np.asarray(data.q, float), np.asarray(data.intensity, float)
    k = float(model.globals["k"].value) or 1.0
    top = float(np.nanmax(intensity)) if intensity.size else 1.0
    q_min, q_max = float(q.min()), float(q.max())
    width_high = max(min(WIDTH_HIGH * q_min, q_max), WIDTH_LOW * q_min * 2.0)
    return {"res_amplitude": (0.0, AMPLITUDE_FACTOR * max(top, 0.0) / abs(k) if top > 0 else INFO["res_amplitude"].upper),
            "res_width": (WIDTH_LOW * q_min, width_high),
            "res_exponent": (INFO["res_exponent"].lower, EXPONENT_HIGH)}


def start(data) -> tuple[float, float]:
    """(w, ν) to start from: the shoulder at the first point and the slope of the first ``LOW_POINTS`` points in
    log–log (ν = 3 when they do not fall)."""
    q, intensity = np.asarray(data.q, float), np.asarray(data.intensity, float)
    good = (q > 0) & (intensity > 0)
    q, intensity = q[good][:LOW_POINTS], intensity[good][:LOW_POINTS]
    exponent = INFO["res_exponent"].value
    if q.size >= 3 and q[-1] > q[0]:
        slope = float(np.polyfit(np.log(q), np.log(intensity), 1)[0])
        if slope < -1.0:
            exponent = min(-slope, EXPONENT_HIGH)
    return (float(q[0]) if q.size else INFO["res_width"].value), float(exponent)


def tied_to_curve(model: FitModel, data, *, with_start: bool = False) -> FitModel:
    """``model`` with the tail's ranges of ``data`` (``limits``), its values held inside them; ``with_start``: w and ν
    also from the first points (``start``) — for a starting model."""
    ranges = limits(model, data)
    width, exponent = start(data) if with_start else (None, None)
    for key, (lower, upper) in ranges.items():
        value = float(model.globals[key].value)
        if with_start and key == "res_width":
            value = width
        elif with_start and key == "res_exponent":
            value = exponent
        model = model.with_parameter(("globals", key), lower=lower, upper=upper, value=min(max(value, lower), upper))
    return model


def undetermined(result, *, finite_only: bool = False) -> tuple:
    """Free values of ``result`` whose 1σ error is not finite, or (``RELATIVE`` values, unless ``finite_only``) above
    ``UNDETERMINED`` × the value. Not counted: a value at an edge of its range (said as such), a scale at 0 and the
    values of a term whose scale is 0 (they change nothing), and a fit that kept its start values (it has no errors)."""
    if getattr(result, "start_kept", False) or getattr(result, "stopped", False):
        return ()
    model, out = result.model, []
    for path, error in result.errors.items():
        parameter = model.get(path)
        if not parameter.free or parameter.tie or path in result.at_bounds or _silent(model, path):
            continue
        if not math.isfinite(error):
            out.append(path)
        elif not finite_only and path[1] in RELATIVE and error > UNDETERMINED * abs(parameter.value):
            out.append(path)
    return tuple(out)


def _silent(model: FitModel, path: tuple) -> bool:
    owner, key = path
    if key in LINEAR:
        return model.get(path).value == 0
    if owner == "globals":
        return any(key in keys and model.globals[keys[0]].value == 0 for keys in (TAIL, DIFFUSE))
    return model.components[owner].value("Int") == 0


def _upper_half(parameter) -> bool:
    return math.isfinite(parameter.upper) and parameter.value >= 0.5 * (parameter.lower + parameter.upper)


def tail_flags(result) -> tuple:
    """The values of the tail and the diffuse term that are a problem while the term is in use (its amplitude free,
    above the lower edge of its range): at an edge of their range, or without a finite error:
    ``(path, "bound" | "undetermined")``."""
    if getattr(result, "start_kept", False) or getattr(result, "stopped", False):
        return ()
    model, out = result.model, []
    for keys in (TAIL, DIFFUSE):
        out += _term_flags(result, model, keys)
    return tuple(out)


def _term_flags(result, model, keys) -> list:
    amplitude = model.globals[keys[0]]
    path_a = ("globals", keys[0])
    if amplitude.value <= 0 or not amplitude.free or amplitude.tie or (
            path_a in result.at_bounds and not _upper_half(amplitude)):
        return []  # no term, or one at the lower edge of its amplitude: not in use
    out = []
    for key in keys:
        path, parameter = ("globals", key), model.globals[key]
        if not parameter.free or parameter.tie:
            continue
        error = result.errors.get(path)
        if path in result.at_bounds:
            out.append((path, "bound"))
        elif error is not None and not math.isfinite(error):
            out.append((path, "undetermined"))
    return out


def flagged(result) -> bool:
    """What a series counts as flagged: the tail in use at an edge of its range or without a finite error, or any
    value without a finite error (``undetermined(finite_only=True)``)."""
    return bool(result is not None and (tail_flags(result) or undetermined(result, finite_only=True)))


__all__ = ["AMPLITUDE_FACTOR", "DIFFUSE", "EXPONENT_HIGH", "TAIL", "UNDETERMINED", "WIDTH_HIGH", "WIDTH_LOW", "flagged", "is_generic",
           "limits", "start", "tail_flags", "tied_to_curve", "undetermined"]
