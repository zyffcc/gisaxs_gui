"""Validate optional roll estimates in a saved instrument profile, PONI or session."""

import math


def measurement_record(value):
    if value is None:
        return None
    error = "Invalid sample roll measurement."
    if not isinstance(value, dict) or not isinstance(value.get("method"), str):
        raise ValueError(error)
    result = dict(value)
    for name in ("horizon", "symmetry", None):
        estimate = result if name is None else result.get(name)
        if estimate is None or (name is None and estimate.get("deg") is None and estimate.get("error_deg") is None):
            continue
        try:
            deg, uncertainty = float(estimate["deg"]), float(estimate["error_deg"])
        except (ValueError, KeyError, TypeError, OverflowError):
            raise ValueError(error) from None
        if not (math.isfinite(deg) and math.isfinite(uncertainty) and uncertainty > 0):
            raise ValueError(error)
        if name:
            result[name] = dict(estimate, deg=deg, error_deg=uncertainty)
        else:
            result.update(deg=deg, error_deg=uncertainty)
    return result
