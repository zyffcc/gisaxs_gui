"""Formatting measurement records for human-readable reports."""

from math import isfinite

from src.gimap.shared.numbers import format_number


def peak_value(peak, key, *, compact=False, value_only=False):
    errors = {"q": "q_err", "d_A": "d_err_A", "fwhm": "fwhm_err"}
    error = (peak.get("fit") or {}).get(errors[key])
    return format_number(peak.get(key), error, compact=compact, value_only=value_only)


def peak_shift(first, last):
    """Display the shift to the precision of the less precise q, in Å⁻¹ and percent of its initial q."""
    errors = [(peak.get("fit") or {}).get("q_err") for peak in (first, last)]
    precision = max(errors) if all(error is not None and isfinite(error) and error > 0 for error in errors) else None
    delta = last["q"] - first["q"]
    sign = "+" if delta >= 0 else "-"
    absolute = format_number(abs(delta), precision, value_only=True, digits=2)
    relative = abs(delta / first["q"])
    percent = (format_number(100 * relative, 100 * precision / first["q"], value_only=True)
               if precision is not None else f"{100 * relative:.1f}")
    return sign + absolute, sign + percent + "%"
