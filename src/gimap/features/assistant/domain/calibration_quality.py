"""Verdicts on calibrations: is a geometry good, and which standard does an image show.

The decisive number is where the standard's lines land in q (``line_check`` of
a calibration): within 0.2 % on average is good, within 0.5 % usable.  The
ring-fit residual in pixels only decides when no line check exists, because a
flat detector model fitted to a tilted wide-angle detector leaves pixel
residuals that say little about the q accuracy.
"""

from __future__ import annotations

import math
from typing import Optional

GOOD_RINGS = 3
"""Fewest rings or lines that make a calibration trustworthy."""
GOOD_RMS_PX = 1.5
"""Largest ring-fit residual (pixels) of a good fit, when no line check exists."""
GOOD_RELATIVE_DQ = 0.002
"""Mean relative q error of the standard's lines for a good calibration."""
USABLE_RELATIVE_DQ = 0.005
"""Mean relative q error up to which a calibration is still usable."""


def rms_px(result: dict) -> float:
    """Ring-fit residual in pixels (infinite when unknown)."""
    value = result.get("rms_residual_px")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else math.inf


def compare_verdict(fits: list[dict], distance_hint: Optional[float] = None) -> str:
    """Which standard an image shows, judged from separate fits (best first)."""
    checked = [(fit, line_quality(fit)) for fit in fits]
    measured = sorted((pair for pair in checked if pair[1] is not None), key=lambda pair: pair[1])
    if measured:
        best, error = measured[0]
        lines = best["line_check"]["lines_checked"]
        if error > USABLE_RELATIVE_DQ:
            return (
                f"none fits well: the best, {best.get('standard')}, still puts its lines {error:.2%} off "
                "their q; the image may show another standard, or the energy or pixel size is wrong"
            )
        if len(measured) == 1 or measured[1][1] >= 1.5 * error:
            return f"clear: {best.get('standard')} — {lines} lines within {error:.2%} of their q"
        return (
            f"ambiguous: {best.get('standard')} ({error:.2%}) and {measured[1][0].get('standard')} "
            f"({measured[1][1]:.2%}) place their lines about equally well; check a log or ask the user"
        )
    best = fits[0]
    rings, rms = int(best.get("matched_rings") or 0), rms_px(best)
    name = best.get("standard")
    if rings < GOOD_RINGS or rms > GOOD_RMS_PX:
        return (
            f"none fits well (best {name}: {rings} rings, rms {rms:.2g} px): the image may show another "
            "standard, the energy or pixel size may be wrong, or the rings are too weak"
        )
    if len(fits) == 1:
        return f"clear: {name} ({rings} rings, rms {rms:.2g} px)"
    second = fits[1]
    second_rings, second_rms = int(second.get("matched_rings") or 0), rms_px(second)
    if rings >= second_rings + 2 or rms <= 0.6 * second_rms:
        return (
            f"clear: {name} ({rings} rings, rms {rms:.2g} px) beats {second.get('standard')} "
            f"({second_rings} rings, rms {second_rms:.2g} px)"
        )
    if distance_hint:
        best_off = abs(best["distance_mm"] - distance_hint) / distance_hint
        second_off = abs(second["distance_mm"] - distance_hint) / distance_hint
        if best_off < 0.1 <= second_off:
            return f"clear: {name}, and its distance agrees with the expected {distance_hint:g} mm"
        if second_off < 0.1 <= best_off:
            return (
                f"probably {second.get('standard')}: its distance agrees with the expected {distance_hint:g} mm, "
                f"although {name} matches as many rings"
            )
    return (
        f"ambiguous: {name} and {second.get('standard')} fit about equally well; check the distance in the "
        "header or a log, or ask the user"
    )


def line_quality(result: dict) -> Optional[float]:
    """Mean relative q error of the standard's lines, when enough lines were measured."""
    check = result.get("line_check") or {}
    if int(check.get("lines_checked") or 0) >= GOOD_RINGS and check.get("mean_relative") is not None:
        return float(check["mean_relative"])
    return None


def assessment(result: dict) -> str:
    """One-line verdict on a calibration: good, usable, doubtful or unreliable, and why."""
    relative = line_quality(result)
    if relative is not None:
        lines = result["line_check"]["lines_checked"]
        if relative <= GOOD_RELATIVE_DQ:
            return f"good: {lines} lines of the standard land within {relative:.2%} of their q on average"
        if relative <= USABLE_RELATIVE_DQ:
            return f"usable: {lines} lines within {relative:.2%} of their q; peak positions good to about that"
        return f"doubtful: the standard's lines are {relative:.2%} off on average; try another image or standard"
    rings = int(result.get("matched_rings") or 0)
    rms = result.get("rms_residual_px")
    rms = float(rms) if isinstance(rms, (int, float)) and math.isfinite(rms) else None
    if rings >= GOOD_RINGS and rms is not None and rms <= GOOD_RMS_PX and result.get("confidence") in ("High", "Medium"):
        return "good: several rings match the standard with a small residual"
    if rings < 2:
        return "unreliable: fewer than two rings matched, the distance is ambiguous"
    return "doubtful: few rings or a large residual; compare the alternatives or another image"


__all__ = [
    "GOOD_RELATIVE_DQ",
    "GOOD_RINGS",
    "GOOD_RMS_PX",
    "USABLE_RELATIVE_DQ",
    "assessment",
    "compare_verdict",
    "line_quality",
    "rms_px",
]
