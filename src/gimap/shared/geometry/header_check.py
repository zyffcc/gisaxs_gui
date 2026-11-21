"""Does the detector distance a frame's own header states agree with the calibration in use?

A header's distance is often nominal (a motor position, the distance of another configuration), so the calibration
stands; but a large disagreement is also what a calibration of another experiment looks like, and only the person can
tell which. Analyze's Data step and the automatic analysis say it in the same words."""

from __future__ import annotations

from typing import Optional

HEADER_DISTANCE_LIMIT = 0.05
"""Relative difference between the header's and the calibration's distance beyond which it is said."""
HEADER_DISTANCE_ITEM = "detector distance"
HEADER_DISTANCE_WHY = (
    "The file header gives a detector distance of {header} mm, the calibration in use {used} mm ({percent} % apart)."
)
HEADER_DISTANCE_HINT = (
    "Check that the calibration belongs to this frame. A header can hold a nominal distance; then the calibration "
    "stands and this line can be ignored."
)


def header_distance_disagreement(
    header_mm: Optional[float], used_mm: Optional[float], limit: float = HEADER_DISTANCE_LIMIT,
) -> Optional[dict]:
    """``{"header", "used", "percent"}`` (as printed: mm with four significant digits, whole per cent) when both
    distances are known and differ by more than ``limit`` of the calibration's; else ``None``."""
    try:
        header, used = float(header_mm), float(used_mm)
    except (TypeError, ValueError):
        return None
    if not (header > 0 and used > 0) or abs(header - used) <= limit * used:
        return None
    return {"header": f"{header:.4g}", "used": f"{used:.4g}", "percent": f"{100 * abs(header - used) / used:.0f}"}


__all__ = ["HEADER_DISTANCE_HINT", "HEADER_DISTANCE_ITEM", "HEADER_DISTANCE_LIMIT", "HEADER_DISTANCE_WHY",
           "header_distance_disagreement"]
