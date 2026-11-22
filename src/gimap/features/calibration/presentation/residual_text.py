"""Ring-point RMS of the calibration, or ring-radius RMS when no point fit was recorded."""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.numbers import format_number

POINTS = "ring points"
RADII = "ring radii"


def shown_rms(candidate) -> tuple[float, str]:
    """``(value in px, what it is: POINTS or RADII)`` of a calibration solution."""
    points = getattr(candidate, "point_rms_px", None)
    if points is not None:
        return float(points), POINTS
    return float(candidate.rms_residual_px), RADII


TITLES = {POINTS: "Ring-point RMS:", RADII: "Ring-radius RMS:"}
"""The title of the RMS row: which residual the value is."""


def rms_text(candidate) -> str:
    value, _kind = shown_rms(candidate)
    return f"{format_number(value)} px"


def rms_title(candidate) -> str:
    return tr(TITLES[shown_rms(candidate)[1]])


def rms_tip(candidate) -> str:
    _, kind = shown_rms(candidate)
    if kind == POINTS:
        tip = tr("RMS distance of the fitted ring points from the rings of this geometry.")
        initial = getattr(candidate, "initial_point_rms_px", None)
        if initial is not None:
            tip += "\n" + trf("Before refinement, the RMS on these points was {rms} px.", rms=format_number(initial))
            flat = candidate.untilted_point_rms_px
            if flat is not None:
                tip += "\n" + trf("Fitting the same points with zero tilt gives {rms} px.", rms=format_number(flat))
        return tip
    return tr("RMS difference between observed and predicted ring radii at the automatic ring-matching step "
              "(azimuthal average).")


__all__ = ["POINTS", "RADII", "TITLES", "rms_text", "rms_tip", "rms_title", "shown_rms"]
