"""The one RMS residual the calibration window shows for a solution, and what it is.

Two residuals exist for a calibration with the tilt refined, and the window used to show both (the RMS row and,
in the tilt's tooltip, another number): the ring radii of the circle fit before the tilt, and the points on the
rings after it. The window now shows one, the residual of the geometry it displays, and says which:

* tilt refined: the RMS distance (px) of the points sampled on the matched rings from the rings of the tilted
  geometry (``tilt_fit.point_rms_px``); the same points without the tilt are named in the tooltip for comparison;
* no tilt: the RMS difference (px) between the radius of each matched ring (azimuthal average about the centre)
  and the radius the geometry predicts (``peak_matcher``)."""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf

POINTS = "ring points"
RADII = "ring radii"


def shown_rms(candidate) -> tuple[float, str]:
    """``(value in px, what it is: POINTS or RADII)`` of a calibration solution."""
    points = getattr(candidate, "point_rms_px", None)
    if getattr(candidate, "tilt_refined", False) and points is not None:
        return float(points), POINTS
    return float(candidate.rms_residual_px), RADII


TITLES = {POINTS: "Ring-point RMS:", RADII: "Ring-radius RMS:"}
"""The title of the RMS row: which residual the value is."""


def rms_text(candidate, digits: int = 3) -> str:
    value, _kind = shown_rms(candidate)
    return f"{value:.{digits}f} px"


def rms_title(candidate) -> str:
    return tr(TITLES[shown_rms(candidate)[1]])


def rms_tip(candidate) -> str:
    value, kind = shown_rms(candidate)
    if kind == POINTS:
        tip = tr("RMS distance of the points sampled on the matched rings from the rings of this geometry, with "
                 "the detector tilt.")
        flat = getattr(candidate, "untilted_point_rms_px", None)
        if flat is not None:
            tip += " " + trf("The same points without the tilt: {flat} px.", flat=f"{flat:.3f}")
        return tip
    return tr("RMS difference between the radius of each matched ring (azimuthal average about the centre) and "
              "the radius this geometry predicts.")


__all__ = ["POINTS", "RADII", "TITLES", "rms_text", "rms_tip", "rms_title", "shown_rms"]
