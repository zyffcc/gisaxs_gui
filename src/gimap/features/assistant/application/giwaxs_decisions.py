"""The standing decisions of a run: what its numbers are and are not, said in every report.

These lines do not change a number; they say what a number means, so that a reader of the report (or of
``report.json``) does not take more from it than the analysis supports:

* ``q`` — the scattering vector is kinematic (vacuum, no refraction correction), with the size of the
  q_z shift refraction can cause at this frame's αi and wavelength (k·sin αi); GIWAXS and GISAXS alike, each in
  its own words (for GISAXS: q_z and the Yoneda band move, q_y and 2π/q* barely);
* ``uncertainties`` — the ± are statistical only, from the bins' own errors (Poisson counting statistics or
  the scatter of each bin's pixels, whichever the frame's reduction used); for GISAXS, the σ of the cuts that
  the fit's χ²ᵣ is normalised by;
* ``flat field`` — when the frame was divided by one: its file and the pixels it left out;
* ``peak search`` — when more candidates qualified than were fitted (``PeakSearch.truncated``), how many,
  and how to fit more;
* ``texture rule`` and ``ring orientation`` — what the texture sentences compare, and the rings whose f depends
  on the bins next to the sample horizon.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from src.gimap.shared.curve_error_description import ERROR_MODELS, curve_error_description

from ..domain import MAX_PEAKS
from ..domain.orientation import NORMAL_CHI, PLANE_CHI, TEXTURE_SIGMAS, WEAK_HERMAN

MORE_PEAKS = "--max-peaks N on the command line; max_peaks of find_peaks or run_standard_pipeline"
"""How to fit more candidates than the default (the GUI's Run Automatic Analysis has no setting for it)."""


REFRACTION = {
    "giwaxs": ("near or below the film's critical angle, refraction shifts q_z by up to about k·sin αi",
               "; in-plane components barely change"),
    "gisaxs": ("near the critical angle refraction mostly shifts q_z, that of the Yoneda band (αf ≈ αc) most, by an "
               "amount of the order of k·sin αi", "; q_y and the length scale 2π/q* of the horizontal cut barely change"),
}
"""What refraction does to the kinematic q of each technique: (the shift, what barely changes)."""
UNCERTAINTIES = {
    "giwaxs": ("the ± of q, d, FWHM and sizes (the peak fits) come from {model}; "
               "the ± of a peak fit whose reduced χ² is above 1 are multiplied by √χ²_r (a fit above χ²_r = 10 is "
               "flagged as a poor fit in the peak table); "
               "the Monte Carlo estimate of σ_f uses the errors of the measured I(χ) bins before mirror filling. "
               "The separate systematic range for f varies the background, exit-angle threshold and wedge fill. "
               "Geometry and refraction uncertainties are excluded"),
    "gisaxs": ("the σ of the cuts' points (the σ column of the saved curves), by which the fit's χ²ᵣ is normalised, "
               "come from {model}; the intervals of R, h and D are the posterior of Predict parameters given these σ "
               "and the model; the geometry (beam centre, distance, αi) and refraction are not included"),
}
"""Which ± a report gives and what they leave out, by technique (``{model}``: the frame's error model)."""


def refraction_decision(geometry: Optional[dict], technique: str = "giwaxs", *, given: bool = False) -> dict:
    """q is kinematic: the q_z shift refraction can cause near or below the film's critical angle, k·sin αi, for
    the αi and wavelength the run used, in the words of ``technique`` (``giwaxs`` or ``gisaxs``). ``given``: the run
    was given its αi (an αi of 0° is then the person's, not the stand-in for a missing one)."""
    geometry = geometry or {}
    critical = geometry.get("critical_angle_deg")
    if critical:
        return {"what": "q", "decision": f"q_z corrected for refraction in the film (αc = {float(critical):g}°)",
                "why": "q_z is the one inside the film, k(√(sin²αf − sin²αc) + √(sin²αi − sin²αc)); q∥ does not "
                       "change; pixels below αc are left out of what uses q_z"}
    wavelength, incidence = geometry.get("wavelength_A"), geometry.get("incidence_deg")
    shifted, barely = REFRACTION.get(technique, REFRACTION["giwaxs"])
    why = shifted
    if wavelength and incidence:
        shift = 2.0 * math.pi / float(wavelength) * math.sin(math.radians(float(incidence)))
        why += f" = {shift:.3g} Å⁻¹ here (αi = {float(incidence):g}°, λ = {float(wavelength):.4g} Å){barely}"
    elif given:
        why += "; αi is 0° as given, so k·sin αi gives no size for that shift"
    else:
        why += "; αi is 0° here (none given), so the size of that shift is not known"
    return {"what": "q", "decision": "kinematic (no refraction correction)", "why": why}


def uncertainty_decision(errors: Optional[str], technique: str = "giwaxs", *, corrections=None) -> dict:
    """State the statistical source for each observable and the separate systematic range for GIWAXS f."""
    model = curve_error_description(errors, corrections or {})
    template = UNCERTAINTIES.get(technique, UNCERTAINTIES["giwaxs"])
    return {"what": "uncertainties", "decision": "statistical ± and systematic range for f" if technique == "giwaxs" else "statistical only", "why": template.format(model=model)}


def flat_field_decision(status: Optional[dict]) -> Optional[dict]:
    """The flat field the frame was divided by (``get_status``: ``corrections.flat_field``): its file and the pixels
    it left out; ``None`` without one."""
    flat = ((status or {}).get("corrections") or {}).get("flat_field")
    if not flat:
        return None
    return {
        "what": "flat field",
        "decision": f"divided by {flat.get('file')} ({flat.get('masked_pixels')} pixels left out)",
        "why": (
            "the reduction divides each corrected pixel by its flat-field response F_ff, normalised to mean 1. "
            f"It excludes nonfinite responses and values at or below {flat.get('floor', 0.05):g}."
        ),
    }


def peak_cap_decision(search, asked: int = MAX_PEAKS) -> Optional[dict]:
    """How many candidate peaks the search found when that was more than it fitted, or the cap that was asked for
    (``None``: every candidate was fitted at the default cap)."""
    if search is None:
        return None
    candidates, cap = int(getattr(search, "candidates", 0)), int(getattr(search, "max_peaks", MAX_PEAKS))
    if candidates > cap:
        return {
            "what": "peak search", "decision": (
                f"{candidates} candidate peaks found; the {cap} most prominent were fitted, "
                f"{len(getattr(search, 'peaks', ()) or ())} of them gave a peak"),
            "why": (
                ("the default cap, for speed" if cap == MAX_PEAKS else "the cap asked for")
                + ": the other candidates were not fitted, so they are not in the peak table, the series hints, the "
                "sector comparison or the rings analysed. A candidate is a maximum of the smoothed I(q) above the "
                "thresholds (3σ, 2 % above the background, 2 bins wide); fitting merges or rejects many of them, so "
                f"fitting all gives fewer peaks than candidates ({MORE_PEAKS})"
            ),
        }
    if int(asked) != MAX_PEAKS:
        return {"what": "peak search", "decision": f"at most {cap} candidates fitted",
                "why": f"as asked; {candidates} qualified, all fitted"}
    return None


def texture_rule_decision(rings: Iterable) -> Optional[dict]:
    """The rule behind the texture sentences, when a ring got a Hermans f."""
    if not any(getattr(ring, "herman", None) is not None for ring in rings):
        return None
    return {
        "what": "texture rule",
        "decision": (
            "\"along the surface normal\" or \"in the sample plane\" is named only when "
            f"|f − f_iso| ≥ max({WEAK_HERMAN:g}, {TEXTURE_SIGMAS:g}σ_f)"
        ),
        "why": (
            f"and only when the I(χ) maximum lies below {NORMAL_CHI:g}° with f − f_iso > 0 (along the normal) or above "
            f"{PLANE_CHI:g}° with f − f_iso < 0 (in the plane); a ring whose maximum lies between {NORMAL_CHI:g}° and "
            f"{PLANE_CHI:g}° is called tilted at that |χ| from its maximum alone, without this margin. f is compared "
            "with f_iso, the f of a random ring over the same measured |χ| (0 only for a ring measured over all of it), "
            "and the margin is at least three statistical uncertainties. The difference is a coverage diagnostic"
        ),
    }


def horizon_decision(rings: Iterable) -> Optional[dict]:
    """Actual pixel exit-angle selection and polarisation used in the orientation reduction."""
    settings = next((ring.orientation_settings for ring in rings if ring.orientation_settings), {})
    threshold = settings.get("min_exit_deg")
    if threshold is None:
        return None
    fraction = settings.get("polarization")
    polarization = ("off" if fraction is None else f"factor {fraction:g}")
    source = settings.get("exit_angle_source", "")
    return {
        "what": "ring orientation",
        "decision": f"Exclude pixels with αf < {threshold:g}° ({source}). Polarisation: {polarization}.",
        "why": "Binned means sample the pole density. Hermans f and the pole fit use (I − b) sin χ. "
               "The systematic range varies the exit threshold, background and missing-wedge fill.",
    }


__all__ = [
    "ERROR_MODELS", "MORE_PEAKS", "flat_field_decision", "horizon_decision", "peak_cap_decision", "refraction_decision",
    "texture_rule_decision", "uncertainty_decision",
]
