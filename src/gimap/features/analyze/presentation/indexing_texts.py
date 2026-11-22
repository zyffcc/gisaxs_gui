"""Sentences of the Phases window, composed in the interface language (``tr``/``trf``; numbers stay numbers)."""

from __future__ import annotations

import math
from typing import Optional

from src.gimap.app.presentation.i18n import tr, trf

from .texts import message_text

MODEL_NAMES = {"gaussian": "Gaussian + isotropic", "maier_saupe": "Maier–Saupe + isotropic"}
BOUND_QUANTITIES = {"fraction_iso": "oriented fraction", "width": "FWHM", "chi0_deg": "direction χ0"}
"""What a fitted parameter at its bound leaves undetermined, in the words of the result."""


def _pm(value: Optional[float], error: Optional[float], digits: int) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if error is None or not math.isfinite(error) or error <= 0:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {error:.{digits}f}"


def orientation_text(spec) -> str:
    h, k, l = spec.indices
    if spec.orientation == "plane":
        return trf("({h} {k} {l}) parallel to the substrate", h=h, k=k, l=l)
    if spec.orientation == "axis":
        return trf("[{h} {k} {l}] along the surface normal", h=h, k=k, l=l)
    return tr("powder (random orientation)")


def phase_cell_text(spec) -> str:
    a, b, c, alpha, beta, gamma = spec.cell
    return f"{a:.4g}, {b:.4g}, {c:.4g}; {alpha:.4g}, {beta:.4g}, {gamma:.4g}"


def symmetry_text(spec) -> str:
    """The Symmetry column of a phase: where its operations come from and what its intensities are. A phase
    without operations says so first, with ⚠, for as long as it is listed."""
    atoms = len(spec.atoms)
    group = f"{spec.space_group}, " if spec.space_group else ""
    if getattr(spec, "positions_only", False):
        return "⚠ " + trf("{group}positions only: no symmetry operations, every hkl, no intensities",
                          group=group)
    if getattr(spec, "symmetry", "") == "symbol":
        text = trf("{group}{count} operations from the space group (none in the CIF)", group=group,
                   count=len(spec.operations))
    elif spec.operations:
        text = trf("{group}{count} operations from the CIF", group=group, count=len(spec.operations))
    else:
        text = trf("lattice centring {centring}", centring=spec.centring)
    if atoms:
        return text + "; " + trf("{atoms} atom sites: |F|² from the atoms", atoms=atoms)
    return text + "; " + tr("no atoms: every allowed reflection has intensity 1")


def phase_notes(spec) -> str:
    """The warnings and the notes of a phase (how its symmetry was found or why it was not), one per line."""
    return "\n".join(message_text(text) for text in (*spec.warnings, *getattr(spec, "notes", ())))


def ring_text(analysis, *, short: bool = False) -> str:
    reduction = analysis.reduction if analysis is not None else None
    window = reduction.markers.get("chi_q_window") if reduction is not None else None
    if not window:
        return tr("No ring of I(χ): open a GIWAXS frame and drag the purple band over I(q) onto a ring.")
    low, high = sorted(float(value) for value in window)
    if short:
        return trf("Pole figure, q = {low}–{high} Å⁻¹", low=f"{low:.3f}", high=f"{high:.3f}")
    return trf("Ring of I(χ): q = {low}–{high} Å⁻¹ (drag the purple band over I(q), or choose a reflection).",
               low=f"{low:.3f}", high=f"{high:.3f}")


def _bounded(text: str, name: str, at_bound) -> str:
    return text + " " + tr("(at its bound)") if name in at_bound else text


def pole_result_text(result) -> str:
    """The result of a pole-figure fit. When a parameter ended at its bound the fit does not determine what it
    stands for: the first line says so and gives the Hermans f of the measured bins (it needs no model), and the
    value at its bound is marked where it is listed."""
    fit = result.fit
    if not fit.ok:
        return trf("No fit: {reason}", reason=tr(fit.message))
    values, errors = fit.values, fit.errors
    profile = fit.profile
    at_bound = tuple(getattr(fit, "at_bound", ()) or ())
    lines = []
    if at_bound:
        lines.append("⚠ " + trf("Not determined by this fit: {what} at its bound. Best determined: Hermans f = {f} "
                                "from the measured bins.",
                                what=", ".join(tr(BOUND_QUANTITIES.get(name, name)) for name in at_bound),
                                f=_pm(profile.herman_measured() if profile is not None else None, None, 3)))
    lines += [
        trf("{model}: oriented fraction {fraction}, FWHM {fwhm}°, direction χ0 = {chi0}°",
            model=tr(MODEL_NAMES.get(fit.model, fit.model)),
            fraction=_bounded(_pm(values.get("oriented_fraction"), errors.get("oriented_fraction"), 3), "fraction_iso",
                              at_bound),
            fwhm=_bounded(_pm(values.get("fwhm_deg"), errors.get("fwhm_deg"), 2), "width", at_bound),
            chi0=_bounded(_pm(values.get("chi0_deg"), errors.get("chi0_deg"), 1), "chi0_deg", at_bound)),
        trf("Hermans f = {f} (fitted distribution, missing wedge filled; oriented part {f_oriented}); "
            "from the measured bins {f_measured}",
            f=_pm(values.get("herman"), errors.get("herman"), 3),
            f_oriented=_pm(values.get("herman_oriented"), errors.get("herman_oriented"), 3),
            f_measured=_pm(profile.herman_measured() if profile is not None else None, None, 3)),
        trf("2D order 2<cos²χ> − 1 = {order}; χ²_red = {chi2} over {points} bins up to χ = {chi_max}°; "
            "weighted coverage {coverage}",
            order=_pm(values.get("order_2d"), errors.get("order_2d"), 3), chi2=f"{fit.chi2_red:.3g}",
            points=fit.points, chi_max=f"{result.chi_max:g}",
            coverage=f"{profile.weighted_coverage:.3f}" if profile is not None else "—"),
        trf("Background {value} counts/pixel ({source}); {lorentz}",
            value=f"{result.background:.4g}", source=tr(result.background_source),
            lorentz=tr("I(χ) already Lorentz-corrected") if result.lorentz else tr("Lorentz factor sin χ applied")),
    ]
    if fit.message:
        lines.append(fit.message)
    return "\n".join(lines)


__all__ = ["BOUND_QUANTITIES", "MODEL_NAMES", "orientation_text", "phase_cell_text", "phase_notes", "pole_result_text",
           "ring_text", "symmetry_text"]
