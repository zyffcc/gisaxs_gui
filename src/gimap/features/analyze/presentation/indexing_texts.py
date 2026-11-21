"""Sentences of the Phases window, composed in the interface language (``tr``/``trf``; numbers stay numbers)."""

from __future__ import annotations

import math
from typing import Optional

from src.gimap.app.presentation.i18n import tr, trf

MODEL_NAMES = {"gaussian": "Gaussian + isotropic", "maier_saupe": "Maier–Saupe + isotropic"}


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
    atoms = len(spec.atoms)
    if spec.operations:
        text = trf("{group}{count} operations from the CIF", group=f"{spec.space_group}, " if spec.space_group else "",
                   count=len(spec.operations))
    else:
        text = trf("lattice centring {centring}", centring=spec.centring)
    if atoms:
        return text + "; " + trf("{atoms} atom sites: |F|² from the atoms", atoms=atoms)
    return text + "; " + tr("no atoms: every allowed reflection has intensity 1")


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


def pole_result_text(result) -> str:
    fit = result.fit
    if not fit.ok:
        return trf("No fit: {reason}", reason=tr(fit.message))
    values, errors = fit.values, fit.errors
    profile = fit.profile
    lines = [
        trf("{model}: oriented fraction {fraction}, FWHM {fwhm}°, direction χ0 = {chi0}°",
            model=tr(MODEL_NAMES.get(fit.model, fit.model)),
            fraction=_pm(values.get("oriented_fraction"), errors.get("oriented_fraction"), 3),
            fwhm=_pm(values.get("fwhm_deg"), errors.get("fwhm_deg"), 2),
            chi0=_pm(values.get("chi0_deg"), errors.get("chi0_deg"), 1)),
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


__all__ = ["MODEL_NAMES", "orientation_text", "phase_cell_text", "pole_result_text", "ring_text", "symmetry_text"]
