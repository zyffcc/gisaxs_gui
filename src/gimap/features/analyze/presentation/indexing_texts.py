"""Sentences of the Phases window, composed in the interface language (``tr``/``trf``; numbers stay numbers)."""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.numbers import format_number
from src.gimap.shared.orientation_notes import coverage_warning, measured_f_text, orientation_notes

from .texts import message_text

MODEL_NAMES = {"gaussian": "Gaussian + isotropic", "gaussian_mosaic": "Gaussian mosaic + isotropic",
               "maier_saupe": "Maier–Saupe + isotropic"}


def percent_text(value, error):
    return format_number(value * 100 if value is not None else None,
                         error * 100 if error is not None else None)


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


def _fit_lines(result):
    fit = result.fit
    values, errors = fit.values, fit.errors
    return [
        trf("{model}: oriented fraction {fraction}, FWHM {fwhm}°, direction χ0 = {chi0}°",
            model=tr(MODEL_NAMES.get(fit.model, fit.model)),
            fraction=_bounded(format_number(values.get("oriented_fraction"), errors.get("oriented_fraction")),
                              "fraction_iso", fit.at_bound),
            fwhm=_bounded(format_number(values.get("fwhm_deg"), errors.get("fwhm_deg")), "width", fit.at_bound),
            chi0=_bounded(format_number(values.get("chi0_deg"), errors.get("chi0_deg")), "chi0_deg", fit.at_bound)),
        trf("Fitted Hermans f = {f} (missing wedge filled, oriented part {f_oriented})",
            f=format_number(values.get("herman"), errors.get("herman")),
            f_oriented=format_number(values.get("herman_oriented"), errors.get("herman_oriented"))),
        trf("2D order 2<cos²χ> − 1 = {order}; χ²_red = {chi2} over {points} bins up to χ = {chi_max}°; "
            "weighted coverage {coverage}",
            order=format_number(values.get("order_2d"), errors.get("order_2d")), chi2=format_number(fit.chi2_red),
            points=fit.points, chi_max=f"{result.chi_max:g}",
            coverage=f"{fit.weighted_coverage:.3f}" if fit.weighted_coverage is not None else "—"),
    ]


def pole_result_text(result) -> str:
    """Show fit quality and the independently measured moment on the requested angular range."""
    fit = result.fit
    lines = []
    warning = coverage_warning(result.systematics, trf)
    if warning:
        lines.append(warning)
    if not fit.ok:
        lines.append(trf("Pole fit not determined ({reasons}).", reasons=tr(fit.message)))
    elif fit.quality_reasons:
        lines.append("⚠ " + trf("Pole fit not determined ({reasons}).",
                                reasons=", ".join(tr(reason) for reason in fit.quality_reasons)))
    measured, error, _ = fit.measured_moment
    if measured is not None:
        text = trf("Measured f = {value}", value=measured_f_text(measured, error, stat_label=tr("stat")))
        limits = result.systematics.get("range")
        if limits:
            text += trf(" [{low}, {high}] (systematic)", low=f"{limits[0]:#.3g}", high=f"{limits[1]:#.3g}")
        text += "   " + trf("Isotropic reference f_iso = {value}", value=f"{fit.herman_isotropic:#.3g}")
        lines.append(text)
    if fit.ok:
        lines.extend(_fit_lines(result))
    lines.append(trf("Background {value} counts/pixel ({source}). 1 − φ is conditional on the background b.",
                     value=format_number(result.background, result.background_sigma), source=tr(result.background_source)))
    if result.min_exit_deg is not None:
        lines.append(trf("Excluded αf < {angle}°. Polarisation {factor}.",
                         angle=f"{result.min_exit_deg:g}", factor=tr("off") if result.polarization is None
                         else f"{result.polarization:g}"))
    if result.systematics.get("incomplete"):
        lines.append(tr("Some systematic estimates are unavailable."))
    lines.extend(orientation_notes(result.systematics, result.systematics, trf))
    return "\n".join(lines)


__all__ = ["MODEL_NAMES", "orientation_text", "percent_text", "phase_cell_text", "phase_notes", "pole_result_text",
           "ring_text", "symmetry_text"]
