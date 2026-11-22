"""Sensitivity of measured Hermans f to the exit-angle cut, ring baseline and low-χ wedge."""

import math
from dataclasses import replace

from .pole_figure import PoleFit, fit_pole_figure, fold_profile


def orientation_systematics(profiles, background, *, background_sigma=None, nominal_fit: PoleFit, chi_max=90.,
                            chi0_deg=0., fraction_iso=None):
    """Compare the crossed threshold and baseline choices, with and without a low-χ model fill.

    Each threshold supplies its own detector-reduced profile. The measured bins are kept in
    each filled result. Only the contiguous gap below the first measured polar angle is filled.
    The envelope includes ±1 propagated fit σ on that extrapolation, conditional on the model.
    A fit which fails to converge supplies no extrapolation. A converged poor fit is flagged.
    """
    fitted = background_sigma is not None and math.isfinite(background_sigma) and background_sigma >= 0
    step = float(background_sigma) if fitted else 0.2 * abs(float(background))
    lower, upper = [], []
    scenarios = []
    incomplete = not {0.5, 1.0, 2.0}.issubset(profiles)
    for multiplier, curve in profiles.items():
        for b in dict.fromkeys((background - step, background, background + step)):
            profile = fold_profile(*curve, b)
            profile = replace(profile, measured=profile.measured & (profile.chi <= chi_max))
            measured = profile.herman_measured()
            scenario = {"threshold_multiplier": float(multiplier), "background": float(b),
                        "measured": measured, "weighted_coverage": profile.weighted_coverage,
                        "model_fill": None, "wedge_sigma": None, "quality_reasons": []}
            scenarios.append(scenario)
            if measured is None:
                incomplete = True
                continue
            lower.append(measured)
            upper.append(measured)
            if profile.measured[0]:
                scenario.update(model_fill=measured, wedge_sigma=0.)
                continue
            if multiplier == 1. and b == background:
                fit = nominal_fit
            elif not nominal_fit.ok or not nominal_fit.converged:
                fit = fit_pole_figure(profile, nominal_fit.model, chi_max=chi_max,
                                      chi0_deg=chi0_deg, fraction_iso=fraction_iso)
            else:
                # Follow the same fitted model and fixed parameters at each perturbation.
                values = nominal_fit.values
                fit = fit_pole_figure(profile, nominal_fit.model, chi_max=chi_max,
                                      chi0_deg=values.get("chi0_deg") if "chi0_deg" in nominal_fit.fixed else None,
                                      fraction_iso=values.get("fraction_iso") if "fraction_iso" in nominal_fit.fixed else None,
                                      start_values=values or None)
            scenario["quality_reasons"] = list(fit.quality_reasons)
            if not fit.ok or not fit.converged:
                incomplete = True
                continue
            filled = fit.values["herman_wedge_fill"]
            sigma = fit.errors["herman_wedge_fill"]
            if not math.isfinite(filled):
                incomplete = True
                continue
            scenario["model_fill"] = filled
            if math.isfinite(sigma):
                scenario["wedge_sigma"] = sigma
            else:
                incomplete = True
                sigma = 0.
            lower.append(filled - sigma)
            upper.append(filled + sigma)
    return {
        "range": [min(lower), max(upper)] if lower else None,
        "background_step": step, "background_rule": "fitted sigma" if fitted else "20% of background",
        "scenarios": scenarios, "incomplete": incomplete,
    }
