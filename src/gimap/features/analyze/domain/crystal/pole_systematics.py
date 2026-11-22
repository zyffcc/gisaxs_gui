"""Sensitivity of measured Hermans f to the exit-angle cut, weak horizon bins, baseline and low-χ wedge."""

import math
from dataclasses import replace

import numpy as np

from .....shared.ring_profile import HORIZON_FRACTION
from .pole_figure import PoleFit, fit_pole_figure, fold_profile


def _wedge_estimate(profile, nominal_fit, reuse_nominal, chi_max):
    """Only a determined nominal fit and a determined scenario fit can supply a wedge extrapolation."""
    if nominal_fit.quality_reasons:
        return None, None, list(nominal_fit.quality_reasons)
    if profile.measured[0]:
        return profile.herman_measured(), 0., []
    if reuse_nominal:
        fit = nominal_fit
    else:
        values = nominal_fit.values
        fit = fit_pole_figure(profile, nominal_fit.model, chi_max=chi_max,
                              chi0_deg=values.get("chi0_deg") if "chi0_deg" in nominal_fit.fixed else None,
                              fraction_iso=values.get("fraction_iso") if "fraction_iso" in nominal_fit.fixed else None,
                              start_values=values)
    if fit.quality_reasons:
        return None, None, list(fit.quality_reasons)
    filled, sigma = fit.values["herman_wedge_fill"], fit.errors["herman_wedge_fill"]
    if not (math.isfinite(filled) and math.isfinite(sigma)):
        return None, None, ["wedge extrapolation or its uncertainty is undefined"]
    return filled, sigma, []


def orientation_systematics(profiles, background, *, background_sigma=None, nominal_fit: PoleFit, chi_max=90.,
                            horizon_angles=None):
    """Cross exit cut, baseline and near-horizon threshold, keeping measured bins in every wedge estimate.

    Folding uses an available measured mirror partner once. A gap on both sides stays empty unless a
    determined model supplies its extrapolation. Its uncertainty is propagated through the filled moment.
    """
    fitted = background_sigma is not None and math.isfinite(background_sigma) and background_sigma >= 0
    step = float(background_sigma) if fitted else 0.2 * abs(float(background))
    scenarios, limits = [], []
    incomplete = not {0.5, 1.0, 2.0}.issubset(profiles)
    allow_model = not nominal_fit.quality_reasons
    nominal_range = None
    for multiplier, curve in profiles.items():
        horizon = (horizon_angles or {}).get(multiplier)
        fractions = (HORIZON_FRACTION, 0., 2 * HORIZON_FRACTION) if horizon is not None else (0.,)
        for b in dict.fromkeys((background - step, background, background + step)):
            # Distinct threshold fractions often retain exactly the same bins.
            estimates = {}
            for fraction in fractions:
                profile = fold_profile(*curve, b, horizon_chi_deg=horizon, horizon_fraction=fraction)
                profile = replace(profile, measured=profile.measured & (profile.chi <= chi_max))
                measured = profile.herman_measured()
                scenario = {"threshold_multiplier": float(multiplier), "background": float(b),
                            "horizon_fraction": fraction, "horizon_excluded_deg": profile.horizon_excluded_deg,
                            "measured": measured, "weighted_coverage": profile.weighted_coverage,
                            "model_fill": None, "wedge_sigma": None, "quality_reasons": []}
                scenarios.append(scenario)
                nominal = multiplier == 1. and b == background
                if nominal and fraction == HORIZON_FRACTION:
                    nominal_range = profile.horizon_excluded_deg
                if measured is None:
                    incomplete = True
                    continue
                limits.append(measured)
                key = profile.measured.tobytes()
                if key not in estimates:
                    reuse = nominal and np.array_equal(profile.measured, nominal_fit.measured_bins)
                    estimates[key] = _wedge_estimate(profile, nominal_fit, reuse, chi_max)
                filled, sigma, reasons = estimates[key]
                scenario.update(model_fill=filled, wedge_sigma=sigma, quality_reasons=reasons)
                if filled is not None:
                    limits.extend((filled - sigma, filled + sigma))
                elif allow_model:
                    incomplete = True
    return {
        "range": [min(limits), max(limits)] if limits else None,
        "background_step": step, "background_rule": "fitted sigma" if fitted else "20% of background",
        "horizon_fraction": HORIZON_FRACTION if (horizon_angles or {}).get(1.) is not None else 0.,
        "horizon_excluded_deg": nominal_range, "model_fill_allowed": allow_model,
        "scenarios": scenarios, "incomplete": incomplete,
    }
