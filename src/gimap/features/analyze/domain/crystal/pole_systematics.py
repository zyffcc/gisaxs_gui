"""Sensitivity of Hermans f to the baseline, exit cut and unknown pole density at either edge."""

import math
from dataclasses import replace

import numpy as np

from .....shared.ring_profile import HORIZON_FRACTION, measured_hermans
from .pole_figure import PoleFit, fit_pole_figure, fold_profile

BAND_DEG = 10  # 1° folded bins
ATTENUATION_FRACTION = 0.75
EMPTY = "unmeasured bins empty"


def _band_mean(density, measured, start, stop):
    """Nonnegative density for the continuation, using only measured bins in the stated band."""
    values = density[start:stop][measured[start:stop]]
    return max(0., float(values.mean())) if values.size else None


def _edge_scenarios(profile, measured_f):
    """Fill only contiguous edge gaps. Every continuation is a density per solid angle."""
    measured, chi = profile.measured, profile.chi
    sine = np.sin(np.radians(chi))
    density = profile.value / sine
    first, last = np.flatnonzero(measured)[[0, -1]]
    first, stop = int(first), int(last) + 1

    # A weak interior bin or an interior detector gap cannot start an attenuated run.
    # Each candidate run is compared with one fixed band immediately before its start.
    edge = stop
    for start in range(max(first + 1, stop - BAND_DEG), stop):
        if not measured[start:stop].all():
            continue
        level = _band_mean(density, measured, max(0, start - BAND_DEG), start)
        if level and np.all(density[start:stop] < ATTENUATION_FRACTION * level):
            edge = start
            break
    high_band = (max(0, edge - BAND_DEG), edge)
    low_band = (first, min(len(chi), first + BAND_DEG))
    high_level = _band_mean(density, measured, *high_band)
    low_level = _band_mean(density, measured, *low_band)
    counts = np.where(measured, profile.value, 0.)
    factors = {EMPTY: measured_f}

    def continued(start, end, level):
        trial = counts.copy()
        trial[start:end] = level * sine[start:end]
        return measured_hermans(chi, trial)[0]

    if stop < len(chi):
        factors["horizon gap continued"] = continued(stop, len(chi), high_level)
    if edge < stop:
        factors["attenuated bins removed, horizon gap empty"] = continued(edge, len(chi), 0.)
        factors["attenuated bins and horizon gap continued"] = continued(edge, len(chi), high_level)
    if first:
        factors["low-angle wedge continued"] = continued(0, first, low_level)
    return {"coverage_factors": factors, "attenuated_deg": (edge, stop) if edge < stop else None,
            "horizon_gap_deg": (stop, len(chi)) if stop < len(chi) else None,
            "low_angle_gap_deg": (0, first) if first else None,
            "horizon_band_deg": high_band, "horizon_band_density": high_level,
            "low_angle_band_deg": low_band, "low_angle_band_density": low_level}


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
    """Cross each measured selection with the edge-density assumptions and any determined model fill."""
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
                            "coverage_factors": {}, "model_fill": None, "wedge_sigma": None, "quality_reasons": []}
                scenarios.append(scenario)
                nominal = multiplier == 1. and b == background
                if nominal and fraction == HORIZON_FRACTION:
                    nominal_range = profile.horizon_excluded_deg
                if measured is None:
                    incomplete = True
                    continue
                scenario.update(_edge_scenarios(profile, measured))
                key = profile.measured.tobytes()
                if key not in estimates:
                    reuse = nominal and np.array_equal(profile.measured, nominal_fit.measured_bins)
                    estimates[key] = _wedge_estimate(profile, nominal_fit, reuse, chi_max)
                filled, sigma, reasons = estimates[key]
                scenario.update(model_fill=filled, wedge_sigma=sigma, quality_reasons=reasons)
                factors = dict(scenario["coverage_factors"])
                if filled is not None:
                    factors.update({"fitted low-angle wedge − 1σ": filled - sigma,
                                    "fitted low-angle wedge + 1σ": filled + sigma})
                elif allow_model:
                    incomplete = True
                for name, factor in factors.items():
                    if factor is None:
                        incomplete = True
                        continue
                    limits.append({"scenario": name, "value": factor, "threshold_multiplier": float(multiplier),
                                   "background": float(b), "horizon_fraction": fraction})
    # When values tie exactly, report the nominal settings if they set the same limit.
    nominal_fraction = HORIZON_FRACTION if (horizon_angles or {}).get(1.) is not None else 0.
    limits.sort(key=lambda item: (item["threshold_multiplier"] != 1., item["background"] != background,
                                  item["horizon_fraction"] != nominal_fraction))
    ends = {}
    if limits:
        ends["min"] = min(limits, key=lambda item: item["value"])
        ends["max"] = max(limits, key=lambda item: item["value"])
    return {
        "range": [ends["min"]["value"], ends["max"]["value"]] if limits else None, "range_sources": ends,
        "background_step": step, "background_rule": "fitted sigma" if fitted else "20% of background",
        "horizon_fraction": nominal_fraction, "continuation_band_deg": BAND_DEG,
        "attenuation_fraction": ATTENUATION_FRACTION,
        "horizon_excluded_deg": nominal_range, "model_fill_allowed": allow_model,
        "scenarios": scenarios, "incomplete": incomplete,
    }
