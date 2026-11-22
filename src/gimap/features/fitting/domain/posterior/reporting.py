"""Diagnostics computed from the finished importance samples. All random draws are local to reporting."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

BOOTSTRAP_DRAWS = 200
BOOTSTRAP_SEED = 7529
VALIDATION_DENSITY = 397 / 2
"""397 fitted q points from 0.02 to 2 nm⁻¹ in tools/benchmarks/recovery_cases.py."""

QZR_COVERAGE = 1.2
"""Physics and Methods §9.13, sphere coverage table (posterior_dwba_pairs.json): 18–33 % at 1.2–1.7
on DWBA cuts fitted at levels 2 and 1 respectively. Applied conservatively to random-cylinder radii too."""
QZR_FAILURE = 2.0
"""Approximate onset of level-2 failure in the same DWBA comparison (bins 1.7–2.3 and 2.3–3.5)."""


def grazing_flags(model, qz, level, reflected_beam=None, known_heights=None):
    """The measured q₊ is independent of the level used in the fit. Heights must be known."""
    if qz is None or not np.isfinite(qz) or qz <= 0:
        return []
    flags = []
    for n, component in enumerate(sorted(model.components, key=lambda c: c.value("R")), 1):
        radius = component.value("R")
        product = qz * radius
        if product <= QZR_COVERAGE:
            continue
        label = f"R_{n}"
        if component.family in ("sphere", "cylinder"):
            if level in (1, 2) and reflected_beam:
                flags.append(f"qzR_cov:{label}")
            if level == 2 and product > QZR_FAILURE:
                flags.append(f"qzR_fail:{label}")
        if component.family in ("cylinder", "vertical_cylinder"):
            height = component.value("h") if "h" in component.keys() else (known_heights or {}).get(str(n))
            if height is not None and 0.8 <= height / (2 * radius) <= 1.25:
                flags.append(f"cyl_sphere:{label}")
    return flags


def density_record(q):
    """Point density of the actual likelihood window, in points per decade of |q|."""
    decades = np.log10(np.max(q) / np.min(q))
    density = float(len(q) / decades) if decades > 0 else None
    ratio = density / VALIDATION_DENSITY if density is not None else None
    return dict(point_count=len(q), points_per_decade=density, validation_points_per_decade=VALIDATION_DENSITY,
                density_notice=ratio is not None and (ratio > 1.5 or ratio < 1 / 1.5))


def bootstrap_log_evidence(log_weights, rng):
    """Bootstrap log(mean(w)) by resampling N final importance weights with replacement."""
    log_weights = np.asarray(log_weights, float)
    offsets = log_weights - np.max(log_weights)
    indices = rng.integers(len(offsets), size=(BOOTSTRAP_DRAWS, len(offsets)))
    return np.max(log_weights) + logsumexp(offsets[indices], axis=1) - np.log(len(offsets))


def comparison_diagnostics(entries, by_bic, composition_weights):
    """Propagate independent mode bootstraps through the same sums and normalization as the comparison."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    records = []
    for result, _mode, _model, _refined, local in entries:
        source = local if local is not None else result
        z = bootstrap_log_evidence(source.log_weights, rng)
        draws.append(z)
        valid = np.all(np.isfinite(z))
        key = "composition_bic_weight" if by_bic else "composition_probability"
        records.append({"composition_count": len(composition_weights), "compared_by": "bic" if by_bic else "evidence", key: composition_weights[result.index],
                        "log_evidence": float(source.log_evidence),
                        "log_evidence_source": "local" if local is not None else "network",
                        "log_evidence_mc_error": float(np.std(z, ddof=1)) if valid else None,
                        "ess_source": "local importance samples" if local is not None else "network proposals"})
    if not by_bic:
        samples = np.array(draws)
        totals = logsumexp(samples, axis=0)
        for composition in composition_weights:
            indices = [i for i, entry in enumerate(entries) if entry[0].index == composition]
            composition_draws = logsumexp(samples[indices], axis=0)
            limits = np.quantile(np.exp(composition_draws - totals), [0.16, 0.84]).tolist()
            for i in indices:
                records[i].update(composition_modes=len(indices),
                                  composition_log_evidence=float(logsumexp([entries[k][4].log_evidence for k in indices])),
                                  composition_log_evidence_mc_error=float(np.std(composition_draws, ddof=1)),
                                  composition_probability_mc68=limits)
    return records


def reflection_quantiles(local, phase, ess_min):
    """Marginal |r| and arg(r) in radians, on the branch centred on the refined phase."""
    from . import encoding as E

    g = E.GLOBAL.start
    re, im = local.samples[:, g + 4], local.samples[:, g + 5]
    angle = phase + (np.arctan2(im, re) - phase + np.pi) % (2 * np.pi) - np.pi
    weights = local.tempered(ess_min)
    out = {}
    for key, values in (("refl_amp", np.hypot(re, im)), ("refl_phase", angle)):
        order = np.argsort(values)
        w = weights[order]
        quantiles = np.interp([.5, .16, .84, .025, .975], (np.cumsum(w) - .5 * w) / w.sum(), values[order])
        out[key] = dict(zip(("median", "low68", "high68", "low95", "high95"), map(float, quantiles)))
    return out


def check_starts(local, ess_min):
    """A small deterministic, systematically resampled copy for the optional MCMC check."""
    if local is None:
        return []
    weights = local.tempered(ess_min)
    count = min(len(weights), 128)
    indices = np.searchsorted(np.cumsum(weights), (np.arange(count) + 0.5) / count)
    return np.nan_to_num(local.samples[indices]).tolist()
