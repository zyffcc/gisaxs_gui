"""Inverse-variance folding of a fibre ring's mean pixel intensity onto its polar angle."""

import numpy as np

SHADOW_FRACTION = 0.2
MONTE_CARLO_SAMPLES = 200


def measured_hermans(chi_deg, counts, sigma=None, seed=0):
    """Return (f, statistical σ, mean cos²χ) for measured, signed counts per polar-angle bin.

    Both counts and their errors include sin χ. Trials with nonpositive total have no ratio.
    """
    total = np.sum(counts)
    if not total > 0.:
        return None, None, None
    projection = np.cos(np.radians(chi_deg)) ** 2
    cos2 = float(np.sum(counts * projection) / total)
    stat = None
    if sigma is not None:
        trials = np.random.default_rng(seed).normal(counts, sigma, size=(MONTE_CARLO_SAMPLES, len(counts)))
        totals = trials.sum(axis=1)
        keep = totals > 0.
        if keep.sum() > 10:
            moments = (trials[keep] * projection).sum(axis=1) / totals[keep]
            stat = float(np.std(1.5 * moments - 0.5))
    return 1.5 * cos2 - 0.5, stat, cos2


def fold_ring(chi_deg, intensity, sigma, background):
    """Return centres, net density, error, measured and shadow masks for 90 one-degree bins."""
    chi = np.asarray(chi_deg, dtype=float)
    value = np.asarray(intensity, dtype=float)
    error = np.asarray(sigma, dtype=float)
    valid = np.isfinite(chi) & (abs(chi) <= 90) & np.isfinite(value) & np.isfinite(error) & (error > 0)
    index = np.clip(np.floor(abs(chi[valid])).astype(int), 0, 89)
    weights = 1.0 / error[valid] ** 2
    total = np.bincount(index, weights=weights, minlength=90)
    counts = np.bincount(index, weights=weights * (value[valid] - background), minlength=90)
    measured = total > 0
    mean = np.divide(counts, total, out=np.full(90, np.nan), where=measured)
    spread = np.divide(1., np.sqrt(total), out=np.full(90, np.nan), where=measured)
    core = (background > 0) & measured & (mean + background < SHADOW_FRACTION * background)
    shadow = measured & (np.convolve(core.astype(int), [1, 1, 1], mode="same") > 0)
    return np.arange(90) + 0.5, mean, spread, measured & ~shadow, shadow
