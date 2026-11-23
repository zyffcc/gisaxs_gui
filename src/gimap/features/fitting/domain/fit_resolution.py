"""Resolution of a horizontal cut. All q values and Gaussian standard deviations are in nm⁻¹."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import roots_legendre

from .cut_model.resolution import table_grid, tabulated

_SPAN = 7.0
_NORM = math.sqrt(2 * math.pi) * math.erf(_SPAN / math.sqrt(2))
_RULES = tuple(roots_legendre(n) for n in (8, 16))
MISSING_DQ = "Resolution from the file needs a dQ or sigma_q column, or NXcanSAS Qdev."
INVALID_DQ = "Every fitted point needs a finite, non-negative dQ for resolution from the file."


def constant_forms(model, q, sigma, cache, geometry, component_form, *, grid=None):
    """The constant-resolution table used for both saved models and Predict."""
    if not model.components:
        return []
    if grid is None:
        grid = table_grid(q, sigma)
    fine = None if cache is None else cache.setdefault(("smearing grid", grid.tobytes()), {})
    forms = [component_form(component, grid, fine, geometry) for component in model.components]
    return list(tabulated(q, grid, np.array(forms), sigma))


def _peak_edges(component, max_q):
    """Split at narrow paracrystal orders and their half widths so quadrature can see their area."""
    if not component.structure or component.interference != "paracrystal":
        return np.empty(0)
    distance, disorder = component.value("D"), component.value("sigma_nn")
    if distance <= 0 or disorder <= 0:
        return np.empty(0)
    # Above q σ_nn = 2 the first harmonic of S already varies smoothly within one period.
    period = 2 * math.pi / distance
    count = int(min(max_q / period + 1, 1 / (math.pi * disorder)))
    centres = np.arange(1, count + 1) * period
    half_width = 2 * np.sinh(0.25 * (centres * distance * disorder)**2) / distance
    edges = np.concatenate((centres - half_width, centres, centres + half_width))
    return np.sort(np.concatenate((-edges, edges)))


def pointwise_form(component, q, dq, geometry, component_form):
    """Integrate T(|q_i + dQ_i z|) against the standard normal in z, independently at every point."""
    if dq is None:
        raise ValueError(MISSING_DQ)
    dq = np.asarray(dq, dtype=float)
    if dq.shape != q.shape or not np.all(np.isfinite(dq) & (dq >= 0)):
        raise ValueError(INVALID_DQ)
    result = np.zeros_like(q)
    sharp = dq == 0
    if sharp.any():
        result[sharp] = component_form(component, q[sharp], geometry=geometry)
    if sharp.all():
        return result
    peaks = _peak_edges(component, np.max(q + _SPAN * dq))
    indices, lows, highs = [], [], []
    for i in np.flatnonzero(~sharp):
        start, end = np.searchsorted(peaks, (q[i] - _SPAN * dq[i], q[i] + _SPAN * dq[i]))
        knots = np.unique(np.r_[[-_SPAN, -2., 0., 2., _SPAN], (peaks[start:end] - q[i]) / dq[i]])
        indices.extend([i] * (knots.size - 1))
        lows.extend(knots[:-1])
        highs.extend(knots[1:])
    owners, lo, hi = np.asarray(indices), np.asarray(lows), np.asarray(highs)
    for depth in range(16):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        estimates = []
        for nodes, weights in _RULES:
            z = mid[:, None] + half[:, None] * nodes
            positions = np.abs(q[owners, None] + dq[owners, None] * z)
            form = component_form(component, positions.ravel(), geometry=geometry).reshape(z.shape)
            estimates.append(half * np.sum(weights * np.exp(-0.5 * z**2) * form, axis=1) / _NORM)
        low, high = estimates
        converged = (np.abs(high - low) <= 1e-7 * np.abs(high) + 1e-15 * half) | (depth == 15)
        np.add.at(result, owners[converged], high[converged])
        if converged.all():
            break
        retry = ~converged
        owners = np.tile(owners[retry], 2)
        lo, hi = np.r_[lo[retry], mid[retry]], np.r_[mid[retry], hi[retry]]
    return result
