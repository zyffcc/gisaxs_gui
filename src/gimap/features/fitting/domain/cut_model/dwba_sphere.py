"""Four-path sphere averages. The two paths at ±Q have the same radial form factor.

For a given cut, the intensity of a single sphere is a function of R and q common to all
the size distributions. For a batch of distributions, tabulate that function in R and
apply their quadrature weights together. Scalar evaluations sum the same paths directly.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from . import envelope


def _coefficients(radii, terms):
    (one, plus), (incident, minus), (exit_, _), (both, _) = terms
    positive, negative = np.exp(-1j * plus * radii), np.exp(-1j * minus * radii)
    return one * positive + both * positive.conj(), incident * negative + exit_ * negative.conj()


def _paths(q, radii, terms, amplitude):
    """|A|² and A for radii (N,1) and q (M,)."""
    plus, minus = terms[0][1], terms[1][1]
    x, y = radii * np.hypot(q, plus), radii * np.hypot(q, minus)
    direct, reflected = _coefficients(radii, terms)
    total = direct * amplitude(x) + reflected * amplitude(y)
    return np.abs(total)**2, total


def _square_envelope(r, weights, a):
    """Sum above aR = X_MIN of w R⁶ env(aR) = 4.5 w (1 + a²R²)/a⁶."""
    first = np.searchsorted(r, envelope.X_MIN / np.maximum(a, 1e-300))
    tail0 = np.r_[np.cumsum(weights[::-1])[::-1], 0.]
    tail2 = np.r_[np.cumsum((weights*r**2)[::-1])[::-1], 0.]
    safe = np.where(first < r.size, a, 1.)
    return 4.5 * (tail0[first] + a**2 * tail2[first]) / safe**6


def _envelope_average(q, r, n, terms, amplitude):
    """The original four-path envelopes after factoring the common R⁶ from the number average.

    For x=aR, y=bR the pair envelope is 4.5 [(1+xy) cos(x-y) + (x-y) sin(x-y)]/(xy)³.
    Only the slow phase (a-b)R remains. Nodes with min(x,y) < X_MIN keep their resolved pair.
    """
    a, b = np.hypot(q, terms[0][1]), np.hypot(q, terms[1][1])
    direct, reflected = _coefficients(r, terms)
    wa, wb = n * abs(direct)**2, n * abs(reflected)**2
    cross = 2 * n * (direct * reflected.conj()).real
    ab = a*b
    phase = r[:, None] * (a-b)
    small = r[:, None] * np.minimum(a, b) < envelope.X_MIN
    cosine, sine = np.cos(phase), phase * np.sin(phase)
    cosine[small] = sine[small] = 0.
    safe = np.where(small.all(axis=0), 1., ab)
    pair = np.einsum("n,nq->q", cross, cosine+sine) + ab * np.einsum("n,nq->q", cross*r**2, cosine)
    result = 4.5 * pair / safe**3 + _square_envelope(r, wa, a) + _square_envelope(r, wb, b)
    if small.any():
        nodes, cols = np.nonzero(small)
        rr = r[nodes]
        x, y = rr*a[cols], rr*b[cols]
        fa, fb = amplitude(x), amplitude(y)
        below = (cross[nodes]*fa*fb + (x < envelope.X_MIN)*wa[nodes]*fa**2
                 + (y < envelope.X_MIN)*wb[nodes]*fb**2)
        result += np.bincount(cols, rr**6 * below, minlength=q.size)
    return result / np.dot(n, r**6)


def _rule_on_grid(radii, weights, step):
    """Put each distribution's quadrature weights on the shared radius grid by sixth-order interpolation."""
    position = radii / step
    left = np.maximum(np.floor(position).astype(int) - 2, 0)
    fraction = position - left
    indices, weights_on_grid = [], []
    for j in range(6):
        coeff = np.ones_like(radii)
        for other in range(6):
            if other != j:
                coeff *= (fraction - other) / (j - other)
        indices.append(left + j)
        weights_on_grid.append(coeff * weights)
    cols, inverse = np.unique(np.stack(indices, axis=-1), return_inverse=True)
    rows = np.repeat(np.arange(radii.shape[0]), 6 * radii.shape[1])
    rule = csr_matrix((np.stack(weights_on_grid, axis=-1).ravel(), (rows, inverse.ravel())),
                      shape=(radii.shape[0], cols.size))
    return cols * step, rule


def _resolved_grid(q, radius_max):
    """A grid for dense cuts. The intensity's lateral Fourier support ends at the diameter 2R_max."""
    if q.size < 64:
        return None
    step = 0.32 / (2 * radius_max)
    position = q / step
    left = np.floor(position).astype(int) - 3
    first, last = int(left.min()), int(left.max()) + 7
    if last - first + 1 > q.size / 2:
        return None
    return np.arange(first, last+1) * step, left - first, position - left


def _interpolate(values, left, position):
    """Eight-point local polynomial on an equally spaced grid (Neville's recurrence)."""
    nodes = np.arange(8)
    part = values[left[:, None] + nodes]
    for order in range(1, 8):
        weight = (position[:, None] - nodes[:-order]) / order
        part = (1 - weight) * part[:, :-1] + weight * part[:, 1:]
    return np.maximum(part[:, 0], 0.)


def average(q, radii, weights, terms, coherent, outer, amplitude):
    """Number-weighted sphere P and C for the four symmetric paths in Geometry.sphere_terms."""
    v2 = weights * radii**6
    norm = v2.sum(axis=1)
    form = np.empty((radii.shape[0], q.size))
    coh = np.zeros_like(form) if coherent else None
    # The envelope switches formula at X_MIN, so use the direct rule across its threshold.
    spread_out = np.all(radii[:, -1] > 1.1 * radii[:, 0])
    if radii.shape[0] >= 16 and spread_out and not (outer or coherent) and q.size:
        max_wave = max(abs(z) for _, z in terms)
        frequency = 2 * (np.hypot(q.max(), max_wave) + max_wave)
        # Six interpolation nodes per value, at least 31 grid points per shortest period.
        step = 0.2 / max(frequency, 1e-30)
        block = max(1, 100_000 // radii.shape[1])
        for begin in range(0, radii.shape[0], block):
            rows = slice(begin, begin + block)
            grid, rule = _rule_on_grid(radii[rows], v2[rows] / norm[rows, None], step)
            width = max(1, 250_000 // grid.size)
            for start in range(0, q.size, width):
                cols = slice(start, start + width)
                intensity, _ = _paths(q[cols], grid[:, None], terms, amplitude)
                form[rows, cols] = np.maximum(rule @ intensity, 0.)
        return form, coh
    # Cap temporaries for a broad distribution or a finely sampled manual cut.
    for row, (r, w) in enumerate(zip(radii, weights)):
        interpolation = _resolved_grid(q, r[-1]) if spread_out and not (coherent or outer) else None
        grid = interpolation[0] if interpolation is not None else q
        resolved = np.empty(grid.size) if interpolation is not None else form[row]
        width = max(1, 250_000 // r.size)
        for start in range(0, grid.size, width):
            cols = slice(start, start + width)
            if outer:
                form[row, cols] = _envelope_average(q[cols], r, w, terms, amplitude)
                continue
            intensity, total = _paths(grid[cols], r[:, None], terms, amplitude)
            resolved[cols] = np.einsum("n,nq->q", v2[row], intensity) / norm[row]
            if coherent:
                mean = np.einsum("n,nq->q", w * r**3, total)
                coh[row, cols] = abs(mean)**2 / norm[row]
        if interpolation is not None:
            form[row] = _interpolate(resolved, *interpolation[1:])
    return form, coh
