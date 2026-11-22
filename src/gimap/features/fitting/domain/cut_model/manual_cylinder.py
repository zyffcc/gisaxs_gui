"""Independent radius and height averages for a randomly oriented cylinder in a manual fit."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import hankel1e, j1, y1

from . import manual_sizes, shapes, sizes


def _rotated_factor(q, law, centre, spread, power, coherent):
    """Average without resolving each high-q oscillation of an unbounded log-normal or Gamma law.

    In the oscillating integral shift the positive radius axis onto r·exp(iφ), φ=min(width, 0.5).
    Both densities are analytic in this sector and decay at infinity. exp(iqr) and H₁(qr) then decay
    along the ray. The non-oscillating part stays on the real axis. The normalization is ⟨rᵖ⟩.
    """
    phi = min(spread, 0.5)
    rotation = np.exp(1j * phi)
    # Complex density times dz, relative to the quadrature's real density times dr.
    if law == "lognormal_median":
        nodes, weights = sizes.nodes(law, centre, spread, power=power, panels=16)
        r, w = nodes[0], weights[0]
        ratio = np.exp((0.5 * phi**2 - 1j * phi * np.log(r / centre)) / spread**2)
    else:
        if power == 2:
            k = 1 / spread**2
            argument = -1j * q * centre / k
            norm = centre**2 * (1 + spread**2)
            square = -2 * np.expm1(-k * np.log1p(argument)).real / (q**2 * norm)
            mean = (2 * np.exp(-k * np.log1p(argument/2)).imag / (q * np.sqrt(norm)) if coherent else None)
            return square, mean
        # A log-radius grid resolves the end at zero for a broad Gamma distribution at large q.
        lo, hi = manual_sizes.gamma_span(np.array([centre]), np.array([spread]), power)
        lo = np.maximum(lo, np.finfo(float).tiny * centre)
        u, weights = sizes._composite(np.log(lo), np.log(hi), 24)
        nodes = np.exp(u)
        r = nodes[0]
        w = weights[0] * r * manual_sizes.gamma_weights(nodes, np.array([centre]), np.array([spread]))[0]
        w /= w.sum()
        ratio = np.exp((1j * phi - (r / centre) * np.expm1(1j*phi)) / spread**2)
    wz = w * ratio
    z = r * rotation
    norm = np.sum(w * r**power)
    xz = z[:, None] * q
    if power == 2:
        # h² sinc²(qh/2) = 2(1−cos(qh))/q².
        square = 2 * (1 - np.einsum("i,ij->j", wz, np.exp(1j*xz)).real) / (q**2 * norm)
        mean = (2 * np.einsum("i,ij->j", wz, np.exp(0.5j*xz)).imag / (q * np.sqrt(norm))
                if coherent else None)
    else:
        # 2 J₁² = |H₁|² + Re(H₁²), and J₁ = Re H₁. Only H₁² and H₁ oscillate.
        x = r[:, None] * q
        real = np.einsum("i,ij->j", w * r**2, j1(x)**2 + y1(x)**2)
        wave = np.zeros(xz.shape, complex)
        significant = xz.imag < 45
        wave[significant] = hankel1e(1, xz[significant]) * np.exp(1j*xz[significant])
        oscillation = np.einsum("i,ij->j", wz * z**2, wave**2).real
        square = 2 * (real + oscillation) / (q**2 * norm)
        mean = (2 * np.einsum("i,ij->j", wz * z, wave).real / (q * np.sqrt(norm))
                if coherent else None)
    return square, mean


@lru_cache(maxsize=16)
def _factor(law, centre, spread, power, amplitude, qmax, coherent, panels):
    """Tabulate a one-dimensional average for the orientation rule. Use finer q spacing near the origin,
    where the largest particles matter, and resolve oscillations up to q times the size spread of order ten."""
    characteristic = sizes.characteristic(law, centre, spread, power)[0]
    extent = sizes.extent(law, centre, spread, power)[0]
    # Uniform spacing in asinh(q/q0): linear near q=0, logarithmic after smearing out the oscillations.
    # The narrow-distribution limit is the original 64 grid points per period.
    q0 = 10 / (characteristic * max(spread, 1e-8))
    period = np.pi if power == 4 else 2 * np.pi
    step = period / (shapes.GRID_PER_PERIOD * characteristic * q0)
    count = max(4, int(np.ceil(np.arcsinh(qmax / q0) / step)) + 2)
    grid = q0 * np.sinh(np.arange(count) * step)
    square = np.empty(count)
    mean = np.empty(count) if coherent else None
    # Low-q points need fewer size nodes. Bound each work array to shapes.CHUNK.
    counts = sizes.panel_counts(grid, extent) if panels is None else np.full(count, panels)
    rotated = ((counts >= 32) & (grid * centre >= 2) & (spread >= 0.005)
               & (law in ("lognormal_median", "schulz")) & (panels is None))
    if np.any(rotated):
        square[rotated], avg = _rotated_factor(grid[rotated], law, centre, spread, power, coherent)
        if coherent:
            mean[rotated] = avg
    for n in np.unique(counts[~rotated]):
        radii, w = sizes.nodes(law, centre, spread, power=power, panels=int(n))
        r, w = radii[0], w[0]
        v2 = w * r**power
        norm = v2.sum()
        idx = np.flatnonzero((counts == n) & ~rotated)
        block = max(1, shapes.CHUNK // r.size)
        for start in range(0, idx.size, block):
            cols = idx[start:start + block]
            a = amplitude(r[:, None] * grid[cols])
            square[cols] = np.einsum("i,ij->j", v2, a*a) / norm
            if coherent:
                mean[cols] = np.einsum("i,ij->j", w * r**(power / 2), a) / np.sqrt(norm)

    def interpolate(q):
        position = np.arcsinh(q / q0)
        part = shapes._cubic(square[None], step, position)[0]
        avg = shapes._cubic(mean[None], step, position)[0] if coherent else None
        return part, avg

    return interpolate, characteristic


def average(q, law, radius, spread, height, h_spread, qz, coherent, panels):
    q = np.hypot(q, qz)
    qmax = q.max(initial=0.0)
    p = np.empty((radius.size, q.size))
    c = np.empty_like(p) if coherent else None
    for i in range(radius.size):
        radial, rscale = _factor(law, radius[i], spread[i], 4, shapes.disc_amplitude, qmax, coherent, panels)
        axial, hscale = _factor(law, height[i], h_spread[i], 2, shapes.sinc_half, qmax, coherent, panels)
        u, weight = shapes.orientation_nodes(qmax, max(rscale, hscale / 2))
        sina = np.sqrt(1.0 - u*u)
        block = max(1, shapes.CHUNK // u.size)
        for start in range(0, q.size, block):
            cols = slice(start, start + block)
            r2, r1 = radial(sina[:, None] * q[cols])
            h2, h1 = axial(u[:, None] * q[cols])
            p[i, cols] = np.einsum("i,ij->j", weight, r2*h2)
            if coherent:
                c[i, cols] = np.einsum("i,ij->j", weight, r1*h1)**2
    return p, c
