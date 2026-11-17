"""Form factors of the three particle families, averaged over their sizes by number, normalised per population.

For a population with number weights n_j and amplitudes V_j A_j(q) (A_j(0) = 1 at q_z = 0):

    P̂(q) = Σ_j n_j V_j² |A_j(q)|² / Σ_j n_j V_j²          (the incoherent average, P̂(0) = 1 in the EIA)
    C(q) = |Σ_j n_j V_j A_j(q)|² / Σ_j n_j V_j²           (the coherent term |<VA>|² of the decoupling approximation)

so a population's scale is its forward intensity, whatever its family. Families:

* ``sphere``: A = Σ_k c_k f(√(q²+Q_k²) R) e^{−iQ_k R}, f(x) = 3 (sin x − x cos x)/x³, the paths (c_k, Q_k) of the
  geometry (``geometry.Geometry.sphere_terms``); V ∝ R³.
* ``vertical_cylinder``: cylinders standing on the substrate, A = 2 J₁(qR)/(qR) in the plane; V ∝ R² at any fixed
  height (the height factor is common to every radius and every path, so it drops out of P̂ and C).
* ``cylinder``: randomly oriented cylinders of radius R and height h with independent distributions,
  A = [2 J₁(|q| R sin α)/(|q| R sin α)] · sin(|q| h cos α / 2)/(|q| h cos α / 2), V ∝ R² h, averaged over
  u = cos α uniform on [0, 1] (isotropic orientation) by composite Gauss–Legendre.

Arrays: ``q`` (N,), radii and weights (B, n) for B parameter sets at once; results (B, N).
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1

from . import sizes

CHUNK = 4_000_000
"""Elements of the largest temporary (B × nodes × q) evaluated at once."""


def sphere_amplitude(x):
    """3 (sin x − x cos x)/x³ (1 at 0); the series below 0.05 avoids the cancellation."""
    x = np.asarray(x, float)
    small = np.abs(x) < 0.05
    xs = np.where(small, 1.0, x)
    out = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / xs**3
    x2 = x * x
    return np.where(small, 1.0 - x2 / 10.0 + x2 * x2 / 280.0 - x2**3 / 15120.0, out)


def disc_amplitude(x):
    """2 J₁(x)/x (1 at 0)."""
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x * x / 8.0 + x**4 / 192.0, 2.0 * j1(xs) / xs)


def sinc_half(x):
    """sin(x/2)/(x/2) (1 at 0)."""
    return np.sinc(np.asarray(x, float) / (2.0 * np.pi))


def _blocks(rows: int, columns: int, per_element: int):
    """(row slice, q slice) blocks whose temporaries hold at most ``CHUNK`` elements (``per_element`` per row and q)."""
    per_element = max(per_element, 1)
    if per_element * columns <= CHUNK:
        step = max(1, CHUNK // (per_element * columns))
        return [(slice(a, min(a + step, rows)), slice(0, columns)) for a in range(0, rows, step)]
    width = max(1, CHUNK // per_element)
    return [(slice(a, a + 1), slice(b, min(b + width, columns))) for a in range(rows) for b in range(0, columns, width)]


def sphere(q, radii, weights, terms, coherent=False):
    """(P̂, C or None) of spheres with the paths ``terms`` = ((c_k, Q_k), …)."""
    q = np.asarray(q, float)
    p_out = np.empty((radii.shape[0], q.size))
    c_out = np.empty_like(p_out) if coherent else None
    single = len(terms) == 1 and terms[0][1] == 0.0
    for rows, cols in _blocks(radii.shape[0], q.size, radii.shape[1] * (1 if single else 2 * len(terms))):
        r, n, qq = radii[rows][:, :, None], weights[rows], q[cols]
        v2 = n * radii[rows] ** 6
        norm = v2.sum(axis=1)
        if single:
            amplitude = sphere_amplitude(qq[None, None, :] * r)
        else:
            amplitude = 0j
            for c, qz in terms:
                amplitude = amplitude + c * sphere_amplitude(np.hypot(qq, qz)[None, None, :] * r) * np.exp(-1j * qz * r)
        p_out[rows, cols] = np.einsum("bn,bnq->bq", v2, np.abs(amplitude) ** 2) / norm[:, None]
        if coherent:
            mean = np.einsum("bn,bnq->bq", n * radii[rows] ** 3, amplitude)
            c_out[rows, cols] = np.abs(mean) ** 2 / norm[:, None]
    return p_out, c_out


def vertical_cylinder(q, radii, weights, coherent=False):
    """(P̂, C or None) of upright cylinders, in the plane."""
    q = np.asarray(q, float)
    p_out = np.empty((radii.shape[0], q.size))
    c_out = np.empty_like(p_out) if coherent else None
    for rows, cols in _blocks(radii.shape[0], q.size, radii.shape[1]):
        amplitude = disc_amplitude(q[cols][None, None, :] * radii[rows][:, :, None])
        v2 = weights[rows] * radii[rows] ** 4
        norm = v2.sum(axis=1)
        p_out[rows, cols] = np.einsum("bn,bnq->bq", v2, amplitude**2) / norm[:, None]
        if coherent:
            mean = np.einsum("bn,bnq->bq", weights[rows] * radii[rows] ** 2, amplitude)
            c_out[rows, cols] = mean**2 / norm[:, None]
    return p_out, c_out


def orientation_nodes(q_max: float, size_extent) -> tuple[np.ndarray, np.ndarray]:
    """Nodes u = cos α on [0, 1] and their weights (sum 1) for the largest size ``size_extent`` (nm)."""
    panels = max(3, sizes.panel_count(q_max, size_extent))
    u, w = sizes._composite(np.zeros(1), np.ones(1), panels)
    return u[0], w[0] / w[0].sum()


GRID_PER_PERIOD = 64
"""Points of the 1D grids of a random cylinder's size averages per period π/R of |F|² (cubic interpolation)."""


def _size_average(grid, nodes_, weights, amplitude, power, coherent):
    """Σ n V² A²/Σ n V² (and the signed Σ n V A/√Σ n V² when ``coherent``) on ``grid`` (B, G), with V ∝ x^(power/2)
    and A = ``amplitude``(grid · x)."""
    mean_sq = np.empty((nodes_.shape[0], grid.size))
    mean = np.empty_like(mean_sq) if coherent else None
    for rows, cols in _blocks(nodes_.shape[0], grid.size, nodes_.shape[1]):
        a = amplitude(grid[cols][None, None, :] * nodes_[rows][:, :, None])
        v2 = weights[rows] * nodes_[rows] ** power
        norm = v2.sum(axis=1)
        mean_sq[rows, cols] = np.einsum("bn,bnq->bq", v2, a * a) / norm[:, None]
        if coherent:
            mean[rows, cols] = np.einsum("bn,bnq->bq", weights[rows] * nodes_[rows] ** (power / 2), a) / np.sqrt(norm)[:, None]
    return mean_sq, mean


def _cubic(values, step, x):
    """Four-point Lagrange interpolation of ``values`` (B, G), sampled at 0, step, 2·step …, at ``x`` (any shape, the
    same for every row): (B, *x.shape)."""
    t = np.asarray(x, float) / step
    i = np.clip(np.floor(t).astype(int) - 1, 0, values.shape[1] - 4)
    f = t - i
    w = (-(f - 1) * (f - 2) * (f - 3) / 6, f * (f - 2) * (f - 3) / 2, -f * (f - 1) * (f - 3) / 2, f * (f - 1) * (f - 2) / 6)
    return sum(wk[None] * values[:, i + k] for k, wk in enumerate(w))


def _grid(q_max, largest, period):
    """Uniform grid from 0 past ``q_max`` with ``GRID_PER_PERIOD`` points per ``period``/``largest``."""
    step = period / (GRID_PER_PERIOD * max(largest, 1e-12))
    count = max(8, int(np.ceil(q_max / step)) + 4)
    return np.arange(count) * step, step


def cylinder(q, radii, r_weights, heights, h_weights, qz=0.0, coherent=False):
    """(P̂, C or None) of randomly oriented cylinders at |q| = √(q² + qz²).

    The radius average ⟨R⁴ (2J₁(xR)/(xR))²⟩/⟨R⁴⟩ depends on x = |q| sin α only, the height average on |q| cos α
    only: both are made once per row on a 1D grid of x (``GRID_PER_PERIOD`` points per period of the largest size)
    and taken at every (α, q) by cubic interpolation, then averaged over α."""
    q_total = np.hypot(np.asarray(q, float), qz)
    q_max = float(q_total.max()) if q_total.size else 0.0
    # |F|² oscillates in u = cos α with period π/(q R) (radius) and 2π/(q h) (height)
    u, wu = orientation_nodes(q_max, max(float(np.max(radii)), 0.5 * float(np.max(heights))))
    sin_a = np.sqrt(np.maximum(1.0 - u * u, 0.0))
    r_grid, r_step = _grid(q_max, float(np.max(radii)), np.pi)
    h_grid, h_step = _grid(q_max, float(np.max(heights)), 2.0 * np.pi)
    radial_sq, radial = _size_average(r_grid, radii, r_weights, disc_amplitude, 4, coherent)
    axial_sq, axial = _size_average(h_grid, heights, h_weights, sinc_half, 2, coherent)
    x_r = sin_a[:, None] * q_total[None, :]
    x_h = u[:, None] * q_total[None, :]
    rows_out = radii.shape[0]
    p_out = np.empty((rows_out, q_total.size))
    c_out = np.empty_like(p_out) if coherent else None
    for rows, cols in _blocks(rows_out, q_total.size, 4 * u.size):
        xr, xh = x_r[:, cols], x_h[:, cols]
        p_out[rows, cols] = np.einsum("u,buq->bq", wu, _cubic(radial_sq[rows], r_step, xr)
                                      * _cubic(axial_sq[rows], h_step, xh))
        if coherent:
            mean = np.einsum("u,buq->bq", wu, _cubic(radial[rows], r_step, xr) * _cubic(axial[rows], h_step, xh))
            c_out[rows, cols] = mean**2
    return p_out, c_out


FAMILIES = ("sphere", "cylinder", "vertical_cylinder")
RADIUS_POWER = {"sphere": 6, "cylinder": 4, "vertical_cylinder": 4}
"""V² ∝ R^power at the other sizes fixed: what turns number weights into intensity weights."""
HEIGHT_POWER = 2

__all__ = ["FAMILIES", "HEIGHT_POWER", "RADIUS_POWER", "cylinder", "disc_amplitude", "orientation_nodes", "sinc_half",
           "sphere", "sphere_amplitude", "vertical_cylinder"]
