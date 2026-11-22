"""Form factors of the three particle families, averaged over their sizes by number, normalised per population.

For a population with number weights n_j and amplitudes V_j A_j(q) (A_j(0) = 1 at q_z = 0):

    P̂(q) = Σ_j n_j V_j² |A_j(q)|² / Σ_j n_j V_j²          (the incoherent average, P̂(0) = 1 in the EIA)
    C(q) = |Σ_j n_j V_j A_j(q)|² / Σ_j n_j V_j²           (the coherent term |<VA>|² of the decoupling approximation)

so a population's scale is its forward intensity, whatever its family. Families:

* ``sphere``: A = Σ_k c_k f(√(q²+Q_k²) R) e^{−iQ_k R}, f(x) = 3 (sin x − x cos x)/x³, the paths (c_k, Q_k) of the
  geometry (``geometry.Geometry.sphere_terms``); V ∝ R³.
* ``vertical_cylinder``: cylinders standing on the substrate, A = 2 J₁(qR)/(qR) in the plane and V ∝ R² at fixed
  height. A manual fit with distributed heights multiplies the radial C by ``upright_height_coherence``.
* ``cylinder``: randomly oriented cylinders of radius R and height h with independent distributions,
  A = [2 J₁(|q| R sin α)/(|q| R sin α)] · sin(|q| h cos α / 2)/(|q| h cos α / 2), V ∝ R² h, averaged over
  α on [0, π/2] with weights sin α (isotropic orientation) by composite Gauss–Legendre.

Arrays: ``q`` (N,), radii and weights (B, n) for B parameter sets at once; results (B, N).
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1

from . import envelope as env
from . import sizes
from . import dwba_sphere

CHUNK = 1_000_000
"""Elements of the largest temporary (B × nodes × q) evaluated at once."""


def sphere_amplitude(x):
    """3 (sin x − x cos x)/x³ (1 at 0); the series below 0.05 avoids the cancellation. In place on one temporary
    or two: it is most of the time of a sphere population's likelihood."""
    x = np.asarray(x, float)
    out = np.cos(x)
    out *= x
    np.subtract(np.sin(x), out, out=out)
    cube = x * x
    cube *= x
    with np.errstate(divide="ignore", invalid="ignore"):
        out /= cube
    out *= 3.0
    small = np.abs(x) < 0.05
    if np.any(small):
        x2 = x[small] ** 2
        out[small] = 1.0 - x2 / 10.0 + x2 * x2 / 280.0 - x2**3 / 15120.0
    return out


def disc_amplitude(x):
    """2 J₁(x)/x (1 at 0)."""
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x * x / 8.0 + x**4 / 192.0, 2.0 * j1(xs) / xs)


def sinc_half(x):
    """sin(x/2)/(x/2) (1 at 0)."""
    return np.sinc(np.asarray(x, float) / (2.0 * np.pi))


def upright_axial_amplitude(heights, coefficients, wave):
    """Sum of the axial paths for each height and detector point, before multiplying by the cylinder volume."""
    amplitude = 0j
    for c, q_z in zip(coefficients, wave):
        x = heights[:, None] * q_z
        amplitude = amplitude + c * sinc_half(x) * np.exp(-0.5j * x)
    return amplitude


def upright_height_coherence(law, height, spread, geometry):
    """|⟨G⟩|²/⟨|G|²⟩ by number, with G the axial amplitude including H. A horizontal row's scale absorbs ⟨|G|²⟩."""
    coefficients, wave = zip(*geometry.sphere_terms())
    q_top = max(abs(q_z) for q_z in wave)
    panels = sizes.panel_count(q_top, sizes.extent(law, height, spread, power=2))
    nodes, weights = sizes.nodes(law, height, spread, power=2, panels=panels)
    heights, weights = nodes[0], weights[0]
    axial = heights * upright_axial_amplitude(heights, coefficients, wave)[:, 0]
    intensity = weights @ (np.abs(axial) ** 2)
    return float(abs(weights @ axial) ** 2 / intensity) if intensity > 0 else 0.0


def _blocks(rows: int, columns: int, per_element: int):
    """(row slice, q slice) blocks whose temporaries hold at most ``CHUNK`` elements (``per_element`` per row and q)."""
    per_element = max(per_element, 1)
    if per_element * columns <= CHUNK:
        step = max(1, CHUNK // (per_element * columns))
        return [(slice(a, min(a + step, rows)), slice(0, columns)) for a in range(0, rows, step)]
    width = max(1, CHUNK // per_element)
    return [(slice(a, a + 1), slice(b, min(b + width, columns))) for a in range(rows) for b in range(0, columns, width)]


def _sphere_envelope(q, radii, weights, terms):
    """P̂ of spheres with every |F|² (or product of two paths) replaced by its envelope where x ≥ ``X_MIN``."""
    p_out = np.empty((radii.shape[0], q.size))
    for rows, cols in _blocks(radii.shape[0], q.size, radii.shape[1] * 4 * len(terms)):
        r, qq = radii[rows][:, :, None], q[cols]
        v2 = weights[rows] * radii[rows] ** 6
        x = [np.hypot(qq, qz)[None, None, :] * r for _c, qz in terms]
        total = 0.0
        for k, (c_k, qz_k) in enumerate(terms):
            for m in range(k, len(terms)):
                c_m, qz_m = terms[m]
                exact = sphere_amplitude(x[k]) * sphere_amplitude(x[m])
                smooth = env.sphere_square(x[k]) if m == k else env.sphere_pair(x[k], x[m])
                pair = np.where(np.minimum(x[k], x[m]) < env.X_MIN, exact, smooth)
                if m == k:
                    total = total + abs(c_k) ** 2 * pair
                else:
                    total = total + 2.0 * np.real(c_k * np.conj(c_m) * np.exp(-1j * (qz_k - qz_m) * r)) * pair
        p_out[rows, cols] = np.einsum("bn,bnq->bq", v2, total) / v2.sum(axis=1)[:, None]
    return p_out


def sphere(q, radii, weights, terms, coherent=False, envelope=False):
    """(P̂, C or None) of spheres with the paths ``terms`` = ((c_k, Q_k), …); with ``envelope``, P̂ of the envelopes
    and C = 0 (``envelope``: for q beyond the size distribution's q_cut)."""
    q = np.asarray(q, float)
    if len(terms) == 4:
        return dwba_sphere.average(q, radii, weights, terms, coherent, envelope, sphere_amplitude)
    if envelope:
        return _sphere_envelope(q, radii, weights, terms), (np.zeros((radii.shape[0], q.size)) if coherent else None)
    p_out = np.empty((radii.shape[0], q.size))
    c_out = np.empty_like(p_out) if coherent else None
    single = len(terms) == 1
    for rows, cols in _blocks(radii.shape[0], q.size, radii.shape[1] * (1 if single else 2 * len(terms))):
        r, n, qq = radii[rows][:, :, None], weights[rows], q[cols]
        v2 = n * radii[rows] ** 6
        norm = v2.sum(axis=1)
        if len(terms) == 1:  # one path: |A|² = |c|² f², the phase e^{−iQR} only in the coherent mean
            c, qz = terms[0]
            f = sphere_amplitude((qq if qz == 0.0 else np.hypot(qq, qz))[None, None, :] * r)
            p_out[rows, cols] = abs(c) ** 2 * np.einsum("bn,bnq->bq", v2, f * f) / norm[:, None]
            if coherent:
                volume = n * radii[rows] ** 3
                if qz == 0.0:
                    mean = np.einsum("bn,bnq->bq", volume, f) ** 2
                else:
                    mean = (np.einsum("bn,bnq->bq", volume * np.cos(qz * radii[rows]), f) ** 2
                            + np.einsum("bn,bnq->bq", volume * np.sin(qz * radii[rows]), f) ** 2)
                c_out[rows, cols] = abs(c) ** 2 * mean / norm[:, None]
            continue
        amplitude = 0j
        for c, qz in terms:
            amplitude = amplitude + c * sphere_amplitude(np.hypot(qq, qz)[None, None, :] * r) * np.exp(-1j * qz * r)
        p_out[rows, cols] = np.einsum("bn,bnq->bq", v2, np.abs(amplitude) ** 2) / norm[:, None]
        if coherent:
            mean = np.einsum("bn,bnq->bq", n * radii[rows] ** 3, amplitude)
            c_out[rows, cols] = np.abs(mean) ** 2 / norm[:, None]
    return p_out, c_out


def vertical_cylinder(q, radii, weights, coherent=False, envelope=False):
    """(P̂, C or None) of upright cylinders, in the plane; with ``envelope``, P̂ of the envelopes and C = 0."""
    q = np.asarray(q, float)
    p_out = np.empty((radii.shape[0], q.size))
    c_out = np.zeros_like(p_out) if coherent and envelope else (np.empty_like(p_out) if coherent else None)
    for rows, cols in _blocks(radii.shape[0], q.size, radii.shape[1]):
        x = q[cols][None, None, :] * radii[rows][:, :, None]
        amplitude = disc_amplitude(x)
        square = np.where(x < env.X_MIN, amplitude**2, env.disc_square(np.maximum(x, env.X_MIN))) if envelope else amplitude**2
        v2 = weights[rows] * radii[rows] ** 4
        norm = v2.sum(axis=1)
        p_out[rows, cols] = np.einsum("bn,bnq->bq", v2, square) / norm[:, None]
        if coherent and not envelope:
            mean = np.einsum("bn,bnq->bq", weights[rows] * radii[rows] ** 2, amplitude)
            c_out[rows, cols] = mean**2 / norm[:, None]
    return p_out, c_out


def orientation_nodes(q_max: float, size_extent) -> tuple[np.ndarray, np.ndarray]:
    """Nodes u = cos α and their weights (sum 1) of the isotropic average for the largest size ``size_extent`` (nm).

    The rule is composite Gauss–Legendre in α on [0, π/2] with weights sin α: q sin α and q cos α are smooth in α,
    whereas in u the argument q √(1 − u²) oscillates ever faster towards u = 1, which a rule in u resolves badly."""
    panels = max(3, sizes.panel_count(q_max, size_extent))
    alpha, w = sizes._composite(np.zeros(1), np.full(1, np.pi / 2), panels)
    w = w[0] * np.sin(alpha[0])
    return np.cos(alpha[0]), w / w.sum()


GRID_PER_PERIOD = 64
"""Points of the 1D grids of a random cylinder's size averages per period π/R of |F|² (cubic interpolation)."""


def _size_average(grid, nodes_, weights, amplitude, power, coherent, square_envelope, cut):
    """Σ n V² A²/Σ n V² (and the signed Σ n V A/√Σ n V² when ``coherent``) on ``grid`` (B, G), with V ∝ x^(power/2)
    and A = ``amplitude``(grid · x); beyond each row's ``cut`` the envelope ``square_envelope`` and a mean of 0."""
    mean_sq = np.empty((nodes_.shape[0], grid.size))
    mean = np.empty_like(mean_sq) if coherent else None
    for rows, cols in _blocks(nodes_.shape[0], grid.size, nodes_.shape[1]):
        x = grid[cols][None, None, :] * nodes_[rows][:, :, None]
        a = amplitude(x)
        step = env.blend(grid[cols], cut[rows])
        smooth = np.where(x >= env.X_MIN, square_envelope(np.maximum(x, env.X_MIN)), a * a)
        square = (1.0 - step[:, None, :]) * a * a + step[:, None, :] * smooth
        v2 = weights[rows] * nodes_[rows] ** power
        norm = v2.sum(axis=1)
        mean_sq[rows, cols] = np.einsum("bn,bnq->bq", v2, square) / norm[:, None]
        if coherent:
            amp = np.einsum("bn,bnq->bq", weights[rows] * nodes_[rows] ** (power / 2), a) / np.sqrt(norm)[:, None]
            mean[rows, cols] = (1.0 - step) * amp
    return mean_sq, mean


def _cubic(values, step, x):
    """Four-point Lagrange interpolation of ``values`` (B, G), sampled at 0, step, 2·step …, at ``x`` (any shape, the
    same for every row): (B, *x.shape)."""
    t = np.asarray(x, float) / step
    i = np.clip(np.floor(t).astype(int) - 1, 0, values.shape[1] - 4)
    f = t - i
    w = (-(f - 1) * (f - 2) * (f - 3) / 6, f * (f - 2) * (f - 3) / 2, -f * (f - 1) * (f - 3) / 2, f * (f - 1) * (f - 2) / 6)
    return sum(wk[None] * values[:, i + k] for k, wk in enumerate(w))


def _grid(q_end, largest, period):
    """Uniform grid from 0 past ``q_end`` with ``GRID_PER_PERIOD`` points per ``period``/``largest``."""
    step = period / (GRID_PER_PERIOD * max(largest, 1e-12))
    count = max(8, int(np.ceil(q_end / step)) + 4)
    return np.arange(count) * step, step


def _envelope_average(x, nodes_, weights, amplitude, power, square_envelope):
    """Σ n V² env(x·R)/Σ n V² at the arguments ``x`` (any shape), each node's exact square below ``X_MIN``: (B, *x.shape)."""
    v2 = weights * nodes_**power
    out = np.zeros((nodes_.shape[0],) + x.shape)
    for j in range(nodes_.shape[1]):
        arg = x[None] * nodes_[:, j].reshape((-1,) + (1,) * x.ndim)
        exact = amplitude(arg) ** 2
        value = np.where(arg >= env.X_MIN, square_envelope(np.maximum(arg, env.X_MIN)), exact)
        out += v2[:, j].reshape((-1,) + (1,) * x.ndim) * value
    return out / v2.sum(axis=1).reshape((-1,) + (1,) * x.ndim)


def _lookup(values, step, end, x, envelope_part):
    """``values`` on the grid at ``x`` ≤ ``end`` (cubic), the envelope average ``envelope_part(x)`` beyond."""
    inside = x <= end
    out = _cubic(values, step, np.where(inside, x, 0.0))
    if not inside.all():
        out = np.where(inside[None], out, envelope_part(np.where(inside, end, x)))
    return out


def cylinder(q, radii, r_weights, heights, h_weights, qz=0.0, coherent=False, cut_r=None, cut_h=None,
             envelope_radii=None, envelope_heights=None):
    """(P̂, C or None) of randomly oriented cylinders at |q| = √(q² + qz²).

    The radius average ⟨R⁴ (2J₁(xR)/(xR))²⟩/⟨R⁴⟩ depends on x = |q| sin α only, the height average on |q| cos α
    only: both are made once per row on a 1D grid of x (``GRID_PER_PERIOD`` points per period of the largest size)
    and taken at every (α, q) by cubic interpolation, then averaged over α. With a cut (``envelope``), each average
    goes over to its envelope between ``cut`` and ``BLEND`` · ``cut``; the grid then ends there, and beyond it the
    envelope average is taken directly on the nodes ``envelope_radii`` / ``envelope_heights`` ((nodes, weights))."""
    q_total = np.hypot(np.asarray(q, float), qz)
    q_max = float(q_total.max()) if q_total.size else 0.0
    rows_out = radii.shape[0]
    cut_r = np.full(rows_out, np.inf) if cut_r is None else np.asarray(cut_r, float)
    cut_h = np.full(rows_out, np.inf) if cut_h is None else np.asarray(cut_h, float)
    end_r = min(q_max, env.BLEND * float(np.max(cut_r)))
    end_h = min(q_max, env.BLEND * float(np.max(cut_h)))
    # |F|² oscillates in u = cos α with period π/(q R) (radius) and 2π/(q h) (height), up to the grids' ends
    u, wu = orientation_nodes(max(end_r, end_h), max(float(np.max(radii)), 0.5 * float(np.max(heights))))
    sin_a = np.sqrt(np.maximum(1.0 - u * u, 0.0))
    r_grid, r_step = _grid(end_r, float(np.max(radii)), np.pi)
    h_grid, h_step = _grid(end_h, float(np.max(heights)), 2.0 * np.pi)
    radial_sq, radial = _size_average(r_grid, radii, r_weights, disc_amplitude, 4, coherent, env.disc_square, cut_r)
    axial_sq, axial = _size_average(h_grid, heights, h_weights, sinc_half, 2, coherent, env.sinc_square, cut_h)
    x_r = sin_a[:, None] * q_total[None, :]
    x_h = u[:, None] * q_total[None, :]
    p_out = np.empty((rows_out, q_total.size))
    c_out = np.empty_like(p_out) if coherent else None
    for rows, cols in _blocks(rows_out, q_total.size, 8 * u.size):
        xr, xh = x_r[:, cols], x_h[:, cols]

        def radial_envelope(x, rows=rows):
            nodes_, node_weights = envelope_radii
            return _envelope_average(x, nodes_[rows], node_weights[rows], disc_amplitude, 4, env.disc_square)

        def axial_envelope(x, rows=rows):
            nodes_, node_weights = envelope_heights
            return _envelope_average(x, nodes_[rows], node_weights[rows], sinc_half, 2, env.sinc_square)

        radial_part = _lookup(radial_sq[rows], r_step, end_r, xr, radial_envelope)
        axial_part = _lookup(axial_sq[rows], h_step, end_h, xh, axial_envelope)
        p_out[rows, cols] = np.einsum("u,buq->bq", wu, radial_part * axial_part)
        if coherent:
            inside = (xr <= end_r)[None] & (xh <= end_h)[None]
            amplitude = (_cubic(radial[rows], r_step, np.minimum(xr, end_r))
                         * _cubic(axial[rows], h_step, np.minimum(xh, end_h)))
            mean = np.einsum("u,buq->bq", wu, np.where(inside, amplitude, 0.0))
            c_out[rows, cols] = mean**2
    return p_out, c_out


FAMILIES = ("sphere", "cylinder", "vertical_cylinder")
RADIUS_POWER = {"sphere": 6, "cylinder": 4, "vertical_cylinder": 4}
"""V² ∝ R^power at the other sizes fixed: what turns number weights into intensity weights."""
HEIGHT_POWER = 2

__all__ = ["FAMILIES", "HEIGHT_POWER", "RADIUS_POWER", "cylinder", "disc_amplitude", "orientation_nodes", "sinc_half",
           "sphere", "sphere_amplitude", "vertical_cylinder"]
