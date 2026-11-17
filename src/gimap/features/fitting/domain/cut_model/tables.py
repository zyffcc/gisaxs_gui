"""Fast size averages of number-weighted distributions from small tables per family (the default path).

Every form factor used here splits exactly into a smooth part and a smooth complex amplitude times an oscillation:

    f(x)² = env(x) + Re[ G(x) e^{iωx} ]          f(x) = Re[ A(x) e^{iω_a x} ]

* sphere f = 3 j₁(x)/x:  env = 9(1+x²)/(2x⁶), G = (9/x⁶)((x²−1)/2 + i x), ω = 2;  A = −3(x + i)/x³, ω_a = 1;
* disc f = 2J₁(x)/x:     env = 2|H₁|²/x², G = (2/x²) H₁² e^{−2ix}, ω = 2;  A = (2/x) H₁ e^{−ix}, ω_a = 1
  (H₁ = J₁ + iY₁, the Hankel function);
* height f = sin(x/2)/(x/2):  env = 2/x², G = −2/x², ω = 1;  A = −2i/x, ω_a = 1/2.

Averaged over sizes x = x̄ t (t = r/mean, number weights n(t), V² ∝ t^p):

    P̂(x̄) = E(x̄) + Re[ e^{iωx̄} B(x̄) ],   E = <n t^p env(x̄t)>/<n t^p>,   B = <n t^p G(x̄t) e^{iωx̄(t−1)}>/<n t^p>

E/env(x̄) is smooth in ln x̄ and ln s (s the relative width); B/G(x̄), once the oscillation e^{iωx̄} is taken out, is
smooth in u = x̄ σ_d (σ_d the relative damping scale of ``envelope``) and ln s, and decays like e^{−(ωu)²/2}. They are
tabulated (``build``; prebuilt in ``data/`` by ``tools/ml1d/build_tables.py``, else built on first use) and
interpolated with four-point Lagrange in each variable; the width grid has a node at ``KINK`` = 0.25, where a Gaussian
by number starts to be cut at R = 0, and no stencil crosses it. B is weighted by 1 − (the smoothstep of
``envelope.blend``) and is 0 beyond u = 2·DAMPING: the model's definition of the average (resolved up to q_cut, its
envelope beyond), the same as the direct quadrature of ``shapes``. The coherent amplitude of the decoupling
approximation is tabulated the same way. Below x̄ = ``X_SMALL`` the averages are taken directly on a few nodes.
Widths below ``S_TABLE`` or above ``S_MAX``, sizes by intensity (whose envelope diverges at R → 0) and paths that beat
(two-path and DWBA spheres) keep the direct quadrature of ``shapes`` (``model.size_averages`` decides).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import j1, y1

from . import envelope, shapes, sizes

S_MIN = 0.005
S_MAX = {"number_gauss": 1.0, "lognormal": 1.0}
S_TABLE = 0.02
"""Narrower distributions are averaged directly (few nodes; their deep minima ask for more than interpolation)."""
KINK = 0.25
X_SMALL = 2.0
X_BIG = 1e5
U_MAX = envelope.BLEND * envelope.DAMPING
S_POINTS, U_STEP, LNX_STEP = 60, 0.04, 0.04
"""Widths per segment (two segments), steps in u and in ln x̄."""
BUILD_PANELS = (64, 4096)
"""Least and most panels a table row is built on: enough for the phase ω u (t − 1)/σ_d across the distribution at
u = U_MAX."""
ENVELOPE_PANELS = 16
"""Panels for E, whose integrand is smooth in t."""
LAWS = ("number_gauss", "lognormal")
DATA = Path(__file__).resolve().parent / "data"
PARAMETERS = dict(version=1, S_MIN=S_MIN, S_MAX=S_MAX, KINK=KINK, X_SMALL=X_SMALL, X_BIG=X_BIG, U_MAX=U_MAX,
                  S_POINTS=S_POINTS, U_STEP=U_STEP, LNX_STEP=LNX_STEP, BUILD_PANELS=list(BUILD_PANELS),
                  ENVELOPE_PANELS=ENVELOPE_PANELS, DAMPING=envelope.DAMPING, BLEND=envelope.BLEND, SPAN=sizes.SPAN)
"""What a prebuilt table must have been built with to be used."""


def _sphere(x):
    x = np.asarray(x, float)
    env = 4.5 * (1.0 + x * x) / x**6
    g = 9.0 / x**6 * (0.5 * (x * x - 1.0) + 1j * x)
    a = -3.0 * (x + 1j) / x**3
    return env, g, a


def _disc(x):
    x = np.asarray(x, float)
    h = j1(x) + 1j * y1(x)
    env = 2.0 * (h.real**2 + h.imag**2) / x**2
    g = 2.0 / x**2 * h**2 * np.exp(-2j * x)
    a = 2.0 / x * h * np.exp(-1j * x)
    return env, g, a


def _sinc(x):
    x = np.asarray(x, float)
    env = 2.0 / x**2
    return env, -2.0 / x**2 + 0j, -2j / x


KINDS = {"sphere": (_sphere, shapes.sphere_amplitude, 6, 2.0, 1.0),
         "disc": (_disc, shapes.disc_amplitude, 4, 2.0, 1.0),
         "sinc": (_sinc, shapes.sinc_half, 2, 1.0, 0.5)}
"""kind: (parts, exact amplitude, power p of V² ∝ t^p, ω, ω_a)."""


def _relative_damping(law: str, rel, power) -> np.ndarray:
    """σ_d = (``envelope.damping_scale``)/mean, a function of the relative width only."""
    rel = np.atleast_1d(np.asarray(rel, float))
    return np.atleast_1d(envelope.damping_scale(law, np.ones_like(rel), rel, power)).astype(float)


def _segment(law, kind, ln_s, u, ln_x):
    parts, _f, power, omega, omega_a = KINDS[kind]
    e_tab = np.empty((ln_s.size, ln_x.size))
    b_tab = np.empty((ln_s.size, u.size), complex)
    a_tab = np.empty((ln_s.size, u.size), complex)
    x_big = np.exp(ln_x)
    env_big = parts(x_big)[0]
    for i, rel in enumerate(np.exp(ln_s)):
        t, n = (a[0] for a in sizes.nodes(law, 1.0, rel, power=power, panels=ENVELOPE_PANELS))
        w = n * t**power
        e_tab[i] = (parts(x_big[:, None] * t[None, :])[0] @ (w / w.sum())) / env_big
        sigma = float(_relative_damping(law, rel, power)[0])
        span = float(sizes.extent(law, 1.0, rel, power)[0])
        panels = int(np.clip(np.ceil(omega * U_MAX * span / (sigma * np.pi)), *BUILD_PANELS))
        t, n = (a[0] for a in sizes.nodes(law, 1.0, rel, power=power, panels=panels))
        w = n * t**power
        w /= w.sum()
        wa = n * t ** (power / 2)
        wa /= np.sqrt(np.sum(n * t**power))
        xbar = u / sigma
        _env, g_x, a_x = parts(xbar)
        b_sum, a_sum = np.zeros(u.size, complex), np.zeros(u.size, complex)
        for block in range(0, t.size, 512):
            tt = t[block:block + 512]
            _e, g_t, a_t = parts(xbar[:, None] * tt[None, :])
            b_sum += (g_t * np.exp(1j * omega * xbar[:, None] * (tt[None, :] - 1.0))) @ w[block:block + 512]
            a_sum += (a_t * np.exp(1j * omega_a * xbar[:, None] * (tt[None, :] - 1.0))) @ wa[block:block + 512]
        b_tab[i], a_tab[i] = b_sum / g_x, a_sum / a_x
    return dict(ln_s=ln_s, e=e_tab, b=b_tab, a=a_tab)


def compute(law: str, kind: str) -> dict:
    """Build the tables of one (law, kind): two width segments split at ``KINK``, grids in u and ln x̄;
    Ẽ = E/env(x̄), B̂ = B/G(x̄), Â = Ba/A(x̄) with Ba = <n t^{p/2} A(x̄t) e^{iω_a x̄(t−1)}>/√<n t^p>."""
    u = np.arange(0.0, U_MAX + 4 * U_STEP, U_STEP)
    u[0] = 1e-6
    ln_x = np.arange(np.log(X_SMALL) - 3 * LNX_STEP, np.log(X_BIG) + 4 * LNX_STEP, LNX_STEP)
    bounds = ((S_MIN, KINK), (KINK, S_MAX[law]))
    segments = [_segment(law, kind, np.linspace(np.log(lo), np.log(hi), S_POINTS), u, ln_x) for lo, hi in bounds]
    return dict(segments=segments, u=u, ln_x=ln_x)


def path(law: str, kind: str, folder=None) -> Path:
    return Path(DATA if folder is None else folder) / f"{law}_{kind}.npz"


def save(law: str, kind: str, folder=None) -> Path:
    """Write the tables of (law, kind) as ``data/<law>_<kind>.npz`` with the parameters they were built with."""
    table = compute(law, kind)
    arrays = dict(u=table["u"], ln_x=table["ln_x"], parameters=np.array(json.dumps(PARAMETERS, sort_keys=True)))
    single = {"e": np.float32, "b": np.complex64, "a": np.complex64, "ln_s": np.float64}
    for number, segment in enumerate(table["segments"]):  # single precision: 6e-8, far below the interpolation
        arrays.update({f"{key}{number}": value.astype(single[key]) for key, value in segment.items()})
    target = path(law, kind, folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    return target


def _load(law: str, kind: str):
    target = path(law, kind)
    if not target.is_file():
        return None
    with np.load(target, allow_pickle=False) as data:
        if json.loads(str(data["parameters"])) != json.loads(json.dumps(PARAMETERS, sort_keys=True)):
            return None
        segments = [dict(ln_s=data[f"ln_s{n}"], e=data[f"e{n}"].astype(float), b=data[f"b{n}"].astype(complex),
                         a=data[f"a{n}"].astype(complex)) for n in range(2)]
        return dict(segments=segments, u=data["u"], ln_x=data["ln_x"])


@lru_cache(maxsize=16)
def build(law: str, kind: str) -> dict:
    """The tables of (law, kind): the prebuilt file when it matches ``PARAMETERS``, else computed now."""
    return _load(law, kind) or compute(law, kind)


def _lagrange(grid: np.ndarray, x: np.ndarray):
    """Indices (…, 4) and four-point Lagrange weights (…, 4) of ``x`` on a uniform ``grid``."""
    step = grid[1] - grid[0]
    t = (np.asarray(x, float) - grid[0]) / step
    i = np.clip(np.floor(t).astype(int) - 1, 0, grid.size - 4)
    f = t - i
    w = np.stack((-(f - 1) * (f - 2) * (f - 3) / 6, f * (f - 2) * (f - 3) / 2, -f * (f - 1) * (f - 3) / 2,
                  f * (f - 1) * (f - 2) / 6), axis=-1)
    return i[..., None] + np.arange(4), w


def _rows(tab: dict, key: str, rel: np.ndarray) -> np.ndarray:
    """The table ``key`` interpolated to every row's width, within the row's segment: (B, columns)."""
    out = None
    for number, segment in enumerate(tab["segments"]):
        mine = rel <= KINK if number == 0 else rel > KINK
        if not mine.any():
            continue
        index, weights = _lagrange(segment["ln_s"], np.log(rel[mine]))
        values = np.einsum("bk,bkc->bc", weights, segment[key][index])
        if out is None:
            out = np.empty((rel.size, values.shape[1]), values.dtype)
        out[mine] = values
    return out


def _along(rows: np.ndarray, grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Each row of ``rows`` (B, G) interpolated at that row's ``x`` (B, N)."""
    index, weights = _lagrange(grid, np.clip(x, grid[0], grid[-1]))
    return np.sum(rows[np.arange(rows.shape[0])[:, None, None], index] * weights, axis=-1)


def _smoothstep(u):
    """``envelope.blend`` in u = q·σ: 0 below DAMPING, 1 above BLEND·DAMPING."""
    t = np.clip((u - envelope.DAMPING) / ((envelope.BLEND - 1.0) * envelope.DAMPING), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def usable(law: str, rel) -> np.ndarray:
    """Rows whose width the tables cover."""
    rel = np.atleast_1d(np.asarray(rel, float))
    return (law in LAWS) & (rel >= S_TABLE) & (rel <= S_MAX.get(law, 0.0))


def _tabulated(law, kind, xbar, rel, coherent):
    """P̂ and the signed amplitude mean at x̄ ≥ X_SMALL, (B, N)."""
    parts, _f, power, omega, omega_a = KINDS[kind]
    tab = build(law, kind)
    env, g_x, a_x = parts(xbar)
    p = env * _along(_rows(tab, "e", rel), tab["ln_x"], np.log(np.minimum(xbar, X_BIG)))
    u = xbar * _relative_damping(law, rel, power)[:, None]
    keep = 1.0 - _smoothstep(u)
    b = _along(_rows(tab, "b", rel), tab["u"], np.minimum(u, U_MAX))
    p = p + keep * np.real(np.exp(1j * omega * xbar) * g_x * b)
    mean = None
    if coherent:
        a = _along(_rows(tab, "a", rel), tab["u"], np.minimum(u, U_MAX))
        mean = keep * np.real(np.exp(1j * omega_a * xbar) * a_x * a)
    return p, mean


def averages(law: str, kind: str, xbar, rel, coherent: bool = False):
    """(P̂, signed amplitude mean <V f>/√<V²> or None) at x̄ = q·mean of shape (B, …) ≥ 0 for the widths ``rel`` (B,)
    that ``usable`` accepts: tables from ``X_SMALL`` on, the exact average on a few nodes below."""
    xbar = np.asarray(xbar, float)
    rel = np.atleast_1d(np.asarray(rel, float))
    shape = xbar.shape
    flat = xbar.reshape(rel.size, -1)
    small = flat < X_SMALL
    p, mean = _tabulated(law, kind, np.maximum(flat, X_SMALL), rel, coherent)
    if small.any():
        _parts, amplitude, power, _omega, _omega_a = KINDS[kind]
        for row in np.flatnonzero(small.any(axis=1)):
            panels = int(sizes.panel_counts(X_SMALL, sizes.extent(law, 1.0, rel[row], power))[0])
            t, n = (a[0] for a in sizes.nodes(law, 1.0, rel[row], power=power, panels=panels))
            x = flat[row, small[row]][:, None] * t[None, :]
            f = amplitude(x)
            w = n * t**power
            p[row, small[row]] = (f * f) @ w / w.sum()
            if coherent:
                mean[row, small[row]] = f @ (n * t ** (power / 2)) / np.sqrt(w.sum())
    return p.reshape(shape), None if mean is None else mean.reshape(shape)


def cylinder(q, law, radius, radius_spread, height, height_spread, qz=0.0, coherent=False):
    """(P̂, C or None) of randomly oriented cylinders (|q| = √(q² + qz²)); rows (B,) the tables cover for both widths.

    Per row the radius and height averages are taken from the tables on a 1D grid of x = q sin α (q cos α), with
    ``shapes.GRID_PER_PERIOD`` points per period of the characteristic size (``sizes.characteristic``), and at every
    orientation node α and q by cubic interpolation, then averaged over α (``shapes.orientation_nodes``, up to
    BLEND · q_cut)."""
    q_total = np.hypot(np.asarray(q, float), qz)
    q_max = float(q_total.max()) if q_total.size else 0.0
    radius, rel_r = np.atleast_1d(radius).astype(float), np.atleast_1d(radius_spread).astype(float)
    height, rel_h = np.atleast_1d(height).astype(float), np.atleast_1d(height_spread).astype(float)
    cut = np.maximum(envelope.q_cut(law, radius, rel_r, 4), envelope.q_cut(law, height, rel_h, 2))
    size_r, size_h = sizes.characteristic(law, radius, rel_r, 4), sizes.characteristic(law, height, rel_h, 2)
    p_out = np.empty((radius.size, q_total.size))
    c_out = np.empty_like(p_out) if coherent else None
    for row in range(radius.size):
        u, wu = shapes.orientation_nodes(min(q_max, envelope.BLEND * float(cut[row])),
                                         max(float(size_r[row]), 0.5 * float(size_h[row])))
        sin_a = np.sqrt(np.maximum(1.0 - u * u, 0.0))
        r_grid, r_step = shapes._grid(q_max, float(size_r[row]), np.pi)
        h_grid, h_step = shapes._grid(q_max, float(size_h[row]), 2.0 * np.pi)
        radial, radial_mean = averages(law, "disc", r_grid[None] * radius[row], rel_r[row:row + 1], coherent)
        axial, axial_mean = averages(law, "sinc", h_grid[None] * height[row], rel_h[row:row + 1], coherent)
        x_r, x_h = sin_a[:, None] * q_total[None, :], u[:, None] * q_total[None, :]
        p_out[row] = wu @ (shapes._cubic(radial, r_step, x_r)[0] * shapes._cubic(axial, h_step, x_h)[0])
        if coherent:
            c_out[row] = (wu @ (shapes._cubic(radial_mean, r_step, x_r)[0]
                                * shapes._cubic(axial_mean, h_step, x_h)[0])) ** 2
    return p_out, c_out


__all__ = ["KINDS", "KINK", "LAWS", "S_MAX", "S_MIN", "S_TABLE", "X_SMALL", "averages", "build", "compute", "cylinder",
           "path", "save", "usable"]
