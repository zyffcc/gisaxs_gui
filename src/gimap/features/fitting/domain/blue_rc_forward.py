"""Versioned local RC form factor; amplitudes never depend on the render grid.

Validated for the local blue-RC branch, not the entire legacy parameter domain.
"""

from functools import lru_cache

import numpy as np
from scipy.special import j1

FORWARD_VERSION = "blue_rc_gauss48_96_v1"


@lru_cache(maxsize=8)
def _legendre(n):
    return np.polynomial.legendre.leggauss(n)


def _nodes(mu, relative_width, n):
    sigma = mu * relative_width
    lo, hi = max(mu - 4 * sigma, 0), mu + 4 * sigma
    x, w = _legendre(n)
    nodes = lo + (x + 1) * (hi - lo) / 2
    weights = w * np.exp(-0.5 * ((nodes - mu) / sigma) ** 2)
    return nodes, weights / weights.sum()


def particle(q, parameters, *, order=48, angles=96):
    q = np.abs(np.asarray(q, float))
    p = parameters
    rr, wr = _nodes(p["R"], p["sigma_R"], order)
    hh, wh = _nodes(p["h"], p["sigma_h"], order)
    xx, ww = _legendre(angles)
    u, weights = (xx + 1) / 2, ww / 2
    x = rr[:, None, None] * np.sqrt(1 - u * u)[None, :, None] * q[None, None, :]
    safe = np.where(abs(x) < 1e-4, 1, x)
    radial = np.where(abs(x) < 1e-4, 1 - x * x / 8 + x**4 / 192, 2 * j1(safe) / safe)
    fm = np.sum(wr[:, None, None] * radial**2, axis=0)
    hm = np.sum(
        wh[:, None, None]
        * np.sinc(hh[:, None, None] * u[None, :, None] * q[None, None, :] / (2 * np.pi)) ** 2,
        axis=0,
    )
    form = np.sum(weights[:, None] * fm * hm, axis=0)
    lp = -np.pi * q * q * (p["D"] * p["sigma_D"]) ** 2
    phi = np.exp(lp)
    return (
        form
        * (-np.expm1(2 * lp))
        / np.maximum((-np.expm1(lp)) ** 2 + 4 * phi * np.sin(0.5 * q * p["D"]) ** 2, 1e-15)
    )


def forward(q, components, globals_):
    q = np.abs(np.asarray(q, float))
    if len(components) != 1 or components[0]["type_id"] != 2:
        raise ValueError("The blue RC forward supports exactly one random cylinder")
    c = components[0]
    return (
        c["amplitude"] * particle(q, c["params"])
        + globals_["background"]
        + globals_["resolution_amplitude"] / (1 + (q / globals_["sigma_Res"]) ** globals_["nu_Res"])
    )
