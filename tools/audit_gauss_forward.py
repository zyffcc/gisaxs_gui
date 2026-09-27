"""Independent quadrature audit of the same continuous particle form factors.

Diagnostic only. Frozen networks were trained on the older finite-grid forward;
this function must not silently replace their label/parameter interpretation.
"""

from functools import lru_cache
import numpy as np
from scipy.special import j1


@lru_cache(maxsize=16)
def legendre(n):
    return np.polynomial.legendre.leggauss(n)


def gaussian_nodes(mu, relative_width, n, nsig):
    sigma = mu * relative_width
    lo = max(mu - nsig * sigma, 0)
    hi = mu + nsig * sigma
    x, w = legendre(n)
    nodes = lo + (x + 1) * (hi - lo) / 2
    weights = w * np.exp(-0.5 * ((nodes - mu) / sigma) ** 2)
    return nodes, weights / weights.sum()


def radial(x):
    safe = np.where(abs(x) < 1e-4, 1, x)
    return np.where(abs(x) < 1e-4, 1 - x * x / 8 + x**4 / 192, 2 * j1(safe) / safe)


def component(q, typ, r, sr, h, sh, dist, sd, d_present, order=24, orientation_order=64):
    q = np.asarray(q)
    if typ == 1:
        rr, w = gaussian_nodes(r, sr, order, 4)
        x = rr[:, None] * q[None, :]
        safe = np.where(abs(x) < 0.1, 1, x)
        form = 3 * (np.sin(safe) - safe * np.cos(safe)) / safe**3
        form = np.where(abs(x) < 0.1, 1 - x * x / 10 + x**4 / 280 - x**6 / 15120, form)
        result = w @ (form * form)
    elif typ == 3:
        rr, w = gaussian_nodes(r, sr, order, 3)
        result = w @ radial(rr[:, None] * q[None, :]) ** 2
    else:
        rr, wr = gaussian_nodes(r, sr, order, 4)
        hh, wh = gaussian_nodes(h, sh, order, 4)
        xx, ww = legendre(orientation_order)
        u = (xx + 1) / 2
        weights = ww / 2
        fm = np.sum(
            wr[:, None, None]
            * radial(rr[:, None, None] * np.sqrt(1 - u * u)[None, :, None] * q[None, None, :]) ** 2,
            axis=0,
        )
        hm = np.sum(
            wh[:, None, None]
            * np.sinc(hh[:, None, None] * u[None, :, None] * q[None, None, :] / (2 * np.pi)) ** 2,
            axis=0,
        )
        result = np.sum(weights[:, None] * fm * hm, axis=0)
    if d_present:
        lp = -np.pi * q * q * (dist * sd) ** 2
        phi = np.exp(lp)
        result *= (-np.expm1(2 * lp)) / np.maximum(
            (-np.expm1(lp)) ** 2 + 4 * phi * np.sin(0.5 * q * dist) ** 2, 1e-15
        )
    return result
