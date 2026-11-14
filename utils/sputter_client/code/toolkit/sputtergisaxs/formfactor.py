'Form factors of flat-bottomed islands, origin at the centre of the base plane.'
from __future__ import annotations

import numpy as np
from scipy.special import j1

_GL = {}


def _gauss_legendre(n: int):
    if n not in _GL:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL[n] = ((x + 1) / 2, w / 2)
    return _GL[n]


def jinc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-6
    out[nz] = 2.0 * j1(x[nz]) / x[nz]
    return out


def cap_profile(a: float, h: float, t: np.ndarray) -> np.ndarray:
    'radius r(z) of a spherical cap at heights z = t*h (t in [0,1]).'
    Rs = (a * a + h * h) / (2.0 * h)
    z = t * h
    return np.sqrt(np.maximum(Rs * Rs - (z - (h - Rs)) ** 2, 0.0))


def hemiellipsoid_profile(a: float, h: float, t: np.ndarray) -> np.ndarray:
    return a * np.sqrt(np.maximum(1.0 - t * t, 0.0))


def volume(a, h, shape: str = "cap"):
    a = np.asarray(a, dtype=float)
    h = np.asarray(h, dtype=float)
    if shape == "cap":
        return np.pi * h * (3 * a * a + h * h) / 6.0
    return 2.0 / 3.0 * np.pi * a * a * h


def slice_tables(qpar: np.ndarray, a_list, h_list, shape: str = "cap", nz: int = 16):
    'Precompute lateral slice amplitudes A[s, k, pix] = w_k * h_s * pi r^2 jinc(q r)'
    t, w = _gauss_legendre(nz)
    prof = cap_profile if shape == "cap" else hemiellipsoid_profile
    q = np.asarray(qpar, dtype=float).ravel()
    A = np.empty((len(a_list), nz, q.size), dtype=np.float64)
    Z = np.empty((len(a_list), nz), dtype=np.float64)
    for s, (a, h) in enumerate(zip(a_list, h_list)):
        r = prof(a, h, t)
        A[s] = (w * h * np.pi * r * r)[:, None] * jinc(np.outer(r, q))
        Z[s] = t * h
    return A, Z


def ff_from_tables(A: np.ndarray, Z: np.ndarray, qz: np.ndarray) -> np.ndarray:
    'F[s, pix] = sum_k A[s,k,pix] exp(i qz[pix] z[s,k]); qz complex, shape (pix,).'
    qz = np.asarray(qz).ravel()
    phase = np.exp(1j * Z[:, :, None] * qz[None, None, :])
    return np.einsum("skp,skp->sp", A, phase)
