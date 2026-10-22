"""Form factors of flat-bottomed islands, origin at the centre of the base plane.

Spherical cap (BornAgain TruncatedSphere(R_s, h, 0)) with base radius a and height h:
    R_s = (a^2 + h^2) / (2 h),  r(z)^2 = R_s^2 - (z - (h - R_s))^2,  0 <= z <= h.
Half-ellipsoid (BornAgain HemiEllipsoid(a, a, h)) is also provided for comparison.
F(q_par, q_z) = sum_k w_k * pi r_k^2 * jinc(q_par r_k) * exp(i q_z z_k)   (Gauss-Legendre in z),
with jinc(x) = 2 J1(x) / x. q_z may be complex (absorbing layer).
"""
from __future__ import annotations

import numpy as np
from scipy.special import j1

_GL = {}


def _gauss_legendre(n: int):
    if n not in _GL:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL[n] = ((x + 1) / 2, w / 2)       # nodes/weights on [0, 1]
    return _GL[n]


def jinc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-6
    out[nz] = 2.0 * j1(x[nz]) / x[nz]
    return out


def cap_profile(a: float, h: float, t: np.ndarray) -> np.ndarray:
    """radius r(z) of a spherical cap at heights z = t*h (t in [0,1])."""
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
    """Precompute lateral slice amplitudes A[s, k, pix] = w_k * h_s * pi r^2 jinc(q r)
    and heights z[s, k]. Independent of q_z, so reused for all four DWBA terms."""
    t, w = _gauss_legendre(nz)
    prof = cap_profile if shape == "cap" else hemiellipsoid_profile
    q = np.asarray(qpar, dtype=float).ravel()
    A = np.empty((len(a_list), nz, q.size), dtype=np.float64)
    Z = np.empty((len(a_list), nz), dtype=np.float64)
    for s, (a, h) in enumerate(zip(a_list, h_list)):
        r = prof(a, h, t)                       # (nz,)
        A[s] = (w * h * np.pi * r * r)[:, None] * jinc(np.outer(r, q))
        Z[s] = t * h
    return A, Z


def ff_from_tables(A: np.ndarray, Z: np.ndarray, qz: np.ndarray) -> np.ndarray:
    """F[s, pix] = sum_k A[s,k,pix] exp(i qz[pix] z[s,k]); qz complex, shape (pix,)."""
    qz = np.asarray(qz).ravel()
    phase = np.exp(1j * Z[:, :, None] * qz[None, None, :])   # (s, k, pix)
    return np.einsum("skp,skp->sp", A, phase)
