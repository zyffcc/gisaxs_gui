"""The one-dimensional paracrystal (Hosemann & Bagchi 1962; for GISAXS Renaud, Lazzari & Leroy 2009).

Neighbour distances Gaussian with mean D and standard deviation σ_nn (``spread`` = σ_nn/D, the relative standard
deviation of the neighbour distance):

    S(q) = (1 − p²) / (1 − 2p cos(qD) + p²),   p = exp(−q² σ_nn² / 2),   S(0) = (σ_nn/D)²

evaluated as −expm1(2 ln p) / (expm1(ln p)² + 4p sin²(qD/2)), which has no cancellation at small q. GIMaP's Fitting
in earlier versions wrote p = exp(−π q² σ_D²): the same function with σ_nn = √(2π) σ_D (``conventions``).
"""

from __future__ import annotations

import numpy as np


def paracrystal(q, distance, spread):
    """S(q) for every row of ``distance`` and ``spread`` (arrays (B,)); result (B, N). Rows with distance or spread
    ≤ 0 have S = 1."""
    q = np.abs(np.asarray(q, float))[None, :]
    distance = np.atleast_1d(np.asarray(distance, float))[:, None]
    spread = np.atleast_1d(np.asarray(spread, float))[:, None]
    off = (distance <= 0) | (spread <= 0)
    log_p = -0.5 * (q * np.where(off, 1.0, spread) * np.where(off, 1.0, distance)) ** 2
    p = np.exp(log_p)
    denominator = np.expm1(log_p) ** 2 + 4.0 * p * np.sin(0.5 * q * distance) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        value = -np.expm1(2.0 * log_p) / denominator
    value = np.where(q == 0, spread**2, value)
    return np.where(off, 1.0, value)


__all__ = ["paracrystal"]
