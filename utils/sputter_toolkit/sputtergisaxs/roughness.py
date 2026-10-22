"""Interface roughness: K-correlation (Palasantzas) power spectrum."""
from __future__ import annotations

import numpy as np


def psd_k_correlation(qpar: np.ndarray, sigma: float, xi: float, hurst: float) -> np.ndarray:
    """2D PSD of C(R) ~ sigma^2 exp(-(R/xi)^(2H)); exact for H = 0.5 (exponential):
    PSD(q) = 4 pi H sigma^2 xi^2 / (1 + q^2 xi^2)^(1 + H)."""
    q = np.asarray(qpar, dtype=float)
    return 4.0 * np.pi * hurst * sigma ** 2 * xi ** 2 / (1.0 + (q * xi) ** 2) ** (1.0 + hurst)
