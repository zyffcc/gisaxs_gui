"""Plane-wave amplitudes in a stratified medium (Parratt recursion, s-polarisation,
small-angle kz). Refractive index n_j = 1 - delta_j + i beta_j is passed as
nd_j = delta_j + i beta_j.

Layer 0 is the ambient (vacuum), layer N-1 the substrate. Thicknesses of layers
1..N-2 are in nm. In layer j the field is
    E_j(z) = T_j exp(-i kz_j (z - z_j)) + R_j exp(+i kz_j (z - z_j)),
with z_j the top of layer j (z up, z_0 = z_1 = 0 at the sample surface).
'sigma' holds interface roughness (nm) at the top of each layer j>=1
(Nevot-Croce factor on the Fresnel coefficient).
"""
from __future__ import annotations

import numpy as np


def kz_in_layers(k: float, alpha: np.ndarray, nds: list[complex]) -> list[np.ndarray]:
    s2 = np.sin(alpha).astype(np.complex128) ** 2
    out = []
    for nd in nds:
        kz = k * np.sqrt(s2 - 2 * nd.real + 2j * nd.imag)   # n^2 - cos^2 ~ sin^2 - 2 delta + 2 i beta
        kz = np.where(kz.imag < 0, -kz, kz)                   # decaying downward branch
        out.append(kz)
    return out


def amplitudes(k: float, alpha: np.ndarray, nds: list[complex], thick: list[float],
               sigma: list[float] | None = None) -> dict:
    """Return per-layer kz, T (down) and R (up) amplitudes at the top of every layer
    for a unit wave incident from the ambient at grazing angle alpha (radians)."""
    alpha = np.asarray(alpha, dtype=float)
    nl = len(nds)
    if sigma is None:
        sigma = [0.0] * nl
    kz = kz_in_layers(k, alpha, nds)
    d = [0.0] + list(thick) + [0.0]              # d[0] ambient, d[-1] substrate (unused)
    assert len(d) == nl
    X = [None] * nl                               # R_j / T_j at top of layer j
    X[nl - 1] = np.zeros_like(kz[0])
    for j in range(nl - 2, -1, -1):
        r = (kz[j] - kz[j + 1]) / (kz[j] + kz[j + 1])
        if sigma[j + 1] > 0:
            r = r * np.exp(-2.0 * kz[j] * kz[j + 1] * sigma[j + 1] ** 2)
        xb = (r + X[j + 1]) / (1.0 + r * X[j + 1])  # ratio at bottom of layer j
        X[j] = xb * np.exp(2j * kz[j] * d[j]) if j > 0 else xb
    T = [None] * nl
    T[0] = np.ones_like(kz[0])
    for j in range(0, nl - 1):
        a = T[j] * (np.exp(1j * kz[j] * d[j]) if j > 0 else 1.0)        # down amplitude at bottom of j
        xb = X[j] * (np.exp(-2j * kz[j] * d[j]) if j > 0 else 1.0)      # ratio at bottom of j
        T[j + 1] = a * (1.0 + xb) / (1.0 + X[j + 1])
    R = [X[j] * T[j] for j in range(nl)]
    return {"kz": kz, "T": T, "R": R, "d": d}


def fields_at_bottom(amp: dict, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Down/up amplitudes referenced to the bottom of layer j (j >= 1)."""
    kz, T, R, d = amp["kz"][j], amp["T"][j], amp["R"][j], amp["d"][j]
    return T * np.exp(1j * kz * d), R * np.exp(-1j * kz * d)


def reflectivity(k: float, alpha: np.ndarray, nds, thick, sigma=None) -> np.ndarray:
    amp = amplitudes(k, alpha, nds, thick, sigma)
    return np.abs(amp["R"][0]) ** 2
