'Plane-wave amplitudes in a stratified medium (Parratt recursion, s-polarisation,'
from __future__ import annotations

import numpy as np


def kz_in_layers(k: float, alpha: np.ndarray, nds: list[complex]) -> list[np.ndarray]:
    s2 = np.sin(alpha).astype(np.complex128) ** 2
    out = []
    for nd in nds:
        kz = k * np.sqrt(s2 - 2 * nd.real + 2j * nd.imag)
        kz = np.where(kz.imag < 0, -kz, kz)
        out.append(kz)
    return out


def amplitudes(k: float, alpha: np.ndarray, nds: list[complex], thick: list[float],
               sigma: list[float] | None = None) -> dict:
    'Return per-layer kz, T (down) and R (up) amplitudes at the top of every layer'
    alpha = np.asarray(alpha, dtype=float)
    nl = len(nds)
    if sigma is None:
        sigma = [0.0] * nl
    kz = kz_in_layers(k, alpha, nds)
    d = [0.0] + list(thick) + [0.0]
    assert len(d) == nl
    X = [None] * nl
    X[nl - 1] = np.zeros_like(kz[0])
    for j in range(nl - 2, -1, -1):
        r = (kz[j] - kz[j + 1]) / (kz[j] + kz[j + 1])
        if sigma[j + 1] > 0:
            r = r * np.exp(-2.0 * kz[j] * kz[j + 1] * sigma[j + 1] ** 2)
        xb = (r + X[j + 1]) / (1.0 + r * X[j + 1])
        X[j] = xb * np.exp(2j * kz[j] * d[j]) if j > 0 else xb
    T = [None] * nl
    T[0] = np.ones_like(kz[0])
    for j in range(0, nl - 1):
        a = T[j] * (np.exp(1j * kz[j] * d[j]) if j > 0 else 1.0)
        xb = X[j] * (np.exp(-2j * kz[j] * d[j]) if j > 0 else 1.0)
        T[j + 1] = a * (1.0 + xb) / (1.0 + X[j + 1])
    R = [X[j] * T[j] for j in range(nl)]
    return {"kz": kz, "T": T, "R": R, "d": d}


def fields_at_bottom(amp: dict, j: int) -> tuple[np.ndarray, np.ndarray]:
    'Down/up amplitudes referenced to the bottom of layer j (j >= 1).'
    kz, T, R, d = amp["kz"][j], amp["T"][j], amp["R"][j], amp["d"][j]
    return T * np.exp(1j * kz * d), R * np.exp(-1j * kz * d)


def reflectivity(k: float, alpha: np.ndarray, nds, thick, sigma=None) -> np.ndarray:
    amp = amplitudes(k, alpha, nds, thick, sigma)
    return np.abs(amp["R"][0]) ** 2
