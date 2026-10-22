"""Lateral correlation: radial paracrystal structure factor (as BornAgain
InterferenceRadialParacrystal with a Gaussian nearest-neighbour distribution)."""
from __future__ import annotations

import numpy as np


def radial_paracrystal(qpar: np.ndarray, D: float, omega: float, damping: float = 1e4) -> np.ndarray:
    """S(q) = Re[(1+P)/(1-P)],  P = exp(-q^2 omega^2 / 2) exp(i q D) exp(-D / damping).
    D: mean nearest-neighbour distance (nm); omega: its standard deviation (nm)."""
    q = np.asarray(qpar, dtype=float)
    P = np.exp(-0.5 * (q * omega) ** 2 + 1j * q * D - D / damping)
    return np.real((1 + P) / (1 - P))


def hex_distance_from_density(n_per_nm2: float) -> float:
    """Nearest-neighbour distance of a hexagonal arrangement with areal density n."""
    return float(np.sqrt(2.0 / (np.sqrt(3.0) * n_per_nm2)))


def density_from_hex_distance(D: float) -> float:
    return float(2.0 / (np.sqrt(3.0) * D * D))
