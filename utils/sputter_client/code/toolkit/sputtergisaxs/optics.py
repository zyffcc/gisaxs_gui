'X-ray optical constants.'
from __future__ import annotations

from pathlib import Path

import numpy as np

ENERGY_EV = 12000.0
WAVELENGTH_NM = 12398.419843 / ENERGY_EV / 10.0


TABLE_12KEV = {
    "vacuum": (0.0, 0.0, 0.0, ""),
    "Au": (1.886e-5, 2.811e-6, 19.32, "Au"),
    "Ag": (1.310e-5, 5.943e-7, 10.49, "Ag"),
    "Si": (3.384e-6, 3.545e-8, 2.33, "Si"),
    "SiO2": (3.187e-6, 1.853e-8, 2.20, "SiO2"),
    "TiO2": (5.394e-6, 1.279e-7, 3.89, "TiO2"),
    "PS": (1.629e-6, 8.627e-10, 1.05, "C8H8"),
    "PMMA": (1.837e-6, 1.558e-9, 1.18, "C5H8O2"),
}


import os

TABLE_FILE = Path(os.environ.get("SGX_OPTICS_TABLE") or Path(__file__).resolve().parents[1] / "assets" / "optics_chantler.npz")
_TAB: dict = {}
_CACHE: dict = {}


def wavelength_nm(energy_ev: float | None = None) -> float:
    return 1239.8419843 / float(energy_ev or ENERGY_EV)


def _table() -> dict:
    if not _TAB and TABLE_FILE.exists():
        z = np.load(TABLE_FILE)
        _TAB["E"] = z["energy_ev"].astype(float)
        for n in TABLE_12KEV:
            if n != "vacuum" and f"delta_{n}" in z:
                _TAB[n] = (z[f"delta_{n}"].astype(float), z[f"beta_{n}"].astype(float))
    return _TAB


def delta_beta(name: str, energy_ev: float | None = None) -> tuple[float, float]:
    '(delta, beta) of a bulk material at the given photon energy (eV); 12 keV table by default.'
    if name == "vacuum":
        return 0.0, 0.0
    if energy_ev is None or abs(energy_ev - ENERGY_EV) < 0.5:
        d, b, _, _ = TABLE_12KEV[name]
        return d, b
    key = (name, round(float(energy_ev), 1))
    if key in _CACHE:
        return _CACHE[key]
    tab = _table()
    if name in tab:
        E = tab["E"]
        if not E[0] <= energy_ev <= E[-1]:
            raise ValueError(f"{energy_ev} eV outside the optics table ({E[0]:.0f}-{E[-1]:.0f} eV)")
        d, b = float(np.interp(energy_ev, E, tab[name][0])), float(np.interp(energy_ev, E, tab[name][1]))
    else:
        try:
            _, _, rho, formula = TABLE_12KEV[name]
            c = material_at_energy(formula, rho, float(energy_ev))
            d, b = c.real, c.imag
        except ImportError as e:
            raise RuntimeError(f"optical constants at {energy_ev} eV need {TABLE_FILE.name} or xraydb") from e
    _CACHE[key] = (d, b)
    return d, b


def nd(name: str, density_scale: float = 1.0, energy_ev: float | None = None) -> complex:
    'Return the refractive decrement as complex delta + i*beta (n = 1 - delta + i beta).'
    d, b = delta_beta(name, energy_ev)
    return complex(d * density_scale, b * density_scale)


def mix(a: complex, b: complex, fraction_a: float) -> complex:
    'Linear (volume-fraction) effective medium of two decrements.'
    return fraction_a * a + (1.0 - fraction_a) * b


def material_at_energy(formula: str, density: float, energy_ev: float) -> complex:
    'Optional: compute delta + i beta with xraydb (pip install xraydb).'
    import xraydb
    delta, beta, _ = xraydb.xray_delta_beta(formula, density, energy_ev)
    return complex(delta, beta)
