'X-ray optical constants at any energy from the tabulated anomalous scattering factors (xray_optics_table.npz).'

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLE = HERE / "xray_optics_table.npz"
RE_NM = 2.8179403262e-6
NA21 = 602.214076
HC_EV_NM = 1239.841984

MATERIALS = {
    "Au": ("Au", 19.32), "Ag": ("Ag", 10.49), "Cu": ("Cu", 8.96), "Pt": ("Pt", 21.45), "Pd": ("Pd", 12.02),
    "Ni": ("Ni", 8.91), "Al": ("Al", 2.70), "Ti": ("Ti", 4.51), "Cr": ("Cr", 7.19), "W": ("W", 19.25),
    "Si": ("Si", 2.329), "SiO2": ("SiO2", 2.2), "TiO2": ("TiO2", 3.9),
    "PS": ("C8H8", 1.05), "PET": ("C10H8O4", 1.38), "PSBMA": ("C11H21NO5S", 1.30), "PNO": ("C8H15NO3", 1.20),
    "PEO": ("C2H4O", 1.13), "PMMA": ("C5H8O2", 1.18), "Polymer": ("C2H4O", 1.13),
}


@lru_cache(maxsize=1)
def _table():
    z = np.load(TABLE, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    index = {str(el): i for i, el in enumerate(z["elements"])}
    return dict(energy=z["energy"].astype(float), f1=z["f1"].astype(float), f2=z["f2"].astype(float),
                z=z["z"].astype(int), mass=z["mass"].astype(float), index=index, meta=meta)


def _check_energy(energy_ev: float) -> None:
    t = _table()
    if not t["energy"][0] <= energy_ev <= t["energy"][-1]:
        raise ValueError(f"energy {energy_ev:.0f} eV outside the optics table ({t['energy'][0]:.0f}-{t['energy'][-1]:.0f} eV)")


def f12(element: str, energy_ev: float) -> tuple:
    "(f1, f2) of an element at an energy (eV): f1 = Z + f' (Chantler), linear interpolation in energy."
    t = _table()
    if element not in t["index"]:
        raise KeyError(f"element {element} not in the optics table ({', '.join(t['index'])})")
    _check_energy(float(energy_ev))
    i = t["index"][element]
    f1 = t["z"][i] + float(np.interp(energy_ev, t["energy"], t["f1"][i]))
    f2 = float(np.interp(energy_ev, t["energy"], t["f2"][i]))
    return f1, f2


def parse_formula(formula: str) -> dict:
    "'C11H21NO5S' -> {'C': 11, 'H': 21, 'N': 1, 'O': 5, 'S': 1} (element symbols and counts, no brackets)."
    parts = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula)
    if not parts or "".join(e + n for e, n in parts) != formula:
        raise ValueError(f"formula {formula!r}: write element symbols and counts, e.g. C8H8 or TiO2")
    out: dict = {}
    for el, n in parts:
        out[el] = out.get(el, 0.0) + (float(n) if n else 1.0)
    return out


def _material(material):
    if isinstance(material, str):
        if material not in MATERIALS:
            raise KeyError(f"material {material} unknown: give (formula, density) or one of {', '.join(MATERIALS)}")
        return MATERIALS[material]
    formula, density = material
    return str(formula), float(density)


def electron_density(material, energy_ev: float) -> tuple:
    '(sum_i n_i f1_i, sum_i n_i f2_i) in electrons per nm^3 of a material at an energy (eV).'
    formula, density = _material(material)
    t = _table()
    counts = parse_formula(formula)
    mol = sum(n * t["mass"][t["index"][el]] for el, n in counts.items())
    scale = density / mol * NA21
    s1 = s2 = 0.0
    for el, n in counts.items():
        a, b = f12(el, energy_ev)
        s1 += scale * n * a
        s2 += scale * n * b
    return s1, s2


def decrement(material, energy_ev: float) -> complex:
    'delta - i beta (refractive index n = 1 - delta - i beta) of a material at an energy (eV).'
    lam = HC_EV_NM / float(energy_ev)
    s1, s2 = electron_density(material, energy_ev)
    pref = RE_NM * lam ** 2 / (2.0 * np.pi)
    return pref * s1 - 1j * pref * s2


def edge(element: str, name: str) -> float:
    edges = _table()["meta"]["edges"].get(element, {})
    if name not in edges:
        raise KeyError(f"no {name} edge of {element} in the table")
    return float(edges[name])


FLUORESCENCE = {"Au": "L3", "Pt": "L3", "Ag": "L3", "Pd": "L3", "W": "L3", "Cu": "K", "Ni": "K", "Ti": "K", "Cr": "K"}
"""The shell whose line the forward model's fluorescence term describes (Au L alpha, Cu K alpha, ...)."""


def fluoresces(metal: str, energy_ev: float) -> bool:
    shell = FLUORESCENCE.get(metal)
    return shell is not None and float(energy_ev) > edge(metal, shell)


def selftest() -> bool:
    'Against the constants iam.py has used: Ag / Si at 12 keV, Cu at 0.105 nm and at Cu K alpha within 2 %;'
    ok = True

    def rec(name, value, want, tol):
        nonlocal ok
        good = abs(value / want - 1) <= tol
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {value:.4g} (iam.py {want:.4g}, tolerance {tol:.0%})")

    e12, e105 = 12000.0, HC_EV_NM / 0.105
    rec("Ag f1 at 12 keV", f12("Ag", e12)[0], 47.0, 0.02)
    rec("Ag f2 at 12 keV", f12("Ag", e12)[1], 2.15, 0.03)
    rec("Si f1 at 12 keV", f12("Si", e12)[0], 14.05, 0.02)
    rec("Cu f1 at 0.105 nm", f12("Cu", e105)[0], 28.2, 0.02)
    rec("Cu f2 at 0.105 nm", f12("Cu", e105)[1], 2.45, 0.02)
    rec("Cu f1 at Cu K alpha", f12("Cu", 8048.0)[0], 27.04, 0.02)
    rec("Au f1 at 12 keV", f12("Au", e12)[0], 64.0, 0.06)
    rec("Au f2 at 12 keV", f12("Au", e12)[1], 9.7, 0.06)
    d = decrement("PS", e12)
    lam = HC_EV_NM / e12
    want = RE_NM * lam ** 2 / (2 * np.pi) * 1.05 / 104.15 * 56 * NA21
    rec("PS delta at 12 keV", d.real, want, 0.01)
    good = fluoresces("Au", 12000.0) and not fluoresces("Au", 11800.0) and fluoresces("Cu", 1239.841984 / 0.105)
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] fluorescence: Au L3 above 11919 eV only, Cu K above 8979 eV")
    print("XRAY OPTICS SELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
