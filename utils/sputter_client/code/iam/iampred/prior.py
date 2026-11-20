'Growth-law prior of the whole-scan refinement (universal sputter model, step 3; 2026-11-20; option, off by default).'

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .process import COMMON, kind_of

SA_FLOOR = 0.15
SP_FLOOR = 0.10
GATE_POWER = 6.0
MIN_SCANS = 2
LAWS = (("D", "D1", "p_D"), ("R", "R1", "p_R"))


class GrowthPrior:
    "The prior of one scan. reference: dict {system: {...}} or a path to one; system 'metal/substrate'."

    def __init__(self, family: str, metal: str, substrate: str, reference, weight: float = 1.0, min_scans: int = MIN_SCANS):
        ref = json.loads(Path(reference).read_text(encoding="utf-8")) if isinstance(reference, (str, Path)) else reference
        self.system = f"{metal}/{substrate}"
        self.weight = float(weight)
        self.min_scans = int(min_scans)
        self.kind = kind_of(family)
        row = (ref or {}).get(self.system) or {}
        self.n_scans = int(row.get("n", 0))
        self.laws, self.note = [], ""
        common = COMMON.get(self.kind, {})
        hb = row.get("h_base") or {}
        self.h_film = float(hb["median"]) if hb.get("median") else None
        if self.kind == "tio2" or not common:
            self.note = "no growth law for this family (porous TiO2: no common law)"
            return
        if self.h_film is None:
            self.note = f"{self.system}: no film onset in the reference"
            return
        for key, k1, kp in LAWS:
            a, p = row.get(k1) or {}, row.get(kp) or {}
            if min(a.get("n", 0), p.get("n", 0)) < self.min_scans or common.get(key) is None:
                continue
            la = math.log(a["median"])
            sa = max(0.5 * math.log(max(a["max"], 1e-9) / max(a["min"], 1e-9)), SA_FLOOR)
            sp = max(0.5 * (p["max"] - p["min"]), SP_FLOOR)
            self.laws.append(dict(quantity=key, name=common[key], ln_v1=la, p=float(p["median"]), s_a=sa, s_p=sp,
                                  n=int(min(a["n"], p["n"]))))
        if not self.laws:
            self.note = f"{self.system}: no law from {self.min_scans} or more scans"

    @property
    def active(self) -> bool:
        return bool(self.laws) and self.weight > 0

    def bind(self, rnames: list) -> "GrowthPrior":
        "The index of each law's quantity among the family's reported quantities (laws without one are dropped)."
        self.laws = [dict(law, q=rnames.index(law["name"])) for law in self.laws if law["name"] in rnames]
        return self

    def gate(self, h: float, after_stop: bool = False) -> float:
        if after_stop or not self.active or not (h > 0):
            return 0.0
        return 1.0 / (1.0 + (h / self.h_film) ** GATE_POWER)

    def residuals(self, h: float, rep, after_stop: bool = False) -> np.ndarray:
        'One row per law (0 where the window is not under the prior).'
        g = self.gate(h, after_stop)
        out = np.zeros(len(self.laws))
        if g <= 0:
            return out
        lh = math.log(h)
        for i, law in enumerate(self.laws):
            v = float(rep[law["q"]])
            if np.isfinite(v):
                sig = math.sqrt(law["s_a"] ** 2 + (lh * law["s_p"]) ** 2)
                out[i] = math.sqrt(self.weight * g) * (v - law["ln_v1"] - law["p"] * lh) / sig
        return out

    def scales(self, h: float, after_stop: bool = False) -> np.ndarray:
        'd residual / d quantity per law (the gate and sigma held fixed): sqrt(weight g) / sigma (0 off the prior).'
        g = self.gate(h, after_stop)
        if g <= 0:
            return np.zeros(len(self.laws))
        lh = math.log(h)
        return np.array([math.sqrt(self.weight * g) / math.sqrt(law["s_a"] ** 2 + (lh * law["s_p"]) ** 2)
                         for law in self.laws])

    def dlnh(self, h: float, rep, after_stop: bool = False) -> np.ndarray:
        'd residual / d ln h per law with the quantity held fixed (2026-11-20, after the Jacobian check of Maxwell'
        g = self.gate(h, after_stop)
        out = np.zeros(len(self.laws))
        if g <= 0:
            return out
        lh = math.log(h)
        r = self.residuals(h, rep, after_stop)
        for i, law in enumerate(self.laws):
            s2 = law["s_a"] ** 2 + (lh * law["s_p"]) ** 2
            out[i] = (-math.sqrt(self.weight * g) * law["p"] / math.sqrt(s2)
                      + r[i] * (-0.5 * GATE_POWER * (1.0 - g) - lh * law["s_p"] ** 2 / s2))
        return out

    def cost(self, h: float, rep, after_stop: bool = False) -> float:
        r = self.residuals(h, rep, after_stop)
        return float(r @ r)

    def describe(self) -> dict:
        return dict(system=self.system, scans=self.n_scans, weight=self.weight, h_film=self.h_film, note=self.note,
                    laws=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in law.items()} for law in self.laws],
                    sa_floor=SA_FLOOR, sp_floor=SP_FLOOR, gate_power=GATE_POWER, min_scans=self.min_scans)


def reference_from_paths(paths_json, exclude=(), keys=("h_cov50", "h_base", "D1", "p_D", "R1", "p_R", "D_film")) -> dict:
    ''
    res = json.loads(Path(paths_json).read_text(encoding="utf-8"))
    groups = {}
    for scan, entry in res.items():
        if scan in set(exclude):
            continue
        groups.setdefault(f"{entry['metal']}/{entry['substrate']}", []).append(entry["descriptors"])
    table = {}
    for system, descs in groups.items():
        row = dict(system=system, n=len(descs))
        for key in keys:
            v = [d[key] for d in descs if d.get(key) is not None]
            if v:
                row[key] = dict(median=float(np.median(v)), min=float(min(v)), max=float(max(v)), n=len(v))
        table[system] = row
    return table
