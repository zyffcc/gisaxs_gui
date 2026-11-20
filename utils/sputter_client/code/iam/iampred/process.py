'Process analysis of a sputter scan: growth laws of the joint-fit path against the deposited amount (2026-11-20).'

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

COMMON = {
    "si": dict(R="lnR", D="lnD", cov="cov", base_mass="base_mass"),
    "polyag": dict(R="lnR_mass", D="lnD_mass", cov="cov", f_base="f_base", film="lnt_film"),
    "cu": dict(R="lnR_mass", D="lnD_mass", cov="cov", f_base="f_base", film="lnt_film"),
    "tio2": dict(R_rel="lnR_rel", cap="cap", film="w_film", f_isl="f_isl"),
}
LOGS = {"lnR", "lnD", "lnR_mass", "lnD_mass", "lnR_rel", "lnt_film"}
BASE_ONSET = 0.5
"""Share of the metal in the continuous layer that marks the film (half of the metal in the closed layer; the dense
underlayer of the model takes a few per cent of the metal from the first nanometre on, so a lower share is not a film)."""


def kind_of(family: str) -> str:
    return family.split("_")[0]


def growth_point(family: str, names: list, reported, h) -> dict:
    'The common quantities (COMMON; logs exponentiated, b = metal share in the continuous layer) of one solution'
    out = {"h": float(h)}
    for key, name in COMMON[kind_of(family)].items():
        if name in names:
            v = float(reported[names.index(name)])
            out[key] = math.exp(v) if name in LOGS and math.isfinite(v) else v
    if "base_mass" in out:
        out["b"] = out["base_mass"] / max(out["h"], 1e-9)
    elif "f_base" in out:
        out["b"] = out["f_base"]
    return {k: (round(v, 6) if math.isfinite(v) else None) for k, v in out.items()}


def path_from_points(t, points, dev5) -> dict:
    "{'t', 'h', 'dev5', quantity: (n,)} from per-window growth_point dicts (by time; a quantity missing in any"
    order = np.argsort(np.asarray(t, float))
    pts = [points[i] for i in order]
    out = dict(t=np.asarray(t, float)[order], dev5=np.asarray(dev5, float)[order])
    for key in pts[0] if pts else ():
        v = [p_.get(key) for p_ in pts]
        if all(x is not None for x in v):
            out[key] = np.array(v, float)
    return out


def path_from_state(state_path, family: str, names: list) -> dict:
    "{'t': (n,), 'h': (n,), 'dev5': (n,), quantity: (n,)} of the joint-fit path (windows with a solution, by time)."
    state = json.load(gzip.open(state_path, "rt", encoding="utf-8"))
    rows = [r for r in state["rows"] if r.get("seq")]
    return path_from_points([float(r["t"]) for r in rows],
                            [growth_point(family, names, r["seq"]["reported"], r["seq"]["h_m"]) for r in rows],
                            [float(r["seq"]["dev5"]) for r in rows])


def _first_crossing(h, v, level):
    'Amount at which v first reaches level (log-linear interpolation in h), None when it never does.'
    for i in range(len(h)):
        if v[i] >= level:
            if i == 0:
                return float(h[0])
            a, b = math.log(max(h[i - 1], 1e-6)), math.log(max(h[i], 1e-6))
            f = (level - v[i - 1]) / max(v[i] - v[i - 1], 1e-12)
            return float(math.exp(a + min(max(f, 0.0), 1.0) * (b - a)))
    return None


def _power(h, v, sel):
    '(exponent, value at h = 1 nm) of v = v1 h^p over the selected windows (least squares in ln h), or (None, None).'
    sel = sel & (h > 0) & (v > 0)
    if sel.sum() < 3 or np.ptp(np.log(h[sel])) < 0.3:
        return None, None
    p, c = np.polyfit(np.log(h[sel]), np.log(v[sel]), 1)
    return float(p), float(math.exp(c))


def _logistic(h, cov, sel):
    '(k, h50) of cov = 1 / (1 + (h50 / h)^k) over the selected windows (linearised: logit cov = k ln h - k ln h50).'
    sel = sel & (h > 0) & (cov > 0.02) & (cov < 0.98)
    if sel.sum() < 3:
        return None, None
    y = np.log(cov[sel] / (1 - cov[sel]))
    k, c = np.polyfit(np.log(h[sel]), y, 1)
    if k <= 0:
        return None, None
    return float(k), float(math.exp(-c / k))


def descriptors(path: dict, family: str) -> dict:
    h = path["h"]
    out = dict(n=len(h), h_max=float(h.max()) if len(h) else None,
               dev5_median=float(np.median(path["dev5"])) if len(h) else None)
    kind = kind_of(family)
    if kind == "tio2":
        for key in ("cap", "film"):
            if key in path and len(h):
                v = path[key]
                top = float(np.max(v))
                out[f"h_{key}50"] = _first_crossing(h, v, 0.5 * top) if top > 0 else None
                out[f"{key}_final"] = float(v[-1])
        if "R_rel" in path:
            out["p_Rrel"], out["Rrel1"] = _power(h, path["R_rel"], np.ones(len(h), bool))
        return out
    cov = path.get("cov")
    b = path.get("b", np.zeros(len(h)))
    out["h_cov50"] = _first_crossing(h, cov, 0.5) if cov is not None else None
    out["h_cov70"] = _first_crossing(h, cov, 0.7) if cov is not None else None
    out["h_base"] = _first_crossing(h, b, BASE_ONSET)
    open_film = h < (out["h_base"] if out["h_base"] is not None else np.inf)
    film = ~open_film
    if "D" in path and film.sum() >= 2:
        out["D_film"] = float(np.median(path["D"][film]))
        out["R_film"] = float(np.median(path["R"][film])) if "R" in path else None
    stop = np.flatnonzero(np.diff(h) < 1e-3 * max(float(h.max()), 1e-9))
    if len(stop) >= 2 and "D" in path:
        i0 = int(stop[0])
        t, D = path["t"], path["D"]
        span = math.log(max(t[-1], 1e-6) / max(t[i0], 1e-6))
        out["h_stop"] = float(h[i0])
        out["D_after_stop_per_decade"] = float(math.log(D[-1] / D[i0]) / span * math.log(10)) if span > 0 else None
    out["p_D"], out["D1"] = _power(h, path["D"], open_film) if "D" in path else (None, None)
    out["p_R"], out["R1"] = _power(h, path["R"], open_film) if "R" in path else (None, None)
    out["cov_k"], out["cov_h50"] = _logistic(h, cov, open_film) if cov is not None else (None, None)
    drops = 0
    for key, rel, absol in (("R", 0.10, None), ("D", 0.05, None), ("cov", None, 0.03)):
        if key not in path:
            continue
        v = path[key]
        for a, c in zip(v, v[1:]):
            drops += int((rel is not None and c < a * (1 - rel)) or (absol is not None and c < a - absol))
    out["drops"] = drops
    return out


STAGES = dict(islands="separate islands (nucleation and growth)", coalescence="coalescence (coverage above 0.5)",
              film="continuous film (half of the metal in the closed layer)",
              after="after deposition (metal amount constant)",
              tio2_clusters="metal clusters in the pores and on the cap layer", tio2_film="metal film")


def stages(path: dict, family: str) -> list:
    'Stage of every window of the path (time order). Sputtering only adds metal, so coverage and the share in the'
    n = len(path["h"])
    if not n:
        return []
    h = path["h"]
    stop = np.flatnonzero(np.diff(h) < 1e-3 * max(float(h.max()), 1e-9))
    i_stop = int(stop[0]) if len(stop) >= 2 else n
    out = []
    if kind_of(family) == "tio2":
        film = np.maximum.accumulate(path["film"]) if "film" in path else np.zeros(n)
        f50 = 0.5 * max(float(film.max()), 1e-12)
        for i in range(n):
            out.append("after" if i > i_stop else "tio2_film" if film[i] >= f50 and film.max() > 0 else "tio2_clusters")
        return out
    cov = np.maximum.accumulate(path["cov"]) if "cov" in path else np.zeros(n)
    b = np.maximum.accumulate(path["b"]) if "b" in path else np.zeros(n)
    for i in range(n):
        out.append("after" if i > i_stop else "film" if b[i] >= BASE_ONSET else "coalescence" if cov[i] >= 0.5 else "islands")
    return out


REFERENCE_KEYS = ("h_cov50", "h_base", "D1", "p_D", "R1", "p_R", "D_film", "h_cap50", "h_film50")


def reference(metal: str, substrate: str) -> "dict | None":
    'The range of the descriptors over the analysed scans of the same metal on the same substrate'
    ref = Path(__file__).with_name("process_reference.json")
    if not ref.exists():
        return None
    table = json.loads(ref.read_text(encoding="utf-8"))
    return table.get(f"{metal}/{substrate}")


def compare(desc: dict, ref: "dict | None") -> dict:
    "Per descriptor: 'within', 'below' or 'above' the system range (None without a range of at least 2 scans)."
    out = {}
    for key in REFERENCE_KEYS:
        v, r = desc.get(key), (ref or {}).get(key)
        if v is None or not r or (ref or {}).get("n", 0) < 2:
            continue
        slack = 1e-3 * max(abs(r["min"]), abs(r["max"]), 1e-9)
        out[key] = "below" if v < r["min"] - slack else "above" if v > r["max"] + slack else "within"
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__.split("\n\n")[-1].strip())
        return 2
    from . import registry as REG
    state, family = Path(argv[0]), argv[1]
    names = [r["name"] for r in REG.load(family)["reported"]]
    path = path_from_state(state, family, names)
    print(json.dumps(descriptors(path, family), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
