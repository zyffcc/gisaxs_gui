'Network inputs, computed by the same code for simulated and measured windows.'

from __future__ import annotations

import numpy as np


def image(s, w, Y, spec, pre_scale=1.0):
    lo, hi = spec["af"]
    pool = int(spec.get("pool", 2))
    ok = w["mask"]
    E = np.where(ok, w["E"], 0.0).sum(0)
    Ys = np.where(ok, Y, 0.0).sum(0)
    P = np.where(ok, pre_scale * w["cfac"] * s.I_pre * w["E"], 0.0).sum(0)
    rows = (np.asarray(s.af) >= lo) & (np.asarray(s.af) <= hi)
    E, Ys, P = E[rows], Ys[rows], P[rows]
    n = (len(E) // pool) * pool

    def pl(A):
        return A[:n].reshape(n // pool, pool, -1).sum(1)

    E2, Y2, P2 = pl(E), pl(Ys), pl(P)
    pos = E2[E2 > 0]
    good = E2 > 0.3 * (np.median(pos) if pos.size else 1.0)
    fl = 1e-3 * np.median(P2[good] / E2[good]) if good.any() else 1e-6
    rate = np.where(good, Y2 / np.maximum(E2, 1e-12), 0.0)
    pre = np.where(good, P2 / np.maximum(E2, 1e-12), 0.0)
    ch = [np.where(good, np.log10(rate + fl), 0.0), np.where(good, np.log10(pre + fl), 0.0)]
    if spec.get("ratio"):
        ch.append(np.where(good, np.log10(rate + fl) - np.log10(pre + fl), 0.0))
    return np.stack(ch + [good.astype(float)]).astype(np.float32)


def scalar_names(cfg):
    out = []
    for nm in cfg["scalars"]:
        out += [f"metal_{m}" for m in cfg["metals"]] if nm == "metal" else [nm]
    return out


def scalars(cfg, s, t, hm_nominal):
    v = []
    for nm in cfg["scalars"]:
        if nm == "ln_hm_nominal":
            v.append(np.log(max(hm_nominal, 0.02)))
        elif nm == "t_rel":
            v.append(float(t) - float(s.t_dep))
        elif nm == "metal":
            v += [float(str(s.metal) == m) for m in cfg["metals"]]
        else:
            raise ValueError(f"unknown scalar {nm}")
    return np.array(v, np.float32)
