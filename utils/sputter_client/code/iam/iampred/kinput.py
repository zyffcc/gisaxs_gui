'Network inputs of the any-wavelength families (2026-11-21, results/iam/anywave_20261121): the image on a fixed k_fz grid'

from __future__ import annotations

import numpy as np

from . import features as FE

GEOMETRY_SCALARS = ("k0", "kiz")


def is_kfz(spec) -> bool:
    return (spec or {}).get("axis") == "kfz"


def kfz_edges(spec) -> np.ndarray:
    lo, hi = (float(v) for v in spec["kfz"])
    return np.linspace(lo, hi, int(spec.get("rows", 58)) + 1)


def overlap(src, dst) -> np.ndarray:
    'W (n_dst, n_src): the share of source interval j inside destination interval i (both edge arrays increasing).'
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    lo = np.maximum(dst[:-1, None], src[None, :-1])
    hi = np.minimum(dst[1:, None], src[None, 1:])
    return np.clip(hi - lo, 0.0, None) / np.diff(src)[None, :]


def _k0():
    import iam as M
    return float(M.K0)


def image(s, w, Y, spec, pre_scale=1.0):
    if not is_kfz(spec):
        return FE.image(s, w, Y, spec, pre_scale)
    ok = w["mask"]
    E = np.where(ok, w["E"], 0.0).sum(0)
    Ys = np.where(ok, Y, 0.0).sum(0)
    P = np.where(ok, pre_scale * w["cfac"] * s.I_pre * w["E"], 0.0).sum(0)
    W = overlap(_k0() * np.sin(np.radians(np.asarray(s.grid.native_edges, float))), kfz_edges(spec))
    E2, Y2, P2 = W @ E, W @ Ys, W @ P
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
    return FE.scalar_names(cfg)


def scalars(cfg, s, t, hm_nominal):
    if not set(GEOMETRY_SCALARS) & set(cfg["scalars"]):
        return FE.scalars(cfg, s, t, hm_nominal)
    k0 = _k0()
    v = []
    for nm in cfg["scalars"]:
        if nm == "k0":
            v.append(k0)
        elif nm == "kiz":
            v.append(k0 * np.sin(np.radians(float(s.cfg.get("ai", 0.40)))))
        else:
            v += list(FE.scalars(dict(cfg, scalars=[nm]), s, t, hm_nominal))
    return np.array(v, np.float32)
