"Read a GIMaP Sputter result folder (read only) on the toolkit's native grid, build the windows (= the fringe"

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

BAND = (0.05, 0.15)
"""q_y band of the fringe tool's vertical cuts (nm^-1)."""


class SourceError(ValueError):
    pass


def _checked(path: Path, expected) -> bytes:
    data = path.read_bytes()
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise SourceError(f"{path.name} does not match the SHA-256 recorded in run.json")
    return data


def _read_csv(path: Path) -> list:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


class Result:
    'One result folder: counts per bin, bin times, production features per bin and the geometry.'

    def __init__(self, folder, grid):
        self.folder = Path(folder)
        binned = self.folder / "binned"
        self.run = json.loads((binned / "run.json").read_text(encoding="utf-8"))
        self.geometry = dict(self.run.get("geometry") or {})
        self.metal = str(self.run.get("metal") or "")
        self.grid = grid
        Ys, Es, rows = [], [], []
        for chunk in self.run.get("chunks") or []:
            name = chunk["file"]
            data = _checked(binned / name, chunk.get("sha256"))
            meta = _checked(binned / (name[:-4] + ".json"), chunk.get("metadata_sha256"))
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                Ys.append(z["Y"].astype(np.float32)); Es.append(z["E"].astype(np.float32))
            rows += json.loads(meta.decode("utf-8"))["rows"]
        Y, E = np.concatenate(Ys), np.concatenate(Es)
        if Y.shape[1:] != (2, len(grid.native_af), len(grid.q)):
            raise SourceError(f"arrays {Y.shape} do not match the native grid")
        order = np.argsort([float(r.get("time_s", 0.0)) for r in rows], kind="stable")
        self.Y, self.E = Y[order], E[order]
        self.rows = [rows[i] for i in order]
        self.time = np.array([float(r["time_s"]) for r in self.rows])
        self.first = np.array([float(r.get("first_time_s", r["time_s"])) for r in self.rows])
        self.last = np.array([float(r.get("last_time_s", r["time_s"])) for r in self.rows])
        frames = {}
        path = binned / "frames.csv"
        if path.is_file():
            for r in _read_csv(path):
                frames[r.get("frame")] = r
        self.features = [frames.get(r.get("frame"), {}) for r in self.rows]
        off = {}
        sp = self.folder / "structure_physical.csv"
        if sp.is_file():
            for r in _read_csv(sp):
                off[r.get("frame")] = str(r.get("beam_off_sample") or "").strip().lower() in ("true", "1", "yes")
        self.beam_off = np.array([off.get(r.get("frame"), False) for r in self.rows], dtype=bool)
        """Windows the toolkit flags as possibly off the sample (left out, as in the fringe tool)."""

    def feature(self, idx, key) -> np.ndarray:
        return np.array([_num(self.features[i].get(key)) for i in idx])


def groups(res: Result) -> list:
    "The fringe tool's time groups (``sputter_fringes.detect.make_groups``) rebuilt from the counts (usable: band"
    qsel = (res.grid.q_edges[:-1] >= BAND[0] - 1e-9) & (res.grid.q_edges[1:] <= BAND[1] + 1e-9)
    usable = np.flatnonzero((res.Y[..., qsel].sum(axis=(1, 2, 3)) >= 1000.0) & ~res.beam_off)
    if not len(usable):
        return []
    start = float(res.time[usable[0]])
    keys = np.floor((res.time[usable] - start) / 1.0 + 1e-9).astype(int)
    out = []
    for k in np.unique(keys):
        mem = usable[keys == k]
        out.append(dict(index=len(out), members=mem, time=float(np.mean(res.time[mem])),
                        t_lo=float(res.first[mem].min()), t_hi=float(res.last[mem].max())))
    return out


def group_for(gs, t):
    lo = np.array([g["t_lo"] for g in gs])
    return gs[max(int(np.searchsorted(lo, t, side="right") - 1), 0)]


def shadow_mask(Y, E, Yt, Et, thr=0.35):
    'Cells darkened on one side only (beam-stop rod or arm, holder shadow): log2 mirror-cell ratio of the window'
    def lr(Yx, Ex, smooth):
        r = np.where(Ex > 0, Yx / np.maximum(Ex, 1e-12), np.nan)
        a, b = r[1], r[0]
        if smooth:
            def sm(x):
                v = np.nan_to_num(x); w = np.isfinite(x).astype(float)
                return uniform_filter(v, 3) / np.maximum(uniform_filter(w, 3), 1e-9)
            a, b = sm(a), sm(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log2(a / b)
    lw, lt = lr(Y, E, False), lr(Yt, Et, True)
    bad1 = (lw < -thr) | (lt < -thr)
    bad0 = (lw > thr) | (lt > thr)

    def dil(b):
        b = np.nan_to_num(b).astype(bool)
        return b | np.roll(b, 1, axis=1) | np.roll(b, -1, axis=1)
    return np.stack([dil(bad0), dil(bad1)])


def first_group_template(res: Result, gs):
    'Counts and exposure of the first group (support and shadow test of the badge protocol).'
    mem = gs[0]["members"]
    return res.Y[mem].astype(float).sum(0), res.E[mem].astype(float).sum(0)


def background_template(res: Result):
    'Background template: the earliest stable second of data extrapolated toward zero metal.'
    Y, E, t = res.Y, res.E, res.time
    af, q = res.grid.native_af, res.grid.q
    hi = ((af > 0.6)[:, None] & (q > 0.3)[None, :])[None]
    r_hi = np.array([(Y[i] * hi).sum() / max((E[i] * hi).sum(), 1e-9) for i in range(len(t))])
    tot = Y.reshape(len(t), -1).sum(1)
    i0 = int(np.argmax(tot >= 1e4))
    while i0 + 1 < len(t) and r_hi[i0] < 0.9 * r_hi[i0 + 1]:
        i0 += 1
    t0 = float(res.first[i0])
    sel = np.where((t >= t[i0]) & (t < t[i0] + 1.0))[0]
    Yt, Et = Y[sel].astype(float).sum(0), E[sel].astype(float).sum(0)
    num = uniform_filter(Yt, size=(1, 1, 3), mode="nearest"); den = uniform_filter(Et, size=(1, 1, 3), mode="nearest")
    rts = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    later = np.where((t >= t[i0]) & (t < t[i0] + 3.0))[0]
    kappa = np.ones_like(rts)
    regions = []
    for a0 in np.arange(0.0, 1.6, 0.1):
        for q0, q1 in ((0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.81)):
            m = ((af >= a0) & (af < a0 + 0.1))[:, None] & ((q >= q0) & (q < q1))[None, :]
            ys = np.array([(Y[i] * m[None]).sum() for i in later]); es = np.array([(E[i] * m[None]).sum() for i in later])
            if es.min() <= 0 or ys.sum() < 200:
                continue
            tb = t[later]
            blocks = np.floor((tb - tb[0]) / 0.5).astype(int)
            tm = np.array([tb[blocks == b].mean() for b in np.unique(blocks)])
            rm = np.array([ys[blocks == b].sum() / es[blocks == b].sum() for b in np.unique(blocks)])
            c = np.polyfit(tm - t0, rm, 1)
            r_win = Yt[:, m].sum() / max(Et[:, m].sum(), 1e-9)
            k = float(np.clip(np.polyval(c, 0.0) / r_win, 0.3, 1.0))
            kappa[:, m] = k
            regions.append(dict(af=[round(float(a0), 2), round(float(a0) + 0.1, 2)], q=[q0, q1], rise_over_3s=float(rm[-1] / rm[0]),
                                factor=k))
    info = dict(skipped_transient_bins=int(i0), stable_start_s=t0, template_window_s=[float(t[sel[0]]), float(t[sel[-1]])],
                counts=float(Yt.sum()), median_factor=float(np.median([r["factor"] for r in regions])) if regions else 1.0,
                regions=regions)
    return rts * kappa, info


class FringeReference:
    'Outputs of the fringe tool for the same result folder (fringe_heights.csv, fringe_summary.json).'

    def __init__(self, folder):
        folder = Path(folder)
        self.rows = _read_csv(folder / "fringe_heights.csv")
        self.summary = json.loads((folder / "fringe_summary.json").read_text(encoding="utf-8"))
        self.time = np.array([_num(r["time_s"]) for r in self.rows])
        meas = [r for r in self.rows if r.get("status") == "measured" and np.isfinite(_num(r.get("H_nm")))]
        self.meas_t = np.array([_num(r["time_s"]) for r in meas])
        self.meas_H = np.array([_num(r["H_nm"]) for r in meas])

    def row(self, group_time):
        i = int(np.argmin(abs(self.time - group_time)))
        if abs(self.time[i] - group_time) > 2e-3:
            raise SourceError("the fringe outputs do not belong to this result folder (group times differ)")
        return self.rows[i]

    def H_bounds_unmeasured(self, t, rate):
        'Bounds of H for a window whose group has no measured H.'
        tt, hh = self.meas_t, self.meas_H
        if not len(tt):
            return (0.3, max(6.0, 1.5 * rate * t + 1.0))
        if t < tt[0]:
            return (0.3, float(hh[0]))
        if t > tt[-1]:
            return (float(hh[-1]), float(hh[-1] + 1.5 * rate * (t - tt[-1]) + 0.5))
        h = float(np.interp(t, tt, hh))
        return (0.75 * h, 1.25 * h)


def tool_minima(text):
    import re
    return [(int(o) if o != "?" else 0, float(a)) for o, a in re.findall(r"n=(\d+|\?)@([0-9.]+)", text or "")]


def make_window(res: Result, gs, t_nominal, Yt, Et, fringes: FringeReference, exclude=None):
    'One registered window: summed counts and exposure of its group, the used cells and the fringe-tool row.'
    g = group_for(gs, t_nominal)
    mem = g["members"]
    Y = res.Y[mem].astype(np.float64).sum(0)
    E = res.E[mem].astype(np.float64).sum(0)
    qs, qse = res.feature(mem, "q_star_nm_inv"), res.feature(mem, "q_star_fit_se_nm_inv")
    qse = qse[np.isfinite(qs)]; qs = qs[np.isfinite(qs)]
    q_se = None
    if len(qs):
        fit = float(np.sqrt(np.nansum(qse ** 2)) / len(qs)) if np.isfinite(qse).any() else 0.0
        scatter = float(np.std(qs, ddof=1) / np.sqrt(len(qs))) if len(qs) > 1 else 0.0
        q_se = max(fit, scatter) or None
    yon = res.feature(mem, "Yoneda_deg"); yon = yon[np.isfinite(yon)]
    row = fringes.row(g["time"])
    sh = shadow_mask(Y, E, Yt, Et)
    mask = (E > 0) & (Et > 0) & (res.grid.native_af >= 0.03)[None, :, None] & ~sh
    n_excl = 0
    if exclude is not None:
        n_excl = int((mask & exclude).sum())
        mask = mask & ~exclude
    H = _num(row.get("H_nm"))
    Hlo, Hhi = _num(row.get("H_lo68_nm")), _num(row.get("H_hi68_nm"))
    return dict(t_nominal=float(t_nominal), group=int(g["index"]), time=g["time"], t_lo=g["t_lo"], t_hi=g["t_hi"],
                members=[int(x) for x in mem], Y=Y, E=E, mask=mask,
                q_star_ref=float(qs.mean()) if len(qs) else None, q_star_se=q_se,
                Yoneda_ref=float(yon.mean()) if len(yon) else None,
                fringe_status=str(row.get("status")), fringe_H=H if np.isfinite(H) else None,
                fringe_minima_text=row.get("minima") or "", fringe_minima=tool_minima(row.get("minima") or ""),
                fringe_H_lo68=Hlo if np.isfinite(Hlo) else None, fringe_H_hi68=Hhi if np.isfinite(Hhi) else None,
                n_used=int(mask.sum()), n_shadow=int(((E > 0) & sh).sum()), n_excluded=n_excl)


__all__ = ["BAND", "FringeReference", "Result", "SourceError", "background_template", "first_group_template",
           "group_for", "groups", "make_window", "shadow_mask", "tool_minima"]
