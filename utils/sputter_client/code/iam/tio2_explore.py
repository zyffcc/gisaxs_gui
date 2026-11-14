'In-situ anchored forward model, step 0: what does the deposited metal change on the sprayed TiO2 NP layer?'

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import iam_paths as IAMP

import numpy as np

CODE = IAMP.GIMAP
sys.path.insert(0, str(CODE))
from utils.sputter_forward.data import Result
from utils.sputter_forward.grid import native_grid

HERE = Path(__file__).resolve().parent
QB = [(0.05, 0.12), (0.12, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.8)]
AB = [(0.05, 0.15), (0.15, 0.25), (0.25, 0.40), (0.40, 0.70), (0.70, 1.20)]


def onset(res, grid, base=(0.3, 1.0)):
    q, af = grid.q, grid.native_af
    sel = ((af >= 0.10) & (af < 0.35))[None, :, None] & ((q >= 0.15) & (q < 1.2))[None, None, :]
    rate = np.array([(res.Y[i] * sel).sum() / max((res.E[i] * sel).sum(), 1e-12) for i in range(len(res.time))])
    b = (res.time >= base[0]) & (res.time <= base[1])
    r0 = float(np.median(rate[b]))
    above = rate > 1.05 * r0
    for i in range(len(rate) - 4):
        if res.time[i] > base[1] and above[i:i + 4].all():
            return float(res.time[i]), r0, rate
    return None, r0, rate


def window_rate(res, lo, hi):
    m = (res.time >= lo) & (res.time < hi)
    Y, E = res.Y[m].sum(0), res.E[m].sum(0)
    return Y, E


def main():
    name, label = sys.argv[1], sys.argv[2]
    times = [float(x) for x in sys.argv[3:]] or [2, 5, 10, 20, 40, 80, 140]
    folder = Path(name) if Path(name).exists() else Path(os.environ["APPDATA"]) / "GIMaP" / "sputter" / name
    grid = native_grid()
    res = Result(folder, grid)
    q, af = grid.q, grid.native_af
    t_dep, r0, rate = onset(res, grid)
    pre_hi = (t_dep or 1.0) - 0.05
    Yp, Ep = window_rate(res, 0.3, pre_hi)
    out = dict(folder=str(folder), t_dep=t_dep, baseline_rate=r0, pre_window=[0.3, pre_hi], q=q.tolist(), af=af.tolist(),
               pre_counts=float(Yp.sum()), windows=[])
    Rp = np.where(Ep > 0, Yp / np.maximum(Ep, 1e-12), np.nan)

    def vprof(Y, E):
        rows = []
        for lo, hi in QB:
            c = (q >= lo) & (q < hi)
            y = Y[:, :, c].sum((0, 2)); e = E[:, :, c].sum((0, 2))
            rows.append(np.where(e > 0, y / np.maximum(e, 1e-12), np.nan))
        return np.array(rows)

    def hprof(Y, E):
        rows = []
        for lo, hi in AB:
            c = (af >= lo) & (af < hi)
            y = Y[:, c, :].sum((0, 1)); e = E[:, c, :].sum((0, 1))
            rows.append(np.where(e > 0, y / np.maximum(e, 1e-12), np.nan))
        return np.array(rows)
    vp0, hp0 = vprof(Yp, Ep), hprof(Yp, Ep)
    out["pre_vprof"], out["pre_hprof"] = vp0.tolist(), hp0.tolist()
    for t in times:
        Y, E = window_rate(res, t - 0.5, t + 0.5)
        if E.sum() <= 0:
            continue
        vp, hp = vprof(Y, E), hprof(Y, E)
        out["windows"].append(dict(t=t, counts=float(Y.sum()), vprof=vp.tolist(), hprof=hp.tolist(),
                                   vratio=(vp / vp0).tolist(), hratio=(hp / hp0).tolist(),
                                   total_ratio=float((Y.sum() / E.sum()) / (Yp.sum() / Ep.sum()))))
    ((IAMP.WORK / "diag")).mkdir(exist_ok=True)
    ((IAMP.WORK / "diag") / f"tio2_{label}.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W = out["windows"]
    fig, ax = plt.subplots(3, len(QB), figsize=(4 * len(QB), 11))
    cm = plt.cm.viridis(np.linspace(0, 1, len(W)))
    for b, (lo, hi) in enumerate(QB):
        a = ax[0, b]; a.semilogy(af, vp0[b], "k", lw=2, label="pre")
        for w, c in zip(W, cm):
            a.semilogy(af, np.array(w["vprof"])[b], color=c, lw=1, label=f"{w['t']:g}s")
        a.set_title(f"q {lo}-{hi} nm-1: vertical"); a.set_xlabel("alpha_f (deg)"); a.set_xlim(0, 1.3)
        a = ax[1, b]
        for w, c in zip(W, cm):
            a.plot(af, np.array(w["vratio"])[b], color=c, lw=1)
        a.axhline(1, color="k", lw=0.5); a.set_title("ratio to pre-state"); a.set_xlim(0, 1.3); a.set_yscale("log")
    for b, (lo, hi) in enumerate(AB):
        a = ax[2, b]; a.loglog(q, hp0[b], "k", lw=2)
        for w, c in zip(W, cm):
            a.loglog(q, np.array(w["hprof"])[b], color=c, lw=1)
        a.set_title(f"alpha_f {lo}-{hi}: lateral"); a.set_xlabel("q_y (nm-1)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{label}: onset {t_dep} s, pre-state window {out['pre_window']}")
    fig.tight_layout(); fig.savefig((IAMP.WORK / "diag") / f"tio2_{label}.png", dpi=80)
    print(json.dumps(dict(label=label, t_dep=t_dep, pre_counts=out["pre_counts"],
                          total_ratio=[(w["t"], round(w["total_ratio"], 3)) for w in W])))


if __name__ == "__main__":
    main()
