'Measured against simulated GISAXS patterns for any set of stored solutions (2026-11-04): per window the measured and'

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iam_paths as IAMP

YONEDA = (0.15, 0.30)
QBAND = (0.20, 0.50)
SUBNAME = {"Si": "Si", "TiO2": "TiO₂", "PS": "PS", "PNO": "PNO", "PSBMA": "PSBMA"}


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    title = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else "测量与模拟的 GISAXS 图样"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    import iam as M
    scans, cols = {}, []
    for e in spec:
        sc = e["scan"]
        if sc not in scans:
            os.environ["IAM_ONSET"] = e.get("onset") or ("override" if M.SCANS[sc]["sub"] == "TiO2" else "toolkit")
            for k in list(scans):
                del scans[k]
            scans[sc] = M.Scan(sc)
        s = scans[sc]
        w = s.window(float(e["t"]), half=e.get("half"))
        mu = np.load(e["mu"])["mu"].astype(float)
        ok = w["mask"]
        cols.append(dict(e=e, metal=s.metal, sub=s.sub, q=s.q, af=s.af, Y=np.where(ok, w["Y"], 0).sum(0),
                         E=np.where(ok, w["E"], 0).sum(0), F=np.where(ok, mu * w["E"], 0).sum(0)))
    n = len(cols)
    fig, ax = plt.subplots(5, n, figsize=(2.9 * n, 14.0), gridspec_kw=dict(height_ratios=[1, 1, 1, 0.8, 0.8]), squeeze=False)
    im3 = None
    for j, c in enumerate(cols):
        q, af, Y, E, F, e = c["q"], c["af"], c["Y"], c["E"], c["F"], c["e"]
        good = E > 0
        rd = np.where(good, Y / np.maximum(E, 1e-12), np.nan)
        rm = np.where(good, F / np.maximum(E, 1e-12), np.nan)
        lo, hi = np.nanpercentile(np.log10(np.maximum(rm, 1e-12)), [2, 99.8])
        ext = [q[0], q[-1], af[0], af[-1]]
        for i, A in enumerate((rd, rm)):
            ax[i, j].imshow(np.log10(np.maximum(A, 10 ** (lo - 1))), origin="lower", aspect="auto", extent=ext,
                            vmin=lo, vmax=hi, cmap="viridis")
            ax[i, j].set_ylim(0, 1.0)
        Ys, Fs = uniform_filter(Y, 3), uniform_filter(F, 3)
        lr = np.where(good & (Fs > 0), np.log(np.maximum(Ys, 0.5) / np.maximum(Fs, 1e-9)), np.nan)
        im3 = ax[2, j].imshow(lr, origin="lower", aspect="auto", extent=ext, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
        ax[2, j].set_ylim(0, 1.0)
        sa = (af >= YONEDA[0]) & (af <= YONEDA[1])
        e1 = E[sa].sum(0); m1 = e1 > 0.5 * np.median(e1[e1 > 0])
        ax[3, j].loglog(q[m1], Y[sa].sum(0)[m1] / e1[m1], "k.", ms=3, label="实测")
        ax[3, j].loglog(q[m1], F[sa].sum(0)[m1] / e1[m1], "-", color="#d6402b", lw=1.3, label="模拟")
        ax[3, j].set_xlabel("q_y (nm⁻¹)", fontsize=8)
        sq = (q >= QBAND[0]) & (q <= QBAND[1])
        e2 = E[:, sq].sum(1); m2 = (e2 > 0.5 * np.median(e2[e2 > 0])) & (af <= 1.0)
        ax[4, j].semilogy(af[m2], Y[:, sq].sum(1)[m2] / e2[m2], "k.", ms=3, label="实测")
        ax[4, j].semilogy(af[m2], F[:, sq].sum(1)[m2] / e2[m2], "-", color="#d6402b", lw=1.3, label="模拟")
        ax[4, j].set_xlabel("α_f (°)", fontsize=8)
        for a in ax[:, j]:
            a.tick_params(labelsize=7)
        d5 = f"dev5 {e['dev5']:.1f}" if e.get("dev5") is not None else ""
        ax[0, j].set_title(f"{e['scan']} {c['metal']}/{SUBNAME.get(c['sub'], c['sub'])} {float(e['t']):g} s\n{e.get('label', '')}\n{d5}",
                           fontsize=8)
    for i, lab in enumerate(["实测 log10(计数/曝光)\nα_f (°)", "模拟（同色标）\nα_f (°)", "ln(实测/模拟)\nα_f (°)",
                             f"横切 α_f {YONEDA[0]}–{YONEDA[1]}°", f"纵切 q_y {QBAND[0]}–{QBAND[1]} nm⁻¹"]):
        ax[i, 0].set_ylabel(lab, fontsize=8)
    ax[3, 0].legend(fontsize=7, frameon=False)
    if im3 is not None:
        fig.colorbar(im3, ax=ax[2, -1], fraction=0.08)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=75)
    print(out)


if __name__ == "__main__":
    main()
