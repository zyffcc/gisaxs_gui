'Main-screen data of a predicted scan for the GUI (posterior model v1, 2026-11-13): for every window the candidates'

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from . import live as LV
from . import registry as REG
from . import runner

WANTED = [("R_nm", "grain radius R", "nm"), ("H_nm", "grain height H", "nm"), ("D_nm", "centre distance D", "nm"),
          ("gap_nm", "gap between grains D - 2R", "nm"), ("width", "width of the radius distribution (log-normal)", ""),
          ("coverage", "coverage", ""), ("amount_nm", "metal amount", "nm"),

          ("f_islands", "share of the metal in resolvable islands", ""),
          ("cap_share", "share of the optically active metal in the cap layer on the TiO2 particles", ""),
          ("depth_nm", "vertical correlation length of the metal network in the pores", "nm"),
          ("xi_nm", "lateral correlation length of the metal network in the pores", "nm"),
          ("film_share", "share of the layer scattering that leaves through the metal film", "")]
POROUS = dict(f_islands="f_isl", cap_share="cap", depth_nm="sigma_z", xi_nm="xi", film_share="w_film")


SOURCES = dict(R_nm=("lnR", "lnR_mass", "lnR_num", "lnR_rel"), D_nm=("lnD", "lnD_mass", "lnD_num"), H_nm=("lnT", "lnH_top"))


def sources(rnames):
    return {k: next(n for n in v if n in rnames) for k, v in SOURCES.items() if any(n in rnames for n in v)}


def sources_text(rnames):
    'sources() for main.json: the height of the polymer / Cu families is the larger of the top height (islands, or'
    src = sources(rnames)
    if src.get("H_nm") == "lnH_top" and "lnt_film" in rnames:
        src["H_nm"] = "larger of lnH_top and lnt_film"
    elif src.get("H_nm") == "lnT":
        src["H_nm"] = "lnT plus the thickness of the base layer"
    src.update({k: v for k, v in POROUS.items() if v in rnames})
    return src


def porous_geometry(p, h_m, morph=None):
    'Where the metal of a solution in a porous TiO2 layer sits (families tio2_*; iam_fit_t / iam2.predict_tio2),'
    if "lnfisl" not in p or "lnkopt" not in p:
        return {}
    try:
        import iam as M
        particle = float(M.R_PARTICLE)
    except (ImportError, AttributeError):
        particle = 75.0
    h_opt = float(np.exp(p["lnkopt"])) * float(h_m)
    kc = float(p.get("kcap", 0.0))
    out = dict(particle_nm=particle, island_amount_nm=float(np.exp(p["lnfisl"])) * float(h_m), cap_amount_nm=kc * h_opt,
               pore_amount_nm=(1.0 - kc) * h_opt)
    if kc > 0:
        c = float(np.exp(p["lncp"])) if "lncp" in p else 1.0
        out.update(cap_cover=c, cap_thickness_nm=kc * h_opt / (c * max(float(p.get("phicap", 1.0)), 1e-6)))
    if "z0" in p:
        out.update(pore_depth_nm=float(p["z0"]), pore_width_nm=float(np.exp(p["lnwb"])) if "lnwb" in p else 0.0)
    if "lnR" in p and "lna" in p:
        out["cluster_height_nm"] = float(np.exp(p["lnR"] + p["lna"]))
    if morph is not None:
        for key, name in (("cluster_distance_nm", "D"), ("cluster_cover", "coverage")):
            try:
                out[key] = float(getattr(morph, name))
            except (AttributeError, TypeError, ValueError):
                pass
    return out


def wanted(pnames, rnames, x, reported, h_m, t_b=0.0, morph=None):
    p = dict(zip(pnames, x))
    r = dict(zip(rnames, reported))
    out = {}
    src = sources(rnames)
    if "R_nm" in src or "lnR" in p:
        out["R_nm"] = float(np.exp(r[src["R_nm"]] if "R_nm" in src else p["lnR"]))
    if "H_nm" in src:
        out["H_nm"] = float(np.exp(r[src["H_nm"]]))
        if src["H_nm"] == "lnH_top" and "lnt_film" in r:
            film = float(np.exp(r["lnt_film"]))
            out["base_nm"] = max(film - out["H_nm"], 0.0)
            out["H_nm"] = max(out["H_nm"], film)
        elif src["H_nm"] == "lnT":
            out["base_nm"] = float(t_b or 0.0)
            out["H_nm"] += out["base_nm"]
    if "D_nm" in src:
        out["D_nm"] = float(np.exp(r[src["D_nm"]]))
    if "R_nm" in out and "D_nm" in out:
        out["gap_nm"] = float(max(out["D_nm"] - 2.0 * out["R_nm"], 0.0))
    if "s" in p:
        out["width"] = float(p["s"])
    if "cov" in r:
        out["coverage"] = float(r["cov"])
    for key, name in POROUS.items():
        if name in r:
            out[key] = float(r[name])
    if h_m is not None:
        out["amount_nm"] = float(h_m)
        out.update(porous_geometry(p, h_m, morph))
    return {k: (round(v, 4) if np.isfinite(v) else None) for k, v in out.items()}


SMOOTH_KEYS = ("R_nm", "D_nm", "H_nm")


def smooth_path(wins, lam=1.0):
    "Sets w['smooth'] (an entry label) in every window with entries: the sequence that minimises"
    W = [w for w in wins if w.get("entries")]
    if not W:
        return

    def jump(a, b):
        d = sum(abs(math.log(max(a[k], 1e-3) / max(b[k], 1e-3))) for k in SMOOTH_KEYS if a.get(k) is not None and b.get(k) is not None)
        if a.get("coverage") is not None and b.get("coverage") is not None:
            d += 2.0 * abs(a["coverage"] - b["coverage"])
        return d

    own = [[e["fit_deviation"] / max(min(x["fit_deviation"] for x in w["entries"]), 1e-6) for e in w["entries"]] for w in W]
    cost, back = [list(own[0])], []
    for i in range(1, len(W)):
        row, brow = [], []
        for j, e in enumerate(W[i]["entries"]):
            opts = [cost[-1][m] + lam * jump(p_, e) for m, p_ in enumerate(W[i - 1]["entries"])]
            m = min(range(len(opts)), key=opts.__getitem__)
            row.append(opts[m] + own[i][j]); brow.append(m)
        cost.append(row); back.append(brow)
    j = min(range(len(cost[-1])), key=cost[-1].__getitem__)
    for i in range(len(W) - 1, -1, -1):
        W[i]["smooth"] = W[i]["entries"][j]["label"]
        if i:
            j = back[i - 1][j]


def _line(wq):
    f = lambda k, fmt: (fmt.format(wq[k]) if wq.get(k) is not None else "-")
    return (f"R {f('R_nm', '{:.1f}')}  H {f('H_nm', '{:.1f}')}  D {f('D_nm', '{:.1f}')} nm\n"
            f"gap {f('gap_nm', '{:.1f}')} nm  width {f('width', '{:.2f}')}  cov {f('coverage', '{:.2f}')}")


def _windows(args):
    cfg, scan, rows, k, outdir, info = args
    from . import adapter as A
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = A.get_scan(scan)
    af, q = np.asarray(s.af, float), np.asarray(s.q, float)
    pn, rn = info["pnames"], [r["name"] for r in cfg["reported"]]
    out = []
    for row in rows:
        t = float(row["t"])
        try:
            ctx = A.Context(cfg, s, t)
            ok = ctx.w["mask"]
            E = np.where(ok, ctx.w["E"], 0.0)
            Es = E.sum(0)
            good = Es > 0
            meas = np.where(good, np.where(ok, ctx.w["Y"], 0.0).sum(0) / np.maximum(Es, 1e-30), np.nan)
            entries = []
            for j, c in enumerate([c for c in row.get("cands", []) if "x" in c][:k], 1):
                entries.append((f"candidate {j}", "candidate", j, c))
            if "refined" in row:
                entries.append(("refined", "refined", None, row["refined"]))
            imgs, recs = [], []
            for label, kind, rank, c in entries:
                (mu, _, _), m, p, h = ctx.render(np.asarray(c["x"], float))
                d5, _ = ctx.dev5(mu)
                imgs.append(np.where(good, (np.where(ok, mu, 0.0) * E).sum(0) / np.maximum(Es, 1e-30), np.nan))
                recs.append(dict(label=label, kind=kind, rank=rank, weight=(round(float(c["weight"]), 4) if "weight" in c else None),
                                 fit_deviation=round(float(d5), 3),
                                 **wanted(pn, rn, c["x"], ctx.reported(m, p, h), h, getattr(m, "t_b", 0.0), m)))
            if not recs:
                raise ValueError("no candidate to show")
            pos = meas[np.isfinite(meas) & (meas > 0)]
            vmin, vmax = (np.log10(np.percentile(pos, 2)), np.log10(np.percentile(pos, 99.8))) if pos.size else (-3, 0)
            n = 1 + len(imgs)
            fig, ax = plt.subplots(1, n, figsize=(3.3 * n + 0.6, 3.9), sharey=True)
            ax = np.atleast_1d(ax)
            panels = [("measured", meas, "")] + [(f"{r_['label']}  (fit deviation {r_['fit_deviation']:.1f})", im, _line(r_)) for r_, im in zip(recs, imgs)]
            for a_, (title, im, text) in zip(ax, panels):
                with np.errstate(divide="ignore", invalid="ignore"):
                    pm = a_.pcolormesh(q, af, np.log10(np.where(im > 0, im, np.nan)), vmin=vmin, vmax=vmax, cmap="viridis", shading="auto")
                a_.set_xscale("log")
                a_.set_title(title, fontsize=8)
                a_.set_xlabel("q_y (nm$^{-1}$)" + ("\n" + text if text else ""), fontsize=8)
                a_.tick_params(labelsize=7)
            ax[0].set_ylabel("exit angle (deg)", fontsize=8)
            fig.colorbar(pm, ax=list(ax), fraction=0.025, pad=0.01).set_label("log10 count rate", fontsize=7)
            fig.suptitle(f"{scan}   t {t:g} s   nominal deposit {ctx.h_nom_ctx:.2f} nm", fontsize=9, x=0.02, ha="left")
            name = f"w_{t:g}.png"
            fig.savefig(Path(outdir) / name, dpi=100, bbox_inches="tight")
            plt.close(fig)
            np.savez_compressed(Path(outdir) / f"w_{t:g}.npz", q=q.astype(np.float32), af=af.astype(np.float32),
                                measured=meas.astype(np.float32), models=np.asarray(imgs, np.float32),
                                labels=np.array([r_["label"] for r_ in recs]))
            best = min(range(len(recs)), key=lambda i: recs[i]["fit_deviation"])
            out.append(dict(t=t, t_rel=round(float(t - s.t_dep), 3), h_nom=round(float(ctx.h_nom_ctx), 4), image=name, arrays=f"w_{t:g}.npz",
                            entries=recs, pick=recs[best]["label"]))
        except (Exception, SystemExit) as e:
            out.append(dict(t=t, error=f"{type(e).__name__}: {str(e)[:160]}"))
        LV.emit("drawn", t=t, entries=len(out[-1].get("entries") or ()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--data", default=None)
    ap.add_argument("--shadow", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--every", type=int, default=1, help="every n-th window (1 = all)")
    a = ap.parse_args()
    out = Path(a.outdir)
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    cfg = REG.load(summ["family"])
    scan = summ["scan"]
    if scan not in cfg["scans"]:
        import types
        from . import predict as PR
        site = runner.site_env(cfg, a.data, a.shadow)
        cfg["scans"][scan], _ = PR._client_entry(cfg, types.SimpleNamespace(scan=scan, like=summ.get("like")),
                                                  Path(site.get("IAM_DATA", ".")))
    with gzip.open(out / "state.json.gz", "rt", encoding="utf-8") as fh:
        st = json.load(fh)
    rows = [r for r in st["rows"] if r.get("cands")][:: max(1, a.every)]
    path = {}
    if (out / "windows.csv").exists():
        for L in csv.DictReader(open(out / "windows.csv", encoding="utf-8")):
            try:
                path[round(float(L["t"]), 3)] = L.get("path_source") or None
            except (KeyError, ValueError):
                pass
    main_dir = out / "main"
    main_dir.mkdir(exist_ok=True)
    LV.emit("drawing", total=len(rows))
    nw = max(1, min(a.workers, len(rows)))
    chunks = [rows[i::nw] for i in range(nw)]
    res = runner.run(cfg, [(cfg, scan, c, a.k, str(main_dir), st["info"]) for c in chunks if c], _windows, nw, data=a.data, shadow=a.shadow)
    if any(c is None for c in res):
        raise SystemExit(f"{scan}: not a scan of family {cfg['family']} (no forward settings for it)")
    wins = sorted((w for c in res for w in c), key=lambda w: w["t"])
    for w in wins:
        w["path"] = path.get(round(w["t"], 3))
    smooth_path(wins)
    doc = dict(scan=scan, family=summ["family"], metal=st["info"].get("metal"), model=summ.get("model"),
               wanted=[dict(key=k_, label=l_, unit=u_) for k_, l_, u_ in WANTED],
               sources=sources_text([r["name"] for r in cfg["reported"]]),
               note="development reference; fit deviation = deviance per cell with a 5 % model-error floor (lower is better)",
               choice="per window: 'pick' = lowest fit deviation, 'smooth' = smooth evolution over the scan (default to show), "
                      "'path' = the source of the time path of iampred.predict",
               windows=wins)
    (main_dir / "main.json").write_bytes((json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    ok = [w for w in wins if "entries" in w]
    print(f"{scan}: {len(ok)} of {len(wins)} windows written to {main_dir} ({sum(len(w['entries']) for w in ok)} model images)")


if __name__ == "__main__":
    main()
