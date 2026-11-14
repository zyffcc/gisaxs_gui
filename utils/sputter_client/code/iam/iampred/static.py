'Static (ex-situ) GISAXS frames of an Au film: one command from the frames of a sample to its structure.'

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IAM = HERE.parent
MODEL = IAM.parent
GEOMETRY = dict(energy_ev=12000.0, sdd_mm=8100.0, pixel_mm=0.172, db_row=1378.0, db_col=852.0, sample_roll_deg=-0.515,
                alpha_i_deg=0.40)
MODES = dict(quick=dict(refine=0, max_frames=4), refine=dict(refine=8, max_frames=4), full=dict(refine=30, max_frames=0))
FAIR = 10.0


SAT_MAX, SAT_REL, SAT_MIN = 2000, 3.0, 300

QUANT = [("h_m_model", "金属等效厚度", "nm", "ln_hm", True), ("height", "膜的总高度（底层 + 晶粒）", "nm", "height", False),
         ("t_b", "底层厚度", "nm", "t_b", False), ("T", "晶粒高出底层的高度", "nm", "T", False),
         ("R", "晶粒半径", "nm", "ln_R", True), ("d_hex", "晶粒中心间距", "nm", "ln_D", True),
         ("eta", "晶粒覆盖率", "", "eta", False), ("rb", "底层相对密度", "", "rb", False)]
FILES = dict(report_md="REPORT.md", report_png="report.png", report_csv="report.csv", patterns_png="patterns.png",
             main_json="main/main.json")


def static_root(arg):
    'Root of the static caches (its own root: the pixel mask differs from the in-situ caches).'
    if arg:
        return Path(arg).resolve()
    if os.environ.get("IAM_STATIC_DATA"):
        return Path(os.environ["IAM_STATIC_DATA"]).resolve()
    sys.path.insert(0, str(IAM))
    import iam_paths as IAMP
    return IAMP.DATA.parent / (IAMP.DATA.name + "_static")


def use_root(root):
    'Point the IAM modules of this process (and of its children) at the static root.'
    os.environ["IAM_DATA"] = str(root)
    sys.path.insert(0, str(IAM))
    import iam_paths as IAMP
    importlib.reload(IAMP)
    return IAMP


def models_dir(arg):
    if arg:
        return Path(arg).resolve()
    if os.environ.get("IAMPRED_MODELS") and (Path(os.environ["IAMPRED_MODELS"]) / "static_nn_v1").exists():
        return Path(os.environ["IAMPRED_MODELS"]) / "static_nn_v1"
    return MODEL.parent / "dist" / "static_nn_v1" / "models"


def frame_list(spec):
    '(kind, base, [entries]) of --frames: a folder (all *.cbf below it), one file, or ZIP[::PREFIX].'
    z, _, prefix = str(spec).partition("::")
    p = Path(z)
    if p.suffix.lower() == ".zip" and p.is_file():
        with zipfile.ZipFile(p) as zf:
            mem = sorted(m for m in zf.namelist() if m.lower().endswith(".cbf") and m.startswith(prefix))
        return "zip", p, mem
    if p.is_dir():
        return "dir", p, sorted(str(f) for f in p.rglob("*.cbf"))
    if p.is_file():
        return "dir", p.parent, [str(p)]
    raise SystemExit(f"{spec}: no such frame folder, file or zip")


def _safe(task):
    'One frame in a worker (zip member or file); a file that is not a frame of this detector gives dict(error).'
    import static_intake as SI
    try:
        return SI._frame(task) if isinstance(task, tuple) else SI._file(task)
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {str(e)[:80]}")


def intake(a, root, label, spec, nominal, reference, note):
    import static_intake as SI
    from concurrent.futures import ProcessPoolExecutor
    kind, base, entries = frame_list(spec)
    if not entries:
        raise SystemExit(f"{spec}: no CBF frames")
    mask = Path(a.mask).resolve() if a.mask else SI.MASK
    msha = hashlib.sha256(mask.read_bytes()).hexdigest()
    marker = dict(file=mask.name, path=str(mask), sha256=msha)
    rm = root / "pixmask.json"
    if rm.exists():
        if json.loads(rm.read_text(encoding="utf-8"))["sha256"] != msha:
            raise SystemExit(f"{root} holds caches of another pixel mask: use another --data root")
    elif root.exists() and any(p.is_dir() for p in root.iterdir()):
        raise SystemExit(f"{root} holds caches without a mask marker: use a new, empty --data root")
    else:
        root.mkdir(parents=True, exist_ok=True)
        rm.write_text(json.dumps(marker, indent=1) + "\n", encoding="utf-8", newline="\n")
    if not (root / "native_grid.npz").exists():
        import shutil
        shutil.copyfile(HERE / "native_grid.npz", root / "native_grid.npz")
    geometry = dict(GEOMETRY, energy_ev=a.energy_ev, sdd_mm=a.sdd_mm, db_row=a.beam_row, db_col=a.beam_col, sample_roll_deg=a.roll)
    skipped = [e for e in entries if "giwaxs" in Path(e).name.lower()]
    entries = [e for e in entries if e not in skipped]
    with ProcessPoolExecutor(a.workers, initializer=SI._init, initargs=(geometry, str(base.parent) if kind == "zip" else None, str(mask))) as pool:
        res = list(pool.map(_safe, [(base.name, m) for m in entries] if kind == "zip" else entries))
    failed = [(e, r["error"]) for e, r in zip(entries, res) if "error" in r]
    entries, res = [e for e, r in zip(entries, res) if "error" not in r], [r for r in res if "error" not in r]
    if skipped or failed:
        print(f"      {label}: {len(skipped)} GIWAXS-named files not read, {len(failed)} files not readable as frames of this detector"
              + (f" (first: {Path(failed[0][0]).name}: {failed[0][1]})" if failed else ""), flush=True)
    if not res:
        raise SystemExit(f"{spec}: no readable GISAXS frame")
    frames = []
    sat_med = float(np.median([r["saturation_masked_pixels"] for r in res])) if len(res) >= 3 else float("inf")
    for e, r in zip(entries, res):
        sat = r["saturation_masked_pixels"]
        why = "no counts inside the mask" if not r["selected_counts"] > 0 else \
            f"{sat} saturated pixels: direct or specular beam not blocked" if sat > SAT_MAX or (sat > SAT_REL * sat_med and sat > SAT_MIN) else ""
        frames.append(dict(member=str(e), raw_scan=Path(e).stem, scan_folder=Path(e).parent.name,
                           state="excluded" if why else "usable", reason=why))
    if not any(f["state"] == "usable" for f in frames):
        raise SystemExit(f"{spec}: no usable frame ({frames[0]['reason']})")
    t = dict(sample=a.sample or label, substrate=a.substrate or ("Si" if reference == "analytic_si" else "bare substrate measured"),
             mode=None, nominal_nm=float(nominal), reference=reference, note=note, frames=frames)
    rows, total = SI.write_cache(root, label, t, res, geometry, marker, f"{spec} (frames read in place)",
                                 alpha_i_note="alpha_i_deg 0.40 is the binning coordinate; the model refines the incidence "
                                 "as eps = alpha_i - 0.40 (prior -0.017 +- 0.010 deg, bounds -0.06 ... +0.03)")
    print(f"      {label}: {len(rows)} frames, {sum(f['state'] == 'usable' for f in frames)} usable, {total:.3g} counts", flush=True)
    return len(rows)


def _readings(s, i, eps):
    'Direct readings of frame i (samples without reference: the substrate term is small against the metal signal).'
    from iampred import observe as OB
    import iam as M
    w = s.window(i)
    s0 = copy.copy(s)
    s0.I_pre = np.zeros_like(s.I_pre)
    out = {}
    try:
        k = OB.direct(s0, w, eps)
        out.update(L_knee=k["L_knee_nm"], knee_status=k["status"])
    except Exception as e:
        out.update(L_knee=None, knee_status=f"failed: {type(e).__name__}")
    try:
        dm = M.dc("Au")
        f = OB.fringe(s0, w, float(s.cfg["nominal_nm"]), 2.0 * float(np.real(dm)), float(M.K0), eps)
        out.update(d_fringe=f["d_nm"] if f["status"] == "fringe" else None, fringe_status=f["status"])
    except Exception as e:
        out.update(d_fringe=None, fringe_status=f"failed: {type(e).__name__}")
    return out


def _sample(out, label):
    'Summary of one predicted sample: values, ranges and sources, plus what the figures need.'
    import iam_static as IS
    rows = list(csv.DictReader(open(out / label / "frames.csv", encoding="utf-8", newline="")))
    rows = [r for r in rows if r["label"] == label]
    s = IS.StaticScan(label)
    recs = {int(r["frame"]): json.loads((out / label / "static_fits" / label / f"{label}_{int(r['frame']):03d}.json").read_text(encoding="utf-8"))
            for r in rows}
    dev = np.array([float(r["dev5_refined"]) for r in rows])
    fit = float(np.median(dev))
    src = "model" if fit <= FAIR else "estimate"
    res = dict(label=label, sample=s.run.get("sample"), substrate=s.run.get("substrate"), nominal_nm=float(s.cfg["nominal_nm"]),
               reference=s.ref, frames=len(rows), frames_imported=len(s.states), fit=fit, fit_network=float(np.median([float(r["dev5_direct"]) for r in rows])),
               refine=int(json.loads((out / label / "summary.json").read_text(encoding="utf-8"))["refine_nfev"]), q={})
    for key, name, unit, col, is_log in QUANT:
        if key == "height":
            v = np.array([float(r["t_b_refined"]) + float(r["T_refined"]) for r in rows])
        else:
            v = np.array([float(r[f"{key}_refined"]) for r in rows])
        lo = np.array([float(r[f"{col}_net68_lo"]) for r in rows]); hi = np.array([float(r[f"{col}_net68_hi"]) for r in rows])
        if is_log:
            lo, hi = np.exp(lo), np.exp(hi)
        val = float(np.median(v))
        res["q"][key] = dict(value=val, lo=float(max(min(v.min(), lo.min(), val), 0.0)), hi=float(max(v.max(), hi.max(), val)), source=src)
    rd = []
    if s.ref == "analytic_si":
        rd = [_readings(s, i, float(recs[i].get("eps", 0.0))) for i in recs]
    for key, st_key in (("L_knee", "knee_status"), ("d_fringe", "fringe_status")):
        v = [r[key] for r in rd if r.get(key) is not None and (key != "L_knee" or r[st_key] == "knee")]
        status = "not read (substrate reference in use)" if not rd else f"read in {len(v)} of {len(rd)} frames" if v else \
            max({r[st_key] for r in rd}, key=[r[st_key] for r in rd].count)
        res["q"][key] = dict(value=float(np.median(v)), lo=float(min(v)), hi=float(max(v)), source="reading", n=len(v)) if v else \
            dict(value=None, lo=None, hi=None, source="none", n=0)
        res["q"][key]["status"] = status

    Y = E = Mu = Sub = 0.0
    for i in recs:
        w = s.window(i)
        Ei = w["E"] * w["mask"]
        z = np.load(out / label / "static_fits" / label / f"{label}_{i:03d}_mu.npz")
        Y = Y + w["Y"] * w["mask"]; E = E + Ei; Mu = Mu + z["mu"] * Ei; Sub = Sub + z["sub"] * Ei
    res["_img"] = dict(Y=Y, E=E, Mu=Mu, Sub=Sub, q=np.asarray(s.q), af=np.asarray(s.af))
    return res


def _fmt(v, unit=""):
    if v is None:
        return "—"
    return (f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}") + (f" {unit}" if unit else "")


SRC_ZH = dict(reading="读数", model="模型值", estimate="估计", none="无")
ST_ZH = {"not read (substrate reference in use)": "用了实测的空基底作参考，不做这项读数", "at range limit": "条纹位置落在搜索范围的边上",
         "no fringe": "没有明显的条纹", "too thin": "膜太薄，条纹不在出射角范围内", "no signal": "信号不足",
         "at lower q limit": "拐点在 q 范围下限之外（尺度更大）", "at upper q limit": "拐点在 q 范围上限之外（尺度更小）"}


def _figures(S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    x = np.array([r["nominal_nm"] for r in S])
    grp = np.array([r["reference"] for r in S])
    one = len(S) == 1

    def pts(ax, key, color, marker, lab):
        v = [r["q"][key] for r in S]
        m = np.array([q["value"] is not None for q in v])
        if key == "rb":
            m &= np.array([r["q"]["t_b"]["value"] >= 0.3 for r in S])
        if not m.any():
            return
        val = np.array([q["value"] if k else np.nan for q, k in zip(v, m)]); lo = np.array([q["lo"] if k else np.nan for q, k in zip(v, m)])
        hi = np.array([q["hi"] if k else np.nan for q, k in zip(v, m)])
        est = np.array([q["source"] == "estimate" for q in v]) & m
        for n, g in enumerate(dict.fromkeys(grp)):
            k = m & (grp == g)
            if k.any():
                ax.errorbar(x[k], val[k], yerr=[val[k] - lo[k], hi[k] - val[k]], color=color, marker=marker, ls="none", lw=0.8, ms=5, capsize=2,
                            mfc=color if g == "analytic_si" else "w", label=lab if n == 0 else None)
                xs = sorted(set(x[k].tolist()))
                if len(xs) > 1:
                    ax.plot(xs, [float(np.median(val[k & (x == v)])) for v in xs], color=color, lw=0.8)
        if est.any():
            ax.plot(x[est], val[est], marker=marker, ls="none", ms=9, mfc="none", mec="0.3")

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    a0 = ax[0, 0]
    pts(a0, "height", "C3", "o", "film height t_b + T (model)")
    pts(a0, "d_fringe", "k", "D", "film thickness (reading, fringe)")
    pts(a0, "h_m_model", "C0", "s", "mass thickness (model)")
    lim = [0.8 * x.min(), 1.25 * x.max()]
    a0.plot(lim, lim, color="0.7", lw=0.8, label="nominal")
    a0.set_title("thickness and height (nm)", fontsize=9)
    pts(ax[0, 1], "t_b", "C2", "o", "base layer t_b"); pts(ax[0, 1], "T", "C1", "s", "grain height above the base T")
    ax[0, 1].set_title("base layer and grains (nm)", fontsize=9)
    pts(ax[0, 2], "R", "C4", "o", "grain radius R"); pts(ax[0, 2], "d_hex", "C5", "s", "grain spacing")
    pts(ax[0, 2], "L_knee", "k", "D", "lateral scale 2 pi / q_knee (reading)")
    ax[0, 2].set_title("lateral sizes (nm)", fontsize=9); ax[0, 2].set_ylim(0, None)
    pts(ax[1, 0], "eta", "C6", "o", "grain coverage"); pts(ax[1, 0], "rb", "C7", "s", "relative base density")
    ax[1, 0].set_ylim(0, 1.1); ax[1, 0].set_title("coverage and density", fontsize=9)
    fill = [r["q"]["h_m_model"]["value"] / r["q"]["height"]["value"] for r in S]
    ax[1, 1].plot(x, fill, "o", color="C9", ms=5, label="filled fraction h_m / (t_b + T)")
    ax[1, 1].set_ylim(0, 1.1); ax[1, 1].set_title("mean metal density of the film", fontsize=9)
    a5 = ax[1, 2]
    a5.axhspan(0, 4, color="#e5f5e0", lw=0); a5.axhspan(4, FAIR, color="#fff7bc", lw=0); a5.axhspan(FAIR, 1e3, color="#fee0d2", lw=0)
    a5.plot(x, [r["fit"] for r in S], "ko", ms=5, label="result")
    a5.plot(x, [r["fit_network"] for r in S], "o", color="0.5", ms=4, mfc="none", label="network alone")
    a5.set_yscale("log"); a5.set_ylim(0.5, max(30.0, 1.5 * max(r["fit"] for r in S)))
    a5.set_title("fit deviation per cell (green: good, yellow: fair, red: model not usable)", fontsize=8)
    for a in ax.ravel():
        a.set_xlabel("nominal thickness (nm)", fontsize=8); a.legend(fontsize=7); a.tick_params(labelsize=8)
        if not one:
            a.set_xscale("log"); a.set_xticks(sorted(set(x.tolist()))); a.set_xticklabels([f"{v:g}" for v in sorted(set(x.tolist()))])
            a.xaxis.set_minor_formatter(NullFormatter())
        for r in S:
            a.annotate(r["label"], (r["nominal_nm"], a.get_ylim()[0]), fontsize=5, rotation=90, va="bottom", ha="right", color="0.5")
    fig.suptitle("static frames: structure of the Au film (filled: directly on Si, open: measured bare substrate as reference; "
                 "large ring: estimate, the model does not describe the frame)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "report.png", dpi=110); plt.close(fig)

    L = S[:12]
    fig, ax = plt.subplots(len(L), 5, figsize=(18, 2.9 * len(L)), squeeze=False)
    for r, smp in enumerate(L):
        g = smp["_img"]; q, af, E = g["q"], g["af"], g["E"]
        rate = lambda A: np.where(E.sum(0) > 0, A.sum(0) / np.maximum(E.sum(0), 1e-30), np.nan)
        D, Mo = rate(g["Y"]), rate(g["Mu"])
        ext = [q[0], q[-1], af[0], af[-1]]
        pos = D[np.isfinite(D) & (D > 0)]
        vmax = np.percentile(np.log10(pos), 99.5); vmin = np.percentile(np.log10(pos), 2)
        for j, (img, ttl) in enumerate(((D, "measured"), (Mo, "forward model of the result"))):
            ax[r, j].imshow(np.log10(np.maximum(img, 1e-30)), origin="lower", extent=ext, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
            ax[r, j].set_xscale("log"); ax[r, j].set_ylim(0, 1.0); ax[r, j].set_ylabel("alpha_f (deg)", fontsize=7)
            ax[r, j].set_title(f"{smp['label']}  {smp['nominal_nm']:g} nm: {ttl}", fontsize=8)
        im = ax[r, 2].imshow(np.log(np.maximum(D, 1e-30) / np.maximum(Mo, 1e-30)), origin="lower", extent=ext, aspect="auto", vmin=-0.5, vmax=0.5, cmap="RdBu_r")
        ax[r, 2].set_xscale("log"); ax[r, 2].set_ylim(0, 1.0); ax[r, 2].set_title("ln(measured / model)", fontsize=8)
        fig.colorbar(im, ax=ax[r, 2], fraction=0.04)

        def cut(A, sel, axis):
            idx = (slice(None), sel, slice(None)) if axis == 1 else (slice(None), slice(None), sel)
            num = A[idx].sum((0, axis)); den = E[idx].sum((0, axis))
            return np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        sel = (af >= 0.15) & (af < 0.30); a3 = ax[r, 3]
        a3.plot(q, cut(g["Y"], sel, 1), "k.", ms=4, label="measured")
        a3.plot(q, cut(g["Mu"], sel, 1), "C3-", lw=1.2, label="model"); a3.plot(q, cut(g["Sub"], sel, 1), ":", color="0.5", lw=1.2, label="substrate part")
        a3.set_xscale("log"); a3.set_yscale("log"); a3.set_title("cut along q_y, alpha_f 0.15-0.30 deg", fontsize=8); a3.set_xlabel("|q_y| (nm^-1)", fontsize=7)
        sel = (q >= 0.25) & (q < 0.45); a4 = ax[r, 4]
        a4.plot(af, cut(g["Y"], sel, 2), "k.", ms=4, label="measured")
        a4.plot(af, cut(g["Mu"], sel, 2), "C3-", lw=1.2, label="model"); a4.plot(af, cut(g["Sub"], sel, 2), ":", color="0.5", lw=1.2, label="substrate part")
        a4.set_yscale("log"); a4.set_xlim(0, 1.0); a4.set_title("cut along alpha_f, q_y 0.25-0.45 nm^-1", fontsize=8); a4.set_xlabel("alpha_f (deg)", fontsize=7)
        a4.text(0.97, 0.05, f"fit deviation {smp['fit']:.2f}", transform=a4.transAxes, ha="right", va="bottom", fontsize=7)
        if r == 0:
            a3.legend(fontsize=6)
        for a in ax[r]:
            a.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out / "patterns.png", dpi=80); plt.close(fig)


def _main_screen(S, out):
    'OUT/main/ for a GUI, in the layout of iampred.present (main.json + one w_<label>.npz per sample): the samples of'
    md = out / "main"
    md.mkdir(exist_ok=True)
    wins = []
    for k, r in enumerate(S, 1):
        q, im = r["q"], r["_img"]
        Y, E, Mu = (np.asarray(im[n], float) for n in ("Y", "E", "Mu"))
        if Y.ndim == 3:
            Y, E, Mu = Y.sum(0), E.sum(0), Mu.sum(0)
        good = E > 0
        meas = np.where(good, Y / np.maximum(E, 1e-30), np.nan)
        model = np.where(good, Mu / np.maximum(E, 1e-30), np.nan)
        name = f"w_{r['label']}.npz"
        np.savez_compressed(md / name, q=np.asarray(im["q"], np.float32), af=np.asarray(im["af"], np.float32),
                            measured=meas.astype(np.float32), models=model[None].astype(np.float32), labels=np.array(["model"]))
        R, D = q["R"]["value"], q["d_hex"]["value"]
        vals = dict(R_nm=R, H_nm=q["height"]["value"], D_nm=D, gap_nm=max(D - 2.0 * R, 0.0), coverage=q["eta"]["value"],
                    amount_nm=q["h_m_model"]["value"], base_nm=q["t_b"]["value"])
        rng = dict(R_nm=[q["R"]["lo"], q["R"]["hi"]], H_nm=[q["height"]["lo"], q["height"]["hi"]], D_nm=[q["d_hex"]["lo"], q["d_hex"]["hi"]],
                   coverage=[q["eta"]["lo"], q["eta"]["hi"]], amount_nm=[q["h_m_model"]["lo"], q["h_m_model"]["hi"]])
        entry = dict(label="model", kind="refined" if r["refine"] else "candidate", rank=None, weight=None, fit_deviation=round(float(r["fit"]), 3),
                     **{k_: round(float(v), 4) for k_, v in vals.items()},
                     ranges={k_: [round(float(a_), 4), round(float(b_), 4)] for k_, (a_, b_) in rng.items()}, source=q["R"]["source"])
        wins.append(dict(t=float(k), t_rel=None, h_nom=round(float(r["nominal_nm"]), 4), label=r["label"], sample=r.get("sample"),
                         substrate=r.get("substrate"), reference=r.get("reference"), frames=r["frames"], image=None, arrays=name,
                         entries=[entry], pick="model", smooth="model", path=None))
    doc = dict(scan=out.name, kind="static", family="static_nn_v1", metal="Au",
               x=dict(key="h_nom", label="nominal thickness", unit="nm"),
               wanted=[dict(key="R_nm", label="grain radius R", unit="nm"), dict(key="H_nm", label="height of the film (base layer + grains)", unit="nm"),
                       dict(key="D_nm", label="centre distance D", unit="nm"), dict(key="gap_nm", label="gap between grains D - 2R", unit="nm"),
                       dict(key="coverage", label="coverage of the grains", unit=""), dict(key="amount_nm", label="metal amount (equivalent thickness)", unit="nm")],
               sources=dict(R_nm="R", D_nm="d_hex", H_nm="t_b + T", coverage="eta", amount_nm="h_m_model", base_nm="t_b"),
               note="development reference; static frames: one entry per sample = the model solution, median over the frames of the "
                    "sample; fit deviation = deviance per cell with a 5 % model-error floor (lower is better)",
               choice="one entry per sample", windows=wins)
    (md / "main.json").write_bytes((json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))


def report(out):
    labels = sorted(p.parent.name for p in out.glob("*/frames.csv"))
    if not labels:
        raise SystemExit(f"{out}: no predicted sample (OUT/<label>/frames.csv)")
    S = [_sample(out, lab) for lab in labels]
    S.sort(key=lambda r: (r["nominal_nm"], r["label"]))
    keys = [q[0] for q in QUANT] + ["d_fringe", "L_knee"]
    with open(out / "report.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["label", "sample", "substrate", "nominal_nm", "reference", "frames", "refine", "fit", "fit_network"]
        for k in keys:
            cols += [k, f"{k}_lo", f"{k}_hi", f"{k}_source"]
        cols += ["fringe_status", "knee_status"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in S:
            row = {c: r[c] for c in cols[:9]}
            for k in keys:
                q = r["q"][k]
                row.update({k: q["value"], f"{k}_lo": q["lo"], f"{k}_hi": q["hi"], f"{k}_source": q["source"]})
            row.update(fringe_status=r["q"]["d_fringe"]["status"], knee_status=r["q"]["L_knee"]["status"])
            w.writerow({c: (round(v, 4) if isinstance(v, float) else v) for c, v in row.items()})
    _figures(S, out)
    _main_screen(S, out)
    md = ["# 静态帧的结构报告（Au 膜，开发参考）", "",
          f"- 样品数：{len(S)}。每个样品的数值是它各帧的中位数；范围合并了各帧之间的差别和网络给出的 68 % 区间。",
          "- 来源：**读数**是直接从计数读出的（沿出射角的膜干涉条纹给膜厚，水平切线的拐点给横向尺度），不经过模型；"
          "**模型值**是网络加正向模型的解（拟合偏差不超过 10）；**估计**是同样的数，但模型没有描述好这一帧（拟合偏差超过 10），只能当量级用。",
          "- 拟合偏差：每个像素格的偏差（含 5 % 模型误差），不超过 4 为好，4–10 尚可，超过 10 说明模型不适用。",
          "- 图：`report.png`（各量随名义厚度的变化），`patterns.png`（实测图、模型图、切线）。逐样品数值在 `report.csv`。", ""]
    for r in S:
        q = r["q"]
        ref = "直接长在 Si 上（解析的基底项）" if r["reference"] == "analytic_si" else f"用实测的空基底帧作基底项（{r['reference']}）"
        grade = "好" if r["fit"] <= 4 else "尚可" if r["fit"] <= FAIR else "模型不适用"
        md += [f"## {r['label']}：名义厚度 {r['nominal_nm']:g} nm", "",
               f"- 基底：{r['substrate']}，{ref}。用了 {r['frames']} 帧（导入 {r['frames_imported']} 帧），精修步数 {r['refine']}。",
               f"- 拟合偏差 {r['fit']:.2f}（{grade}）" + (f"；精修前 {r['fit_network']:.2f}。" if r["refine"] else "（网络加正向模型，没有精修）。"), "",
               "| 量 | 数值 | 范围 | 来源 |", "|---|---|---|---|"]
        f = q["d_fringe"]
        md.append(f"| 膜厚（干涉条纹） | {_fmt(f['value'], 'nm')} | {_fmt(f['lo'])} – {_fmt(f['hi'])} | 读数（{f['n']} 帧） |" if f["value"] is not None
                  else f"| 膜厚（干涉条纹） | — | — | 没有读数：{ST_ZH.get(f['status'], f['status'])}；看下面模型的膜总高度 |")
        for key, name, unit, _, _ in QUANT:
            v = q[key]
            if key == "rb" and q["t_b"]["value"] < 0.3:
                md.append(f"| {name} | — | — | 不适用：底层厚度不足 0.3 nm，膜由分开的晶粒组成 |")
                continue
            md.append(f"| {name} | {_fmt(v['value'], unit)} | {_fmt(v['lo'])} – {_fmt(v['hi'])} | {SRC_ZH[v['source']]} |")
        k = q["L_knee"]
        md.append(f"| 横向特征长度 2π/q（拐点） | {_fmt(k['value'], 'nm')} | {_fmt(k['lo'])} – {_fmt(k['hi'])} | 读数（{k['n']} 帧） |" if k["value"] is not None
                  else f"| 横向特征长度 2π/q（拐点） | — | — | 没有读数：{ST_ZH.get(k['status'], k['status'])}；看上面模型的晶粒间距 |")
        md.append("")
    md += ["## 使用时注意", "",
           "- 金属等效厚度和膜的总高度来自图样的形状，不是绝对定标：入射强度未知，模型里强度系数和质量是耦合的。和名义厚度相差 30 % 以内属于正常。",
           "- 晶粒半径、间距对应图样里的峰或肩；峰不明显的样品这两个量的范围会很宽，按范围而不是按单个数值引用。",
           "- 网络是用这一类静态 Au 样品的模拟训练的（4–20 nm，Si 或 Si 上的纳米颗粒层）；偏离这个范围的样品先看拟合偏差。",
           "- 本报告是开发参考，没有用独立表征（AFM、SEM、XRR）验证过。", ""]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    return [r["label"] for r in S]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames")
    ap.add_argument("--label")
    ap.add_argument("--nominal-nm", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-frames")
    ap.add_argument("--reference")
    ap.add_argument("--sample")
    ap.add_argument("--substrate")
    ap.add_argument("--metal", choices=["Au"], default="Au")
    ap.add_argument("--mode", choices=list(MODES), default="quick")
    ap.add_argument("--max-frames", type=int, default=-1, help="-1: the default of the mode; 0: all usable frames")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--models")
    ap.add_argument("--data")
    ap.add_argument("--mask")
    ap.add_argument("--sdd-mm", type=float, default=GEOMETRY["sdd_mm"])
    ap.add_argument("--energy-ev", type=float, default=GEOMETRY["energy_ev"])
    ap.add_argument("--beam-row", type=float, default=GEOMETRY["db_row"])
    ap.add_argument("--beam-col", type=float, default=GEOMETRY["db_col"])
    ap.add_argument("--roll", type=float, default=GEOMETRY["sample_roll_deg"])
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = static_root(a.data)
    use_root(root)
    steps, status = [], "ok"

    def step(name, fn):
        nonlocal status
        t0 = time.time()
        print(f"[{name}] ...", flush=True)
        try:
            note = fn()
            rc = 0
        except SystemExit as e:
            note, rc = str(e), 1
        except Exception as e:
            note, rc = f"{type(e).__name__}: {e}", 1
        steps.append(dict(step=name, returncode=rc, seconds=round(time.time() - t0, 1), note=note if isinstance(note, str) else None))
        print(f"      {'done' if rc == 0 else 'FAILED: ' + str(note)} in {steps[-1]['seconds']:.0f} s", flush=True)
        if rc != 0 and status == "ok":
            status = f"failed at {name}"
        return rc == 0

    ok = True
    if a.label:
        have = (root / a.label / "binned" / "run.json").exists()
        if not have and not a.frames:
            raise SystemExit(f"{a.label} is not imported yet: give --frames")
        if not have and a.nominal_nm is None:
            raise SystemExit("give --nominal-nm (the nominal thickness of the sample, nm)")
        if a.reference and a.reference_frames:
            raise SystemExit("give --reference or --reference-frames, not both")

        def do_intake():
            if have:
                return "sample already imported: reused"
            ref = a.reference or "analytic_si"
            if a.reference_frames:
                ref = f"{a.label}_ref"
                if not (root / ref / "binned" / "run.json").exists():
                    intake(a, root, ref, a.reference_frames, 0.0, None, "bare substrate frames (iampred.static)")
            elif a.reference and not (root / a.reference / "binned" / "run.json").exists():
                raise SystemExit(f"reference {a.reference} is not imported under {root}")
            intake(a, root, a.label, a.frames, a.nominal_nm, ref, "iampred.static")
            return None

        def do_predict():
            mode = MODES[a.mode]
            n = mode["max_frames"] if a.max_frames < 0 else a.max_frames
            cmd = [sys.executable, str(IAM / "static_nn.py"), "predict", "--models", str(models_dir(a.models)), "--labels", a.label, "--frames", "fit",
                   "--refine", str(mode["refine"]), "--k", "4", "--out", str(out / a.label), "--workers", str(a.workers), "--no-figures"]
            cmd += ["--max-frames", str(n)] if n else []
            log = out / f"analyze_predict_{a.label}.log"
            with open(log, "w", encoding="utf-8") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(IAM), env=dict(os.environ, IAM_DATA=str(root))).returncode
            if rc != 0 or not (out / a.label / "frames.csv").exists():
                raise SystemExit(f"static_nn.py predict failed (see {log})")
            return None

        ok = step("intake", do_intake) and step("predict", do_predict)
    elif a.frames:
        raise SystemExit("give --label with --frames")
    labels = []
    if ok:
        def do_report():
            labels.extend(report(out))
            return None
        ok = step("report", do_report)
    files = {k: (str(out / v) if (out / v).exists() else None) for k, v in FILES.items()}
    res = dict(status=status, kind="static", label=a.label, samples=labels, mode=a.mode, data_root=str(root), steps=steps, files=files,
               note="development reference: not validated against independent characterisation")
    (out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"status: {status}; report: {files['report_md']}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
