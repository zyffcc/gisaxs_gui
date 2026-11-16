'Network for static (ex-situ) GISAXS frames: one frame -> morphology of the grains/base family of iam_static.py'

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iam_static as IS
import iam as M
import iam_paths as IAMP

VERSION = "static_nn_v1"
REPORTED = ["ln_R", "ln_D", "T", "t_b", "ln_hm", "eta", "rb", "fill", "height"]
AMP = ("lnK", "lncs", "lnA", "lnB")
SUBS = AMP + ("lnxi", "Hs", "eps", "sig", "bg", "g")
KEY = ("lnR", "s", "eta", "fcap", "fb", "rb")
POOL = 2
KINDS = ["near", "cross", "wide"]
PROBS = (0.35, 0.40, 0.25)
SIG_U, SIG_AMP, SIG_FM = 0.08, 0.3, 0.3


def mode_of(label):
    run = json.loads((IAMP.DATA / label / "binned" / "run.json").read_text(encoding="utf-8"))
    return "analytic" if (run.get("reference") or "analytic_si") == "analytic_si" else "ref"


def load_records(d):
    '{label: [dict(frame, params, h_m, dev5)]} from <d>/<label>/<label>_<frame>*.json.'
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, "*", "*.json"))):
        r = json.load(open(fn, encoding="utf-8"))
        fr = int(re.search(r"_(\d{3})", os.path.basename(fn)[len(r["label"]):]).group(1))
        out.setdefault(r["label"], []).append(dict(label=r["label"], frame=fr, params=r["params"], h_m=float(r["h_m_model"]),
                                                   dev5=float(r["dev_per_cell"]["0.05"])))
    return out


def features(s, w, Y, mode):
    'Network image (channels, alpha_f rows pooled by POOL, q_y): log10 count rate (both detector sides summed);'
    ok = w["mask"]
    E = np.where(ok, w["E"], 0.0).sum(0)
    Ys = np.where(ok, Y, 0.0).sum(0)
    n = (E.shape[0] // POOL) * POOL

    def pl(A):
        return A[:n].reshape(n // POOL, POOL, -1).sum(1)

    E2, Y2 = pl(E), pl(Ys)
    pos = E2[E2 > 0]
    good = E2 > 0.3 * (np.median(pos) if pos.size else 1.0)
    rate = np.where(good, Y2 / np.maximum(E2, 1e-12), 0.0)
    fl = 1e-3 * np.median(rate[good]) if good.any() else 1e-6
    ch = [np.where(good, np.log10(rate + fl), 0.0)]
    if mode == "ref":
        P2 = pl(np.where(ok, s.I_ref * w["E"], 0.0).sum(0))
        pre = np.where(good, P2 / np.maximum(E2, 1e-12), 0.0)
        flp = 1e-3 * np.median(pre[good]) if good.any() else 1e-6
        ch += [np.where(good, np.log10(pre + flp), 0.0), np.where(good, np.log10(rate + fl) - np.log10(pre + flp), 0.0)]
    return np.stack(ch + [good.astype(float)]).astype(np.float32)


def reported(f, m, p):
    d = f.describe(m, p)
    hm = f.h_nom * float(np.exp(p["lnfm"]))
    H = d["t_b"] + d["T"]
    return np.array([np.log(d["R"]), np.log(d["d_hex"]), d["T"], d["t_b"], np.log(hm), d["eta"], d["rb"], hm / H, H], np.float32)


def forward_hash():
    from iampred import registry as REG
    h = hashlib.sha256(REG.forward_hash().encode())
    h.update((HERE / "iam_static.py").read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def dev5_of(f, x):
    (mu, sub, _), _, _ = f.model(x)
    return float(M.dev5(f.w, mu, f.variance(sub), f=0.05).sum() / f.ok.sum())


def _bank_task(args):
    label, recs = args
    s = IS.StaticScan(label)
    mode = "analytic" if s.ref == "analytic_si" else "ref"
    from scipy.ndimage import gaussian_filter
    R, fr, d5 = [], [], []
    for r in recs:
        f = IS.StaticFit(s, r["frame"], mode)
        x = f.x0(**{k: v for k, v in r["params"].items() if k in f.names})
        (mu, sub, _), _, _ = f.model(x)
        lam = mu * f.w["E"]
        z = np.where(f.ok, np.log((f.w["Y"] + 1.0) / (lam + 1.0)), 0.0)
        wgt = gaussian_filter(f.ok.astype(float), (0, 1.5, 1.5))
        z = np.where(f.ok, gaussian_filter(z, (0, 1.5, 1.5)) / np.maximum(wgt, 1e-6), 0.0)
        R.append(np.clip(z, -1.5, 1.5).astype(np.float16)); fr.append(r["frame"])
        d5.append(float(M.dev5(f.w, mu, f.variance(sub), f=0.05).sum() / f.ok.sum()))
        if abs(d5[-1] - r["dev5"]) > 1e-3 * r["dev5"]:
            print(f"bank {label} #{r['frame']}: dev5 {d5[-1]:.4f} against the record {r['dev5']:.4f}", flush=True)
    return label, mode, np.array(R), fr, d5


def cmd_bank(a):
    recs = load_records(a.records)
    res = _map(_bank_task, [(lab, L) for lab, L in recs.items()], a.workers)
    bad = sum(abs(d - r["dev5"]) > 1e-3 * r["dev5"] for (lab, L), q in zip(recs.items(), res) for d, r in zip(q[4], L))
    if bad and not a.allow_mismatch:
        raise SystemExit(f"{bad} records are not reproduced by the current code and switches (dev5 differs by more than 0.1 %)")
    np.savez_compressed(a.out, R=np.concatenate([r[2] for r in res]), label=np.array(sum(([r[0]] * len(r[3]) for r in res), [])),
                        mode=np.array(sum(([r[1]] * len(r[3]) for r in res), [])), frame=np.array(sum((r[3] for r in res), [])),
                        dev5=np.array(sum((r[4] for r in res), [])))
    print(f"{a.out}: {sum(len(r[3]) for r in res)} residual fields of {len(res)} samples")


def _gen_task(args):
    mode, label, n, seed, recs, bank_path, hold = args
    from scipy.ndimage import gaussian_filter, uniform_filter
    rng = np.random.default_rng(seed)
    s = IS.StaticScan(label)
    own = recs[label]
    others = [r for lab, L in recs.items() if lab != label and lab not in hold for r in L]
    group = [r for lab, L in recs.items() if lab[2] == label[2] and lab not in hold and lab != label for r in L] or own
    bank = np.load(bank_path, allow_pickle=False) if bank_path else None
    if bank is not None:
        bl = bank["label"].astype(str)
        bsel = np.where((bl != label) & ~np.isin(bl, list(hold)))[0]
        BR = bank["R"]
    fits = {}
    rel_ref = None
    if mode == "ref":
        rel_ref = np.where(s.I_ref > 0, np.sqrt(np.maximum(s.V_ref, 0.0)) / np.maximum(s.I_ref, 1e-300), 0.0)
    X, S, P, R, C, K = [], [], [], [], [], []
    nfail, fails, tries, t0 = 0, {}, 0, time.time()
    while len(X) < n and tries < 5 * n + 50:
        tries += 1
        r0 = own[int(rng.integers(len(own)))]
        if r0["frame"] not in fits:
            fits[r0["frame"]] = IS.StaticFit(s, r0["frame"], mode)
        f = fits[r0["frame"]]
        names, lb, ub = f.names, f.lb, f.ub
        span = ub - lb
        kind = int(rng.choice(3, p=PROBS))
        if kind and not others:
            kind = 0
        shape = r0 if kind == 0 else others[int(rng.integers(len(others)))]
        amp = r0 if kind == 0 else group[int(rng.integers(len(group)))]
        hm = shape["h_m"] * float(np.exp(rng.normal(0, 0.1 if kind == 0 else 0.2)))
        x = np.empty(len(names))
        for j, nm in enumerate(names):
            src = amp if nm in SUBS else shape
            v = float(src["params"].get(nm, r0["params"].get(nm, 0.5 * (lb[j] + ub[j]))))
            if nm in AMP:
                v += rng.normal(0, SIG_AMP)
            elif kind == 2 and nm in KEY:
                v = lb[j] + span[j] * (rng.uniform(0, 0.03) if (nm == "fb" and rng.random() < 0.3) else rng.random())
            else:
                v += rng.normal(0, SIG_U) * span[j]
            x[j] = v
        if kind == 2:
            hm = float(np.exp(rng.uniform(np.log(2.0), np.log(25.0))))
        jf = names.index("lnfm")
        x[jf] = np.clip(rng.normal(0, SIG_FM), lb[jf] + 1e-6, ub[jf] - 1e-6)
        x = np.clip(x, lb + 1e-9, ub - 1e-9)
        f.h_nom = hm / float(np.exp(x[jf]))
        try:
            (mu, sub, _), m, p = f.model(x)
            rep = reported(f, m, p)
        except (Exception, SystemExit) as e:
            nfail += 1
            why = f"{type(e).__name__}: {str(e)[:80]}"
            fails[why] = fails.get(why, 0) + 1
            continue
        ok = f.ok
        if rel_ref is not None:
            zf = uniform_filter(rng.normal(0, 1, mu.shape), size=(1, 3, 3), mode="nearest") * 3.0
            mu = mu + np.asarray(sub, float) * rel_ref * zf
        lam = np.where(ok, mu * f.w["E"], 0.0)
        if not np.all(np.isfinite(lam)) or np.any(lam < 0) or not np.all(np.isfinite(rep)):
            nfail += 1
            fails["non-finite or negative model"] = fails.get("non-finite or negative model", 0) + 1
            continue
        field = np.zeros_like(lam)
        if bank is not None and len(bsel):
            field += rng.uniform(0, 1) * BR[int(bsel[rng.integers(len(bsel))])].astype(float)
        g = gaussian_filter(rng.normal(0, 1, lam.shape), (0, 3.0, 3.0))
        field += rng.uniform(0, 0.08) * g / max(g.std(), 1e-9)
        Y = rng.poisson(np.maximum(lam * np.exp(field), 0.0)).astype(float)
        X.append(features(s, f.w, Y, mode).astype(np.float16))
        S.append([np.log(f.h_nom)]); P.append(x.astype(np.float32)); R.append(rep)
        C.append(f"{label}_{r0['frame']:03d}"); K.append(kind)
    f0 = next(iter(fits.values()))
    meta = dict(label=label, pnames=f0.names, lb=f0.lb.tolist(), ub=f0.ub.tolist(), n_made=len(X), n_failed=nfail, fail_reasons=fails,
                seconds=time.time() - t0)
    return np.array(X), np.array(S, np.float32), np.array(P, np.float32), np.array(R, np.float32), C, K, meta


def _map(fn, tasks, workers):
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(fn, tasks))
    return [fn(t) for t in tasks]


def cmd_gen(a):
    recs = load_records(a.records)
    labels = [lab for lab in recs if mode_of(lab) == a.mode]
    hold = sorted(h.strip() for h in a.holdout.split(",") if h.strip())
    if set(hold) - set(recs):
        raise SystemExit(f"--holdout labels without records: {sorted(set(hold) - set(recs))}")
    per = int(np.ceil(a.n / len(labels)))
    tasks = []
    for i, lab in enumerate(labels):
        left, j = per, 0
        while left > 0:
            k = min(a.chunk, left)
            tasks.append((a.mode, lab, k, (a.seed, i, j), recs, a.bank, hold))
            left -= k; j += 1
    t0 = time.time()
    res = [r for r in _map(_gen_task, tasks, a.workers) if len(r[0])]
    pn = {json.dumps([r[6]["pnames"], np.round(r[6]["lb"], 12).tolist(), np.round(r[6]["ub"], 12).tolist()]) for r in res}
    if len(pn) != 1:
        raise SystemExit("parameter lists or bounds differ between the samples")
    m0 = res[0][6]
    fails = {}
    for r in res:
        for k, v in r[6]["fail_reasons"].items():
            fails[k] = fails.get(k, 0) + v
    fam = dict(family=f"static_{a.mode}", version=VERSION, params=m0["pnames"], lb=m0["lb"], ub=m0["ub"], reported=REPORTED, pool=POOL,
               probs=PROBS, sig=[SIG_U, SIG_AMP, SIG_FM], env={k: v for k, v in sorted(os.environ.items()) if k.startswith("IAM_") and
                                                                 k not in ("IAM_DATA", "IAM_WORK", "IAM_GIMAP", "IAM_RAW", "IAM_DIST", "IAM_PY", "IAM_JOBS", "IAM_CHECK")})
    meta = dict(family=fam["family"], family_hash=hashlib.sha256(json.dumps(fam, sort_keys=True).encode()).hexdigest(),
                forward_hash=forward_hash(), pred_hash=VERSION, mode=a.mode, params=m0["pnames"], lb=m0["lb"], ub=m0["ub"],
                reported=REPORTED, scalars=["ln_nominal_nm"], features=dict(pool=POOL), labels=labels, holdout=hold,
                records=str(a.records), bank=a.bank, seed=a.seed, n_requested=a.n, n_made=int(sum(len(r[0]) for r in res)),
                n_failed=int(sum(r[6]["n_failed"] for r in res)), fail_reasons=fails, kinds=KINDS, seconds=time.time() - t0,
                workers=a.workers, env=fam["env"])
    np.savez_compressed(a.out, X=np.concatenate([r[0] for r in res]), S=np.concatenate([r[1] for r in res]),
                        P=np.concatenate([r[2] for r in res]), R=np.concatenate([r[3] for r in res]),
                        ctx=np.array(sum((r[4] for r in res), [])), kind=np.array(sum((r[5] for r in res), []), np.int8),
                        meta=json.dumps(meta, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in ("family", "n_made", "n_failed", "fail_reasons", "seconds")}))


def model_path(models, mode):
    for cand in (Path(models) / f"static_{mode}" / "model.npz", Path(models) / f"static_{mode}.npz"):
        if cand.exists():
            return cand
    raise SystemExit(f"no network for mode '{mode}' under {models} (static_{mode}/model.npz)")


def _predict_task(args):
    label, i, models, refine, k, outdir, rb_late = args
    from iampred import nnmodel as NM
    t0 = time.time()
    s = IS.StaticScan(label)
    mode = "analytic" if s.ref == "analytic_si" else "ref"
    f = IS.StaticFit(s, i, mode)
    if rb_late > 0 and float(s.cfg["nominal_nm"]) >= 8 and "rb" in f.names:
        j = f.names.index("rb")
        f.lb[j] = max(f.lb[j], min(rb_late, f.ub[j] - 1e-6))
    mp = model_path(models, mode)
    P = NM.load(str(mp))
    meta = P.meta
    if list(meta["params"]) != list(f.names):
        raise SystemExit(f"{label}: the network parameters do not match the fitter ({mp})")
    notes = []
    if meta.get("forward_hash") != forward_hash():
        notes.append("forward code changed after training")
    X = features(s, f.w, f.w["Y"], mode)[None]
    S = np.array([[np.log(max(f.h_nom, 0.05))]], np.float32)
    cands = P.candidates(X, S)[0][:k]
    q = P.intervals(X, S)[0]
    tried = []
    for c in cands:
        x = np.clip(np.asarray(c["x"], float), f.lb + 1e-9, f.ub - 1e-9)
        try:
            tried.append((dev5_of(f, x), x, c["weight"], "network"))
            xa = f.amplitudes(x)
            tried.append((dev5_of(f, xa), xa, c["weight"], "network+nnls"))
        except (Exception, SystemExit) as e:
            notes.append(f"candidate failed: {type(e).__name__}: {str(e)[:60]}")
    if not tried:
        raise SystemExit(f"{label} #{i}: no candidate could be rendered")
    best_as_given = min((t for t in tried if t[3] == "network"), key=lambda v: v[0])
    tried.sort(key=lambda v: v[0])
    net = dict(model=str(mp), family=meta.get("family"), family_hash=meta.get("family_hash"), k=len(cands), notes=notes,
               candidates=[dict(dev5=round(t[0], 4), weight=round(float(t[2]), 4), amplitudes=t[3]) for t in tried],
               dev5_as_given=best_as_given[0], dev5_direct=tried[0][0],
               intervals_68={nm: [float(q[j, 1]), float(q[j, 3])] for j, nm in enumerate(meta["reported"])},
               intervals_95={nm: [float(q[j, 0]), float(q[j, 4])] for j, nm in enumerate(meta["reported"])},
               median={nm: float(q[j, 2]) for j, nm in enumerate(meta["reported"])})

    def write(x, opt, sub, t_start, stage=None):
        rec, mu, Pp, m, p = IS.record(f, s, label, i, x, opt, [round(t[0], 4) for t in tried], t_start, mode)
        rec["network"] = net
        od = Path(outdir) / sub / "static_fits" / label
        od.mkdir(parents=True, exist_ok=True)
        (od / f"{label}_{i:03d}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        np.savez_compressed(od / f"{label}_{i:03d}_mu.npz", mu=mu.astype(np.float32), sub=Pp["sub"].astype(np.float32))
        if stage:
            _live_frame(s, f, label, i, rec, mu, stage)
        return rec

    seen, starts = [], []
    for t in tried:
        if not any(abs(t[2] - w) < 1e-12 for w in seen):
            seen.append(t[2]); starts.append(t)
    rd = write(tried[0][1], dict(start=tried[0][3], nfev=0), "direct", t0, "network")
    t_direct = time.time() - t0
    rr = rd
    if refine > 0:
        runs = []
        for t in starts[:2]:
            x, F = f.run(t[1], refine)
            runs.append((F, x, dict(f.last, start=t[3])))
        runs.sort(key=lambda v: v[0])
        net["dev5_refine_runs"] = [round(dev5_of(f, r[1]), 4) for r in runs]
        rr = write(runs[0][1], runs[0][2], ".", t0, "refined")
    else:
        write(tried[0][1], dict(start=tried[0][3], nfev=0), ".", t0)
    print(f"{label} #{i} dev5 network {net['dev5_as_given']:.2f} +nnls {net['dev5_direct']:.2f} refined {rr['dev_per_cell']['0.05']:.2f} "
          f"| h_m {rr['h_m_model']:.2f} t_b {rr['t_b']:.2f} T {rr['T']:.2f} R {rr['R']:.2f} D {rr['d_hex']:.1f} ({time.time() - t0:.0f} s)", flush=True)
    row = dict(label=label, frame=i, mode=mode, nominal_nm=s.cfg["nominal_nm"], dev5_network=net["dev5_as_given"], dev5_direct=net["dev5_direct"],
               dev5_refined=rr["dev_per_cell"]["0.05"], seconds_direct=round(t_direct, 1), seconds_total=round(time.time() - t0, 1))
    for tag, r in (("direct", rd), ("refined", rr)):
        row.update({f"{kk}_{tag}": round(float(r[kk]), 4) for kk in ("h_m_model", "t_b", "T", "R", "d_hex", "eta", "rb")})
    for nm in meta["reported"]:
        row[f"{nm}_net68_lo"], row[f"{nm}_net68_hi"] = (round(v, 4) for v in net["intervals_68"][nm])
    return row


def _live_frame(s, f, label, i, rec, mu, stage):
    'One frame solved, for the live view of a GUI (iampred.live; only when IAMPRED_LIVE names a folder).'
    from iampred import live as LV
    if LV.folder() is None:
        return
    try:
        meas, model = LV.static_frame_images(f.w, mu)
        LV.emit("frame", dict(q=np.asarray(s.q, np.float32), af=np.asarray(s.af, np.float32), measured=meas, model=model),
                label=label, index=int(i), frame=int(i) + 1, nominal_nm=LV.rounded(s.cfg.get("nominal_nm")), stage=stage,
                fit_deviation=LV.rounded(rec.get("dev_per_cell", {}).get("0.05"), 3), values=LV.static_values(rec))
    except Exception:
        pass


def cmd_predict(a):
    labels = [lab for lab, v in IS.TABLE.items() if v.get("mode")] if a.labels == "all" else a.labels.split(",")
    tasks = []
    for lab in labels:
        st = [fr["state"] for fr in IS.TABLE[lab]["frames"]] if lab in IS.TABLE else IS.StaticScan(lab).states
        if a.frames in ("fit", "usable"):
            frames = [i for i, v in enumerate(st) if v == "usable" or (a.frames == "fit" and v == "conditional")]
        else:
            frames = [int(v) for v in a.frames.split(",")]
        if a.max_frames:
            frames = frames[:a.max_frames]
        tasks += [(lab, i, a.models, a.refine, a.k, a.out, a.rb_late) for i in frames]
        try:
            from iampred import live as LV
            LV.emit("solving", label=lab, total=len(frames), refine=a.refine)
        except Exception:
            pass
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = _map(_predict_task, tasks, a.workers)
    with open(Path(a.out) / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    d = np.array([[r["dev5_network"], r["dev5_direct"], r["dev5_refined"]] for r in rows])
    summ = dict(frames=len(rows), samples=len(labels), refine_nfev=a.refine, k=a.k, wall_seconds=round(time.time() - t0, 1),
                cpu_seconds_direct=float(sum(r["seconds_direct"] for r in rows)), cpu_seconds_total=float(sum(r["seconds_total"] for r in rows)),
                median_dev5=dict(network=float(np.median(d[:, 0])), network_nnls=float(np.median(d[:, 1])), refined=float(np.median(d[:, 2]))))
    (Path(a.out) / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ))
    if not a.no_figures:
        import subprocess
        rec = str(Path(a.out) / "static_fits")
        groups = ",".join(sorted({lab[2] for lab in labels}, reverse=True))
        subprocess.run([sys.executable, str(HERE / "static_trend.py"), "--dir", rec, "--out", str(Path(a.out) / "trend")], check=False)
        subprocess.run([sys.executable, str(HERE / "static_schematic.py"), "--dir", rec, "--out", str(Path(a.out) / "schematic"),
                        "--groups", groups], check=False)


CMP = [("h_m_model", "mass thickness h_m (nm)", True), ("height", "film height t_b + T (nm)", True), ("t_b", "base layer t_b (nm)", False),
       ("T", "grain height T (nm)", True), ("R", "grain radius R (nm)", True), ("d_hex", "grain spacing (nm)", True),
       ("eta", "grain coverage", False), ("rb", "base density", False)]


def cmd_compare(a):
    'Network output (frames.csv of predict) against fit records of the same frames: table, summary JSON, figure.'
    import statistics as st
    ref = {}
    for fn in glob.glob(os.path.join(a.records, "*", "*.json")):
        r = json.load(open(fn, encoding="utf-8"))
        ref[(r["label"], int(r["frame"]))] = r
    rows = []
    for d in a.pred:
        with open(Path(d) / "frames.csv", encoding="utf-8") as fh:
            rows += [r for r in csv.DictReader(fh) if (r["label"], int(r["frame"])) in ref]
    if not rows:
        raise SystemExit("no frame of the prediction has a fit record")
    val = {}
    for r in rows:
        f = ref[(r["label"], int(r["frame"]))]
        v = dict(fit={k: float(f[k]) for k, _, _ in CMP if k != "height"}, dev5_fit=float(f["dev_per_cell"]["0.05"]))
        v["fit"]["height"] = v["fit"]["t_b"] + v["fit"]["T"]
        for tag in ("direct", "refined"):
            v[tag] = {k: float(r[f"{k}_{tag}"]) for k, _, _ in CMP if k != "height"}
            v[tag]["height"] = v[tag]["t_b"] + v[tag]["T"]
        v.update(dev5_network=float(r["dev5_network"]), dev5_direct=float(r["dev5_direct"]), dev5_refined=float(r["dev5_refined"]),
                 seconds_direct=float(r["seconds_direct"]), seconds_total=float(r["seconds_total"]), label=r["label"], nominal=float(r["nominal_nm"]))
        val[(r["label"], int(r["frame"]))] = v
    V = list(val.values())
    summ = dict(frames=len(V), samples=len({v["label"] for v in V}), records=str(a.records).replace("\\", "/"),
                median_dev5=dict(fit=st.median(v["dev5_fit"] for v in V), network=st.median(v["dev5_network"] for v in V),
                                 network_nnls=st.median(v["dev5_direct"] for v in V), refined=st.median(v["dev5_refined"] for v in V)),
                refined_within_5pct_of_fit=float(np.mean([v["dev5_refined"] <= 1.05 * v["dev5_fit"] for v in V])),
                refined_below_fit=float(np.mean([v["dev5_refined"] < v["dev5_fit"] for v in V])),
                median_seconds=dict(direct=st.median(v["seconds_direct"] for v in V), total=st.median(v["seconds_total"] for v in V)),
                difference_to_fit={})
    for k, _, rel in CMP:
        for tag in ("direct", "refined"):
            if rel:
                e = [abs(np.log(max(v[tag][k], 1e-6) / max(v["fit"][k], 1e-6))) for v in V]
                summ["difference_to_fit"][f"{k}_{tag}"] = dict(kind="relative", median=float(np.expm1(np.median(e))), p90=float(np.expm1(np.percentile(e, 90))))
            else:
                e = [abs(v[tag][k] - v["fit"][k]) for v in V]
                summ["difference_to_fit"][f"{k}_{tag}"] = dict(kind="absolute", median=float(np.median(e)), p90=float(np.percentile(e, 90)))
    per = []
    for lab in sorted({v["label"] for v in V}, key=lambda s: (s[2], IS.TABLE.get(s, {}).get("mode") or "", IS.TABLE.get(s, {}).get("nominal_nm") or 0)):
        L = [v for v in V if v["label"] == lab]
        row = dict(label=lab, mode=IS.TABLE.get(lab, {}).get("mode"), nominal_nm=L[0]["nominal"], frames=len(L))
        for kk in ("dev5_fit", "dev5_network", "dev5_direct", "dev5_refined"):
            row[kk] = round(st.median(v[kk] for v in L), 3)
        for k, _, _ in CMP:
            for tag in ("fit", "direct", "refined"):
                row[f"{k}_{tag}"] = round(st.median(v[tag][k] for v in L), 3)
        per.append(row)
    summ["samples_table"] = per
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(CMP) + 1, figsize=(3.0 * (len(CMP) + 1), 6.2))
    col = {"7": "C0", "6": "C2", "5": "C4"}
    for i, tag in enumerate(("direct", "refined")):
        for j, (k, lab, rel) in enumerate(CMP):
            x = [v["fit"][k] for v in V]; y = [v[tag][k] for v in V]
            ax[i, j].scatter(x, y, s=8, c=[col.get(v["label"][2], "0.4") for v in V], alpha=0.7)
            lo, hi = min(x + y), max(x + y)
            ax[i, j].plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
            ax[i, j].set_title(f"{lab}\n{'network' if tag == 'direct' else 'network + refinement'}", fontsize=8)
            ax[i, j].set_xlabel("fit", fontsize=7); ax[i, j].set_ylabel(tag, fontsize=7); ax[i, j].tick_params(labelsize=7)
        x = [v["dev5_fit"] for v in V]; y = [v["dev5_direct" if tag == "direct" else "dev5_refined"] for v in V]
        ax[i, -1].scatter(x, y, s=8, c=[col.get(v["label"][2], "0.4") for v in V], alpha=0.7)
        ax[i, -1].plot([min(x), max(x)], [min(x), max(x)], color="0.5", lw=0.8)
        ax[i, -1].set_xscale("log"); ax[i, -1].set_yscale("log")
        ax[i, -1].set_title(f"dev5 per cell\n{'network (best candidate)' if tag == 'direct' else 'network + refinement'}", fontsize=8)
        ax[i, -1].set_xlabel("fit", fontsize=7); ax[i, -1].tick_params(labelsize=7)
    for g, nm in (("7", "Si"), ("6", "RT TiO2"), ("5", "850 C TiO2")):
        ax[0, 0].scatter([], [], c=col[g], s=10, label=nm)
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=85)
    print(json.dumps({k: v for k, v in summ.items() if k != "samples_table"}))


HB = ((0.15, 0.30), (0.45, 0.70))
VB = (0.25, 0.45)


def cmd_maps(a):
    'Per sample: measured 2D image, the image the forward model computes from the network result, and cuts through'
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = sorted({Path(f).parent.name for f in glob.glob(os.path.join(a.pred, "static_fits", "*", "*_mu.npz"))},
                  key=lambda s: (IS.TABLE[s].get("mode") or "", IS.TABLE[s].get("nominal_nm") or 0, s))
    if a.labels != "all":
        labs = [s for s in labs if s in a.labels.split(",")]
    for g in sorted({s[2] for s in labs}, reverse=True):
        L = [s for s in labs if s[2] == g]
        fig, ax = plt.subplots(len(L), 6, figsize=(21, 2.9 * len(L)), squeeze=False)
        for r, lab in enumerate(L):
            s = IS.StaticScan(lab)
            fr = sorted(int(Path(f).stem.split("_")[-2]) for f in glob.glob(os.path.join(a.pred, "static_fits", lab, "*_mu.npz")))
            Y = E = 0.0; M = {}
            for i in fr:
                w = s.window(i)
                Ei = w["E"] * w["mask"]
                Y = Y + w["Y"] * w["mask"]; E = E + Ei
                for tag, sub in (("refined", ""), ("direct", "direct")):
                    z = np.load(Path(a.pred) / sub / "static_fits" / lab / f"{lab}_{i:03d}_mu.npz")
                    M[tag] = M.get(tag, 0.0) + z["mu"] * Ei
                    if tag == "refined":
                        M["sub"] = M.get("sub", 0.0) + z["sub"] * Ei
            rec = [json.load(open(Path(a.pred) / "static_fits" / lab / f"{lab}_{i:03d}.json", encoding="utf-8")) for i in fr]
            d5 = float(np.median([x["dev_per_cell"]["0.05"] for x in rec])); d5n = float(np.median([x["network"]["dev5_direct"] for x in rec]))
            rate = lambda A: np.where(E.sum(0) > 0, A.sum(0) / np.maximum(E.sum(0), 1e-30), np.nan)
            D, R = rate(Y), rate(M["refined"])
            ext = [s.q[0], s.q[-1], s.af[0], s.af[-1]]
            vmax = np.nanpercentile(np.log10(D[D > 0]), 99.5); vmin = np.nanpercentile(np.log10(D[D > 0]), 2)
            for j, (img, ttl) in enumerate(((D, "measured"), (R, "forward model of the network result"))):
                ax[r, j].imshow(np.log10(np.maximum(img, 1e-30)), origin="lower", extent=ext, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
                ax[r, j].set_xscale("log"); ax[r, j].set_ylim(0, 1.0); ax[r, j].set_ylabel("alpha_f (deg)", fontsize=7)
                ax[r, j].set_title(f"{lab[2:].split('_raw')[0].replace('_', '-')}  {IS.TABLE[lab]['mode']} {IS.TABLE[lab]['nominal_nm']:g} nm: {ttl}", fontsize=8)
            im = ax[r, 2].imshow(np.log(np.maximum(D, 1e-30) / np.maximum(R, 1e-30)), origin="lower", extent=ext, aspect="auto", vmin=-0.5, vmax=0.5, cmap="RdBu_r")
            ax[r, 2].set_xscale("log"); ax[r, 2].set_ylim(0, 1.0); ax[r, 2].set_title("ln(measured / model)", fontsize=8)
            fig.colorbar(im, ax=ax[r, 2], fraction=0.04)
            sty = (("refined", "network + refinement", "C3", "-"), ("direct", "network alone", "C0", "--"), ("sub", "substrate part", "0.5", ":"))

            def cut(A, sel, axis):
                idx = (slice(None), sel, slice(None)) if axis == 1 else (slice(None), slice(None), sel)
                num = A[idx].sum((0, axis)); den = E[idx].sum((0, axis))
                return np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
            for j, band in enumerate(HB):
                sel = (s.af >= band[0]) & (s.af < band[1]); x = ax[r, 3 + j]
                x.plot(s.q, cut(Y, sel, 1), "k.", ms=4, label="measured")
                for k, nm, c, ls in sty:
                    x.plot(s.q, cut(M[k], sel, 1), color=c, ls=ls, lw=1.2, label=nm)
                x.set_xscale("log"); x.set_yscale("log"); x.set_title(f"cut along q_y, alpha_f {band[0]:.2f}-{band[1]:.2f} deg", fontsize=8)
            sel = (s.q >= VB[0]) & (s.q < VB[1]); x = ax[r, 5]
            x.plot(s.af, cut(Y, sel, 2), "k.", ms=4, label="measured")
            for k, nm, c, ls in sty:
                x.plot(s.af, cut(M[k], sel, 2), color=c, ls=ls, lw=1.2, label=nm)
            x.set_yscale("log"); x.set_xlim(0, 1.0); x.set_title(f"cut along alpha_f, q_y {VB[0]:.2f}-{VB[1]:.2f} nm^-1", fontsize=8)
            x.text(0.97, 0.05, f"dev5: network {d5n:.2f}, refined {d5:.2f}", transform=x.transAxes, ha="right", va="bottom", fontsize=7)
            if r == 0:
                ax[r, 3].legend(fontsize=6)
            for x in ax[r]:
                x.tick_params(labelsize=7)
        for j in range(5):
            ax[-1, j].set_xlabel("|q_y| (nm^-1)", fontsize=7)
        ax[-1, 5].set_xlabel("alpha_f (deg)", fontsize=7)
        fig.tight_layout(); fig.savefig(f"{a.out}_{g}.png", dpi=80); plt.close(fig)
        print("wrote", f"{a.out}_{g}.png", len(L), "samples")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank"); b.add_argument("--records", required=True); b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=4); b.add_argument("--allow-mismatch", action="store_true")
    g = sub.add_parser("gen"); g.add_argument("mode", choices=["analytic", "ref"]); g.add_argument("--records", required=True)
    g.add_argument("--bank"); g.add_argument("--n", type=int, required=True); g.add_argument("--seed", type=int, required=True)
    g.add_argument("--out", required=True); g.add_argument("--workers", type=int, default=4); g.add_argument("--chunk", type=int, default=50)
    g.add_argument("--holdout", default="")
    p = sub.add_parser("predict"); p.add_argument("--models", required=True); p.add_argument("--labels", default="all")
    p.add_argument("--frames", default="fit"); p.add_argument("--max-frames", type=int, default=0); p.add_argument("--refine", type=int, default=30)
    p.add_argument("--k", type=int, default=4); p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--rb-late", type=float, default=0.8, help="lower bound of the base density for nominal >= 8 nm (0: none)")
    c = sub.add_parser("compare"); c.add_argument("--pred", nargs="+", required=True); c.add_argument("--records", required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("maps"); m.add_argument("--pred", required=True); m.add_argument("--labels", default="all"); m.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(bank=cmd_bank, gen=cmd_gen, predict=cmd_predict, compare=cmd_compare, maps=cmd_maps)[a.cmd](a)


if __name__ == "__main__":
    main()
