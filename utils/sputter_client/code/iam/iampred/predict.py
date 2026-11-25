'Apply a trained family network to measured windows and render every candidate back with the forward model.'

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import features as FE
from . import kinput as KI
from . import live as LV
from . import observe as OB
from . import registry as REG
from . import references as RF
from . import runner

HERE = Path(__file__).resolve().parent
TOL = dict(lnD=0.05, cov=0.03, lnR=0.10)
OFFSETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
S_LN = 0.15
S_FLOOR = {"": 1.0 / 3.0, "nm": 0.5}
GROW = 10.0


def _features(args):
    cfg, scan, times = args[:3]
    total = args[3] if len(args) > 3 else len(times)
    from . import adapter as A
    out = []
    s = A.get_scan(scan)
    if str(s.metal) not in cfg["metals"]:
        return [dict(t=float(t), ok=False, reason=f"scan metal {s.metal} not in family metals {cfg['metals']}")
                for t in times]
    if cfg.get("hosts") and str(s.sub) not in cfg["hosts"]:
        return [dict(t=float(t), ok=False, reason=f"substrate {s.sub} not in family hosts {cfg['hosts']}") for t in times]
    info = None
    for t in times:
        try:
            with np.errstate(invalid="ignore"):
                ctx = A.Context(cfg, s, t)
            if not float(np.sum(ctx.w["E"])) > 0:
                raise ValueError("no frames in this time window")
            X = KI.image(s, ctx.w, np.where(ctx.w["mask"], ctx.w["Y"], 0.0), cfg["features"])
            S = KI.scalars(cfg, s, t, ctx.h_nom_ctx)
            if not (np.all(np.isfinite(X)) and np.all(np.isfinite(S))):
                raise ValueError("network input is not finite")
            if info is None:
                info = dict(pnames=ctx.pnames, lb=ctx.lb.tolist(), ub=ctx.ub.tolist(), metal=str(s.metal),
                            start=A.x_from_reference(ctx, {}).tolist(),
                            t_dep=float(s.t_dep), position_factor=("measured" if getattr(s, "prof", None) is not None
                                                                     else "not available (cfac = 1)"))
            mdl = getattr(ctx.f, "mdl", None)
            eps = float(getattr(mdl, "eps", 0.0) or 0.0)
            obs = OB.direct(s, ctx.w, eps)
            try:
                dm = ctx.M.dc(str(s.metal), f1_au=getattr(mdl, "f1", ctx.M.AU_F1_12))
                obs_f = OB.fringe(s, ctx.w, ctx.h_nom_ctx, 2.0 * float(np.real(dm)), float(ctx.M.K0), eps)
                obs_g = OB.fringe(s, ctx.w, ctx.h_nom_ctx, 2.0 * float(np.real(dm)), float(ctx.M.K0), eps, q_band=OB.GRAIN_Q)
            except Exception as e:
                obs_f = obs_g = dict(d_nm=None, contrast=None, status=f"not read ({type(e).__name__})")
            try:
                dm = ctx.M.dc(str(s.metal), f1_au=getattr(mdl, "f1", ctx.M.AU_F1_12))
                obs_y = OB.yoneda(s, ctx.w, 2.0 * float(np.real(dm)), eps)
                obs_yg = OB.yoneda(s, ctx.w, 2.0 * float(np.real(dm)), eps, q_band=OB.GRAIN_Q)
            except Exception as e:
                obs_y = obs_yg = dict(af_deg=None, rel=None, contrast=None, status=f"not read ({type(e).__name__})")
            out.append(dict(t=float(t), X=X, S=S, h_nom=ctx.h_nom_ctx, t_rel=float(t - s.t_dep), ok=True,
                            n_cells=int(ctx.f.ok.sum()), info=info, obs=obs, obs_fringe=obs_f, obs_grain=obs_g,
                            obs_yoneda=obs_y, obs_yoneda_grain=obs_yg))
        except (Exception, SystemExit) as e:
            out.append(dict(t=float(t), ok=False, reason=f"{type(e).__name__}: {str(e)[:160]}"))
        LV.progress("features", len(out), total, force=len(out) == len(times), family=cfg["family"])
    return out


def _wanted_live(cfg, ctx, x, m, p, h, rep):
    'The quantities of the main screen of one solution, for the live view (None when it is off or they cannot be'
    if LV.folder() is None:
        return None
    try:
        from . import present as PS
        return PS.wanted(ctx.pnames, [r["name"] for r in cfg["reported"]], np.asarray(x, float).tolist(), rep, h,
                         getattr(m, "t_b", 0.0), m)
    except Exception:
        return None


def _live_window(cfg, ctx, s, t, label, kind, mu, d5, values):
    'One window solved, for the live view (iampred.live): its quantities, measured and model image.'
    if values is None:
        return
    try:
        meas, model = LV.window_images(ctx, mu)
        LV.emit("window", dict(q=np.asarray(s.q, np.float32), af=np.asarray(s.af, np.float32), measured=meas,
                               model=model),
                family=cfg["family"], t=float(t), t_rel=LV.rounded(t - s.t_dep, 3), h_nom=LV.rounded(ctx.h_nom_ctx, 4),
                label=label, entry_kind=kind, fit_deviation=LV.rounded(d5, 3), values=values)
    except Exception:
        pass


def _render(args):
    'The candidates of every window of ``items`` drawn by the model (best one kept), then - with ``refine`` - each'
    cfg, scan, items, refine, outdir = args[:5]
    total = args[5] if len(args) > 5 else len(items)
    var = args[6] if len(args) > 6 else None
    from . import adapter as A
    s = A.get_scan(scan)
    rows, todo, vtodo = [], [], []
    seen = [0]
    for it in items:
        t = it["t"]
        ctx = A.Context(cfg, s, t)
        cands, shown = [], []
        for c in it["cands"]:
            x = np.asarray(c["x"], float)
            t0 = time.time()
            seen[0] += 1
            LV.progress("render", len(rows), total, family=cfg["family"], images=seen[0])
            try:
                (mu, _, _), m, p, h = ctx.render(x)
                d5, _ = ctx.dev5(mu)
                rep = ctx.reported(m, p, h)
            except (Exception, SystemExit) as e:
                cands.append(dict(weight=c["weight"], dev5=float("nan"), error=str(e)[:120]))
                shown.append(None)
                continue
            cands.append(dict(weight=c["weight"], dev5=d5, reported=rep.tolist(), x=x.tolist(), h_m=h,
                              seconds=time.time() - t0, mu=mu.astype(np.float32)))
            shown.append(_wanted_live(cfg, ctx, x, m, p, h, rep.tolist()))
        good = [c for c in cands if np.isfinite(c["dev5"])]
        row = dict(scan=scan, t=t, cands=[{k: v for k, v in c.items() if k != "mu"} for c in cands])
        if good:
            b = min(good, key=lambda c: c["dev5"])
            row["best"] = {k: v for k, v in b.items() if k != "mu"}
            row["best_rank"] = cands.index(b) + 1
            np.savez_compressed(Path(outdir) / "mu" / f"{scan}_{t:g}_best.npz", mu=b["mu"])

            _live_window(cfg, ctx, s, t, f"candidate {good.index(b) + 1}", "candidate", b["mu"], b["dev5"],
                         shown[cands.index(b)])
            if refine > 0:
                todo.append((row, t, row["best"]["x"]))
            if var:
                vtodo += [(row, t, v) for v in _variants(ctx, b["x"], var, row)]
        rows.append(row)
        LV.progress("render", len(rows), total, force=len(rows) == len(items), family=cfg["family"], images=seen[0])
    seen = [0]
    for n, (row, t, x_best) in enumerate(todo):
        t0 = time.time()
        ctx = A.Context(cfg, s, t)
        nf = len(ctx.names)
        start, budget = seen[0], refine * (nf + 1)
        _count_images(ctx, seen, lambda: LV.progress("refine", n, total, family=cfg["family"], evaluations=seen[0],
                                                     current=seen[0] - start, budget=budget))
        xb = np.asarray(x_best, float)
        ctx.render(xb)
        xr, _ = ctx.f.run(np.clip(xb[:nf], ctx.f.lb + 1e-9, ctx.f.ub - 1e-9), refine)
        xr = np.r_[xr, xb[nf:]]
        (mu, _, _), m, p, h = ctx.render(xr)
        d5, _ = ctx.dev5(mu)
        rep = ctx.reported(m, p, h)
        row["refined"] = dict(dev5=d5, reported=rep.tolist(), x=xr.tolist(), h_m=h,
                              seconds=time.time() - t0, nfev=refine)
        np.savez_compressed(Path(outdir) / "mu" / f"{scan}_{t:g}_refined.npz", mu=mu.astype(np.float32))
        _live_window(cfg, ctx, s, t, "refined", "refined", mu, d5, _wanted_live(cfg, ctx, xr, m, p, h, rep.tolist()))
        LV.progress("refine", n + 1, total, force=True, family=cfg["family"], evaluations=seen[0], current=0,
                    budget=budget)
    _refine_variants(cfg, s, vtodo, var)
    return rows


def _refine_variants(cfg, s, vtodo, var):
    '--variant: each variant of vtodo [(row, t, variant)] refined on its own (var["refine"]); v["refined"].'
    from . import adapter as A
    for n, (row, t, v) in enumerate(vtodo):
        LV.progress("refine", n, len(vtodo), family=cfg["family"])
        ctx = A.Context(cfg, s, t)
        nf = len(ctx.names)
        t0 = time.time()
        try:
            xv = np.asarray(v["x"], float)
            ctx.render(xv)
            xr, _ = ctx.f.run(np.clip(xv[:nf], ctx.f.lb + 1e-9, ctx.f.ub - 1e-9), var["refine"])
            xr = np.r_[xr, xv[nf:]]
            (mu, _, _), m, p, h = ctx.render(xr)
            d5, _ = ctx.dev5(mu)
            v["refined"] = dict(dev5=d5, reported=ctx.reported(m, p, h).tolist(), x=xr.tolist(), h_m=h,
                                seconds=time.time() - t0, nfev=var["refine"])
        except (Exception, SystemExit) as e:
            v["refined"] = dict(dev5=float("nan"), error=str(e)[:120])
    if vtodo:
        LV.progress("refine", len(vtodo), len(vtodo), force=True, family=cfg["family"])


def _variant_pass(args):
    "--variant in a --reuse run (the second step of analyze's refine mode): the variants of the reused rows, drawn"
    cfg, scan, rows, var = args
    from . import adapter as A
    s = A.get_scan(scan)
    vtodo = []
    for row in rows:
        row.pop("variants", None)
        if "best" in row:
            vtodo += [(row, row["t"], v) for v in _variants(A.Context(cfg, s, row["t"]), start_entry(row)["x"], var, row)]
    _refine_variants(cfg, s, vtodo, var)
    return rows


def start_entry(row):
    "The window's best start: its best candidate, or a candidate of another network (--pool-from) with a lower dev5."
    return min([row["best"]] + [e for e in row.get("foreign", ()) if np.isfinite(e.get("dev5", np.nan))],
               key=lambda e: e["dev5"])


def other_network_items(dirs, rows, info):
    "--pool-from: {t: [dict(src, x)]} - the candidates of other networks' predict runs on this scan (DIR/state.json.gz)"
    by_t = {float(r["t"]): [] for r in rows if "best" in r}
    fams = {}
    for d in dirs:
        st = _load(Path(d) / "state.json.gz")
        fam_, pn = st["key"]["family"], list(st["info"]["pnames"])
        n_ = 0
        for r in st["rows"]:
            t = float(r["t"])
            if t not in by_t:
                continue
            for j, c in enumerate(r.get("cands", ())):
                if "x" not in c or not np.isfinite(c.get("dev5", np.nan)):
                    continue
                x = REG.map_params(pn, c["x"], info["pnames"], info["start"], info["lb"], info["ub"])[0]
                by_t[t].append(dict(src=f"{fam_} candidate {j + 1}", x=x.tolist()))
                n_ += 1
        fams[fam_] = n_
    return by_t, fams


def _foreign_pass(args):
    '--pool-from: the other networks\' candidates of every row drawn with this family\'s model (row["foreign"]).'
    cfg, scan, rows, by_t = args
    from . import adapter as A
    s = A.get_scan(scan)
    for n, row in enumerate(rows):
        LV.progress("render", n, len(rows), family=cfg["family"])
        es = by_t.get(float(row["t"])) or []
        if not es or "best" not in row:
            continue
        ctx = A.Context(cfg, s, row["t"])
        nf = len(ctx.names)
        out = []
        for e in es:
            xv = np.asarray(e["x"], float)
            xv[:nf] = np.clip(xv[:nf], ctx.f.lb + 1e-9, ctx.f.ub - 1e-9)
            try:
                (mu, _, _), m, p, h = ctx.render(xv)
                d5, _ = ctx.dev5(mu)
                out.append(dict(src=e["src"], dev5=d5, reported=ctx.reported(m, p, h).tolist(), x=xv.tolist(), h_m=h))
            except (Exception, SystemExit) as err:
                out.append(dict(src=e["src"], dev5=float("nan"), error=str(err)[:120]))
        row["foreign"] = out
    LV.progress("render", len(rows), len(rows), force=True, family=cfg["family"])
    return rows


def _pool_other_networks(cfg, a, rows, info, dirs, var, nw, notes):
    "--pool-from with --sequence: the other networks' candidates drawn in every window (_foreign_pass), then - with"
    t0 = time.time()
    by_t, fams = other_network_items(dirs, rows, info)
    LV.step("joint.pool", "running", summary="candidates of the other networks drawn with this model")
    chunks = [rows[i::nw] for i in range(nw)]
    rows = sum(runner.run(cfg, [(cfg, a.scan, c, by_t) for c in chunks if c], _foreign_pass, nw, data=a.data,
                          shadow=a.shadow), [])
    rows.sort(key=lambda r: r["t"])
    better = sum(1 for r in rows if "best" in r and start_entry(r) is not r["best"])
    print(f"--pool-from: {sum(fams.values())} candidates of {', '.join(fams) or 'no network'} drawn in "
          f"{sum(bool(r.get('foreign')) for r in rows)} windows, better than this network's best in {better} "
          f"({time.time() - t0:.0f} s)", flush=True)
    notes.append(f"candidates of other networks in the whole-scan pools: "
                 + ", ".join(f"{f_} ({n_})" for f_, n_ in fams.items()) + f"; the best start in {better} windows")
    a._pool_note = dict(families=fams, windows_better=better)
    if var:
        chunks = [rows[i::nw] for i in range(nw)]
        rows = sum(runner.run(cfg, [(cfg, a.scan, c, var) for c in chunks if c], _variant_pass, nw, data=a.data,
                              shadow=a.shadow), [])
        rows.sort(key=lambda r: r["t"])
        print(f"--variant: {sum(len(r.get('variants', ())) for r in rows)} start variants in "
              f"{sum(bool(r.get('variants')) for r in rows)} windows", flush=True)
    return rows


def _variants(ctx, x_best, var, row):
    "--variant: copies of the window's best candidate with the given parameters set (free parameters clipped into"
    ix = {n: i for i, n in enumerate(ctx.pnames)}
    xb = np.asarray(x_best, float)
    if var["min_fb"] > 0 and "fb" in ix and not xb[ix["fb"]] >= var["min_fb"]:
        return []
    nf, out = len(ctx.names), []
    for label, setv in var["variants"]:
        xv = xb.copy()
        for n_, v_ in setv:
            xv[ix[n_]] = v_
        xv[:nf] = np.clip(xv[:nf], ctx.f.lb + 1e-9, ctx.f.ub - 1e-9)
        try:
            (mu, _, _), m, p, h = ctx.render(xv)
            d5, _ = ctx.dev5(mu)
            e = dict(label=label, dev5=d5, reported=ctx.reported(m, p, h).tolist(), x=xv.tolist(), h_m=h)
        except (Exception, SystemExit) as err:
            e = dict(label=label, dev5=float("nan"), error=str(err)[:120], x=xv.tolist())
        row.setdefault("variants", []).append(e)
        if var["refine"] > 0 and np.isfinite(e["dev5"]):
            out.append(e)
    return out


def _count_images(ctx, seen, report):
    'For the live view only: count the model images the refinement computes (``seen[0]``) and call ``report``'
    if LV.folder() is None:
        return
    try:
        resid = ctx.f.resid

        def counted(*args, **kwargs):
            seen[0] += 1
            report()
            return resid(*args, **kwargs)

        ctx.f.resid = counted
    except Exception:
        pass


def _all_times(args):
    cfg, scan, step, nmax = args
    from . import adapter as A
    s = A.get_scan(scan)
    t_end = float(s.res.time[-1]) - 0.5
    t0 = max(float(s.t_dep) + 1.0, float(s.res.time[0]) + 0.5)
    tt = np.arange(np.ceil(t0), t_end, step)
    if nmax and len(tt) > nmax:
        tt = np.arange(np.ceil(t0), t_end, step * int(np.ceil(len(tt) / nmax)))
    return [float(t) for t in tt]


def time_path(pools, reps, names, tol, lam, tx=None, cont=(), lam_c=0.0, cap=np.inf, film=None, free=()):
    'Time-consistent choice of one entry per window (Viterbi). pools: time-ordered [(t, [entry])]; an entry holds'
    idx = {r["name"]: k for k, r in enumerate(reps)}
    mono = [(idx[n], tol.get(n, 0.05), n) for n in names if n in idx]
    use = [i for i, (_, es) in enumerate(pools) if es]
    if not use:
        return {}
    D = [np.array([e["dev5"] + e.get("prior", 0.0) for e in pools[i][1]], float) for i in use]
    R = [np.array([e["reported"] for e in pools[i][1]], float) for i in use]
    best, back = D[0].copy(), []
    for i in range(1, len(use)):
        A, B = R[i - 1], R[i]
        C = np.zeros((len(A), len(B)))
        for k, tl, nm in mono:
            if film is not None and film[use[i]] and nm in free:
                continue
            d = A[:, None, k] - B[None, :, k] - tl
            C += lam * np.where(np.isfinite(d) & (d > 0), d / tl, 0.0) ** 2
        if cont and lam_c > 0:
            dx = abs(float(tx[use[i]] - tx[use[i - 1]]))
            cc = np.zeros_like(C)
            for k, sc, g in cont:
                d = np.abs(A[:, None, k] - B[None, :, k]) - g * dx
                cc += np.where(np.isfinite(d) & (d > 0), d / sc, 0.0) ** 2
            C += lam_c * np.minimum(cc, cap)
        tot = best[:, None] + C
        jp = np.argmin(tot, axis=0)
        back.append(jp)
        best = tot[jp, np.arange(len(B))] + D[i]
    j = int(np.argmin(best))
    out = {}
    for i in range(len(use) - 1, -1, -1):
        out[pools[use[i]][0]] = j
        if i:
            j = int(back[i - 1][j])
    return out


def _entry(c, **kw):
    return dict(dev5=float(c["dev5"]), reported=[float(v) for v in c["reported"]], x=[float(v) for v in c["x"]],
                h_m=float(c["h_m"]), **kw)


def candidate_pools(rows):
    '[(t, entries)]: the rendered network candidates of every window (the pool of the plain time path).'
    return [(r["t"], [_entry(c, src=f"candidate {j + 1}", rank=j + 1) for j, c in enumerate(r["cands"])
                      if np.isfinite(c.get("dev5", np.nan))]) for r in rows]


def scan_pools(sq, use, transplant, log=print):
    "Pools of the whole-scan refinement: candidates + the window's own refined solution + the solutions (refined,"
    pools = candidate_pools(use)
    for (_, es), r in zip(pools, use):
        if "refined" in r and np.isfinite(r["refined"]["dev5"]):
            es.append(_entry(r["refined"], src="refined"))
        for e in r.get("foreign", ()):
            if np.isfinite(e.get("dev5", np.nan)):
                es.append(_entry(e, src=e["src"]))
        for v in r.get("variants", ()):
            if np.isfinite(v.get("dev5", np.nan)):
                es.append(_entry(v, src=f"variant {v['label']}"))
            rv = v.get("refined") or {}
            if np.isfinite(rv.get("dev5", np.nan)):
                es.append(_entry(rv, src=f"variant {v['label']} refined"))
    W = len(use)
    if not transplant or W < 2:
        return pools
    t0 = time.time()
    own = [r["refined"] if "refined" in r and np.isfinite(r["refined"]["dev5"]) else r["best"] for r in use]
    own = [min([o] + [v["refined"] for v in r.get("variants", ()) if np.isfinite((v.get("refined") or {}).get("dev5", np.nan))]
               + [e for e in r.get("foreign", ()) if np.isfinite(e.get("dev5", np.nan))],
               key=lambda e: e["dev5"]) for o, r in zip(own, use)]
    nf = sq.nf
    jobs, src = {}, {}
    for k in range(W):
        js = [k + sg * o for o in OFFSETS for sg in (-1, 1) if 0 <= k + sg * o < W]
        X = np.array([own[j]["x"] for j in js], float)
        X[:, :nf] = np.clip(X[:, :nf], sq.win[k]["lb"] + 1e-9, sq.win[k]["ub"] - 1e-9)
        jobs[k], src[k] = X, js
    res = sq.cross(jobs)
    n = 0
    for k, js in src.items():
        rp_, d5, hh, ok = res[k]
        for i, j in enumerate(js):
            if ok[i] and np.isfinite(d5[i]) and np.all(np.isfinite(rp_[i]) | np.isnan(rp_[i])):
                pools[k][1].append(dict(dev5=float(d5[i]), reported=[float(v) for v in rp_[i]],
                                        x=[float(v) for v in jobs[k][i]], h_m=float(hh[i]),
                                        src=f"from t={use[j]['t']:g}", offset=j - k))
                n += 1
    log(f"pool {sq.scan}: {n} solutions of other windows rendered at {W} windows ({time.time() - t0:.0f} s)")
    return pools


def smooth_scales(reps, use, override):
    '[(name, scale)] of the continuity / smoothness rules. ln quantities: S_LN; others: S_LN x max(median |value|'
    vals = np.array([(r["refined"] if "refined" in r else r["best"])["reported"] for r in use], float)
    out = []
    for k, rp in enumerate(reps):
        if rp["name"] in override:
            sc = override[rp["name"]]
        elif rp.get("unit", "").startswith("ln"):
            sc = S_LN
        else:
            v = np.abs(vals[:, k]); v = v[np.isfinite(v)]
            sc = S_LN * max(float(np.median(v)) if len(v) else 0.0, S_FLOOR.get(rp.get("unit", ""), 1.0 / 3.0))
        out.append((rp["name"], float(sc)))
    return out


def versus(seq, ref):
    'Whole-scan dev5 of a window against its own refined dev5: worse / better beyond max(1, 30 %), else similar.'
    m = max(1.0, 0.3 * ref)
    return "worse" if seq > ref + m else "better" if seq < ref - m else "similar"


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


def _dump(path, obj):
    import gzip
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh, default=_js)


def _load(path):
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def col(rp):
    'CSV column of a reported quantity: ln sizes are written in nm under their plain name (lnD -> D_nm).'
    n = rp["name"]
    if rp.get("unit", "").startswith("ln"):
        return (n[2:] if n.startswith("ln") else n) + "_nm"
    return n


def disp(rp, v):
    return float(np.exp(v)) if rp.get("unit", "").startswith("ln") and np.isfinite(v) else float(v)


def status(rp, t_rel):
    st = rp.get("status", "")
    vr = rp.get("valid_t_rel")
    if vr and not (vr[0] <= t_rel <= vr[1]):
        return f"conditional (outside validated {vr[0]:g}-{vr[1]:g} s)"
    return st


def NM_paths(nn_dir):
    from . import nnmodel as NM
    return NM.member_paths(nn_dir)


def polymer_film(cfg, scan, meta):
    'The polymer film layer of the forward model (2026-11-25, poly_thickness_20261125): {thickness_nm, source}, or None'
    env = REG.scan_env(cfg, scan)
    if env.get("IAM_PFILM") != "1":
        return None
    given = (meta or {}).get("poly_t")
    t = env.get("IAM_POLYT_NM")
    t = float(t) if t else (float(given) if given else None)
    from_intake = given is not None and float(given) > 0 and env.get("IAM_POLYT_NM") == repr(float(given))
    return dict(thickness_nm=t, source="given at intake" if from_intake else "the network family's setting")

def _client_entry(cfg, a, data_root):
    'Family entry of a scan outside the family: metal, onset override, per-scan forward settings (--like).'
    sj = data_root / a.scan / "scan.json"
    meta = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    env = {}
    over = {sc: v.get("env") for sc, v in cfg["scans"].items() if v.get("env")}
    if a.like:
        if a.like not in cfg["scans"]:
            raise SystemExit(f"--like {a.like}: not a scan of family {cfg['family']}")
        env.update(cfg["scans"][a.like].get("env") or {})
    elif over:
        raise SystemExit(f"{cfg['family']}: training scans have their own forward settings {over}; "
                         f"choose the one that fits the new sample with --like <scan>")
    if "t_dep" in meta or "pre_start" in meta:
        env["IAM_ONSET"] = "override"
    lam = meta.get("lam")
    if lam is not None and abs(float(lam) - REG.model_lam(cfg)) > 1e-9:
        problem = REG.lam_problem(meta.get("metal", "?"), float(lam))
        if problem:
            raise SystemExit(f"{a.scan}: {problem}")
        env["IAM_LAM_NM"] = repr(float(lam))
        if REG.lam_outside(meta.get("metal", "?"), float(lam), cfg):
            env["IAM_OPTICS"] = "table"
            print(f"{a.scan}: {float(lam):.5f} nm is away from the wavelength of {cfg['family']} "
                  f"({REG.model_lam(cfg):g} nm): tabulated optics at the scan's energy, the network gives starts only",
                  flush=True)
    pt = meta.get("poly_t")
    if pt is not None and float(pt) > 0 and meta.get("sub") not in (None, "Si", "TiO2"):
        env["IAM_PFILM"], env["IAM_POLYT_NM"] = "1", repr(float(pt))

    pm = data_root / a.scan / "pixmask.json"
    env["IAM_PIXMASK"] = json.loads(pm.read_text(encoding="utf-8"))["sha256"][:12] if pm.exists() else "none"
    return dict(metal=meta.get("metal", "?"), env=env, client=True), meta


FILM_AMOUNT_NM = 6.5
"""--film-stage amount: the nominal amount from which a window counts as film (the report's stage by amount)."""


def film_windows(feats, times, porous=False, how="reading", rows=None, pnames=None):
    "Whether each window of times is in the film stage. how: 'reading' - read from the counts alone as the report"
    from .report import _stages
    by = {float(d["t"]): d for d in feats}
    if how == "amount":
        return [float(by.get(float(t), {}).get("h_nom", 0.0)) >= FILM_AMOUNT_NM for t in times]
    if how == "fb":
        k = list(pnames).index("fb")
        best = {float(r["t"]): r["best"]["x"][k] for r in rows if "best" in r}
        return [best.get(float(t), 0.0) >= 0.5 for t in times]
    lines = []
    for t in times:
        d = by.get(float(t), {})
        L = {"h_nom": float(d.get("h_nom", 0.0))}
        of, og = d.get("obs_fringe") or {}, d.get("obs_grain") or {}
        if of.get("status"):
            L["obs_fringe_status"] = of["status"]
        if of.get("d_nm"):
            L["obs_d_fringe_nm"] = of["d_nm"]
        if og.get("status"):
            L["obs_h_grain_status"] = og["status"]
        if og.get("d_nm"):
            L["obs_h_grain_nm"] = og["d_nm"]
        lines.append(L)
    return [st == "film" for st, _ in _stages(lines, porous)]


def parse_variants(specs, pnames, refine, min_fb):
    '--variant NAME=VALUE,... (repeatable) -> dict(variants=[(label, [(name, value)])], refine, min_fb) for _render,'
    if not specs:
        return None
    out = []
    for spec in specs:
        setv = []
        for it in (x_ for x_ in spec.split(",") if x_):
            n_, _, v_ = it.partition("=")
            if n_ not in pnames:
                raise SystemExit(f"--variant {n_}: not a parameter of the forward model ({', '.join(pnames)})")
            setv.append((n_, float(v_)))
        out.append(("+".join(n_ for n_, _ in setv), setv))
    return dict(variants=out, refine=int(refine), min_fb=float(min_fb))


def start_overrides(xs, times, pnames, start_file, fluct, notes):
    'Start vectors of the whole-scan refinement after the research options (2026-11-23, conform_20261123):'
    import hashlib
    xs = [np.asarray(x, float).copy() for x in xs]
    ix = {n: i for i, n in enumerate(pnames)}
    if start_file:
        raw = Path(start_file).read_bytes()
        table = json.loads(raw.decode("utf-8"))
        table = table.get("windows", table)
        done = []
        for k, t in enumerate(times):
            got = table.get(f"{t:g}") or table.get(str(t)) or table.get(str(float(t)))
            if got:
                for n, v in got.items():
                    if n in ix:
                        xs[k][ix[n]] = float(v)
                done.append(float(t))
        notes["sequence_start"] = dict(file=Path(start_file).name, sha256=hashlib.sha256(raw).hexdigest(), windows=done)
    if fluct:
        if "lnRP" not in ix or "lnAP" not in ix or "fb" not in ix:
            raise SystemExit("--fluct-start: the family has no whole-film density fluctuation (lnRP, lnAP) or no base share fb")
        rp, ap_ = (float(v) for v in fluct.split(","))
        on = []
        for k, t in enumerate(times):
            if xs[k][ix["fb"]] >= 0.5:
                xs[k][ix["lnRP"]], xs[k][ix["lnAP"]] = np.log(rp), np.log(ap_)
                on.append(float(t))
        notes["fluct_start"] = dict(R_P_nm=rp, A_P=ap_, rule="fb >= 0.5 in the start", windows=on)
    return xs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--model", default=None, help="model.npz, or several comma-separated (a seed ensemble: candidates "
                    "pooled); default: $IAMPRED_MODELS/<family>/nn/model.npz and any model_*.npz beside it")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", default=None)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-windows", type=int, default=0, help="--times all: at most this many windows; a longer scan "
                    "gets a multiple of --step (0 = no limit)")
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--sequence", type=int, default=0, help="whole-scan refinement from the time path: maximum number "
                    "of iterations (all windows together; scan-level mass factor and onset shift, no-decrease rule, "
                    "smoothness in ln time)")
    ap.add_argument("--smooth-weight", type=float, default=1.0, help="--sequence: weight of the continuity cost of the "
                    "time path and of the smoothness rows of the whole-scan refinement (dev5 units; 0 = off)")
    ap.add_argument("--smooth", default="", help="--sequence: NAME=SCALE,... overrides the scale of a reported quantity "
                    "(stored transform; 0 = quantity left free)")
    ap.add_argument("--jump-cost", type=float, default=25.0, help="--sequence: largest continuity cost of one step of the "
                    "time path = summed dev5 a change of branch has to gain")
    ap.add_argument("--growth-prior", default=None, help="--sequence: the growth laws of the metal on its substrate as a "
                    "prior on D and R before the film forms (iampred.prior): 'default' = the package's "
                    "process_reference.json, or a reference file of that layout; off when not given")
    ap.add_argument("--growth-prior-weight", type=float, default=1.0, help="--growth-prior: weight (1: a one-sigma "
                    "departure costs one unit of dev5 per window and law)")
    ap.add_argument("--growth-prior-min-scans", type=int, default=2, help="--growth-prior: scans a law needs to be "
                    "used (1: a law of one scan, its spread then only the floors)")
    ap.add_argument("--far-weight", type=float, default=1.0, help="--sequence: weight of the no-decrease rule against "
                    "earlier windows (2 ... 64 places back, beyond one scale; dev5 units; 0 = neighbouring windows only; "
                    "default 1 since 2026-11-20: growth is additive)")
    ap.add_argument("--fringe-weight", type=float, default=0.0, help="--sequence: weight lambda of the fringe term "
                    "(iampred.sequence: each window costs dev5 + lambda x fringe residual^2, the detrended log ratio "
                    "measured / model along the exit angle in q_y 0.1-0.4 nm^-1; 2026-11-23); 0 = off (default)")
    ap.add_argument("--fluct-start", default=None, help="--sequence: R_P,A_P (nm, -): the whole-film density "
                    "fluctuation set on in the start of every film-stage window (base share fb >= 0.5) of a family with "
                    "lnRP / lnAP; the optimiser does not leave the off state by itself (iam_fit4); default: the start as "
                    "chosen")
    ap.add_argument("--sequence-start", default=None, help="--sequence: JSON {t: {parameter: value}} - start values "
                    "of chosen windows that replace those of the time path (research warm starts); applied before "
                    "--fluct-start")
    ap.add_argument("--variant", action="append", default=[], help="NAME=VALUE,... (parameters in their stored "
                    "transform, e.g. lnRP=1.7918,lnAP=0): in every window whose best candidate has fb >= --variant-min-fb, "
                    "a copy of it with these values is drawn and (--variant-refine) refined on its own; the whole-scan "
                    "time path chooses among them and the rest (a start rule for terms the optimiser does not switch on "
                    "by itself; 2026-11-23, conform_20261123 round 9). Repeatable; default: none")
    ap.add_argument("--variant-refine", type=int, default=0, help="--variant: refinement of each variant (as --refine); "
                    "0 = drawn only")
    ap.add_argument("--variant-min-fb", type=float, default=0.5, help="--variant: only windows whose best candidate has "
                    "a base share fb at least this (film stage); 0 = every window")
    ap.add_argument("--film-free", default="", help="--sequence: NAME,... of the no-decrease quantities not charged for a "
                    "step into a film-stage window (read from the counts as the report does); e.g. lnD_mass,cov,lnR_mass: on a "
                    "closed film the grains are relief and need not keep growing like islands (2026-11-23); default none")
    ap.add_argument("--film-stage", choices=["reading", "amount", "fb"], default="reading", help="--film-free: how a "
                    "window is put in the film stage - reading: from the counts as the report does; amount: nominal amount "
                    f">= {FILM_AMOUNT_NM} nm; fb: best candidate with fb >= 0.5")
    ap.add_argument("--transplant", type=int, default=1, help="--sequence: 1 = the solutions of neighbouring windows "
                    "join the pool of a window, 0 = candidates and own refined solution only")
    ap.add_argument("--no-network-step", action="store_true", help="a GUI's step tree: this run only refines (analyze "
                    "runs the chosen network a second time), the network steps were reported by the first run")
    ap.add_argument("--reuse", action="store_true", help="take features, candidates and per-window refinement from "
                    "OUT/state.json.gz when it was written for the same model, windows, --k and --refine")
    ap.add_argument("--pool-from", default="", help="--sequence: DIR,... - predict outputs of other networks on the same "
                    "scan (DIR/state.json.gz): their candidates, parameters carried over by name (registry.map_params), "
                    "are drawn with this family's model in the windows both have and join the whole-scan pools; the "
                    "start variants start from the best of this family's and these candidates (2026-11-24, conform round 18)")
    ap.add_argument("--like", default=None)
    ap.add_argument("--data")
    ap.add_argument("--shadow")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--monotone", default=None, help="reported quantities that may not decrease (time path); "
                    "default: the family's \"monotone\" list, else lnD,cov")
    ap.add_argument("--path-weight", type=float, default=1.0, help="penalty weight of the no-decrease rule in the "
                    "time path and in --sequence (dev5 units)")
    a = ap.parse_args()
    t_start = time.time()
    cfg = REG.load(a.family)
    netfam = REG.network_family(cfg)
    ncfg = cfg if netfam == cfg["family"] else REG.load(netfam)
    if a.model is None:
        if not os.environ.get("IAMPRED_MODELS"):
            raise SystemExit("give --model (or set IAMPRED_MODELS, as run.cmd / run.sh of the client package do)")
        a.model = ",".join(str(p_) for p_ in NM_paths(Path(os.environ["IAMPRED_MODELS"]) / netfam / "nn"))
    from . import nnmodel as NM
    model_paths = [p_ for p_ in str(a.model).split(",") if p_]
    a.model = model_paths[0]
    P = NM.load_many(model_paths, a.k)
    n_net = getattr(P, "n_members", 1)
    meta = P.meta
    names = [r["name"] for r in cfg["reported"]]
    bad = [k for k, x_, y_ in (("family", meta.get("family"), netfam), ("reported", meta["reported"], names),
                               ("scalars", meta.get("scalars"), FE.scalar_names(cfg)),
                               ("features", meta.get("features"), cfg["features"])) if x_ != y_]
    if bad:
        raise SystemExit(f"model / family config mismatch in {bad}: retrain the model for this config")
    warn, notes = [], []
    if n_net > 1:
        notes.append(f"seed ensemble: candidates of {n_net} networks pooled ({a.k} from each)")
    if meta["family_hash"] != ncfg["family_hash"]:
        warn.append("family config changed since training (retrain recommended)")
    if meta["forward_hash"] != cfg["forward_hash"]:
        cp_ = Path(a.model).parent.parent / f"check_{netfam}.json"
        ck = json.loads(cp_.read_text(encoding="utf-8")) if cp_.exists() else {}
        if ck.get("forward_hash") == cfg["forward_hash"] and ck.get("max_rel_diff") is not None and ck["max_rel_diff"] <= 1e-6:
            notes.append("forward-model code changed since training; the current code reproduces every reference fit "
                         f"(check max rel dev5 difference {ck['max_rel_diff']:.1e}): treated as equivalent")
        else:
            warn.append("forward-model code changed since training: candidates are rendered with the current code")
    if meta.get("pred_hash") and meta["pred_hash"] != REG.pred_hash(ncfg):
        warn.append("iampred sample / feature code changed since training")
    for w_ in warn:
        print("WARNING:", w_)
    out = Path(a.out).resolve(); (out / "mu").mkdir(parents=True, exist_ok=True)
    site = runner.site_env(cfg, a.data, a.shadow)
    client = a.scan not in cfg["scans"]
    scan_meta = {}
    if client:
        cfg = dict(cfg); cfg["scans"] = dict(cfg["scans"])
        cfg["scans"][a.scan], scan_meta = _client_entry(cfg, a, Path(site.get("IAM_DATA", ".")))
        refs = {}
    else:
        refs, _ = RF.load(cfg)
    ai_range = ncfg.get("alpha_i_range_deg")
    ai = scan_meta.get("ai")
    if ai_range and ai is not None and not float(ai_range[0]) - 1e-6 <= float(ai) <= float(ai_range[1]) + 1e-6:
        warn.append(f"incidence angle {float(ai):g} deg outside the network's training range "
                    f"{float(ai_range[0]):g}-{float(ai_range[1]):g} deg: its starting solutions may be poor")
        print("WARNING:", warn[-1])
    how = a.times or ("all" if client else "refs")
    if how == "refs":
        times = [r["t"] for r in refs.get(a.scan, [])]
    elif how == "all":
        times = runner.run(cfg, [(cfg, a.scan, a.step, a.max_windows)], _all_times, 1, data=a.data, shadow=a.shadow)[0]
    else:
        times = [float(v) for v in how.split(",")]
    if not times:
        raise SystemExit(f"{a.scan}: no windows for --times {how} (new scans have no reference windows: use all or t1,t2)")
    LV.emit("windows", family=cfg["family"], total=len(times), refine=a.refine)
    if not a.no_network_step:
        tt = [float(t) for t in times]
        LV.step("windows", "done", detail=LV.trace("windows", dict(times=tt, step=a.step, max_windows=a.max_windows)),
                summary=(f"{len(tt)} windows of 1 s, {tt[0]:g}-{tt[-1]:g} s" + (f", every {tt[1] - tt[0]:g} s" if len(tt) > 1 else "")))
        LV.step(f"networks.{cfg['family']}", "running")
    t_start_ = time.time()
    nw = max(1, min(a.workers, len(times)))
    key = json.loads(json.dumps(dict(family=cfg["family"], scan=a.scan, family_hash=cfg["family_hash"],
                                     forward_hash=cfg["forward_hash"], model="+".join(_sha(p_) for p_ in model_paths), k=a.k, refine=a.refine,
                                     like=a.like, times=[float(t) for t in times])))
    pool_dirs = [Path(d_) for d_ in a.pool_from.split(",") if d_]
    st = None
    if a.reuse and (out / "state.json.gz").is_file():
        st = _load(out / "state.json.gz")
        if st.get("key") != key:
            print("--reuse: state.json.gz was written for another model, window list, --k or --refine: computed again",
                  flush=True)
            st = None
    if st is None:
        chunks = [times[i::nw] for i in range(nw)]
        feats = sum(runner.run(cfg, [(cfg, a.scan, c, len(times)) for c in chunks if c], _features, nw, data=a.data,
                               shadow=a.shadow), [])
        feats.sort(key=lambda d: d["t"])
        okf = [d for d in feats if d["ok"]]
        if not okf:
            with open(out / "windows.csv", "w", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh); wr.writerow(["scan", "t", "note"])
                for d in feats:
                    wr.writerow([a.scan, d["t"], d["reason"]])
            raise SystemExit(f"{a.scan}: no usable window; reasons: " + "; ".join(sorted({d['reason'] for d in feats})))
        info = okf[0]["info"]
        if netfam == cfg["family"]:
            mism = [k for k, x_, y_ in (("params", meta["params"], info["pnames"]),
                                        ("bounds", np.round(meta["lb"] + meta["ub"], 9).tolist(),
                                         np.round(info["lb"] + info["ub"], 9).tolist())) if x_ != y_]
        else:
            bnd = lambda nm_, lb_, ub_: dict((n_, (round(l_, 9), round(u_, 9))) for n_, l_, u_ in zip(nm_, lb_, ub_))
            bn, bi = bnd(meta["params"], meta["lb"], meta["ub"]), bnd(info["pnames"], info["lb"], info["ub"])
            mism = [f"bounds of {n_}" for n_ in bn if n_ in bi and bn[n_] != bi[n_]]
        if mism:
            raise SystemExit(f"model / forward-model mismatch in {mism}: the forward code or switches changed since training")
        X = np.stack([d["X"] for d in okf]); S = np.stack([d["S"] for d in okf])
        LV.progress("network", 0, len(okf), force=True, family=cfg["family"])
        cands = P.candidates(X, S)
        if netfam != cfg["family"]:
            for cs_ in cands:
                for c_ in cs_:
                    c_["x"] = REG.map_params(meta["params"], c_["x"], info["pnames"], info["start"], info["lb"],
                                             info["ub"])[0]
        q = P.intervals(X, S, n=300)
        LV.progress("network", len(okf), len(okf), force=True, family=cfg["family"])
        items = [dict(t=d["t"], cands=[dict(weight=c["weight"], x=c["x"].tolist()) for c in cs[:a.k * n_net]])
                 for d, cs in zip(okf, cands)]
        chunks = [items[i::nw] for i in range(nw)]
        var = parse_variants(a.variant, info["pnames"], a.variant_refine, a.variant_min_fb)
        var_now = None if pool_dirs and a.sequence > 0 else var
        rows = sum(runner.run(cfg, [(cfg, a.scan, c, a.refine, str(out), len(items)) + ((var_now,) if var_now else ())
                                    for c in chunks if c], _render, nw, data=a.data, shadow=a.shadow), [])
        rows.sort(key=lambda r: r["t"])
        _dump(out / "state.json.gz", dict(key=key, info=info, q=q, rows=rows,
                                          feats=[{k: v for k, v in d.items() if k not in ("X", "S", "info")} for d in feats]))
    else:
        feats, info, q, rows = st["feats"], st["info"], np.array(st["q"], float), st["rows"]
        for r in rows:
            r.pop("seq", None)
            r.pop("foreign", None)
        okf = [d for d in feats if d["ok"]]
        print(f"--reuse: {len(okf)} windows (features, candidates, per-window refinement) from state.json.gz", flush=True)
        var = parse_variants(a.variant, info["pnames"], a.variant_refine, a.variant_min_fb)
        if var and a.sequence > 0 and not pool_dirs:
            LV.step("joint.pool", "running", summary="film windows: start variants refined on their own")
            t_var = time.time()
            chunks = [rows[i::nw] for i in range(nw)]
            rows = sum(runner.run(cfg, [(cfg, a.scan, c, var) for c in chunks if c], _variant_pass, nw, data=a.data,
                                  shadow=a.shadow), [])
            rows.sort(key=lambda r: r["t"])
            print(f"--variant: {sum(len(r.get('variants', ())) for r in rows)} start variants in "
                  f"{sum(bool(r.get('variants')) for r in rows)} windows ({time.time() - t_var:.0f} s)", flush=True)
    if pool_dirs and a.sequence > 0:
        rows = _pool_other_networks(cfg, a, rows, info, pool_dirs, var, nw, notes)
    if netfam != cfg["family"]:
        new_ = [n_ for n_ in info["pnames"] if n_ not in meta["params"]]
        dropped_ = [n_ for n_ in meta["params"] if n_ not in info["pnames"]]
        notes.append(f"candidates from the network of {netfam}, parameters carried over by name; dropped (not in this "
                     f"family's model): {', '.join(dropped_) or 'none'}; new, at their start values: "
                     f"{', '.join(new_) or 'none'}")
    reps = cfg["reported"]
    mono = ([m_ for m_ in a.monotone.split(",") if m_] if a.monotone is not None
            else list(cfg.get("monotone", ["lnD", "cov"])))
    use = [r for r in rows if "best" in r]
    free = [n_ for n_ in a.film_free.split(",") if n_]
    if [n_ for n_ in free if n_ not in mono]:
        raise SystemExit(f"--film-free {a.film_free}: only no-decrease quantities ({', '.join(mono)})")
    film_ = film_windows(feats, [r["t"] for r in use], porous=cfg["family"].startswith("tio2"), how=a.film_stage,
                         rows=use, pnames=info["pnames"]) if free and use else None
    seq, scales = None, []
    if a.sequence > 0 and use:
        from . import sequence as SQ
        over = {}
        for it in (x_ for x_ in a.smooth.split(",") if x_):
            n_, _, v_ = it.partition("=")
            if n_ not in names:
                raise SystemExit(f"--smooth {n_}: not a reported quantity of {cfg['family']} ({', '.join(names)})")
            over[n_] = float(v_)
        scales = smooth_scales(reps, use, over) if a.smooth_weight > 0 else []
        def seq_log(m_):
            print(m_, flush=True)
            it_ = re.search(r": iteration (\d+) ", m_)
            if it_:
                LV.progress("sequence", int(it_.group(1)), a.sequence, family=cfg["family"])

        sq = SQ.Sequence(cfg, a.scan, [r["t"] for r in use], a.workers, data=a.data, shadow=a.shadow, log=seq_log,
                         fringe_weight=a.fringe_weight)


        from .present import wanted as _wanted
        rn_ = [r["name"] for r in reps]
        tt_ = [r["t"] for r in use]

        def on_iter(k, state, xs_, seconds):
            nf_ = len(xs_[0])
            wins = []
            for i, t_ in enumerate(tt_):
                v_ = _wanted(info["pnames"][:nf_], rn_, list(xs_[i]), [float(q_) for q_ in sq._rep[i]], float(sq._h[i]))
                wins.append(dict(t=float(t_), fit_deviation=LV.rounded(sq._d5[i], 3),
                                 **{kk: vv for kk, vv in v_.items() if kk in ("R_nm", "H_nm", "D_nm", "gap_nm", "width",
                                                                             "coverage", "amount_nm", "base_nm")}))
            LV.emit("joint", family=cfg["family"], iteration=int(k), total=int(a.sequence), seconds=LV.rounded(seconds, 1),
                    median=LV.rounded(state["median_dev5"], 3), windows=wins)
            history.append(dict(iteration=int(k), seconds=LV.rounded(seconds, 1), median=LV.rounded(state["median_dev5"], 3),
                                sum_dev5=LV.rounded(state["sum_dev5"], 3), penalty=LV.rounded(state["penalty"], 3),
                                windows=wins))
            LV.trace("joint", dict(family=cfg["family"], total=int(a.sequence), iterations=history))

        def on_progress(phase, done, n, seconds):
            LV.emit("joint_progress", family=cfg["family"], iteration=int(sq.njac) + (1 if phase == "derivatives" else 0),
                    total=int(a.sequence), phase=phase, done=int(done), windows=int(n), seconds=LV.rounded(seconds, 1))

        history = []
        setup_notes = {}
        sq.on_iter, sq.on_progress = on_iter, on_progress
        try:
            LV.emit("joint_stage", family=cfg["family"], stage="pool", windows=len(use), total=int(a.sequence))
            LV.step("joint.pool", "running")
            t_pool = time.time()
            pools = scan_pools(sq, use, a.transplant, sq.log)
            LV.step("joint.pool", "done", summary=f"{len(use)} windows, {sum(len(es) for _, es in pools)} solutions tried "
                                                  f"in {time.time() - t_pool:.0f} s")
            LV.emit("joint_stage", family=cfg["family"], stage="fit", windows=len(use), total=int(a.sequence))
            LV.step("joint.fit", "running")
            cont = [(k, sc, GROW * sc) for k, (_, sc) in enumerate(scales) if sc > 0]
            prior = None
            if a.growth_prior:
                from . import prior as GP
                ref = Path(__file__).with_name("process_reference.json") if a.growth_prior == "default" else Path(a.growth_prior)
                prior = GP.GrowthPrior(cfg["family"], sq.meta.get("metal", ""), sq.meta.get("sub", ""), ref,
                                       a.growth_prior_weight, a.growth_prior_min_scans).bind(rn_)
                a.growth_prior_used = dict(prior.describe(), active=prior.active)
                print(f"growth prior: {prior.system}, {prior.n_scans} scans, laws "
                      + (", ".join(f"{law['quantity']} = {float(np.exp(law['ln_v1'])):.2f} h^{law['p']:.2f}" for law in prior.laws)
                         or "none") + (f" ({prior.note})" if prior.note else "")
                      + (f", film at {prior.h_film:.2f} nm" if prior.h_film else ""), flush=True)
                if prior.active:
                    for k_, (_, es) in enumerate(pools):
                        for e in es:
                            e["prior"] = prior.cost(e["h_m"], e["reported"], bool(sq.after[k_]))
            path = time_path(pools, reps, mono, TOL, a.path_weight, sq.tx, cont, a.smooth_weight, a.jump_cost,
                             film=film_, free=free)
            chosen = {t: es[path[t]] for t, es in pools}
            _dump(out / "pool.json.gz", dict(key=key, t=[t for t, _ in pools], tx=sq.tx, scales=scales, growth=GROW,
                                             jump_cost=a.jump_cost, smooth_weight=a.smooth_weight, monotone=mono,
                                             path_weight=a.path_weight, tol=TOL, pools=[es for _, es in pools],
                                             chosen=[path[t] for t, _ in pools]))
            x_start = start_overrides([chosen[t]["x"] for t, _ in pools], [t for t, _ in pools], list(info["pnames"]),
                                      a.sequence_start, a.fluct_start, setup_notes)
            if a.variant:
                setup_notes["variants"] = dict(specs=list(a.variant), refine=a.variant_refine, min_fb=a.variant_min_fb,
                                               windows=sorted(float(r["t"]) for r in use if r.get("variants")),
                                               start_from={f"{t:g}": chosen[t].get("src") for t, _ in pools})
            if getattr(a, "_pool_note", None):
                setup_notes["pool_from"] = dict(a._pool_note, start_from_other_network=sorted(
                    float(t) for t, _ in pools if not str(chosen[t].get("src", "")).startswith(("candidate", "refined", "variant", "from t="))))
            if free:
                setup_notes["film_free"] = dict(quantities=free, rule=f"film stage: {a.film_stage}",
                                                windows=[float(r["t"]) for r, f in zip(use, film_) if f])
            a._setup_notes = setup_notes
            sw, seq = SQ.refine(sq, x_start, reps, mono, TOL, a.path_weight, scales,
                                a.smooth_weight, a.sequence, out, a.far_weight, prior, film=film_, film_free=free)
            sh_ = seq.get("shared") or {}
            LV.step("joint.fit", "done", detail="joint.json" if history else None,
                    summary=f"{len(history)} iterations, median fit deviation {seq['end']['median_dev5']:.1f}"
                            + (f", metal amount {sh_['kappa']:.2f} x nominal" if sh_.get("kappa") else "")
                            + f", {(time.time() - t_pool) / 60:.0f} min")
        except BaseException:
            LV.step("joint.fit", "failed", summary="stopped: see the log of this step")
            raise
        finally:
            sq.close()
        for r in rows:
            if r["t"] in sw:
                r["seq"] = sw[r["t"]]
                if "refined" in r:
                    r["seq"]["versus"] = versus(r["seq"]["dev5"], r["refined"]["dev5"])

        _dump(out / "state.json.gz", dict(key=key, info=info, q=q, rows=rows,
                                          feats=[{k: v for k, v in d.items() if k not in ("X", "S", "info")} for d in feats]))
        if seq["shared"] is not None:
            nf = len(info["pnames"]) - 1
            tt = [r["t"] for r in rows if "seq" in r]
            rec = dict(scan=a.scan, family=cfg["family"], source="iampred.predict --sequence", model=str(a.model),
                       family_hash=cfg["family_hash"], forward_hash=cfg["forward_hash"], windows=tt,
                       shared_names=["lnkap", "dt"], local_names=list(info["pnames"][:nf]),
                       stages=dict(seq=dict(
                           x=[float(np.log(seq["shared"]["kappa"])), seq["shared"]["dt_s"]] + [float(v) for t in tt for v in sw[t]["x"][:nf]],
                           sum_dev5=seq["end"]["sum_dev5"], penalty=seq["end"]["penalty"],
                           windows=[dict(t=t, h_m=sw[t]["h_m"], dev5_per_cell=sw[t]["dev5"]) for t in tt])))
            (out / "sequence_record.json").write_bytes((json.dumps(rec, indent=1) + chr(10)).encode("utf-8"))
    else:
        pools = candidate_pools(rows)
        path = time_path(pools, reps, mono, TOL, a.path_weight)
        chosen = {t: es[path[t]] for t, es in pools if t in path}
    chk = None
    cp = Path(a.model).parent.parent / f"check_{cfg['family']}.json"
    if cp.exists():
        chk = json.loads(cp.read_text(encoding="utf-8"))
    chkmap = {(w["scan"], round(w["t"], 3)): w for w in (chk["windows"] if chk else [])}
    header = ["scan", "t", "t_rel", "h_nom", "n_cells", "obs_q_knee", "obs_L_knee_nm", "obs_knee_status",
              "obs_d_fringe_nm", "obs_fringe_contrast", "obs_fringe_status", "obs_h_grain_nm", "obs_h_grain_contrast",
              "obs_h_grain_status", "obs_yoneda_deg", "obs_yoneda_rel", "obs_yoneda_status", "obs_yoneda_grain_deg",
              "obs_yoneda_grain_rel", "obs_yoneda_grain_status", "best_rank", "best_weight", "best_dev5"]
    header += [f"cand{j + 1}_{k}" for j in range(a.k * n_net) for k in ("weight", "dev5")]
    for r in reps:
        header += [col(r), f"{col(r)}_net_p16", f"{col(r)}_net_p84", f"{col(r)}_status"]
    header += ["path_rank", "path_source", "path_dev5"] + [f"{col(r)}_path" for r in reps]
    if a.refine:
        header += ["refined_dev5"] + [f"{col(r)}_refined" for r in reps]
    if seq:
        header += ["seq_dev5"] + (["seq_vs_refined"] if a.refine else []) + [f"{col(r)}_seq" for r in reps]
    header += ["fit_dev5", "fit_dev5_record"] + [f"{col(r)}_fit" for r in reps] + ["note"]
    lines = []
    fi = {d["t"]: i for i, d in enumerate(okf)}
    rmap = {r["t"]: r for r in rows}
    for d in feats:
        if not d["ok"]:
            lines.append({"scan": a.scan, "t": d["t"], "note": d["reason"]})
            continue
        r = rmap[d["t"]]
        i = fi[d["t"]]
        L = {"scan": a.scan, "t": d["t"], "t_rel": round(d["t_rel"], 3), "h_nom": round(d["h_nom"], 4),
             "n_cells": d["n_cells"]}
        ob = d.get("obs")
        if ob:
            L["obs_knee_status"] = ob["status"]
            if ob.get("q_knee"):
                L["obs_q_knee"] = round(ob["q_knee"], 4); L["obs_L_knee_nm"] = round(ob["L_knee_nm"], 2)
        of = d.get("obs_fringe")
        if of:
            L["obs_fringe_status"] = of["status"]
            if of.get("d_nm"):
                L["obs_d_fringe_nm"] = round(of["d_nm"], 2); L["obs_fringe_contrast"] = of["contrast"]
        og = d.get("obs_grain")
        if og:
            L["obs_h_grain_status"] = og["status"]
            if og.get("d_nm"):
                L["obs_h_grain_nm"] = round(og["d_nm"], 2); L["obs_h_grain_contrast"] = og["contrast"]
        oy, oyg = d.get("obs_yoneda"), d.get("obs_yoneda_grain")
        if oy:
            L["obs_yoneda_status"] = oy["status"]
            if oy.get("af_deg"):
                L["obs_yoneda_deg"] = round(oy["af_deg"], 4); L["obs_yoneda_rel"] = round(oy["rel"], 3)
        if oyg:
            L["obs_yoneda_grain_status"] = oyg["status"]
            if oyg.get("af_deg"):
                L["obs_yoneda_grain_deg"] = round(oyg["af_deg"], 4); L["obs_yoneda_grain_rel"] = round(oyg["rel"], 3)
        for j, c in enumerate(r["cands"]):
            L[f"cand{j + 1}_weight"] = round(c["weight"], 4)
            L[f"cand{j + 1}_dev5"] = round(c["dev5"], 4) if np.isfinite(c["dev5"]) else ""
        if "best" in r:
            b = r["best"]
            L["best_rank"] = r["best_rank"]
            L["best_weight"] = round(b["weight"], 4); L["best_dev5"] = round(b["dev5"], 4)
            for k, rp in enumerate(reps):
                L[col(rp)] = disp(rp, b["reported"][k])
                L[f"{col(rp)}_net_p16"] = disp(rp, q[i, k, 1])
                L[f"{col(rp)}_net_p84"] = disp(rp, q[i, k, 3])
                L[f"{col(rp)}_status"] = status(rp, d["t_rel"])
        else:
            L["note"] = "no candidate could be rendered"
        if r["t"] in chosen:
            c = chosen[r["t"]]
            L["path_rank"] = c.get("rank", ""); L["path_source"] = c["src"]; L["path_dev5"] = round(c["dev5"], 4)
            for k, rp in enumerate(reps):
                L[f"{col(rp)}_path"] = disp(rp, c["reported"][k])
        if "refined" in r:
            L["refined_dev5"] = round(r["refined"]["dev5"], 4)
            for k, rp in enumerate(reps):
                L[f"{col(rp)}_refined"] = disp(rp, r["refined"]["reported"][k])
        if "seq" in r:
            L["seq_dev5"] = round(r["seq"]["dev5"], 4)
            L["seq_vs_refined"] = r["seq"].get("versus", "")
            for k, rp in enumerate(reps):
                L[f"{col(rp)}_seq"] = disp(rp, r["seq"]["reported"][k])
        cw = chkmap.get((a.scan, round(d["t"], 3)))
        if cw:
            L["fit_dev5"] = round(cw["dev5_render"], 4)
            L["fit_dev5_record"] = round(cw["dev5_record"], 4)
            for rp in reps:
                L[f"{col(rp)}_fit"] = disp(rp, cw["reported"][rp["name"]])
        lines.append(L)
    with open(out / "windows.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        wr.writeheader()
        for L in lines:
            wr.writerow(L)
    bd = [r["best"]["dev5"] for r in rows if "best" in r]
    c1 = [r["cands"][0]["dev5"] for r in rows if r["cands"] and np.isfinite(r["cands"][0]["dev5"])]
    pairs = [(r["best"]["dev5"], chkmap[(a.scan, round(r["t"], 3))]["dev5_render"]) for r in rows
             if (a.scan, round(r["t"], 3)) in chkmap and "best" in r]
    pd5 = [c["dev5"] for c in chosen.values()]
    srcs = {}
    for c in chosen.values():
        k_ = "other window" if c["src"].startswith("from") else c["src"].split()[0]
        srcs[k_] = srcs.get(k_, 0) + 1
    worse = [r["t"] for r in rows if r.get("seq", {}).get("versus") == "worse"]
    better = [r["t"] for r in rows if r.get("seq", {}).get("versus") == "better"]
    summ = dict(family=cfg["family"], scan=a.scan, new_scan=client, like=a.like, model=str(a.model), network_family=netfam,
                models=model_paths if n_net > 1 else None, n_networks=n_net, model_family_hash=meta["family_hash"], model_forward_hash=meta["forward_hash"],
                family_hash=cfg["family_hash"], forward_hash=cfg["forward_hash"], warnings=warn, notes=notes,
                metal=info["metal"], onset_s=info["t_dep"],
                onset_source=("scan.json (given at intake)" if client and "t_dep" in scan_meta else "forward-model rule"),
                position_factor=info["position_factor"], scan_json=scan_meta or None,
                polymer_film=polymer_film(cfg, a.scan, scan_meta),
                n_windows=len(feats), n_usable=len(okf), n_unusable_reasons=sorted({d["reason"] for d in feats if not d["ok"]}),
                k=a.k, refine_nfev=a.refine,
                median_dev5_first=float(np.median(c1)) if c1 else None, median_dev5_best=float(np.median(bd)) if bd else None,
                median_dev5_refined=float(np.median([r["refined"]["dev5"] for r in rows if "refined" in r])) if a.refine else None,
                median_dev5_path=float(np.median(pd5)) if pd5 else None, monotone=mono, path_weight=a.path_weight,
                path_sources=srcs,
                median_dev5_sequence=float(np.median([r["seq"]["dev5"] for r in rows if "seq" in r])) if seq else None,
                sequence=seq,
                sequence_setup=(dict(smooth_weight=a.smooth_weight, scales=dict(scales), growth_scales_per_ln_time=GROW,
                                     jump_cost=a.jump_cost, far_weight=a.far_weight,
                                     growth_prior=(dict(reference=a.growth_prior, weight=a.growth_prior_weight,
                                                        min_scans=a.growth_prior_min_scans,
                                                        prior=getattr(a, "growth_prior_used", None))
                                                   if a.growth_prior else None),
                                     transplant_offsets=list(OFFSETS) if a.transplant else [],
                                     **({"fringe_weight": a.fringe_weight} if a.fringe_weight > 0 else {}),
                                     **(getattr(a, "_setup_notes", None) or {}))
                                if seq else None),
                n_seq_worse=len(worse) if seq and a.refine else None, seq_worse_t=worse if seq and a.refine else None,
                n_seq_better=len(better) if seq and a.refine else None, reused_state=st is not None,
                median_ratio_best_to_fit=float(np.median([b_ / f_ for b_, f_ in pairs])) if pairs else None,
                reported_notes={col(rp): dict(status=rp.get("status"), valid_t_rel=rp.get("valid_t_rel"), note=rp.get("note"))
                                for rp in reps},
                seconds=time.time() - t_start, seconds_per_window=(time.time() - t_start) / max(len(feats), 1),
                note=("development reference; " + ("TRAIN scan of the family: consistency, not validation" if not client
                                                   else "new scan: no adopted fit to compare with")))
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if not a.no_network_step:
        LV.step(f"networks.{cfg['family']}", "done", summary=(f"median fit deviation {summ['median_dev5_best']:.1f} (best "
                f"candidate), {summ['n_usable']} windows, {time.time() - t_start_:.0f} s" if summ.get("median_dev5_best") is not None
                else f"{summ['n_usable']} windows"))
    print(json.dumps({k: summ[k] for k in ("scan", "n_usable", "median_dev5_first", "median_dev5_best", "median_dev5_path",
                                           "median_dev5_refined", "median_dev5_sequence", "median_ratio_best_to_fit",
                                           "seconds")}))
    if not a.no_figures:
        figures(cfg, a, out, rows, lines, reps)


def figures(cfg, a, out, rows, lines, reps):
    env = dict(os.environ)
    env.update(REG.scan_env(cfg, a.scan))
    env.update(runner.site_env(cfg, a.data, a.shadow))
    spec = []
    pick = rows if len(rows) <= 12 else [rows[i] for i in np.linspace(0, len(rows) - 1, 12).round().astype(int)]
    for r in pick:
        if "best" not in r:
            continue
        tag = "seq" if "seq" in r else "refined" if "refined" in r else "best"
        spec.append(dict(scan=a.scan, t=r["t"], mu=str(out / "mu" / f"{a.scan}_{r['t']:g}_{tag}.npz"),
                         label="network " + dict(seq="+ whole-scan refinement", refined="+ refinement", best="best of K")[tag],
                         dev5=r[tag]["dev5"], half=None, onset=env.get("IAM_ONSET")))
    (out / "patterns_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    code = ("import sys, runpy; sys.path.insert(0, '.'); from iampred import adapter as A; A.register(sys.argv[1]); "
            "sys.argv = ['pattern_gallery.py'] + sys.argv[2:]; runpy.run_path('pattern_gallery.py', run_name='__main__')")
    subprocess.run([sys.executable, "-c", code, a.scan, str(out / "patterns_spec.json"), str(out / "patterns.png"),
                    "--title", f"{cfg['family']} {a.scan}: measured vs network-rendered"],
                   cwd=str(HERE.parent), env=env, check=False)
    evolution_figure(cfg, a.scan, out, lines, reps)


EVOLUTION_COLS = 5


def evolution_figure(cfg, scan, out, lines, reps):
    'OUT/evolution.png from the csv lines (dicts keyed by column; a missing value = a missing key): one panel per'
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    ok = [L for L in lines if "best_dev5" in L]
    if not ok:
        return
    t = np.array([L["t"] for L in ok])
    knee = [L for L in ok if L.get("obs_L_knee_nm")]
    frg = [L for L in ok if L.get("obs_d_fringe_nm") or L.get("obs_h_grain_nm")]
    n = len(reps) + 1 + (1 if knee else 0) + (1 if frg else 0)
    nc = min(EVOLUTION_COLS, n)
    nr = int(np.ceil(n / nc))
    fig, axg = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.1 * nr), squeeze=False)
    ax = list(axg.ravel()[:n])
    for extra in axg.ravel()[n:]:
        extra.set_visible(False)
    if knee:
        axk = ax[len(reps)]
        for st, mk, c_, lab in (("knee", "o", "#54278f", "knee inside the q range"),
                                ("at lower q limit", "^", "#9e9ac8", "at least (q range ends)"),
                                ("at upper q limit", "v", "#9e9ac8", "at most (q range ends)")):
            sel = [L for L in knee if L.get("obs_knee_status") == st]
            if sel:
                axk.plot([L["t"] for L in sel], [L["obs_L_knee_nm"] for L in sel], mk, color=c_, ms=3, label=lab)
        axk.set_xscale("log"); axk.set_yscale("log"); axk.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axk.set_xlabel("t (s)"); axk.legend(fontsize=6)
        axk.set_title("lateral scale in the counts, 2 pi / q_knee (nm)\n[direct observation, no model]", fontsize=8)
    if frg:
        axf = ax[len(reps) + (1 if knee else 0)]
        for st, mk, c_, lab in (("fringe", "o", "#b30000", "film thickness (q 0.06-0.20)"), ("at range limit", "x", "#fc8d59", "at the search limit")):
            sel = [L for L in frg if L.get("obs_fringe_status") == st and L.get("obs_d_fringe_nm")]
            if sel:
                axf.plot([L["t"] for L in sel], [L["obs_d_fringe_nm"] for L in sel], mk, color=c_, ms=3, label=lab)
        sel = [L for L in frg if L.get("obs_h_grain_status") == "fringe" and L.get("obs_h_grain_nm")]
        if sel:
            axf.plot([L["t"] for L in sel], [L["obs_h_grain_nm"] for L in sel], "s", color="#08519c", ms=2.5, mfc="none", label="grain height (q 0.35-1.0)")
        axf.plot(t, [L.get("h_nom", np.nan) for L in ok], "--", color="0.4", lw=1, label="nominal mass thickness")
        axf.set_xscale("log"); axf.set_yscale("log"); axf.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axf.set_xlabel("t (s)"); axf.legend(fontsize=6)
        axf.set_title("film thickness and grain height from the fringes (nm)\n[direct observation, no model]", fontsize=8)
    for k, rp in enumerate(reps):
        c = col(rp)
        y = np.array([L[c] for L in ok]); lo = np.array([L[f"{c}_net_p16"] for L in ok]); hi = np.array([L[f"{c}_net_p84"] for L in ok])
        ax[k].fill_between(t, lo, hi, color="#9ecae1", alpha=0.6, lw=0, label="network head 16-84 %")
        ax[k].plot(t, y, "o", color="#08519c", ms=3, label="best candidate (rendered)")
        if any(f"{c}_path" in L for L in ok):
            ax[k].plot(t, [L.get(f"{c}_path", np.nan) for L in ok], "-", color="#3182bd", lw=1.2, label="time path")
        if any(f"{c}_refined" in L for L in ok):
            ax[k].plot(t, [L.get(f"{c}_refined", np.nan) for L in ok], "s", color="#e6550d", ms=3, label="refined")
        if any(f"{c}_seq" in L for L in ok):
            ax[k].plot(t, [L.get(f"{c}_seq", np.nan) for L in ok], "-D", color="#31a354", ms=3, lw=1.2, label="whole-scan refinement")
        if any(f"{c}_fit" in L for L in ok):
            ax[k].plot(t, [L.get(f"{c}_fit", np.nan) for L in ok], "x", color="k", ms=5, label="adopted fit")
        vr = rp.get("valid_t_rel")
        ax[k].set_xscale("log"); ax[k].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[k].set_xlabel("t (s)")
        ax[k].set_title(f"{rp.get('label', c)}\n[{rp.get('status', '')}{f', {vr[0]:g}-{vr[1]:g} s after onset' if vr else ''}]",
                        fontsize=8)
        if rp.get("unit", "").startswith("ln"):
            ax[k].set_yscale("log")
    ax[-1].plot(t, [L["best_dev5"] for L in ok], "o", color="#08519c", ms=3, label="best candidate")
    if any("path_dev5" in L for L in ok):
        ax[-1].plot(t, [L.get("path_dev5", np.nan) for L in ok], "-", color="#3182bd", lw=1.2, label="time path")
    if any("refined_dev5" in L for L in ok):
        ax[-1].plot(t, [L.get("refined_dev5", np.nan) for L in ok], "s", color="#e6550d", ms=3, label="refined")
    if any("seq_dev5" in L for L in ok):
        ax[-1].plot(t, [L.get("seq_dev5", np.nan) for L in ok], "-D", color="#31a354", ms=3, lw=1.2, label="whole-scan refinement")
        bad = [L for L in ok if L.get("seq_vs_refined") == "worse"]
        if bad:
            ax[-1].plot([L["t"] for L in bad], [L["seq_dev5"] for L in bad], "o", mfc="none", mec="#d62728", ms=8, mew=1.2,
                        label="whole scan clearly worse than the window alone")
    if any("fit_dev5" in L for L in ok):
        ax[-1].plot(t, [L.get("fit_dev5", np.nan) for L in ok], "x", color="k", ms=5, label="adopted fit")
    ax[-1].set_xscale("log"); ax[-1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[-1].set_xlabel("t (s)"); ax[-1].set_title("dev5 per cell (5 % floor)", fontsize=8)
    ax[-1].legend(fontsize=6)
    ax[0].legend(fontsize=6)
    fig.suptitle(f"{cfg['family']} {scan} (development reference)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "evolution.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
