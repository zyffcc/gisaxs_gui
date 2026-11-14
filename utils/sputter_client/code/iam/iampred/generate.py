'Simulated training windows of one family: real-window contexts x sampled forward-model parameters x measured noise.'

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import registry as REG
from . import references as RF
from . import runner

KINDS = ["near", "interp", "cross", "uniform"]
_G = {}


def _bank(path, cfg):
    'Residual bank of the family (check.py): refused when it belongs to another family version; reference windows'
    if path not in _G:
        z = np.load(path, allow_pickle=False)
        if str(z["family_hash"]) != cfg["family_hash"]:
            raise SystemExit(f"{path}: residual bank of another family version (run iampred.check again)")
        if "forward_hash" in z.files and str(z["forward_hash"]) != cfg["forward_hash"]:
            raise SystemExit(f"{path}: residual bank made with other forward code (run iampred.check again)")
        ok = z["ok"].astype(bool) if "ok" in z.files else np.ones(len(z["scan"]), bool)
        _G[path] = dict(R=z["R"][ok], scan=z["scan"].astype(str)[ok], t_rel=z["t_rel"][ok])
    return _G[path]


def _scan_ctx(cfg, scan, times):
    from . import adapter as A
    key = (cfg["family"], scan)
    if _G.get("scan_key") != key:
        _G.pop("scan_obj", None)
        _G["scan_obj"] = A.ScanContexts(cfg, scan, times)
        _G["scan_key"] = key
    return _G["scan_obj"]


def _u_of(ctx, x):
    return (np.asarray(x, float) - ctx.lb) / (ctx.ub - ctx.lb)


def _task(args):
    cfg, scan, refs_all, n, seed, bank_path, hold = args
    from . import adapter as A
    from . import features as FE
    from scipy.ndimage import gaussian_filter, uniform_filter
    rng = np.random.default_rng(seed)
    hold = set(hold)
    refs = refs_all[scan]
    times = [r["t"] for r in refs]
    sc = _scan_ctx(cfg, scan, times)
    pr, nz = cfg["prior"], cfg["noise"]
    probs = np.array([pr["near"], pr["interp"], pr["cross"], pr["uniform"]], float)
    probs /= probs.sum()
    bank = _bank(bank_path, cfg) if (bank_path and nz.get("bank", True)) else None

    if bank is not None:
        other = np.where((bank["scan"] != scan) & ~np.isin(bank["scan"], list(hold)))[0]
        if len(other) == 0:
            other = np.where(~np.isin(bank["scan"], list(hold)))[0]
    else:
        other = np.array([], int)

    U_same = []
    for ctx, r in zip(sc.ctx, refs):
        lnmf = float(np.log(r["h_m"] / max(ctx.h_nom_ctx, 0.002)))
        U_same.append(_u_of(ctx, A.x_from_reference(ctx, r["params"], lnmf)))
    cross_pool = [(s2, r2) for s2, lst in refs_all.items() if s2 != scan and s2 not in hold for r2 in lst]

    I_pre, V_pre = np.asarray(sc.scan.I_pre, float), np.asarray(sc.scan.V_pre, float)
    rel_pre = np.where(I_pre > 0, np.sqrt(np.maximum(V_pre, 0.0)) / np.maximum(I_pre, 1e-300), 0.0)
    X, S, P, R, C, TM, K = [], [], [], [], [], [], []
    nfail, tries, t0, fails = 0, 0, time.time(), {}
    sig_u = float(pr["sigma_u"]); sig_m = float(pr.get("sigma_mass", 0.15))
    while len(X) < n and tries < 5 * n + 50:
        tries += 1
        c = int(rng.integers(len(sc.ctx)))
        ctx = sc.ctx[c]
        lo = (times[c] - times[c - 1]) / 2 if c > 0 else min(0.5, times[c] - float(sc.scan.t_dep))
        hi = (times[c + 1] - times[c]) / 2 if c + 1 < len(times) else (times[c] - times[c - 1]) / 2 if c > 0 else 0.5
        j = float(pr.get("t_jitter", 1.0))
        tm = float(times[c] + rng.uniform(-j * max(lo, 0.0), j * max(hi, 0.0)))
        tm = max(tm, float(sc.scan.t_dep) + 0.1)
        kind = int(rng.choice(4, p=probs))
        if kind == 2 and not cross_pool:
            kind = 0
        npar = len(ctx.pnames)
        sig_vec = np.full(npar, sig_u)
        if ctx.pnames[-1] == A.MASS:
            sig_vec[-1] = sig_m / (ctx.ub[-1] - ctx.lb[-1])
        if kind == 0:
            k = int(np.argmin([abs(t - tm) for t in times]))
            u = U_same[k] + rng.normal(0, 1, npar) * sig_vec
        elif kind == 1:
            k = int(np.searchsorted(times, tm))
            if k <= 0 or k >= len(times):
                u = U_same[min(max(k, 0), len(times) - 1)] + rng.normal(0, 1, npar) * sig_vec
            else:
                lam = (tm - times[k - 1]) / (times[k] - times[k - 1])
                u = (1 - lam) * U_same[k - 1] + lam * U_same[k] + rng.normal(0, 1, npar) * sig_vec / 2
        elif kind == 2 and cross_pool:
            hn = max(ctx.h_nominal(tm), 0.002)
            cand = sorted(cross_pool, key=lambda v: abs(np.log(max(v[1]["h_m"], 0.002)) - np.log(hn)))[:4]
            s2, r2 = cand[int(rng.integers(len(cand)))]
            u = _u_of(ctx, A.x_from_reference(ctx, r2["params"], 0.0)) + rng.normal(0, 1, npar) * sig_vec
            if ctx.pnames[-1] == A.MASS:
                u[-1] = _u_of(ctx, np.r_[np.zeros(npar - 1), rng.normal(0, 2 * sig_m)])[-1]
        else:
            kind = 3
            u = rng.random(npar)
        u = np.clip(u, 0.0, 1.0)
        x = ctx.lb + u * (ctx.ub - ctx.lb)
        try:
            (mu, sub, _), m, p, h = ctx.render(x, tm)
            rep = ctx.reported(m, p, h)
        except (Exception, SystemExit) as e:
            nfail += 1
            why = f"{type(e).__name__}: {str(e)[:80]}"
            fails[why] = fails.get(why, 0) + 1
            continue
        ok = ctx.w["mask"]
        if nz.get("pre_noise", True):
            zf = uniform_filter(rng.normal(0, 1, mu.shape), size=(1, 3, 3), mode="nearest") * 3.0
            mu = mu + np.asarray(sub, float) * rel_pre * zf
        lam_ = np.where(ok, mu * ctx.w["E"], 0.0)
        if not np.all(np.isfinite(lam_)) or np.any(lam_ < 0):
            nfail += 1
            why = "non-finite model" if not np.all(np.isfinite(lam_)) else "negative model intensity"
            fails[why] = fails.get(why, 0) + 1
            continue
        field = np.zeros_like(lam_)
        if bank is not None and len(other):
            b = int(other[rng.integers(len(other))])
            field = field + rng.uniform(*nz["bank_amp"]) * bank["R"][b].astype(float)
        if nz.get("smooth", 0) > 0:
            r = gaussian_filter(rng.normal(0, 1, lam_.shape), (0, 3.0, 3.0))
            field = field + float(nz["smooth"]) * r / max(r.std(), 1e-9)
        lsj = rng.normal(0, float(nz["scale"])) if nz.get("scale", 0) > 0 else 0.0
        lsd = rng.normal(0, float(nz["drift"])) if nz.get("drift", 0) > 0 else 0.0
        Y = rng.poisson(np.maximum(lam_ * np.exp(field + lsj + lsd), 0.0)).astype(float)
        Xf = FE.image(ctx.s, ctx.w, Y, cfg["features"], pre_scale=float(np.exp(lsj)))
        X.append(Xf.astype(np.float16))
        S.append(FE.scalars(cfg, ctx.s, tm, ctx.h_nominal(tm)))
        P.append(x.astype(np.float32)); R.append(rep.astype(np.float32))
        C.append(f"{scan}_{times[c]:g}"); TM.append(tm); K.append(kind)
    meta = dict(scan=scan, pnames=sc.ctx[0].pnames, lb=sc.ctx[0].lb.tolist(), ub=sc.ctx[0].ub.tolist(), n_made=len(X),
                n_failed=nfail, fail_reasons=fails, seconds=time.time() - t0, metal=str(sc.scan.metal))
    return (np.array(X), np.array(S, np.float32), np.array(P, np.float32), np.array(R, np.float32), C, TM, K, meta)


def plan_counts(cfg, refs, n):
    "Samples per scan: proportional to its contexts; with prior.balance == 'metal' each metal gets an equal share."
    w = {sc: float(len(refs[sc])) for sc in cfg["scans"]}
    if cfg["prior"].get("balance") == "metal":
        by = {}
        for sc in cfg["scans"]:
            by.setdefault(cfg["scans"][sc].get("metal", "?"), []).append(sc)
        for mt, lst in by.items():
            tot = sum(w[s] for s in lst)
            for s in lst:
                w[s] = w[s] / tot / len(by)
    tot = sum(w.values())
    return {sc: int(round(n * w[sc] / tot)) for sc in cfg["scans"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--data")
    ap.add_argument("--shadow")
    ap.add_argument("--holdout", default="", help="scans kept out of the cross prior and the residual bank of the others")
    a = ap.parse_args()
    cfg = REG.load(a.family)
    hold = sorted(h.strip() for h in a.holdout.replace("+", ",").split(",") if h.strip())
    if set(hold) - set(cfg["scans"]):
        raise SystemExit(f"--holdout scans not in family: {sorted(set(hold) - set(cfg['scans']))}")
    refs, shas = RF.load(cfg)
    counts = plan_counts(cfg, refs, a.n)
    tasks = []
    for i, sc in enumerate(cfg["scans"]):
        left, j = counts[sc], 0
        while left > 0:
            k = min(a.chunk, left)
            tasks.append((cfg, sc, refs, k, (a.seed, i, j), a.bank, hold))
            left -= k; j += 1
    t0 = time.time()
    res_all = [r for r in runner.run(cfg, tasks, _task, a.workers, data=a.data, shadow=a.shadow) if r is not None]
    per_scan, fail_reasons = {}, {}
    for r in res_all:
        q = per_scan.setdefault(r[7]["scan"], dict(made=0, failed=0))
        q["made"] += r[7]["n_made"]; q["failed"] += r[7]["n_failed"]
        for k_, v_ in r[7]["fail_reasons"].items():
            fail_reasons[k_] = fail_reasons.get(k_, 0) + v_
    res = [r for r in res_all if len(r[0])]
    pn = {json.dumps([r[7]["pnames"], np.round(r[7]["lb"], 12).tolist(), np.round(r[7]["ub"], 12).tolist()]) for r in res}
    if len(pn) != 1:
        raise SystemExit(f"parameter lists or bounds differ between scans (per-scan env changed the fitter): {pn}")
    m0 = res[0][7]
    from . import features as FE
    meta = dict(family=cfg["family"], family_hash=cfg["family_hash"], forward_hash=cfg["forward_hash"],
                params=m0["pnames"], lb=m0["lb"], ub=m0["ub"], reported=[q["name"] for q in cfg["reported"]],
                reported_info=cfg["reported"], scalars=FE.scalar_names(cfg), features=cfg["features"],
                fitter=cfg["fitter"], env=cfg["env"], scans=list(cfg["scans"]), reference_sha256=shas,
                bank=a.bank, seed=a.seed, seed_scheme="SeedSequence(seed, scan index, chunk)", holdout=hold,
                pred_hash=REG.pred_hash(), n_requested=a.n, n_made=int(sum(q["made"] for q in per_scan.values())),
                n_failed=int(sum(q["failed"] for q in per_scan.values())), per_scan=per_scan,
                fail_reasons=fail_reasons, seconds=time.time() - t0, workers=a.workers,
                kinds=KINDS, cpu_seconds=float(sum(r[7]["seconds"] for r in res_all)))
    np.savez_compressed(a.out, X=np.concatenate([r[0] for r in res]), S=np.concatenate([r[1] for r in res]),
                        P=np.concatenate([r[2] for r in res]), R=np.concatenate([r[3] for r in res]),
                        ctx=np.array(sum((r[4] for r in res), [])), t_morph=np.array(sum((r[5] for r in res), []), float),
                        kind=np.array(sum((r[6] for r in res), []), np.int8), meta=json.dumps(meta, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in ("family", "n_made", "n_failed", "seconds", "cpu_seconds")}))


if __name__ == "__main__":
    main()
