'Component separation of a fitted in-situ window (universal sputter model, 2026-11-20).'

from __future__ import annotations

import math

import numpy as np

LABELS = dict(substrate="substrate seen through the metal", background="beam-path background",
              grains="grains / islands", large_clusters="large clusters / patches",
              film="continuous film: density fluctuation / network", voids="voids under the film",
              buried="buried metal clusters", domains="metal-rich domains in the polymer",
              base_roughness="rough top of the continuous layer", fluorescence="fluorescence",
              clusters_in_pores="metal clusters in the pores", clusters_on_cap="metal clusters on the cap layer",
              pore_network="metal network in the pores")
ORDER = ("grains", "large_clusters", "film", "voids", "buried", "domains", "base_roughness", "clusters_in_pores",
         "clusters_on_cap", "pore_network", "substrate", "fluorescence", "background")


def split(ctx, x):
    '(mu, {component: image (like mu)}, closure, (m, p, h)) of the solution x (pnames order) in the window of ctx'
    if ctx.cfg.get("fitter") == "iam_fit_t":
        mu, comps, mph = _split_tio2(ctx, np.asarray(x, float))
    else:
        mu, comps, mph = _split_grains(ctx, np.asarray(x, float))
    comps = {k: np.broadcast_to(v, mu.shape).astype(float) for k, v in comps.items()}
    ok = np.broadcast_to(ctx.f.ok, mu.shape)
    tot = sum(comps.values())
    closure = float(np.max(np.abs(tot - mu)[ok] / np.maximum(np.abs(mu[ok]), 1e-30))) if ok.any() else 0.0
    return mu, {k: comps[k] for k in ORDER if k in comps}, closure, mph


def _split_grains(ctx, x):
    (mu, pr), m, p, h = ctx.render(x, parts=True)
    mdl = ctx.f.mdl
    K = math.exp(p["lnK"])
    sub, isl, fl = pr["sub"], pr["isl"], pr["fl"]
    zero = np.zeros_like(isl)
    piece = lambda name: zero if getattr(mdl, name, None) is None else zero + getattr(mdl, name)
    accL, ff = piece("_accL"), piece("_ff")
    void, bur, dom = piece("_void_part"), piece("_bur_part"), piece("_dom")
    rough = zero
    if getattr(mdl, "rough", None) is not None:
        slab0, _ = m.slab(mdl.s.metal, f1_au=mdl.f1)
        r = mdl.base_rough(m, [(mdl.nu_isl * d, t) for d, t in slab0])
        rough = zero + r
    L = pr.get("isl_L")
    sm = lambda a: K * mdl.smear(a)[None]
    smL = (lambda a: K * mdl.smear_L(a)[None]) if L is not None else sm
    comps = dict(large_clusters=smL(accL), film=(smL(ff) if L is not None else sm(ff)), voids=sm(void), buried=sm(bur),
                 domains=sm(dom), base_roughness=sm(rough),
                 grains=sm(isl - accL - ff - void - bur - dom - rough),
                 fluorescence=sm(np.repeat(fl[:, None], isl.shape[1], 1)))
    shadow = ctx.M.shadow_term(ctx.w, mdl)
    ratio = np.asarray(pr["G"], float)
    rs = mdl.smear_ratio(ratio) if getattr(mdl, "Kaf", None) is not None else ratio
    bg = sub * (float(getattr(mdl, "bg", 0.0)) / np.maximum(rs, 1e-30))[None, :, None]
    comps["substrate"] = sub - bg
    comps["background"] = bg + (0.0 if shadow is None else shadow)
    return mu, {k: v for k, v in comps.items() if k in ("grains", "substrate", "background") or np.any(v != 0)}, (m, p, h)


def _split_tio2(ctx, x):
    (mu, pr), m, p, h = ctx.render(x, parts=True)
    mdl = ctx.f.mdl
    K = math.exp(p["lnK"])
    sm = lambda a: K * mdl.smear(a)[None]
    comps = dict(substrate=pr["sub"], fluorescence=sm(np.repeat(pr["fl"][:, None], pr["isl"].shape[1], 1)))
    if "lnAoz" in ctx.pnames:
        xn = x.copy()
        xn[ctx.pnames.index("lnAoz")] = -200.0
        (_, pr0), _, _, _ = ctx.render(xn, parts=True)
        net, isl0 = pr["isl"] - pr0["isl"], pr0["isl"]
        top0 = pr0.get("isl_top", np.zeros_like(isl0))
    else:
        net, isl0 = np.zeros_like(pr["isl"]), pr["isl"]
        top0 = pr.get("isl_top", np.zeros_like(isl0))
    comps.update(clusters_in_pores=sm(isl0 - top0), clusters_on_cap=sm(top0), pore_network=sm(net))
    comps["background"] = mu - sum(comps.values())
    return mu, {k: v for k, v in comps.items() if k in ("substrate", "clusters_in_pores") or np.any(v != 0)}, (m, p, h)


def shares(ctx, mu, comps) -> dict:
    "Each component's share of the model counts on the fitted cells (sum E c / sum E mu)."
    E = np.where(ctx.f.ok, ctx.w["E"], 0.0)
    tot = float(np.sum(E * mu))
    return {k: round(float(np.sum(E * v)) / max(tot, 1e-300), 5) for k, v in comps.items()}


def images(ctx, comps) -> np.ndarray:
    "(component, af, q) count-rate images on the window's cells, the two detector halves merged like the model"
    ok = ctx.f.ok
    E = np.where(ok, ctx.w["E"], 0.0)
    Es = E.sum(0)
    good = Es > 0
    return np.array([np.where(good, (np.where(ok, v, 0.0) * E).sum(0) / np.maximum(Es, 1e-30), np.nan) for v in comps.values()])


def check(family: str, scan: str, n: int = 2) -> int:
    "Closure check on the family's reference solutions of one scan (first and last of n windows; run in a process"
    from . import adapter as A
    from . import ref_override as RF
    from . import registry as REG
    from . import runner
    cfg = REG.load(family)
    env = REG.scan_env(cfg, scan)
    env.update(runner.site_env(cfg, None, None))
    A.activate(env)
    refs, _ = RF.load_scan(cfg, scan)
    refs = [refs[0], refs[-1]] if n >= 2 and len(refs) > 1 else refs[:1]
    s = A.get_scan(scan)
    bad = 0
    for r in refs:
        ctx = A.Context(cfg, s, r["t"])
        x = A.x_from_reference(ctx, r["params"], float(np.log(r["h_m"] / max(ctx.h_nom_ctx, 0.002))))
        mu, comps, closure, _ = split(ctx, x)
        (mu0, _, _), _, _, _ = ctx.render(x)
        same = float(np.max(np.abs(mu - mu0)[ctx.f.ok] / np.maximum(np.abs(mu0[ctx.f.ok]), 1e-30)))
        sh = shares(ctx, mu, comps)
        ok = closure < 1e-9 and same < 1e-12 and abs(sum(sh.values()) - 1) < 1e-3
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {family} {scan} t={r['t']:g}: closure {closure:.1e}, render {same:.1e}, "
              f"shares {', '.join(f'{k} {v:.3f}' for k, v in sh.items())}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        raise SystemExit(check(sys.argv[2], sys.argv[3]))
    print("python -m iampred.components --check FAMILY SCAN")
    raise SystemExit(2)
