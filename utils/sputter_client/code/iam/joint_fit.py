'Scan-level joint fit (IAM step 2b).'

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("IAM_SMALL", "1")
os.environ.setdefault("IAM_ETAMAX", "0.98")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iam_paths as IAMP

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares

WINDOWS = {"ref16": [3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 110, 140],
           "ref23": [3, 5, 7, 10, 15, 20, 30, 50, 80, 120, 200, 290]}
SIG_SHARED = os.environ.get("IAM_JOINT_SIG", "local") == "shared"
SCALE2_OPT = os.environ.get("IAM_SCALE2", "0") == "1"
SCALE_OPT = os.environ.get("IAM_SCALE", "0") == "1" or SCALE2_OPT
CB = os.environ.get("IAM_CB", "0") == "1" or SCALE_OPT
BASE = os.environ.get("IAM_BASE", "0") == "1" or CB
FIXED = {v.split("=")[0].strip() for v in filter(None, os.environ.get("IAM_FIX", "").split(","))}
STARTS = os.environ.get("IAM_JOINT_STARTS", "fitsLv_d,fitsLq_d" if SCALE2_OPT else "fitsLq_d,fitsLc_d" if SCALE_OPT else
                        "fitsLc_d,fitsLn4_d" if CB else "fitsLn4_d,fitsLn3_d" if BASE else "fits4s_d,fits4b_d").split(",")
SIG = ("sig", 0.0, float(os.environ.get("IAM_JOINT_SIGMAX", "0.1")), 0.02)
SHARED = [("lnkap", np.log(0.5), np.log(2.0), 0.0), ("dt", -1.5, 1.5, 0.0)] + ([SIG] if SIG_SHARED else [])


KAPPA_FIX = float(os.environ["IAM_JOINT_KAPPA"]) if os.environ.get("IAM_JOINT_KAPPA") else None
if KAPPA_FIX is not None:
    SHARED[0] = ("lnkap", np.log(KAPPA_FIX) - 1e-4, np.log(KAPPA_FIX) + 1e-4, float(np.log(KAPPA_FIX)))

LC = os.environ.get("IAM_LC", "0") == "1"
if LC and "phiS" not in FIXED:
    SHARED = SHARED + [("phiS", 0.0, 1.0, float(os.environ.get("IAM_LC_PHI0", "0.5")))]
SHARED_NAMES = [q[0] for q in SHARED]
OUT = IAMP.WORK / os.environ.get("IAM_JOINT_OUT", "joint")
SCALE = dict(cov=0.02, lnh=0.05, needle=0.1, base_m=0.1, base_t=0.3, cb=0.02, nul=0.05, lnD=0.05, lnR=0.10)


PROCESS = os.environ.get("IAM_JOINT_PROCESS", "0") == "1"
PROC_TR = float(os.environ.get("IAM_JOINT_PROC_TR", "10"))


BASEVOID = os.environ.get("IAM_BASEVOID", "0") == "1"
BURIED = os.environ.get("IAM_BURIED", "0") == "1"
if BURIED:
    SCALE["bur_m"] = 0.1
LC_PEN = LC and "lnLc" not in FIXED
if LC_PEN:
    SCALE["lnLc"] = 0.05
PEN_NAMES = ((["base_mass", "base_thickness", "needle"] + (["base_coverage"] if CB else []) + (["nul"] if SCALE2_OPT else [])
              + (["buried_mass"] if BURIED else []) + (["lnD", "lnR_late", "coverage_open"] if PROCESS else []) + (["lnLc"] if LC_PEN else []))
             if BASE else ["coverage", "ln_mean_thickness", "needle"])

_W = {}


def local_par():
    import iam_fit4 as F4
    assert F4.SMALL, "joint_fit needs IAM_SMALL=1"
    drop = ("lnK", "sig", "phiS") if LC else ("lnK", "sig")
    return [q for q in F4.PAR if q[0] not in drop and q[0] not in FIXED] + ([] if SIG_SHARED else [SIG])


def _init(scan, times):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import iam as M
    import iam_fit4 as F4
    s = M.Scan(scan)
    fits = [F4.Fit(s, float(t)) for t in times]
    _W.update(scan=s, fits=fits, meta=dict(t_dep=float(s.t_dep), rate_h=float(s.rate_h),
                                          t_mean=[float(f.w["t_mean"]) for f in fits],
                                          n=[int(f.ok.sum()) for f in fits], pixmask=s.pixmask,
                                          t_sput=(float(s.cfg["t_sput"]) if "t_sput" in s.cfg else None)))
    if s.shadow is not None:
        _W["meta"]["shadow_sha256"] = s.shadow.sha256
    del s.res
    gc.collect()


def _meta(_):
    return _W["meta"]


def h_of(meta, k, lnkap, dt):
    dep = max(meta["t_mean"][k] - meta["t_dep"] - dt, 0.05)
    if meta.get("t_sput") is not None:
        dep = min(dep, meta["t_sput"])
    return float(np.exp(lnkap) * meta["rate_h"] * dep)


def _resid(task):
    'task = (window k, local params (14), shared (lnkap, dt, sig), tag) -> (tag, residual).'
    k, xl, xs, tag = task
    f = _W["fits"][k]
    f.w["h_m"] = h_of(_W["meta"], k, xs[0], xs[1])
    names = [q[0] for q in local_par()]
    p = dict(zip(names, xl))
    if SIG_SHARED:
        p["sig"] = xs[2]
    if "phiS" in SHARED_NAMES:
        p["phiS"] = xs[SHARED_NAMES.index("phiS")]
    x = np.array([p[q[0]] for q in f.free])
    return tag, f.resid(x)


def _final(task):
    k, xl, xs = task
    import iam as M
    f = _W["fits"][k]
    f.w["h_m"] = h_of(_W["meta"], k, xs[0], xs[1])
    names = [q[0] for q in local_par()]
    p = dict(zip(names, xl))
    if SIG_SHARED:
        p["sig"] = xs[2]
    if "phiS" in SHARED_NAMES:
        p["phiS"] = xs[SHARED_NAMES.index("phiS")]
    x = np.array([p[q[0]] for q in f.free])
    (mu, sub, metal), m, pp = f.model(x)
    d = M.dev5(f.w, mu, f.s.V_pre)
    c_tot = min(m.coverage + (m.n_s * float(np.sum(m.sp.wj * np.pi * m.sp.Rj ** 2)) if m.fs > 0 else 0.0), 1.0)
    desc = dict(c_total=c_tot, mean_thickness=f.w["h_m"] / max(c_tot, 1e-3), t=float(f.w["t"]), h_m=f.w["h_m"], R=m.R, s=m.s, eta=m.eta, fcap=m.fcap, T=m.T, D=float(m.D),
                omega=m.omega, coverage=float(m.coverage), fs=float(m.fs), Rs=float(m.sp.R), D_s=float(m.D_s),
                fL=m.fL, RL=float(m.L.R), g=pp["g"], nu=pp["nu"], sig=pp["sig"],
                dev5_per_cell=float(d.sum() / f.ok.sum()), n_cells=int(f.ok.sum()))
    desc.update(M.shadow_desc(f.w, mu, f.ok, f.mdl))
    if "lnkD" in pp:
        desc.update(kD=float(np.exp(pp["lnkD"])), D_sq=float(m.D * np.exp(pp["lnkD"])))
    if BASE:
        desc.update(fb=float(m.fb), rb=float(m.rb), t_b=float(m.t_b), base_mass=float(m.fb * m.h_tot), bg=float(pp["bg"]),
                    cb=float(getattr(m, "cb", 1.0)))
        if "nut" in pp:
            desc["nut"] = float(pp["nut"])
        if "nul" in pp:
            desc["nul"] = float(pp["nul"])
    if LC:
        desc.update(Lc=float(np.exp(pp["lnLc"])), q_c=float(2 * np.pi / np.exp(pp["lnLc"])), phiS=float(pp["phiS"]))
    if "sL" in pp:
        desc.update(sL=float(pp["sL"]), HL=float(np.max(m.L.Hj)) if m.fL > 0 else 0.0)
    if "sigL" in pp:
        desc.update(sigL=float(pp["sigL"]))
    if BASEVOID or BURIED:
        import underlayer as U
        desc.update(U.describe(f.mdl, m))
        if BURIED:
            desc.update(h_surf=float(m.h_tot), fbur=float(pp["fbur"]))
    return k, desc, mu.astype(np.float32)


def grains_T(xl, h_m):
    'Main-structure height T, radius R, total projected coverage of the top structure, base thickness, base mass,'
    import iam4 as M4
    p = dict(zip([q[0] for q in local_par()], xl))
    h_bur = float(p["fbur"]) * h_m if BURIED else 0.0
    h_s = h_m * (1.0 - p["fbur"]) if BURIED else h_m
    kw = dict(RL=np.exp(p["lnRL"]), aL=np.exp(p["lnaL"]), fL=np.exp(p["lnfL"]), Rs=np.exp(p["lnRs"]),
              as_=np.exp(p["lnas"]), cs=p["cs"], oms=p["oms"])
    if CB:
        m = M4.GrainsC(np.exp(p["lnR"]), p["s"], p["eta"], p["fcap"], p["om"], h_s, fb=p["fb"], rb=p["rb"], cb=p["cb"], **kw)
    elif BASE:
        m = M4.GrainsB(np.exp(p["lnR"]), p["s"], p["eta"], p["fcap"], p["om"], h_s, fb=p["fb"], rb=p["rb"], **kw)
    else:
        m = M4.Grains2(np.exp(p["lnR"]), p["s"], p["eta"], p["fcap"], p["om"], h_s, **kw)
    c = m.coverage + (m.n_s * float(np.sum(m.sp.wj * np.pi * m.sp.Rj ** 2)) if m.fs > 0 else 0.0)
    return (m.T, m.R, min(c, 1.0), float(getattr(m, "t_b", 0.0)), float(getattr(m, "fb", 0.0)) * h_s,
            float(getattr(m, "cb", 1.0)), float(m.D), h_bur)


def sigl_start(sig):
    'sigL start of a window whose start record has none: its sig (one smear), or min(sig, IAM_SIGL_START).'
    v = os.environ.get("IAM_SIGL_START", "").strip()
    return float(min(sig, float(v))) if v else float(sig)


def start_params(scan, t):
    'Local start vector from the per-window fits (see module doc) and its source.'
    lp = local_par()
    names = [q[0] for q in lp]
    defaults = {q[0]: q[3] for q in lp}
    if LC and "lnLc" in names:
        import iam_fit4 as F4
        defaults["lnLc"] = F4.lc_start(t)
    if "kv" in names:
        defaults["kv"] = 0.001
    lb = np.array([q[1] for q in lp]); ub = np.array([q[2] for q in lp])
    cands = []
    for br in STARTS:
        p = IAMP.WORK / br / f"{scan}_{t:g}.json"
        if p.exists():
            rr = json.loads(p.read_text(encoding="utf-8"))
            if rr["T"] / (2 * rr["R"]) <= 3:
                cands.append((rr["dev5_per_cell"], br, rr["params"]))
    if cands:
        _, src, r = min(cands, key=lambda c: c[0])
        v = dict(defaults); v.update({k: r[k] for k in names if k in r})
        if "sigL" in names and "sigL" not in r:
            v["sigL"] = sigl_start(r.get("sig", v.get("sig", 0.02)))
        if src == "fits4b_d":
            v["cs"] = 0.02
    else:
        q = IAMP.WORK / "fits2K_d" / f"{scan}_{t:g}.json"
        if not q.exists():
            return None, "none (previous window's start)"
        r = json.loads(q.read_text(encoding="utf-8"))
        v = dict(defaults)
        v.update(lnR=np.log(r["R"]), s=r["s"], eta=r["coverage"], fcap=0.999, om=r["omega"], g=r["g"], nu=r["nu"],
                 lnRL=np.log(r["RL"]), lnaL=np.log(max(r["HL"] / r["RL"], 0.051)), lnfL=np.log(max(r["fL"], 1.1e-4)),
                 cs=0.02, **({"fb": 0.05} if BASE else {}))
        src = "fits2K_d (islands -> fcap 1)"
    x = np.array([v[k] for k in names], float)
    return np.clip(x, lb + 1e-6, ub - 1e-6), src


class Joint:
    def __init__(self, scan, workers, lam, tmin=0.0, windows=None):
        self.scan, self.times = scan, (list(windows) if windows else [t for t in WINDOWS[scan] if t >= tmin])
        self.lp = local_par()
        self.nl, self.nw, self.ns = len(self.lp), len(self.times), len(SHARED)
        self.pool = Pool(workers, initializer=_init, initargs=(scan, self.times))
        self.meta = self.pool.map(_meta, range(workers))[0]
        self.lam = lam
        self.n = self.meta["n"]
        self.off = np.r_[0, np.cumsum(self.n)]
        self.cache_x, self.cache_r = None, None
        self.nev = 0


    def split(self, x):
        return x[: self.ns], x[self.ns:].reshape(self.nw, self.nl)

    def bounds(self):
        lb = np.r_[[q[1] for q in SHARED], np.tile([q[1] for q in self.lp], self.nw)]
        ub = np.r_[[q[2] for q in SHARED], np.tile([q[2] for q in self.lp], self.nw)]
        return lb, ub


    def penalty(self, x):
        if self.lam <= 0:
            return np.zeros(0)
        xs, XL = self.split(x)
        hm = np.array([h_of(self.meta, k, xs[0], xs[1]) for k in range(self.nw)])
        TRC = np.array([grains_T(XL[k], hm[k]) for k in range(self.nw)])
        T, R, c, tb, mb, cb, Dm, hb = TRC.T
        r_n = np.maximum(T / (2 * R) - 3, 0) / SCALE["needle"]
        proc = []
        if PROCESS:
            tt = np.array(self.times, float)
            late = (tt[:-1] >= PROC_TR)
            proc = [np.maximum(np.log(Dm[:-1]) - np.log(Dm[1:]), 0) / SCALE["lnD"],
                    np.where(late, np.maximum(np.log(R[:-1]) - np.log(R[1:]), 0), 0) / SCALE["lnR"],
                    np.where(c[:-1] < 0.95, np.maximum(c[:-1] - c[1:], 0), 0) / SCALE["cov"]]
        if BASE:
            r_m = np.maximum(mb[:-1] - mb[1:], 0) / SCALE["base_m"]
            r_t = np.maximum(tb[:-1] - tb[1:], 0) / SCALE["base_t"]
            if CB:
                ce = cb * np.minimum(1.0, mb / 0.3)
                extra = []
                if SCALE2_OPT:
                    nl = XL[:, [q[0] for q in self.lp].index("nul")]
                    extra = [np.maximum(nl[:-1] - nl[1:], 0) / SCALE["nul"]]
                return np.sqrt(self.lam) * np.r_[(r_m, r_t, r_n, np.maximum(ce[:-1] - ce[1:], 0) / SCALE["cb"], *extra, *proc)]
            bur = [np.maximum(hb[:-1] - hb[1:], 0) / SCALE["bur_m"]] if BURIED else []
            lc = []
            if LC_PEN:
                lL = XL[:, [q[0] for q in self.lp].index("lnLc")]
                lc = [np.maximum(lL[:-1] - lL[1:], 0) / SCALE["lnLc"]]
            return np.sqrt(self.lam) * np.r_[(r_m, r_t, r_n, *bur, *proc, *lc)]
        lh = np.log(hm / np.maximum(c, 1e-3))
        r_c = np.maximum(c[:-1] - c[1:], 0) / SCALE["cov"]
        r_h = np.maximum(lh[:-1] - lh[1:], 0) / SCALE["lnh"]
        return np.sqrt(self.lam) * np.r_[r_c, r_h, r_n]


    def window_resid(self, x):
        xs, XL = self.split(x)
        tasks = [(k, XL[k], xs, k) for k in range(self.nw)]
        out = dict(self.pool.map(_resid, tasks))
        return [out[k] for k in range(self.nw)]

    def fun(self, x):
        self.nev += 1
        rw = self.window_resid(x)
        self.cache_x, self.cache_r = x.copy(), rw
        r = np.concatenate(rw + [self.penalty(x)])
        self.log(x, rw)
        return r

    def jac(self, x):
        if self.cache_x is None or not np.array_equal(x, self.cache_x):
            self.cache_x, self.cache_r = x.copy(), self.window_resid(x)
        r0 = self.cache_r
        lb, ub = self.bounds()
        xs, XL = self.split(x)
        h = 1e-3 * np.maximum(1.0, np.abs(x))
        h = np.where(x + h > ub, -h, h)
        tasks = []
        for j in range(self.ns):
            xs2 = xs.copy(); xs2[j] += h[j]
            tasks += [(k, XL[k], xs2, ("s", j, k)) for k in range(self.nw)]
        for k in range(self.nw):
            for j in range(self.nl):
                xl2 = XL[k].copy(); xl2[j] += h[self.ns + k * self.nl + j]
                tasks.append((k, xl2, xs, ("l", j, k)))
        rows, cols, vals = [], [], []
        for tag, r in self.pool.imap_unordered(_resid, tasks, chunksize=2):
            kind, j, k = tag
            c = j if kind == "s" else self.ns + k * self.nl + j
            d = (r - r0[k]) / h[c]
            rows.append(np.arange(self.off[k], self.off[k + 1], dtype=np.int32)); cols.append(np.full(len(d), c, np.int32))
            vals.append(d.astype(np.float32))
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.off[-1], len(x)))
        if self.lam > 0:
            p0 = self.penalty(x)
            Jp = np.zeros((len(p0), len(x)))
            for c in range(len(x)):
                x2 = x.copy(); x2[c] += h[c]
                Jp[:, c] = (self.penalty(x2) - p0) / h[c]
            J = sp.vstack([J, sp.csr_matrix(Jp)]).tocsr()
        return J

    def log(self, x, rw):
        d = [float(np.sum(r ** 2)) / n for r, n in zip(rw, self.n)]
        pen = float(np.sum(self.penalty(x) ** 2))
        xs, _ = self.split(x)
        print(f"  nev {self.nev:3d}  sum dev5/cell {sum(d):8.2f}  penalty {pen:9.1f}  kappa {np.exp(xs[0]):.3f} "
              f"dt {xs[1]:+.3f}" + (f" sig {xs[2]:.3f}" if SIG_SHARED else "")
              + (f" phiS {xs[SHARED_NAMES.index('phiS')]:.3f}" if "phiS" in SHARED_NAMES else "") + f"  [{' '.join(f'{v:.1f}' for v in d)}]", flush=True)

    def solve(self, x0, nfev):
        lb, ub = self.bounds()
        r = least_squares(self.fun, np.clip(x0, lb + 1e-9, ub - 1e-9), jac=self.jac, bounds=(lb, ub), method="trf",
                          tr_solver="lsmr", x_scale="jac", max_nfev=nfev, ftol=1e-6, xtol=1e-8)
        return r

    def finalize(self, x):
        xs, XL = self.split(x)
        res = self.pool.map(_final, [(k, XL[k], xs) for k in range(self.nw)])
        res.sort(key=lambda v: v[0])
        return [d for _, d, _ in res], [mu for _, _, mu in res]


def own_sig(scan, t):
    for br in ("fits4s_d", "fits4b_d", "fits2K_d") + tuple(STARTS):
        p = IAMP.WORK / br / f"{scan}_{t:g}.json"
        if p.exists():
            return float(json.loads(p.read_text(encoding="utf-8"))["params"].get("sig", 0.02))
    return 0.02


def independent(scan, t):
    'dev5 of the per-window fits for comparison: grains+small (fits4s_d) and the morphology-path choice.'
    out = {}
    for br in ("fits4s_d", "fits4b_d", "fits2K_d"):
        p = IAMP.WORK / br / f"{scan}_{t:g}.json"
        if p.exists():
            out[br] = json.loads(p.read_text(encoding="utf-8"))["dev5_per_cell"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lam", type=float, nargs="+", default=[1e2, 1e3, 1e4])
    ap.add_argument("--nfev-a", type=int, default=25)
    ap.add_argument("--nfev-b", type=int, default=20)
    ap.add_argument("--resume", default=None, help="continue from this stored stage of WORK/joint/<scan>.json")
    ap.add_argument("--warm", default=None, help="<file>:<stage> in WORK/joint: start parameters (local and kappa, dt)")
    ap.add_argument("--tmin", type=float, default=0.0, help="drop windows before this time (s)")
    ap.add_argument("--windows", type=float, nargs="+", default=None, help="window centres (s) instead of WINDOWS[scan]")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    J = Joint(a.scan, a.workers, 0.0, a.tmin, a.windows)
    starts = []
    for t in J.times:
        xl, src = start_params(a.scan, t)
        starts.append((starts[-1][0] if xl is None else xl, src))
    if SIG_SHARED:
        x0 = np.r_[[q[3] for q in SHARED[:2]], float(np.median([own_sig(a.scan, t) for t in J.times])),
                   [q[3] for q in SHARED[3:]], np.concatenate([s for s, _ in starts])]
    else:
        x0 = np.r_[[q[3] for q in SHARED], np.concatenate([s for s, _ in starts])]
    warm_note = None
    if a.warm:
        wf, wst = a.warm.split(":")
        old = json.loads((OUT / wf).read_text(encoding="utf-8"))
        ox = np.array(old["stages"][wst]["x"]); ons, onl = len(old["shared_names"]), len(old["local_names"])
        oxs, oxl = ox[:ons], ox[ons:].reshape(len(old["windows"]), onl)
        x0[:2] = oxs[:2]
        if KAPPA_FIX is not None:
            x0[0] = np.log(KAPPA_FIX)
        if LC:
            for j, nm in enumerate(SHARED_NAMES[2:], 2):
                if nm in old["shared_names"]:
                    x0[j] = oxs[old["shared_names"].index(nm)]
        XL = x0[J.ns:].reshape(J.nw, J.nl)
        for k, t in enumerate(J.times):
            if t not in old["windows"]:
                continue
            ko = old["windows"].index(t)
            for j, nm in enumerate(n for n, *_ in J.lp):
                if nm in old["local_names"]:
                    XL[k, j] = oxl[ko, old["local_names"].index(nm)]
            lnm = [n for n, *_ in J.lp]
            if "sigL" in lnm and "sigL" not in old["local_names"] and "sig" in lnm:
                XL[k, lnm.index("sigL")] = sigl_start(XL[k, lnm.index("sig")])
        warm_note = a.warm
    print(f"{a.scan}: {J.nw} windows, {len(x0)} parameters, cells {J.off[-1]}; starts:",
          {t: src for t, (_, src) in zip(J.times, starts)}, flush=True)
    rec = dict(scan=a.scan, family=("scale-dependent optics with a non-decreasing large-scale weight nul (iam4.GrainsC / Model4f)" if SCALE2_OPT else
                               "scale-dependent optics, base cover cb (iam4.GrainsC / Model4e)" if SCALE_OPT else
                                    "percolating base (cover cb) + grains + small (iam4.GrainsC / Model4d)" if CB else
                                    "dense base + grains + small (iam4.GrainsB / Model4c)" if BASE else
                                    "grains+small (iam4.Model4b)"), fixed=sorted(FIXED), start_branches=STARTS, windows=J.times, shared_names=[q[0] for q in SHARED],
               local_names=[q[0] for q in J.lp], starts={str(t): src for t, (_, src) in zip(J.times, starts)},
               lam=a.lam, penalty_scales=SCALE, sig_start=("median of the start windows' own sig" if SIG_SHARED else "per window: own fit's sig"), quad=os.environ.get("IAM_QUAD", "dense"), stages={})
    if os.environ.get("IAM_LSOFT", "0") == "1":
        rec["family"] += " + soft-base large clusters (iam4.Model4.sL, IAM_LSOFT)"
    if os.environ.get("IAM_SIGL", "0") == "1":
        rec["family"] += " + own alpha_f smear of the large clusters (iam4.Model4.sigL, IAM_SIGL)"
    if LC:
        rec["family"] += " + lc_optics host-optics mix below q_c (iam4.Model4g, IAM_LC)"
        rec["lc_start"] = os.environ.get("IAM_LC_START")
    if BASEVOID or BURIED:
        rec["family"] += " + underlayer" + (" C2 voids in the base" if BASEVOID else "") + (" C3a clusters below it" if BURIED else "")
        if os.environ.get("IAM_BASEVOID_AMP", "0") == "1":
            rec["family"] += " (void amplitude kv, IAM_BASEVOID_AMP)"
    rec["warm_start"] = warm_note
    if KAPPA_FIX is not None:
        rec["kappa_fixed"] = KAPPA_FIX
    rec["pixmask"] = J.meta["pixmask"]
    if "shadow_sha256" in J.meta:
        import iam as M
        rec["instrument"] = dict(shadow_bg=M.SHADOW_BG, shadow_aB=M.SHADOW_AB, shadow_sha256=J.meta["shadow_sha256"])
    plan =[("A_shared", 0.0, a.nfev_a)] + [(f"B_lam{lam:g}", lam, a.nfev_b) for lam in a.lam]
    if a.resume:
        old = json.loads((OUT / f"{a.scan}.json").read_text(encoding="utf-8"))
        rec["start_dev5"], rec["stages"] = old.get("start_dev5"), old["stages"]
        rec["resumed_from"] = a.resume
        x0 = np.array(old["stages"][a.resume]["x"])
        names = [n for n, _, _ in plan]
        plan = plan[names.index(a.resume) + 1:] if a.resume in names else [st for st in plan if st[0] not in old["stages"]]
    else:
        J.fun(x0)
        rec["start_dev5"] = [float(np.sum(r ** 2)) / n for r, n in zip(J.cache_r, J.n)]
    for stage, lam, nfev in plan:
        print(f"stage {stage} (lam {lam:g}, max_nfev {nfev})", flush=True)
        J.lam = lam
        r = J.solve(x0, nfev)
        lb_, ub_ = J.bounds()
        dx_stage = float(np.max(np.abs(r.x - x0) / np.maximum(ub_ - lb_, 1e-12)))
        x0 = r.x
        last = stage == plan[-1][0]
        desc, mus = J.finalize(r.x)
        xs, _ = J.split(r.x)
        J.lam = 1.0
        viol = J.penalty(r.x)
        rec["stages"][stage] = dict(
            lam=lam, status=int(r.status), nfev=int(r.nfev), dx=dx_stage, cost=float(2 * r.cost), seconds=time.time() - t0,
            shared=dict(kappa=float(np.exp(xs[0])), dt=float(xs[1]), **({"sig": float(xs[2])} if SIG_SHARED else {}),
                        **({"phiS": float(xs[SHARED_NAMES.index("phiS")])} if "phiS" in SHARED_NAMES else {})),
            x=[float(v) for v in r.x], windows=desc,
            violations_in_scale_units=dict(zip(PEN_NAMES,
                                               [list(map(float, v)) for v in
                                                np.split(viol, np.cumsum([J.nw if nm == "needle" else J.nw - 1
                                                                          for nm in PEN_NAMES])[:-1])])),
            sum_dev5=float(sum(d["dev5_per_cell"] for d in desc)))
        if last:
            for t, mu in zip(J.times, mus):
                np.savez_compressed(OUT / f"{a.scan}_{t:g}_mu.npz", mu=mu)
        (OUT / f"{a.scan}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    rec["independent_dev5"] = {str(t): independent(a.scan, t) for t in J.times}
    (OUT / f"{a.scan}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    J.pool.close()
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
