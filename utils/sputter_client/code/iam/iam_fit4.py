'IAM v0.4 columnar-grain window fits on Si (iam4.Model4.predict_grains), K fixed by default to 1.1e5 (supported by'

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import iam_paths as IAMP

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iam as M
import iam4 as M4

OUT = IAMP.WORK / os.environ.get("IAM_OUT", "fits4")
PAR = [("lnR", np.log(0.5), np.log(float(os.environ.get("IAM_RMAX", "40"))), np.log(3.0)), ("s", 0.05, 0.6, 0.25), ("eta", 0.15, float(os.environ.get("IAM_ETAMAX", "0.95")), 0.6),
       ("fcap", 0.0, 1.0, 0.7), ("om", 0.05, 0.6, 0.3), ("g", 0.0, 2.0, 0.8), ("nu", 0.0, 1.0, 0.8),
       ("sig", 0.0, float(os.environ.get("IAM_SIGMAX", "0.05")), 0.02), ("lnRL", np.log(6.0), np.log(80.0), np.log(20.0)),
       ("lnaL", np.log(0.05), np.log(1.0), np.log(0.25)), ("lnfL", np.log(1e-4), np.log(float(os.environ.get("IAM_FLMAX", "0.4"))), np.log(0.02)),
       ("lnK", np.log(1e3), np.log(1e7), np.log(1.1e5))]
SCALE2 = os.environ.get("IAM_SCALE2", "0") == "1"
SCALE = os.environ.get("IAM_SCALE", "0") == "1" or SCALE2
CB = os.environ.get("IAM_CB", "0") == "1" or SCALE
NUTGRID = tuple(float(v) for v in os.environ.get("IAM_NUTGRID", "1,0.5").split(",")) if SCALE else (None,)
BASE = os.environ.get("IAM_BASE", "0") == "1" or CB
CBGRID = tuple(float(v) for v in os.environ.get("IAM_CBGRID", "1,0.4").split(",")) if CB else (None,)
SMALL = os.environ.get("IAM_SMALL", "0") == "1" or BASE
FBGRID = tuple(float(v) for v in os.environ.get("IAM_FBGRID", "0.25,0.5,0.8").split(","))
if SMALL:
    PAR = [q for q in PAR if q[0] != "lnRL"] + [("lnRL", np.log(float(os.environ.get("IAM_RLMIN", "10"))), np.log(80.0), np.log(20.0)),
           ("lnRs", np.log(0.5), np.log(4.0), np.log(1.2)), ("lnas", np.log(0.3), np.log(2.0), np.log(1.0)),
           ("cs", 0.0, 0.6, 0.2), ("oms", 0.15, 0.6, 0.35)]
if BASE:
    PAR = PAR + [("fb", 0.0, float(os.environ.get("IAM_FBMAX", "0.9")), 0.5), ("rb", 0.5, 1.0, 0.95), ("bg", 0.0, 1.0, 0.8)]
if CB:
    PAR = PAR + [("cb", 0.05, 1.0, 1.0)]
ROUGH = os.environ.get("IAM_ROUGH", "0") == "1"
if ROUGH:
    PAR = PAR + [("lnAr", np.log(1e-4), np.log(1e4), np.log(float(os.environ.get("IAM_AR0", "1")))),
                 ("lnxi", np.log(5.0), np.log(500.0), np.log(50.0)), ("Hr", 0.1, 1.0, 0.5)]


CONFORM = os.environ.get("IAM_CONFORM", "0") == "1"
if CONFORM and not ROUGH:
    raise SystemExit("IAM_CONFORM needs IAM_ROUGH=1")
if CONFORM:
    PAR = PAR + [("kc", 0.0, 1.0, float(os.environ.get("IAM_KC0", "0.0")))]
if SCALE:
    PAR = PAR + [("nut", 0.0, 1.0, 1.0)]
if SCALE2:
    PAR = PAR + [("nul", 0.0, 1.0, 0.3)]
GSH = os.environ.get("IAM_GSH", "0") == "1"
if GSH:
    PAR = PAR + [("sH", 0.0, 0.4, 0.1)]


if os.environ.get("IAM_SQ", "lma") == "free":
    PAR = PAR + [("lnkD", np.log(0.7), np.log(6.0), 0.0)]

BASEVOID = os.environ.get("IAM_BASEVOID", "0") == "1"
BURIED = os.environ.get("IAM_BURIED", "0") == "1"


PFILL = os.environ.get("IAM_PFILL", "0") == "1"
if PFILL and (not BASE or CB or BURIED):
    raise SystemExit("IAM_PFILL needs IAM_BASE=1 without IAM_CB / IAM_SCALE / IAM_BURIED")


PDOM = os.environ.get("IAM_PDOM", "0") == "1"
if PDOM and not PFILL:
    raise SystemExit("IAM_PDOM needs IAM_PFILL=1")
if PDOM:
    PAR = PAR + [("lnLd", np.log(8.0), np.log(100.0), np.log(30.0)), ("lnXd", np.log(2.0), np.log(60.0), np.log(10.0)),
                 ("lnAd", np.log(1e-6), np.log(2.0), np.log(1e-6))]
if (BASEVOID or BURIED) and (not BASE or CB):
    raise SystemExit("IAM_BASEVOID / IAM_BURIED need IAM_BASE=1 without IAM_CB / IAM_SCALE (Model4c, GrainsB)")
if BASEVOID:
    PAR = PAR + [("lnRv", np.log(0.5), np.log(4.0), np.log(2.0)), ("omv", 0.15, 0.6, 0.3)]
if BURIED:
    PAR = PAR + [("fbur", 0.0, float(os.environ.get("IAM_FBURMAX", "0.5")), 0.1),
                 ("lnRbur", np.log(0.5), np.log(4.0), np.log(1.5)), ("zbur", 0.0, float(os.environ.get("IAM_ZBURMAX", "5.0")), 0.5)]
VOID_LZ = float(os.environ["IAM_BASEVOID_LZ"]) if os.environ.get("IAM_BASEVOID_LZ") else None
VOID_AMP = os.environ.get("IAM_BASEVOID_AMP", "0") == "1"
if VOID_AMP and not BASEVOID:
    raise SystemExit("IAM_BASEVOID_AMP needs IAM_BASEVOID=1")
if VOID_AMP:
    PAR = PAR + [("kv", 0.0, 1.0, 0.5)]
UL_STARTS = ([{"lnRv": np.log(r)} for r in (1.0, 2.0, 3.0)] if BASEVOID else [{}])
if VOID_AMP:
    UL_STARTS = [dict(a, kv=k) for a in UL_STARTS for k in (0.001, 1.0)]
UL_STARTS = [dict(a, **b) for a in UL_STARTS for b in ([{"fbur": 0.1}, {"fbur": 0.3}] if BURIED else [{}])]

LC = os.environ.get("IAM_LC", "0") == "1"
if LC and (not BASE or CB):
    raise SystemExit("IAM_LC needs IAM_BASE=1 without IAM_CB / IAM_SCALE (Model4g on Model4c, GrainsB)")
LC_LNMIN = float(np.log(2 * np.pi / 0.30))
if LC:
    PAR = PAR + [("lnLc", LC_LNMIN, np.log(400.0), np.log(40.0)),
                 ("phiS", 0.0, 1.0, float(os.environ.get("IAM_LC_PHI0", "0.5")))]
    UL_STARTS = [dict(a, **b) for a in UL_STARTS for b in ({}, {"phiS": 0.02})]

LSOFT = os.environ.get("IAM_LSOFT", "0") == "1"
if LSOFT and CB:
    raise SystemExit("IAM_LSOFT is not implemented for IAM_CB / IAM_SCALE (Model4d handles the large clusters itself)")
if LSOFT:
    PAR = PAR + [("sL", 0.0, float(os.environ.get("IAM_LSOFT_MAX", "3.0")), 0.3)]
    UL_STARTS = [dict(a, **b) for a in UL_STARTS for b in ({}, {"sL": 0.001})]

SIGL = os.environ.get("IAM_SIGL", "0") == "1"
if SIGL and CB:
    raise SystemExit("IAM_SIGL is not implemented for IAM_CB / IAM_SCALE (Model4d handles the large clusters itself)")
SIGL_START = float(os.environ.get("IAM_SIGL_START", "0.02"))
if SIGL:
    PAR = PAR + [("sigL", 0.0, float(os.environ.get("IAM_SIGMAX", "0.05")), 0.02)]

FILMFLUCT = os.environ.get("IAM_FILMFLUCT", "0") == "1"
if FILMFLUCT and (not BASE or CB):
    raise SystemExit("IAM_FILMFLUCT needs IAM_BASE=1 without IAM_CB / IAM_SCALE (Model4c, GrainsB)")


FLUCT_PSD = os.environ.get("IAM_FLUCT_PSD", "0") == "1"
if FLUCT_PSD and not FILMFLUCT:
    raise SystemExit("IAM_FLUCT_PSD needs IAM_FILMFLUCT=1")
if FILMFLUCT:
    PAR = PAR + [("lnRP", np.log(2.0 if (FLUCT_PSD or os.environ.get("IAM_FLUCT_TS", "0") == "1") else 8.0), np.log(150.0), np.log(30.0)), ("lnAP", np.log(1e-6), np.log(5.0), np.log(1e-6))]
    UL_STARTS = [dict(a, **b) for a in UL_STARTS for b in ({}, {"lnRP": float(np.log(30.0)), "lnAP": float(np.log(0.3))})]
if FLUCT_PSD:
    PAR = PAR + [("Hf", 0.0, 1.0, 0.5)]


FLUCT_TS = os.environ.get("IAM_FLUCT_TS", "0") == "1"
if FLUCT_TS and (not FILMFLUCT or FLUCT_PSD):
    raise SystemExit("IAM_FLUCT_TS needs IAM_FILMFLUCT=1 without IAM_FLUCT_PSD")
if FLUCT_TS:
    PAR = PAR + [("lnDN", np.log(5.0), np.log(200.0), np.log(30.0))]


def lc_start(t):
    'lnLc start of the window at t (s) from IAM_LC_START (json {t: Lc nm}, inline or a file; piecewise constant, the'
    src = os.environ.get("IAM_LC_START", "").strip()
    if not src:
        return float(np.log(40.0))
    tab = json.loads(src if src.startswith("{") else Path(src).read_text(encoding="utf-8"))
    keys = sorted((float(k), float(v)) for k, v in tab.items())
    below = [v for k, v in keys if k <= float(t) + 1e-9]
    Lc = below[-1] if below else keys[0][1]
    return float(np.clip(np.log(Lc), LC_LNMIN + 1e-6, np.log(400.0) - 1e-6))


class Fit:
    def __init__(self, scan, t):
        self.s, self.w = scan, scan.window(t)
        self.fix = {} if os.environ.get("IAM_FREEK", "0") == "1" else {"lnK": float(np.log(1.1e5))}
        for item in filter(None, os.environ.get("IAM_FIX", "").split(",")):
            k, v = item.split("="); self.fix[k.strip()] = float(v)
        self.free = [p for p in PAR if p[0] not in self.fix]
        self.lb = np.array([p[1] for p in self.free]); self.ub = np.array([p[2] for p in self.free])
        cls = (M4.Model4f if SCALE2 else M4.Model4e if SCALE else M4.Model4d if CB else M4.Model4g if LC else M4.Model4c if BASE else M4.Model4b if SMALL else M4.Model4)
        self.mdl = cls(scan, eps=float(os.environ.get("IAM_EPS", "0")), interf="lma")
        if os.environ.get("IAM_POLYT_NM") and scan.sub in M.POLY:
            self.mdl.set_poly_t(float(os.environ["IAM_POLYT_NM"]))
        self.ok = self.w["mask"]; self.nev = 0

    def full(self, x):
        p = dict(self.fix); p.update({q[0]: v for q, v in zip(self.free, x)})
        for q in PAR:
            p.setdefault(q[0], q[3])
        return p

    def model(self, x, parts=False):
        p = self.full(x)
        if BASE:
            h_surf = self.w["h_m"] * (1.0 - p["fbur"]) if BURIED else self.w["h_m"]
            m = (M4.GrainsC if CB else M4.GrainsB)(np.exp(p["lnR"]), p["s"], p["eta"], p["fcap"], p["om"], h_surf,
                                                 fb=p["fb"], rb=p["rb"], **({"cb": p["cb"]} if CB else {}),
                                                 RL=np.exp(p["lnRL"]), aL=np.exp(p["lnaL"]), fL=np.exp(p["lnfL"]),
                                                 Rs=np.exp(p["lnRs"]), as_=np.exp(p["lnas"]), cs=p["cs"], oms=p["oms"])
        elif SMALL:
            m = M4.Grains2(np.exp(p["lnR"]), p["s"], p["eta"], p["fcap"], p["om"], self.w["h_m"], RL=np.exp(p["lnRL"]),
                           aL=np.exp(p["lnaL"]), fL=np.exp(p["lnfL"]), Rs=np.exp(p["lnRs"]), as_=np.exp(p["lnas"]),
                           cs=p["cs"], oms=p["oms"])
        else:
            m = M4.Grains(np.exp(p["lnR"]), p["s"], p["eta"], p["fcap"], p["om"], self.w["h_m"],
                          RL=np.exp(p["lnRL"]), aL=np.exp(p["lnaL"]), fL=np.exp(p["lnfL"]))
        if GSH:
            m.set_height_spread(p["sH"])
        if PFILL and getattr(self.mdl, "pfilm", False) and self.s.sub in M.POLY:
            m.fill_d = M.dc(self.s.sub)
        if PDOM:
            self.mdl.dom = (float(np.exp(p["lnLd"])), float(np.exp(p["lnXd"])), float(np.exp(p["lnAd"])))
        if "lnkD" in p:
            m.kD = float(np.exp(p["lnkD"]))
        self.mdl.nu_isl = p["nu"]
        if SCALE:
            self.mdl.nu_top = p["nut"]
        if SCALE2:
            self.mdl.nu_L = p["nul"]
        if ROUGH:
            self.mdl.rough = (float(np.exp(p["lnAr"])), float(np.exp(p["lnxi"])), float(p["Hr"]))
        if CONFORM:
            self.mdl.conform = float(p["kc"])
        if BASE:
            self.mdl.bg = p["bg"]
        if BASEVOID:
            self.mdl.void = (float(np.exp(p["lnRv"])), float(p["omv"])) + ((None, VOID_LZ, float(p["kv"])) if VOID_AMP else
                                                                             (None, VOID_LZ) if VOID_LZ else ())
        if BURIED:
            self.mdl.buried = (float(p["fbur"] * self.w["h_m"]), float(np.exp(p["lnRbur"])), float(p["zbur"]))
        if LC:
            self.mdl.Lc, self.mdl.phiS = float(np.exp(p["lnLc"])), float(p["phiS"])
        if LSOFT:
            self.mdl.sL = float(p["sL"])
        if SIGL and (self.mdl.sigL is None or abs(self.mdl.sigL - p["sigL"]) > 1e-12):
            self.mdl.set_smear_L(p["sigL"])
        if FILMFLUCT:
            self.mdl.fluct = (float(np.exp(p["lnRP"])), float(np.exp(p["lnAP"])))
        if FLUCT_PSD:
            self.mdl.fluct_psd, self.mdl.fluct_h = True, float(p["Hf"])
        if FLUCT_TS:
            self.mdl.fluct_d = float(np.exp(p["lnDN"]))
        if abs(self.mdl.sig_af - p["sig"]) > 1e-12:
            self.mdl.set_smear(p["sig"])
        return self.mdl.predict_grains(self.w, m, np.exp(p["lnK"]), g=p["g"], parts=parts), m, p

    def resid(self, x):
        self.nev += 1
        (mu, _, _), _, _ = self.model(x)
        d = M.dev5(self.w, mu, self.s.V_pre)
        return (np.sign(self.w["Y"] - mu * self.w["E"]) * np.sqrt(d))[self.ok]

    def x0(self, **over):
        p = {q[0]: q[3] for q in PAR}
        if LC:
            p["lnLc"] = lc_start(self.w["t"])
        p.update(over)
        return np.clip(np.array([p[q[0]] for q in self.free]), self.lb + 1e-9, self.ub - 1e-9)

    def run(self, x0, nfev):
        r = least_squares(self.resid, x0, bounds=(self.lb, self.ub), x_scale="jac", diff_step=1e-3, max_nfev=nfev)
        self.last = dict(status=int(r.status), nfev=int(r.nfev),
                         dx=float(np.max(np.abs(r.x - x0) / np.maximum(self.ub - self.lb, 1e-12))), F=float(2 * r.cost))
        return r.x, float(2 * r.cost)


def _share_hq(f, scan, part, mu, p):
    'Share of an underlayer part (K x the smeared dsigma/dOmega) in the model intensity over the fitted cells of'
    c = np.broadcast_to(np.exp(p["lnK"]) * f.mdl.smear(part)[None], mu.shape)
    aft = scan.af - getattr(f.mdl, "eps", 0.0)
    reg = ((aft >= 0.34) & (aft < 0.60))[None, :, None] & ((scan.q >= 0.8) & (scan.q < 1.8))[None, None, :] & f.ok
    return np.sum(np.where(reg, c, 0.0)) / max(float(np.sum(np.where(reg, mu, 0.0))), 1e-300)


def main():
    label, times = sys.argv[1], [float(v) for v in sys.argv[2:]]
    OUT.mkdir(parents=True, exist_ok=True)
    scan = M.Scan(label)
    for t in times:
        t0 = time.time(); f = Fit(scan, t); starts = []
        for R in (2.0, 6.0, 15.0):
            for eta in (0.5, 0.85):
                for fc in (0.4, 0.9):
                    for fb in (FBGRID if BASE else (None,)):
                        for cb in CBGRID:
                            for nut in NUTGRID:
                                x = f.x0(lnR=np.log(R), eta=eta, fcap=fc, **({"fb": fb} if BASE else {}),
                                         **({"cb": cb} if CB else {}), **({"nut": nut} if SCALE else {}))
                                starts.append((float(np.sum(f.resid(x) ** 2)), x))
        starts.sort(key=lambda v: v[0])
        short = sorted((f.run(x, 30)[::-1] for _, x in starts[:4]), key=lambda v: v[0])
        cands = [(f"grid{i}", x) for i, (_, x) in enumerate(short[:2])]
        names = {q[0] for q in PAR}
        for br in filter(None, os.environ.get("IAM_WARM", "").split(",")):
            wp = IAMP.WORK / br / f"{label}_{t:g}.json"
            if wp.exists():
                prm = json.loads(wp.read_text(encoding="utf-8"))["params"]
                if SIGL and "sigL" not in prm:
                    prm = dict(prm, sigL=prm.get("sig", 0.02))
                    cands.append((br + "_sigL", f.x0(**dict({k: v for k, v in prm.items() if k in names},
                                                             sigL=min(prm["sigL"], SIGL_START)))))
                for i_ul, ul in enumerate(UL_STARTS):
                    cands.append((br + (f"_ul{i_ul}" if ul else ""),
                                  f.x0(**dict({k: v for k, v in prm.items() if k in names}, **ul))))
                if BASE and "nu" in f.fix and "rb" in prm and abs(prm.get("nu", f.fix["nu"]) - f.fix["nu"]) > 1e-6:

                    q = dict(prm, rb=prm["rb"] * prm["nu"] / f.fix["nu"])
                    cands.append((br + "_rbnu", f.x0(**{k: v for k, v in q.items() if k in names})))
        if os.environ.get("IAM_ONLY_WARM") == "1":
            cands = [c for c in cands if not c[0].startswith("grid")]
            if not cands:
                raise SystemExit(f"IAM_ONLY_WARM: no stored solution for {label} {t:g}")
        runs = []
        for src, x_ in cands:
            xx, Fc = f.run(x_, int(round(200 * M.NFEV_MULT)))
            (mu_c, _, _), _, _ = f.model(xx)
            runs.append((src, Fc, xx, dict(f.last, dev5_per_cell=float(M.dev5(f.w, mu_c, scan.V_pre).sum() / f.ok.sum()))))
        start, F, x, _ = min(runs, key=lambda v: v[1])
        (mu, sub, metal), m, p = f.model(x)
        d = M.dev5(f.w, mu, scan.V_pre); n = int(f.ok.sum()); tot = np.sum(np.where(f.ok, mu, 0))
        rec = dict(scan=label, t=t, h_m=f.w["h_m"], params={k: float(v) for k, v in p.items()}, fixed=f.fix,
                   R=m.R, s=m.s, eta=m.eta, fcap=m.fcap, T=m.T, Hc=m.Hc, Hcap=m.Hcap, sH=float(getattr(m, "sH", 0.0)), D=float(m.D), omega=m.omega,
                   RL=m.L.R, fL=m.fL, coverage=float(m.coverage), g=p["g"], nu=p["nu"], sig=p["sig"],
                   fs=float(getattr(m, "fs", 0.0)), Rs=float(getattr(m, "sp", m.L).R) if SMALL else 0.0,
                   D_s=float(getattr(m, "D_s", 0.0)),
                   fb=float(getattr(m, "fb", 0.0)), rb=float(getattr(m, "rb", 0.0)), t_b=float(getattr(m, "t_b", 0.0)),
                   cb=float(getattr(m, "cb", 1.0)), nut=float(p.get("nut", p["nu"])),
                   bg=float(p.get("bg", 1.0)) if BASE else None, model=type(f.mdl).__name__,
                   K=float(np.exp(p["lnK"])), dev5_per_cell=float(d.sum() / n), n_cells=n, start=start,
                   candidates={src: float(Fc / n) for src, Fc, _, _i in runs},
                   optimizer={src: i_ for src, _, _, i_ in runs},
                   share_sub=float(np.sum(np.where(f.ok, sub, 0)) / tot), seconds=time.time() - t0, pixmask=scan.pixmask,
                   pfilm=bool(getattr(f.mdl, "pfilm", False)), poly_t=getattr(f.mdl, "poly_t", None))
        if LC:
            rec.update(Lc=float(np.exp(p["lnLc"])), phiS=float(p["phiS"]), q_c=float(2 * np.pi / np.exp(p["lnLc"])),
                       lc_start=os.environ.get("IAM_LC_START"))
        if LSOFT:
            rec.update(sL=float(p["sL"]))
        if SIGL:
            rec.update(sigL=float(p["sigL"]))
        if FILMFLUCT:
            rec.update(R_P=float(np.exp(p["lnRP"])), A_P=float(np.exp(p["lnAP"])))
        if FLUCT_PSD:
            rec.update(Hf=float(p["Hf"]))
        extra = {}
        if BASEVOID or BURIED:
            import underlayer as U
            (_, pr), m_, _ = f.model(x, parts=True)
            rec["underlayer"] = U.describe(f.mdl, m_)
            if BURIED:
                rec["h_surf"] = float(f.w["h_m"] * (1.0 - p["fbur"]))
            for k in ("void", "bur"):
                if k in pr:
                    extra[k] = np.asarray(pr[k], np.float32)
                    rec[f"{k}_share_hq"] = float(_share_hq(f, scan, pr[k], mu, p))
        if scan.shadow is not None:
            rec["instrument"] = dict(shadow_bg=M.SHADOW_BG, shadow_aB=M.SHADOW_AB, shadow_sha256=scan.shadow.sha256,
                                     **M.shadow_desc(f.w, mu, f.ok, f.mdl))
        (OUT / f"{label}_{t:g}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        np.savez_compressed(OUT / f"{label}_{t:g}_mu.npz", mu=mu.astype(np.float32), sub=sub.astype(np.float32), **extra)
        print(f"{label} t {t:g} h {f.w['h_m']:.2f}: R {m.R:.2f} s {m.s:.2f} eta {m.eta:.2f} fcap {m.fcap:.2f} T {m.T:.2f} "
              f"D {m.D:.2f} om {m.omega:.2f} | small R {rec['Rs']:.2f} fs {rec['fs']:.3f} Ds {rec['D_s']:.2f} | L R {m.L.R:.1f} f {m.fL:.3f} | g {p['g']:.2f} nu {p['nu']:.2f} "
              f"sig {p['sig']:.3f} K {rec['K']:.3g} | base fb {rec['fb']:.2f} rb {rec['rb']:.2f} t_b {rec['t_b']:.1f} bg {p.get('bg', 1.0):.2f} | dev5/cell {rec['dev5_per_cell']:.2f} ({rec['seconds']:.0f} s)",
              flush=True)


if __name__ == "__main__":
    main()
