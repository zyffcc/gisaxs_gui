'IAM v0.2 TiO2 partition test (phase B). K fixed to the Si value; resolvable islands carry f_isl x h_m (morphology'
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import iam_paths as IAMP
import numpy as np
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iam as M
import iam2 as M2
OUT = IAMP.WORK / os.environ.get("IAM_OUT", "fitsT")
PAR = [("lnR", np.log(0.4), np.log(float(os.environ.get("IAM_RMAX", "15"))), np.log(2.0)), ("lna", np.log(0.2), np.log(2.5), np.log(1.2)),
       ("s", 0.05, 0.6, 0.25), ("om", 0.08, 0.6, 0.3), ("g", 0.0, float(os.environ.get("IAM_GMAX", "0.5")), 0.05),
       ("lnRL", np.log(6.0), np.log(80.0), np.log(20.0)), ("lnaL", np.log(0.05), np.log(1.0), np.log(0.25)),
       ("lnfL", np.log(1e-4), np.log(0.5), np.log(0.03)), ("sig", 0.0, float(os.environ.get("IAM_SIGMAX", "0.05")), 0.015),
       ("lnfisl", np.log(0.02), np.log(float(os.environ.get("IAM_FISLMAX", "1.5"))), np.log(0.4)),
       ("lnkopt", np.log(0.05), np.log(float(os.environ.get("IAM_KOPTMAX", "2.0"))), np.log(1.0)),
       ("lnK", np.log(1e3), np.log(1e7), np.log(1.1e5))]
if os.environ.get("IAM_COAT", "bulk") == "shell":
    PAR = PAR + [("lnhs", np.log(0.2), np.log(80.0), np.log(3.0))]
if os.environ.get("IAM_LAM", "0") == "1":
    PAR = PAR + [("lnlam", np.log(2.0), np.log(300.0), np.log(30.0))]
if os.environ.get("IAM_BURY", "0") == "1":
    PAR = PAR + [("z0", 0.0, float(os.environ.get("IAM_Z0MAX", "340")), 120.0),
                 ("lnwb", np.log(5.0), np.log(150.0), np.log(30.0))]
OZ = os.environ.get("IAM_OZ", "0") == "1"
if OZ:
    PAR = PAR + [("lnAoz", np.log(1e-8), np.log(float(os.environ.get("IAM_AOZMAX", "10"))), np.log(1e-3)),
                 ("lnxio", np.log(3.0), np.log(300.0), np.log(40.0)),
                 ("Hoz", float(os.environ.get("IAM_HOZMIN", "0.1")), 1.5, float(os.environ.get("IAM_HOZ0", "0.8")))]
    if os.environ.get("IAM_OZZ", "0") == "1":
        PAR = PAR + [("lnszo", np.log(0.3), np.log(30.0), np.log(3.0))]
    if os.environ.get("IAM_OZDEPTH", "0") == "1":
        PAR = PAR + [("zn", 0.0, float(os.environ.get("IAM_Z0MAX", "340")), float(os.environ.get("IAM_ZN0", "300"))),
                     ("lnwn", np.log(5.0), np.log(300.0), np.log(50.0))]
if os.environ.get("IAM_CAP", "0") == "1":
    PAR = PAR + [("kcap", 0.0, 1.0, float(os.environ.get("IAM_CAP0", "0.5"))),
                 ("phicap", float(os.environ.get("IAM_PHICAPMIN", "0.05")), 1.0, float(os.environ.get("IAM_PHICAP0", "0.5")))]
    if os.environ.get("IAM_CAPPATCH", "0") == "1":
        PAR = PAR + [("lncp", float(np.log(0.02)), 0.0, float(np.log(float(os.environ.get("IAM_CP0", "0.3")))))]
        if os.environ.get("IAM_CAPSCAT", "0") == "1":
            PAR = PAR + [("lnxic", float(np.log(3.0)), float(np.log(300.0)), float(np.log(30.0)))]
if os.environ.get("IAM_TFILM", "0") == "1":
    PAR = PAR + [("phif", 0.0, 1.0, float(os.environ.get("IAM_PHIF0", "0.2"))),
                 ("lnLf", np.log(10.0), np.log(1000.0), np.log(float(os.environ.get("IAM_LF0", "60"))))]
    if M2.TFILM_X == "fit":
        PAR = PAR + [("lnxf", np.log(1e-3), np.log(float(os.environ.get("IAM_XFMAX", "10"))),
                      np.log(float(os.environ.get("IAM_XF0", "1"))))]
    if M2.TFILM_T == "fit":
        PAR = PAR + [("lntf", np.log(0.5), np.log(float(os.environ.get("IAM_TFMAX", "60"))),
                      np.log(float(os.environ.get("IAM_TF0", "10"))))]
COVMAX = float(os.environ["IAM_COVMAX"]) if os.environ.get("IAM_COVMAX") else None
TCAPMAX = float(os.environ["IAM_TCAPMAX"]) if os.environ.get("IAM_TCAPMAX") else None
DEFAULT_FIX = {"lnK": float(np.log(1.1e5))}
if os.environ.get("IAM_FREEK", "0") == "1":
    DEFAULT_FIX = {}
class Fit:
    def __init__(self, scan, t):
        self.s, self.w = scan, scan.window(t)
        self.fix = dict(DEFAULT_FIX)
        for item in filter(None, os.environ.get("IAM_FIX", "").split(",")):
            k, v = item.split("="); self.fix[k.strip()] = float(v)
        self.free = [p for p in PAR if p[0] not in self.fix]
        self.lb = np.array([p[1] for p in self.free]); self.ub = np.array([p[2] for p in self.free])
        self.mdl = M2.Model2(scan, interf="lma", nu_sub=1.0, eps=0.0, ntheta=4, nphi=5, coarse=(6, 3))
        self.ok = self.w["mask"]; self.nev = 0
    def full(self, x):
        p = dict(self.fix); p.update({q[0]: v for q, v in zip(self.free, x)})
        for q in PAR: p.setdefault(q[0], q[3])
        return p
    def model(self, x, parts=False):
        p = self.full(x)
        fisl, kopt = np.exp(p["lnfisl"]), np.exp(p["lnkopt"])
        m = M2.Morph2(np.exp(p["lnR"]), np.exp(p["lna"]), p["s"], p["om"], self.w["h_m"] * fisl,
                      RL=np.exp(p["lnRL"]), aL=np.exp(p["lnaL"]), fL=np.exp(p["lnfL"]), nsize=5)
        if abs(self.mdl.sig_af - p["sig"]) > 1e-12: self.mdl.set_smear(p["sig"])
        if "lnhs" in p: self.mdl.h_shell = float(np.exp(p["lnhs"]))
        if "kcap" in p: self.mdl.cap_frac, self.mdl.cap_phi = float(p["kcap"]), float(p["phicap"])
        if "lncp" in p: self.mdl.cap_patch = float(np.exp(p["lncp"]))
        if "lnxic" in p: self.mdl.cap_xi = float(np.exp(p["lnxic"]))
        if "lnlam" in p: self.mdl.tio2_lam = float(np.exp(p["lnlam"]))
        if "z0" in p: self.mdl.tio2_bury = (float(p["z0"]), float(np.exp(p["lnwb"])))
        if "lnAoz" in p: self.mdl.oz = (float(np.exp(p["lnAoz"])), float(np.exp(p["lnxio"])), float(p["Hoz"])) \
            + ((float(np.exp(p["lnszo"])),) if "lnszo" in p else ())
        if "zn" in p: self.mdl.oz_depth = (float(p["zn"]), float(np.exp(p["lnwn"])))
        if "phif" in p: self.mdl.tfilm = (float(p["phif"]), float(np.exp(p["lnLf"]))) \
            + ((float(np.exp(p["lntf"])),) if "lntf" in p else ())
        if "lnxf" in p: self.mdl.tfilm_x = float(np.exp(p["lnxf"]))
        out = self.mdl.predict_tio2(self.w, m, np.exp(p["lnK"]), g=p["g"], parts=parts, h_opt=self.w["h_m"] * kopt)
        return out, m, p
    def resid(self, x):
        self.nev += 1
        (mu, _, _), m, _ = self.model(x)
        d = M.dev5(self.w, mu, self.s.V_pre)
        r = (np.sign(self.w["Y"] - mu * self.w["E"]) * np.sqrt(d))[self.ok]
        if COVMAX is not None:
            r = np.r_[r, np.sqrt(r.size) * 10.0 * max(m.coverage - COVMAX, 0.0)]
        if TCAPMAX is not None:
            r = np.r_[r, np.sqrt(np.sum(self.ok)) * max((self.mdl.cap[1] if self.mdl.cap else 0.0) - TCAPMAX, 0.0)]
        return r
    def x0(self, **over):
        p = {q[0]: q[3] for q in PAR}; p.update(over)
        return np.clip(np.array([p[q[0]] for q in self.free]), self.lb + 1e-9, self.ub - 1e-9)
    def run(self, x0, nfev):
        r = least_squares(self.resid, x0, bounds=(self.lb, self.ub), x_scale="jac", diff_step=1e-3, max_nfev=nfev)
        self.last = dict(status=int(r.status), nfev=int(r.nfev),
                         dx=float(np.max(np.abs(r.x - x0) / np.maximum(self.ub - self.lb, 1e-12))), F=float(2 * r.cost))
        return r.x, float(2 * r.cost)
def main():
    M2.TD.assert_fit_allowed()
    label, times = sys.argv[1], [float(v) for v in sys.argv[2:]]
    OUT.mkdir(parents=True, exist_ok=True); scan = M.Scan(label)
    for t in times:
        t0 = time.time(); f = Fit(scan, t); starts = []
        shell = any(q[0] == "lnhs" for q in PAR)
        capg = tuple(float(v) for v in os.environ.get("IAM_CAPGRID", "0.2,0.7").split(","))             if any(q[0] == "kcap" for q in PAR) else (None,)
        lamg = (10.0, 80.0) if any(q[0] == "lnlam" for q in PAR) else (None,)
        buryg = tuple(float(v) for v in os.environ.get("IAM_BURYGRID", "60,160").split(","))             if any(q[0] == "z0" for q in PAR) else (None,)
        ozg = (1e-3, 1e-1) if OZ else (None,)
        only_warm = os.environ.get("IAM_ONLY_WARM") == "1"
        for R in (() if only_warm else (1.2, 2.5)):
            for fi in (0.15, 0.6):
                for ko in (0.3, 1.0):
                    for hf in ((0.3, 1.0) if shell else (None,)):
                        for kc in capg:
                            for lm in lamg:
                              for zb in buryg:
                               for az in ozg:
                                x = f.x0(**({"lnAoz": np.log(az)} if az is not None else {}),
                                         lnR=np.log(R), lnfisl=np.log(fi), lnkopt=np.log(ko),
                                         **({"lnhs": np.log(max(hf * f.w["h_m"], 0.3))} if shell else {}),
                                         **({"kcap": kc} if kc is not None else {}),
                                         **({"lnlam": np.log(lm)} if lm is not None else {}),
                                         **({"z0": zb} if zb is not None else {}))
                                starts.append((float(np.sum(f.resid(x) ** 2)), x))
        starts.sort(key=lambda v: v[0])
        short = sorted((f.run(x, 25)[::-1] for _, x in starts[:3]), key=lambda v: v[0])
        warm = M.warm_starts(f, label, t, {q[0] for q in PAR})
        if os.environ.get("IAM_CAP_WARMK") and any(q[0] == "kcap" for q in f.free):
            ik = [q[0] for q in f.free].index("kcap")
            warm = [(f"{src}+kc{v}", np.r_[x_[:ik], np.clip(float(v), f.lb[ik] + 1e-9, f.ub[ik] - 1e-9), x_[ik + 1:]])
                    for src, x_ in warm for v in os.environ["IAM_CAP_WARMK"].split(",")]
        if os.environ.get("IAM_TF_WARM") and any(q[0] == "lntf" for q in f.free):
            it = [q[0] for q in f.free].index("lntf")
            warm = [(f"{src}+tf{v}", np.r_[x_[:it], np.clip(np.log(float(v)), f.lb[it] + 1e-9, f.ub[it] - 1e-9), x_[it + 1:]])
                    for src, x_ in warm for v in os.environ["IAM_TF_WARM"].split(",")]
        if os.environ.get("IAM_XF_WARM") and all(any(q[0] == k for q in f.free) for k in ("lnxf", "phif")):

            ip, ix = ([q[0] for q in f.free].index(k) for k in ("phif", "lnxf"))
            more = []
            for src, x_ in warm:
                for v in os.environ["IAM_XF_WARM"].split(","):
                    if float(v) > x_[ip] > 0:
                        y = x_.copy()
                        y[ix] = np.clip(x_[ix] + np.log(x_[ip] / float(v)), f.lb[ix] + 1e-9, f.ub[ix] - 1e-9)
                        y[ip] = np.clip(float(v), f.lb[ip] + 1e-9, f.ub[ip] - 1e-9)
                        more.append((f"{src}+pf{v}", y))
            warm = warm + more
        cands = [(f"grid{i}", x) for i, (_, x) in enumerate(short[:2])] + warm
        if only_warm:
            cands = [c for c in cands if not c[0].startswith("grid")]
            if not cands:
                raise SystemExit(f"IAM_ONLY_WARM: no stored solution for {label} {t:g}")
        runs = []
        for src, x_ in cands:
            xx, Fc = f.run(x_, int(round(120 * M.NFEV_MULT)))
            (mu_c, _, _), _, _ = f.model(xx)
            runs.append((src, Fc, xx, dict(f.last, dev5_per_cell=float(M.dev5(f.w, mu_c, scan.V_pre).sum() / f.ok.sum()))))
        start, F, x, _ = min(runs, key=lambda v: v[1])
        (mu, sub, metal), m, p = f.model(x)
        d = M.dev5(f.w, mu, scan.V_pre); n = int(f.ok.sum()); tot = np.sum(np.where(f.ok, mu, 0))
        rec = dict(scan=label, t=t, h_m=f.w["h_m"], params={k: float(v) for k, v in p.items()}, fixed=f.fix,
                   R=m.S.R, H=m.S.R * m.S.a, D=float(m.D), coverage=m.coverage, RL=m.L.R, fL=m.fL,
                   f_isl=float(np.exp(p["lnfisl"])), k_opt=float(np.exp(p["lnkopt"])), g=p["g"], sig=p["sig"],
                   K=float(np.exp(p["lnK"])), dev5_per_cell=float(d.sum() / n), n_cells=n,
                   share_sub=float(np.sum(np.where(f.ok, sub, 0)) / tot), seconds=time.time() - t0,
                   coat=M2.COAT, sig_max=float(f.ub[[q[0] for q in f.free].index('sig')]) if 'sig' not in f.fix else None,
                   g_max=float(os.environ.get("IAM_GMAX", "0.5")),
                   start=start, candidates={src: float(Fc / n) for src, Fc, _, _i in runs},
                   optimizer={src: i_ for src, _, _, i_ in runs}, pixmask=scan.pixmask)
        if "kcap" in p:
            rec.update(cap_place=M2.CAP_PLACE, cap_avg=M2.CAP_AVG, cap_phi=float(p["phicap"]),
                       cap_c=float(np.exp(p["lncp"])) if "lncp" in p else 1.0,
                       cap_mass_nm=float(p["kcap"] * f.w["h_m"] * np.exp(p["lnkopt"])),
                       t_cap_nm=float(f.mdl.cap[1]) if f.mdl.cap else 0.0)
        if "phif" in p:
            rec.update(tfilm_phi_f=float(p["phif"]), tfilm_L_f=float(np.exp(p["lnLf"])), tfilm_n=M2.TFILM_N,
                       tfilm_afr=M2.TFILM_AFR)
            if "lnxf" in p:
                rec.update(tfilm_xf=float(np.exp(p["lnxf"])))
            if M2.TFILM_MET:
                rec.update(tfilm_met=True)
            if M2.TFILM_T != "cap":
                rec.update(tfilm_t=M2.TFILM_T, tfilm_t_f=f.mdl._tfilm_t, tfilm_phi=M2.TFILM_PHI, tfilm_kt=M2.TFILM_KT)
        (OUT / f"{label}_{t:g}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        np.savez_compressed(OUT / f"{label}_{t:g}_mu.npz", mu=mu.astype(np.float32), sub=sub.astype(np.float32))
        print(f"{label} t {t:g} h {f.w['h_m']:.2f}: R {m.S.R:.2f} H {m.S.R*m.S.a:.2f} D {m.D:.2f} cov {m.coverage:.2f} | L R {m.L.R:.1f} f {m.fL:.3f} "
              f"| f_isl {rec['f_isl']:.3f} k_opt {rec['k_opt']:.3f} g {p['g']:.3f} sig {p['sig']:.3f} K {rec['K']:.3g} "
              f"| dev5/cell {rec['dev5_per_cell']:.2f} sub {rec['share_sub']:.2f} ({rec['seconds']:.0f} s)", flush=True)
if __name__ == "__main__":
    main()
