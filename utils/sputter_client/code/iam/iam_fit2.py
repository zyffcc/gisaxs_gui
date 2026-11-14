'IAM v0.2 window fits on Si (sliced EM DWBA checked against BornAgain; large-cluster population; geometry eps / sig).'

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
import iam2 as M2

OUT = IAMP.WORK / os.environ.get("IAM_OUT", "fits2")
PAR = [
    ("lnR", np.log(0.4), np.log(float(os.environ.get("IAM_RMAX", "15"))), np.log(2.0)),
    ("lna", np.log(0.2), np.log(float(os.environ.get("IAM_AMAX", "2.5"))), np.log(1.2)),
    ("s", 0.05, 0.6, 0.25),
    ("om", 0.08, 0.6, 0.3),
    ("g", 0.0, 2.0, 1.0),
    ("nu", 0.0, 1.0, 0.3),
    ("lnRL", np.log(6.0), np.log(80.0), np.log(25.0)),
    ("lnaL", np.log(0.05), np.log(1.0), np.log(0.25)),
    ("lnfL", np.log(1e-4), np.log(0.5), np.log(0.03)),
    ("eps", -0.08, float(os.environ.get("IAM_EPSMAX", "0.04")), -0.03),
    ("sig", 0.0, float(os.environ.get("IAM_SIGMAX", "0.05")), 0.015),
    ("lnK", np.log(1e3), np.log(1e7), np.log(1.13e5)),
]
POLYT = os.environ.get("IAM_POLYT", "0") == "1"
if POLYT:
    PAR = PAR + [("poly_t", 5.0, 200.0, 40.0)]
MASSF = os.environ.get("IAM_MASSF", "0") == "1"
if MASSF:
    PAR = PAR + [("lnfm", np.log(0.3), np.log(3.0), 0.0)]
EMBED = os.environ.get("IAM_EMBED", "0") == "1"
if EMBED:
    PAR = PAR + [("emb", 0.0, 0.95, 0.3)]
OZ = os.environ.get("IAM_OZ", "0") == "1"
if OZ:
    PAR = PAR + [("lnAoz", np.log(1e-8), np.log(1e3), np.log(1e-2)), ("lnxio", np.log(3.0), np.log(300.0), np.log(30.0)),
                 ("Hoz", 0.1, 1.5, 0.8), ("lnszo", np.log(0.3), np.log(30.0), np.log(2.0))]
EMBGRID = tuple(float(v) for v in os.environ.get("IAM_EMBGRID", "0.1,0.5").split(",")) if EMBED else (None,)
POLYTGRID = tuple(float(v) for v in os.environ.get("IAM_POLYTGRID", "20,40,80").split(",")) if POLYT else (None,)


def fixed():
    out = {}
    for item in filter(None, os.environ.get("IAM_FIX", "").split(",")):
        k, v = item.split("=")
        out[k.strip()] = float(v)
    return out


class Fit:
    def __init__(self, scan, t):
        self.s, self.w = scan, scan.window(t)
        self.fix = fixed()
        self.free = [p for p in PAR if p[0] not in self.fix]
        self.lb = np.array([p[1] for p in self.free]); self.ub = np.array([p[2] for p in self.free])
        self.mdl = M2.Model2(scan, interf=os.environ.get("IAM_INTERF", "lma"))
        if self.mdl.interf == "hd":
            self.free = [(q[0], 0.0, 1.0, 0.15) if q[0] == "om" else q for q in self.free]
            self.lb = np.array([q[1] for q in self.free]); self.ub = np.array([q[2] for q in self.free])
        self.ok = self.w["mask"]
        self.nev = 0

    def full(self, x):
        p = dict(self.fix)
        p.update({q[0]: v for q, v in zip(self.free, x)})
        for q in PAR:
            p.setdefault(q[0], q[3])
        return p

    def model(self, x, parts=False):
        p = self.full(x)
        m = M2.Morph2(np.exp(p["lnR"]), np.exp(p["lna"]), p["s"], p["om"], self.w["h_m"] * np.exp(p.get("lnfm", 0.0)),
                      RL=np.exp(p["lnRL"]), aL=np.exp(p["lnaL"]), fL=np.exp(p["lnfL"]))
        self.mdl.nu_isl = p["nu"]
        if os.environ.get("IAM_TIE_NU", "0") == "1":
            self.mdl.nu_sub = p["nu"]
        if abs(self.mdl.eps - p["eps"]) > 1e-12:
            self.mdl.set_geom(p["eps"])
        if abs(self.mdl.sig_af - p["sig"]) > 1e-12:
            self.mdl.set_smear(p["sig"])
        if "poly_t" in p and abs((self.mdl.poly_t or 0.0) - p["poly_t"]) > 1e-9:
            self.mdl.set_poly_t(p["poly_t"])
        if "emb" in p:
            self.mdl.embed_frac = p["emb"]
        if "lnAoz" in p:
            self.mdl.oz = (float(np.exp(p["lnAoz"])), float(np.exp(p["lnxio"])), float(p["Hoz"]), float(np.exp(p["lnszo"])))
        return self.mdl.predict(self.w, m, np.exp(p["lnK"]), g=p["g"], parts=parts), m, p

    def resid(self, x):
        self.nev += 1
        (mu, _, _), _, _ = self.model(x)
        d = M.dev5(self.w, mu, self.s.V_pre)
        return (np.sign(self.w["Y"] - mu * self.w["E"]) * np.sqrt(d))[self.ok]

    def x0(self, **over):
        p = {q[0]: q[3] for q in PAR}
        p.update(over)
        return np.clip(np.array([p[q[0]] for q in self.free]), self.lb + 1e-9, self.ub - 1e-9)

    def run(self, x0, nfev):
        r = least_squares(self.resid, x0, bounds=(self.lb, self.ub), x_scale="jac", diff_step=1e-3, max_nfev=nfev)
        self.last = dict(status=int(r.status), nfev=int(r.nfev),
                         dx=float(np.max(np.abs(r.x - x0) / np.maximum(self.ub - self.lb, 1e-12))), F=float(2 * r.cost))
        return r.x, float(2 * r.cost)


RGRID = tuple(float(v) for v in os.environ.get("IAM_RGRID", "1.2,2.5").split(","))
OMGRID = tuple(float(v) for v in os.environ["IAM_OMGRID"].split(",")) if os.environ.get("IAM_OMGRID") else (None,)
AOZGRID = (1e-3, 1e-1) if OZ else (None,)


def main():
    label, times = sys.argv[1], [float(v) for v in sys.argv[2:]]
    OUT.mkdir(parents=True, exist_ok=True)
    scan = M.Scan(label)
    for t in times:
        t0 = time.time()
        f = Fit(scan, t)
        starts = []
        for R in RGRID:
            for a in (0.8, 1.5):
                for fL in (0.01, 0.08):
                    for pt in POLYTGRID:
                        for eb in EMBGRID:
                          for om in OMGRID:
                           for az in AOZGRID:
                            x = f.x0(lnR=np.log(R), lna=np.log(a), lnfL=np.log(fL), **({"poly_t": pt} if POLYT else {}),
                                     **({"emb": eb} if EMBED else {}), **({"om": om} if om is not None else {}),
                                     **({"lnAoz": np.log(az)} if az is not None else {}))
                            starts.append((float(np.sum(f.resid(x) ** 2)), x))
        starts.sort(key=lambda v: v[0])
        short = sorted((f.run(x, 30)[::-1] for _, x in starts[:4]), key=lambda v: v[0])
        cands = [(f"grid{i}", x) for i, (_, x) in enumerate(short[:2])] + M.warm_starts(f, label, t, {q[0] for q in PAR})
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
        d = M.dev5(f.w, mu, scan.V_pre)
        n = int(f.ok.sum())
        tot = np.sum(np.where(f.ok, mu, 0))
        rec = dict(scan=label, t=t, h_m=f.w["h_m"], cfac=f.w["cfac"], params={k: float(v) for k, v in p.items()},
                   fixed=f.fix, R=m.S.R, H=m.S.R * m.S.a, s=m.S.s, omega=m.omega, D=float(m.D), coverage=m.coverage,
                   RL=m.L.R, HL=m.L.R * m.L.a, fL=m.fL, eta_hd=m.eta_hd, K=float(np.exp(p["lnK"])), g=p["g"], nu=p["nu"], eps=p["eps"],
                   sig=p["sig"], dev5=float(d.sum()), dev5_per_cell=float(d.sum() / n), n_cells=n,
                   sub=scan.sub, poly_t=getattr(f.mdl, "poly_t", None), embed_depth=f.mdl.embed_depth(m),
                   share_sub=float(np.sum(np.where(f.ok, sub, 0)) / tot), nev=f.nev, seconds=time.time() - t0,
                   start=start, candidates={src: float(Fc / n) for src, Fc, _, _i in runs},
                   optimizer={src: i_ for src, _, _, i_ in runs}, pixmask=scan.pixmask)
        (OUT / f"{label}_{t:g}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        np.savez_compressed(OUT / f"{label}_{t:g}_mu.npz", mu=mu.astype(np.float32), sub=sub.astype(np.float32))
        print(f"{label} t {t:g}: R {m.S.R:.2f} H {m.S.R*m.S.a:.2f} s {m.S.s:.2f} om {m.omega:.2f} D {m.D:.2f} cov {m.coverage:.2f} "
              f"eta {m.eta_hd:.2f} | L R {m.L.R:.1f} H {m.L.R*m.L.a:.1f} f {m.fL:.3f} | g {p['g']:.2f} nu {p['nu']:.2f} eps {p['eps']:+.3f} "
              f"sig {p['sig']:.3f} K {np.exp(p['lnK']):.3g} | dev5/cell {rec['dev5_per_cell']:.2f} ({rec['seconds']:.0f} s)",
              flush=True)


if __name__ == "__main__":
    main()
