''

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _k, _v in dict(IAM_BASE="1", IAM_FREEK="1", IAM_FIX="nu=1", IAM_ETAMAX="0.98", IAM_SIGMAX="0.3", IAM_RMAX="60",
                   IAM_FLMAX="0.8", IAM_FBMAX="0.98", IAM_SMOOTH="1", IAM_SMEAR_EDGE="zero").items():
    os.environ.setdefault(_k, _v)

import numpy as np
from scipy.optimize import least_squares, nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iam_paths as IAMP
import iam as M
import iam4 as M4
import iam_fit4 as F4
import iam2 as M2
import iam_fit2 as F2

ROOT = HERE.parent
OUT = IAMP.WORK / "static_fits"
TABLE = json.loads((ROOT / "configs" / "static_frames.json").read_text(encoding="utf-8"))["labels"]
EPS0, EPS_SD, FM_SD = -0.017, 0.010, 0.5
SQ = os.environ.get("IAM_SQ", "lma")
AFMIN = float(os.environ.get("IAM_STATIC_AFMIN", "0.03"))


SIGFIX = float(os.environ["IAM_STATIC_SIG"]) if os.environ.get("IAM_STATIC_SIG") else None
TILT = os.environ.get("IAM_STATIC_TILT", "0") == "1"
TILT_A, TILT_B = 0.8, 0.6


BG = os.environ.get("IAM_STATIC_BG", "")
if BG not in ("", "rad"):
    raise SystemExit(f"IAM_STATIC_BG={BG!r}: only 'rad' is defined")
BG_SD = 0.3


NFIX = os.environ.get("IAM_STATIC_NFIX", "")
if NFIX and not (NFIX == "afm" or (NFIX.startswith("afmx") and float(NFIX[4:]) > 0)):
    raise SystemExit(f"IAM_STATIC_NFIX={NFIX!r}: afm or afmx<factor>")
NFIX_FACTOR = float(NFIX[4:]) if NFIX.startswith("afmx") else 1.0
WARM = os.environ.get("IAM_STATIC_WARM", "")


PATCH = os.environ.get("IAM_STATIC_PATCH", "0") == "1"


NB = os.environ.get("IAM_STATIC_NB", "0") == "1"


DONORS = os.environ.get("IAM_STATIC_DONORS", "")


TIO2K = os.environ.get("IAM_STATIC_TIO2K", "0") == "1"
BOUNDS = {k.strip(): tuple(float(v) for v in r.split(":")) for k, r in
          (item.split("=") for item in filter(None, os.environ.get("IAM_STATIC_BOUNDS", "").split(",")))}


def n_fix_of(label):
    'The fixed grain number density (nm^-2) of a label under IAM_STATIC_NFIX.'
    tab = json.loads((ROOT / "configs" / "static_afm_density.json").read_text(encoding="utf-8"))["labels"]
    if label not in tab:
        raise SystemExit(f"IAM_STATIC_NFIX: no AFM density for {label}")
    return NFIX_FACTOR * float(tab[label]["n_per_nm2"])


def n_fix_radius(eta, s, n, lo, hi):
    'Mean radius R (nm) with eta / (pi <R^2>) = n on the size nodes iam4.Grains uses; clipped to [lo, hi].'
    target = float(eta) / (np.pi * n)
    R = float(np.sqrt(target) * np.exp(-0.5 * s * s))
    for _ in range(4):
        Rj, wj = M.ln_nodes(R, s, M4.GH_X, M4.GH_W)
        R *= float(np.sqrt(target / np.sum(wj * Rj ** 2)))
    return float(np.clip(R, lo, hi))


def _eps_fixed():
    'S15 (2026-11-05): the eps fixed through IAM_FIX (fixes alpha_i = 0.40 + eps and the alpha_f zero together), or None.'
    for item in filter(None, os.environ.get("IAM_FIX", "").split(",")):
        k, v = item.split("=")
        if k.strip() == "eps":
            return float(v)
    return None


EPSFIX = _eps_fixed()


def record_suffix(family="grains", sq=None, afmin=None, sigfix=None, tilt=None, bg=None, epsfix=None, current=True,
                  nfix=None, patch=None, nb=None, donors=None, bounds=None, tio2k=None):
    "Record-name suffix of a fit; with current=True the unset arguments come from this process's switches."
    if current:
        sq = SQ if sq is None else sq
        afmin = AFMIN if afmin is None else afmin
        sigfix = SIGFIX if sigfix is None else sigfix
        tilt = TILT if tilt is None else tilt
        bg = BG if bg is None else bg
        epsfix = EPSFIX if epsfix is None else epsfix
        nfix = NFIX if nfix is None else nfix
        patch = PATCH if patch is None else patch
        nb = NB if nb is None else nb
        donors = DONORS if donors is None else donors
        bounds = BOUNDS if bounds is None else bounds
        tio2k = TIO2K if tio2k is None else tio2k
    suffix = ("" if family == "grains" else f"_{family}") + {"common": "_sqc", "free": "_sqf"}.get(sq or "lma", "")
    if afmin is not None and afmin != 0.03:
        suffix += f"_af{int(round(afmin * 100)):02d}"
    if sigfix is not None:
        suffix += f"_sig{int(round(sigfix * 1e4)):03d}"
    if tilt:
        suffix += "_tlt"
    if bg:
        suffix += "_bg"
    if epsfix is not None:
        suffix += ("_e" if epsfix <= 0 else "_ep") + f"{int(round(abs(epsfix) * 1e4)):03d}"
    if nfix:
        suffix += "_n" + nfix.replace(".", "p")
    if patch:
        suffix += "_pat"
    if nb:
        suffix += "_nb"
    if donors:
        suffix += "_don"
    if tio2k:
        suffix += "_tk"
    for k in sorted(bounds or {}):
        suffix += f"_b{k}{int(round(bounds[k][0] * 100)):02d}"
    return suffix


class StaticScan:
    def __init__(self, label, ref=None, grid=None):
        self.label = label
        run = json.loads((IAMP.DATA / label / "binned" / "run.json").read_text(encoding="utf-8"))
        self.run = run
        self.cfg = dict(metal="Au", sub="Si", nominal_nm=run["nominal_nm"], sample=run["sample"], substrate=run["substrate"])
        self.metal, self.sub = "Au", "Si"
        self.grid = grid or M.scan_grid()
        self.q, self.af = self.grid.q, self.grid.native_af
        self.pixmask = M.pixmask_of(label)
        self.res = M.Result(IAMP.DATA / label, self.grid)
        self.states = [r["state"] for r in self.res.rows]
        self.ref = ref or run.get("reference") or "analytic_si"
        shape = (2, len(self.af), len(self.q))
        if self.ref == "analytic_si":
            self.I_ref = None
            self.I_pre, self.V_pre = np.ones(shape), np.zeros(shape)
        else:
            rs = M.Result(IAMP.DATA / self.ref, self.grid)
            idx = TABLE.get(self.ref, {}).get("reference_frames") or list(range(len(rs.rows)))
            Yp, Ep = rs.Y[idx].sum(0).astype(float), rs.E[idx].sum(0).astype(float)
            self.I_ref, self.V_ref = M.smooth3(Yp, Ep)
            self.I_pre, self.V_pre = self.I_ref.copy(), self.V_ref.copy()

    def bg_template(self, i):
        ''
        if getattr(self, "_bg", None) is None:
            import hashlib
            fn = IAMP.DATA / self.label / "shadow_bg_v3.npz"
            sha = hashlib.sha256(fn.read_bytes()).hexdigest()
            if sha != (fn.parent / (fn.name + ".sha256")).read_text(encoding="utf-8").split()[0]:
                raise SystemExit(f"{self.label}: {fn.name} does not match its sha256 side file")
            z = np.load(fn)
            if z["T"].shape[1:] != (len(self.af), len(self.q)):
                raise SystemExit(f"{self.label}: {fn.name} grid does not match the cache grid")
            self._bg = {int(f): z["T"][j].astype(float) for j, f in enumerate(z["frames"])}
            self.bg_sha256 = sha
        if i not in self._bg:
            raise SystemExit(f"{self.label}: no shadow-background template for frame {i}")
        return self._bg[i]

    def window(self, t):
        i = int(round(t))
        Y, E = self.res.Y[i].astype(float), self.res.E[i].astype(float)
        ok = (E > 0) & (self.af >= AFMIN)[None, :, None]
        if self.I_ref is not None:
            ok &= self.I_ref > 0
        nom = float(self.cfg["nominal_nm"] or 0.0)
        return dict(t=float(i), t_mean=float(i), Y=Y, E=E, mask=ok, cfac=1.0, h_m=max(nom, 0.05), state=self.states[i])


def _sphere_P(q, R0, s):
    'Normalised form factor of spheres with lognormal radii (median R0, width s); P -> 1 for q -> 0.'
    t = np.linspace(-3, 3, 41)
    R, w = R0 * np.exp(s * t), np.exp(-t ** 2 / 2)
    x = q[:, None] * R[None]
    f = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3 * R[None] ** 3
    return (f ** 2 * w).sum(1) / (R ** 6 * w).sum()


def tio2_fine_share(I, af, q, band=(0.05, 0.22), width=0.05):
    'Share f_small(alpha_f, q_y) of the fine-structure (sphere) term in a bare TiO2 image I[side, af, q], and the'
    from scipy.optimize import least_squares, nnls
    J = np.nanmean(np.where(I > 0, I, np.nan), axis=0)
    ok_q = q < 1.0
    r = np.nanmean(J[(af >= band[0]) & (af < band[1])], axis=0)
    m = ok_q & np.isfinite(r) & (r > 0)
    qq, rr = q[m], r[m]

    def res(x):
        return np.log(np.exp(x[0]) * _sphere_P(qq, np.exp(x[1]), x[2]) + np.exp(x[3]) * qq ** -4 + np.exp(x[4])) - np.log(rr)
    best = None
    for R0 in (3.0, 6.0, 12.0):
        o = least_squares(res, [np.log(rr[0]), np.log(R0), 0.3, np.log(rr[0] * qq[0] ** 4 * 0.1), np.log(rr[-1] * 0.3)],
                          bounds=([-30, 0.0, 0.05, -40, -40], [30, np.log(40.0), 0.8, 30, 30]))
        if best is None or o.cost < best.cost:
            best = o
    R0, sw = float(np.exp(best.x[1])), float(best.x[2])
    P = _sphere_P(q, R0, sw)
    basis = np.c_[P, q ** -4.0, q ** -1.5, np.ones_like(q)]
    f = np.zeros((len(af), len(q)))
    for a0 in np.arange(af.min(), af.max() + width, width):
        rows = (af >= a0) & (af < a0 + width)
        if not rows.any():
            continue
        r = np.nanmean(J[rows], axis=0)
        m = np.isfinite(r) & (r > 0)
        if m.sum() < 8:
            continue
        c, _ = nnls(basis[m] / r[m, None], np.ones(m.sum()))
        f[rows] = (c[0] * P / np.maximum(basis @ c, 1e-300))[None, :]
    return np.clip(f, 0.0, 1.0), R0, sw


def extra_par(mode):
    out = [("eps", -0.06, 0.03, EPS0), ("lnfm", np.log(0.2), np.log(3.0), 0.0), ("lnB", np.log(1e-5), np.log(1e3), np.log(0.1))]
    if mode == "ref":
        out.append(("lncs", np.log(1e-3), np.log(1e4), 0.0))
        if TIO2K:
            out.append(("lnks", np.log(0.02), np.log(20.0), 0.0))
    else:
        out += [("lnA", np.log(1e-12), np.log(1e6), 0.0), ("lnxi", np.log(5.0), np.log(3000.0), np.log(200.0)), ("Hs", 0.0, 1.0, 0.5)]
    if TILT:
        out.append(("tilt", 0.0, 0.10, 0.03))
    if BG:
        out.append(("lnaT", np.log(0.2), np.log(5.0), 0.0))
    return out


class _StaticMixin:
    FM_IN_WINDOW = True

    def __init__(self, scan, i, mode):
        self.s, self.mode = scan, mode
        self.w = scan.window(i)
        self.fix = {}
        for item in filter(None, os.environ.get("IAM_FIX", "").split(",")):
            k, v = item.split("="); self.fix[k.strip()] = float(v)
        self.fix.update(self.mode_fix(mode))
        if SIGFIX is not None:
            self.fix["sig"] = SIGFIX
        self.n_fix = None
        if NFIX:
            if not isinstance(self, StaticFit):
                raise SystemExit("IAM_STATIC_NFIX is defined for the grains family only")
            self.n_fix = n_fix_of(scan.label)
            self.fix["lnR"] = 0.0
        base = [q for q in self.base_par() if q[0] not in ("lnK", "eps", "lnfm")]
        self.par = base + [("lnK", np.log(1e-2), np.log(1e10), np.log(1e5))] + extra_par(mode) + self.family_par(mode)
        if BOUNDS:
            if set(BOUNDS) - {q[0] for q in self.par}:
                raise SystemExit(f"IAM_STATIC_BOUNDS: unknown parameter in {sorted(BOUNDS)}")
            self.par = [(q[0], BOUNDS[q[0]][0], BOUNDS[q[0]][1], float(np.clip(q[3], *BOUNDS[q[0]]))) + tuple(q[4:])
                        if q[0] in BOUNDS else q for q in self.par]
        self.free = [p for p in self.par if p[0] not in self.fix]
        self.names = [p[0] for p in self.free]
        self.lb = np.array([p[1] for p in self.free]); self.ub = np.array([p[2] for p in self.free])
        self.mdl = self.make_model(scan)
        self.ok = self.w["mask"]; self.nev = 0
        self.h_nom = self.w["h_m"]
        self.T_bg = scan.bg_template(i) if BG else None

    def full(self, x):
        p = dict(self.fix); p.update({q[0]: v for q, v in zip(self.free, x)})
        for q in self.par:
            p.setdefault(q[0], q[3])
        if self.n_fix is not None:
            lo, hi = next((q[1], q[2]) for q in self.par if q[0] == "lnR")
            p["lnR"] = float(np.log(n_fix_radius(p["eta"], p["s"], self.n_fix, np.exp(lo), np.exp(hi))))
        return p

    def x0(self, **over):
        p = {q[0]: q[3] for q in self.par}; p.update(over)
        return np.clip(np.array([p[q[0]] for q in self.free]), self.lb + 1e-9, self.ub - 1e-9)

    def _prepare(self, p):
        if abs(getattr(self.mdl, "eps", 0.0) - p["eps"]) > 1e-12 or getattr(self, "_eps_done", None) is None:
            self.mdl.set_geom(p["eps"]); self._eps_done = p["eps"]
        if abs(self.mdl.sig_af - p["sig"]) > 1e-12:
            self.mdl.set_smear(p["sig"])

        self.w["h_m"] = self.h_nom * float(np.exp(p["lnfm"])) if self.FM_IN_WINDOW else self.h_nom
        if "cp" in p:
            self.w["h_m"] = self.w["h_m"] / p["cp"]
        if self.mode == "ref":
            self.w["cfac"] = float(np.exp(p["lncs"]))
            if "lnks" in p:
                if getattr(self.s, "f_small", None) is None:
                    self.s.f_small, self.s.tio2_R0, self.s.tio2_s = tio2_fine_share(self.s.I_ref, self.s.af, self.s.q)
                self.s.I_pre = self.s.I_ref * (1.0 + (float(np.exp(p["lnks"])) - 1.0) * self.s.f_small[None])
        else:
            G = self.mdl.pre_factor()
            Gs = self.mdl.smear_ratio(G) if self.mdl.Kaf is not None else G
            xi, H = float(np.exp(p["lnxi"])), float(p["Hs"])
            C = xi ** 2 / (1 + (self.s.q * xi) ** 2) ** (1 + H)
            self.s.I_pre = float(np.exp(p["lnA"])) * (Gs[:, None] * C[None, :])[None].repeat(2, axis=0)
            self.w["cfac"] = 1.0

    def _node_model(self, x, p, deps, sig_eff, parts):
        'The family model at eps + deps with the smear sig_eff (S13 tilt node); x and self.fix are not changed.'
        xk, fix0 = x.copy(), self.fix
        q = dict(p, eps=p["eps"] + deps, sig=sig_eff)
        for k in ("eps", "sig"):
            if k in self.names:
                xk[self.names.index(k)] = q[k]
            else:
                self.fix = dict(self.fix, **{k: q[k]})
        try:
            self._prepare(q)
            return super().model(xk, parts=parts)
        finally:
            self.fix = fix0

    def _tilt_model(self, x, p, parts):
        t = float(p["tilt"])
        if t <= 0:
            return self._node_model(x, p, 0.0, p["sig"], parts)
        sig_eff = float(np.hypot(p["sig"], TILT_B * t))
        acc, m = None, None
        for u, wk in zip(M2.GH5_X, M2.GH5_W):
            out, m, _ = self._node_model(x, p, TILT_A * t * float(u), sig_eff, parts)
            if parts:
                mu, P = out
                P = {k: (wk * v if isinstance(v, np.ndarray) else v) for k, v in P.items()}
                acc = (wk * mu, P) if acc is None else (acc[0] + wk * mu, {k: (acc[1][k] + P[k] if isinstance(P[k], np.ndarray) else P[k]) for k in P})
            else:
                acc = tuple(wk * a for a in out) if acc is None else tuple(a + wk * b for a, b in zip(acc, out))
        return acc, m, p

    def model(self, x, parts=False):
        p = self.full(x)
        if TILT:
            out, m, _ = self._tilt_model(x, p, parts)
            pp = p
        else:
            self._prepare(p)
            out, m, pp = super().model(x, parts=parts)
        B = float(np.exp(p["lnB"]))
        bgT = float(np.exp(p["lnaT"])) * self.T_bg[None] if self.T_bg is not None else 0.0
        if parts:
            mu, P = out
            mu, sub = self.unaltered(mu, P["sub"], p)
            if "cp" in p:
                mu, sub, _ = self.patched(mu, sub, None, p)
            P = dict(P, B=B, sub=sub)
            if self.T_bg is not None:
                P["bgT"] = np.broadcast_to(bgT, mu.shape)
                return (mu + B + bgT, P), m, pp
            return (mu + B, P), m, pp
        mu, sub, metal = out
        mu, sub = self.unaltered(mu, sub, p)
        if "cp" in p:
            mu, sub, metal = self.patched(mu, sub, metal, p)
        if self.T_bg is not None:
            return (mu + B + bgT, sub, metal), m, pp
        return (mu + B, sub, metal), m, pp

    def patched(self, mu, sub, metal, p):
        'IAM_STATIC_PATCH: incoherent mixture of the film patches (area fraction cp) and the bare substrate.'
        cp = float(p["cp"])
        bare = (1 - cp) * self.w["cfac"] * self.s.I_pre
        return cp * mu + bare, cp * sub + bare, None if metal is None else cp * metal

    def unaltered(self, mu, sub, p):
        'Islands family, reference mode: a fraction bu of the reference scattering the metal leaves unchanged'
        if "bu" not in p or self.mode != "ref":
            return mu, sub
        ref = self.w["cfac"] * self.s.I_pre
        return mu + p["bu"] * (ref - sub), (1 - p["bu"]) * sub + p["bu"] * ref

    def variance(self, sub):
        if self.mode != "ref":
            return np.zeros_like(sub)
        r = np.where(self.s.I_ref > 0, sub / np.maximum(self.s.I_ref, 1e-30), 0.0)
        return r * r * self.s.V_ref

    def resid(self, x, f=0.05):
        self.nev += 1
        (mu, sub, _), _, p = self.model(x)
        d = M.dev5(self.w, mu, self.variance(sub), f=f)
        r = (np.sign(self.w["Y"] - mu * self.w["E"]) * np.sqrt(d))[self.ok]
        if self.T_bg is not None:
            return np.r_[r, (p["lnfm"] - 0.0) / FM_SD, (p["eps"] - EPS0) / EPS_SD, p["lnaT"] / BG_SD]
        return np.r_[r, (p["lnfm"] - 0.0) / FM_SD, (p["eps"] - EPS0) / EPS_SD]

    def amplitudes(self, x):
        'Set lnK, the substrate amplitude (lncs or lnA) and lnB of x by non-negative least squares on the rates.'
        names = self.names
        x = x.copy()
        for k, v in (("lnK", 0.0), ("lnB", np.log(1e-5))):
            x[names.index(k)] = np.clip(v, self.lb[names.index(k)] + 1e-9, self.ub[names.index(k)] - 1e-9)
        amp = "lncs" if self.mode == "ref" else "lnA"
        x[names.index(amp)] = 0.0
        (mu, sub, metal), _, _ = self.model(x)
        ok = self.ok
        E, Y = self.w["E"][ok], self.w["Y"][ok]
        w = E / np.sqrt(Y + 1.0)
        Mt = np.broadcast_to(metal[None], mu.shape)[ok] * 1.0
        A = np.c_[sub[ok], Mt, np.ones(ok.sum())] * w[:, None]
        rate = Y / np.maximum(E, 1e-30)
        if self.T_bg is not None:
            rate = rate - (np.exp(self.full(x)["lnaT"]) * np.broadcast_to(self.T_bg[None], mu.shape))[ok]
        c, _ = nnls(A, rate * w)
        c = np.maximum(c, 1e-12)
        for k, v in ((amp, np.log(c[0])), ("lnK", np.log(c[1])), ("lnB", np.log(c[2]))):
            j = names.index(k)
            x[j] = np.clip(v, self.lb[j] + 1e-9, self.ub[j] - 1e-9)
        return x

    def run(self, x0, nfev):
        r = least_squares(self.resid, x0, bounds=(self.lb, self.ub), x_scale="jac", diff_step=1e-3, max_nfev=nfev)
        self.last = dict(status=int(r.status), nfev=int(r.nfev), F=float(2 * r.cost))
        return r.x, float(2 * r.cost)


class StaticFit(_StaticMixin, F4.Fit):
    'Grains/base family (iam4.Model4c): islands are its fcap = 1, fb = 0 limit.'
    family = "Model4c grains/base"

    def base_par(self):
        return F4.PAR

    def family_par(self, mode):
        if PATCH:
            if mode == "ref" or TILT:
                raise SystemExit("IAM_STATIC_PATCH is defined for the analytic Si substrate without the tilt model")
            return [("cp", 0.3, 1.0, 0.7)]
        return []

    def mode_fix(self, mode):
        return {"g": 0.0} if mode == "ref" else {"bg": 0.0}

    def make_model(self, scan):
        return M4.Model4c(scan, eps=0.0, interf="lma")

    def start_grid(self):
        if self.n_fix is not None:
            return [dict(eta=eta, s=s, fcap=fc, fb=fb) for eta in (0.3, 0.6, 0.9) for s in (0.15, 0.4)
                    for fc in (0.4, 0.95) for fb in (0.05, 0.5)]
        kds = (1.0, 2.5) if SQ == "free" else (None,)
        return [dict(lnR=np.log(R), eta=eta, fcap=fc, fb=fb, **({} if kd is None else dict(lnkD=np.log(kd))))
                for R in (2.0, 6.0, 15.0) for eta in (0.5, 0.85) for fc in (0.4, 0.95)
                for fb in (0.05, 0.5) + ((0.001,) if NB else ()) for kd in kds]

    def morph(self, p):
        h_m = self.h_nom * float(np.exp(p["lnfm"])) / float(p.get("cp", 1.0))
        m = M4.GrainsB(np.exp(p["lnR"]), p["s"], p["eta"], p["fcap"], p["om"], h_m, fb=p["fb"], rb=p["rb"],
                          RL=np.exp(p["lnRL"]), aL=np.exp(p["lnaL"]), fL=np.exp(p["lnfL"]), Rs=np.exp(p["lnRs"]),
                          as_=np.exp(p["lnas"]), cs=p["cs"], oms=p["oms"])
        if "lnkD" in p:
            m.kD = float(np.exp(p["lnkD"]))
        return m

    def describe(self, m, p):

        kd = float(getattr(m, "kD", 1.0))
        return dict(R=float(m.R), s=float(m.s), eta=float(m.eta), fcap=float(m.fcap), T=float(m.T), D=float(m.D * kd),
                    D_num=float(m.D), kD=kd,
                    omega=float(m.omega), coverage=float(m.coverage), fb=float(getattr(m, "fb", 0.0)),
                    rb=float(getattr(m, "rb", 0.0)), t_b=float(getattr(m, "t_b", 0.0)), fs=float(getattr(m, "fs", 0.0)),
                    Rs=float(m.sp.R) if hasattr(m, "sp") else None, D_s=float(getattr(m, "D_s", 0.0)),
                    RL=float(m.L.R), fL=float(m.fL), d_hex=float(np.sqrt(2 / np.sqrt(3) / m.n)))


class StaticFitIsl(_StaticMixin, F2.Fit):
    'Islands family (iam2.Model2 hemispheroids, local monodisperse interference); reference mode adds bu.'
    family = "Model2 islands"
    FM_IN_WINDOW = False

    def base_par(self):
        return F2.PAR

    def family_par(self, mode):
        return [("bu", 0.0, 1.0, 0.3)] if mode == "ref" else []

    def mode_fix(self, mode):
        return {"g": 0.0} if mode == "ref" else {}

    def make_model(self, scan):
        return M2.Model2(scan, interf="lma")

    def start_grid(self):
        return [dict(lnR=np.log(R), lna=np.log(a), om=om) for R in (1.5, 3.0, 6.0, 12.0) for a in (0.6, 1.2)
                for om in (0.2, 0.4)]

    def morph(self, p):
        h_m = self.h_nom * float(np.exp(p["lnfm"]))
        return M2.Morph2(np.exp(p["lnR"]), np.exp(p["lna"]), p["s"], p["om"], h_m, RL=np.exp(p["lnRL"]),
                         aL=np.exp(p["lnaL"]), fL=np.exp(p["lnfL"]))

    def describe(self, m, p):
        return dict(R=float(m.S.R), aspect=float(m.S.a), T=float(m.S.R * m.S.a), s=float(m.S.s), D=float(m.D),
                    omega=float(m.omega), coverage=float(m.coverage), RL=float(m.L.R), fL=float(m.fL),
                    bu=float(p.get("bu", 0.0)), d_hex=float(np.sqrt(2 / np.sqrt(3) / m.nS)))


FAMILIES = {"grains": StaticFit, "islands": StaticFitIsl}


def solve(f, extra=(), donors=()):
    'Start grid (R x eta x fcap x fb, amplitudes by NNLS), 40-nfev runs from the best 4, 200-nfev from the best 2;'
    starts = [(float(np.sum(f.resid(x) ** 2)), x, "warm") for x in extra]
    for kw in f.start_grid():
        x = f.amplitudes(f.x0(**kw))
        starts.append((float(np.sum(f.resid(x) ** 2)), x, "grid"))
    starts.sort(key=lambda v: v[0])
    pick = starts[:4]
    if NB and "fb" in f.names:
        j = f.names.index("fb")
        pick = pick + [v for v in starts[4:] if v[1][j] < 0.01][:max(0, 2 - sum(v[1][j] < 0.01 for v in pick))]
    pick = pick + [(0.0, x, f"donor:{name}") for name, x in donors]
    short = sorted((f.run(x, 40)[::-1] + (tag,) for _, x, tag in pick), key=lambda v: v[0])
    runs = []
    for k, (_, x, tag) in enumerate(short[:4 if donors else 3 if NB else 2]):
        xx, F = f.run(x, 200)
        runs.append((F, xx, dict(f.last, start=f"grid{k}" if tag == "grid" else f"{tag}{k}")))
    runs.sort(key=lambda v: v[0])
    F, x, opt = runs[0]
    return F, x, opt, [float(r[0] / f.ok.sum()) for r in runs]


def at_bound(f, x, tol=1e-4):
    'Free parameters within tol x (range) of a bound.'
    span = f.ub - f.lb
    return [n for n, v, lo, hi, w in zip(f.names, x, f.lb, f.ub, span) if v - lo < tol * w or hi - v < tol * w]


def uncertainty(f, x):
    'Gauss-Newton standard errors at x (residuals are deviance-normalised) of R, D, T, eps and the mass factor.'
    r0 = f.resid(x)
    fixed = set(at_bound(f, x))
    J = np.zeros((len(r0), len(x)))
    for j in range(len(x)):
        if f.names[j] in fixed:
            continue
        h = 1e-3 * max(1.0, abs(x[j])) * (1 if x[j] + 1e-3 < f.ub[j] else -1)
        x2 = x.copy(); x2[j] += h
        J[:, j] = (f.resid(x2) - r0) / h
    cov = np.linalg.pinv(J.T @ J)

    def derived(xx):
        p = f.full(xx)
        d = f.describe(f.morph(p), p)
        return np.array([d["R"], d["D"], d["T"], p["eps"], np.exp(p["lnfm"])])
    d0 = derived(x)
    G = np.empty((len(d0), len(x)))
    for j in range(len(x)):
        h = 1e-4 * max(1.0, abs(x[j]))
        x2 = x.copy(); x2[j] = min(x2[j] + h, f.ub[j] - 1e-12)
        G[:, j] = 0.0 if f.names[j] in fixed else (derived(x2) - d0) / max(x2[j] - x[j], 1e-15)
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", G, cov, G), 0.0))
    return dict(zip(["R", "D", "T", "eps", "mass_factor"], [float(v) for v in se]))


def record(f, s, label, i, x, opt, cands, t0, mode):
    (mu, P), m, p = f.model(x, parts=True)
    n = int(f.ok.sum())
    dev = {str(ff): float(M.dev5(f.w, mu, f.variance(P["sub"]), f=ff).sum() / n) for ff in (0.05, 0.03, 0.10, 0.0)}
    tot = float(np.sum(np.where(f.ok, mu, 0)))
    rec = dict(label=label, frame=i, row=s.res.rows[i], state=f.w["state"], mode=mode, reference=s.ref, family=f.family,
               nominal_nm=s.cfg["nominal_nm"], h_m_model=f.h_nom * float(np.exp(p["lnfm"])), mass_factor=float(np.exp(p["lnfm"])),
               sq=os.environ.get("IAM_SQ", "lma"), af_min=AFMIN, sig_fixed=SIGFIX, tilt_model=TILT, at_bound=at_bound(f, x), geometry=s.run.get("geometry"),
               params={k: float(v) for k, v in p.items()}, **f.describe(m, p), eps=float(p["eps"]), sig=float(p["sig"]), K=float(np.exp(p["lnK"])), B=float(np.exp(p["lnB"])),
               dev_per_cell=dev, n_cells=n, share_sub=float(np.sum(np.where(f.ok, P["sub"], 0)) / tot),
               share_B=float(P["B"] * n / max(tot, 1e-30)), share_metal=float(np.sum(np.where(f.ok, mu - P["sub"] - P["B"] - P.get("bgT", 0.0), 0)) / tot),
               optimizer=opt, candidates=cands, se=uncertainty(f, x),
               **(dict(bg_model=BG, bg_template_sha256=s.bg_sha256, aT=float(np.exp(p["lnaT"])),
                       share_bgT=float(np.sum(np.where(f.ok, P["bgT"], 0)) / tot))
                  if f.T_bg is not None else {}),
               **(dict(eps_fixed=EPSFIX) if EPSFIX is not None else {}),
               **(dict(n_fix=NFIX, n_fixed_per_nm2=f.n_fix, n_model_per_nm2=float(m.n), warm=WARM or None)
                  if f.n_fix is not None else {}),
               **(dict(tio2_fine_scale=float(np.exp(p["lnks"])), tio2_fine_R0_nm=s.tio2_R0, tio2_fine_s=s.tio2_s,
                       tio2_fine_share_of_reference=float(np.sum(np.where(f.ok, s.I_ref * s.f_small[None], 0)) / np.sum(np.where(f.ok, s.I_ref, 0))))
                  if "lnks" in p else {}),
               **(dict(patch_cover=float(p["cp"]), h_m_in_patch=f.h_nom * float(np.exp(p["lnfm"])) / float(p["cp"]))
                  if "cp" in p else {}),
               absolute_mass="not stated (K and the mass factor couple; QCM tooling at the sample unknown)",
               seconds=time.time() - t0, pixmask=s.pixmask)
    return rec, mu, P, m, p


def fit_frame(task):
    label, i, mode_ref, outdir = task[:4]
    family = task[4] if len(task) > 4 else "grains"
    t0 = time.time()
    s = StaticScan(label, ref=mode_ref)
    mode = "analytic" if s.ref == "analytic_si" else "ref"
    f = FAMILIES[family](s, i, mode)
    extra = []
    if WARM:
        wp = json.loads((Path(WARM) / label / f"{label}_{i:03d}.json").read_text(encoding="utf-8"))["params"]
        if "cp" in f.names:
            extra += [f.x0(**dict({k: v for k, v in wp.items() if k in f.names}, cp=c)) for c in (0.999, 0.6)]
        else:
            extra.append(f.x0(**{k: v for k, v in wp.items() if k in f.names}))
            if NB and "fb" in f.names:
                extra.append(f.x0(**dict({k: v for k, v in wp.items() if k in f.names}, fb=0.001)))
    donors = []
    if DONORS:
        for dp in sorted((Path(DONORS) / label).glob("*.json")):
            dpar = json.loads(dp.read_text(encoding="utf-8"))["params"]
            donors.append((dp.stem, f.amplitudes(f.x0(**{k: v for k, v in dpar.items() if k in f.names}))))
    F, x, opt, cands = solve(f, extra, donors)
    rec, mu, P, m, p = record(f, s, label, i, x, opt, cands, t0, mode)
    dev = rec["dev_per_cell"]
    suffix = record_suffix(family)
    od = Path(outdir) / label; od.mkdir(parents=True, exist_ok=True)
    (od / f"{label}_{i:03d}{suffix}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    np.savez_compressed(od / f"{label}_{i:03d}{suffix}_mu.npz", mu=mu.astype(np.float32), sub=P["sub"].astype(np.float32))
    print(f"{label} #{i} {family} {f.w['state']} R {rec['R']:.2f} D {rec['D']:.1f} T {rec['T']:.2f} cov {rec['coverage']:.2f} "
          f"fm {rec['mass_factor']:.2f} eps {p['eps']:+.3f} sig {p['sig']:.3f} | dev5 {dev['0.05']:.2f} ({rec['seconds']:.0f} s)", flush=True)
    return rec


def reference_check(label, ref, outdir):
    'S9: each usable frame of `label` against the summed reference frames of `ref`: mu = c I_ref(af - d) + B, with'
    s = StaticScan(label, ref=ref)
    rows = []
    for i, st in enumerate(s.states):
        if st != "usable":
            continue
        w = s.window(i)
        own = (ref == label) and (i in (TABLE[ref].get("reference_frames") or []))
        ok = w["mask"]

        def mu_of(z):
            c, lB, d = z
            I = np.empty_like(s.I_ref)
            for sd in (0, 1):
                for j in range(len(s.q)):
                    I[sd, :, j] = np.interp(s.af - d, s.af, s.I_ref[sd, :, j])
            return np.exp(c) * I + np.exp(lB), I

        def res(z):
            mu, I = mu_of(z)
            V = np.exp(2 * z[0]) * s.V_ref
            dd = M.dev5(w, mu, V)
            return (np.sign(w["Y"] - mu * w["E"]) * np.sqrt(dd))[ok]
        z0 = np.array([np.log(np.sum(w["Y"][ok]) / np.sum((s.I_ref * w["E"])[ok])), np.log(1e-4), 0.0])
        r = least_squares(res, z0, bounds=([-30, np.log(1e-8), -0.05], [30, np.log(1e3), 0.05]), x_scale="jac", diff_step=1e-4)
        mu, _ = mu_of(r.x)
        n = int(ok.sum())
        dev = {str(ff): float(M.dev5(w, mu, np.exp(2 * r.x[0]) * s.V_ref, f=ff).sum() / n) for ff in (0.05, 0.03, 0.10, 0.0)}
        rows.append(dict(frame=i, scan=s.res.rows[i]["scan_folder"], in_reference_sum=own, scale=float(np.exp(r.x[0])),
                         B=float(np.exp(r.x[1])), af_shift=float(r.x[2]), dev_per_cell=dev, n_cells=n))
        print(label, "vs", ref, i, s.res.rows[i]["scan_folder"], "own" if own else "", "dev5 %.2f shift %+.4f" % (dev["0.05"], r.x[2]), flush=True)
    od = Path(outdir); od.mkdir(parents=True, exist_ok=True)
    (od / f"refcheck_{label}_vs_{ref}.json").write_text(json.dumps(dict(label=label, ref=ref, rows=rows), indent=1), encoding="utf-8")
    return rows


def main():
    a = sys.argv[1:]
    cmd, label = a[0], a[1]
    opt = {a[k]: a[k + 1] for k in range(2, len(a) - 1, 2)}
    outdir = opt.get("--out", str(OUT))
    if cmd == "refcheck":
        reference_check(label, opt.get("--ref", label), outdir); return
    s = StaticScan(label, ref=opt.get("--ref"))
    sel = opt.get("--frames", "usable")
    if sel in ("usable", "usable+conditional"):
        frames = [i for i, st in enumerate(s.states) if st == "usable" or (sel.endswith("conditional") and st == "conditional")]
    else:
        frames = [int(v) for v in sel.split(",")]
    tasks = [(label, i, opt.get("--ref"), outdir, opt.get("--family", "grains")) for i in frames]
    nw = int(opt.get("--workers", "1"))
    if nw > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(nw) as pool:
            list(pool.map(fit_frame, tasks))
    else:
        for t in tasks:
            fit_frame(t)


if __name__ == "__main__":
    main()
